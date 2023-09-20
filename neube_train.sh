#!/bin/bash -e

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

set -o nounset

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

export CLI_COLOR=1
RED='\033[1;31m'
GREEN='\033[1;32m'
NOCOLOR='\033[0m'

USAGE="$GREEN bash $0 <data_mode> <train_phase> <base_output_dir> <extra flags to thirdparty.stylegan2_ada_pytorch.train:optional> $NOCOLOR
\n\n
Runs default NeuBE training script (half the epochs reported in paper) with parameters from
our best-performing ablations. Has two sets of flags - for core training and finetuning.
\n\n
Assumes that the pretrained encoder checkpoint is under:\n
$DIR/models/geo_encoders/strong.pt
\n\n
Unless running in 'custom' mode, assumes that the data is located under:\n
$DIR/data
\n\n
Args:\n
  data_mode: styles1, styles2 or custom; will automatically set up datasets for \n
            styles1 and styles2
  train_phase: 'train' or 'finetune'
  base_output_dir: will create run subdir relative to this
  extra_flags: to pass to the train script
\n\n
Important: $NOCOLOR note that training is highly stochastic, as reported in
our paper (Sec. 6.3). \n
We hope that future work will improve on this.
\n
-------------------------------------------------------------------------------\n
Examples:
\n\n
Run default:\n
$GREEN bash $0 styles1 train ./train_out $NOCOLOR
\n\n
Finetune default:
$GREEN bash $0 styles1 finetune ./train_out '--resume=train_out/00000-.../network-snapshot-020000.pkl' $NOCOLOR
\n\n
Run with custom flags:\n
$GREEN bash $0 styles1 train /.train_out '--kimg=10 --wandb_group=test --wandb_project=neube' \$NOCOLOR
"

if [ $# -lt "3" ] || [ $# -gt "4" ]; then
    echo -e "$RED Error: $NOCOLOR Wrong argument number"
    echo
    echo -e $USAGE
    exit
fi

DATA_MODE=$1
PHASE=$2
ODIR=$3
CACHE_DIR=$ODIR/neube_cache
EXP_DIR=$ODIR/neube_exp

mkdir -p $CACHE_DIR
mkdir -p $EXP_DIR

echo "---------------------------------------------------------------------------"
echo -e "$GREEN Parsing flags... $NOCOLOR"

DEFAULT_FLAGS=$(cat train_flags.txt | awk '{printf "%s ", $0}')
ENC_FLAGS="--enc_checkpt=$DIR/models/geo_encoders/strong.pt"
OUT_FLAGS="--cache=$CACHE_DIR --outdir=$EXP_DIR --name_prefix=default_${PHASE}"
echo "   set output flags to: $OUT_FLAGS"
DATA_FLAGS=""

if [ $DATA_MODE == "styles1" ] || [ $DATA_MODE == "styles2" ]; then
    GEOM_EVAL_DATA=triband_${DATA_MODE}:data/astro_datasets/for_eval/triband_geometry_${DATA_MODE}.zip
    GEOM_DATA=triband_new_splines:data/astro_datasets/triband_geometry_splines.zip
    STYLE_DATA=${DATA_MODE}:data/astro_datasets/${DATA_MODE}.zip
    DATA_FLAGS="--data=$STYLE_DATA --geom_data=$GEOM_DATA --geom_metric_data=$GEOM_EVAL_DATA"
    echo "   set data flags to: $DATA_FLAGS"
elif [ $DATA_MODE == "custom" ]; then
    echo -e "  $RED custom mode $NOCOLOR"
    echo "   expecting --data --geom_data to be set in optional flags"
else
    echo -e "$RED Error: $NOCOLOR unexpected mode $DATA_MODE (expected styles1, styles2 or custom)"
    exit
fi

CUSTOM_FLAGS=""
if [ $# -gt 3 ]; then
    CUSTOM_FLAGS="$4"
    echo "   set custom flags to: $CUSTOM_FLAGS"
fi

if [ $PHASE == "train" ]; then
    ALL_FLAGS="$DEFAULT_FLAGS $ENC_FLAGS $OUT_FLAGS $DATA_FLAGS $CUSTOM_FLAGS"
    echo "   using training phase flags"
elif [ $PHASE == "finetune" ]; then
    FINETUNE_FLAGS=$(cat finetune_flags.txt | awk '{printf "%s ", $0}')
    ALL_FLAGS="$DEFAULT_FLAGS $ENC_FLAGS $OUT_FLAGS $DATA_FLAGS $FINETUNE_FLAGS $CUSTOM_FLAGS"
    echo "   using finetune phase flags"
    if [ $# -lt 4 ]; then
        echo -e "$RED Error: $NOCOLOR should set --resume checkpoint in custom flags when finetuning"
        echo "... Exiting"
        echo
        echo -e $USAGE
        exit
    fi
else
    echo -e "$RED Error: $NOCOLOR unexpected phase $PHASE (expected train or finetune)"
    exit
fi

echo
echo -e "$GREEN   Flags set. $NOCOLOR"
echo

echo "---------------------------------------------------------------------------"
echo -e "$GREEN Starting \"${PHASE}\" phase of training. $NOCOLOR"
echo

echo "Flags set to:"
echo "$ALL_FLAGS" | awk '{for (i = 1; i <= NF; ++i) { printf "%s\n", $i}}'
echo

TRAIN_LOG=$EXP_DIR/latest_${DATA_MODE}_${PHASE}_log.txt
echo "Latest train log written to:"
echo "$TRAIN_LOG"
echo

python3 -m thirdparty.stylegan2_ada_pytorch.train $ALL_FLAGS | tee $TRAIN_LOG

echo
echo -e "$GREEN Training for phase ${PHASE} complete $NOCOLOR"
echo
echo "Your output directory is in $EXP_DIR"
echo "Log was written to $TRAIN_LOG"
