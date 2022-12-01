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

USAGE="$GREEN bash $0 <model_name> <line_drawing> <style_id:optional_for_random> <library:optional_for_seed> <output_file_prefix:optional> $NOCOLOR
\n\n
Runs NeuBE Painting Engine to automatically stylize an input line drawing
in a specified style.
\n\n
Assumes that checkpoints are under:\n
$DIR/models/neube/<model_name>
\n\n
-------------------------------------------------------------------------------\n
Examples:
\n\n
To run with a specific style from a brush library:\n
INPUT=forger/images/large_guidance/lamali_sm.png\n
bash $0 style2 \$INPUT playdoh10 models/neube/style2//brush_libs/projected_training_styles.pkl
\n\n
To run with a random seed style:\n
bash $0 style2 \$INPUT 594
\n\n
Tip:\n
To see available styles, see logs of $GREEN bash neube_run.sh $NOCOLOR
"

if [ $# -lt "2" ] || [ $# -gt "5" ]; then
    echo -e "$RED Error: $NOCOLOR Wrong argument number $#"
    echo
    echo -e $USAGE
    exit
fi

MODEL_NAME=$1
MODEL_DIR=models/neube/$MODEL_NAME/
if [ ! -d $MODEL_DIR ]; then
    echo -e "$RED ERROR: $NOCOLOR Model directory $MODEL_DIR not found. Did you extract pretrained checkpoints?"
    echo -e $USAGE
    exit
fi
CHECKPOINT=$MODEL_DIR/snapshot.pkl

INPUT=$2
STYLE=rand
LIB="rand100"  # always random lib
if [ $# -gt "2" ]; then
    STYLE=$3
    LIB="1000"  # seed lib
fi

if [ $# -gt "3" ]; then
    LIB=$4
fi

BNAME=$(basename $INPUT)
echo $BNAME
BNAME="${BNAME%.*}"
OUTPUT_PREFIX="output/stylizations/$STYLE/$BNAME"
if [ $# -eq "5" ]; then
    OUTPUT_PREFIX=$5
fi
echo -e "$GREEN Outputting to: $NOCOLOR ${OUTPUT_PREFIX}_clear_${STYLE}.png"

FLAGS="--gan_checkpoint=$CHECKPOINT --geom_image=$INPUT --feature_blending_level=2"
FLAGS="$FLAGS --color_mode=1 --crop_margin=10 --style_id=$STYLE --library=$LIB --on_white"
FLAGS="$FLAGS --output_file_prefix=$OUTPUT_PREFIX"
echo -e "$GREEN Runnging: $NOCOLOR python -m forger.viz.paint_image_main"
echo -e "$FLAGS"

python -m forger.viz.paint_image_main $FLAGS
