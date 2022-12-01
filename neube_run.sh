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

USAGE="$GREEN bash $0 <model_name> <port:8000 default> <extra flags to forger.ui.run:optional> <library_spec:optional> $NOCOLOR
\n\n
Runs NeuBE Interactive Client-Server interface with selected\n
model as backbone. Provide extra flags to customize.
\n\n
Assumes that checkpoints are under:\n
$DIR/models/neube/<model_name>
\n\n
Library spec specifies how many styles to show in the UI from\n
projected and CLIP brush libraries:\n
randX - cap number of styles displayed in the UI to X random styles for each lib\n
disp - display all styles (default)
\n\n
-------------------------------------------------------------------------------\n
Examples:
\n\n
Run default:\n
$GREEN bash $0 style1 $NOCOLOR
\n\n
Run on custom port:\n
$GREEN bash $0 style2 8008 $NOCOLOR
\n\n
Run with custom flags:\n
$GREEN bash $0 style1 8000 '--log_level=30 --debug_dir=/tmp/neube_debug' $NOCOLOR
\n\n
Example capping number of displayed brush styles from each library to random 9:\n
$GREEN bash $0 style1 8000 '' rand9 $NOCOLOR
"

if [ $# -lt "1" ] || [ $# -gt "4" ]; then
    echo -e "$RED Error: $NOCOLOR Wrong argument number"
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

PORT=8000
if [ $# -gt "1" ]; then
    PORT=$2
fi

EXTRA_FLAGS="--port=$PORT --gan_checkpoint=$CHECKPOINT"
echo -e "$GREEN Setting $NOCOLOR --port=$PORT"
if [ $# -gt "2" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS $3"
fi
echo -e "$GREEN Setting flags $NOCOLOR $EXTRA_FLAGS"

LSPEC="disp"
if [ $# -gt 3 ]; then
    LSPEC="$4"
fi

SAVED_STYLES=$MODEL_DIR/snapshot.saved_zs.txt
CURATED_STYLES=$MODEL_DIR/brush_libs/curated_zs.txt
RANDOM_CLEAR_STYLES=$MODEL_DIR/brush_libs/random5K_clarity0.9.txt
PROJ_STYLES=$MODEL_DIR/brush_libs/projected_training_styles.pkl
PROJ_NOVEL_STYLES=$MODEL_DIR/brush_libs/projected_novel_styles.pkl
PROJ_ART_STYLES=$MODEL_DIR/brush_libs/projected_artwork.pkl
CLIP_STYLES=$MODEL_DIR/brush_libs/clip_brushes.pkl

CURATED_STYLES="Curated_Zs_Random_Choice:rand9:$CURATED_STYLES"
RANDOM_CLEAR_STYLES="Random_Clear_Zs:rand9:$RANDOM_CLEAR_STYLES"
PROJ_STYLES="Projected_Train_Styles:$LSPEC:$PROJ_STYLES"
PROJ_NOVEL_STYLES="Projected_Novel_Styles:$LSPEC:$PROJ_NOVEL_STYLES"
PROJ_ART_STYLES="Projected_Art_Styles:$LSPEC:$PROJ_ART_STYLES"
CLIP_STYLES="CLIP_Styles:$LSPEC:$CLIP_STYLES"

LIBRARIES="Saved_Zs:$LSPEC:default,${CURATED_STYLES},${RANDOM_CLEAR_STYLES},${PROJ_STYLES},${PROJ_NOVEL_STYLES},${PROJ_ART_STYLES},${CLIP_STYLES}"

if [ $MODEL_NAME -eq "style2" ]; then
    # Include extra experiments with CLIP and style2 model
    CLIP_STYLES2=$MODEL_DIR/brush_libs/clip_brushes2.pkl
    CLIP_STYLES3=$MODEL_DIR/brush_libs/clip_brushes3.pkl
    LIBRARIES="${LIBRARIES},CLIP_Styles_Artistic:$LSPEC:${CLIP_STYLES2},CLIP_Styles_Images_Interp:$LSPEC:${CLIP_STYLES3}"
fi

echo -e "$GREEN Setting brush library spec $NOCOLOR $LIBRARIES"
echo ""
echo -e "$GREEN Runing server with command:$NOCOLOR"
echo "python -m forger.ui.run"
echo "   $EXTRA_FLAGS"
echo "   --libraries=\"${LIBRARIES}\""

URL="http://localhost:$PORT/?demo&canvas=2000"
echo $URL
echo -e "$GREEN Note: Zs you save via debug menu are saved in $SAVED_STYLES $NOCOLOR (relaunch to see in menu)"
echo -e "$GREEN Running server... Go to: $NOCOLOR \e]8;;${URL}\a${URL}\e]8;;\a"

python -m forger.ui.run $EXTRA_FLAGS --libraries="${LIBRARIES}"
