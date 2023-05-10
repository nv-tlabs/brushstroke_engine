# Neural Brushstroke Engine

This is the official repository for "Neural Brushstroke Engine: Learning a Latent Style Space of
Interactive Drawing Tools" by Masha Shugrina, Chin-Ying Li and Sanja Fidler. SIGGRAPH Asia 2022.
[See Official Project Website](https://nv-tlabs.github.io/brushstroke_engine/).

<p align="center">
<img src="docs/assets/summary.gif" alt="Neural brushstroke engine summary" width="600px" />
</p>

Neural Brushstroke Engine (NeuBE) includes a GAN model that learns to mimic many drawing media styles
by learning from unlabeled images. The result is a GAN model that can be
directly controlled by user strokes, with style code **z** corresponding to the style of the interactive brush (and not the final image). Together
with a patch-based paining engine, NeuBE allows seamless drawing on a canvas of
any size, user control of stroke color, compositing of strokes on clear background,
and text-based brush search. NeuBE generalizes very well to unseen styles and requires only
about 200 images of different drawing media to train. We also support automatic stylization of line drawings. Try out our drawing interface!


## Requirements

You will need a CUDA-enabled version of PyTorch in your environment. This model comfortably runs on a 12GB GPU,
and might be able to work with less. Training and other scripts may require more memory. We have only run this code on linux systems, but in principle other CUDA-enabled setups should work.

If using a local environment such as [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), create the environment and install PyTorch (we used torch 1.7 and 1.8),
then install requirements. For example, assuming your system supports CUDA 11.1:

```shell
conda create --name art_forger python=3.7
conda activate art_forger
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install -e .
```

We additionally provide a Docker file.


## Datasets and Models

Pretrained models are available in [models.zip](https://drive.google.com/file/d/1Ao-3LZPmOSCZA7GIm5piQ3T6PK63NCZe/view?usp=sharing) (backup link [here](https://drive.google.com/file/d/1cCIBraCUpBygtWuoym4kqAwcPlx1V-rc/view?usp=sharing)). To setup downloaded models
run the following from the **root of this repo**:
```shell
unzip ~/Downloads/models.zip -d .
```
See [Project Website](https://nv-tlabs.github.io/brushstroke_engine/) for training dataset downloads.

## Drawing Interface

To launch web UI using unzipped checkpoints run the following, and go to [localhost:8000/?demo&canvas=2000](http://localhost:8000/?demo&canvas=2000) to draw. 

```shell
MODEL=style1  # or style2 for a model trained on less usual drawing styles
bash neube_run.sh $MODEL  # Run without model argument to get help and options
```

This command launches UI pre-loaded with a menu of brushes and a debug interface to experiment with model settings. 
The `demo` URL
parameter sets the best drawing options automatically; use `canvas` URL parameter to adjust drawing canvas size. 
We recommend Chrome browser on the desktop and Safari on iPad. 

<img src="docs/assets/ui.jpg" alt="drawing" width="100%"/>

## Line Drawing Stylization

We also provide a script to automatically stylize input line drawings given select style.

```shell
MODEL=style2  # or style1 for a model trained on more traditional styles
INPUT=forger/images/large_guidance/lamali_sm.png  # or your own black on white line drawing
STYLE=playdoh10  # tip: see log of neube_run.sh for style names
LIBRARY=models/neube/style2/brush_libs/projected_training_styles.pkl  # style lib of the right model

bash neube_stylize.sh $MODEL $INPUT $STYLE $LIBRARY  # Run without arguments to get help and options
```

Or, to stylize by a random seed used to generate **z**, run:
```shell
Z=594  # seed that deterministically sets the multi-dimensional z-vector
bash neube_stylize.sh $MODEL $INPUT $Z
```

Note that the corresponding python script `forger.viz.paint_image_main` can also produce outputs on clear background and
supports style interpolation, e.g. like this:

<img src="docs/assets/interp_lamali.jpg" alt="drawing" width="100%"/>

## Training

If you need access to our training scripts, please comment on <a href="https://github.com/nv-tlabs/brushstroke_engine/issues/4">this bug</a> so we can prioritize release. 

## Code organization

Our training and model scripts are based on StyleGAN2, and a modified copy of [this repository](https://github.com/NVlabs/stylegan2-ada-pytorch) is copied
here. For legacy reasons some of our training scripts reside in that thirdparty location, while invoking
utilities from the core package.

* [forger](forger) - core python codebase
    * [ui](forger/ui) - web interface connecting to models
* [thirdparty](thirdparty) - core StyleGAN2 training code with some modifications

The name for the python package is inspired by heist plots on art forgery. Before painting can begin, a (human) art forger is first and foremost an expert that knows exactly how to mimic source media by looking at an artwork. Neural Brushstroke Engine is the first step toward acquiring this kind of expertise using AI to learn from unlabeled images and empower the human artist with "forged" digital brushes. 

## Citation

```
@article{shugrina2022neube,
  title={Neural Brushstroke Engine: Learning a Latent Style Space of Interactive Drawing Tools},
  author={Shugrina, Maria and Li, Chin-Ying and Fidler, Sanja},
  journal={ACM Transactions on Graphics (TOG)},
  volume={41},
  number={6},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```