# NeRF: Neural Radiance Fields
### [Project Page](http://tancik.com/nerf) | [Video](https://youtu.be/JuH79E8rdKc) | [Paper](https://arxiv.org/abs/2003.08934) | [Data](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
[![Open Tiny-NeRF in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb)<br>
Tensorflow implementation of optimizing a neural representation for a single scene and rendering new views.<br><br>
[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](http://tancik.com/nerf)  
 [Ben Mildenhall](https://people.eecs.berkeley.edu/~bmild/)\*<sup>1</sup>,
 [Pratul P. Srinivasan](https://people.eecs.berkeley.edu/~pratul/)\*<sup>1</sup>,
 [Matthew Tancik](http://tancik.com/)\*<sup>1</sup>,
 [Jonathan T. Barron](http://jonbarron.info/)<sup>2</sup>,
 [Ravi Ramamoorthi](http://cseweb.ucsd.edu/~ravir/)<sup>3</sup>,
 [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html)<sup>1</sup> <br>
 <sup>1</sup>UC Berkeley, <sup>2</sup>Google Research, <sup>3</sup>UC San Diego  
  \*denotes equal contribution  
in ECCV 2020 (Oral Presentation, Best Paper Honorable Mention)

<img src='imgs/pipeline.jpg'/>

## TL;DR quickstart

To setup a conda environment, download example training data, begin the training process, and launch Tensorboard:
```
conda env create -f environment.yml
conda activate nerf
bash download_example_data.sh
python run_nerf.py --config config_fern.txt
tensorboard --logdir=logs/summaries --port=6006
```
If everything works without errors, you can now go to `localhost:6006` in your browser and watch the "Fern" scene train.

## Setup

Python 3 dependencies:

* Tensorflow 1.15
* matplotlib
* numpy
* imageio
*  configargparse

The LLFF data loader requires ImageMagick.

We provide a conda environment setup file including all of the above dependencies. Create the conda environment `nerf` by running:
```
conda env create -f environment.yml
```

You will also need the [LLFF code](http://github.com/fyusion/llff) (and COLMAP) set up to compute poses if you want to run on your own real data.

## What is a NeRF?

A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Optimizing a NeRF takes between a few hours and a day or two (depending on resolution) and only requires a single GPU. Rendering an image from an optimized NeRF takes somewhere between less than a second and ~30 seconds, again depending on resolution.


## Running code

Here we show how to run our code on two example scenes. You can download the rest of the synthetic and real data used in the paper [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

### Optimizing a NeRF

Run
```
bash download_example_data.sh
```
to get the our synthetic Lego dataset and the LLFF Fern dataset.

To optimize a low-res Fern NeRF:
```
python run_nerf.py --config config_fern.txt
```
After 200k iterations (about 15 hours), you should get a video like this at `logs/fern_test/fern_test_spiral_200000_rgb.mp4`:

![ferngif](https://people.eecs.berkeley.edu/~bmild/nerf/fern_200k_256w.gif)

To optimize a low-res Lego NeRF:
```
python run_nerf.py --config config_lego.txt
```
After 200k iterations, you should get a video like this:

![legogif](https://people.eecs.berkeley.edu/~bmild/nerf/lego_200k_256w.gif)

### Rendering a NeRF

Run
```
bash download_example_weights.sh
```
to get a pretrained high-res NeRF for the Fern dataset. Now you can use [`render_demo.ipynb`](https://github.com/bmild/nerf/blob/master/render_demo.ipynb) to render new views.

### Replicating the paper results

The example config files run at lower resolutions than the quantitative/qualitative results in the paper and video. To replicate the results from the paper, start with the config files in [`paper_configs/`](https://github.com/bmild/nerf/tree/master/paper_configs). Our synthetic Blender data and LLFF scenes are hosted [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and the DeepVoxels data is hosted by Vincent Sitzmann [here](https://drive.google.com/open?id=1lUvJWB6oFtT8EQ_NzBrXnmi25BufxRfl).

### Extracting geometry from a NeRF

Check out [`extract_mesh.ipynb`](https://github.com/bmild/nerf/blob/master/extract_mesh.ipynb) for an example of running marching cubes to extract a triangle mesh from a trained NeRF network. You'll need the install the [PyMCubes](https://github.com/pmneila/PyMCubes) package for marching cubes plus the [trimesh](https://github.com/mikedh/trimesh) and [pyrender](https://github.com/mmatl/pyrender) packages if you want to render the mesh inside the notebook:
```
pip install trimesh pyrender PyMCubes
```

## Generating poses for your own scenes

### Don't have poses?

We recommend using the `imgs2poses.py` script from the [LLFF code](https://github.com/fyusion/llff). Then you can pass the base scene directory into our code using `--datadir <myscene>` along with `-dataset_type llff`. You can take a look at the `config_fern.txt` config file for example settings to use for a forward facing scene. For a spherically captured 360 scene, we recomment adding the `--no_ndc --spherify --lindisp` flags.

### Already have poses!

In `run_nerf.py` and all other code, we use the same pose coordinate system as in OpenGL: the local camera coordinate system of an image is defined in a way that the X axis points to the right, the Y axis upwards, and the Z axis backwards as seen from the image.

Poses are stored as 3x4 numpy arrays that represent camera-to-world transformation matrices. The other data you will need is simple pinhole camera intrinsics (`hwf = [height, width, focal length]`) and near/far scene bounds. Take a look at [our data loading code](https://github.com/bmild/nerf/blob/master/run_nerf.py#L406) to see more.

## Citation

```
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
```
