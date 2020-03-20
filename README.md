# NeRF: Neural Radiance Fields
### [Project](http://tancik.com/nerf) | [Video](https://youtu.be/JuH79E8rdKc) | [Paper](https://arxiv.org/abs/2003.08934) 

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
  

<img src='imgs/pipeline.jpg'/>


## Setup

Python 3 dependencies:

* Tensorflow 1.15 
* matplotlib
* numpy
* imageio
*  configargparse

The LLFF data loader requires ImageMagick.

You will also need the [LLFF code](http://github.com/fyusion/llff) (and COLMAP) set up to compute poses if you want to run on your own real data.

## What is a NeRF? 

A neural radiance field is a simple fully connected network (weights are ~5MB) trained to reproduce input views of a single scene using a rendering loss. The network directly maps from spatial location and viewing direction (5D input) to color and opacity (4D output), acting as the "volume" so we can use volume rendering to differentiably render new views.

Optimizing a NeRF takes between a few hours and a day or two (depending on resolution) and only requires a single GPU. Rendering an image from an optimized NeRF takes somewhere between less than a second and ~30 seconds, again depending on resolution.


## Running code

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
To optimize a low-res Lego NeRF:
```
python run_nerf.py --config config_lego.txt
```

### Rendering a NeRF

Run
```
bash download_example_weights.sh
```
to get a pretrained high-res NeRF for the Fern dataset. Now you can use the `render_demo.ipynb` to render new views.


## Generating poses for your own scenes

We recommend using the `imgs2poses.py` script from the [LLFF code](https://github.com/fyusion/llff). Then you can pass the base scene directory into our code using `--datadir <myscene>` along with `-dataset_type llff`. You can take a look at the `config_fern.txt` config file for example settings to use for a forward facing scene.
