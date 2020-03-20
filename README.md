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
