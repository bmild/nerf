from utils.load_shapenet import load_shapenet_data
from utils.load_blender import load_blender_data
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

images, poses, render_poses, hwf, i_split, obj_split = load_shapenet_data(
                                                    './data/shapenet/ShapeNetRendering/04256520/',
                                                    sample_nums=(5,2,1))

# images, poses, render_poses, hwf, i_split = load_blender_data(
#                                             './data/nerf_synthetic/lego', True, 1, True)

print(hwf)