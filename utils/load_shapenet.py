import os
import numpy as np
import tensorflow as tf
import imageio
from utils.load_blender import pose_spherical
from utils.load_llff import recenter_poses
import h5py

rot90y = np.array([[0, 0, -1],
                   [0, 1, 0],
                   [1, 0, 0]], dtype=np.float32)
rot66z = np.array([[0.407, 0.914, 0.000, 0],
                [-0.914, 0.407, -0.000, 0],
                [-0.000, 0.000, 1.000, 0],
                [0, 0, 0, 1]])


def load_shapenet_data(basedir='./data/shapenet/blender_renderings/', half_res=False, quarter_res=False, sample_nums=(5, 2, 1), fix_objects=None):
    SINGLE_OBJ = False

    all_imgs = []
    all_poses = []

    imgs_dir = os.path.join(basedir,'syn_rgb')


    if fix_objects is not None:
        print('Using specified objects')
        objs = np.array(fix_objects)
    else:
        objs = [obj_name for obj_name in os.listdir(imgs_dir)
                if os.path.isdir(os.path.join(imgs_dir, obj_name))]
        objs = np.random.choice(objs, np.sum(sample_nums), replace=False)

    focal = 210  # DISN fix this to 35 -- this is blender default!

    if sample_nums == (1, 0, 0):
        # signle object mode, doesn't allow i_test
        SINGLE_OBJ = True
        i_split = [[], [], []]
        print('Using single object mode')
    else:
        # replaced with split?
        sample_counts = [0, sample_nums[0], sample_nums[0] +
                         sample_nums[1], sum(sample_nums)]
        i_split = [np.arange(sample_counts[i], sample_counts[i+1])
                   for i in range(3)]

    # tracks the indices for each object
    obj_split = []

    for obj in objs:
        rendering_path = os.path.join(basedir,'syn_rgb', obj)
        renderings = [name for name in os.listdir(rendering_path)
                if name.endswith('.png')]
        renderings.sort()

        pose_path = os.path.join(basedir,'syn_pose', obj)
        poses = [name for name in os.listdir(pose_path)
                if name.endswith('.txt')]
        poses.sort()

        imgs_indices = []
        

        for i, rendering_name in enumerate(renderings):
            imgs_indices.append(len(all_imgs))
            all_imgs.append(imageio.imread(os.path.join(rendering_path, rendering_name)))
            pose = np.loadtxt(os.path.join(pose_path, poses[i]))
            all_poses.append(pose)
        obj_split.append(imgs_indices)

    obj_split = np.array(obj_split)

    if SINGLE_OBJ:
        print(f'Object for training is:{objs}')
        # TODO: shuffle views for single obj
        i_split[1] = np.array([0,8,16,24])
        i_split[0] = np.array([i for i in range(len(all_imgs)) if i not in i_split[1]])
        i_split[2] = np.array([])

    else:
        print(f'Objects for training are:{objs[i_split[0]]}')
        print(f'Objects for validation are:{objs[i_split[1]]}')
        print(f'Objects for testing are:{objs[i_split[2]]}\n')
        # convert object indices in i_split to image indices
        for i in range(len(i_split)):
            i_split[i] = np.concatenate(obj_split[i_split[i]]) if len(i_split[i]) > 0 else np.array([])

    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0)
                             for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)
    # render poses for videos and render only experiments

    H, W = all_imgs[0].shape[:2]

    if quarter_res or half_res:
        factor = 4 if half_res else 2
        H = H//factor
        W = W//factor
        focal = focal/float(factor)
        all_imgs = tf.image.resize_area(all_imgs, [H, W]).numpy()

    all_imgs = np.array(all_imgs).astype(np.float32)
    all_imgs = all_imgs/255.
    all_poses = np.array(all_poses)
    # all_poses = recenter_poses(all_poses)
    all_poses = all_poses.astype(np.float32)
    # all_poses = rot66z @ all_poses

    return all_imgs, all_poses, render_poses, [H, W, focal], i_split, obj_split

