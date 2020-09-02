import os
import numpy as np
import tensorflow as tf
import imageio
from utils.load_blender import pose_spherical
import h5py

rot90y = np.array([[0, 0, -1],
                   [0, 1, 0],
                   [1, 0, 0]], dtype=np.float32)


def load_shapenet_data(basedir='./data/shapenet/ShapeNetRendering/04256520', half_res=False, quarter_res=False, sample_nums=(5, 2, 1), fix_objects=None):
    SINGLE_OBJ = False

    all_imgs = []
    all_poses = []


    if fix_objects is not None:
        print('Using specified objects')
        objs = np.array(fix_objects)
    else:
        objs = [obj_name for obj_name in os.listdir(basedir)
                if os.path.isdir(os.path.join(basedir, obj_name))]
        objs = np.random.choice(objs, np.sum(sample_nums), replace=False)

    rot_mat = get_rotate_matrix(-np.pi / 2)
    focal = 35  # DISN fix this to 35
    # focal = .5 * 137 / np.tan(.5 * 0.436332)

    if sample_nums == (1, 0, 0):
        # signle object mode, doesn't allow i_test
        SINGLE_OBJ = True
        i_split = [[], [], []]
        print('Using single object mode')
    else:
        sample_counts = [0, sample_nums[0], sample_nums[0] +
                         sample_nums[1], sum(sample_nums)]
        i_split = [np.arange(sample_counts[i], sample_counts[i+1])
                   for i in range(3)]

    # tracks the indices for each object
    obj_split = []

    for obj in objs:
        rendering_path = os.path.join(basedir, obj, 'rendering')
        sdf_fl = os.path.join('./../DISN/data/sdf/SDF_v1/02958343/', obj, "ori_sample.h5")
        norm_mat = get_norm_matrix(sdf_fl)

        with open(os.path.join(rendering_path, 'renderings.txt'), 'r') as index_f:
            # load image file names
            img_list = [line.strip() for line in index_f.read().splitlines()]
            # load rotation metadata
            params = np.loadtxt(os.path.join(
                rendering_path, 'rendering_metadata.txt'))
            params_list = [params[num, ...].astype(
                np.float32) for num in range(len(img_list))]

            imgs_indices = []

            for i, img_name in enumerate(img_list):
                # read img
                # add index of the image being read
                imgs_indices.append(len(all_imgs))
                all_imgs.append(imageio.imread(
                    os.path.join(rendering_path, img_name)))

                # calculate pose
                az, el, inl, distance_ratio, fov = params_list[i]
                H, W = all_imgs[-1].shape[:2]

                # TODO: make sure this is correct and find out what normal matrix does
                # DISN version
                # camR, _ = get_img_cam(params_list[j])
                # pose = np.hstack([np.concatenate([camR,[[0,0,0]]]),[[0],[0],[0],[1]]])

                # find out if this can be used
                # obj_rot_mat = np.dot(rot90y, camR)
                K, RT = getBlenderProj(
                    az, el, distance_ratio, img_w=H, img_h=W)

                trans_mat = np.linalg.multi_dot([K, RT, rot_mat, norm_mat])

                pose = np.linalg.inv(np.concatenate([trans_mat,[[0,0,0,1]]]))

                # RT = np.transpose(RT)
                # pose = np.linalg.inv(np.concatenate([RT, [[0], [0], [0], [1]]], axis=1))
                # note that pose here includes translation
                # this information is lost in nerf due to operation pose[:3, :4]

                # trans_mat = np.linalg.multi_dot([K, RT, rot_mat])
                # pose = trans_mat_right

                # NERF version
                # doesn't work!
                # pose = pose_spherical(az, el, 0)

                all_poses.append(pose)

                # assign focal
                # note that this assumes all renderings have same focal
                # if focal is None:
                #     fov = fov / 180 * np.pi
                #     focal = .5 * W / np.tan(.5 * fov)
            obj_split.append(imgs_indices)

    obj_split = np.array(obj_split)

    if SINGLE_OBJ:
        print(f'Object for training is:{objs}')
        # TODO: shuffle views for single obj
        i_split[0:2] = np.split(obj_split[0], [-4])
        i_split[2] = np.array([])

        # print("WARNING: using all train")
        # i_split[0] = np.array(obj_split[0])
        # i_split[1] = np.array(obj_split[0])
        # i_split[2] = np.array([])

        # i_split[1] = np.random.choice(range(len(img_list)), 4, replace=False)
        # i_split[0] = np.array([i for i in range(len(img_list)) if i not in i_split[1]])

    else:
        print(f'Objects for training are:{objs[i_split[0]]}')
        print(f'Objects for validation are:{objs[i_split[1]]}')
        print(f'Objects for testing are:{objs[i_split[2]]}\n')
        # convert object indices in i_split to image indices
        for i in range(len(i_split)):
            i_split[i] = np.concatenate(obj_split[i_split[i]])

    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0)
                             for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)
    # render poses for videos and render only experiments

    if quarter_res or half_res:
        factor = 4 if half_res else 2
        H = H//factor
        W = W//factor
        focal = focal/float(factor)
        all_imgs = tf.image.resize_area(all_imgs, [H, W]).numpy()

    all_imgs = np.array(all_imgs, dtype=np.float32)
    all_imgs = all_imgs/255.
    all_poses = np.array(all_poses, dtype=np.float32)

    return all_imgs, all_poses, render_poses, [H, W, focal], i_split, obj_split


def getBlenderProj(az, el, distance_ratio, img_w=137, img_h=137):
    # camera matrix https://en.wikipedia.org/wiki/Camera_matrix
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    F_MM = 35.  # Focal length
    SENSOR_SIZE_MM = 32.
    PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.
    SKEW = 0.
    CAM_MAX_DIST = 1.75
    CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                          [1.0, -4.371138828673793e-08, -0.0],
                          [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])

    # Calculate intrinsic matrix.
    # 2 atan(35 / 2*32)
    scale = RESOLUTION_PCT / 100
    # print('scale', scale)
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    # print('f_u', f_u, 'f_v', f_v)
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))
    # R_world2obj =  np.transpose((get_inl(np.radians(-az)) @ get_az(np.radians((-el))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                           0,
                                           0)))
    # print('distance', distance_ratio * CAM_MAX_DIST)
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))

    # R: rotation
    # T: translation
    # K: camera calibration
    # The world points are transformed to camera coordinates using the extrinsics parameters. 
    # The camera coordinates are mapped into the image plane using the intrinsics parameters.

    return K, RT

def get_norm_matrix(sdf_h5_file):
    with h5py.File(sdf_h5_file, 'r') as h5_f:
        norm_params = h5_f['norm_params'][:]
        center, m, = norm_params[:3], norm_params[3]
        x,y,z = center[0], center[1], center[2]
        M_inv = np.asarray(
            [[m, 0., 0., 0.],
             [0., m, 0., 0.],
             [0., 0., m, 0.],
             [0., 0., 0., 1.]]
        )
        T_inv = np.asarray(
            [[1.0 , 0., 0., x],
             [0., 1.0 , 0., y],
             [0., 0., 1.0 , z],
             [0., 0., 0., 1.]]
        )
    return np.matmul(T_inv, M_inv)


def get_rotate_matrix(rotation_angle1):
    cosval = np.cos(rotation_angle1)
    sinval = np.sin(rotation_angle1)

    rotation_matrix_x = np.array([[1, 0,        0,      0],
                                  [0, cosval, -sinval, 0],
                                  [0, sinval, cosval, 0],
                                  [0, 0,        0,      1]])
    rotation_matrix_y = np.array([[cosval, 0, sinval, 0],
                                  [0,       1,  0,      0],
                                  [-sinval, 0, cosval, 0],
                                  [0,       0,  0,      1]])
    rotation_matrix_z = np.array([[cosval, -sinval, 0, 0],
                                  [sinval, cosval, 0, 0],
                                  [0,           0,  1, 0],
                                  [0,           0,  0, 1]])
    scale_y_neg = np.array([
        [1, 0,  0, 0],
        [0, -1, 0, 0],
        [0, 0,  1, 0],
        [0, 0,  0, 1]
    ])

    neg = np.array([
        [-1, 0,  0, 0],
        [0, -1, 0, 0],
        [0, 0,  -1, 0],
        [0, 0,  0, 1]
    ])
    # y,z swap = x rotate -90, scale y -1
    # new_pts0[:, 1] = new_pts[:, 2]
    # new_pts0[:, 2] = new_pts[:, 1]
    #
    # x y swap + negative = z rotate -90, scale y -1
    # new_pts0[:, 0] = - new_pts0[:, 1] = - new_pts[:, 2]
    # new_pts0[:, 1] = - new_pts[:, 0]

    # return np.linalg.multi_dot([rotation_matrix_z, rotation_matrix_y, rotation_matrix_y, scale_y_neg, rotation_matrix_z, scale_y_neg, rotation_matrix_x])
    return np.linalg.multi_dot([neg, rotation_matrix_z, rotation_matrix_z, scale_y_neg, rotation_matrix_x])
    # seems wrong


def get_img_cam(param):
    cam_mat, cam_pos = camera_info(degree2rad(param))

    return cam_mat, cam_pos


def degree2rad(params):
    params_new = np.zeros_like(params)
    params_new[0] = np.deg2rad(params[0] + 180.0)
    params_new[1] = np.deg2rad(params[1])
    params_new[2] = np.deg2rad(params[2])
    return params_new


def unit(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def camera_info(param):
    az_mat = get_az(param[0])
    el_mat = get_el(param[1])
    inl_mat = get_inl(param[2])
    cam_mat = np.transpose(np.matmul(np.matmul(inl_mat, el_mat), az_mat))
    cam_pos = get_cam_pos(param)
    return cam_mat, cam_pos


def get_cam_pos(param):
    camX = 0
    camY = 0
    camZ = param[3]  # this is distance ratio
    cam_pos = np.array([camX, camY, camZ])
    return -1 * cam_pos


def get_az(az):
    # theta, y axis, pitch
    cos = np.cos(az)
    sin = np.sin(az)
    mat = np.asarray([cos, 0.0, sin,
                      0.0, 1.0, 0.0,
                      -1.0 * sin, 0.0, cos], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat


def get_el(el):
    # psi, x axis, roll
    cos = np.cos(el)
    sin = np.sin(el)
    mat = np.asarray([1.0, 0.0, 0.0,
                      0.0, cos, -1.0 * sin,
                      0.0, sin, cos], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat


def get_inl(inl):
    # this is phi!
    # rotation around z axis / yaw
    cos = np.cos(inl)
    sin = np.sin(inl)
    # zeros = np.zeros_like(inl)
    # ones = np.ones_like(inl)
    mat = np.asarray([cos, -1.0 * sin, 0.0,
                      sin, cos, 0.0,
                      0.0, 0.0, 1.0], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat
