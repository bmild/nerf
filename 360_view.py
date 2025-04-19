import sys, os
import json
import math
from typing import Dict, Optional, Tuple

import bpy
import mathutils
import numpy as np


class BlenderDatasetGenerator:
    """
    Used to create a dataset of train, test, and val
    images for a NeRF-like model.
    Models this can be used with (that we are sure of at least):
    - NeRF
    - SNeRG
    This kind of dataset may also be referred to as a "360", or "synthetic".
    Credit to Matthew Tancik for providing starter code: https://github.com/bmild/nerf/issues/78
    """

    # Define params
    DEBUG = False
    VIEWS = 500
    RESOLUTION = 800
    DEPTH_SCALE = 1.4
    COLOR_DEPTH = 8
    FORMAT = "PNG"
    RANDOM_VIEWS = True
    UPPER_VIEWS = True
    CIRCLE_FIXED_START = (0.3, 0, 0)

    def __init__(self, dataset_params: Optional[Dict[str, Tuple[int, bool]]]):
        """Maps the dataset split -> (num_images, include_depth_normal)."""
        if dataset_params is None:
            dataset_params = {
                "train": (125, False),
                "val": (125, False),
                "test": (250, True),
            }
        self.dataset_params = dataset_params

    def listify_matrix(self, matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list

    def parent_obj_to_camera(self, b_camera):
        origin = (0, 0, 0)
        b_empty = bpy.data.objects.new("Empty", None)
        b_empty.location = origin
        b_camera.parent = b_empty  # setup parenting

        scn = bpy.context.scene
        scn.collection.objects.link(b_empty)
        bpy.context.view_layer.objects.active = b_empty
        # scn.objects.active = b_empty
        return b_empty

    def generate_data_split(
        self, data_split: str, num_images: int, include_depth_normals=False
    ):
        """
        Generates images that go in a single split of the dataset.
        
        The rotation mode is 'XYZ.'
        
        Parameters:
            data_split(str): one of either "train", "val", or "test"
            num_images(int)
            include_depth_normals(bool): whether or not to dump albedo and
                                         normal images. Default is False.
                                         Note that settings this to True
                                         will triple the number of generated
                                         images.
        Returns: None
        """
        # Make dir to save synthetic images
        if not os.path.exists(self.fp):
            os.makedirs(self.fp)

        # Data to store in JSON file
        out_data = {
            "camera_angle_x": bpy.data.objects["Camera"].data.angle_x,
        }

        # Render Optimizations
        bpy.context.scene.render.use_persistent_data = True

        # Set up rendering of depth map.
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # Add passes for additionally dumping albedo and normals.
        bpy.context.scene.view_layers[
            "View Layer"    # might need to use "RenderLayer", on different versions
        ].use_pass_normal = (
            include_depth_normals
        )
        bpy.context.scene.render.image_settings.file_format = str(self.FORMAT)
        bpy.context.scene.render.image_settings.color_depth = str(self.COLOR_DEPTH)

        if not self.DEBUG:
            # Create input render layer node.
            render_layers = tree.nodes.new("CompositorNodeRLayers")

            depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
            depth_file_output.label = "Depth Output"
            if self.FORMAT == "OPEN_EXR":
                links.new(render_layers.outputs["Depth"], depth_file_output.inputs[0])
            else:
                # Remap as other types can not represent the full range of depth.
                map = tree.nodes.new(type="CompositorNodeMapValue")
                # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
                map.offset = [-0.7]
                map.size = [self.DEPTH_SCALE]
                map.use_min = True
                map.min = [0]
                links.new(render_layers.outputs["Depth"], map.inputs[0])

                links.new(map.outputs[0], depth_file_output.inputs[0])

            normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
            normal_file_output.label = "Normal Output"
            links.new(render_layers.outputs["Normal"], normal_file_output.inputs[0])

        # Background
        bpy.context.scene.render.dither_intensity = 0.0
        bpy.context.scene.render.film_transparent = True

        # Create collection for objects not to render with background
        objs = [
            ob
            for ob in bpy.context.scene.objects
            if ob.type in ("EMPTY") and "Empty" in ob.name
        ]
        bpy.ops.object.delete({"selected_objects": objs})

        scene = bpy.context.scene
        scene.render.resolution_x = self.RESOLUTION
        scene.render.resolution_y = self.RESOLUTION
        scene.render.resolution_percentage = 100

        cam = scene.objects["Camera"]
        cam.location = (0, 4.0, 0.5)
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        b_empty = self.parent_obj_to_camera(cam)
        cam_constraint.target = b_empty

        scene.render.image_settings.file_format = "PNG"  # set output format to .png

        stepsize = 360.0 / num_images

        if not self.DEBUG:
            for output_node in [depth_file_output, normal_file_output]:
                output_node.base_path = ""

        out_data["frames"] = []

        if not self.RANDOM_VIEWS:
            b_empty.rotation_euler = self.CIRCLE_FIXED_START

        # hide the cube that loads on default into the scene (on Blender 2.82 at least)
        if bpy.data.objects["Cube"] is not None:
            bpy.data.objects["Cube"].hide_render = True
            bpy.data.objects["Cube"].hide_viewport = True

        # Take the Pictures!
        for i in range(0, num_images):
            if self.DEBUG:
                i = np.random.randint(0, num_images)
                b_empty.rotation_euler[2] += math.radians(stepsize * i)
            if self.RANDOM_VIEWS:
                scene.render.filepath = self.fp + f"/{data_split}" + "/r_" + str(i)
                if self.UPPER_VIEWS:
                    rot = np.random.uniform(0, 1, size=3) * (1, 0, 2 * np.pi)
                    rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi / 2)
                    b_empty.rotation_euler = rot
                else:
                    b_empty.rotation_euler = np.random.uniform(0, 2 * np.pi, size=3)
            else:
                print(
                    "Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i))
                )
                scene.render.filepath = self.fp + "/r_" + str(i)

            if include_depth_normals:
                depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
                normal_file_output.file_slots[0].path = (
                    scene.render.filepath + "_normal_"
                )

            if self.DEBUG:
                break
            else:
                bpy.ops.render.render(write_still=True)  # render still

            frame_data = {
                "file_path": scene.render.filepath,
                "rotation": math.radians(stepsize),
                "transform_matrix": self.listify_matrix(cam.matrix_world),
            }
            out_data["frames"].append(frame_data)

            if self.RANDOM_VIEWS:
                if self.UPPER_VIEWS:
                    rot = np.random.uniform(0, 1, size=3) * (1, 0, 2 * np.pi)
                    rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi / 2)
                    b_empty.rotation_euler = rot
                else:
                    b_empty.rotation_euler = np.random.uniform(0, 2 * np.pi, size=3)
            else:
                b_empty.rotation_euler[2] += math.radians(stepsize)

        if not self.DEBUG:
            with open(self.fp + "/" + f"transforms_{data_split}.json", "w") as out_file:
                json.dump(out_data, out_file, indent=4)

    def generate_dataset(self, results_path):
        """Generates a synthetic dataset with train/test/val images.
        By default, this method will create:
        - 125 training images
        - 125 validation images
        - 250 test images (not including normal and albedo images,
          which makes for 600 PNGs in total)
        """
        self.fp = bpy.path.abspath(f"//{results_path}")
        for split, params in self.dataset_params.items():
            num_images, include_depth_normal = params
            self.generate_data_split(split, num_images, include_depth_normal)

    @classmethod
    def generate(cls, dataset_params=None, results_path):
        "Class-method equivalent of generate_dataset()."
        data_synthesizer = cls(dataset_params)
        data_synthesizer.generate_dataset(results_path)


# This is what we would actually call in Blender (or as an external script)
if __name__ == "__main__":
    # give the absolute path to the folder where you want the dataset saved
    path = "results"
    BlenderDatasetGenerator.generate(results_path=path)
