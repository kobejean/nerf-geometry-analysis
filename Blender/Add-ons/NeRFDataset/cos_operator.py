import os
import shutil
import math
import mathutils
import bpy
from . import helper, blender_nerf_operator


# global addon script variables
EMPTY_NAME = 'NeRFDataset Sphere'
CAMERA_NAME = 'NeRFDataset Camera'

# camera on sphere operator class
class CameraOnSphere(blender_nerf_operator.NeRFDataset_Operator):
    '''Camera on Sphere Operator'''
    bl_idname = 'object.camera_on_sphere'
    bl_label = 'Camera on Sphere COS'

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        # check if camera is selected : next errors depend on an existing camera
        if camera == None:
            self.report({'ERROR'}, 'Be sure to have a selected camera!')
            return {'FINISHED'}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='COS')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}
        
        if scene.camera_layout_mode == "sphere":
            helper.create_sphere_camera_points(scene)
        elif scene.camera_layout_mode == "circle":
            helper.create_circle_camera_points(scene)
        elif scene.camera_layout_mode == "stereo":
            helper.create_stereo_camera_points(scene)

        helper.setup_depth_map_rendering()

        output_data = self.get_camera_intrinsics(scene, camera)
        depth_map_file_output = helper.find_tagged_nodes(scene.node_tree, "depth_map_file_output")[0]

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.cos_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        os.makedirs(output_path, exist_ok=True)

        if scene.logs: self.save_log_file(scene, output_path, method='COS')

        # initial property might have changed since set_init_props update
        scene.init_output_path = scene.render.filepath

        # other intial properties
        scene.init_sphere_exists = scene.show_sphere
        scene.init_camera_exists = scene.show_camera
        scene.init_frame_end = scene.frame_end
        scene.init_active_camera = camera
        scene.frame_start = 0
        scene.frame_current = 0

        # camera on sphere
        sphere_camera = scene.objects.get(CAMERA_NAME, camera)
        sphere_output_data = self.get_camera_intrinsics(scene, sphere_camera)
        sphere_output_data['frames'] = []
        scene.camera = sphere_camera
    
        if scene.test_data:
            if not scene.show_camera: scene.show_camera = True
            scene.frame_end = scene.frame_start + scene.cos_nb_test_frames - 1

            # testing transforms
            test_frames = self.get_camera_extrinsics(scene, sphere_camera, mode='TEST', method='COS')
            sphere_output_data['frames'] += test_frames
            sphere_output_data['test_filenames'] = [frame["file_path"] for frame in test_frames]

            # rendering
            if scene.render_frames:
                output_test = os.path.join(output_path, 'test')
                os.makedirs(output_test, exist_ok=True)
                scene.rendering = True
                scene.render.filepath = os.path.join(output_test, '') # test frames path
                depth_map_file_output.base_path = output_test
                bpy.ops.render.render('EXEC_DEFAULT', animation=True, write_still=True) # render scene

            scene.frame_start = scene.frame_end + 1

        
        if scene.val_data:
            if not scene.show_camera: scene.show_camera = True
            scene.frame_end = scene.frame_start + scene.cos_nb_val_frames - 1

            # testing transforms
            val_frames = self.get_camera_extrinsics(scene, sphere_camera, mode='VAL', method='COS')
            sphere_output_data['frames'] += val_frames
            sphere_output_data['val_filenames'] = [frame["file_path"] for frame in val_frames]

            # rendering
            if scene.render_frames:
                output_val = os.path.join(output_path, 'val')
                os.makedirs(output_val, exist_ok=True)
                scene.rendering = True
                scene.render.filepath = os.path.join(output_val, '') # validation frames path
                depth_map_file_output.base_path = output_val
                bpy.ops.render.render('EXEC_DEFAULT', animation=True, write_still=True) # render scene

            scene.frame_start = scene.frame_end + 1

        if scene.train_data:
            if not scene.show_camera: scene.show_camera = True
            scene.frame_end = scene.frame_start + scene.cos_nb_train_frames - 1

            # training transforms
            train_frames = self.get_camera_extrinsics(scene, sphere_camera, mode='TRAIN', method='COS')
            sphere_output_data['frames'] += train_frames
            sphere_output_data['train_filenames'] = [frame["file_path"] for frame in train_frames]

            # rendering
            if scene.render_frames:
                output_train = os.path.join(output_path, 'train')
                os.makedirs(output_train, exist_ok=True)
                scene.rendering = True
                scene.render.filepath = os.path.join(output_train, '') # training frames path
                depth_map_file_output.base_path = output_train
                bpy.ops.render.render('EXEC_DEFAULT', animation=True, write_still=True) # render scene

            scene.frame_start = scene.frame_end + 1

        
        self.save_json(output_path, 'transforms.json', sphere_output_data)

        # if frames are rendered, the below code is executed by the handler function
        if not scene.rendering:
            # reset camera settings
            if not scene.init_camera_exists: helper.delete_camera(scene, CAMERA_NAME)
            if not scene.init_sphere_exists:
                objects = bpy.data.objects
                objects.remove(objects[EMPTY_NAME], do_unlink=True)
                scene.show_sphere = False
                scene.sphere_exists = False

            scene.camera = scene.init_active_camera

            # compress dataset and remove folder (only keep zip)
            # shutil.make_archive(output_path, 'zip', output_path) # output filename = output_path
            # shutil.rmtree(output_path)

        return {'FINISHED'}