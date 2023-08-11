import os
import shutil
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

        output_data = self.get_camera_intrinsics(scene, camera)

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
    
        if scene.test_data:
            if not scene.show_camera: scene.show_camera = True
            scene.frame_end = scene.frame_start + scene.cos_nb_test_frames - 1

            # test camera on sphere
            sphere_camera = scene.objects.get(CAMERA_NAME, camera)
            sphere_output_data = self.get_camera_intrinsics(scene, sphere_camera)
            scene.camera = sphere_camera

            # testing transforms
            sphere_output_data['frames'] = self.get_camera_extrinsics(scene, sphere_camera, mode='TEST', method='COS')
            self.save_json(output_path, 'transforms_test.json', sphere_output_data)

            # rendering
            if scene.render_frames:
                output_test = os.path.join(output_path, 'test')
                os.makedirs(output_test, exist_ok=True)
                scene.rendering = True
                scene.render.filepath = os.path.join(output_test, '') # training frames path
                bpy.ops.render.render('EXEC_DEFAULT', animation=True, write_still=True) # render scene

            scene.frame_start = scene.frame_end + 1

        if scene.train_data:
            if not scene.show_camera: scene.show_camera = True
            scene.frame_end = scene.frame_start + scene.cos_nb_train_frames - 1

            # train camera on sphere
            sphere_camera = scene.objects.get(CAMERA_NAME, camera)
            sphere_output_data = self.get_camera_intrinsics(scene, sphere_camera)
            scene.camera = sphere_camera

            # training transforms
            sphere_output_data['frames'] = self.get_camera_extrinsics(scene, sphere_camera, mode='TRAIN', method='COS')
            self.save_json(output_path, 'transforms_train.json', sphere_output_data)
            self.save_json(output_path, 'transforms.json', sphere_output_data)

            # rendering
            if scene.render_frames:
                output_train = os.path.join(output_path, 'train')
                os.makedirs(output_train, exist_ok=True)
                scene.rendering = True
                scene.render.filepath = os.path.join(output_train, '') # training frames path
                bpy.ops.render.render('EXEC_DEFAULT', animation=True, write_still=True) # render scene

            scene.frame_start = scene.frame_end + 1

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