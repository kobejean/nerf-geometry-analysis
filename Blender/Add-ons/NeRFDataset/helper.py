import os
import shutil
import random
import math
import mathutils
import bpy
from bpy.app.handlers import persistent


# global addon script variables
EMPTY_NAME = 'NeRFDataset Sphere'
CAMERA_NAME = 'NeRFDataset Camera'

def create_sphere_camera_points(scene):
    num_test = scene.cos_nb_test_frames if scene.test_data else 0
    num_val = scene.cos_nb_val_frames if scene.val_data else 0
    num_train = scene.cos_nb_train_frames if scene.train_data else 0
    
    total = num_test + num_val + num_train
    segments = num_test + num_val

    # calculate angles
    max_altitude = scene.min_altitude * math.pi / 180
    min_altitude = scene.max_altitude * math.pi / 180
    spread_altitude = max_altitude - min_altitude
    rings = scene.cos_nb_train_frames // segments

    test = []
    val = []
    train = []
    scene.test_points.clear()
    scene.val_points.clear()
    scene.train_points.clear()
    for i in range(total):
        theta = 2 * math.pi * float(i % segments) / segments 
        altitude = min_altitude + spread_altitude * (1 - float(i // segments) / rings)
        phi = math.pi/2 - altitude

        # sample from unit sphere, given theta and phi
        unit_x = math.cos(theta) * math.sin(phi)
        unit_y = math.sin(theta) * math.sin(phi)
        unit_z = math.cos(phi)
        unit = mathutils.Vector((unit_x, unit_y, unit_z))

        # ellipsoid sample : center + rotation @ radius * unit sphere
        point = scene.sphere_radius * mathutils.Vector(scene.sphere_scale) * unit
        rotation = mathutils.Euler(scene.sphere_rotation).to_matrix()
        point = mathutils.Vector(scene.sphere_location) + rotation @ point

        if i < num_test:
            item = scene.test_points.add()
            item.vector = point
        elif i < num_test + num_val:
            item = scene.val_points.add()
            item.vector = point
        else:
            item = scene.train_points.add()
            item.vector = point


def create_circle_camera_points(scene):
    num_test = scene.cos_nb_test_frames if scene.test_data else 0
    num_val = scene.cos_nb_val_frames if scene.val_data else 0
    num_train = scene.cos_nb_train_frames if scene.train_data else 0
    
    total = num_test + num_val + num_train
    segments = num_test + num_val

    # calculate angles

    scene.test_points.clear()
    scene.val_points.clear()
    scene.train_points.clear()        
    
    phi = math.pi / 3

    for i in range(num_test):
        theta = 2 * math.pi * (float(i % segments) / segments + 0.5 / num_train)

        # sample from unit sphere, given theta and phi
        unit_x = math.cos(theta) * math.sin(phi)
        unit_y = math.sin(theta) * math.sin(phi)
        unit_z = math.cos(phi)
        unit = mathutils.Vector((unit_x, unit_y, unit_z))

        # ellipsoid sample : center + rotation @ radius * unit sphere
        point = scene.sphere_radius * mathutils.Vector(scene.sphere_scale) * unit
        rotation = mathutils.Euler(scene.sphere_rotation).to_matrix()
        point = mathutils.Vector(scene.sphere_location) + rotation @ point

        # add point
        item = scene.test_points.add()
        item.vector = point

    for i in range(num_val):
        theta = 2 * math.pi * (float(i % segments) / segments + 0.5 / num_train)

        # sample from unit sphere, given theta and phi
        unit_x = math.cos(theta) * math.sin(phi)
        unit_y = math.sin(theta) * math.sin(phi)
        unit_z = math.cos(phi)
        unit = mathutils.Vector((unit_x, unit_y, unit_z))

        # ellipsoid sample : center + rotation @ radius * unit sphere
        point = scene.sphere_radius * mathutils.Vector(scene.sphere_scale) * unit
        rotation = mathutils.Euler(scene.sphere_rotation).to_matrix()
        point = mathutils.Vector(scene.sphere_location) + rotation @ point

        # add point
        item = scene.val_points.add()
        item.vector = point

    
    for i in range(num_train):
        theta = 2 * math.pi * float(i % num_train) / num_train 

        # sample from unit sphere, given theta and phi
        unit_x = math.cos(theta) * math.sin(phi)
        unit_y = math.sin(theta) * math.sin(phi)
        unit_z = math.cos(phi)
        unit = mathutils.Vector((unit_x, unit_y, unit_z))

        # ellipsoid sample : center + rotation @ radius * unit sphere
        point = scene.sphere_radius * mathutils.Vector(scene.sphere_scale) * unit
        rotation = mathutils.Euler(scene.sphere_rotation).to_matrix()
        point = mathutils.Vector(scene.sphere_location) + rotation @ point

        # add point
        item = scene.train_points.add()
        item.vector = point


def create_stereo_camera_points(scene):
    num_test = scene.cos_nb_test_frames if scene.test_data else 0
    num_val = scene.cos_nb_val_frames if scene.val_data else 0
    num_train = scene.cos_nb_train_frames if scene.train_data else 0
    
    total = num_test + num_val + num_train
    segments = num_test + num_val

    # calculate angles

    scene.test_points.clear()
    scene.val_points.clear()
    scene.train_points.clear()        
    
    phi = math.pi / 3

    for i in range(num_test):
        theta = 2 * math.pi * (float(i % segments) / segments + 0.5 / num_train)

        # sample from unit sphere, given theta and phi
        unit_x = math.cos(theta) * math.sin(phi)
        unit_y = math.sin(theta) * math.sin(phi)
        unit_z = math.cos(phi)
        unit = mathutils.Vector((unit_x, unit_y, unit_z))

        # ellipsoid sample : center + rotation @ radius * unit sphere
        point = scene.sphere_radius * mathutils.Vector(scene.sphere_scale) * unit
        rotation = mathutils.Euler(scene.sphere_rotation).to_matrix()
        point = mathutils.Vector(scene.sphere_location) + rotation @ point

        # add point
        item = scene.test_points.add()
        item.vector = point

    for i in range(num_val):
        theta = 2 * math.pi * (float(i % segments) / segments + 0.5 / num_train)

        # sample from unit sphere, given theta and phi
        unit_x = math.cos(theta) * math.sin(phi)
        unit_y = math.sin(theta) * math.sin(phi)
        unit_z = math.cos(phi)
        unit = mathutils.Vector((unit_x, unit_y, unit_z))

        # ellipsoid sample : center + rotation @ radius * unit sphere
        point = scene.sphere_radius * mathutils.Vector(scene.sphere_scale) * unit
        rotation = mathutils.Euler(scene.sphere_rotation).to_matrix()
        point = mathutils.Vector(scene.sphere_location) + rotation @ point

        # add point
        item = scene.val_points.add()
        item.vector = point

    
    for i in range(num_train):
        theta = 2 * math.pi * float(i % num_train) / num_train 

        # sample from unit sphere, given theta and phi
        unit_x = math.cos(theta) * math.sin(phi)
        unit_y = math.sin(theta) * math.sin(phi)
        unit_z = math.cos(phi)
        unit = mathutils.Vector((unit_x, unit_y, unit_z))

        # ellipsoid sample : center + rotation @ radius * unit sphere
        point = scene.sphere_radius * mathutils.Vector(scene.sphere_scale) * unit
        rotation = mathutils.Euler(scene.sphere_rotation).to_matrix()
        point = mathutils.Vector(scene.sphere_location) + rotation @ point

        # add point
        item = scene.train_points.add()
        item.vector = point



## property poll and update functions

# camera pointer property poll function
def poll_is_camera(self, obj):
    return obj.type == 'CAMERA'

def visualize_sphere(self, context):
    scene = context.scene

    if EMPTY_NAME not in scene.objects.keys() and not scene.sphere_exists:
        # if empty sphere does not exist, create
        bpy.ops.object.empty_add(type='SPHERE') # non default location, rotation and scale here are sometimes not applied, so we enforce them manually below
        empty = context.active_object
        empty.name = EMPTY_NAME
        empty.location = scene.sphere_location
        empty.rotation_euler = scene.sphere_rotation
        empty.scale = scene.sphere_scale
        empty.empty_display_size = scene.sphere_radius

        scene.sphere_exists = True

    elif EMPTY_NAME in scene.objects.keys() and scene.sphere_exists:
        if CAMERA_NAME in scene.objects.keys() and scene.camera_exists:
            delete_camera(scene, CAMERA_NAME)

        objects = bpy.data.objects
        objects.remove(objects[EMPTY_NAME], do_unlink=True)

        scene.sphere_exists = False

def visualize_camera(self, context):
    scene = context.scene

    if CAMERA_NAME not in scene.objects.keys() and not scene.camera_exists:
        if EMPTY_NAME not in scene.objects.keys():
            scene.show_sphere = True

        bpy.ops.object.camera_add()
        camera = context.active_object
        camera.name = CAMERA_NAME
        camera.data.name = CAMERA_NAME
        camera.location = sample_point(scene)
        bpy.data.cameras[CAMERA_NAME].lens = scene.focal

        cam_constraint = camera.constraints.new(type='TRACK_TO')
        cam_constraint.track_axis = 'TRACK_Z' if scene.outwards else 'TRACK_NEGATIVE_Z'
        cam_constraint.up_axis = 'UP_Y'
        cam_constraint.target = bpy.data.objects[EMPTY_NAME]

        scene.camera_exists = True

    elif CAMERA_NAME in scene.objects.keys() and scene.camera_exists:
        objects = bpy.data.objects
        objects.remove(objects[CAMERA_NAME], do_unlink=True)

        for block in bpy.data.cameras:
            if CAMERA_NAME in block.name:
                bpy.data.cameras.remove(block)

        scene.camera_exists = False

def delete_camera(scene, name):
    objects = bpy.data.objects
    objects.remove(objects[name], do_unlink=True)

    scene.show_camera = False
    scene.camera_exists = False

    for block in bpy.data.cameras:
        if name in block.name:
            bpy.data.cameras.remove(block)

# non uniform sampling when stretched or squeezed sphere
def sample_point(scene):
    num_test = scene.cos_nb_test_frames if scene.test_data else 0
    num_val = scene.cos_nb_val_frames if scene.val_data else 0
    num_train = scene.cos_nb_train_frames if scene.train_data else 0
    print(scene.frame_current)

    if scene.frame_current < num_test:
        i = scene.frame_current
        return scene.test_points[i].vector
    elif scene.frame_current < num_test + num_val:
        i = scene.frame_current - num_test
        return scene.val_points[i].vector
    else:
        i = scene.frame_current - num_test - num_val
        return scene.train_points[i].vector

## two way property link between sphere and ui (property and handler functions)
# https://blender.stackexchange.com/questions/261174/2-way-property-link-or-a-filtered-property-display

def properties_ui_upd(self, context):
    can_scene_upd(self, context)

@persistent
def properties_desgraph_upd(scene):
    can_properties_upd(scene)

def properties_ui(self, context):
    scene = context.scene

    if EMPTY_NAME in scene.objects.keys():
        upd_off()
        bpy.data.objects[EMPTY_NAME].location = scene.sphere_location
        bpy.data.objects[EMPTY_NAME].rotation_euler = scene.sphere_rotation
        bpy.data.objects[EMPTY_NAME].scale = scene.sphere_scale
        bpy.data.objects[EMPTY_NAME].empty_display_size = scene.sphere_radius
        upd_on()

    if CAMERA_NAME in scene.objects.keys():
        upd_off()
        bpy.data.cameras[CAMERA_NAME].lens = scene.focal
        bpy.context.scene.objects[CAMERA_NAME].constraints['Track To'].track_axis = 'TRACK_Z' if scene.outwards else 'TRACK_NEGATIVE_Z'
        upd_on()

# if empty sphere modified outside of ui panel, edit panel properties
def properties_desgraph(scene):
    if scene.show_sphere and EMPTY_NAME in scene.objects.keys():
        upd_off()
        scene.sphere_location = bpy.data.objects[EMPTY_NAME].location
        scene.sphere_rotation = bpy.data.objects[EMPTY_NAME].rotation_euler
        scene.sphere_scale = bpy.data.objects[EMPTY_NAME].scale
        scene.sphere_radius = bpy.data.objects[EMPTY_NAME].empty_display_size
        upd_on()

    if scene.show_camera and CAMERA_NAME in scene.objects.keys():
        upd_off()
        scene.focal = bpy.data.cameras[CAMERA_NAME].lens
        scene.outwards = (bpy.context.scene.objects[CAMERA_NAME].constraints['Track To'].track_axis == 'TRACK_Z')
        upd_on()

    if EMPTY_NAME not in scene.objects.keys() and scene.sphere_exists:
        if CAMERA_NAME in scene.objects.keys() and scene.camera_exists:
            delete_camera(scene, CAMERA_NAME)

        scene.show_sphere = False
        scene.sphere_exists = False

    if CAMERA_NAME not in scene.objects.keys() and scene.camera_exists:
        scene.show_camera = False
        scene.camera_exists = False

        for block in bpy.data.cameras:
            if CAMERA_NAME in block.name:
                bpy.data.cameras.remove(block)

    if CAMERA_NAME in scene.objects.keys():
        scene.objects[CAMERA_NAME].location = sample_point(scene)

def empty_fn(self, context): pass

can_scene_upd = properties_ui
can_properties_upd = properties_desgraph

def upd_off():  # make sub function to an empty function
    global can_scene_upd, can_properties_upd
    can_scene_upd = empty_fn
    can_properties_upd = empty_fn
def upd_on():
    global can_scene_upd, can_properties_upd
    can_scene_upd = properties_ui
    can_properties_upd = properties_desgraph


## blender handler functions

# reset properties back to intial
@persistent
def post_render(scene):
    if scene.rendering: # execute this function only when rendering with addon
        method_dataset_name = scene.cos_dataset_name

        scene.rendering = False
        scene.render.filepath = scene.init_output_path # reset filepath


# set initial property values (bpy.data and bpy.context require a loaded scene)
@persistent
def set_init_props(scene):
    filepath = bpy.data.filepath
    filename = bpy.path.basename(filepath)
    default_save_path = filepath[:-len(filename)] # remove file name from blender file path = directoy path

    scene.save_path = default_save_path
    scene.init_frame_step = scene.frame_step
    scene.init_output_path = scene.render.filepath

    bpy.app.handlers.depsgraph_update_post.remove(set_init_props)

# update cos camera when changing frame
@persistent
def cos_camera_update(scene):
    if CAMERA_NAME in scene.objects.keys():
        scene.objects[CAMERA_NAME].location = sample_point(scene)

def find_tagged_nodes(node_tree, tag_value):
    tagged_nodes = []
    for node in node_tree.nodes:
        if "tag" in node.keys() and node["tag"] == tag_value:
            tagged_nodes.append(node)
    return tagged_nodes


def setup_depth_map_rendering():
    print("setup_depth_map_rendering")
    bpy.context.view_layer.use_pass_z = True

    links = None
    needs_setup = not bpy.context.scene.use_nodes

    # Enable Node Usage for the Scene
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # If we have already setup then return
    if find_tagged_nodes(tree, "depth_map_file_output"):
        return


    # Add New Nodes for Depth Map
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    # Add File Output Node
    file_output = tree.nodes.new('CompositorNodeOutputFile')
    file_output.format.file_format = 'OPEN_EXR'
    file_output.format.color_depth = '32'
    file_output.file_slots[0].path = "depth_####"
    file_output["tag"] = "depth_map_file_output"

    if needs_setup:
        composite = tree.nodes.new('CompositorNodeComposite')
        links.new(render_layers.outputs['Image'], composite.inputs['Image'])

    # Link Nodes
    links.new(render_layers.outputs['Depth'], file_output.inputs[0])
