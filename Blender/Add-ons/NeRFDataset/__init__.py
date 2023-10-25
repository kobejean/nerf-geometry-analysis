import bpy
from . import helper, blender_nerf_ui, cos_ui, cos_operator
import mathutils


# blender info
bl_info = {
    'name': 'NeRFDataset',
    'description': 'Generate NeRF Datasets from blender',
    'author': 'Jean Atsumi Flaherty',
    'version': (1, 0, 0),
    'blender': (3, 0, 0),
    'location': '3D View > N panel > NeRFDataset',
    'doc_url': 'https://github.com/kobejean/nerf-geometry-analysis',
    'category': 'Object',
}

# global addon script variables
TRAIN_CAM = 'Train Cam'
TEST_CAM = 'Test Cam'
VERSION = '.'.join(str(x) for x in bl_info['version'])

class VectorPropertyGroup(bpy.types.PropertyGroup):
    vector: bpy.props.FloatVectorProperty(size=3)

class MatrixPropertyGroup(bpy.types.PropertyGroup):
    m00: bpy.props.FloatProperty(name="M00")
    m01: bpy.props.FloatProperty(name="M01")
    m02: bpy.props.FloatProperty(name="M02")
    
    m10: bpy.props.FloatProperty(name="M10")
    m11: bpy.props.FloatProperty(name="M11")
    m12: bpy.props.FloatProperty(name="M12")
    
    m20: bpy.props.FloatProperty(name="M20")
    m21: bpy.props.FloatProperty(name="M21")
    m22: bpy.props.FloatProperty(name="M22")

    def to_mathutils_matrix(self):
        mat = mathutils.Matrix()
        mat[0][0], mat[0][1], mat[0][2] = self.m00, self.m01, self.m02
        mat[1][0], mat[1][1], mat[1][2] = self.m10, self.m11, self.m12
        mat[2][0], mat[2][1], mat[2][2] = self.m20, self.m21, self.m22
        return mat

    def from_mathutils_matrix(self, mat):
        self.m00, self.m01, self.m02 = mat[0][0], mat[0][1], mat[0][2]
        self.m10, self.m11, self.m12 = mat[1][0], mat[1][1], mat[1][2]
        self.m20, self.m21, self.m22 = mat[2][0], mat[2][1], mat[2][2]


bpy.utils.register_class(VectorPropertyGroup)
bpy.utils.register_class(MatrixPropertyGroup)


# addon blender properties
PROPS = [
    # global controllable properties
    ('train_data', bpy.props.BoolProperty(name='Train', description='Construct the training data', default=True) ),
    ('val_data', bpy.props.BoolProperty(name='Val', description='Construct the validation data', default=True) ),
    ('test_data', bpy.props.BoolProperty(name='Test', description='Construct the testing data', default=True) ),
    ('aabb', bpy.props.IntProperty(name='AABB', description='AABB scale as defined in Instant NGP', default=4, soft_min=1, soft_max=128) ),
    ('render_frames', bpy.props.BoolProperty(name='Render Frames', description='Whether training frames should be rendered. If not selected, only the transforms.json files will be generated', default=True) ),
    ('logs', bpy.props.BoolProperty(name='Save Log File', description='Whether to create a log file containing information on the NeRFDataset run', default=False) ),
    ('nerf', bpy.props.BoolProperty(name='NeRF', description='Whether to export the camera transforms.json files in the defaut NeRF file format convention', default=False) ),
    ('save_path', bpy.props.StringProperty(name='Save Path', description='Path to the output directory in which the synthetic dataset will be stored', subtype='DIR_PATH') ),

    # global automatic properties
    ('init_frame_step', bpy.props.IntProperty(name='Initial Frame Step') ),
    ('init_output_path', bpy.props.StringProperty(name='Initial Output Path', subtype='DIR_PATH') ),
    ('rendering', bpy.props.BoolProperty(name='Rendering', description='Whether one of the SOF, TTC or COS methods is rendering', default=False) ),
    ('nerfdataset_version', bpy.props.StringProperty(name='NeRFDataset Version', default=VERSION) ),
    ('camera_train_target', bpy.props.PointerProperty(type=bpy.types.Object, name=TRAIN_CAM, description='Pointer to the training camera', poll=helper.poll_is_camera) ),
    ('camera_test_target', bpy.props.PointerProperty(type=bpy.types.Object, name=TEST_CAM, description='Pointer to the testing camera', poll=helper.poll_is_camera) ),

    # cos controllable properties
    ('cos_dataset_name', bpy.props.StringProperty(name='Name', description='Name of the COS dataset : the data will be stored under <save path>/<name>', default='dataset') ),
    ('camera_layout_mode', bpy.props.EnumProperty(
        items=[
            ("sphere", "Sphere", "Layout cameras in a spherical arrangement"),
            ("golden_spiral", "Golden Spiral Sphere", "Layout cameras in a hemispherical arrangement"),
            ("circle", "Circle", "Layout cameras in a circular arrangement"),
            ("stereo", "Stereo", "Layout cameras for stereo vision"),
            ("line", "Line", "Layout cameras along a line"),

        ],
        name="Camera Layout Mode",
        description="Choose the camera layout mode",
        default="sphere"
    )),
    ('geometry_analysis_type', bpy.props.EnumProperty(
        items=[
            ("sphere", "Sphere", "Allows for spherical geometry evaluation"),
            ("plane", "Plane", "Allows for planar geometry evaluation"),
            ("cube", "Cube", "Allows for cube geometry evaluation"),
            ("line", "Line", "Allows for line geometry evaluation"),
            ("unspecified", "Unspecified", "No specialized geometric evaluation"),
        ],
        name="Geometry Type",
        description="Choose the geometry type",
        default="unspecified"
    )),
    ('geometry_size', bpy.props.FloatVectorProperty(name='Geometry Size', description='Size of geometry', default=(1.0, 1.0, 1.0)) ),
    ('geometry_radius', bpy.props.FloatProperty(name='Geometry Radius', description='Radius of geometry', default=0.5, soft_min=0.01, unit='LENGTH') ),
    ('sphere_location', bpy.props.FloatVectorProperty(name='Location', description='Center position of the training sphere', unit='LENGTH', update=helper.properties_ui_upd) ),
    ('sphere_rotation', bpy.props.FloatVectorProperty(name='Rotation', description='Rotation of the training sphere', unit='ROTATION', update=helper.properties_ui_upd) ),
    ('sphere_scale', bpy.props.FloatVectorProperty(name='Scale', description='Scale of the training sphere in xyz axes', default=(1.0, 1.0, 1.0), update=helper.properties_ui_upd) ),
    ('sphere_radius', bpy.props.FloatProperty(name='Radius', description='Radius scale of the training sphere', default=3.0, soft_min=0.01, unit='LENGTH', update=helper.properties_ui_upd) ),
    ('focal', bpy.props.FloatProperty(name='Lens', description='Focal length of the training camera', default=36, soft_min=1, soft_max=5000, unit='CAMERA', update=helper.properties_ui_upd) ),
    ('seed', bpy.props.IntProperty(name='Seed', description='Random seed for sampling views on the training sphere', default=0) ),
    ('cos_nb_train_frames', bpy.props.IntProperty(name='Train Frames', description='Number of training frames randomly sampled from the training sphere', default=100, soft_min=1) ),
    ('cos_nb_val_frames', bpy.props.IntProperty(name='Validation Frames', description='Number of validation frames randomly sampled from the training sphere', default=10, soft_min=1) ),
    ('cos_nb_test_frames', bpy.props.IntProperty(name='Test Frames', description='Number of training frames randomly sampled from the training sphere', default=10, soft_min=1) ),
    ('min_altitude', bpy.props.FloatProperty(name='Minimum Altitude', description='Minimum altitude angle', default=10, soft_min=-90, soft_max=90) ),
    ('max_altitude', bpy.props.FloatProperty(name='Maximum Altitude', description='Maximum altitude angle', default=90, soft_min=-90, soft_max=90) ),
    ('show_sphere', bpy.props.BoolProperty(name='Sphere', description='Whether to show the training sphere from which random views will be sampled', default=False, update=helper.visualize_sphere) ),
    ('show_camera', bpy.props.BoolProperty(name='Camera', description='Whether to show the training camera', default=False, update=helper.visualize_camera) ),
    # ('upper_views', bpy.props.BoolProperty(name='Upper Views', description='Whether to sample views from the upper hemisphere of the training sphere only', default=False) ),
    ('outwards', bpy.props.BoolProperty(name='Outwards', description='Whether to point the camera outwards of the training sphere', default=False, update=helper.properties_ui_upd) ),

    # cos automatic properties
    ('sphere_exists', bpy.props.BoolProperty(name='Sphere Exists', description='Whether the sphere exists', default=False) ),
    ('init_sphere_exists', bpy.props.BoolProperty(name='Init sphere exists', description='Whether the sphere initially exists', default=False) ),
    ('camera_exists', bpy.props.BoolProperty(name='Camera Exists', description='Whether the camera exists', default=False) ),
    ('init_camera_exists', bpy.props.BoolProperty(name='Init camera exists', description='Whether the camera initially exists', default=False) ),
    ('init_active_camera', bpy.props.PointerProperty(type=bpy.types.Object, name='Init active camera', description='Pointer to initial active camera', poll=helper.poll_is_camera) ),
    ('init_frame_end', bpy.props.IntProperty(name='Initial Frame End') ),
    
    ('test_points', bpy.props.CollectionProperty(type=VectorPropertyGroup, name='Test Points') ),
    ('val_points', bpy.props.CollectionProperty(type=VectorPropertyGroup, name='Val Points') ),
    ('train_points', bpy.props.CollectionProperty(type=VectorPropertyGroup, name='Train Points') ),
    ('test_rotations', bpy.props.CollectionProperty(type=MatrixPropertyGroup, name='Test Rotation Matrices') ),
    ('val_rotations', bpy.props.CollectionProperty(type=MatrixPropertyGroup, name='Val Rotation Matrices') ),
    ('train_rotations', bpy.props.CollectionProperty(type=MatrixPropertyGroup, name='Train Rotation Matrices') ),

]

# classes to register / unregister
CLASSES = [
    blender_nerf_ui.NeRFDataset_UI,
    cos_ui.COS_UI,
    cos_operator.CameraOnSphere
]

# load addon
def register():
    for (prop_name, prop_value) in PROPS:
        setattr(bpy.types.Scene, prop_name, prop_value)

    for cls in CLASSES:
        bpy.utils.register_class(cls)

    # bpy.app.handlers.load_post.append(helper.setup_depth_map_rendering)
    bpy.app.handlers.render_complete.append(helper.post_render)
    bpy.app.handlers.render_cancel.append(helper.post_render)
    bpy.app.handlers.frame_change_post.append(helper.cos_camera_update)
    bpy.app.handlers.depsgraph_update_post.append(helper.properties_desgraph_upd)
    bpy.app.handlers.depsgraph_update_post.append(helper.set_init_props)

# deregister addon
def unregister():
    for (prop_name, _) in PROPS:
        delattr(bpy.types.Scene, prop_name)

    # bpy.app.handlers.load_post.remove(helper.setup_depth_map_rendering)
    bpy.app.handlers.render_complete.remove(helper.post_render)
    bpy.app.handlers.render_cancel.remove(helper.post_render)
    bpy.app.handlers.frame_change_post.remove(helper.cos_camera_update)
    bpy.app.handlers.depsgraph_update_post.remove(helper.properties_desgraph_upd)
    # bpy.app.handlers.depsgraph_update_post.remove(helper.set_init_props)

    for cls in CLASSES:
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()