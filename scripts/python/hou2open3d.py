"""
Functions which bring Open3D functionality into Houdini

TODO:
    Do it in CPP via python? - http://www.sidefx.com/docs/houdini/hom/extendingwithcpp.html#python_sops
    Compare performance
"""

import hou
import open3d
import numpy as np

def read_point_cloud():
    """
    Read Open3d point clouds and covnert to Houdini geometry

    Based on http://www.open3d.org/docs/tutorial/Basic/working_with_numpy.html
    """
    node = hou.pwd()
    node_geo = node.geometry()
    path = node.parm("path").eval()
    
    pcd_load = open3d.read_point_cloud(path)
    
    if not pcd_load.has_points():
        raise hou.NodeWarning("Geometry does not contain any points.")

    # create numpy arrays
    np_pos = np.asarray(pcd_load.points)
    np_n = np.asarray(pcd_load.normals)
    np_cd = np.asarray(pcd_load.colors)

    # position
    node_geo.createPoints(np_pos)

    # normals
    if pcd_load.has_normals():
        node_geo.addAttrib(hou.attribType.Point, "N", default_value=(0.0, 0.0, 0.0), transform_as_normal=True, create_local_variable=False)
        node_geo.setPointFloatAttribValuesFromString("N", np_n, float_type=hou.numericData.Float64)
    
    # colors
    if pcd_load.has_colors():
        node_geo.addAttrib(hou.attribType.Point, "Cd", default_value=(0.0, 0.0, 0.0), transform_as_normal=False, create_local_variable=False)
        node_geo.setPointFloatAttribValuesFromString("Cd", np_cd, float_type=hou.numericData.Float64)

def preprocess_point_cloud():
    """
    Pre-process point cloud - downsample, estimate normals and calculate fpfh features

    Based on http://www.open3d.org/docs/tutorial/Advanced/global_registration.html#extract-geometric-feature
    """
    node = hou.pwd()
    node_geo = node.geometry()
    downsample = node.parm("downsample").eval()
    voxel_size = node.parm("voxel_size").eval()
    estimate_normals = node.parm("estimate_normals").eval()
    max_neighbours = node.parm("max_neighbours").eval()
    compute_fpfh_feature = node.parm("compute_fpfh_feature").eval()
    max_neighbours_fpfh = node.parm("max_neighbours_fpfh").eval()

    # check for attributes
    has_n = bool(node_geo.findPointAttrib("N"))
    has_cd = bool(node_geo.findPointAttrib("Cd"))

    # to numpy
    np_pos_str = node_geo.pointFloatAttribValuesAsString("P", float_type=hou.numericData.Float32)
    np_pos = np.fromstring(np_pos_str, dtype=np.float32).reshape(-1, 3)

    # to open3d
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(np_pos.astype(np.float64))
    
    if has_n:
        np_n_str = node_geo.pointFloatAttribValuesAsString("N", float_type=hou.numericData.Float32)
        np_n = np.fromstring(np_n_str, dtype=np.float32).reshape(-1, 3)
        pcd.normals = open3d.Vector3dVector(np_n.astype(np.float64))
    
    if has_cd:
        np_cd_str = node_geo.pointFloatAttribValuesAsString("Cd", float_type=hou.numericData.Float32)
        np_cd = np.fromstring(np_cd_str, dtype=np.float32).reshape(-1, 3)
        pcd.colors = open3d.Vector3dVector(np_cd.astype(np.float64))

    # preprocess
    if downsample:
        pcd = open3d.voxel_down_sample(pcd, voxel_size)
    
    if estimate_normals:
        open3d.estimate_normals(pcd, open3d.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=max_neighbours))

    pcd_fpfh = None
    if compute_fpfh_feature:
        pcd_fpfh = open3d.compute_fpfh_feature(pcd, open3d.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=max_neighbours_fpfh))
        np_fpfh = np.asarray(pcd_fpfh.data)
        np_fpfh = np.swapaxes(np_fpfh, 1, 0)

    # to houdini
    np_pos = np.asarray(pcd.points)
    np_n = np.asarray(pcd.normals)
    np_cd = np.asarray(pcd.colors)

    if downsample:
        node_geo.deletePoints(node_geo.points())
        node_geo.createPoints(np_pos)
    else:
        node_geo.setPointFloatAttribValuesFromString("P", np_pos, float_type=hou.numericData.Float64)
    
    if has_n or estimate_normals:
        if not has_n:
            node_geo.addAttrib(hou.attribType.Point, "N", default_value=(0.0, 0.0, 0.0), transform_as_normal=True, create_local_variable=False)
        node_geo.setPointFloatAttribValuesFromString("N", np_n, float_type=hou.numericData.Float64)

    if has_cd:
        node_geo.setPointFloatAttribValuesFromString("Cd", np_cd, float_type=hou.numericData.Float64)

    if compute_fpfh_feature:
        node_geo.addAttrib(hou.attribType.Point, "fpfh", default_value=(0.0, )*np_fpfh.shape[1], transform_as_normal=False, create_local_variable=False)
        node_geo.setPointFloatAttribValuesFromString("fpfh", np_fpfh, float_type=hou.numericData.Float64)

def fast_global_registration():
    """
    Execute fast global registration

    Based on http://www.open3d.org/docs/tutorial/Advanced/fast_global_registration.html
    """
    node = hou.pwd()
    node_geo = node.geometry()
    node_geo_target = node.inputs()[1].geometry()
    voxel_size = node.parm("voxel_size").eval()
    transform = node.parm("transform").eval()

    has_fpfh_source = bool(node_geo.findPointAttrib("fpfh"))
    has_fpfh_target = bool(node_geo_target.findPointAttrib("fpfh"))

    if not has_fpfh_source or not has_fpfh_target:
        raise hou.NodeError("One of the inputs does not have 'fpfh' attribute.")

    # to numpy
    np_pos_str_source = node_geo.pointFloatAttribValuesAsString("P", float_type=hou.numericData.Float32)
    np_pos_source = np.fromstring(np_pos_str_source, dtype=np.float32).reshape(-1, 3)
    np_fpfh_str_source = node_geo.pointFloatAttribValuesAsString("fpfh", float_type=hou.numericData.Float32)
    np_fpfh_size = node_geo.findPointAttrib("fpfh").size()
    np_fpfh_source = np.fromstring(np_fpfh_str_source, dtype=np.float32).reshape(-1, np_fpfh_size)
    np_fpfh_source = np.swapaxes(np_fpfh_source, 1, 0)

    np_pos_str_target = node_geo_target.pointFloatAttribValuesAsString("P", float_type=hou.numericData.Float32)
    np_pos_target = np.fromstring(np_pos_str_target, dtype=np.float32).reshape(-1, 3)
    np_fpfh_str_target = node_geo_target.pointFloatAttribValuesAsString("fpfh", float_type=hou.numericData.Float32)
    np_fpfh_target = np.fromstring(np_fpfh_str_target, dtype=np.float32).reshape(-1, np_fpfh_size)
    np_fpfh_target = np.swapaxes(np_fpfh_target, 1, 0)

    # to open3d
    source = open3d.PointCloud()
    source.points = open3d.Vector3dVector(np_pos_source.astype(np.float64))

    source_fpfh = open3d.registration.Feature()
    source_fpfh.resize(np_fpfh_source.shape[0], np_fpfh_source.shape[1])
    source_fpfh.data = np_fpfh_source.astype(np.float64)

    target = open3d.PointCloud()
    target.points = open3d.Vector3dVector(np_pos_target.astype(np.float64))

    target_fpfh = open3d.registration.Feature()
    target_fpfh.resize(np_fpfh_source.shape[0], np_fpfh_source.shape[1])
    target_fpfh.data = np_fpfh_target.astype(np.float64)

    # registration
    registration = open3d.registration_fast_based_on_feature_matching(source, target, source_fpfh, target_fpfh, open3d.FastGlobalRegistrationOption(maximum_correspondence_distance = voxel_size * 0.5))

    registration_xform = hou.Matrix4(registration.transformation)
    registration_xform = registration_xform.transposed()

    # to houdini
    if transform:
        node_geo.transform(registration_xform)
    
    node_geo.addAttrib(hou.attribType.Global, "xform", default_value=(0.0,)*16, create_local_variable=False)
    node_geo.setGlobalAttribValue("xform", registration_xform.asTuple())

def icp_registration():
    """
    Execute Point-to-plane ICP registration

    TODO:
        Add point-to-plane ICP option (and update normals)

    Based on http://www.open3d.org/docs/tutorial/Basic/icp_registration.html#point-to-plane-icp
    """
    node = hou.pwd()
    node_geo = node.geometry()
    node_geo_target = node.inputs()[1].geometry()
    threshold = node.parm("threshold").eval()
    transform = node.parm("transform").eval()
    max_iter = node.parm("max_iter").eval()
    single_pass = node.parm("single_pass").eval()

    has_xform_source = bool(node_geo.findGlobalAttrib("xform"))
    has_n_source = bool(node_geo.findPointAttrib("N"))
    has_n_target = bool(node_geo_target.findPointAttrib("N"))

    if not has_xform_source:
        node_geo.addAttrib(hou.attribType.Global, "xform", default_value=(0.0,)*16, create_local_variable=False)

    if not has_n_source or not has_n_target:
        raise hou.NodeError("One of the inputs does not have 'N' attribute.")
    
    trans_init = node_geo.floatListAttribValue("xform")
    trans_init = np.array(trans_init).reshape(4,4).T

    # to numpy
    np_pos_str_source = node_geo.pointFloatAttribValuesAsString("P", float_type=hou.numericData.Float32)
    np_pos_source = np.fromstring(np_pos_str_source, dtype=np.float32).reshape(-1, 3)
    np_n_str_source = node_geo.pointFloatAttribValuesAsString("N", float_type=hou.numericData.Float32)
    np_n_source = np.fromstring(np_n_str_source, dtype=np.float32).reshape(-1, 3)
    
    np_pos_str_target = node_geo_target.pointFloatAttribValuesAsString("P", float_type=hou.numericData.Float32)
    np_pos_target = np.fromstring(np_pos_str_target, dtype=np.float32).reshape(-1, 3)
    np_n_str_target = node_geo_target.pointFloatAttribValuesAsString("N", float_type=hou.numericData.Float32)
    np_n_target = np.fromstring(np_n_str_target, dtype=np.float32).reshape(-1, 3)

    # to open3d
    source = open3d.PointCloud()
    source.points = open3d.Vector3dVector(np_pos_source.astype(np.float64))
    source.normals = open3d.Vector3dVector(np_n_source.astype(np.float64))

    target = open3d.PointCloud()
    target.points = open3d.Vector3dVector(np_pos_target.astype(np.float64))
    target.normals = open3d.Vector3dVector(np_n_target.astype(np.float64))

    # icp
    #init_evaluation = open3d.evaluate_registration(source, target, threshold, trans_init)

    if single_pass:
        reg_p2l = open3d.registration_icp(source, target, threshold, np.identity(4), open3d.TransformationEstimationPointToPoint(), open3d.ICPConvergenceCriteria(max_iteration=1))
    else:
        reg_p2l = open3d.registration_icp(source, target, threshold, trans_init, open3d.TransformationEstimationPointToPoint(), open3d.ICPConvergenceCriteria(max_iteration=max_iter))

    # print init_evaluation
    # print reg_p2l

    # to houdini
    registration_xform = hou.Matrix4(reg_p2l.transformation)
    registration_xform = registration_xform.transposed()

    if transform:
        node_geo.transform(registration_xform)
    node_geo.setGlobalAttribValue("xform", registration_xform.asTuple())