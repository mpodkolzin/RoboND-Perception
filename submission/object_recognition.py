#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # Voxel Grid Downsampling

    vox = cloud.make_voxel_grid_filter()

    LEAF_SIZE = 0.01

    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    cloud_filtered = vox.filter()

    #filename = 'vox_downsampled.pcd'
    #pcl.save(cloud_filtered, filename)

    # PassThrough filter

    passthrough = cloud_filtered.make_passthrough_filter()

    filter_axis = 'z'

    axis_min = 0.6
    axis_max = 1.1

    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)

    cloud_filtered = passthrough.filter()

    # RANSAC plane segmentation

    seg = cloud_filtered.make_segmenter()

    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    pcl_data_table = cloud.extract(inliers, negative=False)
    pcl_data_objects = cloud.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(pcl_data_objects)
    tree = white_cloud.make_kdtree()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.015)
    ec.set_MinClusterSize(120)
    ec.set_MaxClusterSize(2000)
    cluster_indices = ec.Extract()

    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    ros_cloud_cluster = pcl_to_ros(cluster_cloud)
    ros_cloud_objects = pcl_to_ros(pcl_data_objects)
    ros_cloud_table = pcl_to_ros(pcl_data_table) 

    # TODO: Publish ROS messages
    pcl_cluster_pub.publish(ros_cloud_cluster)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)

# Exercise-3 TODOs: 

    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        pcl_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(pcl_cluster, using_hsv=False)
        normals = get_normals(pcl_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]

        detected_objects_labels.append(label)

        # Publish a label into RViz
	label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))
        

        # Add the detected object to the list of detected objects.
	do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('object_recognition', anonymous=True)


    # TODO: Create Publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)


    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
	rospy.spin()



