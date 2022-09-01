#!/usr/bin/python

import rospy
from rospy import Subscriber, Publisher
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import yaml
import numpy as np

rospy.init_node('zed_split')


def cb(msg):
    now = msg.header.stamp
    ci_l.header.stamp = now
    ci_r.header.stamp = now
    pub_ci_l.publish(ci_l)
    pub_ci_r.publish(ci_r)


def load_yaml(filename):
    stream = file(filename, 'r')
    f = yaml.load(stream,Loader=yaml.FullLoader)
    return f


def getMatrix(src, name):
    rows = src[name]['rows']
    cols = src[name]['cols']
    return np.array(src[name]['data']).reshape((rows, cols))


calib_l = load_yaml('/home/zzq/Code/map_open_vins_v2_ros/src/open_vins/ov_msckf/pointgreyleft.yaml')
calib_r = load_yaml('/home/zzq/Code/map_open_vins_v2_ros/src/open_vins/ov_msckf/pointgreyright.yaml')

bridge = CvBridge()

camMat1 = getMatrix(calib_l, 'camera_matrix')
dis1 = getMatrix(calib_l, 'distortion_coefficients')
R1 = getMatrix(calib_l, 'rectification_matrix')
P1 = getMatrix(calib_l, 'projection_matrix')
camMat2 = getMatrix(calib_r, 'camera_matrix')
dis2 = getMatrix(calib_r, 'distortion_coefficients')
R2 = getMatrix(calib_r, 'rectification_matrix')
P2 = getMatrix(calib_l, 'projection_matrix')

new_size = (648, 314)
map_l1, map_l2 = cv2.initUndistortRectifyMap(camMat1, dis1, R1, P1, new_size, cv2.CV_32FC1)
map_r1, map_r2 = cv2.initUndistortRectifyMap(camMat2, dis2, R2, P2, new_size, cv2.CV_32FC1)


def fill_camera_info(calib):
    ci = CameraInfo()
    ci.width = int(calib['image_width'])
    ci.height = int(calib['image_height'])
    ci.distortion_model = calib['distortion_model']
    ci.D = calib['distortion_coefficients']['data']
    ci.K = calib['camera_matrix']['data']
    ci.R = calib['rectification_matrix']['data']
    ci.P = calib['projection_matrix']['data']
    return ci

ci_l = fill_camera_info(calib_l)
ci_r = fill_camera_info(calib_r)

sub = Subscriber('/camera/left/image_raw', Image, cb)
pub_ci_l = Publisher('/camera/left/camera_info', CameraInfo, queue_size=10)
pub_ci_r = Publisher('/camera/right/camera_info', CameraInfo, queue_size=10)


rospy.spin()

