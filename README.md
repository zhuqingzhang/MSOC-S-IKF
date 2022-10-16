# MSOC-S-IKF

## Introduction

This repository contains the source code of the algorithm __MSOC-S-IKF__ (Multiple State Observability Constraint-Schmidt-Invariant Kalman Filter), which is a consistent and efficient map-based visual inertial localization algorithm. This algorithm is based on the open-sourced framework [Open-VINS](https://github.com/rpng/open_vins). For the detailed introduction and related theories of MSOC-S-IKF, readers can refer to our submitted paper "__Toward Consistent and Efficient Map-based Visual-inertial Localization: Theory Framework and Filter Design__" (The first-version manuscript can be accessed through [this link](https://arxiv.org/abs/2204.12108)).

The folder '__ov_rimsckf__' is the main body of __MSOC-S-IKF__, where we implemented a right-invariant EKF version of Open-VINS, and the right-invriant EKF vins is combined with the Schmidt EKF and the Observability-Constraint technique to get our __MSOC-S-IKF__. 

The folder '**ov_msckf**' contains an extended version of  the original Open-VINS, where we implemented a Schmidt EKF for  map-based visual inertial localization. This block can be used as a baseline to compare with the '**ov_rimsckf**'. The algorithms "**MSC-EKF**" and "**MSC-S-EKF**" in our manuscript are from this package.

The folder '**matches**' provides feature matching information between the query sequences and the maps on  different [used dataset](#dataset) . 

The folder '**ov_eval**' provides the modules for evaluating the algorithms performances. Compared with the original 'ov_eval' from Open-VINS, we added some tools to evaluate the performance of invariant-EKF-based algorithms, e.g., "error_invariant_singlerun" and "error_dataset". 

The folder '**ov_core**' provides some basic modules for '**ov_rimsckf**'/'**ov_msckf**'. 

The folder '**docs**' provides our original paper and the related supplementary material.

## Supplementary Material

The provided **supplementary.pdf** in the folder 'docs' is the SUPPLEMENTARY of our submitted paper "**Toward Consistent and Efficient Map-based Visual-inertial Localization: Theory Framework and Filter Design**". Alternatively, readers can also access the supplementary with [this link](https://drive.google.com/file/d/1TID9CVy3xAso9vs05qDj3s5gU1TJQBwh/view?usp=sharing).  

## Installation
This repository is currently only support ROS1. All the dependencies is the as those in Open-VINS. You can follow the guidance of [Open-VINS Installation](https://docs.openvins.com/gs-installing.html) to install the dependencies.


## Usage

```
$mkdir -p catkin_ws/src
$cd catkin_ws/src
$git clone https://github.com/zhuqingzhang/MSOC-S-IKF.git
$catkin_make
$source devel/setup.bash
##For EuRoC dataset, run the following command
$roslaunch ov_rimsckf pgeneva_ros_eth.launch
##For Kaist dataset, run the following command
$roslaunch ov_rimsckf pgeneva_ros_kaist.launch
##For 4Seasons dataset, run the following command
$roslaunch ov_rimsckf pgeneva_ros_4seasons.launch
##For YQ dataset, run the following command
$roslaunch ov_rimsckf pgeneva_ros_YQ.launch
##For the simulation data, run the following command
@roslaunch ov_rimsckf pgeneva_sim_rect_circle.launch
```

### parameters

Note that before running the roslaunch file, make sure the corresponding parameters in the launch file (e.g., "pgeneva_ros_eth.launch") are correctly configured:

For each launch file, there are some key parameters need to be modified. 

* Make sure the played rosbag is correct.  User may modify the roslaunch parameters "**bag**", "**bag_start**" to decide which rosbag is selected and from which second of the rosbag to play. Alternately, users can choose to comment the rosbag play node in the roslaunch file, and open another terminal to play the rosbag. For example, to play the V203 sequence of EuRoC datasets from the beginning, users can utilize the following command:

  ```
  $rosbag play V2_03_difficult.bag -s 0 --pause
  ```

  For Kaist dataset, users should utilize the official file player provided [here](https://github.com/irapkaist/file_player) to play the bag.

* Modify the roslaunch parameter "**pose_graph_filename**" and "**keyframe_pose_filename**" to choose the correct matching information and map keyframe poses. For example, When we perform localization on V203 against V201, i.e.,  V201 is used as a prior map while V203 is online playing, then "**pose_graph_filename**" should be set as "V203_matches_with_V201.txt", and "**keyframe_pose_filename**" should be set as "V2_01_easy_cam.txt". 

* The parameter "**use_schmidt**" decides whether the algorithm use Schmidt update or not.  If the users use the package "**ov_msckf**" and set "**use_schmidt**" to be "true"/"false", then, the user is running the algorithm "**MSC-S-EKF**"/"**MSC-EKF**" mentioned in our paper. Similarly, if the users use the package "**ov_rimsckf**" and set "**use_schmidt**" to be "true"/"false", then, the user is running the algorithm "**MSOC-S-IKF**"/"**MSC-IKF**" mentioned in our paper. 

* The parameter "**use_prior_map**" decides whether the algorithm is running with a prior map or just running as a pure odometry.



## Example




## <span id="dataset">Used Dataset</span>

In the paper, there are four kinds of datasets are used.

- [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

- [KAIST](https://sites.google.com/view/complex-urban-dataset)

- [4seasons](https://www.4seasons-dataset.com)

- Our own collected dataset, [YQ](https://www.aliyundrive.com/s/GqAnikLnb7k)

