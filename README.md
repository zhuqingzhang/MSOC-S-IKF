# MSOC-S-IKF

## Introduction

This repository contains the source code of the algorithm MSOC-S-IKF (Multiple State Observability Constraint-Schmidt-Invariant Kalman Filter), which is a consistent and efficient map-based visual inertial localization algorithm. This algorithm is based on the open-sourced framework [Open-VINS](https://github.com/rpng/open_vins).

The folder 'ov_rimsckf' is the main body of MSOC-S-IKF, where we implemented a right-invariant EKF version of Open-VINS, and the right-invriant EKF vins is combined with the Schmidt EKF and the Observability-Constraint technique to get our MSOC-S-IKF. 

The folder 'ov_msckf' contains an extended version of  the original Open-VINS, where we implemented a Schmidt EKF for  map-based visual inertial localization. This block can be used as a baseline to compared with the 'ri_msckf'.

The folder 'matches' provides feature matching information between the query sequences and the maps on  different [used dataset](#dataset) . 

The folder 'ov_eval' provides the modules for evaluating the algorithms performances. Compared with the original 'ov_eval' from Open-VINS, we added some tools to evaluate the performance of invariant-EKF-based algorithms, e.g., "error_invariant_singlerun" and "error_dataset". 

The folder 'ov_core' provides some basic modules for 'ri_msckf'/'msckf'. 

The folder 'docs' provides our original paper and the related supplementary material.



## Installation
This repository is currently only support ROS1. All the dependencies is the as those in Open-VINS. You can follow the guidance of [Open-VINS Installation](https://docs.openvins.com/gs-installing.html) to install the dependencies.


## Usage

```
$mkdir -p catkin_ws/src
$cd catkin_ws/src
$git clone https://github.com/zhuqingzhang/MSOC-S-IKF.git
$catkin_make
$source devel/setup.bash
$roslaunch ov_rimsckf pgeneva_ros_eth.launch
```

Note that before running the command of the last line, make sure the corresponding parameters in the launch file "pgeneva_ros_eth.launch" are correctly configured:

For each launch file, there are some key parameters need to be modified. 

* Make sure the played rosbag is correct.  User may modify the roslaunch parameters "bag", "bag_start" to decide which rosbag is selected and from which second of the rosbag to play. Alternately, users can choose to comment the rosbag play node in the roslaunch file, and open another terminal to play the rosbag. For example, to play the V203 sequence of EuRoC datasets, users can utilize the following command:

  ```
  $rosbag play V2_03_difficult.bag -s 0 --pause
  ```

* Modify the roslaunch parameter "pose_graph_filename" and "keyframe_pose_filename" to choose the correct matching information and map keyframe poses. For example, When we perform localization on V203 against V201, i.e.,  V201 is used as a prior map while V203 is online playing, then "pose_graph_filename" should be set as "V203_matches_with_V201.txt", and "keyframe_pose_filename" should be set as "V2_01_easy_cam.txt". 
* The parameter "use_schmidt" decides whether the algorithm use Schmidt update or not
* The parameter "use_prior_map" decides whether the algorithm is running with a prior map or just running as a pure odometry.































## Supplementary Material

The provided supplementary.pdf in the folder 'docs' is the SUPPLEMENTARY of our submitted manuscript "Toward Consistent and Efficient Map-based Visual-inertial Localization: Theory Framework and Filter Design" (Readers can find the first-version manuscript through [this link](https://arxiv.org/abs/2204.12108)). 
Alternatively, readers can also access the supplementary with [this link](https://drive.google.com/file/d/1TID9CVy3xAso9vs05qDj3s5gU1TJQBwh/view?usp=sharing). 


## <span id="dataset">Used Dataset</span>

In the paper, there are four kinds of datasets are used.

- [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

- [KAIST](https://sites.google.com/view/complex-urban-dataset)

- [4seasons](https://www.4seasons-dataset.com)

- Our own collected dataset, [YQ](https://www.aliyundrive.com/s/GqAnikLnb7k)

