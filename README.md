# MSOC-S-IKF

## Introduction

This repository contains the source code of the algorithm MSOC-S-IKF (Multiple State Observability Constraint-Schmidt-Invariant Kalman Filter), which is a consistent and efficient map-based visual inertial localization algorithm. This algorithm is based on the open-sourced framework [Open-VINS](https://github.com/rpng/open_vins).

The folder 'ov_rimsckf' is the main body of MSOC-S-IKF, where we implemented a right-invariant EKF version of Open-VINS, and the right-invriant EKF vins is combined with the Schmidt EKF and the Observability-Constraint technique to get our MSOC-S-IKF. 

The folder 'ov_msckf' contains a extended version of  the original Open-VINS, where we implemented a Schmidt EKF for  map-based visual inertial localization. This block can be used as a baseline to compared with the 'ri_msckf'.

The folder 'matches' provides feature matching information between the query sequences and the maps on  different [used dataset](#dataset) . 

The folder 'ov_eval' provides the modules for evaluating the algorithms performances. Compared with the original 'ov_eval' from Open-VINS, we added some tools to evaluate the performance of invariant-EKF-based algorithms, e.g., "error_invariant_singlerun" and "error_dataset". 

The folder 'ov_core' provides some basic modules for 'ri_msckf'/'msckf'. 

The folder 'docs' provides our original paper and the related supplementary material.



## Installation



## Usage



































## Supplementary Material

The provided supplementary.pdf in the folder 'docs' is the SUPPLEMENTARY of our submitted manuscript "Toward Consistent and Efficient Map-based Visual-inertial Localization: Theory Framework and Filter Design" (Readers can find the first-version manuscript through [this link](https://arxiv.org/abs/2204.12108)). 
Alternatively, readers can also access the supplementrary with [this link](https://drive.google.com/file/d/1TID9CVy3xAso9vs05qDj3s5gU1TJQBwh/view?usp=sharing). 


## <span id="dataset">Used Dataset</span>

In the paper, there are four kinds of datasets are used.

- [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

- [KAIST](https://sites.google.com/view/complex-urban-dataset)

- [4seasons](https://www.4seasons-dataset.com)

- Our own collected dataset, [YQ]()

