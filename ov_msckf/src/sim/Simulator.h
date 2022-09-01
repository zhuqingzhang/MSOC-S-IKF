/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2019 Patrick Geneva
 * Copyright (C) 2019 Kevin Eckenhoff
 * Copyright (C) 2019 Guoquan Huang
 * Copyright (C) 2019 OpenVINS Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#ifndef OV_MSCKF_SIMULATOR_H
#define OV_MSCKF_SIMULATOR_H


#include <fstream>
#include <sstream>
#include <random>
#include <string>
#include <unordered_map>

#include <Eigen/Eigen>
#include <Eigen/StdVector>

#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>


#include "core/VioManagerOptions.h"
#include "sim/BsplineSE3.h"
#include "utils/colors.h"
#include <fstream>

using namespace ov_core;


namespace ov_msckf {



    /**
     * @brief Master simulator class that generated visual-inertial measurements
     *
     * Given a trajectory this will generate a SE(3) @ref ov_core::BsplineSE3 for that trajectory.
     * This allows us to get the inertial measurement information at each timestep during this trajectory.
     * After creating the bspline we will generate an environmental feature map which will be used as our feature measurements.
     * This map will be projected into the frame at each timestep to get our "raw" uv measurements.
     * We inject bias and white noises into our inertial readings while adding our white noise to the uv measurements also.
     * The user should specify the sensor rates that they desire along with the seeds of the random number generators.
     *
     */
    class Simulator {

    public:


        /**
         * @brief Default constructor, will load all configuration variables. load trajectory and 
         * generate all control point(poses) along the trajectory, generate all mappoints
         * @param params_ VioManager parameters. Should have already been loaded from cmd.
         */
        Simulator(VioManagerOptions& params_);

        /**
         * @brief Returns if we are actively simulating
         * @return True if we still have simulation data
         */
        bool ok() {
            return is_running;
        }

        /**
         * @brief Gets the timestamp we have simulated up too
         * @return Timestamp
         */
        double current_timestamp() {
            return timestamp;
        }

        /**
         * @brief Get the simulation state at a specified timestep
         * @param desired_time Timestamp we want to get the state at
         * @param imustate State in the MSCKF ordering: [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
         * @return True if we have a state
         */
        bool get_state(double desired_time, Eigen::Matrix<double,17,1> &imustate);

        /**
         * @brief Gets the next inertial reading if we have one.
         * @param time_imu Time that this measurement occured at
         * @param wm Angular velocity measurement in the inertial frame
         * @param am Linear velocity in the inertial frame
         * @return True if we have a measurement
         */
        bool get_next_imu(double &time_imu, Eigen::Vector3d &wm, Eigen::Vector3d &am);


        /**
         * @brief Gets the next inertial reading if we have one.
         * @param time_cam Time that this measurement occured at
         * @param camids Camera ids that the corresponding vectors match
         * @param feats Noisy uv measurements and ids for the returned time
         * @return True if we have a measurement
         */
        bool get_next_cam(double &time_cam, std::vector<int> &camids, std::vector<std::vector<std::pair<size_t,Eigen::VectorXf>>> &feats);


        /// Returns the true 3d map of features
        std::unordered_map<size_t,Eigen::Vector3d> get_map() {
            return featmap;
        }

        std::unordered_map<size_t,Eigen::Vector3d> get_matching_map() {
            return featmatching;
        }


        /// Access function to get the true parameters (i.e. calibration and settings)
        VioManagerOptions get_true_paramters() {
            return params;
        }

        int get_sim_feats_num(){return id_map;}

        void project_pointcloud_map(const Eigen::Matrix3d &R_CtoG, const Eigen::Vector3d &p_CinG,
                                    const Eigen::Matrix3d &R_CtoG_turb, const Eigen::Vector3d &p_CinG_turb,
                                    const Eigen::Matrix3d &R_kftoG, const Eigen::Vector3d &p_kfinG,
                                    const Eigen::Matrix3d &R_kftoG_turb, const Eigen::Vector3d &p_kfinG_turb,
                                    int camid, const std::unordered_map<size_t,Eigen::Vector3d> &feats,
                                    vector<Eigen::VectorXf> &uvs_cur,
                                    vector<Eigen::VectorXf> &uvs_kf, vector<Eigen::VectorXd> &pts3d_kf);

        bool generate_points_map(const Eigen::Matrix3d &R_CtoG, const Eigen::Vector3d &p_CinG,
                                 const Eigen::Matrix3d &R_CtoG_turb, const Eigen::Vector3d &p_CinG_turb,
                                 const Eigen::Matrix3d &R_kftoG, const Eigen::Vector3d &p_kfinG,
                                 const Eigen::Matrix3d &R_kftoG_turb, const Eigen::Vector3d &p_kfinG_turb,
                                        int camid, const std::unordered_map<size_t,Eigen::Vector3d> &feats,
                                        int numpts, vector<Eigen::VectorXf> &uvs_cur,
                                        vector<Eigen::VectorXf> &uvs_kf, vector<Eigen::VectorXd> &pts3d_kf);
        void  triangulate_3dpoint(const Eigen::Matrix3d &R_CtoG,const Eigen::Vector3d &p_CinG,
                                  const Eigen::Matrix3d &R_kftoG, const Eigen::Vector3d &p_kfinG,
                                  const Eigen::Vector2f &uv_dist,const Eigen::Vector2f &uv_dist_kf,
                                  Eigen::Vector3d &pt3d_turb);

        void gen_matching(vector<Matrix3d> &kfs_pose_R,vector<Vector3d> &kfs_pose_t,
                          vector<Matrix3d> &kfs_pose_R_turb,vector<Vector3d> &kfs_pose_t_turb,
                          Matrix3d &R_CtoG,Vector3d &p_CinG,
                          vector<int> &matches_num,vector<string> &kfs_ts_with_gt,
                          vector<string> &res_kfs_ts,
                          vector<vector<Eigen::VectorXf>> &res_uvs_cur,
                          vector<vector<Eigen::VectorXf>> &res_uvs_kf,
                          vector<vector<Eigen::VectorXd>> &res_pts3d_kf);
        Eigen::Vector2f reproject_uv_norm(Eigen::Vector2f uv_norm_kf, int camid, Eigen::Matrix<double,8,1> cam_d);

        bool single_triangulation(vector<int>& kfindeces,vector<Vector2f>& kfuvs,vector<Vector2f>& kfuvnorms,
        vector<Eigen::Matrix3d>& kfs_pose_R_turb,vector<Eigen::Vector3d>& kfs_pose_t_turb,Vector3d& pt_turb,Vector3d& pt_anchor);

        bool single_gaussnewton(vector<int>& kfindeces,vector<Vector2f>& kfuvs,vector<Vector2f>& kfuvnorms,
        vector<Eigen::Matrix3d>& kfs_pose_R_turb,vector<Eigen::Vector3d>& kfs_pose_t_turb,Vector3d& pt_turb,Vector3d& pt_anchor);

        double compute_error(vector<int>& kfindeces,vector<Vector2f>& kfuvs,vector<Vector2f>& kfuvnorms,
                     vector<Eigen::Matrix3d>& kfs_pose_R_turb,vector<Eigen::Vector3d>& kfs_pose_t_turb,
                     double alpha, double beta, double rho);


        size_t id_matching=0;    //used for generate map matching point
        std::unordered_map<size_t,Eigen::Vector3d> featmatching;
        std::unordered_map<size_t,Eigen::Vector3d> featmatchingturb;
        std::unordered_map<size_t,Eigen::Vector2f> uvsmatching;
        
        ofstream of_point;
        ofstream of_gyro_bias;
        ofstream of_acc_bias;
        /// Our b-spline trajectory
        BsplineSE3 spline;
        BsplineSE3 spline_turb;





    // protected:


        /**
         * @brief This will load the trajectory into memory.
         * @param path_traj Path to the trajectory file that we want to read in.
         */
        void load_data(std::string path_traj);

        void load_data_turb(std::string path_traj);

        /**
         * @brief Projects the passed map features into the desired camera frame.
         * @param R_GtoI Orientation of the IMU pose
         * @param p_IinG Position of the IMU pose
         * @param camid Camera id of the camera sensor we want to project into
         * @param feats Our set of 3d features
         * @return True distorted raw image measurements and their ids for the specified camera
         */
        std::vector<std::pair<size_t,Eigen::VectorXf>> project_pointcloud(const Eigen::Matrix3d &R_GtoI, const Eigen::Vector3d &p_IinG, int camid, const std::unordered_map<size_t,Eigen::Vector3d> &feats);
        
       

        /**
         * @brief Will generate points in the fov of the specified camera
         * @param R_GtoI Orientation of the IMU pose
         * @param p_IinG Position of the IMU pose
         * @param camid Camera id of the camera sensor we want to project into
         * @param[out] feats Map we will append new features to
         * @param numpts Number of points we should generate
         */
        void generate_points(const Eigen::Matrix3d &R_GtoI, const Eigen::Vector3d &p_IinG, int camid, std::unordered_map<size_t,Eigen::Vector3d> &feats, int numpts);

        
         
        //===================================================================
        // Configuration variables
        //===================================================================

        /// True vio manager params (a copy of the parsed ones)
        VioManagerOptions params;

        //===================================================================
        // State related variables
        //===================================================================

        /// Our loaded trajectory data (timestamp(s), q_GtoI, p_IinG)
        std::vector<Eigen::VectorXd> traj_data;
        std::vector<Eigen::VectorXd> traj_data_turb;



        
        /// Our map of 3d features
        size_t id_map = 0;
        std::unordered_map<size_t,Eigen::Vector3d> featmap;  //map of id and 3d global position

        

        /// Mersenne twister PRNG for measurements (IMU)
        std::mt19937 gen_meas_imu;

        /// Mersenne twister PRNG for measurements (CAMERAS)
        std::vector<std::mt19937> gen_meas_cams;

        /// Mersenne twister PRNG for state initialization
        std::mt19937 gen_state_init;

        /// Mersenne twister PRNG for state perturbations
        std::mt19937 gen_state_perturb;

        /// If our simulation is running
        bool is_running;

        //===================================================================
        // Simulation specific variables
        //===================================================================

        /// Current timestamp of the system
        double timestamp;

        /// Last time we had an IMU reading
        double timestamp_last_imu;

        /// Last time we had an CAMERA reading
        double timestamp_last_cam;

        /// Our running acceleration bias
        Eigen::Vector3d true_bias_accel = Eigen::Vector3d::Zero();

        /// Our running gyroscope bias
        Eigen::Vector3d true_bias_gyro = Eigen::Vector3d::Zero();

        // Our history of true biases
        std::vector<double> hist_true_bias_time;
        std::vector<Eigen::Vector3d> hist_true_bias_accel;
        std::vector<Eigen::Vector3d> hist_true_bias_gyro;


    };


}

#endif //OV_MSCKF_SIMULATOR_H
