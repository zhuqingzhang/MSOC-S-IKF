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
#ifndef OV_RIMSCKF_STATE_H
#define OV_RIMSCKF_STATE_H


#include <vector>
#include <unordered_map>

#include "types/Type.h"
#include "types/IMU.h"
#include "types/Vec.h"
#include "types/PoseJPL.h"
#include "types/Landmark.h"
#include "StateOptions.h"
#include "match/KeyframeDatabase.h"
#include "sim/BsplineSE3.h"

using namespace ov_core;
using namespace ov_type;


namespace ov_rimsckf {

    /**
     * @brief State of our filter
     *
     * This state has all the current estimates for the filter.
     * This system is modeled after the MSCKF filter, thus we have a sliding window of clones.
     * We additionally have more parameters for online estimation of calibration and SLAM features.
     * We also have the covariance of the system, which should be managed using the StateHelper class.
     * For localization purpose, we also have nuisance_part to record map keyframes and relative transformation
     * between the Local reference frame and Global reference frame.
     * NOTE: although the state maintain R_GtoI in PoseJPL, for RIMSCKF, we use R_ItoG to formulate
     * propagation equation and updating equation. Therefore, the covariance of rotation part is 
     * also corresponding to R_ItoG instead of R_GtoI.
     * 
     */
    class State {

    public:

        /**
         * @brief Default Constructor (will initialize variables to defaults)
         * @param options_ Options structure containing filter options
         */
        State(StateOptions &options_);

        ~State() {}


        /**
         * @brief Will return the timestep that we will marginalize next.
         * As of right now, since we are using a sliding window, this is the oldest clone.
         * But if you wanted to do a keyframe system, you could selectively marginalize clones.
         * @return timestep of clone we will marginalize
         */
         //loop through the _clones_IMU, find the cloned one with the smallest(oldest) timestamp
        double margtimestep() {
            double time = INFINITY;
            for (std::pair<const double, PoseJPL*> &clone_imu : _clones_IMU) {
                if (clone_imu.first < time) {
                    time = clone_imu.first;
                }
            }
            return time;
        }

        /**
         * @brief Calculates the current max size of the covariance
         * @return Size of the current covariance matrix
         */
        int max_covariance_size() {
            return (int)_Cov.rows();
        }

        int max_nuisance_covariance_size()
        {
            return (int)_Cov_nuisance.rows();
        }

        void insert_keyframe_into_database(Keyframe *kf)
        {
            kfdatabase->update_kf(kf);
            //cout<<kfdatabase->get_internal_data().size()<<endl;
        }

        KeyframeDatabase* get_kfdataBase()
        {
            return kfdatabase;
        }

         void init_spline(std::vector<Eigen::VectorXd>& traj_data)
        {
          spline.feed_trajectory(traj_data);
        }

        void set_points_gt(std::unordered_map<size_t,Eigen::Vector3d> &featmap)
        {
          _featmap = featmap;
        }

        bool get_state(double desired_time, Eigen::Matrix<double,17,1> &imustate)
        {
            imustate.setZero();
            imustate(4) = 1;

            // Current state values
            Eigen::Matrix3d R_GtoI;
            Eigen::Vector3d p_IinG, w_IinI, v_IinG;

            // Get the pose, velocity, and acceleration
            bool success_vel = spline.get_velocity(desired_time, R_GtoI, p_IinG, w_IinI, v_IinG);

            // Find the bounding bias values
            

            // Finally lets create the current state
            imustate(0,0) = desired_time;
            imustate.block(1,0,4,1) = rot_2_quat(R_GtoI);
            imustate.block(5,0,3,1) = p_IinG;
            imustate.block(8,0,3,1) = v_IinG;

            return success_vel;
        }

        /// Current timestamp (should be the last update time!)
        double _timestamp;

        double _timestamp_approx;

        /// Struct containing filter options
        StateOptions _options;

        /// Pointer to the "active" IMU state (q_GtoI, p_IinG, v_IinG, bg, ba)
        IMU *_imu;

        /// Map between imaging times and clone poses (q_GtoIi, p_IiinG)
        std::map<double, PoseJPL*> _clones_IMU;

        /// Our current set of SLAM features (3d positions). map between feature_id and its Landmark
        std::unordered_map<size_t, Landmark*> _features_SLAM;

        /// Time offset base IMU to camera (t_imu = t_cam + t_off)
        Vec *_calib_dt_CAMtoIMU;

        /// Calibration poses for each camera (R_ItoC, p_IinC)
        std::unordered_map<size_t, PoseJPL*> _calib_IMUtoCAM;

        /// Camera intrinsics
        std::unordered_map<size_t, Vec*> _cam_intrinsics;

        /// What distortion model we are using (false=radtan, true=fisheye)
        std::unordered_map<size_t, bool> _cam_intrinsics_model;

        ///modified by zzq.
        ///the nuisance part of SKF. Map between keyframe(from the map) timestamp and pose
        std::map<double, PoseJPL*> _clones_Keyframe;

        PoseJPL* transform_vio_to_map;  //for msckf use

        PoseJPL* transform_map_to_vio; //for rimsckf use

        bool set_transform = false;

        std::unordered_map<double,Eigen::Vector3d> p_vio_in_map;
        std::unordered_map<double,Eigen::Matrix3d> R_vio_to_map;
        
        ofstream of_points;

        bool iter=false;

        int iter_count=0;

        bool have_match=false;

        Eigen::MatrixXd Hx_last;
        Eigen::MatrixXd Hn_last;
        double _avg_error_2_last;

        bool zupt=false;


    // private:

        KeyframeDatabase *kfdatabase;
        // Define that the state helper is a friend class of this class
        // This will allow it to access the below functions which should normally not be called
        // This prevents a developer from thinking that the "insert clone" will actually correctly add it to the covariance
        friend class StateHelper;
        
        double noise=1;

        double v=1000;
        double a=0.4;
        double V=1;

        double b=0.85;
        int count=0;
        MatrixXd sigma;

        Eigen::Vector3d last_kf_match_position;
        double distance_last_match=0;

        vector<double> delay_clones;  //record the timestamp of clones_IMU with match map kf
        
        vector<vector<Feature*>> delay_loop_features;

        /// Covariance of all active variables
        Eigen::MatrixXd _Cov;

        Eigen::MatrixXd _Cov_last;

        Eigen::MatrixXd _Cov_iter;

        ///Covriance of all nuisance variables
        Eigen::MatrixXd _Cov_nuisance;

        ///Cross-Covariance of  active-nuisance variable;
        Eigen::MatrixXd _Cross_Cov_AN;

        Eigen::MatrixXd _Cross_Cov_AN_last;

        Eigen::MatrixXd _Cross_Cov_AN_iter;

        /// Vector of variables
        std::vector<Type*> _variables;

        std::vector<Eigen::MatrixXd> _variables_last;

        /// Vector of nuisance_variables
        std::vector<Type*> _nuisance_variables;

        BsplineSE3 spline;

        std::unordered_map<size_t,Eigen::Vector3d> _featmap;        


    };

}

#endif //OV_RIMSCKF_STATE_H