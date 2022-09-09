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
#include "UpdaterSLAM.h"



using namespace ov_core;
using namespace ov_rimsckf;



void UpdaterSLAM::delayed_init(State *state, std::vector<Feature*>& feature_vec) {

    // Return if no features
    if(feature_vec.empty())
        return;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements
        int ct_meas = 0;
        for(const auto &pair : (*it0)->timestamps) {
            ct_meas += (*it0)->timestamps[pair.first].size();
        }

        // Remove if we don't have enough
        if(ct_meas < 3) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }

    }
    rT1 =  boost::posix_time::microsec_clock::local_time();

    // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
    std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
    for(const auto &clone_calib : state->_calib_IMUtoCAM) {

        // For this camera, create the vector of camera poses
        std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
        for(const auto &clone_imu : state->_clones_IMU) {

            // Get current camera pose
            Eigen::Matrix<double,3,3> R_GtoCi = clone_calib.second->Rot()*clone_imu.second->Rot();
            Eigen::Matrix<double,3,1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose()*clone_calib.second->pos();

            // Append to our map
            clones_cami.insert({clone_imu.first,FeatureInitializer::ClonePose(R_GtoCi,p_CioinG)});

        }

        // Append to our map
        clones_cam.insert({clone_calib.first,clones_cami});

    }

    // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
    auto it1 = feature_vec.begin();
    while(it1 != feature_vec.end()) {

        // Triangulate the feature and remove if it fails
        bool success = initializer_feat->single_triangulation(*it1, clones_cam);
        if(!success) {
            (*it1)->to_delete = true;
            it1 = feature_vec.erase(it1);
            continue;
        }

        // Gauss-newton refine the feature
        success = initializer_feat->single_gaussnewton(*it1, clones_cam);
        if(!success) {
            (*it1)->to_delete = true;
            it1 = feature_vec.erase(it1);
            continue;
        }
        it1++;

    }
    rT2 =  boost::posix_time::microsec_clock::local_time();

    // 4. Compute linear system for each feature, nullspace project, and reject
    auto it2 = feature_vec.begin();
    cout<<"num of landmark feats to be initialized : "<<feature_vec.size()<<endl;
    while(it2 != feature_vec.end()) {


        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        
        // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        auto feat_rep = ((int)feat.featid < state->_options.max_aruco_features)? state->_options.feat_rep_aruco : state->_options.feat_rep_slam;
        feat.feat_representation = feat_rep;
        //we only support gloabl 3D landmarks for invariant ekf right now.
        assert(feat_rep==LandmarkRepresentation::Representation::GLOBAL_3D);
        if(feat_rep==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
            feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        }

        // Save the position and its fej value
        if(LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
            feat.anchor_cam_id = (*it2)->anchor_cam_id;
            feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;
            feat.p_FinA = (*it2)->p_FinA;
            feat.p_FinA_fej = (*it2)->p_FinA;
        } else {
            feat.p_FinG = (*it2)->p_FinG;
            feat.p_FinG_fej = (*it2)->p_FinG;
        }

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f;
        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        std::vector<Type*> Hx_order;

        // Get the Jacobian for this feature
        UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

        // If we are doing the single feature representation, then we need to remove the bearing portion
        // To do so, we project the bearing portion onto the state and depth Jacobians and the residual.
        // This allows us to directly initialize the feature as a depth-old feature
        if(feat_rep==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

            // Append the Jacobian in respect to the depth of the feature
            Eigen::MatrixXd H_xf = H_x;
            H_xf.conservativeResize(H_x.rows(), H_x.cols()+1);
            H_xf.block(0, H_x.cols(), H_x.rows(), 1) = H_f.block(0,H_f.cols()-1,H_f.rows(),1);
            H_f.conservativeResize(H_f.rows(), H_f.cols()-1);

            // Nullspace project the bearing portion
            // This takes into account that we have marginalized the bearing already
            // Thus this is crucial to ensuring estimator consistency as we are not taking the bearing to be true
            UpdaterHelper::nullspace_project_inplace(H_f, H_xf, res);

            // Split out the state portion and feature portion
            H_x = H_xf.block(0,0,H_xf.rows(),H_xf.cols()-1);
            H_f = H_xf.block(0,H_xf.cols()-1,H_xf.rows(),1);

        }

        // Create feature pointer (we will always create it of size three since we initialize the single invese depth as a msckf anchored representation)
        int landmark_size = (feat_rep==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE)? 1 : 3;
        Landmark* landmark = new Landmark(landmark_size);
        landmark->_featid = feat.featid;
        landmark->_feat_representation = feat_rep;
        if(LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
            landmark->_anchor_cam_id = feat.anchor_cam_id;
            landmark->_anchor_clone_timestamp = feat.anchor_clone_timestamp;
            landmark->set_from_xyz(feat.p_FinA, false);
            landmark->set_from_xyz(feat.p_FinA_fej, true);
        } else {
            landmark->set_from_xyz(feat.p_FinG, false);
            landmark->set_from_xyz(feat.p_FinG_fej, true);
        }

        // Measurement noise matrix
        double sigma_pix_sq = ((int)feat.featid < state->_options.max_aruco_features)? _options_aruco.sigma_pix_sq : _options_slam.sigma_pix_sq;
        Eigen::MatrixXd R = sigma_pix_sq*Eigen::MatrixXd::Identity(res.rows(), res.rows());

        // Try to initialize, delete new pointer if we failed
        double chi2_multipler = ((int)feat.featid < state->_options.max_aruco_features)? _options_aruco.chi2_multipler : _options_slam.chi2_multipler;
        if (StateHelper::initialize(state, landmark, Hx_order, H_x, H_f, R, res, chi2_multipler)) {
            state->_features_SLAM.insert({(*it2)->featid, landmark});
            cout<<"new initialized landmark id: "<<landmark->_featid<<endl;
            (*it2)->to_delete = true;
            it2++;
        } else {
            delete landmark;
            (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            
            continue;
        }

    //     //test landmark reprojection
    //     cout<<"test landmark reprojection"<<endl;
    //     Eigen::Vector3d p_FinG = landmark->get_xyz(false);
    //     for (auto const& pair : feat.timestamps) {

    //     // Our calibration between the IMU and CAMi frames
    //     cout<<"for cameraid "<<pair.first<<endl;
    //     Vec* distortion = state->_cam_intrinsics.at(pair.first);
    //     PoseJPL* calibration = state->_calib_IMUtoCAM.at(pair.first);
    //     Eigen::Matrix<double,3,3> R_ItoC = calibration->Rot();
    //     Eigen::Matrix<double,3,1> p_IinC = calibration->pos();
    //     //Following Jacobian is derived based in R_CtoI and p_CinI
    //     Eigen::Matrix<double,3,3> R_CtoI = R_ItoC.transpose();
    //     Eigen::Matrix<double,3,1> p_CinI = -R_CtoI*p_IinC;
    //     Eigen::Matrix<double,8,1> cam_d = distortion->value();
    //     cout<<"R_CtoI: "<<endl<<R_CtoI<<endl<<"p_CinI: "<<p_CinI.transpose()<<endl;
        

    //     // Loop through all measurements for this specific camera
    //     for (size_t m = 0; m < feat.timestamps[pair.first].size(); m++) {

    //         //=========================================================================
    //         //=========================================================================
    //         // Get current IMU clone state
    //         PoseJPL* clone_Ii = state->_clones_IMU.at(feat.timestamps[pair.first].at(m));
    //         // Get current feature in IMU frame
    //         Eigen::Matrix<double,3,1> p_FinIi = clone_Ii->Rot()*(p_FinG-clone_Ii->pos());
    //         // Project the current feature into the current frame of reference to get predicted measurement
    //         Eigen::Matrix<double,3,1> p_FinCi = R_ItoC*p_FinIi+p_IinC;
    //         Eigen::Matrix<double,2,1> uv_norm;
    //         uv_norm << p_FinCi(0)/p_FinCi(2),p_FinCi(1)/p_FinCi(2);
    //         // Distort the normalized coordinates (false=radtan, true=fisheye)
    //         Eigen::Matrix<double,2,1> uv_dist;

    //         // Calculate distortion uv and jacobian
    //         if(state->_cam_intrinsics_model.at(pair.first)) {

    //             // Calculate distorted coordinates for fisheye
    //             double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
    //             double theta = std::atan(r);
    //             double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

    //             // Handle when r is small (meaning our xy is near the camera center)
    //             double inv_r = (r > 1e-8)? 1.0/r : 1.0;
    //             double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

    //             // Calculate distorted coordinates for fisheye
    //             double x1 = uv_norm(0)*cdist;
    //             double y1 = uv_norm(1)*cdist;
    //             uv_dist(0) = cam_d(0)*x1 + cam_d(2);
    //             uv_dist(1) = cam_d(1)*y1 + cam_d(3);

    //         } else {

    //             // Calculate distorted coordinates for radial
    //             double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
    //             double r_2 = r*r;
    //             double r_4 = r_2*r_2;
    //             double x1 = uv_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm(0)*uv_norm(1)+cam_d(7)*(r_2+2*uv_norm(0)*uv_norm(0));
    //             double y1 = uv_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm(1)*uv_norm(1))+2*cam_d(7)*uv_norm(0)*uv_norm(1);
    //             uv_dist(0) = cam_d(0)*x1 + cam_d(2);
    //             uv_dist(1) = cam_d(1)*y1 + cam_d(3);

    //         }

    //         // Our residual
    //         Eigen::Matrix<double,2,1> uv_m;
    //         uv_m << (double)feat.uvs[pair.first].at(m)(0), (double)feat.uvs[pair.first].at(m)(1);
    //         cout<<"clone IMU pose at "<<to_string(feat.timestamps[pair.first].at(m))<<" orientation: "
    //         <<clone_Ii->quat().transpose()<<" position: "<<clone_Ii->pos().transpose()<<endl;
    //         cout<<"uv_m: "<<uv_m.transpose()<<" uv_dist: "<<uv_dist.transpose()<<endl;
    //         cout<<"measures error: "<<(uv_m - uv_dist).transpose()<<endl;

    //         cout<<" current feature observed timestamp: "<<to_string(feat.timestamps[pair.first].at(m))<<endl;   

    //     }

    //   }
    //   cout<<"end test landmark reprojection"<<endl;

    }
    rT3 =  boost::posix_time::microsec_clock::local_time();

    // Debug print timing information
    //if(!feature_vec.empty()) {
    //    printf("[SLAM-DELAY]: %.4f seconds to clean\n",(rT1-rT0).total_microseconds() * 1e-6);
    //    printf("[SLAM-DELAY]: %.4f seconds to triangulate\n",(rT2-rT1).total_microseconds() * 1e-6);
    //    printf("[SLAM-DELAY]: %.4f seconds initialize (%d features)\n",(rT3-rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
    //    printf("[SLAM-DELAY]: %.4f seconds total\n",(rT3-rT1).total_microseconds() * 1e-6);
    //}

}

void UpdaterSLAM::delayed_init_skf(State *state, std::vector<Feature*>& feature_vec) {

    // Return if no features
    if(feature_vec.empty())
        return;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements
        int ct_meas = 0;
        for(const auto &pair : (*it0)->timestamps) {
            ct_meas += (*it0)->timestamps[pair.first].size();
        }

        // Remove if we don't have enough
        if(ct_meas < 3) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }

    }
    rT1 =  boost::posix_time::microsec_clock::local_time();

    // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
    std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
    for(const auto &clone_calib : state->_calib_IMUtoCAM) {

        // For this camera, create the vector of camera poses
        std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
        for(const auto &clone_imu : state->_clones_IMU) {

            // Get current camera pose
            Eigen::Matrix<double,3,3> R_GtoCi = clone_calib.second->Rot()*clone_imu.second->Rot();
            Eigen::Matrix<double,3,1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose()*clone_calib.second->pos();

            // Append to our map
            clones_cami.insert({clone_imu.first,FeatureInitializer::ClonePose(R_GtoCi,p_CioinG)});

        }

        // Append to our map
        clones_cam.insert({clone_calib.first,clones_cami});

    }

    // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
    auto it1 = feature_vec.begin();
    while(it1 != feature_vec.end()) {

        // Triangulate the feature and remove if it fails
        bool success = initializer_feat->single_triangulation(*it1, clones_cam);
        if(!success) {
            (*it1)->to_delete = true;
            it1 = feature_vec.erase(it1);
            continue;
        }

        // Gauss-newton refine the feature
        success = initializer_feat->single_gaussnewton(*it1, clones_cam);
        if(!success) {
            (*it1)->to_delete = true;
            it1 = feature_vec.erase(it1);
            continue;
        }
        it1++;

    }
    rT2 =  boost::posix_time::microsec_clock::local_time();

    // 4. Compute linear system for each feature, nullspace project, and reject
    auto it2 = feature_vec.begin();
    while(it2 != feature_vec.end()) {


        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        feat.keyframe_matched_obs = (*it2)->keyframe_matched_obs;
        //in this case, we don't have keyframe constrains.
        assert(feat.keyframe_matched_obs.empty());

        // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        auto feat_rep = ((int)feat.featid < state->_options.max_aruco_features)? state->_options.feat_rep_aruco : state->_options.feat_rep_slam;
        feat.feat_representation = feat_rep;
        if(feat_rep==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
            feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        }

        // Save the position and its fej value
        if(LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
            feat.anchor_cam_id = (*it2)->anchor_cam_id;
            feat.anchor_clone_timestamp = (*it2)->anchor_clone_timestamp;
            feat.p_FinA = (*it2)->p_FinA;
            feat.p_FinA_fej = (*it2)->p_FinA;
        } else {
            feat.p_FinG = (*it2)->p_FinG;
            feat.p_FinG_fej = (*it2)->p_FinG;
        }

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f;
        Eigen::MatrixXd H_x;
        Eigen::MatrixXd H_n;
        Eigen::VectorXd res;
        std::vector<Type*> Hx_order;
        std::vector<Type*> Hn_order;

        // Get the Jacobian for this feature
        UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

        // If we are doing the single feature representation, then we need to remove the bearing portion
        // To do so, we project the bearing portion onto the state and depth Jacobians and the residual.
        // This allows us to directly initialize the feature as a depth-old feature
        if(feat_rep==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

            // Append the Jacobian in respect to the depth of the feature
            Eigen::MatrixXd H_xf = H_x;
            H_xf.conservativeResize(H_x.rows(), H_x.cols()+1);
            H_xf.block(0, H_x.cols(), H_x.rows(), 1) = H_f.block(0,H_f.cols()-1,H_f.rows(),1);
            H_f.conservativeResize(H_f.rows(), H_f.cols()-1);

            // Nullspace project the bearing portion
            // This takes into account that we have marginalized the bearing already
            // Thus this is crucial to ensuring estimator consistency as we are not taking the bearing to be true
            UpdaterHelper::nullspace_project_inplace(H_f, H_xf, res);

            // Split out the state portion and feature portion
            H_x = H_xf.block(0,0,H_xf.rows(),H_xf.cols()-1);
            H_f = H_xf.block(0,H_xf.cols()-1,H_xf.rows(),1);

        }

        // Create feature pointer (we will always create it of size three since we initialize the single invese depth as a msckf anchored representation)
        int landmark_size = (feat_rep==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE)? 1 : 3;
        Landmark* landmark = new Landmark(landmark_size);
        landmark->_featid = feat.featid;
        landmark->_feat_representation = feat_rep;
        if(LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
            landmark->_anchor_cam_id = feat.anchor_cam_id;
            landmark->_anchor_clone_timestamp = feat.anchor_clone_timestamp;
            landmark->set_from_xyz(feat.p_FinA, false);
            landmark->set_from_xyz(feat.p_FinA_fej, true);
        } else {
            landmark->set_from_xyz(feat.p_FinG, false);
            landmark->set_from_xyz(feat.p_FinG_fej, true);
        }

        // Measurement noise matrix
        double sigma_pix_sq = ((int)feat.featid < state->_options.max_aruco_features)? _options_aruco.sigma_pix_sq : _options_slam.sigma_pix_sq;
        Eigen::MatrixXd R = sigma_pix_sq*Eigen::MatrixXd::Identity(res.rows(), res.rows());
        H_n.conservativeResize(res.rows(),0);

        // Try to initialize, delete new pointer if we failed
        double chi2_multipler = ((int)feat.featid < state->_options.max_aruco_features)? _options_aruco.chi2_multipler : _options_slam.chi2_multipler;
        if (StateHelper::initialize_skf(state, landmark, Hx_order, Hn_order,H_x, H_n, H_f, R, res, chi2_multipler)) {
            state->_features_SLAM.insert({(*it2)->featid, landmark});
            (*it2)->to_delete = true;
            it2++;
        } else {
            delete landmark;
            (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
        }

    }
    rT3 =  boost::posix_time::microsec_clock::local_time();

    // Debug print timing information
    //if(!feature_vec.empty()) {
    //    printf("[SLAM-DELAY]: %.4f seconds to clean\n",(rT1-rT0).total_microseconds() * 1e-6);
    //    printf("[SLAM-DELAY]: %.4f seconds to triangulate\n",(rT2-rT1).total_microseconds() * 1e-6);
    //    printf("[SLAM-DELAY]: %.4f seconds initialize (%d features)\n",(rT3-rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
    //    printf("[SLAM-DELAY]: %.4f seconds total\n",(rT3-rT1).total_microseconds() * 1e-6);
    //}

}

void UpdaterSLAM::update(State *state, std::vector<Feature*>& feature_vec) {


    // Return if no features
    if(feature_vec.empty())
        return;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }


    // 1. Clean all feature measurements and make sure they all have valid clone times
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements
        int ct_meas = 0;
        for(const auto &pair : (*it0)->timestamps) {
            ct_meas += (*it0)->timestamps[pair.first].size();
        }

        // Get the landmark and its representation
        // For single depth representation we need at least two measurement
        // This is because we do nullspace projection
        Landmark* landmark = state->_features_SLAM.at((*it0)->featid);
        int required_meas = (landmark->_feat_representation==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE)? 3 : 2;

        // Remove if we don't have enough
        if(ct_meas < 1) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else if(ct_meas < required_meas) {
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }

    }
    rT1 =  boost::posix_time::microsec_clock::local_time();

    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    for(size_t i=0; i<feature_vec.size(); i++) {
        for (const auto &pair : feature_vec.at(i)->timestamps) {
            max_meas_size += 2*feature_vec.at(i)->timestamps[pair.first].size();
        }
    }

    // Calculate max possible state size (i.e. the size of our covariance)
    size_t max_hx_size = state->max_covariance_size();

    // Large Jacobian, residual, and measurement noise of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(max_meas_size,max_meas_size);
    std::unordered_map<Type*,size_t> Hx_mapping;
    std::vector<Type*> Hx_order_big;
    size_t ct_jacob = 0;
    size_t ct_meas = 0;

    // 4. Compute linear system for each feature, nullspace project, and reject
    auto it2 = feature_vec.begin();
    cout<<"num of landmarks used to update: "<<feature_vec.size()<<endl;
    while(it2 != feature_vec.end()) {

        // Ensure we have the landmark and it is the same
        assert(state->_features_SLAM.find((*it2)->featid) != state->_features_SLAM.end());
        assert(state->_features_SLAM.at((*it2)->featid)->_featid == (*it2)->featid);

        // Get our landmark from the state
        Landmark* landmark = state->_features_SLAM.at((*it2)->featid);

        // Convert the state landmark into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        cout<<"landmark id: "<<feat.featid<<endl;

        // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        feat.feat_representation = landmark->_feat_representation;
        assert(feat.feat_representation==LandmarkRepresentation::Representation::GLOBAL_3D);
        // if(landmark->_feat_representation==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        //     feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        // }

        // Save the position and its fej value
        if(LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
            feat.anchor_cam_id = landmark->_anchor_cam_id;
            feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
            feat.p_FinA = landmark->get_xyz(false);
            feat.p_FinA_fej = landmark->get_xyz(true);
        } else {
            feat.p_FinG = landmark->get_xyz(false);
            feat.p_FinG_fej = landmark->get_xyz(true);
        }

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f;
        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        std::vector<Type*> Hx_order;

        // Get the Jacobian for this feature
        UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

        // Place Jacobians in one big Jacobian, since the landmark is already in our state vector
        // H_xf=[H_x,H_f], is the augmented matrix, since landmark is also an element of the state. by zzq
        Eigen::MatrixXd H_xf = H_x;
        if(landmark->_feat_representation==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

            // Append the Jacobian in respect to the depth of the feature
            //As the landmark is represented by only the depth(i.e. depth is the element of the state
            //instead of the full 3d landmark), so we just need to append the last column(depth) into Jacobians. by zzq
            H_xf.conservativeResize(H_x.rows(), H_x.cols()+1);
            H_xf.block(0, H_x.cols(), H_x.rows(), 1) = H_f.block(0,H_f.cols()-1,H_f.rows(),1);
            H_f.conservativeResize(H_f.rows(), H_f.cols()-1);

            // Nullspace project the bearing portion
            // This takes into account that we have marginalized the bearing already
            // Thus this is crucial to ensuring estimator consistency as we are not taking the bearing to be true
            UpdaterHelper::nullspace_project_inplace(H_f, H_xf, res);

        } else {

            // Else we have the full feature in our state, so just append it
            H_xf.conservativeResize(H_x.rows(), H_x.cols()+H_f.cols());
            H_xf.block(0, H_x.cols(), H_x.rows(), H_f.cols()) = H_f;

        }

        // Append to our Jacobian order vector
        std::vector<Type*> Hxf_order = Hx_order;
        Hxf_order.push_back(landmark);

        /// Chi2 distance check
        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hxf_order);
        Eigen::MatrixXd S = H_xf*P_marg*H_xf.transpose();
        double sigma_pix_sq = ((int)feat.featid < state->_options.max_aruco_features)? _options_aruco.sigma_pix_sq : _options_slam.sigma_pix_sq;
        S.diagonal() += sigma_pix_sq*Eigen::VectorXd::Ones(S.rows());
        double chi2 = res.dot(S.llt().solve(res));

        // Get our threshold (we precompute up to 500 but handle the case that it is more)
        double chi2_check;
        if(res.rows() < 500) {
            chi2_check = chi_squared_table[res.rows()];
        } else {
            boost::math::chi_squared chi_squared_dist(res.rows());
            chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
            printf(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
        }

        // Check if we should delete or not
        double chi2_multipler = ((int)feat.featid < state->_options.max_aruco_features)? _options_aruco.chi2_multipler : _options_slam.chi2_multipler;
        
         cout<<"chi2 vs. chi2check: "<<chi2<<" vs. "<<chi2_multipler*chi2_check<<endl;
        if(chi2 > chi2_multipler*chi2_check) {
            if((int)feat.featid < state->_options.max_aruco_features)
                printf(YELLOW "[SLAM-UP]: rejecting aruco tag %d for chi2 thresh (%.3f > %.3f)\n" RESET,(int)feat.featid,chi2,chi2_multipler*chi2_check);
            printf(GREEN "slam feature chi2_x is too large\n" RESET );
            (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            continue;
        }
       
        // Debug print when we are going to update the aruco tags
        if((int)feat.featid < state->_options.max_aruco_features)
            printf("[SLAM-UP]: accepted aruco tag %d for chi2 thresh (%.3f < %.3f)\n",(int)feat.featid,chi2,chi2_multipler*chi2_check);

        // We are good!!! Append to our large H vector
        size_t ct_hx = 0;
        for(const auto &var : Hxf_order) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping.find(var)==Hx_mapping.end()) {
                Hx_mapping.insert({var,ct_jacob});
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas,Hx_mapping[var],H_xf.rows(),var->size()) = H_xf.block(0,ct_hx,H_xf.rows(),var->size());
            ct_hx += var->size();

        }

        // Our isotropic measurement noise
        R_big.block(ct_meas,ct_meas,res.rows(),res.rows()) *= sigma_pix_sq;

        // Append our residual and move forward
        res_big.block(ct_meas,0,res.rows(),1) = res;
        ct_meas += res.rows();
        it2++;

    }
    rT2 =  boost::posix_time::microsec_clock::local_time();

    // We have appended all features to our Hx_big, res_big
    // Delete it so we do not reuse information
    for(size_t f=0; f < feature_vec.size(); f++) {
        feature_vec[f]->to_delete = true;
    }

    // Return if we don't have anything and resize our matrices
    if(ct_meas < 1) {
        return;
    }
    assert(ct_meas<=max_meas_size);
    assert(ct_jacob<=max_hx_size);
    res_big.conservativeResize(ct_meas,1);
    Hx_big.conservativeResize(ct_meas,ct_jacob);
    R_big.conservativeResize(ct_meas,ct_meas);

    // 5. With all good SLAM features update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
    rT3 =  boost::posix_time::microsec_clock::local_time();

    // Debug print timing information
    //printf("[SLAM-UP]: %.4f seconds to clean\n",(rT1-rT0).total_microseconds() * 1e-6);
    //printf("[SLAM-UP]: %.4f seconds creating linear system\n",(rT2-rT1).total_microseconds() * 1e-6);
    //printf("[SLAM-UP]: %.4f seconds to update (%d feats of %d size)\n",(rT3-rT2).total_microseconds() * 1e-6, (int)feature_vec.size(), (int)Hx_big.rows());
    //printf("[SLAM-UP]: %.4f seconds total\n",(rT3-rT1).total_microseconds() * 1e-6);

}


void UpdaterSLAM::update_skf(State *state, std::vector<Feature*>& feature_vec) {


    // Return if no features
    if(feature_vec.empty())
        return;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }


    // 1. Clean all feature measurements and make sure they all have valid clone times
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements
        int ct_meas = 0;
        for(const auto &pair : (*it0)->timestamps) {
            ct_meas += (*it0)->timestamps[pair.first].size();
        }

        // Get the landmark and its representation
        // For single depth representation we need at least two measurement
        // This is because we do nullspace projection
        Landmark* landmark = state->_features_SLAM.at((*it0)->featid);
        int required_meas = (landmark->_feat_representation==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE)? 3 : 2;

        // Remove if we don't have enough
        if(ct_meas < 1) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else if(ct_meas < required_meas) {
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }

    }
    rT1 =  boost::posix_time::microsec_clock::local_time();

    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    size_t max_meas_n_size = 0;
    for(size_t i=0; i<feature_vec.size(); i++) {
        for (const auto &pair : feature_vec.at(i)->timestamps) {
            max_meas_size += 2*feature_vec.at(i)->timestamps[pair.first].size();
        }
        //it should be zero;
        if(state->set_transform)
            max_meas_n_size += 2*feature_vec.at(i)->keyframe_matched_obs.size();
    }
    assert(max_meas_n_size==0);


    // Calculate max possible state size (i.e. the size of our covariance)
    size_t max_hx_size = state->max_covariance_size();
    size_t max_hn_size= state->max_nuisance_covariance_size();

    // Large Jacobian, residual, and measurement noise of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size+max_meas_n_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size+max_meas_n_size, max_hx_size);
    Eigen::MatrixXd Hn_big = Eigen::MatrixXd::Zero(max_meas_size+max_meas_n_size,max_hn_size); //[H_nuisance] for here, as the feas are not observed by kf, H_nuisnace = 0
    Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(max_meas_size+max_meas_n_size,max_meas_size+max_meas_n_size);
    std::unordered_map<Type*,size_t> Hx_mapping;
    std::vector<Type*> Hx_order_big;//[imu,clones_imu...]
    std::vector<Type*> Hn_order_big; //[clones_kf....]
    size_t ct_jacob = 0;
    size_t ct_jacob_n = 0;
    size_t ct_meas = 0;

    // 4. Compute linear system for each feature, nullspace project, and reject
    auto it2 = feature_vec.begin();
    while(it2 != feature_vec.end()) {

        // Ensure we have the landmark and it is the same
        assert(state->_features_SLAM.find((*it2)->featid) != state->_features_SLAM.end());
        assert(state->_features_SLAM.at((*it2)->featid)->_featid == (*it2)->featid);

        // Get our landmark from the state
        Landmark* landmark = state->_features_SLAM.at((*it2)->featid);

        // Convert the state landmark into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;

        // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        feat.feat_representation = landmark->_feat_representation;
        if(landmark->_feat_representation==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
            feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        }

        // Save the position and its fej value
        if(LandmarkRepresentation::is_relative_representation(feat.feat_representation)) {
            feat.anchor_cam_id = landmark->_anchor_cam_id;
            feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
            feat.p_FinA = landmark->get_xyz(false);
            feat.p_FinA_fej = landmark->get_xyz(true);
        } else {
            feat.p_FinG = landmark->get_xyz(false);
            feat.p_FinG_fej = landmark->get_xyz(true);
        }

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f;
        Eigen::MatrixXd H_x;
        Eigen::VectorXd res;
        std::vector<Type*> Hx_order;

        // Get the Jacobian for this feature
        UpdaterHelper::get_feature_jacobian_full(state, feat, H_f, H_x, res, Hx_order);

        // Place Jacobians in one big Jacobian, since the landmark is already in our state vector
        // H_xf=[H_x,H_f], is the augmented matrix, since landmark is also an element of the state. by zzq
        Eigen::MatrixXd H_xf = H_x;
        if(landmark->_feat_representation==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

            // Append the Jacobian in respect to the depth of the feature
            //As the landmark is represented by only the depth(i.e. depth is the element of the state
            //instead of the full 3d landmark), so we just need to append the last column(depth) into Jacobians. by zzq
            H_xf.conservativeResize(H_x.rows(), H_x.cols()+1);
            H_xf.block(0, H_x.cols(), H_x.rows(), 1) = H_f.block(0,H_f.cols()-1,H_f.rows(),1);
            H_f.conservativeResize(H_f.rows(), H_f.cols()-1);

            // Nullspace project the bearing portion
            // This takes into account that we have marginalized the bearing already
            // Thus this is crucial to ensuring estimator consistency as we are not taking the bearing to be true
            UpdaterHelper::nullspace_project_inplace(H_f, H_xf, res);

        } else {

            // Else we have the full feature in our state, so just append it
            H_xf.conservativeResize(H_x.rows(), H_x.cols()+H_f.cols());
            H_xf.block(0, H_x.cols(), H_x.rows(), H_f.cols()) = H_f;

        }

        // Append to our Jacobian order vector
        std::vector<Type*> Hxf_order = Hx_order;
        Hxf_order.push_back(landmark);

        /// Chi2 distance check
        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hxf_order);
        Eigen::MatrixXd S = H_xf*P_marg*H_xf.transpose();
        double sigma_pix_sq = ((int)feat.featid < state->_options.max_aruco_features)? _options_aruco.sigma_pix_sq : _options_slam.sigma_pix_sq;
        S.diagonal() += sigma_pix_sq*Eigen::VectorXd::Ones(S.rows());
        double chi2 = res.dot(S.llt().solve(res));

        // Get our threshold (we precompute up to 500 but handle the case that it is more)
        double chi2_check;
        if(res.rows() < 500) {
            chi2_check = chi_squared_table[res.rows()];
        } else {
            boost::math::chi_squared chi_squared_dist(res.rows());
            chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
            printf(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res.rows());
        }

        // Check if we should delete or not
        double chi2_multipler = ((int)feat.featid < state->_options.max_aruco_features)? _options_aruco.chi2_multipler : _options_slam.chi2_multipler;
        if(chi2 > chi2_multipler*chi2_check) {
            if((int)feat.featid < state->_options.max_aruco_features)
                printf(YELLOW "[SLAM-UP]: rejecting aruco tag %d for chi2 thresh (%.3f > %.3f)\n" RESET,(int)feat.featid,chi2,chi2_multipler*chi2_check);
            (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            continue;
        }

        // Debug print when we are going to update the aruco tags
        if((int)feat.featid < state->_options.max_aruco_features)
            printf("[SLAM-UP]: accepted aruco tag %d for chi2 thresh (%.3f < %.3f)\n",(int)feat.featid,chi2,chi2_multipler*chi2_check);

        // We are good!!! Append to our large H vector
        size_t ct_hx = 0;
        for(const auto &var : Hxf_order) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping.find(var)==Hx_mapping.end()) {
                Hx_mapping.insert({var,ct_jacob});
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas,Hx_mapping[var],H_xf.rows(),var->size()) = H_xf.block(0,ct_hx,H_xf.rows(),var->size());
            ct_hx += var->size();

        }

        // Our isotropic measurement noise
        R_big.block(ct_meas,ct_meas,res.rows(),res.rows()) *= sigma_pix_sq;

        // Append our residual and move forward
        res_big.block(ct_meas,0,res.rows(),1) = res;
        ct_meas += res.rows();
        it2++;

    }
    rT2 =  boost::posix_time::microsec_clock::local_time();

    // We have appended all features to our Hx_big, res_big
    // Delete it so we do not reuse information
    for(size_t f=0; f < feature_vec.size(); f++) {
        feature_vec[f]->to_delete = true;
    }

    // Return if we don't have anything and resize our matrices
    if(ct_meas < 1) {
        return;
    }
    assert(ct_meas<=max_meas_size+max_meas_n_size);
    assert(ct_jacob<=max_hx_size);
    assert(ct_jacob_n<=max_hn_size);
    res_big.conservativeResize(ct_meas,1);
    Hx_big.conservativeResize(ct_meas,ct_jacob);
    R_big.conservativeResize(ct_meas,ct_meas);
    Hn_big.conservativeResize(ct_meas,ct_jacob_n);

    // 5. With all good SLAM features update the state
    if(!state->_options.full_ekf)
        StateHelper::SKFUpdate(state, Hx_order_big,Hn_order_big, Hx_big,Hn_big, res_big, R_big,false);
    else
        StateHelper::EKFMAPUpdate(state, Hx_order_big,Hn_order_big, Hx_big,Hn_big, res_big, R_big,false);
    rT3 =  boost::posix_time::microsec_clock::local_time();

    // Debug print timing information
    //printf("[SLAM-UP]: %.4f seconds to clean\n",(rT1-rT0).total_microseconds() * 1e-6);
    //printf("[SLAM-UP]: %.4f seconds creating linear system\n",(rT2-rT1).total_microseconds() * 1e-6);
    //printf("[SLAM-UP]: %.4f seconds to update (%d feats of %d size)\n",(rT3-rT2).total_microseconds() * 1e-6, (int)feature_vec.size(), (int)Hx_big.rows());
    //printf("[SLAM-UP]: %.4f seconds total\n",(rT3-rT1).total_microseconds() * 1e-6);

}



void UpdaterSLAM::change_anchors(State* state) {

    // Return if we do not have enough clones
    if ((int)state->_clones_IMU.size() <= state->_options.max_clone_size) {
        return;
    }

    // Get the marginalization timestep, and change the anchor for any feature seen from it
    // NOTE: for now we have anchor the feature in the same camera as it is before
    // NOTE: this also does not change the representation of the feature at all right now
    double marg_timestep = state->margtimestep();
    for (auto &f : state->_features_SLAM) {
        // Skip any features that are in the global frame
        if(f.second->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D
            || f.second->_feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH)
            continue;
        // Else lets see if it is anchored in the clone that will be marginalized
        assert(marg_timestep <= f.second->_anchor_clone_timestamp);
        if (f.second->_anchor_clone_timestamp == marg_timestep) {
            perform_anchor_change(state, f.second, state->_timestamp, f.second->_anchor_cam_id);
        }
    }

}




void UpdaterSLAM::perform_anchor_change(State* state, Landmark* landmark, double new_anchor_timestamp, size_t new_cam_id) {

    // Assert that this is an anchored representation
    assert(LandmarkRepresentation::is_relative_representation(landmark->_feat_representation));
    assert(landmark->_anchor_cam_id != -1);

    // Create current feature representation
    UpdaterHelper::UpdaterHelperFeature old_feat;
    old_feat.featid = landmark->_featid;
    old_feat.feat_representation = landmark->_feat_representation;
    old_feat.anchor_cam_id = landmark->_anchor_cam_id;
    old_feat.anchor_clone_timestamp = landmark->_anchor_clone_timestamp;
    old_feat.p_FinA = landmark->get_xyz(false);
    old_feat.p_FinA_fej = landmark->get_xyz(true);

    // Get Jacobians of p_FinG wrt old representation
    Eigen::MatrixXd H_f_old;
    std::vector<Eigen::MatrixXd> H_x_old;
    std::vector<Type*> x_order_old;
    UpdaterHelper::get_feature_jacobian_representation(state, old_feat, H_f_old, H_x_old, x_order_old);

    // Create future feature representation
    UpdaterHelper::UpdaterHelperFeature new_feat;
    new_feat.featid = landmark->_featid;
    new_feat.feat_representation = landmark->_feat_representation;
    new_feat.anchor_cam_id = new_cam_id;
    new_feat.anchor_clone_timestamp = new_anchor_timestamp;

    //==========================================================================
    //==========================================================================

    // OLD: anchor camera position and orientation
    Eigen::Matrix<double,3,3> R_GtoIOLD = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->Rot();
    Eigen::Matrix<double,3,3> R_GtoOLD = state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->Rot()*R_GtoIOLD;
    Eigen::Matrix<double,3,1> p_OLDinG = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->pos()-R_GtoOLD.transpose()*state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->pos();

    // NEW: anchor camera position and orientation
    Eigen::Matrix<double,3,3> R_GtoINEW = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->Rot();
    Eigen::Matrix<double,3,3> R_GtoNEW = state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->Rot()*R_GtoINEW;
    Eigen::Matrix<double,3,1> p_NEWinG = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->pos()-R_GtoNEW.transpose()*state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->pos();

    // Calculate transform between the old anchor and new one
    Eigen::Matrix<double,3,3> R_OLDtoNEW = R_GtoNEW*R_GtoOLD.transpose();
    Eigen::Matrix<double,3,1> p_OLDinNEW = R_GtoNEW*(p_OLDinG-p_NEWinG);
    new_feat.p_FinA = R_OLDtoNEW*landmark->get_xyz(false)+p_OLDinNEW;

    //==========================================================================
    //==========================================================================

    // OLD: anchor camera position and orientation
    Eigen::Matrix<double,3,3> R_GtoIOLD_fej = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->Rot_fej();
    Eigen::Matrix<double,3,3> R_GtoOLD_fej = state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->Rot()*R_GtoIOLD_fej;
    Eigen::Matrix<double,3,1> p_OLDinG_fej = state->_clones_IMU.at(old_feat.anchor_clone_timestamp)->pos_fej()-R_GtoOLD_fej.transpose()*state->_calib_IMUtoCAM.at(old_feat.anchor_cam_id)->pos();

    // NEW: anchor camera position and orientation
    Eigen::Matrix<double,3,3> R_GtoINEW_fej = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->Rot_fej();
    Eigen::Matrix<double,3,3> R_GtoNEW_fej = state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->Rot()*R_GtoINEW_fej;
    Eigen::Matrix<double,3,1> p_NEWinG_fej = state->_clones_IMU.at(new_feat.anchor_clone_timestamp)->pos_fej()-R_GtoNEW_fej.transpose()*state->_calib_IMUtoCAM.at(new_feat.anchor_cam_id)->pos();

    // Calculate transform between the old anchor and new one
    Eigen::Matrix<double,3,3> R_OLDtoNEW_fej = R_GtoNEW_fej*R_GtoOLD_fej.transpose();
    Eigen::Matrix<double,3,1> p_OLDinNEW_fej = R_GtoNEW_fej*(p_OLDinG_fej-p_NEWinG_fej);
    new_feat.p_FinA_fej = R_OLDtoNEW_fej*landmark->get_xyz(true)+p_OLDinNEW_fej;

    // Get Jacobians of p_FinG wrt new representation
    Eigen::MatrixXd H_f_new;
    std::vector<Eigen::MatrixXd> H_x_new;
    std::vector<Type*> x_order_new;
    UpdaterHelper::get_feature_jacobian_representation(state, new_feat, H_f_new, H_x_new, x_order_new);

    //==========================================================================
    //==========================================================================

    // New phi order is just the landmark
    std::vector<Type*> phi_order_NEW;
    phi_order_NEW.push_back(landmark);

    // Loop through all our orders and append them
    std::vector<Type*> phi_order_OLD;
    int current_it = 0;
    std::map<Type*, int> Phi_id_map;
    for (const auto &var: x_order_old) {
        if(Phi_id_map.find(var) == Phi_id_map.end()) {
            Phi_id_map.insert({var,current_it});
            phi_order_OLD.push_back(var);
            current_it += var->size();
        }
    }
    for (const auto &var: x_order_new) {
        if(Phi_id_map.find(var) == Phi_id_map.end()) {
            Phi_id_map.insert({var,current_it});
            phi_order_OLD.push_back(var);
            current_it += var->size();
        }
    }
    Phi_id_map.insert({landmark,current_it});
    phi_order_OLD.push_back(landmark);
    current_it += landmark->size();

    // Anchor change Jacobian
    int phisize = (new_feat.feat_representation!=LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
    Eigen::MatrixXd Phi = Eigen::MatrixXd::Zero(phisize, current_it);
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(phisize, phisize);

    // Inverse of our new representation
    // pf_new_error = Hfnew^{-1}*(Hfold*pf_olderror+Hxold*x_olderror-Hxnew*x_newerror)
    Eigen::MatrixXd H_f_new_inv;
    if(phisize==1) {
        H_f_new_inv = 1.0/H_f_new.squaredNorm()*H_f_new.transpose();
    } else {
        H_f_new_inv = H_f_new.colPivHouseholderQr().solve(Eigen::Matrix<double,3,3>::Identity());
    }

    // Place Jacobians for old anchor
    for (size_t i=0; i<H_x_old.size(); i++) {
        Phi.block(0,Phi_id_map.at(x_order_old[i]),phisize,x_order_old[i]->size()).noalias() += H_f_new_inv*H_x_old[i];
    }

    // Place Jacobians for old feat
    Phi.block(0,Phi_id_map.at(landmark),phisize,phisize) = H_f_new_inv*H_f_old;

    // Place Jacobians for new anchor
    for (size_t i=0; i<H_x_new.size(); i++) {
        Phi.block(0,Phi_id_map.at(x_order_new[i]),phisize,x_order_new[i]->size()).noalias() -= H_f_new_inv*H_x_new[i];
    }

    // Perform covariance propagation
    StateHelper::EKFPropagation(state, phi_order_NEW, phi_order_OLD, Phi, Q);

    // Set state from new feature
    landmark->_featid = new_feat.featid;
    landmark->_feat_representation = new_feat.feat_representation;
    landmark->_anchor_cam_id = new_feat.anchor_cam_id;
    landmark->_anchor_clone_timestamp = new_feat.anchor_clone_timestamp;
    landmark->set_from_xyz(new_feat.p_FinA, false);
    landmark->set_from_xyz(new_feat.p_FinA_fej, true);
    landmark->has_had_anchor_change = true;

}





