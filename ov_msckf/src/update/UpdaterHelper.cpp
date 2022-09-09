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
#include "UpdaterHelper.h"


using namespace ov_core;
using namespace ov_msckf;


void UpdaterHelper::get_feature_jacobian_representation(State* state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                        std::vector<Eigen::MatrixXd> &H_x, std::vector<Type*> &x_order) {

    // Global XYZ representation
    if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D) {
        H_f.resize(3,3);
        H_f.setIdentity();
        return;
    }

    // Global inverse depth representation
    if (feature.feat_representation == LandmarkRepresentation::Representation::GLOBAL_FULL_INVERSE_DEPTH) {

        // Get the feature linearization point
        Eigen::Matrix<double,3,1> p_FinG = (state->_options.do_fej)? feature.p_FinG_fej : feature.p_FinG;

        // Get inverse depth representation (should match what is in Landmark.cpp)
        double g_rho = 1/p_FinG.norm();
        double g_phi = std::acos(g_rho*p_FinG(2));
        //double g_theta = std::asin(g_rho*p_FinG(1)/std::sin(g_phi));
        double g_theta = std::atan2(p_FinG(1),p_FinG(0));
        Eigen::Matrix<double,3,1> p_invFinG;
        p_invFinG(0) = g_theta;
        p_invFinG(1) = g_phi;
        p_invFinG(2) = g_rho;

        // Get inverse depth bearings
        double sin_th = std::sin(p_invFinG(0,0));
        double cos_th = std::cos(p_invFinG(0,0));
        double sin_phi = std::sin(p_invFinG(1,0));
        double cos_phi = std::cos(p_invFinG(1,0));
        double rho = p_invFinG(2,0);

        // Construct the Jacobian
        H_f.resize(3,3);
        H_f << -(1.0/rho)*sin_th*sin_phi, (1.0/rho)*cos_th*cos_phi, -(1.0/(rho*rho))*cos_th*sin_phi,
                (1.0/rho)*cos_th*sin_phi, (1.0/rho)*sin_th*cos_phi, -(1.0/(rho*rho))*sin_th*sin_phi,
                0.0, -(1.0/rho)*sin_phi, -(1.0/(rho*rho))*cos_phi;
        return;
    }


    //======================================================================
    //======================================================================
    //======================================================================


    // Assert that we have an anchor pose for this feature
    assert(feature.anchor_cam_id!=-1);

    // Anchor pose orientation and position, and camera calibration for our anchor camera
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
    Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
    Eigen::Vector3d p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
    Eigen::Vector3d p_FinA = feature.p_FinA;

    // If I am doing FEJ, I should FEJ the anchor states (should we fej calibration???)
    // Also get the FEJ position of the feature if we are
    if(state->_options.do_fej) {
        // "Best" feature in the global frame
        Eigen::Vector3d p_FinG_best = R_GtoI.transpose() * R_ItoC.transpose()*(feature.p_FinA - p_IinC) + p_IinG;
        // Transform the best into our anchor frame using FEJ
        R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot_fej();
        p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos_fej();
        p_FinA = (R_GtoI.transpose()*R_ItoC.transpose()).transpose()*(p_FinG_best - p_IinG) + p_IinC;
    }
    Eigen::Matrix3d R_CtoG = R_GtoI.transpose()*R_ItoC.transpose();

    // Jacobian for our anchor pose
    Eigen::Matrix<double,3,6> H_anc;
    H_anc.block(0,0,3,3).noalias() = -R_GtoI.transpose()*skew_x(R_ItoC.transpose()*(p_FinA-p_IinC));
    H_anc.block(0,3,3,3).setIdentity();

    // Add anchor Jacobians to our return vector
    x_order.push_back(state->_clones_IMU.at(feature.anchor_clone_timestamp));
    H_x.push_back(H_anc);

    // Get calibration Jacobians (for anchor clone)
    if (state->_options.do_calib_camera_pose) {
        Eigen::Matrix<double,3,6> H_calib;
        H_calib.block(0,0,3,3).noalias() = -R_CtoG*skew_x(p_FinA-p_IinC);
        H_calib.block(0,3,3,3) = -R_CtoG;
        x_order.push_back(state->_calib_IMUtoCAM.at(feature.anchor_cam_id));
        H_x.push_back(H_calib);
    }

    // If we are doing anchored XYZ feature
    if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_3D) {
        H_f = R_CtoG;
        return;
    }

    // If we are doing full inverse depth
    if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_FULL_INVERSE_DEPTH) {

        // Get inverse depth representation (should match what is in Landmark.cpp)
        double a_rho = 1/p_FinA.norm();
        double a_phi = std::acos(a_rho*p_FinA(2));
        double a_theta = std::atan2(p_FinA(1),p_FinA(0));
        Eigen::Matrix<double,3,1> p_invFinA;
        p_invFinA(0) = a_theta;
        p_invFinA(1) = a_phi;
        p_invFinA(2) = a_rho;

        // Using anchored inverse depth
        double sin_th = std::sin(p_invFinA(0,0));
        double cos_th = std::cos(p_invFinA(0,0));
        double sin_phi = std::sin(p_invFinA(1,0));
        double cos_phi = std::cos(p_invFinA(1,0));
        double rho = p_invFinA(2,0);
        //assert(p_invFinA(2,0)>=0.0);

        // Jacobian of anchored 3D position wrt inverse depth parameters
        Eigen::Matrix<double,3,3> d_pfinA_dpinv;
        d_pfinA_dpinv << -(1.0/rho)*sin_th*sin_phi, (1.0/rho)*cos_th*cos_phi, -(1.0/(rho*rho))*cos_th*sin_phi,
                (1.0/rho)*cos_th*sin_phi, (1.0/rho)*sin_th*cos_phi, -(1.0/(rho*rho))*sin_th*sin_phi,
                0.0, -(1.0/rho)*sin_phi, -(1.0/(rho*rho))*cos_phi;
        H_f = R_CtoG*d_pfinA_dpinv;
        return;
    }

    // If we are doing the MSCKF version of inverse depth
    if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH) {

        // Get inverse depth representation (should match what is in Landmark.cpp)
        Eigen::Matrix<double,3,1> p_invFinA_MSCKF;
        p_invFinA_MSCKF(0) = p_FinA(0)/p_FinA(2);
        p_invFinA_MSCKF(1) = p_FinA(1)/p_FinA(2);
        p_invFinA_MSCKF(2) = 1/p_FinA(2);

        // Using the MSCKF version of inverse depth
        double alpha = p_invFinA_MSCKF(0,0);
        double beta = p_invFinA_MSCKF(1,0);
        double rho = p_invFinA_MSCKF(2,0);

        // Jacobian of anchored 3D position wrt inverse depth parameters
        Eigen::Matrix<double,3,3> d_pfinA_dpinv;
        d_pfinA_dpinv << (1.0/rho), 0.0, -(1.0/(rho*rho))*alpha,
                0.0, (1.0/rho), -(1.0/(rho*rho))*beta,
                0.0, 0.0, -(1.0/(rho*rho));
        H_f = R_CtoG*d_pfinA_dpinv;
        return;
    }

    /// CASE: Estimate single depth of the feature using the initial bearing
    if (feature.feat_representation == LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {

        // Get inverse depth representation (should match what is in Landmark.cpp)
        double rho = 1.0/p_FinA(2);
        Eigen::Vector3d bearing = rho*p_FinA;

        // Jacobian of anchored 3D position wrt inverse depth parameters
        Eigen::Vector3d d_pfinA_drho;
        d_pfinA_drho << -(1.0/(rho*rho))*bearing;
        H_f = R_CtoG*d_pfinA_drho;
        return;

    }

    // Failure, invalid representation that is not programmed
    assert(false);

}



void UpdaterHelper::get_feature_jacobian_intrinsics(State* state, const Eigen::Vector2d &uv_norm, bool isfisheye, Eigen::Matrix<double,8,1> cam_d, Eigen::Matrix<double,2,2> &dz_dzn, Eigen::Matrix<double,2,8> &dz_dzeta) {

    // Calculate distortion uv and jacobian
    if(isfisheye) {

        // Calculate distorted coordinates for fisheye
        double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
        double theta = std::atan(r);
        double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

        // Handle when r is small (meaning our xy is near the camera center)
        double inv_r = (r > 1e-8)? 1.0/r : 1.0;
        double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

        // Jacobian of distorted pixel to "normalized" pixel
        Eigen::Matrix<double,2,2> duv_dxy = Eigen::Matrix<double,2,2>::Zero();
        duv_dxy << cam_d(0), 0, 0, cam_d(1);

        // Jacobian of "normalized" pixel to normalized pixel
        Eigen::Matrix<double,2,2> dxy_dxyn = Eigen::Matrix<double,2,2>::Zero();
        dxy_dxyn << theta_d*inv_r, 0, 0, theta_d*inv_r;

        // Jacobian of "normalized" pixel to r
        Eigen::Matrix<double,2,1> dxy_dr = Eigen::Matrix<double,2,1>::Zero();
        dxy_dr << -uv_norm(0)*theta_d*inv_r*inv_r, -uv_norm(1)*theta_d*inv_r*inv_r;

        // Jacobian of r pixel to normalized xy
        Eigen::Matrix<double,1,2> dr_dxyn = Eigen::Matrix<double,1,2>::Zero();
        dr_dxyn << uv_norm(0)*inv_r, uv_norm(1)*inv_r;

        // Jacobian of "normalized" pixel to theta_d
        Eigen::Matrix<double,2,1> dxy_dthd = Eigen::Matrix<double,2,1>::Zero();
        dxy_dthd << uv_norm(0)*inv_r, uv_norm(1)*inv_r;

        // Jacobian of theta_d to theta
        double dthd_dth = 1+3*cam_d(4)*std::pow(theta,2)+5*cam_d(5)*std::pow(theta,4)+7*cam_d(6)*std::pow(theta,6)+9*cam_d(7)*std::pow(theta,8);

        // Jacobian of theta to r
        double dth_dr = 1/(r*r+1);

        // Total Jacobian wrt normalized pixel coordinates
        dz_dzn = duv_dxy*(dxy_dxyn+(dxy_dr+dxy_dthd*dthd_dth*dth_dr)*dr_dxyn);

        // Compute the Jacobian in respect to the intrinsics if we are calibrating
        if(state->_options.do_calib_camera_intrinsics) {

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm(0)*cdist;
            double y1 = uv_norm(1)*cdist;

            // Jacobian
            dz_dzeta(0,0) = x1;
            dz_dzeta(0,2) = 1;
            dz_dzeta(0,4) = cam_d(0)*uv_norm(0)*inv_r*std::pow(theta,3);
            dz_dzeta(0,5) = cam_d(0)*uv_norm(0)*inv_r*std::pow(theta,5);
            dz_dzeta(0,6) = cam_d(0)*uv_norm(0)*inv_r*std::pow(theta,7);
            dz_dzeta(0,7) = cam_d(0)*uv_norm(0)*inv_r*std::pow(theta,9);
            dz_dzeta(1,1) = y1;
            dz_dzeta(1,3) = 1;
            dz_dzeta(1,4) = cam_d(1)*uv_norm(1)*inv_r*std::pow(theta,3);
            dz_dzeta(1,5) = cam_d(1)*uv_norm(1)*inv_r*std::pow(theta,5);
            dz_dzeta(1,6) = cam_d(1)*uv_norm(1)*inv_r*std::pow(theta,7);
            dz_dzeta(1,7) = cam_d(1)*uv_norm(1)*inv_r*std::pow(theta,9);
        }


    } else {

        // Calculate distorted coordinates for radial
        double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
        double r_2 = r*r;
        double r_4 = r_2*r_2;

        // Jacobian of distorted pixel to normalized pixel
        double x = uv_norm(0);
        double y = uv_norm(1);
        double x_2 = uv_norm(0)*uv_norm(0);
        double y_2 = uv_norm(1)*uv_norm(1);
        double x_y = uv_norm(0)*uv_norm(1);
        dz_dzn(0,0) = cam_d(0)*((1+cam_d(4)*r_2+cam_d(5)*r_4)+(2*cam_d(4)*x_2+4*cam_d(5)*x_2*r)+2*cam_d(6)*y+(2*cam_d(7)*x+4*cam_d(7)*x));
        dz_dzn(0,1) = cam_d(0)*(2*cam_d(4)*x_y+4*cam_d(5)*x_y*r+2*cam_d(6)*x+2*cam_d(7)*y);
        dz_dzn(1,0) = cam_d(1)*(2*cam_d(4)*x_y+4*cam_d(5)*x_y*r+2*cam_d(6)*x+2*cam_d(7)*y);
        dz_dzn(1,1) = cam_d(1)*((1+cam_d(4)*r_2+cam_d(5)*r_4)+(2*cam_d(4)*y_2+4*cam_d(5)*y_2*r)+2*cam_d(7)*x+(2*cam_d(6)*y+4*cam_d(6)*y));

        // Compute the Jacobian in respect to the intrinsics if we are calibrating
        if(state->_options.do_calib_camera_intrinsics) {

            // Calculate distorted coordinates for radtan
            double x1 = uv_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm(0)*uv_norm(1)+cam_d(7)*(r_2+2*uv_norm(0)*uv_norm(0));
            double y1 = uv_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm(1)*uv_norm(1))+2*cam_d(7)*uv_norm(0)*uv_norm(1);

            // Jacobian
            dz_dzeta(0,0) = x1;
            dz_dzeta(0,2) = 1;
            dz_dzeta(0,4) = cam_d(0)*uv_norm(0)*r_2;
            dz_dzeta(0,5) = cam_d(0)*uv_norm(0)*r_4;
            dz_dzeta(0,6) = 2*cam_d(0)*uv_norm(0)*uv_norm(1);
            dz_dzeta(0,7) = cam_d(0)*(r_2+2*uv_norm(0)*uv_norm(0));
            dz_dzeta(1,1) = y1;
            dz_dzeta(1,3) = 1;
            dz_dzeta(1,4) = cam_d(1)*uv_norm(1)*r_2;
            dz_dzeta(1,5) = cam_d(1)*uv_norm(1)*r_4;
            dz_dzeta(1,6) = cam_d(1)*(r_2+2*uv_norm(1)*uv_norm(1));
            dz_dzeta(1,7) = 2*cam_d(1)*uv_norm(0)*uv_norm(1);
        }

    }


}



void UpdaterHelper::get_feature_jacobian_full(State* state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                              Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<Type*> &x_order) {

    // Total number of measurements for this feature
    int total_meas = 0;
    for (auto const& pair : feature.timestamps) {
        total_meas += (int)pair.second.size();
    }
    // cout<<"num of meas of current feature: "<<total_meas<<endl;

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    std::unordered_map<Type*,size_t> map_hx;
    for (auto const& pair : feature.timestamps) {

        // Our extrinsics and intrinsics
        PoseJPL *calibration = state->_calib_IMUtoCAM.at(pair.first);
        Vec *distortion = state->_cam_intrinsics.at(pair.first);

        // If doing calibration extrinsics
        if(state->_options.do_calib_camera_pose) {
            map_hx.insert({calibration,total_hx});
            x_order.push_back(calibration);
            total_hx += calibration->size();
        }

        // If doing calibration intrinsics
        if(state->_options.do_calib_camera_intrinsics) {
            map_hx.insert({distortion,total_hx});
            x_order.push_back(distortion);
            total_hx += distortion->size();
        }

        // Loop through all measurements for this specific camera
        for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

            // Add this clone if it is not added already
            PoseJPL *clone_Ci = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
            if(map_hx.find(clone_Ci) == map_hx.end()) {
                map_hx.insert({clone_Ci,total_hx});
                x_order.push_back(clone_Ci);
                total_hx += clone_Ci->size();
            }

        }

    }

    // If we are using an anchored representation, make sure that the anchor is also added
    if (LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {

        // Assert we have a clone
        assert(feature.anchor_cam_id != -1);

        // Add this anchor if it is not added already
        PoseJPL *clone_Ai = state->_clones_IMU.at(feature.anchor_clone_timestamp);
        if(map_hx.find(clone_Ai) == map_hx.end()) {
            map_hx.insert({clone_Ai,total_hx});
            x_order.push_back(clone_Ai);
            total_hx += clone_Ai->size();
        }

        // Also add its calibration if we are doing calibration
        if(state->_options.do_calib_camera_pose) {
            // Add this anchor if it is not added already
            PoseJPL *clone_calib = state->_calib_IMUtoCAM.at(feature.anchor_cam_id);
            if(map_hx.find(clone_calib) == map_hx.end()) {
                map_hx.insert({clone_calib,total_hx});
                x_order.push_back(clone_calib);
                total_hx += clone_calib->size();
            }
        }

    }

    //=========================================================================
    //=========================================================================

    // Calculate the position of this feature in the global frame
    // If anchored, then we need to calculate the position of the feature in the global
    Eigen::Vector3d p_FinG = feature.p_FinG;
    if(LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
        // Assert that we have an anchor pose for this feature
        assert(feature.anchor_cam_id!=-1);
        // Get calibration for our anchor camera
        Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
        Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
        // Anchor pose orientation and position
        Eigen::Matrix<double,3,3> R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
        Eigen::Matrix<double,3,1> p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
        // Feature in the global frame
        p_FinG = R_GtoI.transpose() * R_ItoC.transpose()*(feature.p_FinA - p_IinC) + p_IinG;
    }

    // Calculate the position of this feature in the global frame FEJ
    // If anchored, then we can use the "best" p_FinG since the value of p_FinA does not matter
    Eigen::Vector3d p_FinG_fej = feature.p_FinG_fej;
    if(LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
        p_FinG_fej = p_FinG;
    }

    Eigen::Vector3d p_FinG_true = Eigen::Vector3d::Zero();
    if(state->_options.use_gt)
    {
      int featid = feature.featid - state->_options.max_aruco_features - 1;
      cout<<"featureid: "<<featid<<" _featmap size: "<<state->_featmap.size()<<endl;
      assert(state->_featmap.find(featid)!=state->_featmap.end());
      p_FinG_true = state->_featmap.at(featid);
    }
    //=========================================================================
    //=========================================================================

    // Allocate our residual and Jacobians
    int c = 0;
    int jacobsize = (feature.feat_representation!=LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
    res = Eigen::VectorXd::Zero(2*total_meas);
    H_f = Eigen::MatrixXd::Zero(2*total_meas,jacobsize);
    H_x = Eigen::MatrixXd::Zero(2*total_meas,total_hx);

    // Derivative of p_FinG in respect to feature representation.
    // This only needs to be computed once and thus we pull it out of the loop below
    Eigen::MatrixXd dpfg_dlambda;
    std::vector<Eigen::MatrixXd> dpfg_dx;
    std::vector<Type*> dpfg_dx_order;
    UpdaterHelper::get_feature_jacobian_representation(state, feature, dpfg_dlambda, dpfg_dx, dpfg_dx_order);

    // Assert that all the ones in our order are already in our local jacobian mapping
    for(auto &type : dpfg_dx_order) {
        assert(map_hx.find(type)!=map_hx.end());
    }

    // Loop through each camera for this feature
    for (auto const& pair : feature.timestamps) {

        // Our calibration between the IMU and CAMi frames
        Vec* distortion = state->_cam_intrinsics.at(pair.first);
        PoseJPL* calibration = state->_calib_IMUtoCAM.at(pair.first);
        Eigen::Matrix<double,3,3> R_ItoC = calibration->Rot();
        Eigen::Matrix<double,3,1> p_IinC = calibration->pos();
        Eigen::Matrix<double,8,1> cam_d = distortion->value();

        // Loop through all measurements for this specific camera
        for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

            //=========================================================================
            //=========================================================================

            // Get current IMU clone state
            PoseJPL* clone_Ii = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));
            Matrix<double,17,1> state_true_i = Matrix<double,17,1>::Zero();
            IMU* imu_true_i = new IMU();
            if(state->_options.use_gt)
            {
              bool success = state->get_state(feature.timestamps[pair.first].at(m),state_true_i);
              assert(success);
              imu_true_i->set_value(state_true_i.block(1,0,16,1));
            }
            
            

            Eigen::Matrix<double,3,3> R_GtoIi = clone_Ii->Rot();
            Eigen::Matrix<double,3,1> p_IiinG = clone_Ii->pos();


            // Get current feature in the IMU
            Eigen::Matrix<double,3,1> p_FinIi = R_GtoIi*(p_FinG-p_IiinG);

           

            // Project the current feature into the current frame of reference to get predicted measurement
            Eigen::Matrix<double,3,1> p_FinCi = R_ItoC*p_FinIi+p_IinC;
            Eigen::Matrix<double,2,1> uv_norm;
            uv_norm << p_FinCi(0)/p_FinCi(2),p_FinCi(1)/p_FinCi(2);

            // Distort the normalized coordinates (false=radtan, true=fisheye)
            Eigen::Matrix<double,2,1> uv_dist;

            // Calculate distortion uv and jacobian
            if(state->_cam_intrinsics_model.at(pair.first)) {

                // Calculate distorted coordinates for fisheye
                double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
                double theta = std::atan(r);
                double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

                // Handle when r is small (meaning our xy is near the camera center)
                double inv_r = (r > 1e-8)? 1.0/r : 1.0;
                double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

                // Calculate distorted coordinates for fisheye
                double x1 = uv_norm(0)*cdist;
                double y1 = uv_norm(1)*cdist;
                uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                uv_dist(1) = cam_d(1)*y1 + cam_d(3);

            } else {

                // Calculate distorted coordinates for radial
                double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
                double r_2 = r*r;
                double r_4 = r_2*r_2;
                double x1 = uv_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm(0)*uv_norm(1)+cam_d(7)*(r_2+2*uv_norm(0)*uv_norm(0));
                double y1 = uv_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm(1)*uv_norm(1))+2*cam_d(7)*uv_norm(0)*uv_norm(1);
                uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                uv_dist(1) = cam_d(1)*y1 + cam_d(3);

            }

            // Our residual
            Eigen::Matrix<double,2,1> uv_m;
            uv_m << (double)feature.uvs[pair.first].at(m)(0), (double)feature.uvs[pair.first].at(m)(1);
            res.block(2*c,0,2,1) = uv_m - uv_dist;
            // cout<<"uv_m: "<<uv_m.transpose()<<" uv_dist: "<<uv_dist.transpose()<<endl;
            // cout<<"measure error: "<< (uv_m-uv_dist).transpose() <<endl;


            //=========================================================================
            //=========================================================================

            // If we are doing first estimate Jacobians, then overwrite with the first estimates
            if(state->_options.do_fej) {
                R_GtoIi = clone_Ii->Rot_fej();
                p_IiinG = clone_Ii->pos_fej();
                //R_ItoC = calibration->Rot_fej();
                //p_IinC = calibration->pos_fej();
                p_FinIi = R_GtoIi*(p_FinG_fej-p_IiinG);
                p_FinCi = R_ItoC*p_FinIi+p_IinC;
                //uv_norm << p_FinCi(0)/p_FinCi(2),p_FinCi(1)/p_FinCi(2);
                //cam_d = state->get_intrinsics_CAM(pair.first)->fej();
            }

            if(state->_options.use_gt)
            {   

                Eigen::Matrix<double,3,3> R_GtoIi_true = imu_true_i->Rot();
                Eigen::Matrix<double,3,1> p_IiinG_true = imu_true_i->pos();

                R_GtoIi = R_GtoIi_true;
                p_IiinG = p_IiinG_true;
                //R_ItoC = calibration->Rot_fej();
                //p_IinC = calibration->pos_fej();
                p_FinIi = R_GtoIi_true*(p_FinG_true-p_IiinG_true);
                p_FinCi = R_ItoC*p_FinIi+p_IinC;
            }

            // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
            Eigen::Matrix<double,2,2> dz_dzn = Eigen::Matrix<double,2,2>::Zero();
            Eigen::Matrix<double,2,8> dz_dzeta = Eigen::Matrix<double,2,8>::Zero();
            UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm, state->_cam_intrinsics_model.at(pair.first), cam_d, dz_dzn, dz_dzeta);

            // Normalized coordinates in respect to projection function
            Eigen::Matrix<double,2,3> dzn_dpfc = Eigen::Matrix<double,2,3>::Zero();
            dzn_dpfc << 1/p_FinCi(2),0,-p_FinCi(0)/(p_FinCi(2)*p_FinCi(2)),
                    0, 1/p_FinCi(2),-p_FinCi(1)/(p_FinCi(2)*p_FinCi(2));

            // Derivative of p_FinCi in respect to p_FinIi
            Eigen::Matrix<double,3,3> dpfc_dpfg = R_ItoC*R_GtoIi;

            // Derivative of p_FinCi in respect to camera clone state
            Eigen::Matrix<double,3,6> dpfc_dclone = Eigen::Matrix<double,3,6>::Zero();
            dpfc_dclone.block(0,0,3,3).noalias() = R_ItoC*skew_x(p_FinIi);
            dpfc_dclone.block(0,3,3,3) = -dpfc_dpfg;

            //=========================================================================
            //=========================================================================


            // Precompute some matrices
            Eigen::Matrix<double,2,3> dz_dpfc = dz_dzn*dzn_dpfc;
            Eigen::Matrix<double,2,3> dz_dpfg = dz_dpfc*dpfc_dpfg;

            // CHAINRULE: get the total feature Jacobian
            H_f.block(2*c,0,2,H_f.cols()).noalias() = dz_dpfg*dpfg_dlambda;

            // CHAINRULE: get state clone Jacobian
            H_x.block(2*c,map_hx[clone_Ii],2,clone_Ii->size()).noalias() = dz_dpfc*dpfc_dclone;

            // CHAINRULE: loop through all extra states and add their
            // NOTE: we add the Jacobian here as we might be in the anchoring pose for this measurement
            for(size_t i=0; i<dpfg_dx_order.size(); i++) {
                H_x.block(2*c,map_hx[dpfg_dx_order.at(i)],2,dpfg_dx_order.at(i)->size()).noalias() += dz_dpfg*dpfg_dx.at(i);
            }


            //=========================================================================
            //=========================================================================


            // Derivative of p_FinCi in respect to camera calibration (R_ItoC, p_IinC)
            if(state->_options.do_calib_camera_pose) {

                // Calculate the Jacobian
                Eigen::Matrix<double,3,6> dpfc_dcalib = Eigen::Matrix<double,3,6>::Zero();
                dpfc_dcalib.block(0,0,3,3) = skew_x(p_FinCi-p_IinC);
                dpfc_dcalib.block(0,3,3,3) = Eigen::Matrix<double,3,3>::Identity();

                // Chainrule it and add it to the big jacobian
                H_x.block(2*c,map_hx[calibration],2,calibration->size()).noalias() += dz_dpfc*dpfc_dcalib;

            }

            // Derivative of measurement in respect to distortion parameters
            if(state->_options.do_calib_camera_intrinsics) {
                H_x.block(2*c,map_hx[distortion],2,distortion->size()) = dz_dzeta;
            }

            // Move the Jacobian and residual index forward
            c++;

        }

    }


}

void UpdaterHelper::compute_meas(Vector3d p_FinG, Vector3d p_GinW, Matrix3d R_GtoW, Vector3d p_KFinW, Matrix3d R_KFtoW, VectorXd kf_d, Vector2d& z_fdist)
{
            //get the point from VIO frame to World frame
            Vector3d p_finw=R_GtoW*p_FinG+p_GinW;
            //get the point from World frame to KF frame
            Vector3d p_finkf=R_KFtoW.transpose()*(p_finw-p_KFinW);
            //get point from KF frame to normal frame
            Vector2d z_finnorm;
            z_finnorm<<p_finkf(0)/p_finkf(2),p_finkf(1)/p_finkf(2);
            //get the normal point distort
            double xn=z_finnorm(0);
            double yn=z_finnorm(1);
            double fx=kf_d(0);
            double fy=kf_d(1);
            double cx=kf_d(2);
            double cy=kf_d(3);
            double k1=kf_d(4);
            double k2=kf_d(5);
            double p1=kf_d(6);
            double p2=kf_d(7);
            double r2=xn*xn+yn*yn;
            double x=xn*(1+k1*r2+k2*r2*r2)+2*p1*xn*yn+p2*(r2+2*xn*xn);
            double y=yn*(1+k1*r2+k2*r2*r2)+p1*(r2+2*yn*yn)+2*p2*xn*yn;
            z_fdist<<fx*x+cx,fy*y+cy;
}


bool UpdaterHelper::get_feature_jacobian_kf(State *state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                            Eigen::MatrixXd &H_n,Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<Type *> &n_order,
                                            std::vector<Type *> &x_order) {
    // Total number of measurements for this feature
    int total_meas = 0;

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    int total_hn = 0;
    std::unordered_map<Type*,size_t> map_hx;
    std::unordered_map<Type*,size_t> map_hn;
    //nuisance part
    for (auto const& pair : feature.keyframe_matched_obs) {
        for(auto const& match : pair.second)
        {
            total_meas++;
            double kf_id = match.first;
            // Keyframe* keyframe=state->get_kfdataBase()->get_keyframe(kf_id);
            PoseJPL *kf_pose=state->_clones_Keyframe[kf_id];
            assert(kf_pose!=nullptr);
            if(map_hn.find(kf_pose)==map_hn.end())
            {
                map_hn.insert({kf_pose,total_hn});
                n_order.push_back(kf_pose);
                total_hn += kf_pose->size();
            }
        }

    }
    

    // add the current frame state
    // this part is active state part

    // Add this state if it is not added already
   
    // PoseJPL *clone_Cur=nullptr;
    // clone_Cur = state->_clones_IMU.at(state->_timestamp);
    // assert(clone_Cur!=nullptr);
    // if(map_hx.find(clone_Cur) == map_hx.end()) {
    //     map_hx.insert({clone_Cur,total_hx});
    //     x_order.push_back(clone_Cur);
    //     total_hx += clone_Cur->size();
    // }

    //add related clone_IMU
    for(int i=0;i<feature.timestamps[0].size();i++)
    {
           total_meas++;
        // cout<<"feature.timestamps[0]["<<i<<"] :"<<feature.timestamps[0][i]<<endl;
        // if(feature.timestamps[0][i]==state->delay_clones[state->delay_clones.size()-1])
        // if(i==feature.timestamps[0].size()-1)
        {
            // cout<<"have current state: "<<i<<endl;
            // sleep(1);
            PoseJPL* clone=nullptr;
            clone=state->_clones_IMU.at(feature.timestamps[0][i]);
            assert(clone!=nullptr);
            if(map_hx.find(clone) == map_hx.end()) {
            map_hx.insert({clone,total_hx});
            x_order.push_back(clone);
            total_hx += clone->size();
           }
           
        }

    }
    assert(total_meas>=2);
    // // Also add its calibration if we are doing calibration
    // if(state->_options.do_calib_camera_pose) {
    //     // Add this anchor if it is not added already
    //     PoseJPL *clone_calib = state->_calib_IMUtoCAM.at(0); //we use the left cam;
    //     if(map_hx.find(clone_calib) == map_hx.end()) {
    //         map_hx.insert({clone_calib,total_hx});
    //         x_order.push_back(clone_calib);
    //         total_hx += clone_calib->size();
    //     }
    // }

    // if(state->_options.do_calib_camera_intrinsics)
    // {
    //     Vec *distortion = state->_cam_intrinsics.at(0);
    //     map_hx.insert({distortion,total_hx});
    //     x_order.push_back(distortion);
    //     total_hx += distortion->size();
    // }

    

    //we also need to add transformation between map and vio
    if(state->set_transform) //normally, if we are in this function, the state->set_transform should be already set true
    {
        PoseJPL *transform = state->transform_vio_to_map;
        assert(transform!=nullptr);
        if(map_hx.find(transform)==map_hx.end())
        {
            map_hx.insert({transform,total_hx});
            x_order.push_back(transform);
            total_hx += transform->size();
        }
    }

    //=========================================================================
    //=========================================================================

    // // Calculate the position of this feature in the global frame
    // // If anchored, then we need to calculate the position of the feature in the global(vio system reference)
    // Eigen::Vector3d p_FinG = feature.p_FinG;
    // if(LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    //     // Assert that we have an anchor pose for this feature
    //     assert(feature.anchor_cam_id!=-1);
    //     // Get calibration for our anchor camera
    //     Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
    //     Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
    //     // Anchor pose orientation and position
    //     Eigen::Matrix<double,3,3> R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
    //     Eigen::Matrix<double,3,1> p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
    //     // Feature in the global frame
    //     p_FinG = R_GtoI.transpose() * R_ItoC.transpose()*(feature.p_FinA - p_IinC) + p_IinG;
    // }

    // // Calculate the position of this feature in the global frame FEJ
    // // If anchored, then we can use the "best" p_FinG since the value of p_FinA does not matter
    // Eigen::Vector3d p_FinG_fej = feature.p_FinG_fej;
    // if(LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    //     p_FinG_fej = p_FinG;
    // }

    //=========================================================================
    //=========================================================================

    // Allocate our residual and Jacobians
    int c = 0;
    // int jacobsize = (feature.feat_representation!=LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
    int jacobsize = 3; //the observed feature is represented by 3d with respect to the loop_keyframe
    //for each feature, it has two reprojection error, in current cam frame and map_kf frame
    res = Eigen::VectorXd::Zero(2*total_meas);
    H_f = Eigen::MatrixXd::Zero(2*total_meas,jacobsize);
    H_x = Eigen::MatrixXd::Zero(2*total_meas,total_hx);
    H_n = Eigen::MatrixXd::Zero(2*total_meas,total_hn);

    vector<double> keyframe_id;
    vector<size_t> keyframe_feature_id;
    for(auto const& pair: feature.keyframe_matched_obs)
    {
        keyframe_id.clear();
        keyframe_feature_id.clear();
        for(auto const& match: pair.second)
        {
            double kf_id = match.first;
            size_t kf_feature_id = match.second;
            keyframe_id.push_back(kf_id);
            keyframe_feature_id.push_back(kf_feature_id);
        }
        //as each keyframe has a 3d point represented in itself, we choose the first kf as anchor kf,
        //and reproject its 3d point into kfs and current frame
        double kf_id = keyframe_id[0];
        Keyframe* anchor_kf = state->get_kfdataBase()->get_keyframe(kf_id);
        assert(anchor_kf != nullptr);
        size_t kf_feature_id = keyframe_feature_id[0];

        //now we reproject anchor_kf's 3d point into current frame
        //get the kf position in world frame.
        PoseJPL* anchor_kf_w = state->_clones_Keyframe[kf_id];
        Eigen::Matrix<double,3,3> R_AnchorKFtoW = anchor_kf_w->Rot();
        Eigen::Matrix<double,3,1> p_AnchorKFinW = anchor_kf_w->pos();

        //get the transform between the VIO(G) to World(W)
        PoseJPL* tranform = state->transform_vio_to_map;
        Eigen::Matrix<double,3,3> R_GtoW_linp = state->transform_vio_to_map->Rot_linp();
        Eigen::Matrix<double,3,1> p_GinW_linp = state->transform_vio_to_map->pos_linp();
        Eigen::Matrix<double,3,3> R_GtoW = state->transform_vio_to_map->Rot();
        Eigen::Matrix<double,3,1> p_GinW = state->transform_vio_to_map->pos();
        if(!state->iter)
        {
          assert(p_GinW_linp==p_GinW);
        }
        Eigen::Matrix<double,3,3> R_GtoW_true = Eigen::Matrix<double,3,3>::Identity();
        Eigen::Matrix<double,3,1> p_GinW_true = Eigen::Matrix<double,3,1>::Zero();

        double ts = pair.first; //ts

        cv::Point3f p_financhor = anchor_kf->point_3d_map.at(ts)[kf_feature_id];
        Vector3d p_FinAnchorKF(double(p_financhor.x),double(p_financhor.y),double(p_financhor.z));
        cv::Point3f p_financhor_linp = anchor_kf->point_3d_linp_map.at(ts)[kf_feature_id];
        Vector3d p_FinAnchorKF_linp(double(p_financhor_linp.x),double(p_financhor_linp.y),double(p_financhor_linp.z));
        assert(p_financhor==p_financhor_linp);
        // cv::Point2f uv_ob = anchor_kf->matched_point_2d_uv_map.at(ts)[kf_feature_id];
        // double x = uv_ob.x;
        // double y = uv_ob.y;
        // Vector2d uv_OB(x,y);
        // double uv_x = feature.uvs[0][0](0);
        // double uv_y = feature.uvs[0][0](1);
        // Vector2d uv_feat(uv_x,uv_y);
        // assert(uv_OB == uv_feat); //check if feature link is correct

        //transform the feature to map reference;
        //hk
        Eigen::Vector3d p_FinW =  R_AnchorKFtoW * p_FinAnchorKF + p_AnchorKFinW;
        Eigen::Vector3d p_FinW_linp = R_AnchorKFtoW * p_FinAnchorKF_linp + p_AnchorKFinW;
        if(!state->iter)
        {
          assert(p_FinW==p_FinW_linp);
        }

        //dhk_dpfk
        Eigen::Matrix<double,3,3> dpfw_dpfk=R_AnchorKFtoW;

        //dhk_dxn
        Eigen::Matrix<double,3,6> dpfw_dxnanchor=Eigen::Matrix<double,3,6>::Zero();
        dpfw_dxnanchor.block(0,0,3,3) = skew_x(R_AnchorKFtoW*p_FinAnchorKF_linp);
        dpfw_dxnanchor.block(0,3,3,3) = Eigen::Matrix3d::Identity();

        //transform the feature in map reference into vio reference
        //hw
        Eigen::Vector3d p_FinG = R_GtoW.transpose()*(p_FinW-p_GinW);
        Eigen::Vector3d p_FinG_linp = R_GtoW_linp.transpose()*(p_FinW_linp-p_GinW_linp);
        Eigen::Vector3d p_FinG_true = R_GtoW_true.transpose()*(p_FinW-p_GinW_true);
        //dhw_dpfw
        Eigen::Matrix<double,3,3> dpfg_dpfw = R_GtoW_linp.transpose();
        if(state->_options.use_gt)
        {
          dpfg_dpfw = R_GtoW_true.transpose();
        }
        //dhw_dxtrans
        Eigen::Matrix<double,3,6> dpfg_dxtrans = Eigen::Matrix<double,3,6>::Zero();
        dpfg_dxtrans.block(0,0,3,3) =-R_GtoW_linp.transpose()*skew_x(p_FinW_linp-p_GinW_linp);
        dpfg_dxtrans.block(0,3,3,3) = -R_GtoW_linp.transpose();
        if(state->_options.use_gt)
        {
          dpfg_dxtrans.block(0,0,3,3) =-R_GtoW_true.transpose()*skew_x(p_FinW-p_GinW_true);
          dpfg_dxtrans.block(0,3,3,3) = -R_GtoW_true.transpose();
        }

        //transform the feature in vio reference into current camera frame
        PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
        Matrix3d R_ItoC_linp = extrinsics->Rot_linp();
        Vector3d p_IinC_linp = extrinsics->pos_linp();
        Matrix3d R_ItoC = extrinsics->Rot();
        Vector3d p_IinC = extrinsics->pos();
        assert(p_IinC_linp==p_IinC);
        for(int i=0;i<feature.timestamps[0].size();i++)
        {
            PoseJPL *clone_Cur = state->_clones_IMU.at(feature.timestamps[0][i]);
            assert(clone_Cur!=nullptr);

            Eigen::Matrix<double,17,1> state_true_cur = Eigen::Matrix<double,17,1>::Zero();
            IMU* imu_true_cur = new IMU();
            if(state->_options.use_gt)
            {
              bool success = state->get_state(feature.timestamps[0][i],state_true_cur);
              assert(success);
            }
            imu_true_cur->set_value(state_true_cur.block(1,0,16,1));

            Matrix3d R_GtoI_fej = clone_Cur->Rot_fej();
            Vector3d p_IinG_fej = clone_Cur->pos_fej();
            Matrix3d R_GtoI_linp = clone_Cur->Rot_linp();
            Vector3d p_IinG_linp = clone_Cur->pos_linp();
            Matrix3d R_GtoI = clone_Cur->Rot();
            Vector3d p_IinG = clone_Cur->pos();
            Matrix3d R_GtoI_true = imu_true_cur->Rot();
            Vector3d p_IinG_true = imu_true_cur->pos();
            // assert(R_GtoI_linp==R_GtoI);
            //ht
            Eigen::Vector3d p_FinC = R_ItoC*R_GtoI*(p_FinG-p_IinG) + p_IinC;
            Eigen::Vector3d p_FinC_linp = R_ItoC_linp*R_GtoI_linp*(p_FinG_linp-p_IinG_linp) + p_IinC_linp;
            Eigen::Vector3d p_FinC_true = R_ItoC_linp*R_GtoI_true*(p_FinG_true-p_IinG_true) + p_IinC_linp;
            //dht_dpfg
            // Eigen::Matrix3d dpfc_dpfg = R_ItoC_linp*R_GtoI_fej;
            Eigen::Matrix3d dpfc_dpfg = R_ItoC_linp*R_GtoI_linp;
            if(state->_options.use_gt)
            {
              dpfc_dpfg = R_ItoC_linp * R_GtoI_true;
            }
            //dht_dxcur
            Eigen::Matrix<double,3,6> dpfc_dxcur = Eigen::Matrix<double,3,6>::Zero();
            // dpfc_dxcur.block(0,0,3,3) = R_ItoC_linp*skew_x(R_GtoI_fej*(p_FinG_linp-p_IinG_fej));
            // dpfc_dxcur.block(0,3,3,3) = -R_ItoC_linp*R_GtoI_fej;
            dpfc_dxcur.block(0,0,3,3) = R_ItoC_linp*skew_x(R_GtoI_linp*(p_FinG_linp-p_IinG_linp));
            dpfc_dxcur.block(0,3,3,3) = -R_ItoC_linp*R_GtoI_linp;
            if(state->_options.use_gt)
            {
              dpfc_dxcur.block(0,0,3,3) = R_ItoC_linp*skew_x(R_GtoI_true*(p_FinG_true-p_IinG_true));
              dpfc_dxcur.block(0,3,3,3) = -R_ItoC_linp*R_GtoI_true;
            }


            Eigen::Matrix<double,3,6> dpfc_dxe=Eigen::Matrix<double,3,6>::Zero();
            // if(state->_options.do_calib_camera_pose)
            // {

            //     dpfc_dxe.block(0,0,3,3)=skew_x(R_ItoC_linp*R_GtoI_linp*(p_FinG_linp-p_IinG_linp));
            //     dpfc_dxe.block(0,3,3,3)=Matrix3d::Identity();
            // }

            //transform the feature in current camera into normal frame
            //hp
            Eigen::Matrix<double,2,1> uv_norm;
            uv_norm<<p_FinC(0)/p_FinC(2),p_FinC(1)/p_FinC(2);
            Eigen::Matrix<double,2,1> uv_norm_linp;
            uv_norm_linp<<p_FinC_linp(0)/p_FinC_linp(2),p_FinC_linp(1)/p_FinC_linp(2);
            Eigen::Matrix<double,2,1> uv_norm_true;
            if(state->_options.use_gt)
            {
              uv_norm_true<<p_FinC_true(0)/p_FinC_true(2),p_FinC_true(1)/p_FinC_true(2);
            }
            


            // Distort the normalized coordinates (false=radtan, true=fisheye)
            Eigen::Matrix<double,2,1> uv_dist;
            Vec* distortion = state->_cam_intrinsics.at(0);
            Eigen::Matrix<double,8,1> cam_d = distortion->linp();
            // Calculate distortion uv and jacobian
            if(state->_cam_intrinsics_model.at(0)) {

                // Calculate distorted coordinates for fisheye
                double r = std::sqrt(uv_norm_linp(0)*uv_norm_linp(0)+uv_norm_linp(1)*uv_norm_linp(1));
                double theta = std::atan(r);
                double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

                // Handle when r is small (meaning our xy is near the camera center)
                double inv_r = (r > 1e-8)? 1.0/r : 1.0;
                double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

                // Calculate distorted coordinates for fisheye
                double x1 = uv_norm_linp(0)*cdist;
                double y1 = uv_norm_linp(1)*cdist;
                uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                uv_dist(1) = cam_d(1)*y1 + cam_d(3);

            } else {

                // Calculate distorted coordinates for radial
                double r = std::sqrt(uv_norm_linp(0)*uv_norm_linp(0)+uv_norm_linp(1)*uv_norm_linp(1));
                double r_2 = r*r;
                double r_4 = r_2*r_2;
                double x1 = uv_norm_linp(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm_linp(0)*uv_norm_linp(1)+cam_d(7)*(r_2+2*uv_norm_linp(0)*uv_norm_linp(0));
                double y1 = uv_norm_linp(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm_linp(1)*uv_norm_linp(1))+2*cam_d(7)*uv_norm_linp(0)*uv_norm_linp(1);
                uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                uv_dist(1) = cam_d(1)*y1 + cam_d(3);

            }

            double uv_x = feature.uvs[0][i](0);
            double uv_y = feature.uvs[0][i](1);
            Vector2d uv_feat(uv_x,uv_y);

            // Our residual
            cout<<"measure feature: "<<uv_feat.transpose()<<endl;
            cout<<"predict feature in curimage: "<<uv_dist.transpose()<<endl;
            // if(uv_dist(0)<0||uv_dist(1)<0)
            // {
            //   return false;
            // }
            res.block(2*c,0,2,1) = uv_feat - uv_dist;


            // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
            //dhd_dzn
            Eigen::Matrix<double,2,2> dz_dzn = Eigen::Matrix<double,2,2>::Zero();
            Eigen::Matrix<double,2,8> dz_dzeta = Eigen::Matrix<double,2,8>::Zero();
            if(state->_options.use_gt)
            {
              uv_norm_linp = uv_norm_true;
            }
            UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_linp, state->_cam_intrinsics_model.at(0), cam_d, dz_dzn, dz_dzeta);

            //Compute Jacobian of transforming feature in kf frame into nomalized uv frame;
            //dhp_dpfc
            Eigen::Matrix<double,2,3> dzn_dpfc = Eigen::Matrix<double,2,3>::Zero();

            // p_FinC_linp = R_ItoC_linp*R_GtoI_fej*(p_FinG_linp-p_IinG_fej) + p_IinC_linp;
            if(state->_options.use_gt)
            {
              p_FinC_linp = p_FinC_true;
            }
            
            dzn_dpfc << 1/p_FinC_linp(2),0,-p_FinC_linp(0)/(p_FinC_linp(2)*p_FinC_linp(2)),
                    0, 1/p_FinC_linp(2),-p_FinC_linp(1)/(p_FinC_linp(2)*p_FinC_linp(2));


            // Precompute some matrices
            Eigen::Matrix<double,2,3> dz_dpfc = dz_dzn*dzn_dpfc;
            Eigen::Matrix<double,2,3> dz_dpfg = dz_dpfc*dpfc_dpfg;
            Eigen::Matrix<double,2,3> dz_dpfw = dz_dpfg*dpfg_dpfw;

            // CHAINRULE: get the total feature Jacobian
            if(!state->_options.ptmeas)
                H_f.block(2*c,0,2,H_f.cols()).noalias() = dz_dpfw*dpfw_dpfk;

            H_n.block(2*c, map_hn[anchor_kf_w],2,anchor_kf_w->size()).noalias() = dz_dpfw*dpfw_dxnanchor;

            // if(!state->iter)

            // {
            // if(feature.timestamps[0][i]==state->_timestamp)
            // if(feature.timestamps[0][i]==state->delay_clones[state->delay_clones.size()-1])
            {
                H_x.block(2*c,map_hx[clone_Cur],2,clone_Cur->size()).noalias()=dz_dpfc*dpfc_dxcur;


            }
            H_x.block(2*c,map_hx[tranform],2,tranform->size()).noalias() = dz_dpfg*dpfg_dxtrans;
            // }

            c++;
        }

        //now we finish the observation that reprojecting the 3d point in anchor kf to clone_frames;

        //compute reprojecting the 3d point in anchor kf into achor kf:
        Eigen::Matrix<double,9,1> kf_db = anchor_kf->_intrinsics;
        Eigen::Matrix<double,8,1> kf_d = kf_db.block(0,0,8,1);
        bool is_fisheye = int(kf_db(8,0));
        Eigen::Matrix<double,2,1> uv_norm_kf;
        uv_norm_kf<<p_FinAnchorKF(0)/p_FinAnchorKF(2),p_FinAnchorKF(1)/p_FinAnchorKF(2);
        Eigen::Matrix<double,2,1> uv_norm_kf_linp;
        uv_norm_kf_linp<<p_FinAnchorKF_linp(0)/p_FinAnchorKF_linp(2),p_FinAnchorKF_linp(1)/p_FinAnchorKF_linp(2);

        Eigen::Matrix<double,2,1> uv_dist_kf;
        if(is_fisheye) {

            // Calculate distorted coordinates for fisheye
            double r = std::sqrt(uv_norm_kf_linp(0)*uv_norm_kf_linp(0)+uv_norm_kf_linp(1)*uv_norm_kf_linp(1));
            double theta = std::atan(r);
            double theta_d = theta+kf_d(4)*std::pow(theta,3)+kf_d(5)*std::pow(theta,5)+kf_d(6)*std::pow(theta,7)+kf_d(7)*std::pow(theta,9);

            // Handle when r is small (meaning our xy is near the camera center)
            double inv_r = (r > 1e-8)? 1.0/r : 1.0;
            double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm_kf_linp(0)*cdist;
            double y1 = uv_norm_kf_linp(1)*cdist;
            uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
            uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

        } else {

            // Calculate distorted coordinates for radial
            double r = std::sqrt(uv_norm_kf_linp(0)*uv_norm_kf_linp(0)+uv_norm_kf_linp(1)*uv_norm_kf_linp(1));
            double r_2 = r*r;
            double r_4 = r_2*r_2;
            double x1 = uv_norm_kf_linp(0)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+2*kf_d(6)*uv_norm_kf_linp(0)*uv_norm_kf_linp(1)+kf_d(7)*(r_2+2*uv_norm_kf_linp(0)*uv_norm_kf_linp(0));
            double y1 = uv_norm_kf_linp(1)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+kf_d(6)*(r_2+2*uv_norm_kf_linp(1)*uv_norm_kf_linp(1))+2*kf_d(7)*uv_norm_kf_linp(0)*uv_norm_kf_linp(1);
            uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
            uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

        }

        // Our residual

        Eigen::Matrix<double,2,1> uv_m;
        uv_m << (double)anchor_kf->point_2d_uv_map.at(ts)[kf_feature_id].x, (double)anchor_kf->point_2d_uv_map.at(ts)[kf_feature_id].y;
        cout<<"measure feature: "<<uv_m.transpose()<<endl;
        cout<<"predict feature in mapkf: "<<uv_dist_kf.transpose()<<endl;
        res.block(2*c,0,2,1) = uv_m - uv_dist_kf;

        Eigen::Matrix<double,2,2> dzkf_dzn = Eigen::Matrix<double,2,2>::Zero();
        Eigen::Matrix<double,2,8> dzkf_dzeta = Eigen::Matrix<double,2,8>::Zero();
        UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_kf_linp, is_fisheye, kf_d, dzkf_dzn, dzkf_dzeta);

        Eigen::Matrix<double,2,3> dzn_dpfk = Eigen::Matrix<double,2,3>::Zero();
        dzn_dpfk << 1/p_FinAnchorKF_linp(2),0,-p_FinAnchorKF_linp(0)/(p_FinAnchorKF_linp(2)*p_FinAnchorKF_linp(2)),
                0, 1/p_FinAnchorKF_linp(2),-p_FinAnchorKF_linp(1)/(p_FinAnchorKF_linp(2)*p_FinAnchorKF_linp(2));

        if(!state->_options.ptmeas)
        {
            H_f.block(2*c,0,2,H_f.cols()).noalias()=dzkf_dzn*dzn_dpfk;
            // Move the Jacobian and residual index forward
            c++;
        }


        //now compute reprojecting the 3d point in anchor frame to other kf
        for(int i = 1;i<keyframe_id.size();i++)
        {
           kf_id = keyframe_id[i];
           kf_feature_id = keyframe_feature_id[i];
           Keyframe* keyframe = state->get_kfdataBase()->get_keyframe(kf_id);
           assert(keyframe != nullptr);
           PoseJPL* kf_w=state->_clones_Keyframe[kf_id];
           Eigen::Matrix<double,3,3> R_KFtoW = kf_w->Rot();
           Eigen::Matrix<double,3,1> p_KFinW = kf_w->pos();

            cv::Point3f p_finkf = keyframe->point_3d_map.at(ts)[kf_feature_id];
            Vector3d p_FinKFOb(double(p_finkf.x),double(p_finkf.y),double(p_finkf.z));
            cv::Point3f p_finkf_linp = keyframe->point_3d_linp_map.at(ts)[kf_feature_id];
            Vector3d p_FinKFOb_linp(double(p_finkf_linp.x),double(p_finkf_linp.y),double(p_finkf_linp.z));
            assert(p_finkf==p_finkf_linp);

           Eigen::Vector3d p_FinKF = R_KFtoW.transpose()*(p_FinW-p_KFinW);
           Eigen::Vector3d p_FinKF_linp=R_KFtoW.transpose()*(p_FinW_linp-p_KFinW);
           cout<<"p_FinKF: "<<p_FinKF.transpose()<<" "<<p_FinKFOb.transpose()<<endl;
//           sleep(1);

           uv_norm_kf<<p_FinKF(0)/p_FinKF(2),p_FinKF(1)/p_FinKF(2);
           uv_norm_kf_linp<<p_FinKF_linp(0)/p_FinKF_linp(2),p_FinKF_linp(1)/p_FinKF_linp(2);

           if(is_fisheye) {

            // Calculate distorted coordinates for fisheye
            double r = std::sqrt(uv_norm_kf_linp(0)*uv_norm_kf_linp(0)+uv_norm_kf_linp(1)*uv_norm_kf_linp(1));
            double theta = std::atan(r);
            double theta_d = theta+kf_d(4)*std::pow(theta,3)+kf_d(5)*std::pow(theta,5)+kf_d(6)*std::pow(theta,7)+kf_d(7)*std::pow(theta,9);

            // Handle when r is small (meaning our xy is near the camera center)
            double inv_r = (r > 1e-8)? 1.0/r : 1.0;
            double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm_kf_linp(0)*cdist;
            double y1 = uv_norm_kf_linp(1)*cdist;
            uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
            uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

            }
            else {

                // Calculate distorted coordinates for radial
                double r = std::sqrt(uv_norm_kf_linp(0)*uv_norm_kf_linp(0)+uv_norm_kf_linp(1)*uv_norm_kf_linp(1));
                double r_2 = r*r;
                double r_4 = r_2*r_2;
                double x1 = uv_norm_kf_linp(0)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+2*kf_d(6)*uv_norm_kf_linp(0)*uv_norm_kf_linp(1)+kf_d(7)*(r_2+2*uv_norm_kf_linp(0)*uv_norm_kf_linp(0));
                double y1 = uv_norm_kf_linp(1)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+kf_d(6)*(r_2+2*uv_norm_kf_linp(1)*uv_norm_kf_linp(1))+2*kf_d(7)*uv_norm_kf_linp(0)*uv_norm_kf_linp(1);
                uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
                uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

            }

            uv_m << (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].x, (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].y;
            cout<<"measure feature: "<<uv_m.transpose()<<endl;
            cout<<"predict feature in mapkf: "<<uv_dist_kf.transpose()<<endl;
            res.block(2*c,0,2,1) = uv_m - uv_dist_kf;


            dzkf_dzn = Eigen::Matrix<double,2,2>::Zero();
            dzkf_dzeta = Eigen::Matrix<double,2,8>::Zero();
            UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_kf_linp, is_fisheye, kf_d, dzkf_dzn, dzkf_dzeta);

            dzn_dpfk = Eigen::Matrix<double,2,3>::Zero();
            dzn_dpfk << 1/p_FinKF_linp(2),0,-p_FinKF_linp(0)/(p_FinKF_linp(2)*p_FinKF_linp(2)),
                    0, 1/p_FinKF_linp(2),-p_FinKF_linp(1)/(p_FinKF_linp(2)*p_FinKF_linp(2));

            Matrix<double,3,3> dpfk_dpfanchor = R_KFtoW.transpose()*R_AnchorKFtoW;
            if(!state->_options.ptmeas)
                H_f.block(2*c,0,2,H_f.cols()).noalias() = dzkf_dzn*dzn_dpfk*dpfk_dpfanchor;

            Matrix<double,3,6> dpfk_dxnanchor=Matrix<double,3,6>::Zero();
            dpfk_dxnanchor.block(0,0,3,3)=R_KFtoW.transpose()*skew_x(R_AnchorKFtoW*p_FinAnchorKF_linp);
            dpfk_dxnanchor.block(0,3,3,3)=R_KFtoW.transpose();

            H_n.block(2*c, map_hn[anchor_kf_w],2,anchor_kf_w->size()).noalias() = dzkf_dzn*dzn_dpfk*dpfk_dxnanchor;

            Matrix<double,3,6> dpfk_dxn=Matrix<double,3,6>::Zero();
            dpfk_dxn.block(0,0,3,3)=-R_KFtoW.transpose()*skew_x(p_FinW_linp-p_KFinW);
            dpfk_dxn.block(0,3,3,3)=-R_KFtoW.transpose();

            H_n.block(2*c, map_hn[kf_w],2,kf_w->size()).noalias() = dzkf_dzn*dzn_dpfk*dpfk_dxn;

            c++;

        }


    }
    return true;


    // for (auto const& pair : feature.keyframe_matched_obs)
    // {
    //     for(auto const& match : pair.second )
    //     {
    //         double kf_id=match.first;
    //         size_t kf_feature_id=match.second;
    //         Keyframe* keyframe=state->get_kfdataBase()->get_keyframe(kf_id);
    //         Eigen::Matrix<double,9,1> kf_db = keyframe->_intrinsics;
    //         Eigen::Matrix<double,8,1> kf_d = kf_db.block(0,0,8,1);
    //         bool is_fisheye= int(kf_db(8,0));
    //         //get the kf position in world frame.
    //         PoseJPL* kf_w=state->_clones_Keyframe[kf_id];
    //         Eigen::Matrix<double,3,3> R_KFtoW = kf_w->Rot();
    //         Eigen::Matrix<double,3,1> p_KFinW = kf_w->pos();

    //         //get the transform between the VIO(G) to World(W)
    //         PoseJPL* tranform=state->transform_vio_to_map;
    //         Eigen::Matrix<double,3,3> R_GtoW_linp = state->transform_vio_to_map->Rot_linp();
    //         Eigen::Matrix<double,3,1> p_GinW_linp = state->transform_vio_to_map->pos_linp();
    //         Eigen::Matrix<double,3,3> R_GtoW = state->transform_vio_to_map->Rot();
    //         Eigen::Matrix<double,3,1> p_GinW = state->transform_vio_to_map->pos();

    //         double ts=state->_timestamp_approx; //ts for YQ
    //         // double ts=state->_timestamp;
    //         // ts=floor(ts*100)/100.0;  //ts for euroc /kaist
    //         ////// ts=round(ts*10000)/10000.0; //ts for YQ

    //         cv::Point3f p_fink=keyframe->point_3d_map.at(ts)[kf_feature_id];
    //         Vector3d p_FinKF(double(p_fink.x),double(p_fink.y),double(p_fink.z));
    //         cv::Point3f p_fink_linp=keyframe->point_3d_linp_map.at(ts)[kf_feature_id];
    //         Vector3d p_FinKF_linp(double(p_fink_linp.x),double(p_fink_linp.y),double(p_fink_linp.z));

    //         cv::Point2f uv_ob=keyframe->matched_point_2d_uv_map.at(ts)[kf_feature_id];
    //         double x=uv_ob.x;
    //         double y=uv_ob.y;
    //         Vector2d uv_OB(x,y);
    //         double uv_x=feature.uvs[0][0](0);
    //         double uv_y=feature.uvs[0][0](1);
    //         Vector2d uv_feat(uv_x,uv_y);
    //         assert(uv_OB==uv_feat); //check if feature link is correct

    //         //transform the feature to map reference;
    //         //hk
    //         Eigen::Vector3d p_FinW =  R_KFtoW * p_FinKF + p_KFinW;
    //         Eigen::Vector3d p_FinW_linp= R_KFtoW * p_FinKF_linp + p_KFinW;
            
    //         //dhk_dpfk
    //         Eigen::Matrix<double,3,3> dpfw_dpfk=R_KFtoW;

    //         //dhk_dxn
    //         Eigen::Matrix<double,3,6> dpfw_dxn=Eigen::Matrix<double,3,6>::Zero();
    //         dpfw_dxn.block(0,0,3,3)=skew_x(R_KFtoW*p_FinKF_linp);
    //         dpfw_dxn.block(0,3,3,3)=Eigen::Matrix3d::Identity();

    //         //transform the feature in map reference into vio reference
    //         //hw
    //         Eigen::Vector3d p_FinG = R_GtoW.transpose()*(p_FinW-p_GinW);
    //         Eigen::Vector3d p_FinG_linp = R_GtoW_linp.transpose()*(p_FinW_linp-p_GinW_linp);
    //         //dhw_dpfw
    //         Eigen::Matrix<double,3,3> dpfg_dpfw= R_GtoW_linp.transpose();
    //         //dhw_dxtrans
    //         Eigen::Matrix<double,3,6> dpfg_dxtrans=Eigen::Matrix<double,3,6>::Zero();
    //         dpfg_dxtrans.block(0,0,3,3)=-R_GtoW_linp.transpose()*skew_x(p_FinW_linp-p_GinW_linp);
    //         dpfg_dxtrans.block(0,3,3,3)=-R_GtoW_linp.transpose();

    //         //transform the feature in vio reference into current camera frame
    //         PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
    //         Matrix3d R_ItoC_linp=extrinsics->Rot_linp();
    //         Vector3d p_IinC_linp=extrinsics->pos_linp();
    //         Matrix3d R_ItoC=extrinsics->Rot();
    //         Vector3d p_IinC=extrinsics->pos();

    //         PoseJPL *clone_Cur = state->_clones_IMU.at(state->_timestamp);
    //         Matrix3d R_GtoI_linp = clone_Cur->Rot_linp();
    //         Vector3d p_IinG_linp = clone_Cur->pos_linp();
    //         Matrix3d R_GtoI = clone_Cur->Rot();
    //         Vector3d p_IinG = clone_Cur->pos();
    //         //ht
    //         Eigen::Vector3d p_FinC = R_ItoC*R_GtoI*(p_FinG-p_IinG) + p_IinC;
    //         Eigen::Vector3d p_FinC_linp = R_ItoC_linp*R_GtoI_linp*(p_FinG_linp-p_IinG_linp) + p_IinC_linp;
    //         //dht_dpfg
    //         Eigen::Matrix3d dpfc_dpfg = R_ItoC_linp*R_GtoI_linp;
    //         //dht_dxcur
    //         Eigen::Matrix<double,3,6> dpfc_dxcur=Eigen::Matrix<double,3,6>::Zero();
    //         dpfc_dxcur.block(0,0,3,3)=R_ItoC_linp*skew_x(R_GtoI_linp*(p_FinG_linp-p_IinG_linp));
    //         dpfc_dxcur.block(0,3,3,3)=-R_ItoC_linp*R_GtoI_linp;
            
    //         Eigen::Matrix<double,3,6> dpfc_dxe=Eigen::Matrix<double,3,6>::Zero();;
    //         if(state->_options.do_calib_camera_pose)
    //         {

    //            dpfc_dxe.block(0,0,3,3)=skew_x(R_ItoC_linp*R_GtoI_linp*(p_FinG_linp-p_IinG_linp));
    //            dpfc_dxe.block(0,3,3,3)=Matrix3d::Identity();
    //         }
            
    //         //transform the feature in current camera into normal frame
    //         //hp
    //         Eigen::Matrix<double,2,1> uv_norm;
    //         uv_norm<<p_FinC(0)/p_FinC(2),p_FinC(1)/p_FinC(2);
    //         Eigen::Matrix<double,2,1> uv_norm_linp;
    //         uv_norm_linp<<p_FinC_linp(0)/p_FinC_linp(2),p_FinC_linp(1)/p_FinC_linp(2);


    //         // Distort the normalized coordinates (false=radtan, true=fisheye)
    //         Eigen::Matrix<double,2,1> uv_dist;
    //         Vec* distortion = state->_cam_intrinsics.at(0);
    //         Eigen::Matrix<double,8,1> cam_d = distortion->value();
    //         // Calculate distortion uv and jacobian
    //         if(state->_cam_intrinsics_model.at(0)) {

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
    //         cout<<"measure feature: "<<uv_feat.transpose()<<endl;
    //         cout<<"predict feature in curimage: "<<uv_dist.transpose()<<endl;
    //         res.block(2*c,0,2,1) = uv_feat - uv_dist;


    //         // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
    //         //dhd_dzn
    //         Eigen::Matrix<double,2,2> dz_dzn = Eigen::Matrix<double,2,2>::Zero();
    //         Eigen::Matrix<double,2,8> dz_dzeta = Eigen::Matrix<double,2,8>::Zero();
    //         UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_linp, state->_cam_intrinsics_model.at(0), cam_d, dz_dzn, dz_dzeta);

    //         //Compute Jacobian of transforming feature in kf frame into nomalized uv frame;
    //         //dhp_dpfc
    //         Eigen::Matrix<double,2,3> dzn_dpfc = Eigen::Matrix<double,2,3>::Zero();
    //         dzn_dpfc << 1/p_FinC_linp(2),0,-p_FinC_linp(0)/(p_FinC_linp(2)*p_FinC_linp(2)),
    //                 0, 1/p_FinC_linp(2),-p_FinC_linp(1)/(p_FinC_linp(2)*p_FinC_linp(2));


    //         // Precompute some matrices
    //         Eigen::Matrix<double,2,3> dz_dpfc = dz_dzn*dzn_dpfc;
    //         Eigen::Matrix<double,2,3> dz_dpfg = dz_dpfc*dpfc_dpfg;
    //         Eigen::Matrix<double,2,3> dz_dpfw = dz_dpfg*dpfg_dpfw;

    //         // CHAINRULE: get the total feature Jacobian
    //         H_f.block(2*c,0,2,H_f.cols()).noalias() = dz_dpfw*dpfw_dpfk;

    //         H_n.block(2*c, map_hn[kf_w],2,kf_w->size()).noalias() = dz_dpfw*dpfw_dxn;

    //         H_x.block(2*c,map_hx[clone_Cur],2,clone_Cur->size()).noalias()=dz_dpfc*dpfc_dxcur;
    //         H_x.block(2*c,map_hx[tranform],2,tranform->size()).noalias() = dz_dpfg*dpfg_dxtrans;
    //          c++;
    //         // if(state->_options.do_calib_camera_pose)
    //         // {
    //         //     H_x.block(2*c,map_hx[extrinsics],2,extrinsics->size()).noalias()=dz_dpfc*dpfc_dxe;
    //         // }
    //         // if(state->_options.do_calib_camera_intrinsics)
    //         // {
    //         //     H_x.block(2*c,map_hx[distortion],2,distortion->size()).noalias()=dz_dzeta;
    //         // }

    //         // reproject the feature into kf
    //         Eigen::Matrix<double,2,1> uv_norm_kf;
    //         uv_norm_kf<<p_FinKF(0)/p_FinKF(2),p_FinKF(1)/p_FinKF(2);
    //         Eigen::Matrix<double,2,1> uv_norm_kf_linp;
    //         uv_norm_kf_linp<<p_FinKF_linp(0)/p_FinKF_linp(2),p_FinKF_linp(1)/p_FinKF_linp(2);

    //         Eigen::Matrix<double,2,1> uv_dist_kf;
    //         if(is_fisheye) {

    //             // Calculate distorted coordinates for fisheye
    //             double r = std::sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
    //             double theta = std::atan(r);
    //             double theta_d = theta+kf_d(4)*std::pow(theta,3)+kf_d(5)*std::pow(theta,5)+kf_d(6)*std::pow(theta,7)+kf_d(7)*std::pow(theta,9);

    //             // Handle when r is small (meaning our xy is near the camera center)
    //             double inv_r = (r > 1e-8)? 1.0/r : 1.0;
    //             double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

    //             // Calculate distorted coordinates for fisheye
    //             double x1 = uv_norm_kf(0)*cdist;
    //             double y1 = uv_norm_kf(1)*cdist;
    //             uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
    //             uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

    //         } else {

    //             // Calculate distorted coordinates for radial
    //             double r = std::sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
    //             double r_2 = r*r;
    //             double r_4 = r_2*r_2;
    //             double x1 = uv_norm_kf(0)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+2*kf_d(6)*uv_norm_kf(0)*uv_norm_kf(1)+kf_d(7)*(r_2+2*uv_norm_kf(0)*uv_norm_kf(0));
    //             double y1 = uv_norm_kf(1)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+kf_d(6)*(r_2+2*uv_norm_kf(1)*uv_norm_kf(1))+2*kf_d(7)*uv_norm_kf(0)*uv_norm_kf(1);
    //             uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
    //             uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

    //         }

    //         // Our residual
           
    //         Eigen::Matrix<double,2,1> uv_m;
    //         uv_m << (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].x, (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].y;
    //         cout<<"measure feature: "<<uv_m.transpose()<<endl;
    //         cout<<"predict feature in mapkf: "<<uv_dist_kf.transpose()<<endl;
    //         res.block(2*c,0,2,1) = uv_m - uv_dist_kf;

    //         Eigen::Matrix<double,2,2> dzkf_dzn = Eigen::Matrix<double,2,2>::Zero();
    //         Eigen::Matrix<double,2,8> dzkf_dzeta = Eigen::Matrix<double,2,8>::Zero();
    //         UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_kf_linp, is_fisheye, kf_d, dzkf_dzn, dzkf_dzeta);
             
    //         Eigen::Matrix<double,2,3> dzn_dpfk = Eigen::Matrix<double,2,3>::Zero();
    //         dzn_dpfk << 1/p_FinKF_linp(2),0,-p_FinKF_linp(0)/(p_FinKF_linp(2)*p_FinKF_linp(2)),
    //                 0, 1/p_FinKF_linp(2),-p_FinKF_linp(1)/(p_FinKF_linp(2)*p_FinKF_linp(2));
            
    //         H_f.block(2*c,0,2,H_f.cols()).noalias()=dzkf_dzn*dzn_dpfk;
    //        // Move the Jacobian and residual index forward
    //         c++;



    //         // assert(state->of_points.is_open());
    //         // state->of_points<<"current feat: ("<<uv_feat.transpose()<<") ("<<uv_dist.transpose()<<") kf feat("<<uv_m.transpose()<<") ("<<uv_dist_kf.transpose()<<") ";


    //         //test Jacobian
    //         // Vector2d z_fdist,z_fdist_turb;
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist,state->_cam_intrinsics_model.at(0));

    //         // Vector3d turb(0.0001, -0.003, 0.003);
    //         // //turb on p_FinKF:
    //         // compute_meas2(p_FinKF+turb, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on p_FinG:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfw*dpfw_dpfk*turb).transpose()<<endl;

    //         // //turb on R_GtoW;
    //         // Eigen::Matrix<double, 4, 1> dq;
    //         // dq << .5 * turb, 1.0;
    //         // dq = ov_core::quatnorm(dq);
    //         // Vector4d q=quat_multiply(dq, state->transform_vio_to_map->quat());
    //         // JPLQuat *q_JPL=new JPLQuat();
    //         // q_JPL->set_value(q);
    //         // Matrix3d R_GtoW_turb=q_JPL->Rot();
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW_turb,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on R_GtoW:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfg*(dpfg_dxtrans.block(0,0,3,3))*turb).transpose()<<endl;

    //         // //turb on p_GinW:
    //         // compute_meas2(p_FinKF, p_GinW+turb, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on p_GinW:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfg*(dpfg_dxtrans.block(0,3,3,3))*turb).transpose()<<endl;

    //         // //turb on R_KFtoW:
    //         // q=quat_multiply(dq, kf_w->quat());
    //         // q_JPL->set_value(q);
    //         // Matrix3d R_KFtoW_turb=q_JPL->Rot();
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW_turb, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on R_KFtoW:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfw*(dpfw_dxn.block(0,0,3,3))*turb).transpose()<<endl;

    //         // //turb on p_KFinW:
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW+turb, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on p_KFinW:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfw*(dpfw_dxn.block(0,3,3,3))*turb).transpose()<<endl;

    //         // //turb on R_GtoI:
    //         // q=quat_multiply(dq,clone_Cur->quat());
    //         // q_JPL->set_value(q);
    //         // Matrix3d R_GtoI_turb=q_JPL->Rot();
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI_turb,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on R_GtoI:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfc*(dpfc_dxcur.block(0,0,3,3))*turb).transpose()<<endl;
           
    //         // //turb on p_IinG
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG+turb, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on p_IinG:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfc*(dpfc_dxcur.block(0,3,3,3))*turb).transpose()<<endl;

            
    //         // if(state->_options.do_calib_camera_pose)
    //         // {
    //         //     //turb on R_ItoC
    //         //     q=quat_multiply(dq,extrinsics->quat());
    //         //     q_JPL->set_value(q);
    //         //     Matrix3d R_ItoC_turb=q_JPL->Rot();
    //         //     compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC_turb,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         //     cout<<"turb on R_ItoC:"<<endl;
    //         //     cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         //     cout<<"diff jacob: "<<(dz_dpfc*(dpfc_dxe.block(0,0,3,3))*turb).transpose()<<endl;

    //         //     //turb on p_IinC
    //         //     compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC+turb, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         //     cout<<"turb on R_ItoC:"<<endl;
    //         //     cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         //     cout<<"diff jacob: "<<(dz_dpfc*(dpfc_dxe.block(0,3,3,3))*turb).transpose()<<endl;

    //         // }

            
    //     }

    // }

}


bool UpdaterHelper::get_feature_jacobian_kf_transfej(State *state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                             Eigen::MatrixXd &H_n,Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<Type *> &n_order,
                                             std::vector<Type *> &x_order) {
    // Total number of measurements for this feature
    int total_meas = 0;

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    int total_hn = 0;
    std::unordered_map<Type*,size_t> map_hx;
    std::unordered_map<Type*,size_t> map_hn;
    //nuisance part
    for (auto const& pair : feature.keyframe_matched_obs) {
        for(auto const& match : pair.second)
        {
            total_meas++;
            double kf_id = match.first;
            // Keyframe* keyframe=state->get_kfdataBase()->get_keyframe(kf_id);
            PoseJPL *kf_pose=state->_clones_Keyframe[kf_id];
            if(map_hn.find(kf_pose)==map_hn.end())
            {
                map_hn.insert({kf_pose,total_hn});
                n_order.push_back(kf_pose);
                total_hn += kf_pose->size();
            }
        }

    }


    // add the current frame state
    // this part is active state part

    // Add this state if it is not added already

    // PoseJPL *clone_Cur=nullptr;
    // clone_Cur = state->_clones_IMU.at(state->_timestamp);
    // assert(clone_Cur!=nullptr);
    // if(map_hx.find(clone_Cur) == map_hx.end()) {
    //     map_hx.insert({clone_Cur,total_hx});
    //     x_order.push_back(clone_Cur);
    //     total_hx += clone_Cur->size();
    // }

    //add related clone_IMU
    for(int i=0;i<feature.timestamps[0].size();i++)
    {
        total_meas++;
        // cout<<"feature.timestamps[0]["<<i<<"] :"<<feature.timestamps[0][i]<<endl;
        // if(feature.timestamps[0][i]==state->delay_clones[state->delay_clones.size()-1])
        // if(i==feature.timestamps[0].size()-1)
        {
            // cout<<"have current state: "<<i<<endl;
            // sleep(1);
            PoseJPL* clone=nullptr;
            clone=state->_clones_IMU.at(feature.timestamps[0][i]);
            assert(clone!=nullptr);
            if(map_hx.find(clone) == map_hx.end()) {
                map_hx.insert({clone,total_hx});
                x_order.push_back(clone);
                total_hx += clone->size();
            }

        }

    }
    assert(total_meas>=2);
    // // Also add its calibration if we are doing calibration
    // if(state->_options.do_calib_camera_pose) {
    //     // Add this anchor if it is not added already
    //     PoseJPL *clone_calib = state->_calib_IMUtoCAM.at(0); //we use the left cam;
    //     if(map_hx.find(clone_calib) == map_hx.end()) {
    //         map_hx.insert({clone_calib,total_hx});
    //         x_order.push_back(clone_calib);
    //         total_hx += clone_calib->size();
    //     }
    // }

    // if(state->_options.do_calib_camera_intrinsics)
    // {
    //     Vec *distortion = state->_cam_intrinsics.at(0);
    //     map_hx.insert({distortion,total_hx});
    //     x_order.push_back(distortion);
    //     total_hx += distortion->size();
    // }



    //we also need to add transformation between map and vio
    if(state->set_transform) //normally, if we are in this function, the state->set_transform should be already set true
    {
        PoseJPL *transform = state->transform_vio_to_map;
        if(map_hx.find(transform)==map_hx.end())
        {
            map_hx.insert({transform,total_hx});
            x_order.push_back(transform);
            total_hx += transform->size();
        }
    }

    //=========================================================================
    //=========================================================================

    // // Calculate the position of this feature in the global frame
    // // If anchored, then we need to calculate the position of the feature in the global(vio system reference)
    // Eigen::Vector3d p_FinG = feature.p_FinG;
    // if(LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    //     // Assert that we have an anchor pose for this feature
    //     assert(feature.anchor_cam_id!=-1);
    //     // Get calibration for our anchor camera
    //     Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
    //     Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
    //     // Anchor pose orientation and position
    //     Eigen::Matrix<double,3,3> R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
    //     Eigen::Matrix<double,3,1> p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
    //     // Feature in the global frame
    //     p_FinG = R_GtoI.transpose() * R_ItoC.transpose()*(feature.p_FinA - p_IinC) + p_IinG;
    // }

    // // Calculate the position of this feature in the global frame FEJ
    // // If anchored, then we can use the "best" p_FinG since the value of p_FinA does not matter
    // Eigen::Vector3d p_FinG_fej = feature.p_FinG_fej;
    // if(LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    //     p_FinG_fej = p_FinG;
    // }

    //=========================================================================
    //=========================================================================

    // Allocate our residual and Jacobians
    int c = 0;
    // int jacobsize = (feature.feat_representation!=LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
    int jacobsize = 3; //the observed feature is represented by 3d with respect to the loop_keyframe
    //for each feature, it has two reprojection error, in current cam frame and map_kf frame
    res = Eigen::VectorXd::Zero(2*total_meas);
    H_f = Eigen::MatrixXd::Zero(2*total_meas,jacobsize);
    H_x = Eigen::MatrixXd::Zero(2*total_meas,total_hx);
    H_n = Eigen::MatrixXd::Zero(2*total_meas,total_hn);

    vector<double> keyframe_id;
    vector<size_t> keyframe_feature_id;
    for(auto const& pair: feature.keyframe_matched_obs)
    {
        keyframe_id.clear();
        keyframe_feature_id.clear();
        for(auto const& match: pair.second)
        {
            double kf_id = match.first;
            size_t kf_feature_id = match.second;
            keyframe_id.push_back(kf_id);
            keyframe_feature_id.push_back(kf_feature_id);
        }
        //as each keyframe has a 3d point represented in itself, we choose the first kf as anchor kf,
        //and reproject its 3d point into kfs and current frame
        double kf_id = keyframe_id[0];
        Keyframe* anchor_kf = state->get_kfdataBase()->get_keyframe(kf_id);
        assert(anchor_kf != nullptr);
        size_t kf_feature_id = keyframe_feature_id[0];

        //now we reproject anchor_kf's 3d point into current frame
        //get the kf position in world frame.
        PoseJPL* anchor_kf_w = state->_clones_Keyframe[kf_id];
        Eigen::Matrix<double,3,3> R_AnchorKFtoW = anchor_kf_w->Rot();
        Eigen::Matrix<double,3,1> p_AnchorKFinW = anchor_kf_w->pos();

        //get the transform between the VIO(G) to World(W)
        PoseJPL* tranform = state->transform_vio_to_map;
        Eigen::Matrix<double,3,3> R_GtoW_linp = state->transform_vio_to_map->Rot_linp();
        Eigen::Matrix<double,3,1> p_GinW_linp = state->transform_vio_to_map->pos_linp();
        Eigen::Matrix<double,3,3> R_GtoW_fej = state->transform_vio_to_map->Rot_fej();
        Eigen::Matrix<double,3,1> p_GinW_fej = state->transform_vio_to_map->pos_fej();
        // if(state->_options.trans_fej)
        // {
        //     R_GtoW_linp=R_GtoW_fej;
        //     p_GinW_linp=p_GinW_fej;
        // }
        Eigen::Matrix<double,3,3> R_GtoW = state->transform_vio_to_map->Rot();
        Eigen::Matrix<double,3,1> p_GinW = state->transform_vio_to_map->pos();

        if(!state->iter)
        {
          assert(p_GinW_linp==p_GinW);
        }

        double ts = pair.first; //ts

        cv::Point3f p_financhor = anchor_kf->point_3d_map.at(ts)[kf_feature_id];
        Vector3d p_FinAnchorKF(double(p_financhor.x),double(p_financhor.y),double(p_financhor.z));
        cv::Point3f p_financhor_linp = anchor_kf->point_3d_linp_map.at(ts)[kf_feature_id];
        Vector3d p_FinAnchorKF_linp(double(p_financhor_linp.x),double(p_financhor_linp.y),double(p_financhor_linp.z));
        assert(p_financhor==p_financhor_linp);
        // cv::Point2f uv_ob = anchor_kf->matched_point_2d_uv_map.at(ts)[kf_feature_id];
        // double x = uv_ob.x;
        // double y = uv_ob.y;
        // Vector2d uv_OB(x,y);
        // double uv_x = feature.uvs[0][0](0);
        // double uv_y = feature.uvs[0][0](1);
        // Vector2d uv_feat(uv_x,uv_y);
        // assert(uv_OB == uv_feat); //check if feature link is correct

        //transform the feature to map reference;
        //hk
        Eigen::Vector3d p_FinW =  R_AnchorKFtoW * p_FinAnchorKF + p_AnchorKFinW;
        Eigen::Vector3d p_FinW_linp = R_AnchorKFtoW * p_FinAnchorKF_linp + p_AnchorKFinW;
        if(!state->iter)
        {
          assert(p_FinW==p_FinW_linp);
        }

        //dhk_dpfk
        Eigen::Matrix<double,3,3> dpfw_dpfk=R_AnchorKFtoW;

        //dhk_dxn
        Eigen::Matrix<double,3,6> dpfw_dxnanchor=Eigen::Matrix<double,3,6>::Zero();
        dpfw_dxnanchor.block(0,0,3,3) = skew_x(R_AnchorKFtoW*p_FinAnchorKF_linp);
        dpfw_dxnanchor.block(0,3,3,3) = Eigen::Matrix3d::Identity();

        //transform the feature in map reference into vio reference
        //hw
        Eigen::Vector3d p_FinG = R_GtoW.transpose()*(p_FinW-p_GinW);
        Eigen::Vector3d p_FinG_fej = R_GtoW_fej.transpose()*(p_FinW_linp-p_GinW_fej);
        Eigen::Vector3d p_FinG_linp = R_GtoW_linp.transpose()*(p_FinW_linp-p_GinW_linp);
        //dhw_dpfw
        Eigen::Matrix<double,3,3> dpfg_dpfw = R_GtoW_fej.transpose();
        // Eigen::Matrix<double,3,3> dpfg_dpfw = R_GtoW_linp.transpose();
        //dhw_dxtrans
        Eigen::Matrix<double,3,6> dpfg_dxtrans = Eigen::Matrix<double,3,6>::Zero();
        dpfg_dxtrans.block(0,0,3,3) =-R_GtoW_fej.transpose()*skew_x(p_FinW_linp-p_GinW_fej);
        dpfg_dxtrans.block(0,3,3,3) = -R_GtoW_fej.transpose();
        // dpfg_dxtrans.block(0,0,3,3) =-R_GtoW_linp.transpose()*skew_x(p_FinW_linp-p_GinW_linp);
        // dpfg_dxtrans.block(0,3,3,3) = -R_GtoW_linp.transpose();

        //transform the feature in vio reference into current camera frame
        PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
        Matrix3d R_ItoC_linp = extrinsics->Rot_linp();
        Vector3d p_IinC_linp = extrinsics->pos_linp();
        Matrix3d R_ItoC = extrinsics->Rot();
        Vector3d p_IinC = extrinsics->pos();
        assert(p_IinC_linp==p_IinC);
        for(int i=0;i<feature.timestamps[0].size();i++)
        {
            PoseJPL *clone_Cur = state->_clones_IMU.at(feature.timestamps[0][i]);
            assert(feature.timestamps[0][i]==state->_timestamp);
            assert(clone_Cur!=nullptr);
            Matrix3d R_GtoI_linp = clone_Cur->Rot_linp();
            Vector3d p_IinG_linp = clone_Cur->pos_linp();
            Matrix3d R_GtoI_fej = clone_Cur->Rot_fej();
            Vector3d p_IinG_fej = clone_Cur->pos_fej();
            Matrix3d R_GtoI = clone_Cur->Rot();
            Vector3d p_IinG = clone_Cur->pos();
            // assert(R_GtoI_linp==R_GtoI);
            //ht
            Eigen::Vector3d p_FinC = R_ItoC*R_GtoI*(p_FinG-p_IinG) + p_IinC;
            Eigen::Vector3d p_FinC_linp = R_ItoC_linp*R_GtoI_linp*(p_FinG_linp-p_IinG_linp) + p_IinC_linp;
            Eigen::Vector3d p_FinC_fej = R_ItoC_linp*R_GtoI_fej*(p_FinG_fej-p_IinG_fej)+p_IinC_linp;
            //dht_dpfg   
            Eigen::Matrix3d dpfc_dpfg = R_ItoC_linp*R_GtoI_fej;
            // Eigen::Matrix3d dpfc_dpfg = R_ItoC_linp*R_GtoI_linp;
            //dht_dxcur
            Eigen::Matrix<double,3,6> dpfc_dxcur = Eigen::Matrix<double,3,6>::Zero();
            dpfc_dxcur.block(0,0,3,3) = R_ItoC_linp*skew_x(R_GtoI_fej*(p_FinG_fej-p_IinG_fej));
            dpfc_dxcur.block(0,3,3,3) = -R_ItoC_linp*R_GtoI_fej;
            // dpfc_dxcur.block(0,0,3,3) = R_ItoC_linp*skew_x(R_GtoI_linp*(p_FinG_linp-p_IinG_linp));
            // dpfc_dxcur.block(0,3,3,3) = -R_ItoC_linp*R_GtoI_linp;

            Eigen::Matrix<double,3,6> dpfc_dxe=Eigen::Matrix<double,3,6>::Zero();;
            // if(state->_options.do_calib_camera_pose)
            // {

            //     dpfc_dxe.block(0,0,3,3)=skew_x(R_ItoC_linp*R_GtoI_linp*(p_FinG_linp-p_IinG_linp));
            //     dpfc_dxe.block(0,3,3,3)=Matrix3d::Identity();
            // }

            //transform the feature in current camera into normal frame
            //hp
            Eigen::Matrix<double,2,1> uv_norm;
            uv_norm<<p_FinC(0)/p_FinC(2),p_FinC(1)/p_FinC(2);
            Eigen::Matrix<double,2,1> uv_norm_linp;
            uv_norm_linp<<p_FinC_linp(0)/p_FinC_linp(2),p_FinC_linp(1)/p_FinC_linp(2);
            Eigen::Matrix<double,2,1> uv_norm_fej;
            uv_norm_fej<<p_FinC_fej(0)/p_FinC_fej(2),p_FinC_fej(1)/p_FinC_fej(2);
            // uv_norm_linp=uv_norm;

            // Distort the normalized coordinates (false=radtan, true=fisheye)
            Eigen::Matrix<double,2,1> uv_dist;
            Vec* distortion = state->_cam_intrinsics.at(0); 
            Eigen::Matrix<double,8,1> cam_d = distortion->linp();
            // Calculate distortion uv and jacobian
            if(state->_cam_intrinsics_model.at(0)) {

                // Calculate distorted coordinates for fisheye
                double r = std::sqrt(uv_norm_linp(0)*uv_norm_linp(0)+uv_norm_linp(1)*uv_norm_linp(1));
                double theta = std::atan(r);
                double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

                // Handle when r is small (meaning our xy is near the camera center)
                double inv_r = (r > 1e-8)? 1.0/r : 1.0;
                double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

                // Calculate distorted coordinates for fisheye
                double x1 = uv_norm_linp(0)*cdist;
                double y1 = uv_norm_linp(1)*cdist;
                uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                uv_dist(1) = cam_d(1)*y1 + cam_d(3);

            } else {

                // Calculate distorted coordinates for radial
                double r = std::sqrt(uv_norm_linp(0)*uv_norm_linp(0)+uv_norm_linp(1)*uv_norm_linp(1));
                double r_2 = r*r;
                double r_4 = r_2*r_2;
                double x1 = uv_norm_linp(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm_linp(0)*uv_norm_linp(1)+cam_d(7)*(r_2+2*uv_norm_linp(0)*uv_norm_linp(0));
                double y1 = uv_norm_linp(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm_linp(1)*uv_norm_linp(1))+2*cam_d(7)*uv_norm_linp(0)*uv_norm_linp(1);
                uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                uv_dist(1) = cam_d(1)*y1 + cam_d(3);

            }

            double uv_x = feature.uvs[0][i](0);
            double uv_y = feature.uvs[0][i](1);
            Vector2d uv_feat(uv_x,uv_y);

            // Our residual
            cout<<"measure feature: "<<uv_feat.transpose()<<endl;
            cout<<"predict feature in curimage: "<<uv_dist.transpose()<<endl;
            // if(uv_dist(0)<0||uv_dist(1)<0)
            // {
            //   return false;
            // }
            res.block(2*c,0,2,1) = uv_feat - uv_dist;


            // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
            //dhd_dzn
            Eigen::Matrix<double,2,2> dz_dzn = Eigen::Matrix<double,2,2>::Zero();
            Eigen::Matrix<double,2,8> dz_dzeta = Eigen::Matrix<double,2,8>::Zero();
            UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_linp, state->_cam_intrinsics_model.at(0), cam_d, dz_dzn, dz_dzeta);

            //Compute Jacobian of transforming feature in kf frame into nomalized uv frame;
            //dhp_dpfc
            Eigen::Matrix<double,2,3> dzn_dpfc = Eigen::Matrix<double,2,3>::Zero();
            p_FinC_fej=p_FinC_linp;
            dzn_dpfc << 1/p_FinC_fej(2),0,-p_FinC_fej(0)/(p_FinC_fej(2)*p_FinC_fej(2)),
                    0, 1/p_FinC_fej(2),-p_FinC_fej(1)/(p_FinC_fej(2)*p_FinC_fej(2));


            // Precompute some matrices
            Eigen::Matrix<double,2,3> dz_dpfc = dz_dzn*dzn_dpfc;
            Eigen::Matrix<double,2,3> dz_dpfg = dz_dpfc*dpfc_dpfg;
            Eigen::Matrix<double,2,3> dz_dpfw = dz_dpfg*dpfg_dpfw;

            // CHAINRULE: get the total feature Jacobian
            H_f.block(2*c,0,2,H_f.cols()).noalias() = dz_dpfw*dpfw_dpfk;

            H_n.block(2*c, map_hn[anchor_kf_w],2,anchor_kf_w->size()).noalias() = dz_dpfw*dpfw_dxnanchor;

            // if(!state->iter)

            // {
            // if(feature.timestamps[0][i]==state->_timestamp)
            // if(feature.timestamps[0][i]==state->delay_clones[state->delay_clones.size()-1])
            {
                H_x.block(2*c,map_hx[clone_Cur],2,clone_Cur->size()).noalias()=dz_dpfc*dpfc_dxcur;

            }
            H_x.block(2*c,map_hx[tranform],2,tranform->size()).noalias() = dz_dpfg*dpfg_dxtrans;
            // }

            c++;
        }

        //now we finish the observation that reprojecting the 3d point in anchor kf to clone_frames;

        //compute reprojecting the 3d point in anchor kf into achor kf:
        Eigen::Matrix<double,9,1> kf_db = anchor_kf->_intrinsics;
        Eigen::Matrix<double,8,1> kf_d = kf_db.block(0,0,8,1);
        bool is_fisheye = int(kf_db(8,0));
        Eigen::Matrix<double,2,1> uv_norm_kf;
        uv_norm_kf<<p_FinAnchorKF(0)/p_FinAnchorKF(2),p_FinAnchorKF(1)/p_FinAnchorKF(2);
        Eigen::Matrix<double,2,1> uv_norm_kf_linp;
        uv_norm_kf_linp<<p_FinAnchorKF_linp(0)/p_FinAnchorKF_linp(2),p_FinAnchorKF_linp(1)/p_FinAnchorKF_linp(2);
        
        // uv_norm_kf_linp=uv_norm_kf;
        Eigen::Matrix<double,2,1> uv_dist_kf;
        if(is_fisheye) {

            // Calculate distorted coordinates for fisheye
            double r = std::sqrt(uv_norm_kf_linp(0)*uv_norm_kf_linp(0)+uv_norm_kf_linp(1)*uv_norm_kf_linp(1));
            double theta = std::atan(r);
            double theta_d = theta+kf_d(4)*std::pow(theta,3)+kf_d(5)*std::pow(theta,5)+kf_d(6)*std::pow(theta,7)+kf_d(7)*std::pow(theta,9);

            // Handle when r is small (meaning our xy is near the camera center)
            double inv_r = (r > 1e-8)? 1.0/r : 1.0;
            double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm_kf_linp(0)*cdist;
            double y1 = uv_norm_kf_linp(1)*cdist;
            uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
            uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

        } else {

            // Calculate distorted coordinates for radial
            double r = std::sqrt(uv_norm_kf_linp(0)*uv_norm_kf_linp(0)+uv_norm_kf_linp(1)*uv_norm_kf_linp(1));
            double r_2 = r*r;
            double r_4 = r_2*r_2;
            double x1 = uv_norm_kf_linp(0)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+2*kf_d(6)*uv_norm_kf_linp(0)*uv_norm_kf_linp(1)+kf_d(7)*(r_2+2*uv_norm_kf_linp(0)*uv_norm_kf_linp(0));
            double y1 = uv_norm_kf_linp(1)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+kf_d(6)*(r_2+2*uv_norm_kf_linp(1)*uv_norm_kf_linp(1))+2*kf_d(7)*uv_norm_kf_linp(0)*uv_norm_kf_linp(1);
            uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
            uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

        }

        // Our residual

        Eigen::Matrix<double,2,1> uv_m;
        uv_m << (double)anchor_kf->point_2d_uv_map.at(ts)[kf_feature_id].x, (double)anchor_kf->point_2d_uv_map.at(ts)[kf_feature_id].y;
        cout<<"uv_norm_kf_linp: "<<uv_norm_kf_linp.transpose()<<endl;
        cout<<"p_FinAnchorKF: "<<p_FinAnchorKF.transpose()<<endl;
        cout<<"intrinsice: "<<kf_d.transpose()<<endl;
        cout<<"measure feature: "<<uv_m.transpose()<<endl;
        cout<<"predict feature in mapkf: "<<uv_dist_kf.transpose()<<endl;
        res.block(2*c,0,2,1) = uv_m - uv_dist_kf;

        Eigen::Matrix<double,2,2> dzkf_dzn = Eigen::Matrix<double,2,2>::Zero();
        Eigen::Matrix<double,2,8> dzkf_dzeta = Eigen::Matrix<double,2,8>::Zero();
        UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_kf_linp, is_fisheye, kf_d, dzkf_dzn, dzkf_dzeta);

        Eigen::Matrix<double,2,3> dzn_dpfk = Eigen::Matrix<double,2,3>::Zero();
        dzn_dpfk << 1/p_FinAnchorKF_linp(2),0,-p_FinAnchorKF_linp(0)/(p_FinAnchorKF_linp(2)*p_FinAnchorKF_linp(2)),
                0, 1/p_FinAnchorKF_linp(2),-p_FinAnchorKF_linp(1)/(p_FinAnchorKF_linp(2)*p_FinAnchorKF_linp(2));

        H_f.block(2*c,0,2,H_f.cols()).noalias()=dzkf_dzn*dzn_dpfk;
        // Move the Jacobian and residual index forward
        c++;

        //now compute reprojecting the 3d point in anchor frame to other kf
        for(int i = 1;i<keyframe_id.size();i++)
        {
            kf_id = keyframe_id[i];
            kf_feature_id = keyframe_feature_id[i];
            Keyframe* keyframe = state->get_kfdataBase()->get_keyframe(kf_id);
            assert(keyframe != nullptr);
            PoseJPL* kf_w=state->_clones_Keyframe[kf_id];
            Eigen::Matrix<double,3,3> R_KFtoW = kf_w->Rot();
            Eigen::Matrix<double,3,1> p_KFinW = kf_w->pos();

            cv::Point3f p_finkf = keyframe->point_3d_map.at(ts)[kf_feature_id];
            Vector3d p_FinKFOb(double(p_finkf.x),double(p_finkf.y),double(p_finkf.z));
            cv::Point3f p_finkf_linp = keyframe->point_3d_linp_map.at(ts)[kf_feature_id];
            Vector3d p_FinKFOb_linp(double(p_finkf_linp.x),double(p_finkf_linp.y),double(p_finkf_linp.z));
            assert(p_finkf==p_finkf_linp);

            Eigen::Vector3d p_FinKF = R_KFtoW.transpose()*(p_FinW-p_KFinW);
            Eigen::Vector3d p_FinKF_linp=R_KFtoW.transpose()*(p_FinW_linp-p_KFinW);
            cout<<"p_FinKF: "<<p_FinKF.transpose()<<" "<<p_FinKFOb.transpose()<<endl;
//           sleep(1);

            uv_norm_kf<<p_FinKF(0)/p_FinKF(2),p_FinKF(1)/p_FinKF(2);
            uv_norm_kf_linp<<p_FinKF_linp(0)/p_FinKF_linp(2),p_FinKF_linp(1)/p_FinKF_linp(2);
            // uv_norm_kf_linp=uv_norm_kf;
            if(is_fisheye) {

                // Calculate distorted coordinates for fisheye
                double r = std::sqrt(uv_norm_kf_linp(0)*uv_norm_kf_linp(0)+uv_norm_kf_linp(1)*uv_norm_kf_linp(1));
                double theta = std::atan(r);
                double theta_d = theta+kf_d(4)*std::pow(theta,3)+kf_d(5)*std::pow(theta,5)+kf_d(6)*std::pow(theta,7)+kf_d(7)*std::pow(theta,9);

                // Handle when r is small (meaning our xy is near the camera center)
                double inv_r = (r > 1e-8)? 1.0/r : 1.0;
                double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

                // Calculate distorted coordinates for fisheye
                double x1 = uv_norm_kf_linp(0)*cdist;
                double y1 = uv_norm_kf_linp(1)*cdist;
                uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
                uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

            }
            else {

                // Calculate distorted coordinates for radial
                double r = std::sqrt(uv_norm_kf_linp(0)*uv_norm_kf_linp(0)+uv_norm_kf_linp(1)*uv_norm_kf_linp(1));
                double r_2 = r*r;
                double r_4 = r_2*r_2;
                double x1 = uv_norm_kf_linp(0)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+2*kf_d(6)*uv_norm_kf_linp(0)*uv_norm_kf_linp(1)+kf_d(7)*(r_2+2*uv_norm_kf_linp(0)*uv_norm_kf_linp(0));
                double y1 = uv_norm_kf_linp(1)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+kf_d(6)*(r_2+2*uv_norm_kf_linp(1)*uv_norm_kf_linp(1))+2*kf_d(7)*uv_norm_kf_linp(0)*uv_norm_kf_linp(1);
                uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
                uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

            }

            uv_m << (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].x, (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].y;
            cout<<"measure feature: "<<uv_m.transpose()<<endl;
            cout<<"predict feature in mapkf: "<<uv_dist_kf.transpose()<<endl;
            res.block(2*c,0,2,1) = uv_m - uv_dist_kf;


            dzkf_dzn = Eigen::Matrix<double,2,2>::Zero();
            dzkf_dzeta = Eigen::Matrix<double,2,8>::Zero();
            UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_kf_linp, is_fisheye, kf_d, dzkf_dzn, dzkf_dzeta);

            dzn_dpfk = Eigen::Matrix<double,2,3>::Zero();
            dzn_dpfk << 1/p_FinKF_linp(2),0,-p_FinKF_linp(0)/(p_FinKF_linp(2)*p_FinKF_linp(2)),
                    0, 1/p_FinKF_linp(2),-p_FinKF_linp(1)/(p_FinKF_linp(2)*p_FinKF_linp(2));

            Matrix<double,3,3> dpfk_dpfanchor = R_KFtoW.transpose()*R_AnchorKFtoW;

            H_f.block(2*c,0,2,H_f.cols()).noalias() = dzkf_dzn*dzn_dpfk*dpfk_dpfanchor;

            Matrix<double,3,6> dpfk_dxnanchor=Matrix<double,3,6>::Zero();
            dpfk_dxnanchor.block(0,0,3,3)=R_KFtoW.transpose()*skew_x(R_AnchorKFtoW*p_FinAnchorKF_linp);
            dpfk_dxnanchor.block(0,3,3,3)=R_KFtoW.transpose();

            H_n.block(2*c, map_hn[anchor_kf_w],2,anchor_kf_w->size()).noalias() = dzkf_dzn*dzn_dpfk*dpfk_dxnanchor;

            Matrix<double,3,6> dpfk_dxn=Matrix<double,3,6>::Zero();
            dpfk_dxn.block(0,0,3,3)=-R_KFtoW.transpose()*skew_x(p_FinW_linp-p_KFinW);
            dpfk_dxn.block(0,3,3,3)=-R_KFtoW.transpose();

            H_n.block(2*c, map_hn[kf_w],2,kf_w->size()).noalias() = dzkf_dzn*dzn_dpfk*dpfk_dxn;

            c++;

        }


    }
    return true;


    // for (auto const& pair : feature.keyframe_matched_obs)
    // {
    //     for(auto const& match : pair.second )
    //     {
    //         double kf_id=match.first;
    //         size_t kf_feature_id=match.second;
    //         Keyframe* keyframe=state->get_kfdataBase()->get_keyframe(kf_id);
    //         Eigen::Matrix<double,9,1> kf_db = keyframe->_intrinsics;
    //         Eigen::Matrix<double,8,1> kf_d = kf_db.block(0,0,8,1);
    //         bool is_fisheye= int(kf_db(8,0));
    //         //get the kf position in world frame.
    //         PoseJPL* kf_w=state->_clones_Keyframe[kf_id];
    //         Eigen::Matrix<double,3,3> R_KFtoW = kf_w->Rot();
    //         Eigen::Matrix<double,3,1> p_KFinW = kf_w->pos();

    //         //get the transform between the VIO(G) to World(W)
    //         PoseJPL* tranform=state->transform_vio_to_map;
    //         Eigen::Matrix<double,3,3> R_GtoW_linp = state->transform_vio_to_map->Rot_linp();
    //         Eigen::Matrix<double,3,1> p_GinW_linp = state->transform_vio_to_map->pos_linp();
    //         Eigen::Matrix<double,3,3> R_GtoW = state->transform_vio_to_map->Rot();
    //         Eigen::Matrix<double,3,1> p_GinW = state->transform_vio_to_map->pos();

    //         double ts=state->_timestamp_approx; //ts for YQ
    //         // double ts=state->_timestamp;
    //         // ts=floor(ts*100)/100.0;  //ts for euroc /kaist
    //         ////// ts=round(ts*10000)/10000.0; //ts for YQ

    //         cv::Point3f p_fink=keyframe->point_3d_map.at(ts)[kf_feature_id];
    //         Vector3d p_FinKF(double(p_fink.x),double(p_fink.y),double(p_fink.z));
    //         cv::Point3f p_fink_linp=keyframe->point_3d_linp_map.at(ts)[kf_feature_id];
    //         Vector3d p_FinKF_linp(double(p_fink_linp.x),double(p_fink_linp.y),double(p_fink_linp.z));

    //         cv::Point2f uv_ob=keyframe->matched_point_2d_uv_map.at(ts)[kf_feature_id];
    //         double x=uv_ob.x;
    //         double y=uv_ob.y;
    //         Vector2d uv_OB(x,y);
    //         double uv_x=feature.uvs[0][0](0);
    //         double uv_y=feature.uvs[0][0](1);
    //         Vector2d uv_feat(uv_x,uv_y);
    //         assert(uv_OB==uv_feat); //check if feature link is correct

    //         //transform the feature to map reference;
    //         //hk
    //         Eigen::Vector3d p_FinW =  R_KFtoW * p_FinKF + p_KFinW;
    //         Eigen::Vector3d p_FinW_linp= R_KFtoW * p_FinKF_linp + p_KFinW;

    //         //dhk_dpfk
    //         Eigen::Matrix<double,3,3> dpfw_dpfk=R_KFtoW;

    //         //dhk_dxn
    //         Eigen::Matrix<double,3,6> dpfw_dxn=Eigen::Matrix<double,3,6>::Zero();
    //         dpfw_dxn.block(0,0,3,3)=skew_x(R_KFtoW*p_FinKF_linp);
    //         dpfw_dxn.block(0,3,3,3)=Eigen::Matrix3d::Identity();

    //         //transform the feature in map reference into vio reference
    //         //hw
    //         Eigen::Vector3d p_FinG = R_GtoW.transpose()*(p_FinW-p_GinW);
    //         Eigen::Vector3d p_FinG_linp = R_GtoW_linp.transpose()*(p_FinW_linp-p_GinW_linp);
    //         //dhw_dpfw
    //         Eigen::Matrix<double,3,3> dpfg_dpfw= R_GtoW_linp.transpose();
    //         //dhw_dxtrans
    //         Eigen::Matrix<double,3,6> dpfg_dxtrans=Eigen::Matrix<double,3,6>::Zero();
    //         dpfg_dxtrans.block(0,0,3,3)=-R_GtoW_linp.transpose()*skew_x(p_FinW_linp-p_GinW_linp);
    //         dpfg_dxtrans.block(0,3,3,3)=-R_GtoW_linp.transpose();

    //         //transform the feature in vio reference into current camera frame
    //         PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
    //         Matrix3d R_ItoC_linp=extrinsics->Rot_linp();
    //         Vector3d p_IinC_linp=extrinsics->pos_linp();
    //         Matrix3d R_ItoC=extrinsics->Rot();
    //         Vector3d p_IinC=extrinsics->pos();

    //         PoseJPL *clone_Cur = state->_clones_IMU.at(state->_timestamp);
    //         Matrix3d R_GtoI_linp = clone_Cur->Rot_linp();
    //         Vector3d p_IinG_linp = clone_Cur->pos_linp();
    //         Matrix3d R_GtoI = clone_Cur->Rot();
    //         Vector3d p_IinG = clone_Cur->pos();
    //         //ht
    //         Eigen::Vector3d p_FinC = R_ItoC*R_GtoI*(p_FinG-p_IinG) + p_IinC;
    //         Eigen::Vector3d p_FinC_linp = R_ItoC_linp*R_GtoI_linp*(p_FinG_linp-p_IinG_linp) + p_IinC_linp;
    //         //dht_dpfg
    //         Eigen::Matrix3d dpfc_dpfg = R_ItoC_linp*R_GtoI_linp;
    //         //dht_dxcur
    //         Eigen::Matrix<double,3,6> dpfc_dxcur=Eigen::Matrix<double,3,6>::Zero();
    //         dpfc_dxcur.block(0,0,3,3)=R_ItoC_linp*skew_x(R_GtoI_linp*(p_FinG_linp-p_IinG_linp));
    //         dpfc_dxcur.block(0,3,3,3)=-R_ItoC_linp*R_GtoI_linp;

    //         Eigen::Matrix<double,3,6> dpfc_dxe=Eigen::Matrix<double,3,6>::Zero();;
    //         if(state->_options.do_calib_camera_pose)
    //         {

    //            dpfc_dxe.block(0,0,3,3)=skew_x(R_ItoC_linp*R_GtoI_linp*(p_FinG_linp-p_IinG_linp));
    //            dpfc_dxe.block(0,3,3,3)=Matrix3d::Identity();
    //         }

    //         //transform the feature in current camera into normal frame
    //         //hp
    //         Eigen::Matrix<double,2,1> uv_norm;
    //         uv_norm<<p_FinC(0)/p_FinC(2),p_FinC(1)/p_FinC(2);
    //         Eigen::Matrix<double,2,1> uv_norm_linp;
    //         uv_norm_linp<<p_FinC_linp(0)/p_FinC_linp(2),p_FinC_linp(1)/p_FinC_linp(2);


    //         // Distort the normalized coordinates (false=radtan, true=fisheye)
    //         Eigen::Matrix<double,2,1> uv_dist;
    //         Vec* distortion = state->_cam_intrinsics.at(0);
    //         Eigen::Matrix<double,8,1> cam_d = distortion->value();
    //         // Calculate distortion uv and jacobian
    //         if(state->_cam_intrinsics_model.at(0)) {

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
    //         cout<<"measure feature: "<<uv_feat.transpose()<<endl;
    //         cout<<"predict feature in curimage: "<<uv_dist.transpose()<<endl;
    //         res.block(2*c,0,2,1) = uv_feat - uv_dist;


    //         // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
    //         //dhd_dzn
    //         Eigen::Matrix<double,2,2> dz_dzn = Eigen::Matrix<double,2,2>::Zero();
    //         Eigen::Matrix<double,2,8> dz_dzeta = Eigen::Matrix<double,2,8>::Zero();
    //         UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_linp, state->_cam_intrinsics_model.at(0), cam_d, dz_dzn, dz_dzeta);

    //         //Compute Jacobian of transforming feature in kf frame into nomalized uv frame;
    //         //dhp_dpfc
    //         Eigen::Matrix<double,2,3> dzn_dpfc = Eigen::Matrix<double,2,3>::Zero();
    //         dzn_dpfc << 1/p_FinC_linp(2),0,-p_FinC_linp(0)/(p_FinC_linp(2)*p_FinC_linp(2)),
    //                 0, 1/p_FinC_linp(2),-p_FinC_linp(1)/(p_FinC_linp(2)*p_FinC_linp(2));


    //         // Precompute some matrices
    //         Eigen::Matrix<double,2,3> dz_dpfc = dz_dzn*dzn_dpfc;
    //         Eigen::Matrix<double,2,3> dz_dpfg = dz_dpfc*dpfc_dpfg;
    //         Eigen::Matrix<double,2,3> dz_dpfw = dz_dpfg*dpfg_dpfw;

    //         // CHAINRULE: get the total feature Jacobian
    //         H_f.block(2*c,0,2,H_f.cols()).noalias() = dz_dpfw*dpfw_dpfk;

    //         H_n.block(2*c, map_hn[kf_w],2,kf_w->size()).noalias() = dz_dpfw*dpfw_dxn;

    //         H_x.block(2*c,map_hx[clone_Cur],2,clone_Cur->size()).noalias()=dz_dpfc*dpfc_dxcur;
    //         H_x.block(2*c,map_hx[tranform],2,tranform->size()).noalias() = dz_dpfg*dpfg_dxtrans;
    //          c++;
    //         // if(state->_options.do_calib_camera_pose)
    //         // {
    //         //     H_x.block(2*c,map_hx[extrinsics],2,extrinsics->size()).noalias()=dz_dpfc*dpfc_dxe;
    //         // }
    //         // if(state->_options.do_calib_camera_intrinsics)
    //         // {
    //         //     H_x.block(2*c,map_hx[distortion],2,distortion->size()).noalias()=dz_dzeta;
    //         // }

    //         // reproject the feature into kf
    //         Eigen::Matrix<double,2,1> uv_norm_kf;
    //         uv_norm_kf<<p_FinKF(0)/p_FinKF(2),p_FinKF(1)/p_FinKF(2);
    //         Eigen::Matrix<double,2,1> uv_norm_kf_linp;
    //         uv_norm_kf_linp<<p_FinKF_linp(0)/p_FinKF_linp(2),p_FinKF_linp(1)/p_FinKF_linp(2);

    //         Eigen::Matrix<double,2,1> uv_dist_kf;
    //         if(is_fisheye) {

    //             // Calculate distorted coordinates for fisheye
    //             double r = std::sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
    //             double theta = std::atan(r);
    //             double theta_d = theta+kf_d(4)*std::pow(theta,3)+kf_d(5)*std::pow(theta,5)+kf_d(6)*std::pow(theta,7)+kf_d(7)*std::pow(theta,9);

    //             // Handle when r is small (meaning our xy is near the camera center)
    //             double inv_r = (r > 1e-8)? 1.0/r : 1.0;
    //             double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

    //             // Calculate distorted coordinates for fisheye
    //             double x1 = uv_norm_kf(0)*cdist;
    //             double y1 = uv_norm_kf(1)*cdist;
    //             uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
    //             uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

    //         } else {

    //             // Calculate distorted coordinates for radial
    //             double r = std::sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
    //             double r_2 = r*r;
    //             double r_4 = r_2*r_2;
    //             double x1 = uv_norm_kf(0)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+2*kf_d(6)*uv_norm_kf(0)*uv_norm_kf(1)+kf_d(7)*(r_2+2*uv_norm_kf(0)*uv_norm_kf(0));
    //             double y1 = uv_norm_kf(1)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+kf_d(6)*(r_2+2*uv_norm_kf(1)*uv_norm_kf(1))+2*kf_d(7)*uv_norm_kf(0)*uv_norm_kf(1);
    //             uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
    //             uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

    //         }

    //         // Our residual

    //         Eigen::Matrix<double,2,1> uv_m;
    //         uv_m << (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].x, (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].y;
    //         cout<<"measure feature: "<<uv_m.transpose()<<endl;
    //         cout<<"predict feature in mapkf: "<<uv_dist_kf.transpose()<<endl;
    //         res.block(2*c,0,2,1) = uv_m - uv_dist_kf;

    //         Eigen::Matrix<double,2,2> dzkf_dzn = Eigen::Matrix<double,2,2>::Zero();
    //         Eigen::Matrix<double,2,8> dzkf_dzeta = Eigen::Matrix<double,2,8>::Zero();
    //         UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_kf_linp, is_fisheye, kf_d, dzkf_dzn, dzkf_dzeta);

    //         Eigen::Matrix<double,2,3> dzn_dpfk = Eigen::Matrix<double,2,3>::Zero();
    //         dzn_dpfk << 1/p_FinKF_linp(2),0,-p_FinKF_linp(0)/(p_FinKF_linp(2)*p_FinKF_linp(2)),
    //                 0, 1/p_FinKF_linp(2),-p_FinKF_linp(1)/(p_FinKF_linp(2)*p_FinKF_linp(2));

    //         H_f.block(2*c,0,2,H_f.cols()).noalias()=dzkf_dzn*dzn_dpfk;
    //        // Move the Jacobian and residual index forward
    //         c++;



    //         // assert(state->of_points.is_open());
    //         // state->of_points<<"current feat: ("<<uv_feat.transpose()<<") ("<<uv_dist.transpose()<<") kf feat("<<uv_m.transpose()<<") ("<<uv_dist_kf.transpose()<<") ";


    //         //test Jacobian
    //         // Vector2d z_fdist,z_fdist_turb;
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist,state->_cam_intrinsics_model.at(0));

    //         // Vector3d turb(0.0001, -0.003, 0.003);
    //         // //turb on p_FinKF:
    //         // compute_meas2(p_FinKF+turb, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on p_FinG:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfw*dpfw_dpfk*turb).transpose()<<endl;

    //         // //turb on R_GtoW;
    //         // Eigen::Matrix<double, 4, 1> dq;
    //         // dq << .5 * turb, 1.0;
    //         // dq = ov_core::quatnorm(dq);
    //         // Vector4d q=quat_multiply(dq, state->transform_vio_to_map->quat());
    //         // JPLQuat *q_JPL=new JPLQuat();
    //         // q_JPL->set_value(q);
    //         // Matrix3d R_GtoW_turb=q_JPL->Rot();
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW_turb,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on R_GtoW:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfg*(dpfg_dxtrans.block(0,0,3,3))*turb).transpose()<<endl;

    //         // //turb on p_GinW:
    //         // compute_meas2(p_FinKF, p_GinW+turb, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on p_GinW:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfg*(dpfg_dxtrans.block(0,3,3,3))*turb).transpose()<<endl;

    //         // //turb on R_KFtoW:
    //         // q=quat_multiply(dq, kf_w->quat());
    //         // q_JPL->set_value(q);
    //         // Matrix3d R_KFtoW_turb=q_JPL->Rot();
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW_turb, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on R_KFtoW:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfw*(dpfw_dxn.block(0,0,3,3))*turb).transpose()<<endl;

    //         // //turb on p_KFinW:
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW+turb, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on p_KFinW:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfw*(dpfw_dxn.block(0,3,3,3))*turb).transpose()<<endl;

    //         // //turb on R_GtoI:
    //         // q=quat_multiply(dq,clone_Cur->quat());
    //         // q_JPL->set_value(q);
    //         // Matrix3d R_GtoI_turb=q_JPL->Rot();
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI_turb,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on R_GtoI:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfc*(dpfc_dxcur.block(0,0,3,3))*turb).transpose()<<endl;

    //         // //turb on p_IinG
    //         // compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG+turb, R_GtoI,p_IinC, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         // cout<<"turb on p_IinG:"<<endl;
    //         // cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         // cout<<"diff jacob: "<<(dz_dpfc*(dpfc_dxcur.block(0,3,3,3))*turb).transpose()<<endl;


    //         // if(state->_options.do_calib_camera_pose)
    //         // {
    //         //     //turb on R_ItoC
    //         //     q=quat_multiply(dq,extrinsics->quat());
    //         //     q_JPL->set_value(q);
    //         //     Matrix3d R_ItoC_turb=q_JPL->Rot();
    //         //     compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC, R_ItoC_turb,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         //     cout<<"turb on R_ItoC:"<<endl;
    //         //     cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         //     cout<<"diff jacob: "<<(dz_dpfc*(dpfc_dxe.block(0,0,3,3))*turb).transpose()<<endl;

    //         //     //turb on p_IinC
    //         //     compute_meas2(p_FinKF, p_GinW, R_GtoW,p_KFinW, R_KFtoW, p_IinG, R_GtoI,p_IinC+turb, R_ItoC,cam_d, z_fdist_turb,state->_cam_intrinsics_model.at(0));
    //         //     cout<<"turb on R_ItoC:"<<endl;
    //         //     cout<<"diff: "<<(z_fdist_turb-z_fdist).transpose()<<endl;
    //         //     cout<<"diff jacob: "<<(dz_dpfc*(dpfc_dxe.block(0,3,3,3))*turb).transpose()<<endl;

    //         // }


    //     }

    // }

}

void UpdaterHelper::compute_meas2(Vector3d p_FinKF,Vector3d p_GinW, Matrix3d R_GtoW, Vector3d p_KFinW, 
                             Matrix3d R_KFtoW, Vector3d p_IinG, Matrix3d R_GtoI, Vector3d p_IinC, 
                             Matrix3d R_ItoC,VectorXd cam_d, Vector2d& z_fdist, bool is_fisheye)
{
 


            //transform the feature in KF to map reference;
            //hk
            Eigen::Vector3d p_FinW =  R_KFtoW * p_FinKF + p_KFinW;

            //transform the feature in map reference into vio reference
            //hw
            Eigen::Vector3d p_FinG = R_GtoW.transpose()*(p_FinW-p_GinW);

            //ht
            Eigen::Vector3d p_FinC = R_ItoC*R_GtoI*(p_FinG-p_IinG) + p_IinC;
            
            //transform the feature in current camera into normal frame
            //hp
            Eigen::Matrix<double,2,1> uv_norm;
            uv_norm<<p_FinC(0)/p_FinC(2),p_FinC(1)/p_FinC(2);
            if(is_fisheye) {

                // Calculate distorted coordinates for fisheye
                double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
                double theta = std::atan(r);
                double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

                // Handle when r is small (meaning our xy is near the camera center)
                double inv_r = (r > 1e-8)? 1.0/r : 1.0;
                double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

                // Calculate distorted coordinates for fisheye
                double x1 = uv_norm(0)*cdist;
                double y1 = uv_norm(1)*cdist;
                z_fdist(0) = cam_d(0)*x1 + cam_d(2);
                z_fdist(1) = cam_d(1)*y1 + cam_d(3);

            } else {

                // Calculate distorted coordinates for radial
                double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
                double r_2 = r*r;
                double r_4 = r_2*r_2;
                double x1 = uv_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm(0)*uv_norm(1)+cam_d(7)*(r_2+2*uv_norm(0)*uv_norm(0));
                double y1 = uv_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm(1)*uv_norm(1))+2*cam_d(7)*uv_norm(0)*uv_norm(1);
                z_fdist(0) = cam_d(0)*x1 + cam_d(2);
                z_fdist(1) = cam_d(1)*y1 + cam_d(3);

            }    
}


void UpdaterHelper::nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {

    // Apply the left nullspace of H_f to all variables
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all index
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_f.cols(); ++n) {
        for (int m = (int) H_f.rows() - 1; m > n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (H_x.block(m - 1, 0, 2, H_x.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
        }
    }

    // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
    // NOTE: need to eigen3 eval here since this experiences aliasing!
    //H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
    H_x = H_x.block(H_f.cols(),0,H_x.rows()-H_f.cols(),H_x.cols()).eval();
    res = res.block(H_f.cols(),0,res.rows()-H_f.cols(),res.cols()).eval();

    // Sanity check
    assert(H_x.rows()==res.rows());
}


void UpdaterHelper::nullspace_project_inplace_with_nuisance(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_n,Eigen::VectorXd &res) {

    // Apply the left nullspace of H_f to all variables
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all i                                                                                                                                       ndex
    MatrixXd Hx_big=MatrixXd::Zero(H_x.rows(),H_x.cols()+H_n.cols());
    Hx_big.block(0,0,H_x.rows(),H_x.cols())=H_x;
    Hx_big.block(0,H_x.cols(),H_n.rows(),H_n.cols())=H_n;

    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_f.cols(); ++n) {
        for (int m = (int) H_f.rows() - 1; m > n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (Hx_big.block(m - 1, 0, 2, Hx_big.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            // (H_n.block(m - 1, 0, 2, H_n.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
        }
    }

    // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
    // NOTE: need to eigen3 eval here since this experiences aliasing!
    //H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
    Hx_big=Hx_big.block(H_f.cols(),0,Hx_big.rows()-H_f.cols(),Hx_big.cols()).eval();
    H_x = Hx_big.block(0,0,Hx_big.rows(),H_x.cols()).eval();
    H_n = Hx_big.block(0,H_x.cols(),Hx_big.rows(),H_n.cols()).eval();
    res = res.block(H_f.cols(),0,res.rows()-H_f.cols(),res.cols()).eval();

    // Sanity check
    assert(H_x.rows()==res.rows());
    assert(H_n.rows()==res.rows());
}

void UpdaterHelper::nullspace_project_inplace_with_nuisance_noise(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_n,Eigen::MatrixXd &noise, Eigen::VectorXd &res) {

    // Apply the left nullspace of H_f to all variables
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all i                                                                                                                                       ndex
    MatrixXd Hx_big=MatrixXd::Zero(H_x.rows(),H_x.cols()+H_n.cols());
    Hx_big.block(0,0,H_x.rows(),H_x.cols())=H_x;
    Hx_big.block(0,H_x.cols(),H_n.rows(),H_n.cols())=H_n;

    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_f.cols(); ++n) {
        for (int m = (int) H_f.rows() - 1; m > n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (Hx_big.block(m - 1, 0, 2, Hx_big.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            // (H_n.block(m - 1, 0, 2, H_n.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (noise.block(m-1,0,2,noise.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (noise.block(0,m-1,noise.rows(),2)).applyOnTheRight(0, 1, tempHo_GR.adjoint());
        }
    }

    // The H_f jacobian max rank is 3 if it is a 3d position, thus size of the left nullspace is Hf.rows()-3
    // NOTE: need to eigen3 eval here since this experiences aliasing!
    //H_f = H_f.block(H_f.cols(),0,H_f.rows()-H_f.cols(),H_f.cols()).eval();
    Hx_big=Hx_big.block(H_f.cols(),0,Hx_big.rows()-H_f.cols(),Hx_big.cols()).eval();
    H_x = Hx_big.block(0,0,Hx_big.rows(),H_x.cols()).eval();
    H_n = Hx_big.block(0,H_x.cols(),Hx_big.rows(),H_n.cols()).eval();
    res = res.block(H_f.cols(),0,res.rows()-H_f.cols(),res.cols()).eval();
    noise = noise.block(H_f.cols(),H_f.cols(),noise.rows()-H_f.cols(),noise.cols()-H_f.cols()).eval();

    // Sanity check
    assert(H_x.rows()==res.rows());
    assert(H_n.rows()==res.rows());
    assert(noise.rows()==res.rows());
}





void UpdaterHelper::measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res) {


    // Return if H_x is a fat matrix (there is no need to compress in this case)
    if(H_x.rows() <= H_x.cols())
        return;

    // Do measurement compression through givens rotations
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all index
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n=0; n<H_x.cols(); n++) {
        for (int m=(int)H_x.rows()-1; m>n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_x(m-1,n), H_x(m,n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_x.block(m-1,n,2,H_x.cols()-n)).applyOnTheLeft(0,1,tempHo_GR.adjoint());
            (res.block(m-1,0,2,1)).applyOnTheLeft(0,1,tempHo_GR.adjoint());
        }
    }

    // If H is a fat matrix, then use the rows
    // Else it should be same size as our state
    int r = std::min(H_x.rows(),H_x.cols());

    // Construct the smaller jacobian and residual after measurement compression
    assert(r<=H_x.rows());
    H_x.conservativeResize(r, H_x.cols());
    res.conservativeResize(r, res.cols());

}

void UpdaterHelper::measurement_compress_inplace_with_nuisance(Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_n,
                                                               Eigen::VectorXd &res)

{
    

    assert(H_x.rows()==H_n.rows());
    Eigen::MatrixXd H_all=MatrixXd::Zero(H_x.rows(),H_x.cols()+H_n.cols());
    H_all.block(0,0,H_x.rows(),H_x.cols())=H_x;
    H_all.block(0,H_x.cols(),H_n.rows(),H_n.cols())=H_n;
    // Return if H_x is a fat matrix (there is no need to compress in this case)
    if(H_all.rows() <= H_all.cols())
        return;

    // Do measurement compression through givens rotations
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all index
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n=0; n<H_all.cols(); n++) {
        for (int m=(int)H_all.rows()-1; m>n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_all(m-1,n), H_all(m,n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_all.block(m-1,n,2,H_all.cols()-n)).applyOnTheLeft(0,1,tempHo_GR.adjoint());
            (res.block(m-1,0,2,1)).applyOnTheLeft(0,1,tempHo_GR.adjoint());
        }
    }

    // If H is a fat matrix, then use the rows
    // Else it should be same size as our state
    int r = std::min(H_all.rows(),H_all.cols());

    // Construct the smaller jacobian and residual after measurement compression
    assert(r<=H_all.rows());
    H_all.conservativeResize(r,H_all.cols());
    H_x.conservativeResize(r, H_x.cols());
    H_n.conservativeResize(r,H_n.cols());
    res.conservativeResize(r, res.cols());
    H_x=H_all.block(0,0,r,H_x.cols());
    H_n=H_all.block(0,H_x.cols(),r,H_n.cols());

}


void UpdaterHelper::measurement_compress_inplace_with_nuisance_noise(Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_n,Eigen::MatrixXd &noise,
                                                               Eigen::VectorXd &res)

{
    

    assert(H_x.rows()==H_n.rows());
    Eigen::MatrixXd H_all=MatrixXd::Zero(H_x.rows(),H_x.cols()+H_n.cols());
    H_all.block(0,0,H_x.rows(),H_x.cols())=H_x;
    H_all.block(0,H_x.cols(),H_n.rows(),H_n.cols())=H_n;
    // Return if H_x is a fat matrix (there is no need to compress in this case)
    if(H_all.rows() <= H_all.cols())
        return;

    // Do measurement compression through givens rotations
    // Based on "Matrix Computations 4th Edition by Golub and Van Loan"
    // See page 252, Algorithm 5.2.4 for how these two loops work
    // They use "matlab" index notation, thus we need to subtract 1 from all index
    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n=0; n<H_all.cols(); n++) {
        for (int m=(int)H_all.rows()-1; m>n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_all(m-1,n), H_all(m,n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_all.block(m-1,n,2,H_all.cols()-n)).applyOnTheLeft(0,1,tempHo_GR.adjoint());
            (res.block(m-1,0,2,1)).applyOnTheLeft(0,1,tempHo_GR.adjoint());
            (noise.block(m-1,0,2,noise.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (noise.block(0,m-1,noise.rows(),2)).applyOnTheRight(0, 1, tempHo_GR.adjoint());
        }
    }

    // If H is a fat matrix, then use the rows
    // Else it should be same size as our state
    int r = std::min(H_all.rows(),H_all.cols());

    // Construct the smaller jacobian and residual after measurement compression
    assert(r<=H_all.rows());
    H_all.conservativeResize(r,H_all.cols());
    H_x.conservativeResize(r, H_x.cols());
    H_n.conservativeResize(r,H_n.cols());
    res.conservativeResize(r, res.cols());
    H_x=H_all.block(0,0,r,H_x.cols());
    H_n=H_all.block(0,H_x.cols(),r,H_n.cols());
    noise.conservativeResize(r,r);

}

void UpdaterHelper::get_feature_jacobian_loc(State *state, UpdaterHelperFeature &feature,
                                             Eigen::MatrixXd &H_x, Eigen::VectorXd &res,
                                             std::vector<Type *> &x_order)
{
    int total_meas = 0;

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    int total_hn = 0;
    std::unordered_map<Type*,size_t> map_hx;
   
    

    // add the current frame state
    // this part is active state part

    // Add this state if it is not added already
   
    // PoseJPL *clone_Cur=nullptr;
    // clone_Cur = state->_clones_IMU.at(state->_timestamp);
    // assert(clone_Cur!=nullptr);
    // if(map_hx.find(clone_Cur) == map_hx.end()) {
    //     map_hx.insert({clone_Cur,total_hx});
    //     x_order.push_back(clone_Cur);
    //     total_hx += clone_Cur->size();
    // }

    //add related clone_IMU
    for(int i=0;i<feature.timestamps[0].size();i++)
    {
           total_meas++;
        // cout<<"feature.timestamps[0]["<<i<<"] :"<<feature.timestamps[0][i]<<endl;
        // if(feature.timestamps[0][i]==state->delay_clones[state->delay_clones.size()-1])
        // if(i==feature.timestamps[0].size()-1)
        {
            // cout<<"have current state: "<<i<<endl;
            // sleep(1);
            PoseJPL* clone=nullptr;
            clone=state->_clones_IMU.at(feature.timestamps[0][i]);
            assert(clone!=nullptr);
            if(map_hx.find(clone) == map_hx.end()) {
            map_hx.insert({clone,total_hx});
            x_order.push_back(clone);
            total_hx += clone->size();
           }
           
        }

    }
    assert(total_meas>=1);
    // // Also add its calibration if we are doing calibration
    // if(state->_options.do_calib_camera_pose) {
    //     // Add this anchor if it is not added already
    //     PoseJPL *clone_calib = state->_calib_IMUtoCAM.at(0); //we use the left cam;
    //     if(map_hx.find(clone_calib) == map_hx.end()) {
    //         map_hx.insert({clone_calib,total_hx});
    //         x_order.push_back(clone_calib);
    //         total_hx += clone_calib->size();
    //     }
    // }

    // if(state->_options.do_calib_camera_intrinsics)
    // {
    //     Vec *distortion = state->_cam_intrinsics.at(0);
    //     map_hx.insert({distortion,total_hx});
    //     x_order.push_back(distortion);
    //     total_hx += distortion->size();
    // }

    

    //we also need to add transformation between map and vio
    if(state->set_transform) //normally, if we are in this function, the state->set_transform should be already set true
    {
        PoseJPL *transform = state->transform_vio_to_map;
        assert(transform!=nullptr);
        if(map_hx.find(transform)==map_hx.end())
        {
            map_hx.insert({transform,total_hx});
            x_order.push_back(transform);
            total_hx += transform->size();
        }
    }

    //=========================================================================
    //=========================================================================

    // // Calculate the position of this feature in the global frame
    // // If anchored, then we need to calculate the position of the feature in the global(vio system reference)
    // Eigen::Vector3d p_FinG = feature.p_FinG;
    // if(LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    //     // Assert that we have an anchor pose for this feature
    //     assert(feature.anchor_cam_id!=-1);
    //     // Get calibration for our anchor camera
    //     Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->Rot();
    //     Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(feature.anchor_cam_id)->pos();
    //     // Anchor pose orientation and position
    //     Eigen::Matrix<double,3,3> R_GtoI = state->_clones_IMU.at(feature.anchor_clone_timestamp)->Rot();
    //     Eigen::Matrix<double,3,1> p_IinG = state->_clones_IMU.at(feature.anchor_clone_timestamp)->pos();
    //     // Feature in the global frame
    //     p_FinG = R_GtoI.transpose() * R_ItoC.transpose()*(feature.p_FinA - p_IinC) + p_IinG;
    // }

    // // Calculate the position of this feature in the global frame FEJ
    // // If anchored, then we can use the "best" p_FinG since the value of p_FinA does not matter
    // Eigen::Vector3d p_FinG_fej = feature.p_FinG_fej;
    // if(LandmarkRepresentation::is_relative_representation(feature.feat_representation)) {
    //     p_FinG_fej = p_FinG;
    // }

    //=========================================================================
    //=========================================================================

    // Allocate our residual and Jacobians
    int c = 0;
    // int jacobsize = (feature.feat_representation!=LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
    int jacobsize = 3; //the observed feature is represented by 3d with respect to the loop_keyframe
    //for each feature, it has two reprojection error, in current cam frame and map_kf frame
    res = Eigen::VectorXd::Zero(2*total_meas);
    H_x = Eigen::MatrixXd::Zero(2*total_meas,total_hx);

    vector<double> keyframe_id;
    vector<size_t> keyframe_feature_id;
    for(auto const& pair: feature.keyframe_matched_obs)
    {
        keyframe_id.clear();
        keyframe_feature_id.clear();
        for(auto const& match: pair.second)
        {
            double kf_id = match.first;
            size_t kf_feature_id = match.second;
            keyframe_id.push_back(kf_id);
            keyframe_feature_id.push_back(kf_feature_id);
        }
        //as each keyframe has a 3d point represented in itself, we choose the first kf as anchor kf,
        //and reproject its 3d point into kfs and current frame
        double kf_id = keyframe_id[0];
        Keyframe* anchor_kf = state->get_kfdataBase()->get_keyframe(kf_id);
        assert(anchor_kf != nullptr);
        size_t kf_feature_id = keyframe_feature_id[0];

        //now we reproject anchor_kf's 3d point into current frame
        //get the kf position in world frame.
        PoseJPL* anchor_kf_w = state->_clones_Keyframe[kf_id];
        Eigen::Matrix<double,3,3> R_AnchorKFtoW = anchor_kf_w->Rot();
        Eigen::Matrix<double,3,1> p_AnchorKFinW = anchor_kf_w->pos();

        //get the transform between the VIO(G) to World(W)
        PoseJPL* tranform = state->transform_vio_to_map;
        Eigen::Matrix<double,3,3> R_GtoW_linp = state->transform_vio_to_map->Rot_linp();
        Eigen::Matrix<double,3,1> p_GinW_linp = state->transform_vio_to_map->pos_linp();
        Eigen::Matrix<double,3,3> R_GtoW = state->transform_vio_to_map->Rot();
        Eigen::Matrix<double,3,1> p_GinW = state->transform_vio_to_map->pos();
        if(!state->iter)
        {
          assert(p_GinW_linp==p_GinW);
        }
        Eigen::Matrix<double,3,3> R_GtoW_true = Eigen::Matrix<double,3,3>::Identity();
        Eigen::Matrix<double,3,1> p_GinW_true = Eigen::Matrix<double,3,1>::Zero();

        double ts = pair.first; //ts

        cv::Point3f p_financhor = anchor_kf->point_3d_map.at(ts)[kf_feature_id];
        Vector3d p_FinAnchorKF(double(p_financhor.x),double(p_financhor.y),double(p_financhor.z));
        cv::Point3f p_financhor_linp = anchor_kf->point_3d_linp_map.at(ts)[kf_feature_id];
        Vector3d p_FinAnchorKF_linp(double(p_financhor_linp.x),double(p_financhor_linp.y),double(p_financhor_linp.z));
        assert(p_financhor==p_financhor_linp);
        // cv::Point2f uv_ob = anchor_kf->matched_point_2d_uv_map.at(ts)[kf_feature_id];
        // double x = uv_ob.x;
        // double y = uv_ob.y;
        // Vector2d uv_OB(x,y);
        // double uv_x = feature.uvs[0][0](0);
        // double uv_y = feature.uvs[0][0](1);
        // Vector2d uv_feat(uv_x,uv_y);
        // assert(uv_OB == uv_feat); //check if feature link is correct

        //transform the feature to map reference;
        //hk
        Eigen::Vector3d p_FinW =  R_AnchorKFtoW * p_FinAnchorKF + p_AnchorKFinW;
        Eigen::Vector3d p_FinW_linp = R_AnchorKFtoW * p_FinAnchorKF_linp + p_AnchorKFinW;
        if(!state->iter)
        {
          assert(p_FinW==p_FinW_linp);
        }

        //dhk_dpfk
        Eigen::Matrix<double,3,3> dpfw_dpfk=R_AnchorKFtoW;

        //dhk_dxn
        Eigen::Matrix<double,3,6> dpfw_dxnanchor=Eigen::Matrix<double,3,6>::Zero();
        dpfw_dxnanchor.block(0,0,3,3) = skew_x(R_AnchorKFtoW*p_FinAnchorKF_linp);
        dpfw_dxnanchor.block(0,3,3,3) = Eigen::Matrix3d::Identity();

        //transform the feature in map reference into vio reference
        //hw
        Eigen::Vector3d p_FinG = R_GtoW.transpose()*(p_FinW-p_GinW);
        Eigen::Vector3d p_FinG_linp = R_GtoW_linp.transpose()*(p_FinW_linp-p_GinW_linp);
        Eigen::Vector3d p_FinG_true = R_GtoW_true.transpose()*(p_FinW-p_GinW_true);
        //dhw_dpfw
        Eigen::Matrix<double,3,3> dpfg_dpfw = R_GtoW_linp.transpose();
        if(state->_options.use_gt)
        {
          dpfg_dpfw = R_GtoW_true.transpose();
        }
        //dhw_dxtrans
        Eigen::Matrix<double,3,6> dpfg_dxtrans = Eigen::Matrix<double,3,6>::Zero();
        dpfg_dxtrans.block(0,0,3,3) =-R_GtoW_linp.transpose()*skew_x(p_FinW_linp-p_GinW_linp);
        dpfg_dxtrans.block(0,3,3,3) = -R_GtoW_linp.transpose();
        if(state->_options.use_gt)
        {
          dpfg_dxtrans.block(0,0,3,3) =-R_GtoW_true.transpose()*skew_x(p_FinW-p_GinW_true);
          dpfg_dxtrans.block(0,3,3,3) = -R_GtoW_true.transpose();
        }

        //transform the feature in vio reference into current camera frame
        PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
        Matrix3d R_ItoC_linp = extrinsics->Rot_linp();
        Vector3d p_IinC_linp = extrinsics->pos_linp();
        Matrix3d R_ItoC = extrinsics->Rot();
        Vector3d p_IinC = extrinsics->pos();
        assert(p_IinC_linp==p_IinC);
        for(int i=0;i<feature.timestamps[0].size();i++)
        {
            PoseJPL *clone_Cur = state->_clones_IMU.at(feature.timestamps[0][i]);
            assert(clone_Cur!=nullptr);

            Eigen::Matrix<double,17,1> state_true_cur = Eigen::Matrix<double,17,1>::Zero();
            IMU* imu_true_cur = new IMU();
            if(state->_options.use_gt)
            {
              bool success = state->get_state(feature.timestamps[0][i],state_true_cur);
              assert(success);
            }
            imu_true_cur->set_value(state_true_cur.block(1,0,16,1));

            Matrix3d R_GtoI_fej = clone_Cur->Rot_fej();
            Vector3d p_IinG_fej = clone_Cur->pos_fej();
            Matrix3d R_GtoI_linp = clone_Cur->Rot_linp();
            Vector3d p_IinG_linp = clone_Cur->pos_linp();
            Matrix3d R_GtoI = clone_Cur->Rot();
            Vector3d p_IinG = clone_Cur->pos();
            Matrix3d R_GtoI_true = imu_true_cur->Rot();
            Vector3d p_IinG_true = imu_true_cur->pos();
            // assert(R_GtoI_linp==R_GtoI);
            //ht
            Eigen::Vector3d p_FinC = R_ItoC*R_GtoI*(p_FinG-p_IinG) + p_IinC;
            Eigen::Vector3d p_FinC_linp = R_ItoC_linp*R_GtoI_linp*(p_FinG_linp-p_IinG_linp) + p_IinC_linp;
            Eigen::Vector3d p_FinC_true = R_ItoC_linp*R_GtoI_true*(p_FinG_true-p_IinG_true) + p_IinC_linp;
            //dht_dpfg
            // Eigen::Matrix3d dpfc_dpfg = R_ItoC_linp*R_GtoI_fej;
            Eigen::Matrix3d dpfc_dpfg = R_ItoC_linp*R_GtoI_linp;
            if(state->_options.use_gt)
            {
              dpfc_dpfg = R_ItoC_linp * R_GtoI_true;
            }
            //dht_dxcur
            Eigen::Matrix<double,3,6> dpfc_dxcur = Eigen::Matrix<double,3,6>::Zero();
            // dpfc_dxcur.block(0,0,3,3) = R_ItoC_linp*skew_x(R_GtoI_fej*(p_FinG_linp-p_IinG_fej));
            // dpfc_dxcur.block(0,3,3,3) = -R_ItoC_linp*R_GtoI_fej;
            dpfc_dxcur.block(0,0,3,3) = R_ItoC_linp*skew_x(R_GtoI_linp*(p_FinG_linp-p_IinG_linp));
            dpfc_dxcur.block(0,3,3,3) = -R_ItoC_linp*R_GtoI_linp;
            if(state->_options.use_gt)
            {
              dpfc_dxcur.block(0,0,3,3) = R_ItoC_linp*skew_x(R_GtoI_true*(p_FinG_true-p_IinG_true));
              dpfc_dxcur.block(0,3,3,3) = -R_ItoC_linp*R_GtoI_true;
            }


            Eigen::Matrix<double,3,6> dpfc_dxe=Eigen::Matrix<double,3,6>::Zero();
            // if(state->_options.do_calib_camera_pose)
            // {

            //     dpfc_dxe.block(0,0,3,3)=skew_x(R_ItoC_linp*R_GtoI_linp*(p_FinG_linp-p_IinG_linp));
            //     dpfc_dxe.block(0,3,3,3)=Matrix3d::Identity();
            // }

            //transform the feature in current camera into normal frame
            //hp
            Eigen::Matrix<double,2,1> uv_norm;
            uv_norm<<p_FinC(0)/p_FinC(2),p_FinC(1)/p_FinC(2);
            Eigen::Matrix<double,2,1> uv_norm_linp;
            uv_norm_linp<<p_FinC_linp(0)/p_FinC_linp(2),p_FinC_linp(1)/p_FinC_linp(2);
            Eigen::Matrix<double,2,1> uv_norm_true;
            if(state->_options.use_gt)
            {
              uv_norm_true<<p_FinC_true(0)/p_FinC_true(2),p_FinC_true(1)/p_FinC_true(2);
            }
            


            // Distort the normalized coordinates (false=radtan, true=fisheye)
            Eigen::Matrix<double,2,1> uv_dist;
            Vec* distortion = state->_cam_intrinsics.at(0);
            Eigen::Matrix<double,8,1> cam_d = distortion->linp();
            // Calculate distortion uv and jacobian
            if(state->_cam_intrinsics_model.at(0)) {

                // Calculate distorted coordinates for fisheye
                double r = std::sqrt(uv_norm_linp(0)*uv_norm_linp(0)+uv_norm_linp(1)*uv_norm_linp(1));
                double theta = std::atan(r);
                double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

                // Handle when r is small (meaning our xy is near the camera center)
                double inv_r = (r > 1e-8)? 1.0/r : 1.0;
                double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

                // Calculate distorted coordinates for fisheye
                double x1 = uv_norm_linp(0)*cdist;
                double y1 = uv_norm_linp(1)*cdist;
                uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                uv_dist(1) = cam_d(1)*y1 + cam_d(3);

            } else {

                // Calculate distorted coordinates for radial
                double r = std::sqrt(uv_norm_linp(0)*uv_norm_linp(0)+uv_norm_linp(1)*uv_norm_linp(1));
                double r_2 = r*r;
                double r_4 = r_2*r_2;
                double x1 = uv_norm_linp(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm_linp(0)*uv_norm_linp(1)+cam_d(7)*(r_2+2*uv_norm_linp(0)*uv_norm_linp(0));
                double y1 = uv_norm_linp(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm_linp(1)*uv_norm_linp(1))+2*cam_d(7)*uv_norm_linp(0)*uv_norm_linp(1);
                uv_dist(0) = cam_d(0)*x1 + cam_d(2);
                uv_dist(1) = cam_d(1)*y1 + cam_d(3);

            }

            double uv_x = feature.uvs[0][i](0);
            double uv_y = feature.uvs[0][i](1);
            Vector2d uv_feat(uv_x,uv_y);

            // Our residual
            cout<<"measure feature: "<<uv_feat.transpose()<<endl;
            cout<<"predict feature in curimage: "<<uv_dist.transpose()<<endl;
            
            res.block(2*c,0,2,1) = uv_feat - uv_dist;


            // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
            //dhd_dzn
            Eigen::Matrix<double,2,2> dz_dzn = Eigen::Matrix<double,2,2>::Zero();
            Eigen::Matrix<double,2,8> dz_dzeta = Eigen::Matrix<double,2,8>::Zero();
            if(state->_options.use_gt)
            {
              uv_norm_linp = uv_norm_true;
            }
            UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_linp, state->_cam_intrinsics_model.at(0), cam_d, dz_dzn, dz_dzeta);

            //Compute Jacobian of transforming feature in kf frame into nomalized uv frame;
            //dhp_dpfc
            Eigen::Matrix<double,2,3> dzn_dpfc = Eigen::Matrix<double,2,3>::Zero();

            // p_FinC_linp = R_ItoC_linp*R_GtoI_fej*(p_FinG_linp-p_IinG_fej) + p_IinC_linp;
            if(state->_options.use_gt)
            {
              p_FinC_linp = p_FinC_true;
            }
            
            dzn_dpfc << 1/p_FinC_linp(2),0,-p_FinC_linp(0)/(p_FinC_linp(2)*p_FinC_linp(2)),
                    0, 1/p_FinC_linp(2),-p_FinC_linp(1)/(p_FinC_linp(2)*p_FinC_linp(2));


            // Precompute some matrices
            Eigen::Matrix<double,2,3> dz_dpfc = dz_dzn*dzn_dpfc;
            Eigen::Matrix<double,2,3> dz_dpfg = dz_dpfc*dpfc_dpfg;
            Eigen::Matrix<double,2,3> dz_dpfw = dz_dpfg*dpfg_dpfw;

          

            {
                H_x.block(2*c,map_hx[clone_Cur],2,clone_Cur->size()).noalias()=dz_dpfc*dpfc_dxcur;


            }
            H_x.block(2*c,map_hx[tranform],2,tranform->size()).noalias() = dz_dpfg*dpfg_dxtrans;
            // }

            c++;
        }


    }
}



