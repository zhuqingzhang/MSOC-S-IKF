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
using namespace ov_rimsckf;


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


    
    //make sure that the feature representaion is global3D, as the following Jacobians is derived 
    //based on global3D representation.
    assert(!LandmarkRepresentation::is_relative_representation(feature.feat_representation));


    //=========================================================================
    //=========================================================================

    // Derivative of p_FinG in respect to feature representation. i.e. d(p_FinG)/d(lamda...)
    // This only needs to be computed once and thus we pull it out of the loop below
    Eigen::MatrixXd dpfg_dlambda;
    std::vector<Eigen::MatrixXd> dpfg_dx;
    std::vector<Type*> dpfg_dx_order;
    UpdaterHelper::get_feature_jacobian_representation(state, feature, dpfg_dlambda, dpfg_dx, dpfg_dx_order);

    // Assert that all the ones in our order are already in our local jacobian mapping
    for(auto &type : dpfg_dx_order) {
        assert(map_hx.find(type)!=map_hx.end());
    }
    
   
    //For RIMSCKF, the error of feature needs to be bond with a clone pose(let's say R_AIinG , p_AIinG):
    //error_feature = /hat{feature} - /hat{R_AIinG}*R_AIinG^{\tilde}*feaure
    //NOTE: alough the clone pose is recorded with the form of R_GinI, p_IinG,
    //the Jacobian is derived based on R_IinG,p_IinG on mainfold SE(3).
    Eigen::Matrix<double,3,3> R_AItoG;
    Eigen::Matrix<double,3,1> p_AIinG;
    Eigen::Matrix<double,3,3> R_AItoG_true;
    Eigen::Matrix<double,3,1> p_AIinG_true;
    PoseJPL* clone_AIi = new PoseJPL();
    //we use the current IMU state as anchor frame to formulate the error_feature
    // double anchor_timestamp = -1.0;
    double anchor_timestamp = state->_timestamp;
    assert(state->_clones_IMU.find(anchor_timestamp)!=state->_clones_IMU.end());
    // assert((state->_clones_IMU[anchor_timestamp]->quat()-state->_imu->quat()).norm()<1e-5);
    // assert((state->_clones_IMU[anchor_timestamp]->pos()-state->_imu->pos()).norm()<1e-5);
    clone_AIi = state->_clones_IMU.at(anchor_timestamp);
    Eigen::Matrix<double,17,1> state_true_anchor = Eigen::Matrix<double,17,1>::Zero();
    IMU* imu_true_anchor = new IMU();
    if(state->_options.use_gt)
    {
      bool success = state->get_state(anchor_timestamp,state_true_anchor);
      assert(success);
    }
    imu_true_anchor->set_value(state_true_anchor.block(1,0,16,1));
    if(map_hx.find(clone_AIi) == map_hx.end()) {
                map_hx.insert({clone_AIi,total_hx});
                x_order.push_back(clone_AIi);
                total_hx += clone_AIi->size();
    }
    
    // Allocate our residual and Jacobians
    int c = 0;
    int jacobsize = (feature.feat_representation!=LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) ? 3 : 1;
    res = Eigen::VectorXd::Zero(2*total_meas);
    H_f = Eigen::MatrixXd::Zero(2*total_meas,jacobsize);
    H_x = Eigen::MatrixXd::Zero(2*total_meas,total_hx);


    Eigen::Vector3d p_FinG = feature.p_FinG;
    Eigen::Vector3d p_FinG_true;
    if(state->_options.use_gt)
    {
      int featid = feature.featid - state->_options.max_aruco_features - 1;
      cout<<"featureid: "<<featid<<" _featmap size: "<<state->_featmap.size()<<endl;
      assert(state->_featmap.find(featid)!=state->_featmap.end());
      p_FinG_true = state->_featmap.at(featid);
    }

    // Loop through each camera for this feature
    for (auto const& pair : feature.timestamps) {

        // Our calibration between the IMU and CAMi frames
        //cout<<"for cameraid "<<pair.first<<endl;
        Vec* distortion = state->_cam_intrinsics.at(pair.first);
        PoseJPL* calibration = state->_calib_IMUtoCAM.at(pair.first);
        Eigen::Matrix<double,3,3> R_ItoC = calibration->Rot();
        Eigen::Matrix<double,3,1> p_IinC = calibration->pos();
        //Following Jacobian is derived based in R_CtoI and p_CinI
        Eigen::Matrix<double,3,3> R_CtoI = R_ItoC.transpose();
        Eigen::Matrix<double,3,1> p_CinI = -R_CtoI*p_IinC;
        Eigen::Matrix<double,8,1> cam_d = distortion->value();
        

        // Loop through all measurements for this specific camera
        for (size_t m = 0; m < feature.timestamps[pair.first].size(); m++) {

            //=========================================================================
            //=========================================================================
            // Get current IMU clone state
            PoseJPL* clone_Ii = state->_clones_IMU.at(feature.timestamps[pair.first].at(m));

            Eigen::Matrix<double,17,1> state_true_cur = Eigen::Matrix<double,17,1>::Zero();
            IMU* imu_true_cur = new IMU();
            if(state->_options.use_gt)
            {
              bool success = state->get_state(feature.timestamps[pair.first].at(m),state_true_cur);
              assert(success);
            }
            imu_true_cur->set_value(state_true_cur.block(1,0,16,1));



            // Get current feature in IMU frame
            Eigen::Matrix<double,3,1> p_FinIi = clone_Ii->Rot()*(p_FinG-clone_Ii->pos());
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


            // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
            Eigen::Matrix<double,2,2> dz_dzn = Eigen::Matrix<double,2,2>::Zero();
            Eigen::Matrix<double,2,8> dz_dzeta = Eigen::Matrix<double,2,8>::Zero();
            UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm, state->_cam_intrinsics_model.at(pair.first), cam_d, dz_dzn, dz_dzeta);
            
            // Normalized coordinates in respect to projection function
            if(state->_options.use_gt)
            {
              p_FinIi = imu_true_cur->Rot()*(p_FinG_true-imu_true_cur->pos());
            // Project the current feature into the current frame of reference to get predicted measurement
              p_FinCi = R_ItoC*p_FinIi+p_IinC;
            }
            Eigen::Matrix<double,2,3> dzn_dpfc = Eigen::Matrix<double,2,3>::Zero();
            dzn_dpfc << 1/p_FinCi(2),0,-p_FinCi(0)/(p_FinCi(2)*p_FinCi(2)),
                    0, 1/p_FinCi(2),-p_FinCi(1)/(p_FinCi(2)*p_FinCi(2));

            // cout<<"anchor_timestamp: "<<to_string(anchor_timestamp)<<" current_timestamp: "<<to_string(state->_timestamp)
            // <<" current feature observed timestamp: "<<to_string(feature.timestamps[pair.first].at(m))<<endl;
            if(anchor_timestamp==-1.0||feature.timestamps[pair.first].at(m)==anchor_timestamp)
            {
                //if not set anchor_clone_pose, we use current clone pose to bind with feature error
                if(anchor_timestamp==-1.0)
                {
                    anchor_timestamp=feature.timestamps[pair.first].at(m);
                }
                if(state->_options.use_gt)
                {
                  bool success = state->get_state(anchor_timestamp,state_true_anchor);
                  assert(success);
                }
                imu_true_anchor->set_value(state_true_anchor.block(1,0,16,1));
                
                clone_AIi = state->_clones_IMU.at(anchor_timestamp);
                R_AItoG = clone_AIi->Rot().transpose();
                p_AIinG = clone_AIi->pos();
                R_AItoG_true = imu_true_anchor->Rot().transpose();
                p_AIinG_true = imu_true_anchor->pos();

                Eigen::Matrix<double,3,3> R_GtoAC = R_ItoC * R_AItoG.transpose();
                Eigen::Matrix<double,3,3> R_GtoAC_true = R_ItoC * R_AItoG_true.transpose();
                
                //compute d(feature_camera)/dx
                //1. d(feature_camera)/d(pose_AI)
                Eigen::Matrix<double,3,6> dpfc_dAI = Eigen::Matrix<double,3,6>::Zero();
                dpfc_dAI.block(0,3,3,3) = -R_GtoAC;
                if(state->_options.use_gt)
                {
                  dpfc_dAI.block(0,3,3,3) = -R_GtoAC_true;
                }

                //2. d(feature_camera)/d(feature_global)
                Eigen::Matrix<double,3,3> dpfc_dpfg = Eigen::Matrix<double,3,3>::Identity();
                dpfc_dpfg = R_GtoAC;
                if(state->_options.use_gt)
                {
                  dpfc_dpfg = R_GtoAC_true;
                }

                // Precompute some matrices
                Eigen::Matrix<double,2,3> dz_dpfc = dz_dzn*dzn_dpfc;
                Eigen::Matrix<double,2,3> dz_dpfg = dz_dpfc*dpfc_dpfg;

                // CHAINRULE: get the total feature Jacobian
                H_f.block(2*c,0,2,H_f.cols()).noalias() = dz_dpfg*dpfg_dlambda; //when feature is GLOBAL3D, dpfg_dlambda=Identity;
                // CHAINRULE: get state clone Jacobian
                H_x.block(2*c,map_hx[clone_AIi],2,clone_AIi->size()).noalias() = dz_dpfc*dpfc_dAI;
                //cout<<"in anchor, id is: "<<map_hx[clone_AIi]<<endl;
                // CHAINRULE: loop through all extra states and add their
                // NOTE: we add the Jacobian here as we might be in the anchoring pose for this measurement
                //for GLOBAL3D representation, dpfg_dx_order is empty
                for(size_t i=0; i<dpfg_dx_order.size(); i++) {
                    H_x.block(2*c,map_hx[dpfg_dx_order.at(i)],2,dpfg_dx_order.at(i)->size()).noalias() += dz_dpfg*dpfg_dx.at(i);
                }


                
                // Jacobian of extrinsics
                Eigen::Matrix<double,3,6> dpfc_dex = Eigen::Matrix<double,3,6>::Zero();
                if(state->_options.do_calib_camera_pose) {
                    dpfc_dex.block(0,0,3,3) = - R_GtoAC * skew_x(p_AIinG-p_FinG) * R_AItoG;
                    dpfc_dex.block(0,3,3,3) = - R_GtoAC * R_AItoG;
                    if(state->_options.use_gt)
                    {
                      dpfc_dex.block(0,0,3,3) = - R_GtoAC_true * skew_x(p_AIinG_true-p_FinG_true) * R_AItoG_true;
                      dpfc_dex.block(0,3,3,3) = - R_GtoAC_true * R_AItoG_true;
                    }

                    // Chainrule it and add it to the big jacobian
                    H_x.block(2*c,map_hx[calibration],2,calibration->size()).noalias() += dz_dpfc*dpfc_dex;
                } 

                // Jacobian of intrinsics
                if(state->_options.do_calib_camera_intrinsics) {
                    H_x.block(2*c,map_hx[distortion],2,distortion->size()) = dz_dzeta;
                }

                // Move the Jacobian and residual index forward
                c++;
            }
            else
            {
                //if not the anchor_clone_pose, then the jacobian would be related with 
                //current clone pose and anchor clone pose

                Eigen::Matrix<double,3,3> R_ItoG = clone_Ii->Rot().transpose();
                Eigen::Matrix<double,3,1> p_IinG = clone_Ii->pos();
                Eigen::Matrix<double,3,3> R_ItoG_true = imu_true_cur->Rot().transpose();
                Eigen::Matrix<double,3,1> p_IinG_true = imu_true_cur->pos();

                Eigen::Matrix<double,3,3> R_GtoIC = R_ItoC * R_ItoG.transpose();
                Eigen::Matrix<double,3,3> R_GtoIC_true = R_ItoC * R_ItoG_true.transpose();

                //compute d(feature_camera)/dx

                //1. d(feature_camera)/d(pose_I)
                Eigen::Matrix<double,3,6> dpfc_dI = Eigen::Matrix<double,3,6>::Zero();
                dpfc_dI.block(0,0,3,3) = R_GtoIC * skew_x(p_FinG);
                dpfc_dI.block(0,3,3,3) = -R_GtoIC;
                if(state->_options.use_gt)
                {
                  dpfc_dI.block(0,0,3,3) = R_GtoIC_true * skew_x(p_FinG_true);
                  dpfc_dI.block(0,3,3,3) = -R_GtoIC_true;
                }

                //2. d(feature_camera)/d(pose_A)
                Eigen::Matrix<double,3,6> dpfc_dAI = Eigen::Matrix<double,3,6>::Zero();
                dpfc_dAI.block(0,0,3,3) = -R_GtoIC * skew_x(p_FinG); 
                if(state->_options.use_gt)
                {
                  dpfc_dAI.block(0,0,3,3) = -R_GtoIC_true * skew_x(p_FinG_true); 
                }

                //3. d(feature_camera)/d(feature_global)
                Eigen::Matrix<double,3,3> dpfc_dpfg = Eigen::Matrix<double,3,3>::Identity();
                dpfc_dpfg = R_GtoIC;
                if(state->_options.use_gt)
                {
                  dpfc_dpfg = R_GtoIC_true;
                }

                // Precompute some matrices
                Eigen::Matrix<double,2,3> dz_dpfc = dz_dzn*dzn_dpfc;
                Eigen::Matrix<double,2,3> dz_dpfg = dz_dpfc*dpfc_dpfg;


                // CHAINRULE: get the total feature Jacobian
                H_f.block(2*c,0,2,H_f.cols()).noalias() = dz_dpfg*dpfg_dlambda; //when feature is GLOBAL3D, dpfg_dlambda=Identity;
                // CHAINRULE: get state clone Jacobian
                H_x.block(2*c,map_hx[clone_AIi],2,clone_AIi->size()).noalias() = dz_dpfc*dpfc_dAI;
                H_x.block(2*c,map_hx[clone_Ii],2,clone_Ii->size()).noalias() = dz_dpfc*dpfc_dI;
                //cout<<"anchor id is: "<<map_hx[clone_AIi]<<endl;
                //cout<<"current id is: "<<map_hx[clone_Ii]<<endl;
                // CHAINRULE: loop through all extra states and add their
                // NOTE: we add the Jacobian here as we might be in the anchoring pose for this measurement
                //for GLOBAL3D representation, dpfg_dx_order is empty
                for(size_t i=0; i<dpfg_dx_order.size(); i++) {
                    H_x.block(2*c,map_hx[dpfg_dx_order.at(i)],2,dpfg_dx_order.at(i)->size()).noalias() += dz_dpfg*dpfg_dx.at(i);
                }



                // Jacobian of extrinsics
                Eigen::Matrix<double,3,6> dpfc_dex = Eigen::Matrix<double,3,6>::Zero();
                if(state->_options.do_calib_camera_pose) {
                    dpfc_dex.block(0,0,3,3) = - R_GtoIC * skew_x(p_IinG-p_FinG) * R_ItoG;
                    dpfc_dex.block(0,3,3,3) = - R_GtoIC * R_ItoG;
                    if(state->_options.use_gt)
                    {
                      dpfc_dex.block(0,0,3,3) = - R_GtoIC_true * skew_x(p_IinG_true-p_FinG_true) * R_ItoG_true;
                      dpfc_dex.block(0,3,3,3) = - R_GtoIC_true * R_ItoG_true;
                    }
                    // Chainrule it and add it to the big jacobian
                    H_x.block(2*c,map_hx[calibration],2,calibration->size()).noalias() += dz_dpfc*dpfc_dex;
                } 

                // Jacobian of intrinsics
                if(state->_options.do_calib_camera_intrinsics) {
                    H_x.block(2*c,map_hx[distortion],2,distortion->size()) = dz_dzeta;
                }

                // Move the Jacobian and residual index forward
                c++;
            }
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


void UpdaterHelper::get_feature_jacobian_loc(State *state, UpdaterHelperFeature &feature,
                                             Eigen::MatrixXd &H_x, Eigen::VectorXd &res,
                                             std::vector<Type *> &x_order){

  // Total number of measurements for this feature
    int total_meas = 0;

    // Compute the size of the states involved with this feature
    int total_hx = 0;
    int total_hn = 0;
    std::unordered_map<Type*,size_t> map_hx;


    //add related clone_IMU
    for(int i=0;i<feature.timestamps[0].size();i++)
    {
      total_meas++;
    
      PoseJPL* clone=nullptr;
      clone=state->_clones_IMU.at(feature.timestamps[0][i]);
      assert(clone!=nullptr);
      if(map_hx.find(clone) == map_hx.end()) {
      map_hx.insert({clone,total_hx});
      x_order.push_back(clone);
      total_hx += clone->size();
      }
    }

    //if there is no dealy update, it should only be 2, only observed by cuurent frame.
    assert(total_meas>=1);
   

    

    //we also need to add transformation between map and vio
    if(state->set_transform) //normally, if we are in this function, the state->set_transform should be already set true
    {
        PoseJPL *transform = state->transform_map_to_vio;
        if(map_hx.find(transform)==map_hx.end())
        {
            map_hx.insert({transform,total_hx});
            x_order.push_back(transform);
            total_hx += transform->size();
        }
    }


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

        //get the kf position in world frame.
        PoseJPL* anchor_kf_w = state->_clones_Keyframe[kf_id];
        assert(anchor_kf_w != nullptr);
        Eigen::Matrix<double,3,3> R_AnchorKFtoW = anchor_kf_w->Rot();
        Eigen::Matrix<double,3,1> p_AnchorKFinW = anchor_kf_w->pos();

        //get the transform between the VIO(G) to World(W)
        PoseJPL* tranform = state->transform_map_to_vio;
        Eigen::Matrix<double,3,3> R_WtoG = state->transform_map_to_vio->Rot();
        Eigen::Matrix<double,3,1> p_WinG = state->transform_map_to_vio->pos();

        Eigen::Matrix<double,3,3> R_WtoG_linp = state->transform_map_to_vio->Rot_linp();
        Eigen::Matrix<double,3,1> p_WinG_linp = state->transform_map_to_vio->pos_linp();

        // Eigen::Matrix<double,3,3> R_WtoG_true = state->transform_map_to_vio->Rot_linp();
        // Eigen::Matrix<double,3,1> p_WinG_true = state->transform_map_to_vio->pos_linp();
        Eigen::Matrix<double,3,3> R_WtoG_true = Eigen::Matrix<double,3,3>::Identity();
        Eigen::Matrix<double,3,1> p_WinG_true = Eigen::Matrix<double,3,1>::Zero();

        if(!state->iter)
        {
            assert(R_WtoG==R_WtoG_linp);
            assert(p_WinG_linp==p_WinG);
        }
        double ts = pair.first; //ts

        cv::Point3f p_financhor = anchor_kf->point_3d_map.at(ts)[kf_feature_id];
        Vector3d p_FinAnchorKF(double(p_financhor.x),double(p_financhor.y),double(p_financhor.z));
      

        //transform the feature to map reference; we use this feature(p_FinW) in the map to do feature reprojection
        Eigen::Vector3d p_FinW =  R_AnchorKFtoW * p_FinAnchorKF + p_AnchorKFinW;

        //first we reproject p_FinW into IMU clones frame and compute the related Jacobians

        //transform p_FinW to p_FinG
        Eigen::Vector3d p_FinG = R_WtoG * p_FinW + p_WinG;
        Eigen::Vector3d p_FinG_linp = R_WtoG_linp * p_FinW + p_WinG_linp;
        Eigen::Vector3d p_FinG_true = R_WtoG_true * p_FinW + p_WinG_true;

        //transform the feature in vio reference into IMU clones camera frame
        PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
        Matrix3d R_ItoC = extrinsics->Rot_linp();
        Vector3d p_IinC = extrinsics->pos_linp();
        Matrix3d R_CtoI = R_ItoC.transpose();
        Vector3d p_CinI = -R_CtoI * p_IinC;
        for(int i=0;i<feature.timestamps[0].size();i++)
        {
            PoseJPL *clone_Cur = state->_clones_IMU.at(feature.timestamps[0][i]);
            assert(clone_Cur!=nullptr);
            Matrix3d R_GtoI = clone_Cur->Rot();
            Vector3d p_IinG = clone_Cur->pos();
            Matrix3d R_GtoI_linp = clone_Cur->Rot_linp();
            Vector3d p_IinG_linp = clone_Cur->pos_linp();
            Matrix<double,17,1> state_true_cur = Matrix<double,17,1>::Zero();
            IMU* imu_true_cur = new IMU();
            if(state->_options.use_gt)
            {
              bool success = state->get_state(feature.timestamps[0][i],state_true_cur);
              assert(success);
            }
            imu_true_cur->set_value(state_true_cur.block(1,0,16,1));

            Matrix3d R_GtoI_true = imu_true_cur->Rot();
            Vector3d p_IinG_true = imu_true_cur->pos();

            if(!state->iter)
            {
                assert(R_GtoI==R_GtoI_linp);
                assert(p_IinG==p_IinG_linp);
            }
            // assert(R_GtoI_linp==R_GtoI);
            //ht
            Eigen::Vector3d p_FinC = R_ItoC*R_GtoI*(p_FinG-p_IinG) + p_IinC;
            Eigen::Matrix3d R_GtoC = R_ItoC * R_GtoI;

            Eigen::Vector3d p_FinC_linp = R_ItoC*R_GtoI_linp*(p_FinG_linp-p_IinG_linp) + p_IinC;
            Eigen::Matrix3d R_GtoC_linp = R_ItoC * R_GtoI_linp;

            Eigen::Vector3d p_FinC_true = R_ItoC*R_GtoI_true*(p_FinG_true-p_IinG_true) + p_IinC;
            Eigen::Matrix3d R_GtoC_true = R_ItoC * R_GtoI_true;

            //dht_dpfw
            Eigen::Matrix3d dpfc_dpfw = - R_GtoC_linp * R_WtoG_linp;
            if(state->_options.use_gt)
            {
              dpfc_dpfw = - R_GtoC_true * R_WtoG_true;
            }
            //dht_dxtrans
            Eigen::Matrix<double,3,6> dpfc_dxtrans = Eigen::Matrix<double,3,6>::Zero();
            dpfc_dxtrans.block(0,0,3,3) = - R_GtoC_linp * skew_x(R_WtoG_linp * p_FinW);
            dpfc_dxtrans.block(0,3,3,3) = R_GtoC_linp;
            if(state->_options.use_gt)
            {
              dpfc_dxtrans.block(0,0,3,3) = - R_GtoC_true * skew_x(R_WtoG_true * p_FinW);
              dpfc_dxtrans.block(0,3,3,3) = R_GtoC_true;
            }
            //dht_dxcur
            Eigen::Matrix<double,3,6> dpfc_dxcur = Eigen::Matrix<double,3,6>::Zero();
            dpfc_dxcur.block(0,0,3,3) =  R_GtoC_linp * skew_x(R_WtoG_linp * p_FinW);;
            dpfc_dxcur.block(0,3,3,3) = - R_GtoC_linp;
            if(state->_options.use_gt)
            {
              dpfc_dxcur.block(0,0,3,3) =  R_GtoC_true * skew_x(R_WtoG_true * p_FinW);;
              dpfc_dxcur.block(0,3,3,3) = - R_GtoC_true;
            }

            Eigen::Matrix<double,3,6> dpfc_dxe=Eigen::Matrix<double,3,6>::Zero();;
           

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
            cout<<"measure feature: "<<uv_feat.transpose();
            cout<<" predict feature in curimage: "<<uv_dist.transpose()<<endl;
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
            if(state->_options.use_gt)
            {
              p_FinC_linp = p_FinC_true;
            }
            dzn_dpfc << 1/p_FinC_linp(2),0,-p_FinC_linp(0)/(p_FinC_linp(2)*p_FinC_linp(2)),
                    0, 1/p_FinC_linp(2),-p_FinC_linp(1)/(p_FinC_linp(2)*p_FinC_linp(2));


            // Precompute some matrices
            Eigen::Matrix<double,2,3> dz_dpfc = dz_dzn*dzn_dpfc;
            Eigen::Matrix<double,2,3> dz_dpfw = dz_dpfc*dpfc_dpfw;


            
            H_x.block(2*c,map_hx[clone_Cur],2,clone_Cur->size()).noalias()=dz_dpfc*dpfc_dxcur;


            H_x.block(2*c,map_hx[tranform],2,tranform->size()).noalias() = dz_dpfc * dpfc_dxtrans; 

            c++;
        }

    }

 
 }


bool UpdaterHelper::get_feature_jacobian_oc(State *state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
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

    //add related clone_IMU
    for(int i=0;i<feature.timestamps[0].size();i++)
    {
      total_meas++;
  
      PoseJPL* clone=nullptr;
      clone=state->_clones_IMU.at(feature.timestamps[0][i]);
      assert(clone!=nullptr);
      if(map_hx.find(clone) == map_hx.end()) {
      map_hx.insert({clone,total_hx});
      x_order.push_back(clone);
      total_hx += clone->size();
      }
    }
    //if there is no dealy update, it should only be 2, only observed by cuurent frame.
    assert(total_meas>=2);
    // // Also add its calibration if we are doing calibration

    

    //we also need to add transformation between map and vio
    if(state->set_transform) //normally, if we are in this function, the state->set_transform should be already set true
    {
        PoseJPL *transform = state->transform_map_to_vio;
        if(map_hx.find(transform)==map_hx.end())
        {
            map_hx.insert({transform,total_hx});
            x_order.push_back(transform);
            total_hx += transform->size();
        }
    }

    //=========================================================================
    //=========================================================================

   
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

   //!for oc, we need to perform observability constraint
   //! [dh_dpLG, dh_dRLG]* u = 0; --> A*u=0
   //! u=[(R_LG0*p_F)^, I, ]^T
   //! dh_dpI=-dh_dpLG, dh_dRI=-dh_dR_LG, dh_dpF=dh_dpLG*R_LG0

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

        //get the kf position in world frame.
        PoseJPL* anchor_kf_w = state->_clones_Keyframe[kf_id];
        Eigen::Matrix<double,3,3> R_AnchorKFtoW = anchor_kf_w->Rot();
        Eigen::Matrix<double,3,1> p_AnchorKFinW = anchor_kf_w->pos();

        //get the transform between the VIO(G) to World(W)
        PoseJPL* tranform = state->transform_map_to_vio;
        Eigen::Matrix<double,3,3> R_WtoG = state->transform_map_to_vio->Rot();
        Eigen::Matrix<double,3,1> p_WinG = state->transform_map_to_vio->pos();
        Eigen::Matrix<double,3,3> R_WtoG_linp = state->transform_map_to_vio->Rot_linp();
        Eigen::Matrix<double,3,1> p_WinG_linp = state->transform_map_to_vio->pos_linp();
        Eigen::Matrix<double,3,3> R_WtoG_fej = state->transform_map_to_vio->Rot_fej();
        Eigen::Matrix<double,3,1> p_WinG_fej = state->transform_map_to_vio->pos_fej();
        Eigen::Matrix<double,3,3> R_WtoG_true = Eigen::Matrix<double,3,3>::Identity();
        Eigen::Matrix<double,3,1> p_WinG_true = Eigen::Matrix<double,3,1>::Zero();
        // cout<<" R_WtoG: "<< R_WtoG.transpose()<<"  R_WtoG_linp:"<< R_WtoG_linp.transpose()<<endl;
        // cout<<" p_WinG: "<< p_WinG.transpose()<<"  p_WinG_linp:"<< p_WinG_linp.transpose()<<endl;


        double ts = pair.first; //ts

        cv::Point3f p_financhor = anchor_kf->point_3d_map.at(ts)[kf_feature_id];
        Vector3d p_FinAnchorKF(double(p_financhor.x),double(p_financhor.y),double(p_financhor.z));
        

        //transform the feature to map reference; we use this feature(p_FinW) in the map to do feature reprojection
        Eigen::Vector3d p_FinW =  R_AnchorKFtoW * p_FinAnchorKF + p_AnchorKFinW;

        //first we reproject p_FinW into IMU clones frame and compute the related Jacobians

        //transform p_FinW to p_FinG
        Eigen::Vector3d p_FinG = R_WtoG * p_FinW + p_WinG;
        Eigen::Vector3d p_FinG_linp = R_WtoG_linp * p_FinW + p_WinG_linp;
        Eigen::Vector3d p_FinG_true = R_WtoG_true * p_FinW + p_WinG_true;
        // cout<<"p_FinG: "<<p_FinG.transpose()<<" p_FinG_linp:"<<p_FinG_linp.transpose()<<endl;

        //transform the feature in vio reference into IMU clones camera frame
        PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
        Matrix3d R_ItoC = extrinsics->Rot_linp();
        Vector3d p_IinC = extrinsics->pos_linp();
        Matrix3d R_CtoI = R_ItoC.transpose();
        Vector3d p_CinI = -R_CtoI * p_IinC;
        

        MatrixXd u = MatrixXd::Zero(6,3);
        MatrixXd A = MatrixXd::Zero(2,6);
        //fill in u
        u.block(3,0,3,3) = skew_x(R_WtoG_fej*p_FinW);
        u.block(0,0,3,3) = Eigen::Matrix3d::Identity();

        for(int i=0;i<feature.timestamps[0].size();i++)
        {
            PoseJPL *clone_Cur = state->_clones_IMU.at(feature.timestamps[0][i]);
            Eigen::Matrix<double,17,1> state_true_cur = Eigen::Matrix<double,17,1>::Zero();
            IMU* imu_true_cur = new IMU();
            if(state->_options.use_gt)
            {
              bool success = state->get_state(feature.timestamps[0][i],state_true_cur);
              assert(success);
            }
            imu_true_cur->set_value(state_true_cur.block(1,0,16,1));

            assert(clone_Cur!=nullptr);
            Matrix3d R_GtoI = clone_Cur->Rot();
            Vector3d p_IinG = clone_Cur->pos();
            Matrix3d R_GtoI_linp = clone_Cur->Rot_linp();
            Vector3d p_IinG_linp = clone_Cur->pos_linp();
            Matrix3d R_GtoI_true = imu_true_cur->Rot();
            Vector3d p_IinG_true = imu_true_cur->pos(); 
            // assert(R_GtoI_linp==R_GtoI);
            //ht
            Eigen::Vector3d p_FinC = R_ItoC*R_GtoI*(p_FinG-p_IinG) + p_IinC;
            Eigen::Matrix3d R_GtoC = R_ItoC * R_GtoI;
            Eigen::Vector3d p_FinC_linp = R_ItoC*R_GtoI_linp*(p_FinG_linp-p_IinG_linp) + p_IinC;
            Eigen::Matrix3d R_GtoC_linp = R_ItoC * R_GtoI_linp;
            Eigen::Vector3d p_FinC_true = R_ItoC*R_GtoI_true*(p_FinG_true-p_IinG_true) + p_IinC;
            Eigen::Matrix3d R_GtoC_true = R_ItoC * R_GtoI_true;
            // cout<<"p_FinC: "<<p_FinC.transpose()<<" p_FinC_linp:"<<p_FinC_linp.transpose()<<endl;


            //dht_dpfw
            Eigen::Matrix3d dpfc_dpfw =  R_GtoC_linp * R_WtoG_linp;
            if(state->_options.use_gt)
            {
              dpfc_dpfw =  R_GtoC_true* R_WtoG_true;
            }
            //dht_dxtrans
            Eigen::Matrix<double,3,6> dpfc_dxtrans = Eigen::Matrix<double,3,6>::Zero();
            dpfc_dxtrans.block(0,0,3,3) = - R_GtoC_linp * skew_x(R_WtoG_linp * p_FinW);
            dpfc_dxtrans.block(0,3,3,3) = R_GtoC_linp;
            if(state->_options.use_gt)
            {
              dpfc_dxtrans.block(0,0,3,3) = - R_GtoC_true * skew_x(R_WtoG_true * p_FinW);
              dpfc_dxtrans.block(0,3,3,3) = R_GtoC_true;
            }
            //dht_dxcur
            Eigen::Matrix<double,3,6> dpfc_dxcur = Eigen::Matrix<double,3,6>::Zero();
            dpfc_dxcur.block(0,0,3,3) =  R_GtoC_linp * skew_x(R_WtoG_linp * p_FinW);;
            dpfc_dxcur.block(0,3,3,3) = - R_GtoC_linp;
            if(state->_options.use_gt)
            {
              dpfc_dxcur.block(0,0,3,3) =  R_GtoC_true * skew_x(R_WtoG_true * p_FinW);;
              dpfc_dxcur.block(0,3,3,3) = - R_GtoC_true;
            }

            Eigen::Matrix<double,3,6> dpfc_dxe=Eigen::Matrix<double,3,6>::Zero();;
           

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
            Eigen::Matrix<double,2,1> tmp_uv;
            tmp_uv=uv_norm_linp;
            // uv_norm_linp=uv_norm;
            // cout<<"uv_norm: "<<uv_norm.transpose()<<" uv_norm_linp:"<<uv_norm_linp.transpose()<<endl;

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
            cout<<"measure feature: "<<uv_feat.transpose();
            cout<<" predict feature in curimage: "<<uv_dist.transpose()<<endl;
           
            res.block(2*c,0,2,1) = uv_feat - uv_dist;


            // Compute Jacobians in respect to normalized image coordinates and possibly the camera intrinsics
            //dhd_dzn
            Eigen::Matrix<double,2,2> dz_dzn = Eigen::Matrix<double,2,2>::Zero();
            Eigen::Matrix<double,2,8> dz_dzeta = Eigen::Matrix<double,2,8>::Zero();
            uv_norm_linp=tmp_uv;
            if(state->_options.use_gt)
            {
              uv_norm_linp = uv_norm_true;
              
            }
            UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_linp, state->_cam_intrinsics_model.at(0), cam_d, dz_dzn, dz_dzeta);

            //Compute Jacobian of transforming feature in kf frame into nomalized uv frame;
            //dhp_dpfc
            Eigen::Matrix<double,2,3> dzn_dpfc = Eigen::Matrix<double,2,3>::Zero();
            if(state->_options.use_gt)
            {
              p_FinC_linp=p_FinC_true;
            }
            dzn_dpfc << 1/p_FinC_linp(2),0,-p_FinC_linp(0)/(p_FinC_linp(2)*p_FinC_linp(2)),
                    0, 1/p_FinC_linp(2),-p_FinC_linp(1)/(p_FinC_linp(2)*p_FinC_linp(2));


            // Precompute some matrices
            Eigen::Matrix<double,2,3> dz_dpfc = dz_dzn*dzn_dpfc;
            Eigen::Matrix<double,2,3> dz_dpfw = dz_dpfc*dpfc_dpfw;
            
            //now we use newset Jacobian to compute the optimized Jacobian that satisfy the oc
            A.block(0,0,2,6)= dz_dpfc * dpfc_dxtrans; 

            MatrixXd A_opt = A - A*u*(u.transpose()*u).llt().solve(u.transpose());
            // cout<<"A_opt: "<<endl<<A_opt<<endl;
            // cout<<"A: "<<endl<<A<<endl;

            MatrixXd dh_dxcur = Eigen::Matrix<double,2,6>::Zero();
            dh_dxcur = -A_opt;
            
            MatrixXd dh_dpF = A_opt.block(0,3,2,3)*R_WtoG_fej;
            



            // CHAINRULE: get the total feature Jacobian
            // if(!state->_options.ptmeas)
            //     H_f.block(2*c,0,2,H_f.cols()).noalias() = dz_dpfw;
            if(!state->_options.ptmeas)
                H_f.block(2*c,0,2,H_f.cols()).noalias() = dh_dpF;

           
            H_x.block(2*c,map_hx[clone_Cur],2,clone_Cur->size()).noalias()=dh_dxcur;

            H_x.block(2*c,map_hx[tranform],2,tranform->size()).noalias() = A_opt; 


            c++;

        }

        //now we finish the observation that reprojecting the 3d point in world to clone_frames;

        //compute reprojecting the 3d point in world  into the achor kf:
        Eigen::Matrix<double,9,1> kf_db = anchor_kf->_intrinsics;
        Eigen::Matrix<double,8,1> kf_d = kf_db.block(0,0,8,1);
        bool is_fisheye = int(kf_db(8,0));
        Eigen::Matrix<double,2,1> uv_norm_kf;
        uv_norm_kf<<p_FinAnchorKF(0)/p_FinAnchorKF(2),p_FinAnchorKF(1)/p_FinAnchorKF(2);
        

        Eigen::Matrix<double,2,1> uv_dist_kf;
        if(is_fisheye) {

            // Calculate distorted coordinates for fisheye
            double r = std::sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
            double theta = std::atan(r);
            double theta_d = theta+kf_d(4)*std::pow(theta,3)+kf_d(5)*std::pow(theta,5)+kf_d(6)*std::pow(theta,7)+kf_d(7)*std::pow(theta,9);

            // Handle when r is small (meaning our xy is near the camera center)
            double inv_r = (r > 1e-8)? 1.0/r : 1.0;
            double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm_kf(0)*cdist;
            double y1 = uv_norm_kf(1)*cdist;
            uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
            uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

        } else {

            // Calculate distorted coordinates for radial
            double r = std::sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
            double r_2 = r*r;
            double r_4 = r_2*r_2;
            double x1 = uv_norm_kf(0)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+2*kf_d(6)*uv_norm_kf(0)*uv_norm_kf(1)+kf_d(7)*(r_2+2*uv_norm_kf(0)*uv_norm_kf(0));
            double y1 = uv_norm_kf(1)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+kf_d(6)*(r_2+2*uv_norm_kf(1)*uv_norm_kf(1))+2*kf_d(7)*uv_norm_kf(0)*uv_norm_kf(1);
            uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
            uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);
        }

        // Our residual

        Eigen::Matrix<double,2,1> uv_m;
        uv_m << (double)anchor_kf->point_2d_uv_map.at(ts)[kf_feature_id].x, (double)anchor_kf->point_2d_uv_map.at(ts)[kf_feature_id].y;
        cout<<"measure feature: "<<uv_m.transpose();
        cout<<" predict feature in mapkf: "<<uv_dist_kf.transpose()<<endl;
        res.block(2*c,0,2,1) = uv_m - uv_dist_kf;

        Eigen::Matrix<double,2,2> dzkf_dzn = Eigen::Matrix<double,2,2>::Zero();
        Eigen::Matrix<double,2,8> dzkf_dzeta = Eigen::Matrix<double,2,8>::Zero();
        UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_kf, is_fisheye, kf_d, dzkf_dzn, dzkf_dzeta);

        Eigen::Matrix<double,2,3> dzn_dpfk = Eigen::Matrix<double,2,3>::Zero();
        dzn_dpfk << 1/p_FinAnchorKF(2),0,-p_FinAnchorKF(0)/(p_FinAnchorKF(2)*p_FinAnchorKF(2)),
                0, 1/p_FinAnchorKF(2),-p_FinAnchorKF(1)/(p_FinAnchorKF(2)*p_FinAnchorKF(2));

        Eigen::Matrix3d dpfk_dpfw = R_AnchorKFtoW.transpose();
        Eigen::Matrix<double,3,6> dpfk_dxnanchor = Eigen::Matrix<double,3,6>::Zero();
        dpfk_dxnanchor.block(0,0,3,3) = R_AnchorKFtoW.transpose() * skew_x(p_FinW); 
        dpfk_dxnanchor.block(0,3,3,3) = -R_AnchorKFtoW.transpose();

        H_n.block(2*c, map_hn[anchor_kf_w],2,anchor_kf_w->size()).noalias() = dzkf_dzn*dzn_dpfk*dpfk_dxnanchor;
        
        if(!state->_options.ptmeas)
        {
            H_f.block(2*c,0,2,H_f.cols()).noalias()=dzkf_dzn*dzn_dpfk*dpfk_dpfw;
            // Move the Jacobian and residual index forward
            
        }
        c++;

        //now compute reprojecting the 3d point in world frame to other kf
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

           Eigen::Vector3d p_FinKF = R_KFtoW.transpose()*(p_FinW-p_KFinW);
          //  cout<<"p_FinKF: "<<p_FinKF.transpose()<<" "<<p_FinKFOb.transpose()<<endl;
//           sleep(1);

           uv_norm_kf<<p_FinKF(0)/p_FinKF(2),p_FinKF(1)/p_FinKF(2);
           
           if(is_fisheye) {

            // Calculate distorted coordinates for fisheye
            double r = std::sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
            double theta = std::atan(r);
            double theta_d = theta+kf_d(4)*std::pow(theta,3)+kf_d(5)*std::pow(theta,5)+kf_d(6)*std::pow(theta,7)+kf_d(7)*std::pow(theta,9);

            // Handle when r is small (meaning our xy is near the camera center)
            double inv_r = (r > 1e-8)? 1.0/r : 1.0;
            double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm_kf(0)*cdist;
            double y1 = uv_norm_kf(1)*cdist;
            uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
            uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

            }
            else {

                // Calculate distorted coordinates for radial
                double r = std::sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
                double r_2 = r*r;
                double r_4 = r_2*r_2;
                double x1 = uv_norm_kf(0)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+2*kf_d(6)*uv_norm_kf(0)*uv_norm_kf(1)+kf_d(7)*(r_2+2*uv_norm_kf(0)*uv_norm_kf(0));
                double y1 = uv_norm_kf(1)*(1+kf_d(4)*r_2+kf_d(5)*r_4)+kf_d(6)*(r_2+2*uv_norm_kf(1)*uv_norm_kf(1))+2*kf_d(7)*uv_norm_kf(0)*uv_norm_kf(1);
                uv_dist_kf(0) = kf_d(0)*x1 + kf_d(2);
                uv_dist_kf(1) = kf_d(1)*y1 + kf_d(3);

            }

            uv_m << (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].x, (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].y;
            cout<<"measure feature: "<<uv_m.transpose();
            cout<<" predict feature in mapkf: "<<uv_dist_kf.transpose()<<endl;
            res.block(2*c,0,2,1) = uv_m - uv_dist_kf;


            dzkf_dzn = Eigen::Matrix<double,2,2>::Zero();
            dzkf_dzeta = Eigen::Matrix<double,2,8>::Zero();
            UpdaterHelper::get_feature_jacobian_intrinsics(state, uv_norm_kf, is_fisheye, kf_d, dzkf_dzn, dzkf_dzeta);

            dzn_dpfk = Eigen::Matrix<double,2,3>::Zero();
            dzn_dpfk << 1/p_FinKF(2),0,-p_FinKF(0)/(p_FinKF(2)*p_FinKF(2)),
                    0, 1/p_FinKF(2),-p_FinKF(1)/(p_FinKF(2)*p_FinKF(2));

            dpfk_dpfw = R_KFtoW.transpose();
            Matrix<double,3,6> dpfk_dxkf = Matrix<double,3,6>::Zero();
            dpfk_dxkf.block(0,0,3,3) = R_KFtoW.transpose() * skew_x(p_FinW);
            dpfk_dxkf.block(0,3,3,3) = -R_KFtoW.transpose();

            if(!state->_options.ptmeas)
                H_f.block(2*c,0,2,H_f.cols()).noalias() = dzkf_dzn*dzn_dpfk*dpfk_dpfw;

            H_n.block(2*c, map_hn[kf_w],2,kf_w->size()).noalias() = dzkf_dzn*dzn_dpfk*dpfk_dxkf;

            c++;

        }

    }
    return true;


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



