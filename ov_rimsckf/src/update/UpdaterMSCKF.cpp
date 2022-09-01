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
#include "UpdaterMSCKF.h"



using namespace ov_core;
using namespace ov_rimsckf;
using namespace std;


void UpdaterMSCKF::update_skf(State *state, std::vector<Feature *> &feature_vec) {

    cout<<"in update_skf"<<endl;
    // Return if no features
    if(feature_vec.empty())
        return;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    //
    cout<<"0"<<endl;
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }
    cout<<"1"<<endl;
    // 1. Clean all feature measurements and make sure they all have valid clone times
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature measurements that are not observed by clones (measurements timsstamp are not found in clonetimes)
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements by slide windows
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
    cout<<"2"<<endl;
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

        // Append
        clones_cam.insert({clone_calib.first,clones_cami});

    }
    cout<<"3"<<endl;
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

    cout<<"finish 3"<<endl;
    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    size_t max_meas_n_size=0;
    for(size_t i=0; i<feature_vec.size(); i++) {
        for (const auto &pair : feature_vec.at(i)->timestamps) {
            max_meas_size += 2*feature_vec.at(i)->timestamps[pair.first].size();
        }
        //keyframe linked observations. It should be zero;
        if(state->set_transform)
            max_meas_n_size+=2*feature_vec.at(i)->keyframe_matched_obs.size();
    }
    assert(max_meas_n_size==0);

    // Calculate max possible state size (i.e. the size of our covariance)
    // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
    size_t max_hx_size = state->max_covariance_size();//current state covariance size
    for(auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();  //minus the size of slam_features
    }

    size_t max_hn_size= state->max_nuisance_covariance_size();


    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size+max_meas_n_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size+max_meas_n_size, max_hx_size); //[Hx ]
    Eigen::MatrixXd Hn_big = Eigen::MatrixXd::Zero(max_meas_size+max_meas_n_size,max_hn_size); //[H_nuisance] for here, as the feas are not observed by kf, H_nuisnace = 0
    std::unordered_map<Type*,size_t> Hx_mapping;
    std::vector<Type*> Hx_order_big; //[imu,clones_imu...]
    std::vector<Type*> Hn_order_big; //[clones_kf....]
    size_t ct_jacob = 0;
    size_t ct_jacob_n = 0;
    size_t ct_meas = 0;
    cout<<"4"<<endl;
    // 4. Compute linear system for each feature, nullspace project, and reject
    auto it2 = feature_vec.begin();
    while(it2 != feature_vec.end()) {

        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;

        

        // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        feat.feat_representation = state->_options.feat_rep_msckf;
        assert(feat.feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D);
        // if(state->_options.feat_rep_msckf==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        //     feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        // }

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

//        Eigen::MatrixXd
//        UpdaterHelper::get_feature_jacobian_kf()

        // Nullspace project
        UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

        /// Chi2 distance check
        //get the covariance correspondes to the elements in Hx_order. by zzq
        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
        Eigen::MatrixXd S = H_x*P_marg*H_x.transpose();
        S.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S.rows()); //HPH^T+R
        double chi2 = res.dot(S.llt().solve(res)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
        // the result is the Mahalanobis distance

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
        if(chi2 > _options.chi2_multipler*chi2_check) {
            (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            //cout << "featid = " << feat.featid << endl;
            //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
            //cout << "res = " << endl << res.transpose() << endl;
            continue;
        }

        // We are good!!! Append to our large H vector
        size_t ct_hx = 0;
        for(const auto &var : Hx_order) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping.find(var)==Hx_mapping.end()) {
                Hx_mapping.insert({var,ct_jacob});
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas,Hx_mapping[var],H_x.rows(),var->size()) = H_x.block(0,ct_hx,H_x.rows(),var->size());
            ct_hx += var->size();

        }

        // Append our residual and move forward
        res_big.block(ct_meas,0,res.rows(),1) = res;
        ct_meas += res.rows();
        it2++;

    }
    rT3 =  boost::posix_time::microsec_clock::local_time();
     cout<<"finish 4"<<endl;
    // We have appended all features to our Hx_big, res_big
    // Delete it so we do not reuse information
    for (size_t f=0; f < feature_vec.size(); f++) {
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


     cout<<"5"<<endl;
    // 5. Perform measurement compression
    UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    Hn_big.conservativeResize(Hx_big.rows(),ct_jacob_n);

    if(Hx_big.rows() < 1) {
        return;
    }
    rT4 =  boost::posix_time::microsec_clock::local_time();


    // Our noise is isotropic, so make it here after our compression
    Eigen::MatrixXd R_big = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());
    cout<<"6"<<endl;
    // 6. With all good features update the state
    if(!state->_options.full_ekf)
        StateHelper::SKFUpdate(state, Hx_order_big, Hn_order_big, Hx_big, Hn_big, res_big, R_big,false);
    else
        StateHelper::EKFMAPUpdate(state, Hx_order_big, Hn_order_big, Hx_big, Hn_big, res_big, R_big,false);
    rT5 =  boost::posix_time::microsec_clock::local_time();


}


bool UpdaterMSCKF::update_skf_with_KF(State *state, std::vector<Feature *> &feature_vec) {
    // Return if no features
    if(feature_vec.empty())
        return false;
    std::cout<<"at the begining, feature size is"<<feature_vec.size()<<endl;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    //
    cout<<"0"<<endl;
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    cout<<"1"<<endl;
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature measurements that are not observed by clones (measurements timsstamp are not found in clonetimes)
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements by slide windows
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
    std::cout<<"2"<<endl;
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
        // Append
        clones_cam.insert({clone_calib.first,clones_cami});

    }

    // 3. Try to triangulate all MSCKF or new SLAM features that have measurements
    // here we triangulate the features only use clones cameras measurements.(without keyframe measurements)
    cout<<"3"<<endl;
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
    cout<<"after 3, feature size is:"<<feature_vec.size()<<endl;


    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    size_t max_meas_n_size = 0;
    for(size_t i=0; i<feature_vec.size(); i++) {
        for (const auto &pair : feature_vec.at(i)->timestamps) {
            max_meas_size += 2*feature_vec.at(i)->timestamps[pair.first].size();
        }
        assert(!feature_vec.at(i)->keyframe_matched_obs.empty());
        max_meas_n_size += 2*feature_vec.at(i)->keyframe_matched_obs.size();
    }
    if(feature_vec.empty()){  
        return false;
    }
    assert(max_meas_n_size>0);


    // Calculate max possible state size (i.e. the size of our covariance)
    // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
    size_t max_hx_size = state->max_covariance_size();//current state covariance size
    for(auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();  //minus the size of slam_features
    }

    size_t max_hn_size= state->max_nuisance_covariance_size();


    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size+max_meas_n_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size+max_meas_n_size, max_hx_size); //[Hx ]
    Eigen::MatrixXd Hn_big = Eigen::MatrixXd::Zero(max_meas_size+max_meas_n_size,max_hn_size); //[H_nuisance] 
    std::unordered_map<Type*,size_t> Hx_mapping;
    std::unordered_map<Type*,size_t> Hn_mapping;
    std::vector<Type*> Hx_order_big; //[imu,clones_imu...]
    std::vector<Type*> Hn_order_big; //[clones_kf....]
    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    size_t ct_jacob_n=0;

    // 4. Compute linear system for each feature, nullspace project, and reject
    cout<<"4"<<endl;
    cout<<"max_meas_n_size: "<<max_meas_n_size<<" max_hn_size: "<<max_hn_size<<endl;
    auto it2 = feature_vec.begin();
    while(it2 != feature_vec.end()) {

        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        feat.keyframe_matched_obs = (*it2)->keyframe_matched_obs;

        // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        feat.feat_representation = state->_options.feat_rep_msckf;
        if(state->_options.feat_rep_msckf==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
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
        Eigen::MatrixXd H_f_x;
        Eigen::MatrixXd H_f_n;
        Eigen::MatrixXd H_x;  //jacobian of obsevation of sliding windows
        Eigen::MatrixXd H_n;  //jacobian of observation of kf relating to nuisance part
        Eigen::MatrixXd H_n_x; //jacobian of observatin of kf relating to state
        Eigen::VectorXd res_x;
        Eigen::VectorXd res_n;
        std::vector<Type*> Hx_order;
        std::vector<Type*> Hn_order; //we use this to record the related n when compute H_n;
        std::vector<Type*> Hnx_order;//when compute H_n, it could related to x and n. we use this to record the related x


        // Get the Jacobian for this feature
        UpdaterHelper::get_feature_jacobian_full(state, feat, H_f_x, H_x, res_x, Hx_order);
        cout<<"get feature jacobian full"<<endl;
        cout<<"H_f_x.cols(): "<<H_f_x.cols()<<" H_x.cols(): "<<H_x.cols()<<endl;
        cout<<"H_f_x.rows(): "<<H_f_x.rows()<<endl;
        

        UpdaterHelper::get_feature_jacobian_kf(state,feat,H_f_n,H_n,H_n_x,res_n,Hn_order,Hnx_order);
        cout<<"get feature jacobian kf"<<endl;
        cout<<"H_f_n.cols(): "<<H_f_n.cols()<<" H_n_x.cols(): "<<H_n_x.cols()<<" H_n.cols(): "<<H_n.cols()<<endl;
        cout<<"H_f_n.rows(): "<<H_f_n.rows()<<endl;
        assert(H_n.rows()==res_n.rows());
        assert(H_n_x.rows()==res_n.rows());
        assert(H_f_n.rows()==res_n.rows());
        
        //Now we need to construct a bigger Hx(i.e H_stack), Hn and Hf to put H_x,H_n_x into H_x_stack, put H_n into H_n_stack, put H_f_x,H_f_n into Hf_stack.
        //                                       x   nuisance
        // H_stack=[H_x_stack | H_n_stack]  =  [ H_x  |     ]     Hf_stack=[H_f_x]
        //                                     [H_n_x |  H_n]              [H_f_n]
        assert(H_f_x.cols()==H_f_n.cols());
        MatrixXd Hf_stack=MatrixXd::Zero(H_f_x.rows()+H_f_n.rows(),H_f_x.cols());
        Hf_stack.block(0,0,H_f_x.rows(),H_f_x.cols())=H_f_x;
        Hf_stack.block(H_f_x.rows(),0,H_f_n.rows(),H_f_n.cols())=H_f_n;

        MatrixXd Hx_stack=MatrixXd::Zero(H_x.rows()+H_n_x.rows(),max_hx_size);
        MatrixXd Hn_stack=MatrixXd::Zero(Hx_stack.rows(),max_hn_size);

        VectorXd r_stack=VectorXd::Zero(res_x.rows()+res_n.rows());

        
        size_t ct_jacob_small=0;
        size_t ct_meas_small=0;
        size_t ct_jacob_n_small=0;
        std::unordered_map<Type*,size_t> Hx_mapping_small;
        std::unordered_map<Type*,size_t> Hn_mapping_small;
        std::vector<Type*> Hx_order_stack; //[imu,clones_imu...]
        std::vector<Type*> Hn_order_stack; //[clones_kf....]
        
        size_t ct_hx = 0;
        for(const auto &var : Hx_order) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping_small.find(var)==Hx_mapping_small.end()) {
                Hx_mapping_small.insert({var,ct_jacob_small});  //记录var在Hx_stack中的哪一列
                Hx_order_stack.push_back(var);
                ct_jacob_small += var->size();
            }

            // Append to our large Jacobian
            Hx_stack.block(ct_meas_small,Hx_mapping_small[var],H_x.rows(),var->size()) = H_x.block(0,ct_hx,H_x.rows(),var->size());
            ct_hx += var->size();
        }

        // Append our residual and move forward
        r_stack.block(ct_meas_small,0,res_x.rows(),1) = res_x;
        ct_meas_small += res_x.rows();

        //Append H_n_x into the Hx_stack
        size_t ct_hnx = 0;
        for(const auto &var : Hnx_order){
            if(Hx_mapping_small.find(var)==Hx_mapping_small.end())
            {
                Hx_mapping_small.insert({var,ct_jacob_small});
                Hx_order_stack.push_back(var);
                ct_jacob_small += var->size();
            }
            Hx_stack.block(ct_meas_small,Hx_mapping_small[var],H_n_x.rows(),var->size())= H_n_x.block(0,ct_hnx,H_n_x.rows(),var->size());
            ct_hnx += var->size();
        }

        //Append H_n into Hn_stack
        size_t ct_hn =0;
        for(const auto &var : Hn_order){
            if(Hn_mapping_small.find(var)==Hn_mapping_small.end())
            {
                Hn_mapping_small.insert({var,ct_jacob_n_small});
                Hn_order_stack.push_back(var);
                ct_jacob_n_small += var->size();
            }
            Hn_stack.block(ct_meas_small,Hn_mapping_small[var],H_n.rows(),var->size()) = H_n.block(0,ct_hn,H_n.rows(),var->size());
            ct_hn += var->size();
        }

        // Append our residual_n and move forward
        r_stack.block(ct_meas_small,0,res_n.rows(),1) = res_n;
        ct_meas_small += res_n.rows();
        
        assert(ct_meas_small==H_f_x.rows()+H_f_n.rows());
        assert(ct_jacob_small<=max_hx_size);
        assert(ct_jacob_n_small<=max_hn_size);
        Hx_stack.conservativeResize(ct_meas_small,ct_jacob_small);
        Hn_stack.conservativeResize(ct_meas_small,ct_jacob_n_small);
        r_stack.conservativeResize(ct_meas_small);

        //Now we have Hx_stack, Hn_stack, Hf_stack, perform nullspace project
        UpdaterHelper::nullspace_project_inplace_with_nuisance(Hf_stack, Hx_stack, Hn_stack, r_stack);
        cout<<"nullspace project inplace with nuisance"<<endl;
        

        // // Nullspace project
        // UpdaterHelper::nullspace_project_inplace(H_f_x, H_x, res_x);
        // cout<<"nullspace project inplace"<<endl;

        // if(max_meas_n_size>=4)
        // {
        //     UpdaterHelper::nullspace_project_inplace_with_nuisance(H_f_n, H_n_x, H_n, res_n);
        //      cout<<"nullspace project inplace with nuisance"<<endl;
        // }
        
        /// Chi2 distance check
        //get the covariance correspondes to the elements in Hx_order. by zzq
        // Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
        // cout<<"P_marg"<<endl;
        Eigen::MatrixXd P_n_marg=StateHelper::get_marginal_nuisance_covariance(state,Hn_order_stack);
         cout<<"P_n_marg"<<endl;
        Eigen::MatrixXd P_xn_marg=StateHelper::get_marginal_cross_covariance(state,Hx_order_stack,Hn_order_stack);
         cout<<"P_xn_marg"<<endl;
        Eigen::MatrixXd P_x_marg=StateHelper::get_marginal_covariance(state,Hx_order_stack);
         cout<<"P_x_marg"<<endl;
        // Eigen::MatrixXd S1 = H_x*P_marg*H_x.transpose();  //S1=HPH^T
        //  S2= [Hx,Hn]* [Pxx Pxn] * [Hx^T ]
        //               [Pnx Pnn]   [Hn^T ]
        Eigen::MatrixXd S2(Hn_stack.rows(),Hn_stack.rows());
        S2 = Hn_stack*P_n_marg*Hn_stack.transpose();
        S2 +=Hx_stack*P_x_marg*Hx_stack.transpose();
        S2 += Hx_stack*P_xn_marg*Hn_stack.transpose();
        S2 += Hn_stack*P_xn_marg.transpose()*Hx_stack.transpose();
        // S1.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S1.rows()); //HPH^T+R
        S2.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S2.rows()); 

        // double chi2_x = res_x.dot(S1.llt().solve(res_x)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
        double chi2_x = r_stack.dot(S2.llt().solve(r_stack));

        // the result is the Mahalanobis distance

        // Get our threshold (we precompute up to 500 but handle the case that it is more)
        double chi2_check_1,chi2_check_2;
        if(r_stack.rows() < 500) {
            chi2_check_1 = chi_squared_table[r_stack.rows()];
        } else {
            boost::math::chi_squared chi_squared_dist(r_stack.rows());
            chi2_check_1 = boost::math::quantile(chi_squared_dist, 0.95);
            printf(GREEN "chi2_check over the residual limit - %d\n" RESET, (int)r_stack.rows());
        }
        // if(res_n.rows() < 500) {
        //     chi2_check_2 = chi_squared_table[res_n.rows()];
        // } else {
        //     boost::math::chi_squared chi_squared_dist(res_n.rows());
        //     chi2_check_2 = boost::math::quantile(chi_squared_dist, 0.95);
        //     printf(YELLOW "chi2_check over the residual limit - %d\n" RESET, (int)res_n.rows());
        // }

        // Check if we should delete or not
        if(chi2_x > _options.chi2_multipler*chi2_check_1) {
            (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            //cout << "featid = " << feat.featid << endl;
            //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
            //cout << "res = " << endl << res.transpose() << endl;
            printf(GREEN "chi2_x is too large\n" RESET );
            continue;
        }
        // else if(chi2_n > _options.chi2_multipler*chi2_check_2){
        //     (*it2)->to_delete = true;
        //     it2 = feature_vec.erase(it2);
        //     cout<<"chi2_n is to large"<<endl;
        //     continue;
        // }


        // We are good!!! Append to our large H vector
        //                                      x       nuisance
        // H_big=[H_x_big | H_n_big]  =  [ Hx_stack  | Hn_stack    ]
        //  Note that H_x_big might have wider cols than Hx_stack, and so does H_n_big and Hn_stack

        ct_hx = 0;
        for(const auto &var : Hx_order_stack) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping.find(var)==Hx_mapping.end()) {
                Hx_mapping.insert({var,ct_jacob});  //记录var在Hx_big中的哪一列
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas,Hx_mapping[var],Hx_stack.rows(),var->size()) = Hx_stack.block(0,ct_hx,Hx_stack.rows(),var->size());
            ct_hx += var->size();

        }  

        //Append Hn_stack into Hn_big
        ct_hn =0;
        for(const auto &var : Hn_order_stack){
            if(Hn_mapping.find(var)==Hn_mapping.end())
            {
                Hn_mapping.insert({var,ct_jacob_n});
                Hn_order_big.push_back(var);
                ct_jacob_n += var->size();
            }
            Hn_big.block(ct_meas,Hn_mapping[var],Hn_stack.rows(),var->size()) = Hn_stack.block(0,ct_hn,Hn_stack.rows(),var->size());
            ct_hn += var->size();
        }

        // Append our residual_n and move forward
        res_big.block(ct_meas,0,r_stack.rows(),1) = r_stack;
        ct_meas += r_stack.rows();

        it2++;

        //save this feature into file
        assert(state->of_points.is_open());
        Vector4d q_vio=state->_imu->quat();  //q_GtoI in JPL, i.e. q_ItoG in Hamilton
        Vector3d t_vio=state->_imu->pos();
        Quaterniond q(q_vio(3),q_vio(0),q_vio(1),q_vio(2));
        Matrix3d R_ViotoMap= state->transform_vio_to_map->Rot();
        Vector3d t_VioinMap= state->transform_vio_to_map->pos();
        Quaterniond q_map;
        q_map=R_ViotoMap*q;
        Vector3d t_map=R_ViotoMap*t_vio+t_VioinMap;
        Vector3d p_FinW=R_ViotoMap*feat.p_FinG+t_VioinMap;
        //timestamp, q_vio.x q_vio.y q_vio.z q_vio.w t_vio.x t_vio.y t_vio.z q_map.x ...., feature_vio.x,y,z feature_world.x,y,z
        state->of_points<<to_string(state->_timestamp)<<" "<<q_vio(0)<<" "<<q_vio(1)<<" "<<q_vio(2)<<" "<<q_vio(3)<<
        " "<<t_vio(0)<<" "<<t_vio(1)<<" "<<t_vio(2)<<" "<<q_map.x()<<" "<<q_map.y()<<" "<<q_map.z()<<" "<<q_map.w()<<
        " "<<t_map(0)<<" "<<t_map(1)<<" "<<t_map(2)<<" "<<
        feat.p_FinG(0,0)<<" "<<feat.p_FinG(1,0)<<" "<<feat.p_FinG(2,0)<<" "<<p_FinW(0)<<" "<<p_FinW(1)<<" "<<p_FinW(2)<<
        endl;



    }
    rT3 =  boost::posix_time::microsec_clock::local_time();


    // We have appended all features to our Hx_big, Hn_big , res_big
    // Delete it so we do not reuse information
    for (size_t f=0; f < feature_vec.size(); f++) {
        feature_vec[f]->to_delete = true;
    }

    // Return if we don't have anything and resize our matrices
    if(ct_meas < 1) {
        return false;
    }
    assert(ct_meas<=max_meas_size+max_meas_n_size);
    assert(ct_jacob<=max_hx_size);
    assert(ct_jacob_n<=max_hn_size);
    res_big.conservativeResize(ct_meas,1);
    Hx_big.conservativeResize(ct_meas,ct_jacob);
    Hn_big.conservativeResize(ct_meas,ct_jacob_n);


    // 5. Perform measurement compression
    //UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    cout<<"5"<<endl;
    cout<<"Hx_big.row: "<<Hx_big.rows()<<" "<<"Hx_big col: "<<Hx_big.cols()<<" Hn_big col: "<<Hn_big.cols()<<endl;
    UpdaterHelper::measurement_compress_inplace_with_nuisance(Hx_big,Hn_big,res_big);

    //    UpdaterHelper::measurement_compress_inplace(Hx_big,res_big); 
    //    Hn_big.conservativeResize(Hx_big.rows(),ct_jacob_n); 
    
    if(Hx_big.rows() < 1) {
        return false;
    }
    rT4 =  boost::posix_time::microsec_clock::local_time();


    // Our noise is isotropic, so make it here after our compression
    Eigen::MatrixXd R_big = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());

    // 6. With all good features update the state
    std::cout<<"6"<<endl;
    StateHelper::SKFUpdate(state, Hx_order_big, Hn_order_big, Hx_big, Hn_big, res_big, R_big,false);
    rT5 =  boost::posix_time::microsec_clock::local_time();
    return true;
}



bool UpdaterMSCKF::update_skf_with_KF2(State *state, std::vector<Feature *> &feature_vec) {
    // Return if no features
    if(feature_vec.empty())
        return false;
    std::cout<<"at the begining, feature size is"<<feature_vec.size()<<endl;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    //
    cout<<"0"<<endl;
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    cout<<"1"<<endl;
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature measurements that are not observed by clones (measurements timsstamp are not found in clonetimes)
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements by slide windows
        int ct_meas = 0;
        for(const auto &pair : (*it0)->timestamps) {
            ct_meas += (*it0)->timestamps[pair.first].size();
        }

        // Remove if we don't have enough
        if(ct_meas < 1) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }

    }
    rT1 =  boost::posix_time::microsec_clock::local_time();

    // // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
    // std::cout<<"2"<<endl;
    // std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
    // for(const auto &clone_calib : state->_calib_IMUtoCAM) {

    //     // For this camera, create the vector of camera poses
    //     std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    //     for(const auto &clone_imu : state->_clones_IMU) {

    //         // Get current camera pose
    //         Eigen::Matrix<double,3,3> R_GtoCi = clone_calib.second->Rot()*clone_imu.second->Rot();
    //         Eigen::Matrix<double,3,1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose()*clone_calib.second->pos();

    //         // Append to our map
    //         clones_cami.insert({clone_imu.first,FeatureInitializer::ClonePose(R_GtoCi,p_CioinG)});

    //     }
    //     // Append
    //     clones_cam.insert({clone_calib.first,clones_cami});

    // }

    
    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    size_t max_meas_n_size = 0;
    for(size_t i=0; i<feature_vec.size(); i++) {
        for (const auto &pair : feature_vec.at(i)->timestamps) {
            max_meas_size += 4*feature_vec.at(i)->timestamps[pair.first].size();
        }
        assert(!feature_vec.at(i)->keyframe_matched_obs.empty());
        //each match provide 2 pair of reproject error (reproject into kf and current frame)
        max_meas_n_size += 4*feature_vec.at(i)->keyframe_matched_obs.size();
    }
    if(feature_vec.empty()){  
        return false;
    }
    assert(max_meas_n_size>0);
    //as for each feature in feature_vec, it only observed by current frame and loop_keyframe,  max_meas_size should equal to max_meas_n_size;
    assert(max_meas_size==max_meas_n_size);


    // Calculate max possible state size (i.e. the size of our covariance)
    // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
    size_t max_hx_size = state->max_covariance_size();//current state covariance size
    for(auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();  //minus the size of slam_features
    }

    size_t max_hn_size= state->max_nuisance_covariance_size();


    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size); //[Hx ]
    Eigen::MatrixXd Hn_big = Eigen::MatrixXd::Zero(max_meas_size,max_hn_size); //[H_nuisance] 
    std::unordered_map<Type*,size_t> Hx_mapping;
    std::unordered_map<Type*,size_t> Hn_mapping;
    std::vector<Type*> Hx_order_big; //[imu,clones_imu...]
    std::vector<Type*> Hn_order_big; //[clones_kf....]
    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    size_t ct_jacob_n=0;

    // 4. Compute linear system for each feature, nullspace project, and reject
    cout<<"4"<<endl;
    cout<<"max_meas_n_size: "<<max_meas_n_size<<" max_hn_size: "<<max_hn_size<<endl;
    auto it2 = feature_vec.begin();
    int feat_id_index=0;
    assert(state->of_points.is_open());
    state->of_points<<endl<<"timestamp: "<<to_string(state->_timestamp)<<endl;
    while(it2 != feature_vec.end()) {

        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        feat.keyframe_matched_obs = (*it2)->keyframe_matched_obs;

        // // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        // feat.feat_representation = state->_options.feat_rep_msckf;
        // if(state->_options.feat_rep_msckf==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        //     feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        // }

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f_x;
        Eigen::MatrixXd H_f_n;
        Eigen::MatrixXd H_x;  //jacobian of obsevation of sliding windows
        Eigen::MatrixXd H_n;  //jacobian of observation of kf relating to nuisance part
        Eigen::MatrixXd H_n_x; //jacobian of observatin of kf relating to state
        Eigen::VectorXd res_x;
        Eigen::VectorXd res_n;
        std::vector<Type*> Hx_order;
        std::vector<Type*> Hn_order; //we use this to record the related n when compute H_n;
        std::vector<Type*> Hnx_order;//when compute H_n, it could related to x and n. we use this to record the related x
        

        UpdaterHelper::get_feature_jacobian_kf2(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        // bool flag=false;
        // for(int i=0;i<res_n.rows();i++)
        // {
        //     if(res_n(i,0)>200)
        //     {
        //         flag=true;
        //         break;
        //     }
        // }
        // if(flag)
        // {
        //     (*it2)->to_delete = true;
        //     it2 = feature_vec.erase(it2);
        //     feat_id_index++;
        //     continue;
        // }

        // cout<<"get feature jacobian kf"<<endl;
        // cout<<"H_f_n.cols(): "<<H_f_n.cols()<<" H_n_x.cols(): "<<H_x.cols()<<" H_n.cols(): "<<H_n.cols()<<endl;
        // cout<<"H_f_n.rows(): "<<H_f_n.rows()<<endl;
        assert(H_n.rows()==res_n.rows());
        assert(H_x.rows()==res_n.rows());
        assert(H_f_n.rows()==res_n.rows());

        UpdaterHelper::nullspace_project_inplace_with_nuisance(H_f_n, H_x, H_n, res_n);
        // cout<<"nullspace project inplace with nuisance"<<endl;


        Eigen::MatrixXd P_n_marg=StateHelper::get_marginal_nuisance_covariance(state,Hn_order);
        //  cout<<"P_n_marg"<<endl;
        Eigen::MatrixXd P_xn_marg=StateHelper::get_marginal_cross_covariance(state,Hx_order,Hn_order);
        //  cout<<"P_xn_marg"<<endl;
        Eigen::MatrixXd P_x_marg=StateHelper::get_marginal_covariance(state,Hx_order);
        //  cout<<"P_x_marg"<<endl;
        // Eigen::MatrixXd S1 = H_x*P_marg*H_x.transpose();  //S1=HPH^T
        //  S2= [Hx,Hn]* [Pxx Pxn] * [Hx^T ]
        //               [Pnx Pnn]   [Hn^T ]
        Eigen::MatrixXd S2(H_x.rows(),H_x.rows());
        S2 = H_n*P_n_marg*H_n.transpose();
        S2 +=H_x*P_x_marg*H_x.transpose();
        S2 += H_x*P_xn_marg*H_n.transpose();
        S2 += H_n*P_xn_marg.transpose()*H_x.transpose();
        // S1.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S1.rows()); //HPH^T+R
        S2.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S2.rows()); 

        // double chi2_x = res_x.dot(S1.llt().solve(res_x)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
        double chi2_x = res_n.dot(S2.llt().solve(res_n));
        // the result is the Mahalanobis distance

        // Get our threshold (we precompute up to 500 but handle the case that it is more)
        double chi2_check_1,chi2_check_2;
        if(res_n.rows() < 500) {
            chi2_check_1 = chi_squared_table[res_n.rows()];
        } else {
            boost::math::chi_squared chi_squared_dist(res_n.rows());
            chi2_check_1 = boost::math::quantile(chi_squared_dist, 0.95);
            printf(GREEN "chi2_check over the residual limit - %d\n" RESET, (int)res_n.rows());
        }
        // Check if we should delete or not
        if(chi2_x > _options.chi2_multipler*chi2_check_1) {

           (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            feat_id_index++;
            //cout << "featid = " << feat.featid << endl;
            //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
            //cout << "res = " << endl << res.transpose() << endl;
            printf(GREEN "chi2_x is too large\n" );
            continue;
        }


        //Now we need to put H_x,H_n into H_x_big,H_n_big for each feat
        //                             x    nuisance
        //                           [... | ...] 
        //H_big=[H_x_big | H_n_big ]=[H_x | H_n]
        //                           [... | ...]
        
        size_t ct_hx = 0;
        for(const auto &var : Hx_order) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping.find(var)==Hx_mapping.end()) {
                Hx_mapping.insert({var,ct_jacob});  //记录var在Hx_big中的哪一列
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas,Hx_mapping[var],H_x.rows(),var->size()) = H_x.block(0,ct_hx,H_x.rows(),var->size());
            ct_hx += var->size();
        }

        size_t ct_hn =0;
        for(const auto &var : Hn_order){
            if(Hn_mapping.find(var)==Hn_mapping.end())
            {
                Hn_mapping.insert({var,ct_jacob_n});
                Hn_order_big.push_back(var);
                ct_jacob_n += var->size();
            }
            Hn_big.block(ct_meas,Hn_mapping[var],H_n.rows(),var->size()) = H_n.block(0,ct_hn,H_n.rows(),var->size());
            ct_hn += var->size();
        }

        // Append our residual_n and move forward
        res_big.block(ct_meas,0,res_n.rows(),1) = res_n;
        ct_meas += res_n.rows();


        it2++;

        // //save this feature into file
        // assert(state->of_points.is_open());
        // Vector4d q_vio=state->_imu->quat();  //q_GtoI in JPL, i.e. q_ItoG in Hamilton
        // Vector3d t_vio=state->_imu->pos();
        // Quaterniond q(q_vio(3),q_vio(0),q_vio(1),q_vio(2));
        // Matrix3d R_ViotoMap= state->transform_vio_to_map->Rot();
        // Vector3d t_VioinMap= state->transform_vio_to_map->pos();
        // Quaterniond q_map;
        // q_map=R_ViotoMap*q;
        // Vector3d t_map=R_ViotoMap*t_vio+t_VioinMap;
        // assert(feat.keyframe_matched_obs.size()==1);
        // Keyframe* keyframe=nullptr;
        // for(auto match : feat.keyframe_matched_obs)
        // {
        //     for(auto kf: match.second)
        //     {
        //         keyframe=state->get_kfdataBase()->get_keyframe(kf.first);
        //     }
        //     assert(keyframe!=nullptr);
        // }
        // cv::Point3f p_fink=keyframe->point_3d[feat_id_index];
        // assert(keyframe->matched_point_2d_uv[feat_id_index].x==feat.uvs[0].at(0)(0));//feat.uvs should only be observed by cam0 once 
        // assert(keyframe->matched_point_2d_uv[feat_id_index].y==feat.uvs[0].at(0)(1));
        // Vector3d p_FinK(p_fink.x,p_fink.y,p_fink.z);
        // Vector3d p_FinW=keyframe->_Pose_KFinWorld->Rot()*p_FinK+keyframe->_Pose_KFinWorld->pos();
        // Vector3d p_FinVio=R_ViotoMap.transpose()*(p_FinW-t_VioinMap);
        // //timestamp, q_vio.x q_vio.y q_vio.z q_vio.w t_vio.x t_vio.y t_vio.z q_map.x ...., feature_vio.x,y,z feature_world.x,y,z
        // state->of_points<<to_string(state->_timestamp)<<" "<<q_vio(0)<<" "<<q_vio(1)<<" "<<q_vio(2)<<" "<<q_vio(3)<<
        // " "<<t_vio(0)<<" "<<t_vio(1)<<" "<<t_vio(2)<<" "<<q_map.x()<<" "<<q_map.y()<<" "<<q_map.z()<<" "<<q_map.w()<<
        // " "<<t_map(0)<<" "<<t_map(1)<<" "<<t_map(2)<<" "<<
        // p_FinVio(0,0)<<" "<<p_FinVio(1,0)<<" "<<p_FinVio(2,0)<<" "<<p_FinW(0)<<" "<<p_FinW(1)<<" "<<p_FinW(2)<<
        // endl;

        feat_id_index++;



    }
    rT3 =  boost::posix_time::microsec_clock::local_time();


    // We have appended all features to our Hx_big, Hn_big , res_big
    // Delete it so we do not reuse information
    for (size_t f=0; f < feature_vec.size(); f++) {
        feature_vec[f]->to_delete = true;
    }

    // Return if we don't have anything and resize our matrices
    if(ct_meas < 1) {
        return false;
    }
    assert(ct_meas<=max_meas_size);
    assert(ct_jacob<=max_hx_size);
    assert(ct_jacob_n<=max_hn_size);
    res_big.conservativeResize(ct_meas,1);
    Hx_big.conservativeResize(ct_meas,ct_jacob);
    Hn_big.conservativeResize(ct_meas,ct_jacob_n);


    // 5. Perform measurement compression
    //UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    cout<<"5"<<endl;
    cout<<"Hx_big.row: "<<Hx_big.rows()<<" "<<"Hx_big col: "<<Hx_big.cols()<<" Hn_big col: "<<Hn_big.cols()<<endl;
    UpdaterHelper::measurement_compress_inplace_with_nuisance(Hx_big,Hn_big,res_big);

    //    UpdaterHelper::measurement_compress_inplace(Hx_big,res_big); 
    //    Hn_big.conservativeResize(Hx_big.rows(),ct_jacob_n); 
    
    if(Hx_big.rows() < 1) {
        return false;
    }
    rT4 =  boost::posix_time::microsec_clock::local_time();


    // Our noise is isotropic, so make it here after our compression
    Eigen::MatrixXd R_big = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());

    // 6. With all good features update the state
    std::cout<<"6"<<endl;
    StateHelper::SKFUpdate(state, Hx_order_big, Hn_order_big, Hx_big, Hn_big, res_big, R_big,false);
    rT5 =  boost::posix_time::microsec_clock::local_time();
    return true;
}


bool UpdaterMSCKF::update_skf_with_KF3(State *state, std::vector<Feature *> &feature_vec, bool iterative) {
    // Return if no features
    if(feature_vec.empty())
        return false;
    std::cout<<"in rimsckf update_skf_with_KF3, at the begining, feature size is"<<feature_vec.size()<<endl;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    //
    cout<<"0"<<endl;
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    cout<<"1"<<endl;
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature measurements that are not observed by clones (measurements timsstamp are not found in clonetimes)
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements by slide windows
        int ct_meas = 0;
        for(const auto &pair : (*it0)->timestamps) {
            ct_meas += (*it0)->timestamps[pair.first].size();
        }

        // Remove if we don't have enough
        if(ct_meas < 1) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }

    }
    rT1 =  boost::posix_time::microsec_clock::local_time();
    std::cout<<"after clean old measurement, feature size is"<<feature_vec.size()<<endl;

    // // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
    // std::cout<<"2"<<endl;
    // std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
    // for(const auto &clone_calib : state->_calib_IMUtoCAM) {

    //     // For this camera, create the vector of camera poses
    //     std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    //     for(const auto &clone_imu : state->_clones_IMU) {

    //         // Get current camera pose
    //         Eigen::Matrix<double,3,3> R_GtoCi = clone_calib.second->Rot()*clone_imu.second->Rot();
    //         Eigen::Matrix<double,3,1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose()*clone_calib.second->pos();

    //         // Append to our map
    //         clones_cami.insert({clone_imu.first,FeatureInitializer::ClonePose(R_GtoCi,p_CioinG)});

    //     }
    //     // Append
    //     clones_cam.insert({clone_calib.first,clones_cami});

    // }

    
    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    for(size_t i=0; i<feature_vec.size(); i++) {
        assert(!feature_vec.at(i)->keyframe_matched_obs.empty());
        for(auto match: feature_vec.at(i)->keyframe_matched_obs)
        {
            max_meas_size += 2*match.second.size();  //for each feature,we have  a 3d point represented in a match kf, and the 3d point could be projected to each of matched kfframe
        }
        max_meas_size += 2*feature_vec.at(i)->uvs[0].size(); //for each feature, it could be projected into curframe;
        
    }
    if(feature_vec.empty()){  
        return false;
    }
    assert(max_meas_size>0);
    

    // Calculate max possible state size (i.e. the size of our covariance)
    // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
    size_t max_hx_size = state->max_covariance_size();//current state covariance size
    for(auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();  //minus the size of slam_features
    }

    size_t max_hn_size= state->max_nuisance_covariance_size();


    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd noise_big = Eigen::MatrixXd::Zero(max_meas_size,max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size); //[Hx ]
    Eigen::MatrixXd Hn_big = Eigen::MatrixXd::Zero(max_meas_size,max_hn_size); //[H_nuisance] 
    Eigen::VectorXd res_big_last = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big_last = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size); //[Hx ]
    Eigen::MatrixXd Hn_big_last = Eigen::MatrixXd::Zero(max_meas_size,max_hn_size); //[H_nuisance] 
    std::unordered_map<Type*,size_t> Hx_mapping;
    std::unordered_map<Type*,size_t> Hn_mapping;
    std::vector<Type*> Hx_order_big; //[imu,clones_imu...]
    std::vector<Type*> Hn_order_big; //[clones_kf....]
    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    size_t ct_jacob_n=0;

    // 4. Compute linear system for each feature, nullspace project, and reject
    cout<<"4"<<endl;
    auto it2 = feature_vec.begin();
    int feat_id_index=0;
    // assert(state->of_points.is_open());
    // state->of_points<<"in updaterMSCKFwithKF, timestamp:"<<" "<<to_string(state->_timestamp)<<" how many kf for each feature: "<<endl;
    double avg_chi2=0;
    double avg_error_1=0;
    double avg_error_2=0;
    int num_features=feature_vec.size();
    int failed_features=0;
    int valid_features=0;
    while(it2 != feature_vec.end()) {

        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        feat.keyframe_matched_obs = (*it2)->keyframe_matched_obs;
        // cout<<"observed by "<<(*it2)->keyframe_matched_obs.size()<<endl;
        // sleep(2);

        // // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        // feat.feat_representation = state->_options.feat_rep_msckf;
        // if(state->_options.feat_rep_msckf==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        //     feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        // }

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f_x;
        Eigen::MatrixXd H_f_x_last;
        Eigen::MatrixXd H_f_n;
        Eigen::MatrixXd H_f_n_last;
        Eigen::MatrixXd H_x;  //jacobian of obsevation of sliding windows
        Eigen::MatrixXd H_x_last;
        Eigen::MatrixXd H_n;  //jacobian of observation of kf relating to nuisance part
        Eigen::MatrixXd H_n_last;
        Eigen::MatrixXd H_n_x; //jacobian of observatin of kf relating to state
        Eigen::VectorXd res_x;
        Eigen::VectorXd res_n;
        Eigen::VectorXd res_n_last;
        std::vector<Type*> Hx_order;
        std::vector<Type*> Hn_order; //we use this to record the related n when compute H_n;
        std::vector<Type*> Hnx_order;//when compute H_n, it could related to x and n. we use this to record the related x
        std::vector<Type*> Hx_order_last;
        std::vector<Type*> Hn_order_last; //we use this to record the related n when compute H_n;
        std::vector<Type*> Hnx_order_last;

        // UpdaterHelper::get_feature_jacobian_kf3(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        // if(state->iter==false)
        //kf3 for invariant error, kf4 for invariant error for kfposes,srd error for pt, kf5 for standard error
        // bool success = UpdaterHelper::get_feature_jacobian_kf4(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        bool success = UpdaterHelper::get_feature_jacobian_oc(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        if(!success)
        {
          (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            failed_features++;
            feat_id_index++;
            continue;
        }

        double error_r_1=sqrt(res_n.transpose()*res_n);
        double error_r_2=error_r_1/res_n.rows();
        avg_error_1+=error_r_1;
        avg_error_2+=error_r_2;
        //  cout<<"H_x: "<<endl<<H_x<<endl<<"H_n: "<<endl<<H_n<<endl<<" H_f_n:"<<endl<<H_f_n<<endl;
        //  sleep(1);
        // if((res_n(0)>20||res_n(1)>20)&&state->_options.opt_thred>100)
        // {
        //   (*it2)->to_delete = true;
        //     it2 = feature_vec.erase(it2);
        //     feat_id_index++;
        //     continue;
        // }
        // else if(state->iter)
        // {
        //     UpdaterHelper::get_feature_jacobian_ukf(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        // }
        // if(state->iter)
        // {
        //     UpdaterHelper::get_feature_jacobian_kf3(state,feat,H_f_n_last,H_n_last,H_x_last,res_n_last,Hn_order_last,Hx_order_last);
            
        // }
    
        state->of_points<<res_n.rows()/4<<" ";

        
        


        // cout<<"get feature jacobian kf"<<endl;
        cout<<"H_f_n.cols(): "<<H_f_n.cols()<<" H_x.cols(): "<<H_x.cols()<<" H_n.cols(): "<<H_n.cols()<<endl;
        cout<<"H_f_n.rows(): "<<H_f_n.rows()<<endl;
        assert(H_n.rows()==res_n.rows());
        assert(H_x.rows()==res_n.rows());
        assert(H_f_n.rows()==res_n.rows());
        MatrixXd noise=_options.sigma_pix_sq*MatrixXd::Identity(res_n.rows(),res_n.rows());
//         if(!state->_options.ptmeas)
//         {
// //            UpdaterHelper::nullspace_project_inplace_with_nuisance(H_f_n, H_x, H_n, res_n);
//             Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(H_f_n.rows(), H_f_n.cols());
//             qr.compute(H_f_n);
//             Eigen::MatrixXd Q = qr.householderQ();
//             Eigen::MatrixXd Q1 = Q.block(0,0,Q.rows(),3);
//             Eigen::MatrixXd Q2 = Q.block(0,3,Q.rows(),Q.cols()-3);
//             H_x=Q2.transpose()*H_x;
//             H_n=Q2.transpose()*H_n;
//             res_n=Q2.transpose()*res_n;
//             noise=Q2.transpose()*noise*Q2;
//         }
        if(!state->_options.ptmeas)
            UpdaterHelper::nullspace_project_inplace_with_nuisance_noise(H_f_n, H_x, H_n,noise, res_n);
        

        // if(state->iter)
        //     UpdaterHelper::nullspace_project_inplace_with_nuisance(H_f_n_last, H_x_last, H_n_last, res_n_last);
        // cout<<"nullspace project inplace with nuisance"<<endl;


        Eigen::MatrixXd P_n_marg=StateHelper::get_marginal_nuisance_covariance(state,Hn_order);
        //  cout<<"P_n_marg"<<endl;
        Eigen::MatrixXd P_xn_marg=StateHelper::get_marginal_cross_covariance(state,Hx_order,Hn_order);
        //  cout<<"P_xn_marg"<<endl;
        Eigen::MatrixXd P_x_marg=StateHelper::get_marginal_covariance(state,Hx_order);
        //  cout<<"P_x_marg"<<endl;
        // Eigen::MatrixXd S1 = H_x*P_marg*H_x.transpose();  //S1=HPH^T
        //  S2= [Hx,Hn]* [Pxx Pxn] * [Hx^T ]
        //               [Pnx Pnn]   [Hn^T ]
        Eigen::MatrixXd S2(H_x.rows(),H_x.rows());
        S2 = H_n*P_n_marg*H_n.transpose();
        S2 +=H_x*P_x_marg*H_x.transpose();
        S2 += H_x*P_xn_marg*H_n.transpose();
        S2 += H_n*P_xn_marg.transpose()*H_x.transpose();
        // S1.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S1.rows()); //HPH^T+R
        S2.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S2.rows()); 

        // double chi2_x = res_x.dot(S1.llt().solve(res_x)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
        double chi2_x = res_n.dot(S2.llt().solve(res_n));
        // the result is the Mahalanobis distance

        // Get our threshold (we precompute up to 500 but handle the case that it is more)
        double chi2_check_1,chi2_check_2;
        if(res_n.rows() < 500) {
            chi2_check_1 = chi_squared_table[res_n.rows()];
        } else {
            boost::math::chi_squared chi_squared_dist(res_n.rows());
            chi2_check_1 = boost::math::quantile(chi_squared_dist, 0.95);
            printf(GREEN "chi2_check over the residual limit - %d\n" RESET, (int)res_n.rows());
        }
        // Check if we should delete or not
        cout<<"chi2 vs. chi2check: "<<chi2_x<<" vs. "<<_options.chi2_multipler*chi2_check_1<<endl;
        cout<<"observed by "<<(*it2)->keyframe_matched_obs.at(state->_timestamp_approx).size()<<endl;
        cout<<"res: "<<res_n.norm()/res_n.rows()<<endl;
        // sleep(1);
        if(chi2_x > _options.chi2_multipler*chi2_check_1) {

           (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            feat_id_index++;
            //cout << "featid = " << feat.featid << endl;
            //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
            //cout << "res = " << endl << res.transpose() << endl;
            printf(GREEN "global observation chi2_x is too large\n" RESET );
            // sleep(3);
            continue;
        }




        // if(res_n.norm()/res_n.rows()>20&&state->_options.opt_thred>100)
        // {
        //   (*it2)->to_delete = true;
        //     it2 = feature_vec.erase(it2);
        //     feat_id_index++;
        //     continue;
        // }

        

        // assert(state->of_points.is_open());
        // state->of_points<<chi2_x<<" ";
        avg_chi2+=chi2_x;



        //Now we need to put H_x,H_n into H_x_big,H_n_big for each feat
        //                             x    nuisance
        //                           [... | ...] 
        //H_big=[H_x_big | H_n_big ]=[H_x | H_n]
        //                           [... | ...]
        
        size_t ct_hx = 0;
        for(const auto &var : Hx_order) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping.find(var)==Hx_mapping.end()) {
                Hx_mapping.insert({var,ct_jacob});  //记录var在Hx_big中的哪一列
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas,Hx_mapping[var],H_x.rows(),var->size()) = H_x.block(0,ct_hx,H_x.rows(),var->size());
            // if(state->iter)
            // Hx_big_last.block(ct_meas,Hx_mapping[var],H_x_last.rows(),var->size()) = H_x_last.block(0,ct_hx,H_x_last.rows(),var->size());
            ct_hx += var->size();
        }

        size_t ct_hn =0;
        for(const auto &var : Hn_order){
            if(Hn_mapping.find(var)==Hn_mapping.end())
            {
                Hn_mapping.insert({var,ct_jacob_n});
                Hn_order_big.push_back(var);
                ct_jacob_n += var->size();
            }
            Hn_big.block(ct_meas,Hn_mapping[var],H_n.rows(),var->size()) = H_n.block(0,ct_hn,H_n.rows(),var->size());
            // if(state->iter)
            // Hn_big_last.block(ct_meas,Hn_mapping[var],H_n_last.rows(),var->size()) = H_n.block(0,ct_hn,H_n_last.rows(),var->size());
            ct_hn += var->size();
        }

        // Append our residual_n and move forward
        res_big.block(ct_meas,0,res_n.rows(),1) = res_n;

        noise_big.block(ct_meas,ct_meas,noise.rows(),noise.cols())=noise;


        // if(state->iter)
        //     res_big_last.block(ct_meas,0,res_n_last.rows(),1) = res_n_last;
        ct_meas += res_n.rows();


        it2++;

        // //save this feature into file
        // assert(state->of_points.is_open());
        // Vector4d q_vio=state->_imu->quat();  //q_GtoI in JPL, i.e. q_ItoG in Hamilton
        // Vector3d t_vio=state->_imu->pos();
        // Quaterniond q(q_vio(3),q_vio(0),q_vio(1),q_vio(2));
        // Matrix3d R_ViotoMap= state->transform_vio_to_map->Rot();
        // Vector3d t_VioinMap= state->transform_vio_to_map->pos();
        // Quaterniond q_map;
        // q_map=R_ViotoMap*q;
        // Vector3d t_map=R_ViotoMap*t_vio+t_VioinMap;
        // assert(feat.keyframe_matched_obs.size()==1);
        // Keyframe* keyframe=nullptr;
        // for(auto match : feat.keyframe_matched_obs)
        // {
        //     for(auto kf: match.second)
        //     {
        //         keyframe=state->get_kfdataBase()->get_keyframe(kf.first);
        //     }
        //     assert(keyframe!=nullptr);
        // }
        // cv::Point3f p_fink=keyframe->point_3d[feat_id_index];
        // assert(keyframe->matched_point_2d_uv[feat_id_index].x==feat.uvs[0].at(0)(0));//feat.uvs should only be observed by cam0 once 
        // assert(keyframe->matched_point_2d_uv[feat_id_index].y==feat.uvs[0].at(0)(1));
        // Vector3d p_FinK(p_fink.x,p_fink.y,p_fink.z);
        // Vector3d p_FinW=keyframe->_Pose_KFinWorld->Rot()*p_FinK+keyframe->_Pose_KFinWorld->pos();
        // Vector3d p_FinVio=R_ViotoMap.transpose()*(p_FinW-t_VioinMap);
        // //timestamp, q_vio.x q_vio.y q_vio.z q_vio.w t_vio.x t_vio.y t_vio.z q_map.x ...., feature_vio.x,y,z feature_world.x,y,z
        // state->of_points<<to_string(state->_timestamp)<<" "<<q_vio(0)<<" "<<q_vio(1)<<" "<<q_vio(2)<<" "<<q_vio(3)<<
        // " "<<t_vio(0)<<" "<<t_vio(1)<<" "<<t_vio(2)<<" "<<q_map.x()<<" "<<q_map.y()<<" "<<q_map.z()<<" "<<q_map.w()<<
        // " "<<t_map(0)<<" "<<t_map(1)<<" "<<t_map(2)<<" "<<
        // p_FinVio(0,0)<<" "<<p_FinVio(1,0)<<" "<<p_FinVio(2,0)<<" "<<p_FinW(0)<<" "<<p_FinW(1)<<" "<<p_FinW(2)<<
        // endl;

        feat_id_index++;
        valid_features++;



    }
    // sleep(5);
    rT3 =  boost::posix_time::microsec_clock::local_time();
    avg_chi2/=double(num_features);
    avg_error_1/=double(valid_features);
    avg_error_2/=double(valid_features);
    cout<<"avg_error_1: "<<avg_error_1<<endl;
    double failed_rate = double(failed_features)/double(num_features);
    // state->of_points<<endl<<"avg chi2: "<<avg_chi2<<" avg_error_1: "<<avg_error_1<<" avg_error_2: "<<avg_error_2<<endl;
    cout<<"score: "<<avg_error_1<<endl;
    cout<<"num_feature: "<<num_features<<" failed feature: "<<failed_features<<" rate: "<<failed_rate<<endl;
    // sleep(10);
    
    


    // We have appended all features to our Hx_big, Hn_big , res_big
    // Delete it so we do not reuse information
    for(size_t f=0; f < feature_vec.size(); f++) {
        feature_vec[f]->to_delete = true;
    }

    // Return if we don't have anything and resize our matrices
    if(ct_meas < 5) {
        return false;
    }
    assert(ct_meas<=max_meas_size);
    assert(ct_jacob<=max_hx_size);
    assert(ct_jacob_n<=max_hn_size);
    res_big.conservativeResize(ct_meas,1);
    Hx_big.conservativeResize(ct_meas,ct_jacob);
    Hn_big.conservativeResize(ct_meas,ct_jacob_n);
    noise_big.conservativeResize(ct_meas,ct_meas);
    
    // if(state->iter)
    // {
    //     res_big_last.conservativeResize(ct_meas,1);
    //     Hx_big_last.conservativeResize(ct_meas,ct_jacob);
    //     Hn_big_last.conservativeResize(ct_meas,ct_jacob_n);
    // }


    // 5. Perform measurement compression
    //UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    cout<<"5"<<endl;
    cout<<"Hx_big.row: "<<Hx_big.rows()<<" "<<"Hx_big col: "<<Hx_big.cols()<<" Hn_big col: "<<Hn_big.cols()<<endl;
    // cout<<"big noise before compress: "<<endl<<noise_big<<endl;
    
   
   UpdaterHelper::measurement_compress_inplace_with_nuisance(Hx_big,Hn_big,res_big);
    // Eigen::MatrixXd H_all=MatrixXd::Zero(Hx_big.rows(),Hx_big.cols()+Hn_big.cols());
    // H_all.block(0,0,Hx_big.rows(),Hx_big.cols())=Hx_big;
    // H_all.block(0,Hx_big.cols(),Hn_big.rows(),Hn_big.cols())=Hn_big;
    // int Hx_cols=Hx_big.cols();
    // int Hn_cols=Hn_big.cols();
    // if(H_all.rows()>(H_all.cols()))
    // {
    //     Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(H_all.rows(), H_all.cols());
    //     qr.compute(H_all);
    //     Eigen::MatrixXd Q = qr.householderQ();
    //     cout<<"Q size: "<<Q.rows()<<" "<<Q.cols()<<endl;
    //     Eigen::MatrixXd Q1 = Q.block(0,0,Q.rows(),H_all.cols());
    //     Eigen::MatrixXd Q2 = Q.block(0,H_all.cols(),Q.rows(),Q.cols()-H_all.cols());
    //     H_all=Q1.transpose()*H_all;
    //     res_big=Q1.transpose()*res_big;
    //     Hx_big=H_all.block(0,0,H_all.rows(),Hx_cols);
    //     Hn_big=H_all.block(0,Hx_cols,H_all.rows(),Hn_cols);
    //     noise_big=Q1.transpose()*noise_big*Q1;
    // }

    // UpdaterHelper::measurement_compress_inplace_with_nuisance_noise(Hx_big,Hn_big,noise_big,res_big);
    // cout<<"big noise after compress: "<<endl<<noise_big<<endl;
    // sleep(10);
    // if(state->iter)
    // {
    //     UpdaterHelper::measurement_compress_inplace_with_nuisance(Hx_big_last,Hn_big_last,res_big_last);
    // }

    //    UpdaterHelper::measurement_compress_inplace(Hx_big,res_big); 
    //    Hn_big.conservativeResize(Hx_big.rows(),ct_jacob_n); 
    
    if(Hx_big.rows() < 1) {
        return false;
    }
    rT4 =  boost::posix_time::microsec_clock::local_time();

    // if(state->iter==true)   
    // {
    //     double now=(Hx_big*Hx_big.transpose()).determinant();
    //     double last=(_Hx_big_last*_Hx_big_last.transpose()).determinant();
    //     if(avg_error_2>_avg_error_2_last||now>last)  //if error not descrease, we should stop the iteration and use the last iteration information.
    //     // if(now>last)
    //     {
    //         // iterative=false;
    //         // state->iter=false;
           
    //         Hx_big=_Hx_big_last;
    //         Hn_big=_Hn_big_last;
    //         res_big=_res_big_last;
    //     }
        
        
    // }
    // else if(state->iter==false)  //if update is not iterative yet, we need to judge whether it should iterative update.
    
    // sleep(1);
    // if(state->iter_count==0)
    // {
    //     _avg_error_2_last=avg_error_2;
    // }
    
    // sleep(1);

    // if(state->iter==true)
    // {
    //     if(avg_error_2>state->_avg_error_2_last)
    //     {
            
    //         cout<<"avg_error_2 larger,stop"<<endl;
    //         for(int i=0;i<state->_variables.size();i++)
    //         {
    //             state->_variables.at(i)->set_value(state->_variables_last[i]);
    //             state->_variables.at(i)->set_linp(state->_variables_last[i]);
    //         }
    //         state->_Cov=state->_Cov_last;
    //         state->_Cross_Cov_AN=state->_Cross_Cov_AN_last;
    //         state->iter=false;
    //         return true;
    //     }
    //     else
    //     {
    //         state->_avg_error_2_last=avg_error_2;
    //     }
    // }
    if(state->iter==false)
    {
        if((avg_error_1>state->_options.opt_thred||failed_rate>0.1)&&state->_options.opt_thred<100)
        {
            state->iter=true;
            state->iter_count=0;
            // iterative=true;
            // _avg_error_2_last=avg_error_2;        get_feature_jacobian_kf4
            // _Hx_big_last=Hx_big;
            // _Hn_big_last=Hn_big;
            // _res_big_last=res_big;
            // state->_avg_error_2_last=avg_error_2;
            // state->_Cov_iter=state->_Cov;
            // state->_Cross_Cov_AN_iter=state->_Cross_Cov_AN;
            return false; //return false when using optimize;
        }
    }



   

    // Our noise is isotropic, so make it here after our compression
  //  Eigen::MatrixXd R_big = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());
  Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());

  //  cout<<"Hx_big: "<<endl<<Hx_big<<endl<<"Hn_big: "<<endl<<Hn_big<<endl;
    //   Eigen::MatrixXd R_big=noise_big;
    // 6. With all good features update the state
    std::cout<<"6"<<endl;
    if(state->_options.full_ekf)
        StateHelper::EKFMAPUpdate(state, Hx_order_big, Hn_order_big, Hx_big, Hn_big, res_big, R_big,iterative);
    else
    {
        
        StateHelper::SKFUpdate(state, Hx_order_big, Hn_order_big, Hx_big, Hn_big, res_big, R_big,iterative);
    }

    

        
    rT5 =  boost::posix_time::microsec_clock::local_time();
    return true;
}


bool UpdaterMSCKF::update_skf_with_KF4(State *state, std::vector<Feature *> &feature_vec, bool iterative) {
    // Return if no features
    if(feature_vec.empty())
        return false;
    std::cout<<"at the begining, feature size is"<<feature_vec.size()<<endl;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    //
    cout<<"0"<<endl;
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    cout<<"1"<<endl;
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature measurements that are not observed by clones (measurements timsstamp are not found in clonetimes)
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements by slide windows
        int ct_meas = 0;
        for(const auto &pair : (*it0)->timestamps) {
            ct_meas += (*it0)->timestamps[pair.first].size();
        }

        // Remove if we don't have enough
        if(ct_meas < 1) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }

    }
    rT1 =  boost::posix_time::microsec_clock::local_time();

    // // 2. Create vector of cloned *CAMERA* poses at each of our clone timesteps
    // std::cout<<"2"<<endl;
    // std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
    // for(const auto &clone_calib : state->_calib_IMUtoCAM) {

    //     // For this camera, create the vector of camera poses
    //     std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
    //     for(const auto &clone_imu : state->_clones_IMU) {

    //         // Get current camera pose
    //         Eigen::Matrix<double,3,3> R_GtoCi = clone_calib.second->Rot()*clone_imu.second->Rot();
    //         Eigen::Matrix<double,3,1> p_CioinG = clone_imu.second->pos() - R_GtoCi.transpose()*clone_calib.second->pos();

    //         // Append to our map
    //         clones_cami.insert({clone_imu.first,FeatureInitializer::ClonePose(R_GtoCi,p_CioinG)});

    //     }
    //     // Append
    //     clones_cam.insert({clone_calib.first,clones_cami});

    // }

    
    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    for(size_t i=0; i<feature_vec.size(); i++) {
        assert(!feature_vec.at(i)->keyframe_matched_obs.empty());
        max_meas_size += 4*feature_vec.at(i)->keyframe_matched_obs.size();
    }
    if(feature_vec.empty()){  
        return false;
    }
    assert(max_meas_size>0);
    

    // Calculate max possible state size (i.e. the size of our covariance)
    // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
    size_t max_hx_size = state->max_covariance_size();//current state covariance size
    for(auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();  //minus the size of slam_features
    }

    size_t max_hn_size= state->max_nuisance_covariance_size();


    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size); //[Hx ]
    Eigen::MatrixXd Hn_big = Eigen::MatrixXd::Zero(max_meas_size,max_hn_size); //[H_nuisance] 
    std::unordered_map<Type*,size_t> Hx_mapping;
    std::unordered_map<Type*,size_t> Hn_mapping;
    std::vector<Type*> Hx_order_big; //[imu,clones_imu...]
    std::vector<Type*> Hn_order_big; //[clones_kf....]
    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    size_t ct_jacob_n=0;

    // 4. Compute linear system for each feature, nullspace project, and reject
    cout<<"4"<<endl;
    auto it2 = feature_vec.begin();
    int feat_id_index=0;
    assert(state->of_points.is_open());
    state->of_points<<"in updaterMSCKFwithKF, timestamp:"<<" "<<to_string(state->_timestamp)<<endl;
    double avg_chi2=0;
    double avg_error_1=0;
    double avg_error_2=0;
    while(it2 != feature_vec.end()) {

        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        feat.keyframe_matched_obs = (*it2)->keyframe_matched_obs;

        // // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        // feat.feat_representation = state->_options.feat_rep_msckf;
        // if(state->_options.feat_rep_msckf==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        //     feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        // }

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f_x;
        Eigen::MatrixXd H_f_n;
        Eigen::MatrixXd H_x;  //jacobian of obsevation of sliding windows
        Eigen::MatrixXd H_n;  //jacobian of observation of kf relating to nuisance part
        Eigen::MatrixXd H_n_x; //jacobian of observatin of kf relating to state
        Eigen::VectorXd res_x;
        Eigen::VectorXd res_n;
        std::vector<Type*> Hx_order;
        std::vector<Type*> Hn_order; //we use this to record the related n when compute H_n;
        std::vector<Type*> Hnx_order;//when compute H_n, it could related to x and n. we use this to record the related x
        

        // UpdaterHelper::get_feature_jacobian_kf3(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        UpdaterHelper::get_feature_jacobian_kf5(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        state->of_points<<res_n.rows()/4<<" ";

        
        double error_r_1=sqrt(res_n.transpose()*res_n);
        double error_r_2=error_r_1/res_n.rows();
        avg_error_1+=error_r_1;
        avg_error_2+=error_r_2;
        // bool flag=false;
        // for(int i=0;i<res_n.rows();i++)
        // {
        //     if(res_n(i,0)>200)
        //     {
        //         flag=true;
        //         break;
        //     }
        // }
        // if(flag)
        // {
        //     (*it2)->to_delete = true;
        //     it2 = feature_vec.erase(it2);
        //     feat_id_index++;
        //     continue;
        // }

        // cout<<"get feature jacobian kf"<<endl;
        cout<<"H_f_n.cols(): "<<H_f_n.cols()<<" H_x.cols(): "<<H_x.cols()<<" H_n.cols(): "<<H_n.cols()<<endl;
        cout<<"H_f_n.rows(): "<<H_f_n.rows()<<endl;
        assert(H_n.rows()==res_n.rows());
        assert(H_x.rows()==res_n.rows());
        assert(H_f_n.rows()==res_n.rows());

        UpdaterHelper::nullspace_project_inplace_with_nuisance(H_f_n, H_x, H_n, res_n);
        // cout<<"nullspace project inplace with nuisance"<<endl;


        Eigen::MatrixXd P_n_marg=StateHelper::get_marginal_nuisance_covariance(state,Hn_order);
        //  cout<<"P_n_marg"<<endl;
        Eigen::MatrixXd P_xn_marg=StateHelper::get_marginal_cross_covariance(state,Hx_order,Hn_order);
        //  cout<<"P_xn_marg"<<endl;
        Eigen::MatrixXd P_x_marg=StateHelper::get_marginal_covariance(state,Hx_order);
        //  cout<<"P_x_marg"<<endl;
        // Eigen::MatrixXd S1 = H_x*P_marg*H_x.transpose();  //S1=HPH^T
        //  S2= [Hx,Hn]* [Pxx Pxn] * [Hx^T ]
        //               [Pnx Pnn]   [Hn^T ]
        Eigen::MatrixXd S2(H_x.rows(),H_x.rows());
        S2 = H_n*P_n_marg*H_n.transpose();
        S2 +=H_x*P_x_marg*H_x.transpose();
        S2 += H_x*P_xn_marg*H_n.transpose();
        S2 += H_n*P_xn_marg.transpose()*H_x.transpose();
        // S1.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S1.rows()); //HPH^T+R
        S2.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S2.rows()); 

        // double chi2_x = res_x.dot(S1.llt().solve(res_x)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
        double chi2_x = res_n.dot(S2.llt().solve(res_n));
        // the result is the Mahalanobis distance

        // Get our threshold (we precompute up to 500 but handle the case that it is more)
        double chi2_check_1,chi2_check_2;
        if(res_n.rows() < 500) {
            chi2_check_1 = chi_squared_table[res_n.rows()];
        } else {
            boost::math::chi_squared chi_squared_dist(res_n.rows());
            chi2_check_1 = boost::math::quantile(chi_squared_dist, 0.95);
            printf(GREEN "chi2_check over the residual limit - %d\n" RESET, (int)res_n.rows());
        }
        // Check if we should delete or not
        if(chi2_x > _options.chi2_multipler*chi2_check_1) {

           (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            feat_id_index++;
            //cout << "featid = " << feat.featid << endl;
            //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
            //cout << "res = " << endl << res.transpose() << endl;
            printf(GREEN "chi2_x is too large\n" );
            continue;
        }

        

        // assert(state->of_points.is_open());
        // state->of_points<<chi2_x<<" ";
        avg_chi2+=chi2_x;


        //Now we need to put H_x,H_n into H_x_big,H_n_big for each feat
        //                             x    nuisance
        //                           [... | ...] 
        //H_big=[H_x_big | H_n_big ]=[H_x | H_n]
        //                           [... | ...]
        
        size_t ct_hx = 0;
        for(const auto &var : Hx_order) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping.find(var)==Hx_mapping.end()) {
                Hx_mapping.insert({var,ct_jacob});  //记录var在Hx_big中的哪一列
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas,Hx_mapping[var],H_x.rows(),var->size()) = H_x.block(0,ct_hx,H_x.rows(),var->size());
            ct_hx += var->size();
        }

        size_t ct_hn =0;
        for(const auto &var : Hn_order){
            if(Hn_mapping.find(var)==Hn_mapping.end())
            {
                Hn_mapping.insert({var,ct_jacob_n});
                Hn_order_big.push_back(var);
                ct_jacob_n += var->size();
            }
            Hn_big.block(ct_meas,Hn_mapping[var],H_n.rows(),var->size()) = H_n.block(0,ct_hn,H_n.rows(),var->size());
            ct_hn += var->size();
        }

        // Append our residual_n and move forward
        res_big.block(ct_meas,0,res_n.rows(),1) = res_n;
        ct_meas += res_n.rows();


        it2++;

        // //save this feature into file
        // assert(state->of_points.is_open());
        // Vector4d q_vio=state->_imu->quat();  //q_GtoI in JPL, i.e. q_ItoG in Hamilton
        // Vector3d t_vio=state->_imu->pos();
        // Quaterniond q(q_vio(3),q_vio(0),q_vio(1),q_vio(2));
        // Matrix3d R_ViotoMap= state->transform_vio_to_map->Rot();
        // Vector3d t_VioinMap= state->transform_vio_to_map->pos();
        // Quaterniond q_map;
        // q_map=R_ViotoMap*q;
        // Vector3d t_map=R_ViotoMap*t_vio+t_VioinMap;
        // assert(feat.keyframe_matched_obs.size()==1);
        // Keyframe* keyframe=nullptr;
        // for(auto match : feat.keyframe_matched_obs)
        // {
        //     for(auto kf: match.second)
        //     {
        //         keyframe=state->get_kfdataBase()->get_keyframe(kf.first);
        //     }
        //     assert(keyframe!=nullptr);
        // }
        // cv::Point3f p_fink=keyframe->point_3d[feat_id_index];
        // assert(keyframe->matched_point_2d_uv[feat_id_index].x==feat.uvs[0].at(0)(0));//feat.uvs should only be observed by cam0 once 
        // assert(keyframe->matched_point_2d_uv[feat_id_index].y==feat.uvs[0].at(0)(1));
        // Vector3d p_FinK(p_fink.x,p_fink.y,p_fink.z);
        // Vector3d p_FinW=keyframe->_Pose_KFinWorld->Rot()*p_FinK+keyframe->_Pose_KFinWorld->pos();
        // Vector3d p_FinVio=R_ViotoMap.transpose()*(p_FinW-t_VioinMap);
        // //timestamp, q_vio.x q_vio.y q_vio.z q_vio.w t_vio.x t_vio.y t_vio.z q_map.x ...., feature_vio.x,y,z feature_world.x,y,z
        // state->of_points<<to_string(state->_timestamp)<<" "<<q_vio(0)<<" "<<q_vio(1)<<" "<<q_vio(2)<<" "<<q_vio(3)<<
        // " "<<t_vio(0)<<" "<<t_vio(1)<<" "<<t_vio(2)<<" "<<q_map.x()<<" "<<q_map.y()<<" "<<q_map.z()<<" "<<q_map.w()<<
        // " "<<t_map(0)<<" "<<t_map(1)<<" "<<t_map(2)<<" "<<
        // p_FinVio(0,0)<<" "<<p_FinVio(1,0)<<" "<<p_FinVio(2,0)<<" "<<p_FinW(0)<<" "<<p_FinW(1)<<" "<<p_FinW(2)<<
        // endl;

        feat_id_index++;



    }
    rT3 =  boost::posix_time::microsec_clock::local_time();
    avg_chi2/=double(feature_vec.size());
    avg_error_1/=double(feature_vec.size());
    avg_error_2/=double(feature_vec.size());
    // state->of_points<<endl<<"avg chi2: "<<avg_chi2<<" avg_error_1: "<<avg_error_1<<" avg_error_2: "<<avg_error_2<<endl;
    
    
    state->of_points<<endl;


    // We have appended all features to our Hx_big, Hn_big , res_big
    // Delete it so we do not reuse information
    for(size_t f=0; f < feature_vec.size(); f++) {
        feature_vec[f]->to_delete = true;
    }

    // Return if we don't have anything and resize our matrices
    if(ct_meas < 1) {
        return false;
    }
    assert(ct_meas<=max_meas_size);
    assert(ct_jacob<=max_hx_size);
    assert(ct_jacob_n<=max_hn_size);
    res_big.conservativeResize(ct_meas,1);
    Hx_big.conservativeResize(ct_meas,ct_jacob);
    Hn_big.conservativeResize(ct_meas,ct_jacob_n);


    // 5. Perform measurement compression
    //UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    cout<<"5"<<endl;
    cout<<"Hx_big.row: "<<Hx_big.rows()<<" "<<"Hx_big col: "<<Hx_big.cols()<<" Hn_big col: "<<Hn_big.cols()<<endl;
    UpdaterHelper::measurement_compress_inplace_with_nuisance(Hx_big,Hn_big,res_big);

    //    UpdaterHelper::measurement_compress_inplace(Hx_big,res_big); 
    //    Hn_big.conservativeResize(Hx_big.rows(),ct_jacob_n); 
    
    if(Hx_big.rows() < 1) {
        return false;
    }
    rT4 =  boost::posix_time::microsec_clock::local_time();

    // if(state->iter==true)   
    // {
    //     if(avg_error_2>_avg_error_2_last)  //if error not descrease, we should stop the iteration and use the last iteration information.
    //     {
    //         iterative=false;
    //         state->iter=false;
    //         Hx_big=_Hx_big_last;
    //         Hn_big=_Hn_big_last;
    //         res_big=_res_big_last;
    //     }
    //     else   //else we should record the current information 
    //     {
    //         _Hx_big_last=Hx_big;
    //         _Hn_big_last=Hn_big;
    //         _res_big_last=res_big;
    //         _avg_error_2_last=avg_error_2;
    //     }
        
    // }
    // else if(state->iter==false)  //if update is not iterative yet, we need to judge whether it should iterative update.
    if(state->iter==false)
    {
        if(avg_error_2>state->_options.opt_thred)
        {
            state->iter=true;
            iterative=true;
            _avg_error_2_last=avg_error_2;
            _Hx_big_last=Hx_big;
            _Hn_big_last=Hn_big;
            _res_big_last=res_big;
            return false; //return false when using optimize;
        }
    }


    // Our noise is isotropic, so make it here after our compression
    double scale=1;
    if(state->iter==true)
    {
        scale=(Hx_big*Hx_big.transpose()).determinant()/(_Hx_big_last*_Hx_big_last.transpose()).determinant();
        cout<<"scale: "<<scale<<endl;
        sleep(5);
    }
    Eigen::MatrixXd R_big = scale*_options.sigma_pix_sq*Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());

    // 6. With all good features update the state
    std::cout<<"6"<<endl;
    StateHelper::SKFUpdate(state, Hx_order_big, Hn_order_big, Hx_big, Hn_big, res_big, R_big,iterative);
    rT5 =  boost::posix_time::microsec_clock::local_time();
    return true;
}



bool UpdaterMSCKF::update_skf_with_KF6(State *state, vector<Keyframe*> kfs,int anchor_id, bool iterative, Matrix3d R_kf_cur,
                                 Vector3d t_kf_cur, int flag){

    //in this case, we use R_kf_cur,t_kf_cur as observation, 
    //use Pose_KFinW, tranform_world_vio, Pose_CurinVins to compute the prediction.
    // r should be 6*1, Hx should be 6*(6+6), Hn should be 6*6
    //there is the case that kfs are more than one.

    int num_kf=kfs.size();
    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(6*num_kf);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(6*num_kf, 12); //[Hx ]
    if(flag!=2)
    {
        Hx_big.conservativeResize(6*num_kf,6);
    }
    Eigen::MatrixXd Hn_big = Eigen::MatrixXd::Zero(6*num_kf, 6*num_kf); //[H_nuisance] 
    std::unordered_map<Type*,size_t> Hx_mapping;
    std::unordered_map<Type*,size_t> Hn_mapping;
    std::vector<Type*> Hx_order_big; //[imu,clones_imu...]
    std::vector<Type*> Hn_order_big; //[clones_kf....]
    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    size_t ct_jacob_n=0;

    

    
    for(int i=0;i<num_kf;i++)
    {
       Keyframe* kf=kfs[i];
       assert(kf!=nullptr);
       PoseJPL* kf_pose=state->_clones_Keyframe[kf->time_stamp];
       if(Hn_mapping.find(kf_pose)==Hn_mapping.end()){
           Hn_mapping.insert({kf_pose,ct_jacob_n});
           Hn_order_big.push_back(kf_pose);
           ct_jacob_n+=kf_pose->size();
       }
    }

    if(flag==1||flag==2)
    {
        PoseJPL *clone_Cur=nullptr;
        clone_Cur = state->_clones_IMU.at(state->_timestamp);
        assert(clone_Cur!=nullptr);
        if(Hx_mapping.find(clone_Cur) == Hx_mapping.end()) {
        Hx_mapping.insert({clone_Cur,ct_jacob});
        Hx_order_big.push_back(clone_Cur);
        ct_jacob += clone_Cur->size();
        }
    }
    


    if(flag==0||flag==2)
    {
        PoseJPL *transform = state->transform_vio_to_map;
        if(Hx_mapping.find(transform)==Hx_mapping.end())
        {
            Hx_mapping.insert({transform,ct_jacob});
            Hx_order_big.push_back(transform);
            ct_jacob += transform->size();
        }
    }
   
    int c=0;
    for(int i=0;i<num_kf;i++)
    {
         //begin our observation function and jacobians
        Keyframe* kf=kfs[i];
        assert(kf!=nullptr);
        PoseJPL* kf_pose=state->_clones_Keyframe[kf->time_stamp];
        Matrix3d R_W_KF=kf->_Pose_KFinWorld->Rot();
        Vector3d t_W_KF=kf->_Pose_KFinWorld->pos();

        PoseJPL *transform = state->transform_vio_to_map;
        Matrix3d R_W_VIO=state->transform_vio_to_map->Rot();
        Vector3d t_W_VIO=state->transform_vio_to_map->pos();

        PoseJPL *clone_Cur=nullptr;
        clone_Cur = state->_clones_IMU.at(state->_timestamp);
        assert(clone_Cur!=nullptr);
        Matrix3d R_I_VIO=state->_clones_IMU.at(state->_timestamp)->Rot();
        Vector3d t_VIO_I=state->_clones_IMU.at(state->_timestamp)->pos();

        Matrix3d R_C_I=state->_calib_IMUtoCAM.at(0)->Rot();
        Vector3d t_C_I=state->_calib_IMUtoCAM.at(0)->pos();

        //h1, the observation function of rotation
        Matrix3d R_KF_C=R_W_KF.transpose()*R_W_VIO*R_I_VIO.transpose()*R_C_I.transpose();
        Vector3d t_VIO_C=t_VIO_I-R_I_VIO.transpose()*R_C_I.transpose()*t_C_I;
        Vector3d t_W_C=t_W_VIO+R_W_VIO*t_VIO_C;
        //h2, the observation function of pretranslation
        Vector3d t_KF_C=R_W_KF.transpose()*(t_W_C-t_W_KF);

        Matrix3d R_KF_I=R_W_KF.transpose()*R_W_VIO*R_I_VIO.transpose();

        // dh1_dRIV
        Matrix3d dh1_dRIV=Jl_so3_inv(log_so3(R_KF_I*R_C_I.transpose()))*R_KF_I;
        //dh1_dtVI
        Matrix3d dh1_dtVI=Matrix3d::Zero();
        //dh2_dRIV
        Matrix3d dh2_dRIV=R_KF_I*skew_x(R_C_I.transpose()*t_C_I);
        //dh2_dtVI
        Matrix3d dh2_dtVI=R_W_KF.transpose()*R_W_VIO;

        if(flag==1||flag==2)
        {
            Hx_big.block(6*c+0,Hx_mapping[clone_Cur],3,3)=dh1_dRIV;
            Hx_big.block(6*c+0,Hx_mapping[clone_Cur]+3,3,3)=dh1_dtVI;
            Hx_big.block(6*c+3,Hx_mapping[clone_Cur],3,3)=dh2_dRIV;
            Hx_big.block(6*c+3,Hx_mapping[clone_Cur]+3,3,3)=dh2_dtVI;
        }
        


        //dh1_dRWV
        Matrix3d dh1_dRWV=Jl_so3_inv(log_so3(R_KF_I*R_C_I.transpose()))*R_W_KF.transpose();
        //dh1_dtWV;
        Matrix3d dh1_dtWV=Matrix3d::Zero();
        //dh2_dRWV
        Matrix3d dh2_dRWV=R_W_KF.transpose()*skew_x(R_W_VIO*t_VIO_C);
        //dh2_dtWV
        Matrix3d dh2_dtWV=R_W_KF.transpose();
        

        if(flag==0||flag==2)
        {
            Hx_big.block(6*c+0,Hx_mapping[transform],3,3)=dh1_dRWV;
            Hx_big.block(6*c+0,Hx_mapping[transform]+3,3,3)=dh1_dtWV;
            Hx_big.block(6*c+3,Hx_mapping[transform],3,3)=dh2_dRWV;
            Hx_big.block(6*c+3,Hx_mapping[transform]+3,3,3)=dh2_dtWV;
        }

        
        //dh1_dRWKF
        Matrix3d dh1_dRWKF=-Jl_so3_inv(log_so3(R_KF_I*R_C_I.transpose()))*R_W_KF.transpose();
        //dh1_dtWKF
        Matrix3d dh1_dtWKF=Matrix3d::Zero();
        //dh2_dRWKF
        Matrix3d dh2_dRWKF=-R_W_KF.transpose()*skew_x(t_W_C-t_W_KF);
        //dh2_dtWKF
        Matrix3d dh2_dtWKF=-R_W_KF.transpose();

        Hn_big.block(6*c+0,Hn_mapping[kf_pose],3,3)=dh1_dRWKF;
        Hn_big.block(6*c+0,Hn_mapping[kf_pose]+3,3,3)=dh1_dtWKF;
        Hn_big.block(6*c+3,Hn_mapping[kf_pose],3,3)=dh2_dRWKF;
        Hn_big.block(6*c+3,Hn_mapping[kf_pose]+3,3,3)=dh2_dtWKF;


        //R_kf_cur=dR*R_KF_C --> dR=R_kf_cur* R_KF_C.transpose()  --> dtheta= log_so3(R_kf_cur*R_KF_C.transpose());
        Matrix3d obs_R=R_kf_cur;
        Vector3d obs_t=t_kf_cur;
        if(i!=anchor_id)
        {
            Keyframe* anchor=kfs[anchor_id];
            assert(anchor!=nullptr);
            Matrix3d R_W_anchor=anchor->_Pose_KFinWorld->Rot();
            Vector3d t_W_anchor=anchor->_Pose_KFinWorld->pos();
            Matrix3d R_K_anchor=R_W_KF.transpose()*R_W_anchor;
            Vector3d t_K_anchor=R_W_KF.transpose()*(t_W_anchor-t_W_KF);
            obs_R=R_K_anchor*R_kf_cur;
            obs_t=t_K_anchor+R_K_anchor*t_kf_cur;
        }

        Vector3d dtheta=log_so3(obs_R*R_KF_C.transpose());
        Vector3d dt=obs_t-t_KF_C;
        cout<<"dtheta: "<<dtheta<<" dt: "<<dt<<endl;
        res_big.block(6*c+0,0,3,1)=dtheta;
        res_big.block(6*c+3,0,3,1)=dt;
        c++;
    }
    cout<<"res: "<<res_big.norm()<<endl;
    if(res_big.norm()>10)
    return false;
    // sleep(1);

    MatrixXd R=5*MatrixXd::Identity(res_big.rows(),res_big.rows());


    StateHelper::SKFUpdate(state, Hx_order_big, Hn_order_big, Hx_big, Hn_big, res_big, R,iterative);


    return true;
}


bool UpdaterMSCKF::update_noskf_with_KF(State *state, std::vector<Feature *> &feature_vec, bool iterative)
{
    // Return if no features
    if(feature_vec.empty())
        return false;
    std::cout<<"at the begining, feature size is"<<feature_vec.size()<<endl;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    //
    cout<<"0"<<endl;
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    cout<<"1"<<endl;
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature measurements that are not observed by clones (measurements timsstamp are not found in clonetimes)
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements by slide windows
        int ct_meas = 0;
        for(const auto &pair : (*it0)->timestamps) {
            ct_meas += (*it0)->timestamps[pair.first].size();
        }

        // Remove if we don't have enough
        if(ct_meas < 1) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }

    }
    rT1 =  boost::posix_time::microsec_clock::local_time();
    std::cout<<"after clean old measurement, feature size is"<<feature_vec.size()<<endl;


    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    for(size_t i=0; i<feature_vec.size(); i++) {
        max_meas_size += 2*feature_vec.at(i)->uvs[0].size(); //for each feature, it could be projected into curframe;
    }
    if(feature_vec.empty()){  
        return false;
    }
    assert(max_meas_size>0);
    

    // Calculate max possible state size (i.e. the size of our covariance)
    // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
    size_t max_hx_size = state->max_covariance_size();//current state covariance size
    for(auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();  //minus the size of slam_features
    }



    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd noise_big = Eigen::MatrixXd::Zero(max_meas_size,max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size); //[Hx ]
    Eigen::VectorXd res_big_last = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big_last = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size); //[Hx ]
    std::unordered_map<Type*,size_t> Hx_mapping;
    std::unordered_map<Type*,size_t> Hn_mapping;
    std::vector<Type*> Hx_order_big; //[imu,clones_imu...]
    std::vector<Type*> Hn_order_big; //[clones_kf....]
    size_t ct_jacob = 0;
    size_t ct_meas = 0;
    size_t ct_jacob_n=0;

    // 4. Compute linear system for each feature, nullspace project, and reject
    cout<<"4"<<endl;
    auto it2 = feature_vec.begin();
    int feat_id_index=0;
    // assert(state->of_points.is_open());
    // state->of_points<<"in updaterMSCKFwithKF, timestamp:"<<" "<<to_string(state->_timestamp)<<" how many kf for each feature: "<<endl;
    double avg_chi2=0;
    double avg_error_1=0;
    double avg_error_2=0;
    double num_features = feature_vec.size();
    while(it2 != feature_vec.end()) {

        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        feat.keyframe_matched_obs = (*it2)->keyframe_matched_obs;

        // // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        // feat.feat_representation = state->_options.feat_rep_msckf;
        // if(state->_options.feat_rep_msckf==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        //     feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        // }

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f_x;
        Eigen::MatrixXd H_f_x_last;
        Eigen::MatrixXd H_f_n;
        Eigen::MatrixXd H_f_n_last;
        Eigen::MatrixXd H_x;  //jacobian of obsevation of sliding windows
        Eigen::MatrixXd H_x_last;
        Eigen::MatrixXd H_n;  //jacobian of observation of kf relating to nuisance part
        Eigen::MatrixXd H_n_last;
        Eigen::MatrixXd H_n_x; //jacobian of observatin of kf relating to state
        Eigen::VectorXd res_x;
        Eigen::VectorXd res_n;
        Eigen::VectorXd res_n_last;
        std::vector<Type*> Hx_order;
        std::vector<Type*> Hn_order; //we use this to record the related n when compute H_n;
        std::vector<Type*> Hnx_order;//when compute H_n, it could related to x and n. we use this to record the related x
        std::vector<Type*> Hx_order_last;
        std::vector<Type*> Hn_order_last; //we use this to record the related n when compute H_n;
        std::vector<Type*> Hnx_order_last;

        // UpdaterHelper::get_feature_jacobian_kf3(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        // if(state->iter==false)
        UpdaterHelper::get_feature_jacobian_loc(state,feat,H_x,res_n,Hx_order);

        // else if(state->iter)
        // {
        //     UpdaterHelper::get_feature_jacobian_ukf(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        // }
        // if(state->iter)
        // {
        //     UpdaterHelper::get_feature_jacobian_kf3(state,feat,H_f_n_last,H_n_last,H_x_last,res_n_last,Hn_order_last,Hx_order_last);
            
        // }
    
        state->of_points<<res_n.rows()/4<<" ";

        
        double error_r_1=sqrt(res_n.transpose()*res_n);
        double error_r_2=error_r_1/res_n.rows();
        avg_error_1+=error_r_1;
        avg_error_2+=error_r_2;


        // cout<<"get feature jacobian kf"<<endl;
        assert(H_x.rows()==res_n.rows());
        MatrixXd noise=_options.sigma_pix_sq*MatrixXd::Identity(res_n.rows(),res_n.rows());


        Eigen::MatrixXd P_x_marg=StateHelper::get_marginal_covariance(state,Hx_order);
        //  cout<<"P_x_marg"<<endl;
        // Eigen::MatrixXd S1 = H_x*P_marg*H_x.transpose();  //S1=HPH^T
        //  S2= [Hx,Hn]* [Pxx Pxn] * [Hx^T ]
        //               [Pnx Pnn]   [Hn^T ]
        Eigen::MatrixXd S2(H_x.rows(),H_x.rows());
        S2 =H_x*P_x_marg*H_x.transpose();
        S2.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S2.rows()); 

        // double chi2_x = res_x.dot(S1.llt().solve(res_x)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
        double chi2_x = res_n.dot(S2.llt().solve(res_n));
        // the result is the Mahalanobis distance

        // Get our threshold (we precompute up to 500 but handle the case that it is more)
        double chi2_check_1,chi2_check_2;
        if(res_n.rows() < 500) {
            chi2_check_1 = chi_squared_table[res_n.rows()];
        } else {
            boost::math::chi_squared chi_squared_dist(res_n.rows());
            chi2_check_1 = boost::math::quantile(chi_squared_dist, 0.95);
            printf(GREEN "chi2_check over the residual limit - %d\n" RESET, (int)res_n.rows());
        }
        // Check if we should delete or not
        cout<<"chi2 vs. chi2check: "<<chi2_x<<" vs. "<<_options.chi2_multipler*chi2_check_1<<endl;
        if(chi2_x > _options.chi2_multipler*chi2_check_1) {

           (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            feat_id_index++;
            //cout << "featid = " << feat.featid << endl;
            //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
            //cout << "res = " << endl << res.transpose() << endl;
            printf(GREEN "chi2_x is too large\n" RESET );
            continue;
        }

        

        // assert(state->of_points.is_open());
        // state->of_points<<chi2_x<<" ";
        avg_chi2+=chi2_x;


        //Now we need to put H_x,H_n into H_x_big,H_n_big for each feat
        //                             x    nuisance
        //                           [... | ...] 
        //H_big=[H_x_big | H_n_big ]=[H_x | H_n]
        //                           [... | ...]
        
        size_t ct_hx = 0;
        for(const auto &var : Hx_order) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping.find(var)==Hx_mapping.end()) {
                Hx_mapping.insert({var,ct_jacob});  //记录var在Hx_big中的哪一列
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas,Hx_mapping[var],H_x.rows(),var->size()) = H_x.block(0,ct_hx,H_x.rows(),var->size());
            // if(state->iter)
            // Hx_big_last.block(ct_meas,Hx_mapping[var],H_x_last.rows(),var->size()) = H_x_last.block(0,ct_hx,H_x_last.rows(),var->size());
            ct_hx += var->size();
        }


        // Append our residual_n and move forward
        res_big.block(ct_meas,0,res_n.rows(),1) = res_n;

        noise_big.block(ct_meas,ct_meas,noise.rows(),noise.cols())=noise;


        // if(state->iter)
        //     res_big_last.block(ct_meas,0,res_n_last.rows(),1) = res_n_last;
        ct_meas += res_n.rows();


        it2++;

        feat_id_index++;



    }
    // sleep(5);
    rT3 =  boost::posix_time::microsec_clock::local_time();
    avg_chi2/=double(feature_vec.size());
    avg_error_1/=double(feature_vec.size());
    avg_error_2/=double(num_features);
    cout<<"score: "<<avg_error_2<<endl;


    // We have appended all features to our Hx_big, Hn_big , res_big
    // Delete it so we do not reuse information
    for(size_t f=0; f < feature_vec.size(); f++) {
        feature_vec[f]->to_delete = true;
    }

    // Return if we don't have anything and resize our matrices
    if(ct_meas < 1) {
        return false;
    }
    assert(ct_meas<=max_meas_size);
    assert(ct_jacob<=max_hx_size);
    res_big.conservativeResize(ct_meas,1);
    Hx_big.conservativeResize(ct_meas,ct_jacob);
    noise_big.conservativeResize(ct_meas,ct_meas);
    
    // if(state->iter)
    // {
    //     res_big_last.conservativeResize(ct_meas,1);
    //     Hx_big_last.conservativeResize(ct_meas,ct_jacob);
    //     Hn_big_last.conservativeResize(ct_meas,ct_jacob_n);
    // }
    // 5. Perform measurement compression
    //UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    cout<<"5"<<endl;
    cout<<"Hx_big.row: "<<Hx_big.rows()<<" "<<"Hx_big col: "<<Hx_big.cols()<<endl;
    // cout<<"big noise before compress: "<<endl<<noise_big<<endl;
    
   
   UpdaterHelper::measurement_compress_inplace(Hx_big,res_big);
    // Eigen::MatrixXd H_all=MatrixXd::Zero(Hx_big.rows(),Hx_big.cols()+Hn_big.cols());
    // H_all.block(0,0,Hx_big.rows(),Hx_big.cols())=Hx_big;
    // H_all.block(0,Hx_big.cols(),Hn_big.rows(),Hn_big.cols())=Hn_big;
    // int Hx_cols=Hx_big.cols();
    // int Hn_cols=Hn_big.cols();
    // if(H_all.rows()>(H_all.cols()))
    // {
    //     Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(H_all.rows(), H_all.cols());
    //     qr.compute(H_all);
    //     Eigen::MatrixXd Q = qr.householderQ();
    //     cout<<"Q size: "<<Q.rows()<<" "<<Q.cols()<<endl;
    //     Eigen::MatrixXd Q1 = Q.block(0,0,Q.rows(),H_all.cols());
    //     Eigen::MatrixXd Q2 = Q.block(0,H_all.cols(),Q.rows(),Q.cols()-H_all.cols());
    //     H_all=Q1.transpose()*H_all;
    //     res_big=Q1.transpose()*res_big;
    //     Hx_big=H_all.block(0,0,H_all.rows(),Hx_cols);
    //     Hn_big=H_all.block(0,Hx_cols,H_all.rows(),Hn_cols);
    //     noise_big=Q1.transpose()*noise_big*Q1;
    // }

    // UpdaterHelper::measurement_compress_inplace_with_nuisance_noise(Hx_big,Hn_big,noise_big,res_big);
    // cout<<"big noise after compress: "<<endl<<noise_big<<endl;
    // sleep(10);
    // if(state->iter)
    // {
    //     UpdaterHelper::measurement_compress_inplace_with_nuisance(Hx_big_last,Hn_big_last,res_big_last);
    // }

    //    UpdaterHelper::measurement_compress_inplace(Hx_big,res_big); 
    //    Hn_big.conservativeResize(Hx_big.rows(),ct_jacob_n); 
    
    if(Hx_big.rows() < 1) {
        return false;
    }
    rT4 =  boost::posix_time::microsec_clock::local_time();

    // if(state->iter==true)   
    // {
    //     double now=(Hx_big*Hx_big.transpose()).determinant();
    //     double last=(_Hx_big_last*_Hx_big_last.transpose()).determinant();
    //     if(avg_error_2>_avg_error_2_last||now>last)  //if error not descrease, we should stop the iteration and use the last iteration information.
    //     // if(now>last)
    //     {
    //         // iterative=false;
    //         // state->iter=false;
           
    //         Hx_big=_Hx_big_last;
    //         Hn_big=_Hn_big_last;
    //         res_big=_res_big_last;
    //     }
        
        
    // }
    // else if(state->iter==false)  //if update is not iterative yet, we need to judge whether it should iterative update.
    
    if(state->iter==false)
    {
        if(state->distance_last_match>state->_options.opt_thred&&state->_options.opt_thred<100)
        {
            state->iter=true;
            state->iter_count=0;
            // iterative=true;
            // _avg_error_2_last=avg_error_2;        get_feature_jacobian_kf4
            // _Hx_big_last=Hx_big;
            // _Hn_big_last=Hn_big;
            // _res_big_last=res_big;
            // state->_avg_error_2_last=avg_error_2;
            // state->_Cov_iter=state->_Cov;
            // state->_Cross_Cov_AN_iter=state->_Cross_Cov_AN;
            return false; //return false when using optimize;
        }
    }

    // Our noise is isotropic, so make it here after our compression
   Eigen::MatrixXd R_big = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());
    //   Eigen::MatrixXd R_big=noise_big;
    // 6. With all good features update the state
    std::cout<<"6"<<endl;
   
        
    StateHelper::EKFUpdate(state, Hx_order_big,  Hx_big, res_big, R_big);


        
    rT5 =  boost::posix_time::microsec_clock::local_time();
    return true;
}

void UpdaterMSCKF::update(State *state, std::vector<Feature*>& feature_vec) {

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

        // Clean the feature measurements that are not observed by clones (measurements timsstamp are not found in clonetimes)
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


    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    for(size_t i=0; i<feature_vec.size(); i++) {
        for (const auto &pair : feature_vec.at(i)->timestamps) {
            max_meas_size += 2*feature_vec.at(i)->timestamps[pair.first].size();
        }
    }

    // Calculate max possible state size (i.e. the size of our covariance)
    // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
    size_t max_hx_size = state->max_covariance_size();//current state covariance size
    for(auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();  //minus the size of slam_features
    }

    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size);
    std::unordered_map<Type*,size_t> Hx_mapping;
    std::vector<Type*> Hx_order_big;
    size_t ct_jacob = 0;
    size_t ct_meas = 0;


    // 4. Compute linear system for each feature, nullspace project, and reject
    auto it2 = feature_vec.begin();
    cout<<"num of features used to update: "<<feature_vec.size()<<endl;
    while(it2 != feature_vec.end()) {

        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        

        // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        feat.feat_representation = state->_options.feat_rep_msckf;
        // right now, we only support GLOBAL_3D 
        assert(feat.feat_representation == LandmarkRepresentation::Representation::GLOBAL_3D);
        // if(state->_options.feat_rep_msckf==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        //     feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        // }

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

        // Nullspace project
        UpdaterHelper::nullspace_project_inplace(H_f, H_x, res);

        /// Chi2 distance check
        //get the covariance correspondes to the elements in Hx_order. by zzq
        Eigen::MatrixXd P_marg = StateHelper::get_marginal_covariance(state, Hx_order);
        Eigen::MatrixXd S = H_x*P_marg*H_x.transpose();
        S.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S.rows()); //HPH^T+R
        double chi2 = res.dot(S.llt().solve(res)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
                                                   // the result is the Mahalanobis distance

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
        if(chi2 > _options.chi2_multipler*chi2_check) {
            (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            printf(GREEN "msckf feature chi2_x is too large\n" RESET );
            //cout << "featid = " << feat.featid << endl;
            //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
            //cout << "res = " << endl << res.transpose() << endl;
            continue;
        }

        // We are good!!! Append to our large H vector
        size_t ct_hx = 0; //id of each var in Hx along Hx's col
        for(const auto &var : Hx_order) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping.find(var)==Hx_mapping.end()) {
                Hx_mapping.insert({var,ct_jacob});
                Hx_order_big.push_back(var);
                ct_jacob += var->size();  //id of each var in Hx_big along Hx_big's col
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas,Hx_mapping[var],H_x.rows(),var->size()) = H_x.block(0,ct_hx,H_x.rows(),var->size());
            ct_hx += var->size();

        }

        // Append our residual and move forward
        res_big.block(ct_meas,0,res.rows(),1) = res;
        ct_meas += res.rows();
        it2++;

    }
    rT3 =  boost::posix_time::microsec_clock::local_time();

    // We have appended all features to our Hx_big, res_big
    // Delete it so we do not reuse information
    for (size_t f=0; f < feature_vec.size(); f++) {
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


    // 5. Perform measurement compression
    UpdaterHelper::measurement_compress_inplace(Hx_big, res_big);
    if(Hx_big.rows() < 1) {
        return;
    }
    rT4 =  boost::posix_time::microsec_clock::local_time();


    // Our noise is isotropic, so make it here after our compression
    Eigen::MatrixXd R_big = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());

    // 6. With all good features update the state
    StateHelper::EKFUpdate(state, Hx_order_big, Hx_big, res_big, R_big);
    rT5 =  boost::posix_time::microsec_clock::local_time();

    // Debug print timing information
    //printf("[MSCKF-UP]: %.4f seconds to clean\n",(rT1-rT0).total_microseconds() * 1e-6);
    //printf("[MSCKF-UP]: %.4f seconds to triangulate\n",(rT2-rT1).total_microseconds() * 1e-6);
    //printf("[MSCKF-UP]: %.4f seconds create system (%d features)\n",(rT3-rT2).total_microseconds() * 1e-6, (int)feature_vec.size());
    //printf("[MSCKF-UP]: %.4f seconds compress system\n",(rT4-rT3).total_microseconds() * 1e-6);
    //printf("[MSCKF-UP]: %.4f seconds update state (%d size)\n",(rT5-rT4).total_microseconds() * 1e-6, (int)res_big.rows());
    //printf("[MSCKF-UP]: %.4f seconds total\n",(rT5-rT1).total_microseconds() * 1e-6);

}


bool UpdaterMSCKF::update_initial_transform(State *state, std::vector<Feature *>& feature_vec){
  if(feature_vec.empty())
        return false;
    std::cout<<"in rimsckf update_initial_transform, at the begining, feature size is"<<feature_vec.size()<<endl;

    // Start timing
    boost::posix_time::ptime rT0, rT1, rT2, rT3, rT4, rT5, rT6, rT7;
    rT0 =  boost::posix_time::microsec_clock::local_time();

    // 0. Get all timestamps our clones are at (and thus valid measurement times)
    //
    cout<<"0"<<endl;
    std::vector<double> clonetimes;
    for(const auto& clone_imu : state->_clones_IMU) {
        clonetimes.emplace_back(clone_imu.first);
    }

    // 1. Clean all feature measurements and make sure they all have valid clone times
    cout<<"1"<<endl;
    auto it0 = feature_vec.begin();
    while(it0 != feature_vec.end()) {

        // Clean the feature measurements that are not observed by clones (measurements timsstamp are not found in clonetimes)
        (*it0)->clean_old_measurements(clonetimes);

        // Count how many measurements by slide windows
        int ct_meas = 0;
        for(const auto &pair : (*it0)->timestamps) {
            ct_meas += (*it0)->timestamps[pair.first].size();
        }

        // Remove if we don't have enough
        if(ct_meas < 1) {
            (*it0)->to_delete = true;
            it0 = feature_vec.erase(it0);
        } else {
            it0++;
        }

    }
    rT1 =  boost::posix_time::microsec_clock::local_time();
    std::cout<<"after clean old measurement, feature size is"<<feature_vec.size()<<endl;

    
    // Calculate the max possible measurement size
    size_t max_meas_size = 0;
    for(size_t i=0; i<feature_vec.size(); i++) {
        assert(!feature_vec.at(i)->keyframe_matched_obs.empty());
        for(auto match: feature_vec.at(i)->keyframe_matched_obs)
        {
            max_meas_size += 2*match.second.size();  //for each feature,we have  a 3d point represented in a match kf, and the 3d point could be projected to each of matched kfframe
        }
        max_meas_size += 2*feature_vec.at(i)->uvs[0].size(); //for each feature, it could be projected into curframe;
        
    }
    if(feature_vec.empty()){  
        return false;
    }
    assert(max_meas_size>0);
    

    // Calculate max possible state size (i.e. the size of our covariance)
    // NOTE: that when we have the single inverse depth representations, those are only 1dof in size
    size_t max_hx_size = state->max_covariance_size();//current state covariance size
    for(auto &landmark : state->_features_SLAM) {
        max_hx_size -= landmark.second->size();  //minus the size of slam_features
    }

    size_t max_hn_size= state->max_nuisance_covariance_size();


    // Large Jacobian and residual of *all* features for this update
    Eigen::VectorXd res_big = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd noise_big = Eigen::MatrixXd::Zero(max_meas_size,max_meas_size);
    Eigen::MatrixXd Hx_big = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size); //[Hx ]
    Eigen::MatrixXd Ha_big = Eigen::MatrixXd::Zero(max_meas_size,max_hx_size);
    Eigen::MatrixXd Ht_big = Eigen::MatrixXd::Zero(max_meas_size,6); //relative transformation has the dimension 6
    Eigen::MatrixXd Hn_big = Eigen::MatrixXd::Zero(max_meas_size,max_hn_size); //[H_nuisance] 
    Eigen::VectorXd res_big_last = Eigen::VectorXd::Zero(max_meas_size);
    Eigen::MatrixXd Hx_big_last = Eigen::MatrixXd::Zero(max_meas_size, max_hx_size); //[Hx ]
    Eigen::MatrixXd Hn_big_last = Eigen::MatrixXd::Zero(max_meas_size,max_hn_size); //[H_nuisance] 
    std::unordered_map<Type*,size_t> Hx_mapping;
    std::unordered_map<Type*,size_t> Ha_mapping;
    std::unordered_map<Type*,size_t> Ht_mapping;
    std::unordered_map<Type*,size_t> Hn_mapping;
    std::vector<Type*> Hx_order_big; //[imu,clones_imu...]
    std::vector<Type*> Ha_order_big;
    std::vector<Type*> Ht_order_big;
    std::vector<Type*> Hn_order_big; //[clones_kf....]
    size_t ct_jacob = 0;
    size_t ct_jacob_a = 0;
    size_t ct_jacob_t = 0;
    size_t ct_meas = 0;
    size_t ct_jacob_n=0;

    // 4. Compute linear system for each feature, nullspace project, and reject
    cout<<"4"<<endl;
    auto it2 = feature_vec.begin();
    int feat_id_index=0;
    // assert(state->of_points.is_open());
    // state->of_points<<"in updaterMSCKFwithKF, timestamp:"<<" "<<to_string(state->_timestamp)<<" how many kf for each feature: "<<endl;
    double avg_chi2=0;
    double avg_error_1=0;
    double avg_error_2=0;
    int num_features=feature_vec.size();
    while(it2 != feature_vec.end()) {

        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;
        feat.keyframe_matched_obs = (*it2)->keyframe_matched_obs;
        // cout<<"observed by "<<(*it2)->keyframe_matched_obs.size()<<endl;
        // sleep(2);

        // // If we are using single inverse depth, then it is equivalent to using the msckf inverse depth
        // feat.feat_representation = state->_options.feat_rep_msckf;
        // if(state->_options.feat_rep_msckf==LandmarkRepresentation::Representation::ANCHORED_INVERSE_DEPTH_SINGLE) {
        //     feat.feat_representation = LandmarkRepresentation::Representation::ANCHORED_MSCKF_INVERSE_DEPTH;
        // }

        // Our return values (feature jacobian, state jacobian, residual, and order of state jacobian)
        Eigen::MatrixXd H_f_x;
        Eigen::MatrixXd H_f_x_last;
        Eigen::MatrixXd H_f_n;
        Eigen::MatrixXd H_f_n_last;
        Eigen::MatrixXd H_x;  //jacobian of obsevation of sliding windows
        Eigen::MatrixXd H_x_last;
        Eigen::MatrixXd H_n;  //jacobian of observation of kf relating to nuisance part
        Eigen::MatrixXd H_n_last;
        Eigen::MatrixXd H_n_x; //jacobian of observatin of kf relating to state
        Eigen::VectorXd res_x;
        Eigen::VectorXd res_n;
        Eigen::VectorXd res_n_last;
        std::vector<Type*> Hx_order;
        std::vector<Type*> Hn_order; //we use this to record the related n when compute H_n;
        std::vector<Type*> Hnx_order;//when compute H_n, it could related to x and n. we use this to record the related x
        std::vector<Type*> Hx_order_last;
        std::vector<Type*> Hn_order_last; //we use this to record the related n when compute H_n;
        std::vector<Type*> Hnx_order_last;

        // UpdaterHelper::get_feature_jacobian_kf3(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        // if(state->iter==false)
        //kf3 for invariant error, kf4 for invariant error for kfposes,srd error for pt, kf5 for standard error
        bool success = UpdaterHelper::get_feature_jacobian_kf3(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        if(!success)
        {
          (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            feat_id_index++;
            continue;
        }
        cout<<"H_x: "<<endl<<H_x<<endl<<"H_n: "<<endl<<H_n<<endl<<" H_f_n:"<<endl<<H_f_n<<endl;

        // cout<<"get feature jacobian kf"<<endl;
        cout<<"H_f_n.cols(): "<<H_f_n.cols()<<" H_x.cols(): "<<H_x.cols()<<" H_n.cols(): "<<H_n.cols()<<endl;
        cout<<"H_f_n.rows(): "<<H_f_n.rows()<<endl;
        assert(H_n.rows()==res_n.rows());
        assert(H_x.rows()==res_n.rows());
        assert(H_f_n.rows()==res_n.rows());
        MatrixXd noise=_options.sigma_pix_sq*MatrixXd::Identity(res_n.rows(),res_n.rows());

        if(!state->_options.ptmeas)
        {
          // UpdaterHelper::nullspace_project_inplace_with_nuisance_noise(H_f_n, H_x, H_n,noise, res_n);
          // UpdaterHelper::nullspace_project_inplace_with_nuisance(H_f_n, H_x, H_n, res_n);

        }
            

        cout<<"after nullspace"<<endl<<"H_x: "<<endl<<H_x<<endl<<"H_n: "<<endl<<H_n<<endl<<" H_f_n:"<<endl<<H_f_n<<endl;
        

        // Eigen::MatrixXd P_n_marg=StateHelper::get_marginal_nuisance_covariance(state,Hn_order);
        // //  cout<<"P_n_marg"<<endl;
        // Eigen::MatrixXd P_xn_marg=StateHelper::get_marginal_cross_covariance(state,Hx_order,Hn_order);
        // //  cout<<"P_xn_marg"<<endl;
        // Eigen::MatrixXd P_x_marg=StateHelper::get_marginal_covariance(state,Hx_order);
        // //  cout<<"P_x_marg"<<endl;
        // // Eigen::MatrixXd S1 = H_x*P_marg*H_x.transpose();  //S1=HPH^T
        // //  S2= [Hx,Hn]* [Pxx Pxn] * [Hx^T ]
        // //               [Pnx Pnn]   [Hn^T ]
        // Eigen::MatrixXd S2(H_x.rows(),H_x.rows());
        // S2 = H_n*P_n_marg*H_n.transpose();
        // S2 +=H_x*P_x_marg*H_x.transpose();
        // S2 += H_x*P_xn_marg*H_n.transpose();
        // S2 += H_n*P_xn_marg.transpose()*H_x.transpose();
        // // S1.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S1.rows()); //HPH^T+R
        // S2.diagonal() += _options.sigma_pix_sq*Eigen::VectorXd::Ones(S2.rows()); 

        // // double chi2_x = res_x.dot(S1.llt().solve(res_x)); // r.dot(S^-1*r) S is the measurement covariance, S^-1 is information matrix
        // double chi2_x = res_n.dot(S2.llt().solve(res_n));
        // // the result is the Mahalanobis distance

        // // Get our threshold (we precompute up to 500 but handle the case that it is more)
        // double chi2_check_1,chi2_check_2;
        // if(res_n.rows() < 500) {
        //     chi2_check_1 = chi_squared_table[res_n.rows()];
        // } else {
        //     boost::math::chi_squared chi_squared_dist(res_n.rows());
        //     chi2_check_1 = boost::math::quantile(chi_squared_dist, 0.95);
        //     printf(GREEN "chi2_check over the residual limit - %d\n" RESET, (int)res_n.rows());
        // }
        // // Check if we should delete or not
        // cout<<"chi2 vs. chi2check: "<<chi2_x<<" vs. "<<_options.chi2_multipler*chi2_check_1<<endl;
        // cout<<"observed by "<<(*it2)->keyframe_matched_obs.at(state->_timestamp_approx).size()<<endl;
        // cout<<"res: "<<res_n.norm()/res_n.rows()<<endl;
        // // sleep(1);
        // if(chi2_x > _options.chi2_multipler*chi2_check_1) {

        //    (*it2)->to_delete = true;
        //     it2 = feature_vec.erase(it2);
        //     feat_id_index++;
        //     //cout << "featid = " << feat.featid << endl;
        //     //cout << "chi2 = " << chi2 << " > " << _options.chi2_multipler*chi2_check << endl;
        //     //cout << "res = " << endl << res.transpose() << endl;
        //     printf(GREEN "global observation chi2_x is too large\n" RESET );
        //     // sleep(3);
        //     continue;
        // }

        //* Now we need to stack H_x, H_n into H_x_big, H_n_big for each feat
        //                                          x   t  
        //                                        [...|...]
        // H_big = [H_x_big| H_n_big]    =        H_x|H_n]
        //                                        [...|...]
       
        size_t ct_hx = 0;
        for(const auto &var : Hx_order) {

            // Ensure that this variable is in our Jacobian
            if(Hx_mapping.find(var)==Hx_mapping.end()) {
                Hx_mapping.insert({var,ct_jacob});  //记录var在Hx_big中的哪一列
                Hx_order_big.push_back(var);
                ct_jacob += var->size();
            }

            // Append to our large Jacobian
            Hx_big.block(ct_meas,Hx_mapping[var],H_x.rows(),var->size()) = H_x.block(0,ct_hx,H_x.rows(),var->size());
            // if(state->iter)
            // Hx_big_last.block(ct_meas,Hx_mapping[var],H_x_last.rows(),var->size()) = H_x_last.block(0,ct_hx,H_x_last.rows(),var->size());
            ct_hx += var->size();
        }

        size_t ct_hn =0;
        for(const auto &var : Hn_order){
            if(Hn_mapping.find(var)==Hn_mapping.end())
            {
                Hn_mapping.insert({var,ct_jacob_n});
                Hn_order_big.push_back(var);
                ct_jacob_n += var->size();
            }
            Hn_big.block(ct_meas,Hn_mapping[var],H_n.rows(),var->size()) = H_n.block(0,ct_hn,H_n.rows(),var->size());
            // if(state->iter)
            // Hn_big_last.block(ct_meas,Hn_mapping[var],H_n_last.rows(),var->size()) = H_n.block(0,ct_hn,H_n_last.rows(),var->size());
            ct_hn += var->size();
        }

        // Append our residual_n and move forward
        res_big.block(ct_meas,0,res_n.rows(),1) = res_n;

        noise_big.block(ct_meas,ct_meas,noise.rows(),noise.cols())=noise;

        ct_meas += res_n.rows();


        it2++;

        feat_id_index++;



    }
    // sleep(5);
    rT3 =  boost::posix_time::microsec_clock::local_time();
    
    
    


    // We have appended all features to our Hx_big, Hn_big , res_big
    // Delete it so we do not reuse information
    for(size_t f=0; f < feature_vec.size(); f++) {
        feature_vec[f]->to_delete = true;
    }

    // Return if we don't have anything and resize our matrices
    if(ct_meas < 5) {
        return false;
    }
    assert(ct_meas<=max_meas_size);
    assert(ct_jacob<=max_hx_size);
    assert(ct_jacob_n<=max_hn_size);
    res_big.conservativeResize(ct_meas,1);
    Hx_big.conservativeResize(ct_meas,ct_jacob);
    Hn_big.conservativeResize(ct_meas,ct_jacob_n);
    noise_big.conservativeResize(ct_meas,ct_meas);
    

    cout<<"5"<<endl;
    cout<<"Hx_big.row: "<<Hx_big.rows()<<" "<<"Hx_big col: "<<Hx_big.cols()<<" Hn_big col: "<<Hn_big.cols()<<endl;
    
   
   UpdaterHelper::measurement_compress_inplace_with_nuisance(Hx_big,Hn_big,res_big);

    if(Hx_big.rows() < 1) {
        return false;
    }
    rT4 =  boost::posix_time::microsec_clock::local_time();

    //*split H_x into H_a and H_t  where H_a is related to active variables except relative transformation
    //* H_t is related to the relative transformation.
    PoseJPL *transform = state->transform_map_to_vio;
    std::unordered_map<Type*,size_t> map_hx;
    std::unordered_map<Type*,size_t> map_ha;
    std::unordered_map<Type*,size_t> map_ht;
    int count_x = 0;
    int count_a = 0;
    int count_t = 0;
    for(auto var : Hx_order_big)
    {
      if(map_hx.find(var)==map_hx.end())
      {
        if(var == transform)
        {
          Ht_order_big.push_back(var);
          map_hx.insert({var,count_x});
          map_ht.insert({var,count_t});
          Ht_big.block(0,count_t,Hx_big.rows(),var->size()) = Hx_big.block(0,count_x,Hx_big.rows(),var->size());
          count_x += var->size();
          count_t += var->size();
        }
        else
        {
          Ha_order_big.push_back(var);
          map_hx.insert({var,count_x});
          map_ha.insert({var,count_a});
          Ha_big.block(0,count_a,Hx_big.rows(),var->size()) = Hx_big.block(0,count_x,Hx_big.rows(),var->size());
          count_x += var->size();
          count_a += var->size();
        } 
      }
    }

    
    assert(Ha_order_big.size()>0);
    assert(Ht_order_big.size()>0);
    assert(Ha_order_big.size()+Ht_order_big.size()==Hx_order_big.size());
    assert(count_x == count_a + count_t);
    Ha_big.conservativeResize(Hx_big.rows(),count_a);
    Ht_big.conservativeResize(Hx_big.rows(),count_t);

    cout<<"Hx_big:"<<endl<<Hx_big<<endl<<"Hn_big: "<<endl<<Hn_big<<endl;
    cout<<"Ha_big:"<<endl<<Ha_big<<endl<<"Ht_big:"<<endl<<Ht_big<<endl;
    // sleep(2);


    // Our noise is isotropic, so make it here after our compression
   Eigen::MatrixXd R_big = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());
    //   Eigen::MatrixXd R_big=noise_big;
    // 6. With all good features update the state
   
  //  StateHelper::SKFUpdate(state, Hx_order_big, Hn_order_big, Hx_big, Hn_big, res_big, R_big,false);
   StateHelper::init_transform_update(state, Ha_order_big, Ht_order_big, Hn_order_big, Ha_big, Ht_big, Hn_big, res_big, R_big);

    

        
    rT5 =  boost::posix_time::microsec_clock::local_time();
    return true;
}







