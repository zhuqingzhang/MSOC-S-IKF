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
using namespace ov_msckf;
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



bool UpdaterMSCKF::update_skf_with_KF(State *state, std::vector<Feature *> &feature_vec, bool iterative) {
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

        bool success = false;
        if(!state->_options.trans_fej)
            success = UpdaterHelper::get_feature_jacobian_kf(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        else
            success = UpdaterHelper::get_feature_jacobian_kf_transfej(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);

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

        assert(H_n.rows()==res_n.rows());
        assert(H_x.rows()==res_n.rows());
        assert(H_f_n.rows()==res_n.rows());
        MatrixXd noise=_options.sigma_pix_sq*MatrixXd::Identity(res_n.rows(),res_n.rows());

        if(!state->_options.ptmeas)
            UpdaterHelper::nullspace_project_inplace_with_nuisance_noise(H_f_n, H_x, H_n,noise, res_n);
        

       

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

        //TODO: whether delete the map observation?
        // // Check if we should delete or not
        if(chi2_x >_options.chi2_multipler*chi2_check_1) {

           (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            feat_id_index++;

            printf(GREEN "chi2_x is too large\n" RESET);
            continue;
        }

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
            
            ct_hn += var->size();
        }

        // Append our residual_n and move forward
        res_big.block(ct_meas,0,res_n.rows(),1) = res_n;

        noise_big.block(ct_meas,ct_meas,noise.rows(),noise.cols())=noise;



        ct_meas += res_n.rows();


        it2++;

        feat_id_index++;
        valid_features++;
    }

    rT3 =  boost::posix_time::microsec_clock::local_time();
    avg_chi2/=double(feature_vec.size());
    avg_error_1/=double(valid_features);
    avg_error_2/=double(valid_features);
    double failed_rate = double(failed_features)/double(num_features);

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
    noise_big.conservativeResize(ct_meas,ct_meas);
    

    // 5. Perform measurement compression
    cout<<"5"<<endl;
    cout<<"Hx_big.row: "<<Hx_big.rows()<<" "<<"Hx_big col: "<<Hx_big.cols()<<" Hn_big col: "<<Hn_big.cols()<<endl;
    
   
   UpdaterHelper::measurement_compress_inplace_with_nuisance(Hx_big,Hn_big,res_big);
    
    
    if(Hx_big.rows() < 1) {
        return false;
    }
    rT4 =  boost::posix_time::microsec_clock::local_time();

    if(state->iter==false)
    {
        if((avg_error_1>state->_options.opt_thred||failed_rate>0.1)&&state->_options.opt_thred<100)
        {
            state->iter=true;
            state->iter_count=0;
            iterative=true;
            if(state->_options.iter_num==0) 
                return false; //return false when using relinear;
        }
    }
    // sleep(10);



   

    // Our noise is isotropic, so make it here after our compression
   Eigen::MatrixXd R_big = Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());
    // 6. With all good features update the state

    StateHelper::SKFUpdate(state, Hx_order_big, Hn_order_big, Hx_big, Hn_big, res_big, R_big,iterative);
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
    while(it2 != feature_vec.end()) {

        // Convert our feature into our current format
        UpdaterHelper::UpdaterHelperFeature feat;
        feat.featid = (*it2)->featid;
        feat.uvs = (*it2)->uvs;
        feat.uvs_norm = (*it2)->uvs_norm;
        feat.timestamps = (*it2)->timestamps;

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
  // sleep(5);
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

        bool success = false;
        if(!state->_options.trans_fej)
            success = UpdaterHelper::get_feature_jacobian_kf(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);
        else
            success = UpdaterHelper::get_feature_jacobian_kf_transfej(state,feat,H_f_n,H_n,H_x,res_n,Hn_order,Hx_order);

        if(!success)
        {
          (*it2)->to_delete = true;
            it2 = feature_vec.erase(it2);
            feat_id_index++;
            continue;
        }
        assert(H_n.rows()==res_n.rows());
        assert(H_x.rows()==res_n.rows());
        assert(H_f_n.rows()==res_n.rows());
        MatrixXd noise=_options.sigma_pix_sq*MatrixXd::Identity(res_n.rows(),res_n.rows());

      
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
    

    
   
   UpdaterHelper::measurement_compress_inplace_with_nuisance(Hx_big,Hn_big,res_big);

    if(Hx_big.rows() < 1) {
        return false;
    }
    rT4 =  boost::posix_time::microsec_clock::local_time();

    //*split H_x into H_a and H_t  where H_a is related to active variables except relative transformation
    //* H_t is related to the relative transformation.
    PoseJPL *transform = state->transform_vio_to_map;
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



    // Our noise is isotropic, so make it here after our compression
   Eigen::MatrixXd R_big = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());
    // 6. With all good features update the state
   StateHelper::init_transform_update(state, Ha_order_big, Ht_order_big, Hn_order_big, Ha_big, Ht_big, Hn_big, res_big, R_big);

    rT5 =  boost::posix_time::microsec_clock::local_time();
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

        UpdaterHelper::get_feature_jacobian_loc(state,feat,H_x,res_n,Hx_order);
    
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
            printf(GREEN "chi2_x is too large\n" RESET );
            continue;
        }

        

      
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


        // Append our residual_n and move forward
        res_big.block(ct_meas,0,res_n.rows(),1) = res_n;

        noise_big.block(ct_meas,ct_meas,noise.rows(),noise.cols())=noise;


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
    
    // 5. Perform measurement compression
   UpdaterHelper::measurement_compress_inplace(Hx_big,res_big);
   
    
    if(Hx_big.rows() < 1) {
        return false;
    }
    rT4 =  boost::posix_time::microsec_clock::local_time();
    if(state->iter==false)
    {
        if(state->distance_last_match>state->_options.opt_thred&&state->_options.opt_thred<100)
        {
            state->iter=true;
            state->iter_count=0;
            return false; //return false when using optimize;
        }
    }

    // Our noise is isotropic, so make it here after our compression
   Eigen::MatrixXd R_big = _options.sigma_pix_sq*Eigen::MatrixXd::Identity(res_big.rows(),res_big.rows());
    // 6. With all good features update the state
    std::cout<<"6"<<endl;
    
    StateHelper::EKFUpdate(state, Hx_order_big,  Hx_big, res_big, R_big);

    rT5 =  boost::posix_time::microsec_clock::local_time();
    return true;
}












