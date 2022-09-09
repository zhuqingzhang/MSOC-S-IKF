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
#include "Propagator.h"
#include <boost/date_time.hpp>


using namespace ov_core;
using namespace ov_rimsckf;




void Propagator::propagate_and_clone(State* state, double timestamp) {

    // If the difference between the current update time and state is zero
    // We should crash, as this means we would have two clones at the same time!!!!
    if(state->_timestamp == timestamp) {
        printf(RED "Propagator::propagate_and_clone(): Propagation called again at same timestep at last update timestep!!!!\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // We should crash if we are trying to propagate backwards
    if(state->_timestamp > timestamp) {
        printf(RED "Propagator::propagate_and_clone(): Propagation called trying to propagate backwards in time!!!!\n" RESET);
        printf(RED "Propagator::propagate_and_clone(): desired propagation = %.4f\n" RESET, (timestamp-state->_timestamp));
        std::exit(EXIT_FAILURE);
    }

    //===================================================================================
    //===================================================================================
    //===================================================================================

    // Set the last time offset value if we have just started the system up
    if(!have_last_prop_time_offset) {
        last_prop_time_offset = state->_calib_dt_CAMtoIMU->value()(0);
        have_last_prop_time_offset = true;
    }

    // Get what our IMU-camera offset should be (t_imu = t_cam + calib_dt)
    double t_off_new = state->_calib_dt_CAMtoIMU->value()(0);

    // First lets construct an IMU vector of measurements we need
    //both time0 and time1 are imu timestamp
    double time0 = state->_timestamp+last_prop_time_offset; //current state time step k|k
    double time1 = timestamp+t_off_new;  //the time step we need to predict k+1|k
    vector<IMUDATA> prop_data = Propagator::select_imu_readings(imu_data,time0,time1);

    // We are going to sum up all the state transition matrices, so we can do a single large multiplication at the end
    // Phi_summed = Phi_i*Phi_summed
    // Q_summed = Phi_i*Q_summed*Phi_i^T + Q_i
    // After summing we can multiple the total phi to get the updated covariance
    // We will then add the noise to the IMU portion of the state

    //compute the size of propagation matrix. For RI-MSCKF, the propagation matrix
    //should be related to R_G_I, p_G_I, v_G_I, bg,ba, landmark_features, 6 DOF features( only translation is considered);
    int size_imu = state->_imu->size();
    assert(size_imu==15);
    int size_landmarks = state->_features_SLAM.size() * 3;
    
    int size_6DOFfeatures = 0;
    if(state->set_transform)
        size_6DOFfeatures = 1 * 3; //for localization, we only have 1 6DOF feature, i.e. transformation between local and global reference
    
    int prop_size = size_imu + size_landmarks + size_6DOFfeatures;
    Eigen::MatrixXd Phi_summed = Eigen::MatrixXd::Identity(prop_size,prop_size);
    Eigen::MatrixXd Qd_summed = Eigen::MatrixXd::Zero(prop_size,prop_size);
    double dt_summed = 0;

    // Loop through all IMU messages, and use them to move the state forward in time
    // This uses the zero'th order quat, and then constant acceleration discrete
     boost::posix_time::ptime t = boost::posix_time::microsec_clock::local_time();
    if(prop_data.size() > 1) {
        for(size_t i=0; i<prop_data.size()-1; i++) {

            // Get the next state Jacobian and noise Jacobian for this IMU reading
            Eigen::MatrixXd F = Eigen::MatrixXd::Zero(prop_size,prop_size);
            Eigen::MatrixXd Qdi = Eigen::MatrixXd::Zero(prop_size,prop_size);

            boost::posix_time::ptime t1 = boost::posix_time::microsec_clock::local_time();
            predict_and_compute(state, prop_data.at(i), prop_data.at(i+1), F, Qdi);
            //Here F is equal to Phi in documents. Qdi is equal to G_k*Q_d*G_k^T in documents
            boost::posix_time::ptime t2 = boost::posix_time::microsec_clock::local_time();
            // printf("predict and compute time: %f s\n", (t2-t1).total_microseconds() * 1e-6);

            // Next we should propagate our IMU covariance
            // Pii' = F*Pii*F.transpose() + G*Q*G.transpose()
            // Pci' = F*Pci and Pic' = Pic*F.transpose()
            // NOTE: Here we are summing the state transition F so we can do a single mutiplication later
            // NOTE: Phi_summed = Phi_i*Phi_summed
            // NOTE: Q_summed = Phi_i*Q_summed*Phi_i^T + G*Q_i*G^T
            t1 = boost::posix_time::microsec_clock::local_time();
            //F have special structrue
            //[A B 0]
            //[0 I 0]
            //[0 C I]    where A corresponding to the R,p,v of IMU, I corresponding to bias,
            //           B correponding to cross part of bias and other IMU parameters,     
            //           C correponding to cross part of bias and features (3dof and 6dof),   
            //so that we could split F into blocks to acclerate computation
            //IMU part
            Phi_summed.block(0,0,size_imu,size_imu) = 
                  F.block(0,0,size_imu,size_imu) *
                  Phi_summed.block(0,0,size_imu,size_imu);
            if(prop_size>size_imu) //we have feature parts (i,e, C)
            {
               Phi_summed.block(size_imu,size_imu-6,prop_size-size_imu,6) = 
               F.block(size_imu,size_imu-6,prop_size-size_imu,6) + Phi_summed.block(size_imu,size_imu-6,prop_size-size_imu,6);
            }

            // Phi_summed = F * Phi_summed;
            Qd_summed = F * Qd_summed * F.transpose() + Qdi;
            Qd_summed = 0.5*(Qd_summed+Qd_summed.transpose());
            dt_summed += prop_data.at(i+1).timestamp-prop_data.at(i).timestamp;
            t2 = boost::posix_time::microsec_clock::local_time();
            // printf("predict and compute after time : %f s\n", (t2-t1).total_microseconds() * 1e-6);
        }
    }
     boost::posix_time::ptime t_ = boost::posix_time::microsec_clock::local_time();
     printf("predict and compute accume time: %f s\n", (t_-t).total_microseconds() * 1e-6);
    // cout<<"***Phi_summed: "<<endl<<Phi_summed<<endl<<"***Qd_summed: "<<endl<<Qd_summed<<endl;

    // Last angular velocity (used for cloning when estimating time offset)
    Eigen::Matrix<double,3,1> last_w = Eigen::Matrix<double,3,1>::Zero();
    if(prop_data.size() > 1) last_w = prop_data.at(prop_data.size()-2).wm - state->_imu->bias_g();
    else if(!prop_data.empty()) last_w = prop_data.at(prop_data.size()-1).wm - state->_imu->bias_g();

    // Do the update to the covariance with our "summed" state transition and IMU noise addition...
    std::vector<Type*> Phi_order;
    Phi_order.push_back(state->_imu);
    cout<<"Phi_order size: "<<Phi_order.size()<<endl;
    cout<<"imu id and size: "<<state->_imu->id()<<" "<<state->_imu->size()<<endl;
    // propogate the covariance
    if(state->_features_SLAM.size()>0)
    {
        auto iter = state->_features_SLAM.begin();
        while(iter != state->_features_SLAM.end())
        {
            Phi_order.push_back(iter->second);
            iter++;
        }
    }
    if(state->set_transform)
    {
        Phi_order.push_back(state->transform_map_to_vio->p());
    }
    boost::posix_time::ptime t3 = boost::posix_time::microsec_clock::local_time();
    cout<<"Phi_order size: "<<Phi_order.size()<<endl;
    StateHelper::EKFPropagation(state, Phi_order, Phi_order, Phi_summed, Qd_summed);
    boost::posix_time::ptime t4 = boost::posix_time::microsec_clock::local_time();
    printf("EKFPropagation time: %f s\n", (t4-t3).total_microseconds() * 1e-6);

    // Set timestamp data
    state->_timestamp = timestamp;
    last_prop_time_offset = t_off_new;

    // Now perform stochastic cloning
    StateHelper::augment_clone(state, last_w);

}


//it use current state "state" to propagate to the state at "timestamp",
// and store the propagate values in state_plus(q,p,v,wm) without am
// it does not handle the propagation of covariance
void Propagator::fast_state_propagate(State *state, double timestamp, Eigen::Matrix<double,13,1> &state_plus) {

    // Set the last time offset value if we have just started the system up
    if(!have_last_prop_time_offset) {
        last_prop_time_offset = state->_calib_dt_CAMtoIMU->value()(0);
        have_last_prop_time_offset = true;
    }

    // Get what our IMU-camera offset should be (t_imu = t_cam + calib_dt)
    double t_off_new = state->_calib_dt_CAMtoIMU->value()(0);

    // First lets construct an IMU vector of measurements we need
    double time0 = state->_timestamp+last_prop_time_offset;
    double time1 = timestamp+t_off_new;
    vector<IMUDATA> prop_data = Propagator::select_imu_readings(imu_data,time0,time1);

    // Save the original IMU state
    Eigen::VectorXd orig_val = state->_imu->value();
    Eigen::VectorXd orig_fej = state->_imu->fej();

    // Loop through all IMU messages, and use them to move the state forward in time
    // This uses the zero'th order quat, and then constant acceleration discrete
    if(prop_data.size() > 1) {
        for(size_t i=0; i<prop_data.size()-1; i++) {

            // Time elapsed over interval
            double dt = prop_data.at(i+1).timestamp-prop_data.at(i).timestamp;
            //assert(data_plus.timestamp>data_minus.timestamp);

            // Corrected imu measurements
            Eigen::Matrix<double,3,1> w_hat = prop_data.at(i).wm - state->_imu->bias_g();
            Eigen::Matrix<double,3,1> a_hat = prop_data.at(i).am - state->_imu->bias_a();
            Eigen::Matrix<double,3,1> w_hat2 = prop_data.at(i+1).wm - state->_imu->bias_g();
            Eigen::Matrix<double,3,1> a_hat2 = prop_data.at(i+1).am - state->_imu->bias_a();

            // Compute the new state mean value
            Eigen::Vector4d new_q;
            Eigen::Vector3d new_v, new_p;
            if(state->_options.use_rk4_integration) predict_mean_rk4(state, dt, w_hat, a_hat, w_hat2, a_hat2, new_q, new_v, new_p);
            else predict_mean_discrete(state, dt, w_hat, a_hat, w_hat2, a_hat2, new_q, new_v, new_p);

            //Now replace imu estimate and fej with propagated values
            Eigen::Matrix<double,16,1> imu_x = state->_imu->value();
            imu_x.block(0,0,4,1) = new_q;
            imu_x.block(4,0,3,1) = new_p;
            imu_x.block(7,0,3,1) = new_v;
            state->_imu->set_value(imu_x);
            state->_imu->set_fej(imu_x);

        }
    }

    // Now record what the predicted state should be
    state_plus = Eigen::Matrix<double,13,1>::Zero();
    state_plus.block(0,0,4,1) = state->_imu->quat();
    state_plus.block(4,0,3,1) = state->_imu->pos();
    state_plus.block(7,0,3,1) = state->_imu->vel();
    if(prop_data.size() > 1) state_plus.block(10,0,3,1) = prop_data.at(prop_data.size()-2).wm - state->_imu->bias_g();
    else if(!prop_data.empty()) state_plus.block(10,0,3,1) = prop_data.at(prop_data.size()-1).wm - state->_imu->bias_g();

    // Finally replace the imu with the original state we had
    state->_imu->set_value(orig_val);
    state->_imu->set_fej(orig_fej);

}




std::vector<Propagator::IMUDATA> Propagator::select_imu_readings(const std::vector<IMUDATA>& imu_data, double time0, double time1) {

    // Our vector imu readings
    std::vector<Propagator::IMUDATA> prop_data;

    // Ensure we have some measurements in the first place!
    if(imu_data.empty()) {
        printf(YELLOW "Propagator::select_imu_readings(): No IMU measurements. IMU-CAMERA are likely messed up!!!\n" RESET);
        return prop_data;
    }

    // Loop through and find all the needed measurements to propagate with
    // Note we split measurements based on the given state time, and the update timestamp
    for(size_t i=0; i<imu_data.size()-1; i++) {

        // START OF THE INTEGRATION PERIOD
        // If the next timestamp is greater then our current state time
        // And the current is not greater then it yet...
        // Then we should "split" our current IMU measurement
        if(imu_data.at(i+1).timestamp > time0 && imu_data.at(i).timestamp < time0) {
            IMUDATA data = Propagator::interpolate_data(imu_data.at(i),imu_data.at(i+1), time0);
            prop_data.push_back(data);
            //printf("propagation #%d = CASE 1 = %.3f => %.3f\n", (int)i,data.timestamp-prop_data.at(0).timestamp,time0-prop_data.at(0).timestamp);
            continue;
        }

        // MIDDLE OF INTEGRATION PERIOD
        // If our imu measurement is right in the middle of our propagation period
        // Then we should just append the whole measurement time to our propagation vector
        if(imu_data.at(i).timestamp >= time0 && imu_data.at(i+1).timestamp <= time1) {
            prop_data.push_back(imu_data.at(i));
            //printf("propagation #%d = CASE 2 = %.3f\n",(int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp);
            continue;
        }

        // END OF THE INTEGRATION PERIOD
        // If the current timestamp is greater then our update time
        // We should just "split" the NEXT IMU measurement to the update time,
        // NOTE: we add the current time, and then the time at the end of the interval (so we can get a dt)
        // NOTE: we also break out of this loop, as this is the last IMU measurement we need!
        if(imu_data.at(i+1).timestamp > time1) {
            // If we have a very low frequency IMU then, we could have only recorded the first integration (i.e. case 1) and nothing else
            // In this case, both the current IMU measurement and the next is greater than the desired intepolation, thus we should just cut the current at the desired time
            // Else, we have hit CASE2 and this IMU measurement is not past the desired propagation time, thus add the whole IMU reading
            if(imu_data.at(i).timestamp > time1) {
                IMUDATA data = interpolate_data(imu_data.at(i-1), imu_data.at(i), time1);
                prop_data.push_back(data);
                //printf("propagation #%d = CASE 3.1 = %.3f => %.3f\n", (int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp,imu_data.at(i).timestamp-time0);
            } else {
                prop_data.push_back(imu_data.at(i));
                //printf("propagation #%d = CASE 3.2 = %.3f => %.3f\n", (int)i,imu_data.at(i).timestamp-prop_data.at(0).timestamp,imu_data.at(i).timestamp-time0);
            }
            // If the added IMU message doesn't end exactly at the camera time
            // Then we need to add another one that is right at the ending time
            if(prop_data.at(prop_data.size()-1).timestamp != time1) {
                IMUDATA data = interpolate_data(imu_data.at(i), imu_data.at(i+1), time1);
                prop_data.push_back(data);
                //printf("propagation #%d = CASE 3.3 = %.3f => %.3f\n", (int)i,data.timestamp-prop_data.at(0).timestamp,data.timestamp-time0);
            }
            break;
        }

    }

    // Check that we have at least one measurement to propagate with
    if(prop_data.empty()) {
        printf(YELLOW "Propagator::select_imu_readings(): No IMU measurements to propagate with (%d of 2). IMU-CAMERA are likely messed up!!!\n" RESET, (int)prop_data.size());
        return prop_data;
    }

    // If we did not reach the whole integration period (i.e., the last inertial measurement we have is smaller then the time we want to reach)
    // Then we should just "stretch" the last measurement to be the whole period (case 3 in the above loop)
    //TODO:why there is no implementation about this case?
    //if(time1-imu_data.at(imu_data.size()-1).timestamp > 1e-3) {
    //    printf(YELLOW "Propagator::select_imu_readings(): Missing inertial measurements to propagate with (%.6f sec missing). IMU-CAMERA are likely messed up!!!\n" RESET, (time1-imu_data.at(imu_data.size()-1).timestamp));
    //    return prop_data;
    //}

    // Loop through and ensure we do not have an zero dt values
    // This would cause the noise covariance to be Infinity
    for (size_t i=0; i < prop_data.size()-1; i++) {
        if (std::abs(prop_data.at(i+1).timestamp-prop_data.at(i).timestamp) < 1e-12) {
            printf(YELLOW "Propagator::select_imu_readings(): Zero DT between IMU reading %d and %d, removing it!\n" RESET, (int)i, (int)(i+1));
            prop_data.erase(prop_data.begin()+i);
            i--;
        }
    }

    // Check that we have at least one measurement to propagate with
    if(prop_data.size() < 2) {
        printf(YELLOW "Propagator::select_imu_readings(): No IMU measurements to propagate with (%d of 2). IMU-CAMERA are likely messed up!!!\n" RESET, (int)prop_data.size());
        return prop_data;
    }

    // Success :D
    return prop_data;

}




void Propagator::predict_and_compute(State *state, const IMUDATA data_minus, const IMUDATA data_plus,
                                     Eigen::MatrixXd &F, Eigen::MatrixXd &Qd) {

    // Set them to zero
    F.setZero();
    Qd.setZero();
    int prop_size = F.rows();

    // Time elapsed over interval
    double dt = data_plus.timestamp-data_minus.timestamp;
    //assert(data_plus.timestamp>data_minus.timestamp);

    // Corrected imu measurements
    Eigen::Matrix<double,3,1> w_hat = data_minus.wm - state->_imu->bias_g();
    Eigen::Matrix<double,3,1> a_hat = data_minus.am - state->_imu->bias_a();
    Eigen::Matrix<double,3,1> w_hat2 = data_plus.wm - state->_imu->bias_g();
    Eigen::Matrix<double,3,1> a_hat2 = data_plus.am - state->_imu->bias_a();

    // Compute the new state mean value
    Eigen::Vector4d new_q;
    Eigen::Vector3d new_v, new_p;
    if(state->_options.use_rk4_integration) predict_mean_rk4(state, dt, w_hat, a_hat, w_hat2, a_hat2, new_q, new_v, new_p);
    else predict_mean_discrete(state, dt, w_hat, a_hat, w_hat2, a_hat2, new_q, new_v, new_p);

    // Get the locations of each entry of the imu state
    int th_id = state->_imu->q()->id()-state->_imu->id();
    int p_id = state->_imu->p()->id()-state->_imu->id();
    int v_id = state->_imu->v()->id()-state->_imu->id();
    int bg_id = state->_imu->bg()->id()-state->_imu->id();
    int ba_id = state->_imu->ba()->id()-state->_imu->id();
    int landmark_id = ba_id + state->_imu->ba()->size();
    int total_size = landmark_id;
    

    // Get the size of feature landamrks;
    int size_landmarks = state->_features_SLAM.size();

    // Get the size of 6DOF feature  (for localization, it is the translation part of transformation between local and global reference)
    int size_6doffeature = 1;

    
    // Allocate noise Jacobian
    // As the structure of A and W is sparse, we directly compute the value of each small block to improving computational effeciency.
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(prop_size,3*4);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(prop_size,prop_size);
    Eigen::MatrixXd A2 = Eigen::MatrixXd::Zero(prop_size,prop_size); //A*A
    Eigen::MatrixXd A3 = Eigen::MatrixXd::Zero(prop_size,prop_size);  //A*A*A
    Eigen::Matrix3d R_ItoG = state->_imu->Rot().transpose();
    if(!state->_options.use_gt)
    {
      A.block(th_id,bg_id,3,3) = -R_ItoG;
      A.block(p_id,v_id,3,3) = Eigen::Matrix3d::Identity();
      A.block(p_id,bg_id,3,3) = -skew_x(state->_imu->pos())*R_ItoG;
      A.block(v_id,th_id,3,3) = skew_x(-_gravity);
      A.block(v_id,bg_id,3,3) = -skew_x(state->_imu->vel())*R_ItoG;
      A.block(v_id,ba_id,3,3) = -R_ItoG;

      A2.block(p_id,th_id,3,3) = skew_x(-_gravity);
      A2.block(p_id,bg_id,3,3) = -skew_x(state->_imu->vel()) * R_ItoG;
      A2.block(p_id,ba_id,3,3) = -R_ItoG;
      A2.block(v_id,bg_id,3,3) = -skew_x(-_gravity) * R_ItoG;

      A3.block(p_id,bg_id,3,3) = -skew_x(-_gravity) * R_ItoG;

      W.block(th_id,0,3,3) = R_ItoG;
      W.block(p_id,0,3,3) = skew_x(state->_imu->pos())*R_ItoG;
      W.block(v_id,0,3,3) = skew_x(state->_imu->vel())*R_ItoG;
      W.block(v_id,3,3,3) = R_ItoG;

      W.block(bg_id,6,3,3) = Eigen::Matrix3d::Identity();
      W.block(ba_id,9,3,3) = Eigen::Matrix3d::Identity(); 
      
      if(state->_features_SLAM.size()>0)
      {
          auto iter = state->_features_SLAM.begin();
          while(iter != state->_features_SLAM.end())
          {
              if(iter->second->_feat_representation!=LandmarkRepresentation::GLOBAL_3D)
              {
                  printf(RED "Only Support GLOBAL_3D Landmarks right now, EXITING!#!@#!@#\n" RESET);
                  std::exit(EXIT_FAILURE);
              }
              A.block(landmark_id,bg_id,3,3) = -skew_x(iter->second->get_xyz(false))*R_ItoG;
              W.block(landmark_id,0,3,3) = skew_x(iter->second->get_xyz(false))*R_ItoG;
              landmark_id += iter->second->size();
              iter++;
          }
      }
      total_size = landmark_id;

      if(state->set_transform)
      {
          int f6dof_id = landmark_id;
          A.block(f6dof_id,bg_id,3,3) = -skew_x(state->transform_map_to_vio->pos())*R_ItoG;
          W.block(f6dof_id,0,3,3) = skew_x(state->transform_map_to_vio->pos())*R_ItoG;
          landmark_id += state->transform_map_to_vio->p()->size();
      }
      total_size = landmark_id;
      assert(total_size==prop_size);
    }
    else if(state->_options.use_gt){
      Eigen::Matrix<double,17,1> state_last = Eigen::Matrix<double,17,1>::Zero();
      Eigen::Matrix<double,17,1> state_cur = Eigen::Matrix<double,17,1>::Zero();
      bool success1 = state->get_state(data_minus.timestamp,state_last);
      IMU* imu_last = new IMU();
      imu_last->set_value(state_last.block(1,0,16,1));
      assert(success1);
      bool success2 = state->get_state(data_plus.timestamp,state_cur);
      assert(success2);
      IMU* imu_cur = new IMU();
      imu_cur->set_value(state_cur.block(1,0,16,1));
      R_ItoG = imu_last->Rot().transpose();

      A.block(th_id,bg_id,3,3) = -R_ItoG;
      A.block(p_id,v_id,3,3) = Eigen::Matrix3d::Identity();
      A.block(p_id,bg_id,3,3) = -skew_x(imu_last->pos())*R_ItoG;
      A.block(v_id,th_id,3,3) = skew_x(-_gravity);
      A.block(v_id,bg_id,3,3) = -skew_x(imu_last->vel())*R_ItoG;
      A.block(v_id,ba_id,3,3) = -R_ItoG;

      A2.block(p_id,th_id,3,3) = skew_x(-_gravity);
      A2.block(p_id,bg_id,3,3) = -skew_x(imu_last->vel()) * R_ItoG;
      A2.block(p_id,ba_id,3,3) = -R_ItoG;
      A2.block(v_id,bg_id,3,3) = -skew_x(-_gravity) * R_ItoG;

      A3.block(p_id,bg_id,3,3) = -skew_x(-_gravity) * R_ItoG;

      W.block(th_id,0,3,3) = R_ItoG;
      W.block(p_id,0,3,3) = skew_x(imu_last->pos())*R_ItoG;
      W.block(v_id,0,3,3) = skew_x(imu_last->vel())*R_ItoG;
      W.block(v_id,3,3,3) = R_ItoG;

      W.block(bg_id,6,3,3) = Eigen::Matrix3d::Identity();
      W.block(ba_id,9,3,3) = Eigen::Matrix3d::Identity(); 
      
      if(state->_features_SLAM.size()>0)
      {
          auto iter = state->_features_SLAM.begin();
          while(iter != state->_features_SLAM.end())
          {
              if(iter->second->_feat_representation!=LandmarkRepresentation::GLOBAL_3D)
              {
                  printf(RED "Only Support GLOBAL_3D Landmarks right now, EXITING!#!@#!@#\n" RESET);
                  std::exit(EXIT_FAILURE);
              }
              A.block(landmark_id,bg_id,3,3) = -skew_x(iter->second->get_xyz(false))*R_ItoG;
              W.block(landmark_id,0,3,3) = skew_x(iter->second->get_xyz(false))*R_ItoG;
              landmark_id += iter->second->size();
              iter++;
          }
      }
      total_size = landmark_id;

      if(state->set_transform)
      {
          Vector3d p_mapinvio = Vector3d::Zero();
          
          int f6dof_id = landmark_id;

          // A.block(f6dof_id,bg_id,3,3) = -skew_x(state->transform_map_to_vio->pos())*R_ItoG;
          // W.block(f6dof_id,0,3,3) = skew_x(state->transform_map_to_vio->pos())*R_ItoG;
          A.block(f6dof_id,bg_id,3,3) = -skew_x(p_mapinvio)*R_ItoG;
          W.block(f6dof_id,0,3,3) = skew_x(p_mapinvio)*R_ItoG;
          landmark_id += state->transform_map_to_vio->p()->size();
      }
      total_size = landmark_id;
      assert(total_size==prop_size);
    }
    

    // cout<<"A: "<<endl<<A<<endl<<"W: "<<endl<<W<<endl;

    //here we still use continuous covariance
    Eigen::MatrixXd Qc = Eigen::MatrixXd::Zero(12,12);
    Qc.block(0,0,3,3) = _noises.sigma_w_2*Eigen::Matrix<double,3,3>::Identity();
    Qc.block(3,3,3,3) = _noises.sigma_a_2*Eigen::Matrix<double,3,3>::Identity();
    Qc.block(6,6,3,3) = _noises.sigma_wb_2*Eigen::Matrix<double,3,3>::Identity();
    Qc.block(9,9,3,3) = _noises.sigma_ab_2*Eigen::Matrix<double,3,3>::Identity();
    // cout<<"Qc: "<<endl<<Qc<<endl;


    F = Eigen::MatrixXd::Identity(prop_size,prop_size) + A*dt + 0.5*A2*dt*dt + (1.0/6.0)*A3*dt*dt*dt;
    Qd = F*W*Qc*W.transpose()*F.transpose()*dt;
    Qd = 0.5*(Qd+Qd.transpose());

    // cout<<"F: "<<endl<<F<<endl<<"Qd: "<<endl<<Qd<<endl; 

    //Now replace imu estimate and fej with propagated values
    Eigen::Matrix<double,16,1> imu_x = state->_imu->value();
    imu_x.block(0,0,4,1) = new_q;
    imu_x.block(4,0,3,1) = new_p;
    imu_x.block(7,0,3,1) = new_v;
    state->_imu->set_value(imu_x);
    state->_imu->set_fej(imu_x);
    state->_imu->set_linp(imu_x);

}




void Propagator::predict_mean_discrete(State *state, double dt,
                                        const Eigen::Vector3d &w_hat1, const Eigen::Vector3d &a_hat1,
                                        const Eigen::Vector3d &w_hat2, const Eigen::Vector3d &a_hat2,
                                        Eigen::Vector4d &new_q, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p) {

    // If we are averaging the IMU, then do so
    Eigen::Vector3d w_hat = w_hat1;
    Eigen::Vector3d a_hat = a_hat1;
    if (state->_options.imu_avg) {
        w_hat = .5*(w_hat1+w_hat2);
        a_hat = .5*(a_hat1+a_hat2);
    }

    // Pre-compute things
    double w_norm = w_hat.norm();
    Eigen::Matrix<double,4,4> I_4x4 = Eigen::Matrix<double,4,4>::Identity();
    Eigen::Matrix<double,3,3> R_Gtoi = state->_imu->Rot();

    //here we use Roation matrix to propagate orientation
    Eigen::Matrix3d R_itoG= state->_imu->Rot().transpose();

    // Orientation: Equation (101) and (103) and of Trawny indirect TR
    Eigen::Matrix<double,4,4> bigO;
    Eigen::Matrix3d R_itoG_new = R_itoG * exp_so3(w_hat * dt);
    
    new_q = rot_2_quat(R_itoG_new.transpose());
    //new_q = rot_2_quat(exp_so3(-w_hat*dt)*R_Gtoi);

    // Velocity: just the acceleration in the local frame, minus global gravity
    new_v = state->_imu->vel() + R_Gtoi.transpose()*a_hat*dt - _gravity*dt;

    // Position: just velocity times dt, with the acceleration integrated twice
    new_p = state->_imu->pos() + state->_imu->vel()*dt + 0.5*R_Gtoi.transpose()*a_hat*dt*dt - 0.5*_gravity*dt*dt;

}



void Propagator::predict_mean_rk4(State *state, double dt,
                                  const Eigen::Vector3d &w_hat1, const Eigen::Vector3d &a_hat1,
                                  const Eigen::Vector3d &w_hat2, const Eigen::Vector3d &a_hat2,
                                  Eigen::Vector4d &new_q, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p) {

    // Pre-compute things
    Eigen::Vector3d w_hat = w_hat1;
    Eigen::Vector3d a_hat = a_hat1;
    Eigen::Vector3d w_alpha = (w_hat2-w_hat1)/dt;
    Eigen::Vector3d a_jerk = (a_hat2-a_hat1)/dt;

    // y0 ================
    Eigen::Vector4d q_0 = state->_imu->quat();
    Eigen::Vector3d p_0 = state->_imu->pos();
    Eigen::Vector3d v_0 = state->_imu->vel();

    // k1 ================
    Eigen::Vector4d dq_0 = {0,0,0,1};
    Eigen::Vector4d q0_dot = 0.5*Omega(w_hat)*dq_0;
    Eigen::Vector3d p0_dot = v_0;
    Eigen::Matrix3d R_Gto0 = quat_2_Rot(quat_multiply(dq_0,q_0));
    Eigen::Vector3d v0_dot = R_Gto0.transpose()*a_hat-_gravity;

    Eigen::Vector4d k1_q = q0_dot*dt;
    Eigen::Vector3d k1_p = p0_dot*dt;
    Eigen::Vector3d k1_v = v0_dot*dt;

    // k2 ================
    w_hat += 0.5*w_alpha*dt;
    a_hat += 0.5*a_jerk*dt;

    Eigen::Vector4d dq_1 = quatnorm(dq_0+0.5*k1_q);
    //Eigen::Vector3d p_1 = p_0+0.5*k1_p;
    Eigen::Vector3d v_1 = v_0+0.5*k1_v;

    Eigen::Vector4d q1_dot = 0.5*Omega(w_hat)*dq_1;
    Eigen::Vector3d p1_dot = v_1;
    Eigen::Matrix3d R_Gto1 = quat_2_Rot(quat_multiply(dq_1,q_0));
    Eigen::Vector3d v1_dot = R_Gto1.transpose()*a_hat-_gravity;

    Eigen::Vector4d k2_q = q1_dot*dt;
    Eigen::Vector3d k2_p = p1_dot*dt;
    Eigen::Vector3d k2_v = v1_dot*dt;

    // k3 ================
    Eigen::Vector4d dq_2 = quatnorm(dq_0+0.5*k2_q);
    //Eigen::Vector3d p_2 = p_0+0.5*k2_p;
    Eigen::Vector3d v_2 = v_0+0.5*k2_v;

    Eigen::Vector4d q2_dot = 0.5*Omega(w_hat)*dq_2;
    Eigen::Vector3d p2_dot = v_2;
    Eigen::Matrix3d R_Gto2 = quat_2_Rot(quat_multiply(dq_2,q_0));
    Eigen::Vector3d v2_dot = R_Gto2.transpose()*a_hat-_gravity;

    Eigen::Vector4d k3_q = q2_dot*dt;
    Eigen::Vector3d k3_p = p2_dot*dt;
    Eigen::Vector3d k3_v = v2_dot*dt;

    // k4 ================
    w_hat += 0.5*w_alpha*dt;
    a_hat += 0.5*a_jerk*dt;

    Eigen::Vector4d dq_3 = quatnorm(dq_0+k3_q);
    //Eigen::Vector3d p_3 = p_0+k3_p;
    Eigen::Vector3d v_3 = v_0+k3_v;

    Eigen::Vector4d q3_dot = 0.5*Omega(w_hat)*dq_3;
    Eigen::Vector3d p3_dot = v_3;
    Eigen::Matrix3d R_Gto3 = quat_2_Rot(quat_multiply(dq_3,q_0));
    Eigen::Vector3d v3_dot = R_Gto3.transpose()*a_hat-_gravity;

    Eigen::Vector4d k4_q = q3_dot*dt;
    Eigen::Vector3d k4_p = p3_dot*dt;
    Eigen::Vector3d k4_v = v3_dot*dt;

    // y+dt ================
    Eigen::Vector4d dq = quatnorm(dq_0+(1.0/6.0)*k1_q+(1.0/3.0)*k2_q+(1.0/3.0)*k3_q+(1.0/6.0)*k4_q);
    new_q = quat_multiply(dq, q_0);
    new_p = p_0+(1.0/6.0)*k1_p+(1.0/3.0)*k2_p+(1.0/3.0)*k3_p+(1.0/6.0)*k4_p;
    new_v = v_0+(1.0/6.0)*k1_v+(1.0/3.0)*k2_v+(1.0/3.0)*k3_v+(1.0/6.0)*k4_v;

}


void Propagator::predictNewState(State *state, double dt,
                                  const Eigen::Vector3d &w_hat1, const Eigen::Vector3d &a_hat1,
                                  const Eigen::Vector3d &w_hat2, const Eigen::Vector3d &a_hat2,
                                  Eigen::Vector4d &new_q, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p) {

  // TODO: Will performing the forward integration using
  //    the inverse of the quaternion give better accuracy?
  Eigen::Vector3d gyro = w_hat1;
  Eigen::Vector3d acc = a_hat1;
  if (state->_options.imu_avg) {
        gyro = .5*(w_hat1+w_hat2);
        acc = .5*(a_hat1+a_hat2);
    }

  double gyro_norm = gyro.norm();
  Matrix4d Omega = Matrix4d::Zero();
  Omega.block<3, 3>(0, 0) = -skew_x(gyro);
  Omega.block<3, 1>(0, 3) = gyro;
  Omega.block<1, 3>(3, 0) = -gyro;

  Vector4d q = state->_imu->quat();
  Vector3d v = state->_imu->vel();
  Vector3d p = state->_imu->pos();

  // Some pre-calculation
  Vector4d dq_dt, dq_dt2;
  if (gyro_norm > 1e-5) {
    dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;
    dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;
  }
  else {
    dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *
      cos(gyro_norm*dt*0.5) * q;
    dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *
      cos(gyro_norm*dt*0.25) * q;
  }

  Matrix3d dR_dt_transpose = quat_2_Rot(dq_dt).transpose();
  Matrix3d dR_dt2_transpose = quat_2_Rot(dq_dt2).transpose();  

  // k1 = f(tn, yn)
  Vector3d k1_v_dot = quat_2_Rot(q).transpose()*acc - _gravity;
  Vector3d k1_p_dot = v;

  // k2 = f(tn+dt/2, yn+k1*dt/2)
  Vector3d k1_v = v + k1_v_dot*dt/2;
  Vector3d k2_v_dot = dR_dt2_transpose*acc - _gravity;
  Vector3d k2_p_dot = k1_v;

  // k3 = f(tn+dt/2, yn+k2*dt/2)
  Vector3d k2_v = v + k2_v_dot*dt/2;
  Vector3d k3_v_dot = dR_dt2_transpose*acc- _gravity;
  Vector3d k3_p_dot = k2_v;

  // k4 = f(tn+dt, yn+k3*dt)
  Vector3d k3_v = v + k3_v_dot*dt;
  Vector3d k4_v_dot = dR_dt_transpose*acc - _gravity;
  Vector3d k4_p_dot = k3_v;

  // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  new_q = dq_dt;
  quatnorm(q);
  new_v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
  new_p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);

  return;
}

