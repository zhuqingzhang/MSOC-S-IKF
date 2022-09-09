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
#include "StateHelper.h"
#include <boost/filesystem.hpp>


using namespace ov_core;
using namespace ov_rimsckf;


void StateHelper::EKFPropagation(State *state, const std::vector<Type*> &order_NEW, const std::vector<Type*> &order_OLD,
                                 const Eigen::MatrixXd &Phi, const Eigen::MatrixXd &Q) {

    // We need at least one old and new variable
    if (order_NEW.empty() || order_OLD.empty()) {
        printf(RED "StateHelper::EKFPropagation() - Called with empty variable arrays!\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    //Phi is in order of R_ItoG, p_IinG, v_IinG, bg, ba, features_landmarks, 6DOF features
    //we should collect their covariance from the state->_Cov and compose the covariance in order.
    
    std::vector<int> Phi_id; //location of each variable in Phi
    std::vector<int> _Cov_id; //location of each variable in state->_Cov;
    std::vector<int> var_size; //the size of each variable
    int current_id = 0;
    int size_order_OLD = 0;
    cout<<"order_OLD size: "<<order_OLD.size()<<endl;
    for (Type *var:order_OLD)
    {
        Phi_id.push_back(current_id);
        _Cov_id.push_back(var->id());
        cout<<"var id:"<<var->id()<<endl;
        var_size.push_back(var->size());
        cout<<"var size:"<<var->size()<<endl;
        current_id += var->size();
        size_order_OLD += var->size();
        cout<<"size_order_OLD: "<<size_order_OLD<<endl;
    }
    cout<<"ekfpropagator:: size_order_OLD: "<< size_order_OLD<<endl;
    //other variables that are not propagated with IMU
    std::vector<Type*> order_notprop;
    std::vector<int> id_notprop;  
    std::vector<int> var_size_notprop;
    int notprop_size = 0;
    if(state->set_transform)
    {
        if(order_OLD.size()==1) //only have imu state, it is zupt.
        {
            order_notprop.push_back(state->transform_map_to_vio);
            id_notprop.push_back(notprop_size);
            var_size_notprop.push_back(state->transform_map_to_vio->size());
            notprop_size += state->transform_map_to_vio->size();
        }
        else
        {
            order_notprop.push_back(state->transform_map_to_vio->q());
            id_notprop.push_back(notprop_size);
            var_size_notprop.push_back(state->transform_map_to_vio->q()->size());
            notprop_size += state->transform_map_to_vio->q()->size();
        }
    }

    auto iter = state->_clones_IMU.begin();
    while(iter!=state->_clones_IMU.end())
    {
        order_notprop.push_back(iter->second);
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(iter->second->size());
        notprop_size += iter->second->size();
        iter++;
    }
    
    for (int i = 0; i < state->_options.num_cameras; i++) {

        if(state->_options.do_calib_camera_pose)
        {
            order_notprop.push_back(state->_calib_IMUtoCAM[i]);
            id_notprop.push_back(notprop_size);
            var_size_notprop.push_back(state->_calib_IMUtoCAM[i]->size());
            notprop_size += state->_calib_IMUtoCAM[i]->size();
        }
        
        if (state->_options.do_calib_camera_intrinsics) {
            order_notprop.push_back(state->_cam_intrinsics[i]);
            id_notprop.push_back(notprop_size);
            var_size_notprop.push_back(state->_cam_intrinsics[i]->size());
            notprop_size += state->_cam_intrinsics[i]->size();
        }
    }

    //time delay
    if (state->_options.do_calib_camera_timeoffset) {
        order_notprop.push_back(state->_calib_dt_CAMtoIMU);
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(state->_calib_dt_CAMtoIMU->size());
        notprop_size += state->_calib_dt_CAMtoIMU->size();
    }
    

    if(state->zupt)
    {
      auto it=state->_features_SLAM.begin();
      while(it!=state->_features_SLAM.end())
      {
        order_notprop.push_back(it->second);
        id_notprop.push_back(notprop_size);
        var_size_notprop.push_back(it->second->size());
        notprop_size += it->second->size();
        it++;
      }
    }

    cout<<"ekfpropagator:: notprop_size: "<< notprop_size<<endl;
    cout<<"Cov size: "<<state->_Cov.rows()<<endl;
    
    
    assert(Phi.cols()==size_order_OLD);
    assert(state->_Cov.rows()==(size_order_OLD + notprop_size));
    cout<<"1"<<endl;
    //collect covariance from state->_Cov
    Eigen::MatrixXd Cov = Eigen::MatrixXd::Zero(size_order_OLD,size_order_OLD);
    for(int i=0; i<_Cov_id.size(); i++)
    {
        for(int j=0; j<_Cov_id.size(); j++)
        {
            Cov.block(Phi_id[i],Phi_id[j],var_size[i],var_size[j]) = 
            state->_Cov.block(_Cov_id[i],_Cov_id[j],var_size[i],var_size[j]);
        }
    }
    cout<<"2"<<endl;
    // cout<<"collected Cov: "<<endl<<Cov<<endl;
    // cout<<"original Cov: "<<endl<<state->_Cov.block(0,0,state->_imu->size(),state->_imu->size())<<endl;
    //collect cross_covariance from state->_Cov
    Eigen::MatrixXd Cross_Cov = Eigen::MatrixXd::Zero(size_order_OLD,notprop_size);
    for(int i=0; i<_Cov_id.size(); i++)
    {
        for(int j=0; j<order_notprop.size(); j++)
        {
            Cross_Cov.block(Phi_id[i],id_notprop[j],var_size[i],var_size_notprop[j]) = 
            state->_Cov.block(_Cov_id[i],order_notprop[j]->id(),var_size[i],var_size_notprop[j]);
        }
    }
    cout<<"3"<<endl;
    // cout<<"collected Cross_Cov: "<<endl<<Cross_Cov<<endl;
    // cout<<"original Cross_cov: "<<endl<<state->_Cov.block(0,state->_imu->size(),size_order_OLD,notprop_size)<<endl;


    //propagate the covariance
    Eigen::MatrixXd Cov_new = Phi*Cov*Phi.transpose() + Q;
    Eigen::MatrixXd Cross_Cov_new = Phi*Cross_Cov;
    cout<<"4"<<endl;
    //reassgin the Cov_new to state->_Cov
    for(int i=0; i<_Cov_id.size(); i++)
    {
        for(int j=0; j<_Cov_id.size();j++)
        {
            state->_Cov.block(_Cov_id[i],_Cov_id[j],var_size[i],var_size[j]) =
            Cov_new.block(Phi_id[i],Phi_id[j],var_size[i],var_size[j]);
        }
        
        for(int k=0; k<order_notprop.size(); k++)
        {
            state->_Cov.block(_Cov_id[i],order_notprop[k]->id(),var_size[i],var_size_notprop[k]) =
            Cross_Cov_new.block(Phi_id[i],id_notprop[k],var_size[i],var_size_notprop[k]);

            state->_Cov.block(order_notprop[k]->id(),_Cov_id[i],var_size_notprop[k],var_size[i]) =
            Cross_Cov_new.block(Phi_id[i],id_notprop[k],var_size[i],var_size_notprop[k]).transpose();
        }
    }
    cout<<"5"<<endl;
    // cout<<"updated Cov_new: "<<endl<<Cov_new<<endl;
    // cout<<"updated original Cov: "<<endl<<state->_Cov.block(0,0,state->_imu->size(),state->_imu->size())<<endl;
    // cout<<"updated Cross_Cov_new: "<<endl<<Cross_Cov_new<<endl;
    // cout<<"updated original Cross_Cov: "<<endl<<state->_Cov.block(0,state->_imu->size(),size_order_OLD,notprop_size)<<endl;

    //check negative
    Eigen::VectorXd diags = state->_Cov.diagonal();
    cout<<"6"<<endl;
    bool found_neg = false;
    for(int i=0; i<diags.rows(); i++) {
        if(diags(i) < 0.0) {
            printf(RED "StateHelper::EKFPropagation() - diagonal at %d is %.2f\n" RESET,i,diags(i));
            found_neg = true;
        }
    }
    assert(!found_neg);
    

    //if we have nuisance part, we should propagte the nuisance covariance
    //Phi_CovAN =[Phi] [P_AN]
    if(state->_options.use_schmidt)
    {
        Eigen::MatrixXd Phi_CovAN = Eigen::MatrixXd::Zero(Phi.rows(),state->_Cov_nuisance.cols());
        if(state->_nuisance_variables.size()>0)
        {
            for(int i=0; i<_Cov_id.size(); i++)
            {
                Phi_CovAN.noalias() += Phi.block(0,Phi_id[i],Phi.rows(),var_size[i]) * 
                                    state->_Cross_Cov_AN.block(_Cov_id[i],0,var_size[i],state->_Cov_nuisance.cols());
            }
            //Update state->_Cross_Cov_AN
            for(int i=0; i<_Cov_id.size(); i++)
            {
                state->_Cross_Cov_AN.block(_Cov_id[i],0,var_size[i],state->_Cov_nuisance.cols()) = 
                Phi_CovAN.block(Phi_id[i],0,var_size[i],Phi_CovAN.cols());
            }
        }
    }
    


    // // We are good to go!
    // int start_id = order_NEW.at(0)->id();
    // int phi_size = Phi.rows();
    // int total_size = state->_Cov.rows();
    // state->_Cov.block(start_id,0,phi_size,total_size) = Cov_PhiT.transpose();
    // state->_Cov.block(0,start_id,total_size,phi_size) = Cov_PhiT;
    // state->_Cov.block(start_id,start_id,phi_size,phi_size) = Phi_Cov_PhiT;

  

}



void StateHelper::EKFUpdate(State *state, const std::vector<Type *> &H_order, const Eigen::MatrixXd &H,
                            const Eigen::VectorXd &res, const Eigen::MatrixXd &R) {

    //==========================================================
    //==========================================================
    // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
    assert(res.rows() == R.rows());
    assert(H.rows() == res.rows());
    Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());

    // Get the location in small jacobian for each measuring variable
    int current_it = 0;
    std::vector<int> H_id;
    for (Type *meas_var: H_order) {
        H_id.push_back(current_it);
        current_it += meas_var->size();
    }
    int state_size=current_it;

    //==========================================================
    //==========================================================
    // For each active variable find its M = P*H^T
    for (Type *var: state->_variables) {
        // Sum up effect of each subjacobian = K_i= \sum_m (P_im Hm^T)
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for (size_t i = 0; i < H_order.size(); i++) {
            Type *meas_var = H_order[i];
            M_i.noalias() += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                             H.block(0, H_id[i], H.rows(), meas_var->size()).transpose();
        }
        M_a.block(var->id(), 0, var->size(), res.rows()) = M_i;
    }

    //==========================================================
    //==========================================================
    // Get covariance of the involved terms
    Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);

    // Residual covariance S = H*Cov*H' + R
    Eigen::MatrixXd S(R.rows(), R.rows());
    S.triangularView<Eigen::Upper>() = H * P_small * H.transpose();
    S.triangularView<Eigen::Upper>() += R;
    //Eigen::MatrixXd S = H * P_small * H.transpose() + R;

    // Invert our S (should we use a more stable method here??)
    Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
    S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
    Eigen::MatrixXd K = M_a * Sinv.selfadjointView<Eigen::Upper>();
    //Eigen::MatrixXd K = M_a * S.inverse();

    // Update Covariance
    state->_Cov.triangularView<Eigen::Upper>() -= K * M_a.transpose();
    state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
    //Cov -= K * M_a.transpose();
    //Cov = 0.5*(Cov+Cov.transpose());

    // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
    Eigen::VectorXd diags = state->_Cov.diagonal();
    bool found_neg = false;
    for(int i=0; i<diags.rows(); i++) {
        if(diags(i) < 0.0) {
            printf(RED "StateHelper::EKFUpdate() - diagonal at %d is %.2f\n" RESET,i,diags(i));
            found_neg = true;
        }
    }
    assert(!found_neg);

    // Calculate our delta and update all our active states
    Eigen::MatrixXd delta_res = Eigen::MatrixXd::Zero(res.rows(),1);
        if(state->iter!=false)// we have relinearized value for map update;
        {
            //x_est-x_lin
            Eigen::VectorXd x_error=Eigen::VectorXd::Zero(state_size);
            int id=0;
            for (Type *meas_var: H_order) {
                if(meas_var->size()==6)
                {
                    // PoseJPL* var=dynamic_cast<PoseJPL*>(meas_var);
                    // Eigen::Matrix3d R_est=var->Rot();
                    // Eigen::Matrix3d R_lin=var->Rot_linp();
                    // Eigen::Vector3d R_error=Eigen::Vector3d::Zero();
                    // Eigen::Vector3d p_error=var->pos_linp()-R_lin*R_est.transpose()*var->pos();
                    // Eigen::VectorXd pos_error=Eigen::VectorXd::Zero(6);
                    // pos_error<<R_error,p_error;
                    // x_error.block(id,0,meas_var->size(),1)=pos_error;
                    if(meas_var->id()==state->transform_map_to_vio->id())
                    {
                      PoseJPL* var=dynamic_cast<PoseJPL*>(meas_var);
                      Eigen::Matrix3d R_est=state->_imu->Rot().transpose();
                      Eigen::Matrix3d R_lin=state->_imu->Rot_linp().transpose();
                      Eigen::Vector4d q_est=var->q()->value();
                      Eigen::Vector4d q_lin=var->q()->linp();
                      Eigen::Vector3d q_error = compute_error(q_est,q_lin);
                      Eigen::Vector3d p_error = var->pos()-var->pos_linp();
                      Eigen::VectorXd pos_error=Eigen::VectorXd::Zero(6);
                      pos_error<<q_error,p_error;
                      x_error.block(id,0,meas_var->size(),1)=pos_error;
                    }
                    else
                    {
                      PoseJPL* var=dynamic_cast<PoseJPL*>(meas_var);
                      Eigen::Matrix3d R_est=var->Rot();
                      Eigen::Matrix3d R_lin=var->Rot_linp();
                      Eigen::Vector4d q_est=var->q()->value();
                      Eigen::Vector4d q_lin=var->q()->linp();
                      Eigen::Vector3d q_error = compute_error(q_est,q_lin);
                      Eigen::Vector3d R_error=Eigen::Vector3d::Zero();
                      Eigen::Vector3d p_error = var->pos()-var->pos_linp();
                      Eigen::VectorXd pos_error=Eigen::VectorXd::Zero(6);
                      pos_error<<q_error,p_error;
                      x_error.block(id,0,meas_var->size(),1)=pos_error;
                    }
                }
                else
                {
                    x_error.block(id,0,meas_var->size(),1)=meas_var->value()-meas_var->linp();
                }
                id+=meas_var->size();
            }
            assert(id==state_size);
            // cout<<"x_error: "<<x_error.transpose()<<endl;
            // sleep(10);
            //nuisnace part value are not update, so its value() and linp() are always the same, so x_error is zero
            //so here we dont need to consider the nuisance part.
            delta_res=H*x_error;
        }

    Eigen::VectorXd dx = K * (res-delta_res);
 
    
    UpdateVarInvariant(state, dx);
    // cout<<"imu state position: "<<state->_imu->pos().transpose()<<" imu clone state postion: "<<state->_clones_IMU[state->_timestamp]->pos().transpose()<<endl;
    // cout<<"imu state orientation: "<<state->_imu->quat().transpose()<<" imu clone state orientation: "<<state->_clones_IMU[state->_timestamp]->quat().transpose()<<endl;
    cout<<"state timestamp: "<<to_string(state->_timestamp)<<" imu orientation: "<<state->_imu->quat().transpose()<<" imu position: "<<state->_imu->pos().transpose()<<endl;
    // for (size_t i = 0; i < state->_variables.size(); i++) {
    //     state->_variables.at(i)->update(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
    // }

}

void StateHelper::UpdateVarInvariant(State *state, const Eigen::VectorXd dx)
{
    
    //first update imu stuff
    VectorXd delta_x_imu = dx.block(state->_imu->id(),0,state->_imu->size(),1);

    cout<<"0"<<endl;

    if (delta_x_imu.segment<3>(6).norm() > 0.5 ||
        delta_x_imu.segment<3>(3).norm() > 1.0) {
        printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
        printf("delta position: %f\n", delta_x_imu.segment<3>(3).norm());
        printf(YELLOW "update is too large");
        //return;
    }
    Matrix<double,3,2> d_imu; //velocity and position
    d_imu.block<3,1>(0,0) = expSE_3(delta_x_imu.head<3>(),delta_x_imu.segment<3>(6));
    d_imu.block<3,1>(0,1) = expSE_3(delta_x_imu.head<3>(),delta_x_imu.segment<3>(3));
    //Note that dR is I to G; state->_imu->Rot() is R_GtoI
    Matrix3d dR = exp_so3(delta_x_imu.head<3>());
    state->_imu->q()->set_value(rot_2_quat(state->_imu->Rot()*dR.transpose()));
    state->_imu->v()->set_value(dR * state->_imu->vel() + d_imu.block<3,1>(0,0));
    state->_imu->p()->set_value(dR * state->_imu->pos() + d_imu.block<3,1>(0,1));
    state->_imu->bg()->set_value(state->_imu->bias_g() + delta_x_imu.segment<3>(9));
    state->_imu->ba()->set_value(state->_imu->bias_a() + delta_x_imu.segment<3>(12));
    //don't forget to set the value of imu!!
    Matrix<double,16,1> imu_value;
    imu_value<<state->_imu->quat(),state->_imu->pos(),state->_imu->vel(),state->_imu->bias_g(),state->_imu->bias_a();
    state->_imu->set_value(imu_value);
    state->_imu->set_linp(imu_value);
    cout<<"1"<<endl;
    
    //update feature_SLAM if we have
    auto feat = state->_features_SLAM.begin();
    while(feat!=state->_features_SLAM.end())
    {
    //    cout<<"landmark id: "<<feat->second->_featid<<" before update: "<<feat->second->value().transpose();
        Vector3d d_feat = expSE_3(delta_x_imu.head<3>(),dx.segment<3>(feat->second->id()));
        feat->second->set_value(dR * feat->second->get_xyz(false) + d_feat);
        feat->second->set_linp(dR * feat->second->get_xyz(false) + d_feat);
        // cout<<"after update: "<<feat->second->value().transpose()<<endl;
        feat++;

    }
    cout<<"2"<<endl;
    
    //update 6DOF feature if we have
    if(state->set_transform)
    {
        Vector3d d_trans = expSE_3(delta_x_imu.head<3>(),dx.segment<3>(state->transform_map_to_vio->p()->id()));
        state->transform_map_to_vio->p()->set_value(dR * state->transform_map_to_vio->pos() + d_trans);
        dR = exp_so3(dx.segment<3>(state->transform_map_to_vio->q()->id()));
        state->transform_map_to_vio->q()->set_value(rot_2_quat(dR * state->transform_map_to_vio->Rot()));
        Matrix<double,7,1> pose;
        pose<<state->transform_map_to_vio->quat(),state->transform_map_to_vio->pos();
        state->transform_map_to_vio->set_value(pose);
        state->transform_map_to_vio->set_linp(pose);
    }
    cout<<"3"<<endl;

    //update clone_IMU
    auto clone = state->_clones_IMU.begin();
    while(clone != state->_clones_IMU.end())
    {
        VectorXd delta_clone = dx.segment<6>(clone->second->id());
        Vector3d dx_clone = expSE_3(delta_clone.head<3>(),delta_clone.tail<3>());
        dR = exp_so3(delta_clone.head<3>());
        clone->second->q()->set_value(rot_2_quat(clone->second->Rot() * dR.transpose()));
        clone->second->p()->set_value(dR * clone->second->pos() + dx_clone);
        Matrix<double,7,1> pose;
        pose<<clone->second->quat(),clone->second->pos();
        clone->second->set_value(pose);
        clone->second->set_linp(pose);
        clone++;
    }
    cout<<"4"<<endl;

    //extrinsics and intrinsics
    for (int i = 0; i < state->_options.num_cameras; i++) {

        if(state->_options.do_calib_camera_pose)
        {
            Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM[i]->Rot();
            Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM[i]->pos();
            Eigen::Matrix3d R_CtoI = R_ItoC.transpose();
            Eigen::Vector3d p_CinI = -R_CtoI * p_IinC;
            Eigen::VectorXd dx_CtoI = dx.block(state->_calib_IMUtoCAM[i]->id(),0,state->_calib_IMUtoCAM[i]->size(),1);
            assert(dx_CtoI.rows()==6);
            dR = exp_so3(dx_CtoI.head<3>());
            Eigen::Vector3d dx = expSE_3(dx_CtoI.head<3>(),dx_CtoI.tail<3>());
            p_CinI = dR * p_CinI + dx;
            R_CtoI = dR * R_CtoI;
            R_ItoC = R_CtoI.transpose();
            p_IinC = -R_ItoC*p_CinI;
            Eigen::Vector4d q_ItoC = rot_2_quat(R_ItoC);
            Eigen::Matrix<double,7,1> pose_ItoC;
            pose_ItoC<<q_ItoC,p_IinC;
            state->_calib_IMUtoCAM[i]->set_value(pose_ItoC);
            state->_calib_IMUtoCAM[i]->set_linp(pose_ItoC);
        }
        
        if (state->_options.do_calib_camera_intrinsics) {
            state->_cam_intrinsics[i]->update_linp(dx.block(state->_cam_intrinsics[i]->id(),0,state->_cam_intrinsics[i]->size(),1));
            state->_cam_intrinsics[i]->update(dx.block(state->_cam_intrinsics[i]->id(),0,state->_cam_intrinsics[i]->size(),1));
        }
    }
    cout<<"5"<<endl;

    //time delay
    if (state->_options.do_calib_camera_timeoffset) {
        state->_calib_dt_CAMtoIMU->update_linp(dx.block(state->_calib_dt_CAMtoIMU->id(),0,state->_calib_dt_CAMtoIMU->size(),1));
        state->_calib_dt_CAMtoIMU->update(dx.block(state->_calib_dt_CAMtoIMU->id(),0,state->_calib_dt_CAMtoIMU->size(),1));
    }
    cout<<"6"<<endl; 
  
}


Eigen::MatrixXd StateHelper::ConstructSEN(State *state)
{
    /* R p v f1...fN p_trans  0
       0 1 0  0... 0    0     0
       0 0 1  0... 0    0     0
       0 0 0  1... 0    0     0
       . . .  .... .    .     .
       0 0 0  0... 1    0     0
       0 0 0  0... 0    1     0
       0 0 0  0... 0    0    R_trans  

    */
    
    int size_big = 0;
    size_big += state->_imu->q()->size();
    size_big += 2; // p,v
    if(state->_features_SLAM.size()>0)
    {
        size_big += state->_features_SLAM.size();
    }
    if(state->set_transform)
    {
        size_big += 1; //p_trans
        size_big += state->transform_map_to_vio->q()->size();
    }

    Eigen::MatrixXd SEN = Eigen::MatrixXd::Identity(size_big,size_big);
    int id=0;
    SEN.block(0,id,3,3) = state->_imu->Rot().transpose();
    id += 3;
    SEN.block(0,id,3,1) = state->_imu->pos();
    id += 1;
    SEN.block(0,id,3,1) = state->_imu->vel();
    id += 1;
    auto iter = state->_features_SLAM.begin();
    while(iter != state->_features_SLAM.end())
    {
        SEN.block(0,id,3,1) = iter->second->get_xyz(false);
        id += 1;
        iter++;
    }
    if(state->set_transform)
    {
        SEN.block(0,id,3,1) = state->transform_map_to_vio->pos();
        id += 1;
        SEN.block(id,id,3,3) = state->transform_map_to_vio->Rot();
        id += 3;
    }
    assert(id == size_big);

    return SEN;

}


Eigen::VectorXd StateHelper::Construct_dxSEN(State *state, const Eigen::VectorXd dx)
{
    int size_big = 0;
    size_big += state->_imu->pose()->size();
    size_big += state->_imu->v()->size();
    if(state->_features_SLAM.size()>0)
    {
        size_big += state->_features_SLAM.size() * 3;
    }
    if(state->set_transform)
    {
        size_big += state->transform_map_to_vio->size();
    }

    Eigen::VectorXd dx_SEN = Eigen::VectorXd::Zero(size_big,1);
    int id = 0;
    dx_SEN.block(id,0,state->_imu->q()->size(),1) = dx.block(state->_imu->q()->id(),0,state->_imu->q()->size(),1);
    id += state->_imu->q()->size();
    dx_SEN.block(id,0,state->_imu->p()->size(),1) = dx.block(state->_imu->p()->id(),0,state->_imu->p()->size(),1);
    id += state->_imu->p()->size();
    dx_SEN.block(id,0,state->_imu->v()->size(),1) = dx.block(state->_imu->v()->id(),0,state->_imu->v()->size(),1);
    id += state->_imu->v()->size();

    auto iter = state->_features_SLAM.begin();
    while(iter != state->_features_SLAM.end())
    {
        dx_SEN.block(id,0,iter->second->size(),1) = dx.block(iter->second->id(),0,iter->second->size(),1);
        id += iter->second->size();
        iter++;
    }

    if(state->set_transform)
    {
        dx_SEN.block(id,0,state->transform_map_to_vio->p()->size(),1) = 
        dx.block(state->transform_map_to_vio->p()->id(),0,state->transform_map_to_vio->p()->size(),1);
        id += state->transform_map_to_vio->p()->size();
        dx_SEN.block(id,0,state->transform_map_to_vio->q()->size(),1) = 
        dx.block(state->transform_map_to_vio->q()->id(),0,state->transform_map_to_vio->q()->size(),1);
        id += state->transform_map_to_vio->q()->size();
    }
    assert(id == size_big);

    return dx_SEN;
}


void StateHelper::SeparateSEN(State *state, Eigen::MatrixXd X_SEN)
{
    int id=0;
    Eigen::Matrix3d R_IinG = X_SEN.block(0,id,3,3);
    Eigen::Vector4d q_GinI = rot_2_quat(R_IinG.transpose());
    id += 3;
    Eigen::Vector3d p_IinG = X_SEN.block(0,id,3,1);
    id += 1;
    Eigen::Vector3d v_IinG = X_SEN.block(0,id,3,1);
    id += 1;
    state->_imu->q()->set_value(q_GinI);
    state->_imu->p()->set_value(p_IinG);
    state->_imu->v()->set_value(v_IinG);
    
    auto iter = state->_features_SLAM.begin();
    while(iter != state->_features_SLAM.end())
    {
        cout<<"landmark id: "<<iter->second->_featid<<" before update: "<<iter->second->value().transpose();
        iter->second->set_value(X_SEN.block(0,id,3,1));
        cout<<"after update: "<<iter->second->value().transpose()<<endl;
        id += 1;
        iter++;
    }
    if(state->set_transform)
    {
        Eigen::Vector3d p_WinG = X_SEN.block(0,id,3,1);
        id += 1;
        Eigen::Matrix3d R_WtoG = X_SEN.block(id,id,3,3);
        Eigen::Vector4d q_WtoG = rot_2_quat(R_WtoG);
        id += 3;
        Eigen::Matrix<double,7,1> trans_WtoG;
        trans_WtoG<<q_WtoG,p_WinG;
        state->transform_map_to_vio->set_value(trans_WtoG);
    }
    assert(id == X_SEN.rows());
}


void StateHelper::UpdateClonePose(State *state, const Eigen::VectorXd dx)
{
    auto iter = state->_clones_IMU.begin();
    while(iter != state->_clones_IMU.end())
    {
        int id = iter->second->id();
        Eigen::Matrix3d R_clonetoG = iter->second->Rot().transpose();
        Eigen::Vector3d p_cloneinG = iter->second->pos();
        Eigen::Matrix4d Pose_SE3 = Eigen::Matrix4d::Identity();
        Pose_SE3.block(0,0,3,3) = R_clonetoG;
        Pose_SE3.block(0,3,3,1) = p_cloneinG;
        Eigen::VectorXd dx_clone = dx.block(id,0,iter->second->size(),1);
        Eigen::Matrix4d dX_clone = exp_se3(dx_clone);
        Pose_SE3 = dX_clone * Pose_SE3;
        R_clonetoG = Pose_SE3.block(0,0,3,3);
        p_cloneinG = Pose_SE3.block(0,3,3,1);
        Eigen::Vector4d q_Gtoclone = rot_2_quat(R_clonetoG.transpose());
        Eigen::Matrix<double,7,1> pose_clone;
        pose_clone<<q_Gtoclone,p_cloneinG;
        iter->second->set_value(pose_clone);

        iter++;
    }
}

void StateHelper::UpdateParameters(State *state, const Eigen::VectorXd dx)
{
    //first update bg and ba;
    int id = state->_imu->bg()->id();
    cout<<"bg id: "<<id;
    state->_imu->bg()->update(dx.block(id,0,state->_imu->bg()->size(),1));
    id = state->_imu->ba()->id();
    cout<<" ba id: "<<id<<endl;
    state->_imu->ba()->update(dx.block(id,0,state->_imu->ba()->size(),1));

    //extrinsics and intrinsics
    for (int i = 0; i < state->_options.num_cameras; i++) {

        if(state->_options.do_calib_camera_pose)
        {
            Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM[i]->Rot();
            Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM[i]->pos();
            Eigen::Matrix4d Pose_CtoI = Eigen::Matrix4d::Identity();
            Pose_CtoI.block(0,0,3,3) = R_ItoC.transpose();
            Pose_CtoI.block(0,3,3,1) = -R_ItoC.transpose() * p_IinC;
            id = state->_calib_IMUtoCAM[i]->id();
            Eigen::VectorXd dx_CtoI = dx.block(id,0,state->_calib_IMUtoCAM[i]->size(),1);
            Eigen::Matrix4d dX_CtoI = exp_se3(dx_CtoI);
            Pose_CtoI = dX_CtoI * Pose_CtoI;
            R_ItoC = Pose_CtoI.block(0,0,3,3).transpose();
            p_IinC = -R_ItoC * Pose_CtoI.block(0,3,3,1);
            Eigen::Vector4d q_ItoC = rot_2_quat(R_ItoC);
            Eigen::Matrix<double,7,1> pose_ItoC;
            pose_ItoC<<q_ItoC,p_IinC;
            state->_calib_IMUtoCAM[i]->set_value(pose_ItoC);
        }
        
        if (state->_options.do_calib_camera_intrinsics) {
            id = state->_cam_intrinsics[i]->id();
            state->_cam_intrinsics[i]->update(dx.block(id,0,state->_cam_intrinsics[i]->size(),1));
        }
    }

    //time delay
    if (state->_options.do_calib_camera_timeoffset) {
        id = state->_calib_dt_CAMtoIMU->id();
        state->_calib_dt_CAMtoIMU->update(dx.block(id,0,state->_calib_dt_CAMtoIMU->size(),1));
    }
}
void StateHelper::init_transform_update(State *state, const std::vector<Type *> &Ha_order, const std::vector<Type *> &Ht_order, const std::vector<Type *> &Hn_order,
                                           const Eigen::MatrixXd &Ha,const Eigen::MatrixXd &Ht, const Eigen::MatrixXd &H_nuisance,
                                            const Eigen::VectorXd &res, Eigen::MatrixXd &R){

    cout<<"in StateHelper::SKFUpdate"<<endl;
    assert(res.rows() == R.rows());
    assert(Ha.rows() == res.rows());
    assert(Ha.rows()== H_nuisance.rows());
    assert(Ht.rows() == Ha.rows());
    int nuisance_size=state->_Cov_nuisance.rows();
    int trans_size = state->transform_map_to_vio->size();
    int active_size=state->_Cov.rows()- trans_size;



    
    //* M_a = P_aa * H_a.transpose()
    Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(active_size, res.rows());
    //* M_n = P_nn * H_n.transpose();
    Eigen::MatrixXd M_n = Eigen::MatrixXd::Zero(nuisance_size, res.rows());

    int current_it = 0;
    std::vector<int> Ha_id;
    for(Type *meas_var: Ha_order)
    {
      Ha_id.push_back(current_it);
      current_it += meas_var->size();
    }
    
    current_it = 0;
    std::vector<int> Ht_id;
    for(Type *meas_var: Ht_order)
    {
      Ht_id.push_back(current_it);
      current_it += meas_var->size();
    }

    current_it=0;
    std::vector<int> Hn_id;
    for (Type *meas_n_var: Hn_order)
    {
        Hn_id.push_back(current_it);
        current_it+=meas_n_var->size();
    }

    Eigen::MatrixXd Paa_small = StateHelper::get_marginal_covariance(state, Ha_order);
    Eigen::MatrixXd Pnn_small = StateHelper::get_marginal_covariance(state, Hn_order);
    Eigen::MatrixXd A = Ha*Paa_small*Ha.transpose() + H_nuisance*Pnn_small*H_nuisance.transpose() + R;
    Eigen::MatrixXd Ainv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
    A.selfadjointView<Eigen::Upper>().llt().solveInPlace(Ainv);
    Eigen::MatrixXd tmp = (Ht.transpose() * Ainv * Ht ).inverse();
    Eigen::MatrixXd S_inv = Ainv - Ainv * Ht * tmp * Ht.transpose() * Ainv;



    //==========================================================
    //==========================================================
    // For each active variable find its M = P*H^T

    //[ Pxx   Pxn][ Hx^T]  -->[ Pxx*Hx^T+Pxn*Hn^T ] = [ L_active ]
    //[ Pxn^T Pnn][ Hn^T]     [Pxn^T*Hx^T+Pnn*Hn^T]   [L_nuisance]

    //K=M*S^-1 --> [ L_active ] * S^-1
    //             [L_nuisance]
    //1: Pxx*Hx^T
    cout<<"1"<<endl;
    bool skip_trans = false;
    for (Type *var: state->_variables)
    {
      if(var == state->transform_map_to_vio)
      {
          skip_trans = true;
          continue;
      }
      Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
      for(size_t i=0; i<Ha_order.size();i++)
      {
        Type *meas_var = Ha_order[i];
        M_i.noalias() += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                             Ha.block(0, Ha_id[i], Ha.rows(), meas_var->size()).transpose();
      }
      if(skip_trans)
      {
        M_a.block(var->id()-state->transform_map_to_vio->size(),0,var->size(),res.rows()).noalias() += M_i;
      }
      else
      {
        M_a.block(var->id(),0,var->size(),res.rows()).noalias() += M_i;
      }
      

    }
    assert(skip_trans);


    //4: Pnn*Hn^T
    for(Type *var: state->_nuisance_variables){
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(),res.rows());
        Eigen::MatrixXd M_i_last = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for(size_t i=0; i<Hn_order.size();i++){
            Type *meas_n_var=Hn_order[i];
            M_i.noalias()+=state->_Cov_nuisance.block(var->id(),meas_n_var->id(),var->size(),meas_n_var->size()) *
                           H_nuisance.block(0,Hn_id[i],H_nuisance.rows(),meas_n_var->size()).transpose();

        }
        M_n.block(var->id(),0,var->size(),res.rows()).noalias() += M_i;

    }

    Eigen::VectorXd error_a = M_a * S_inv * res;

    Eigen::VectorXd error_t = tmp * Ht.transpose()*Ainv *res;

    Eigen::VectorXd error = Eigen::VectorXd::Zero(state->_Cov.rows());

    assert(error.rows() == error_a.rows()+error_t.rows());
    int trans_id = state->transform_map_to_vio->id();
    error.block(0,0,trans_id,1) = error_a.block(0,0,trans_id,1);
    error.block(trans_id,0,trans_size,1) = error_t;
    error.block(trans_id+trans_size,0,error_a.rows()-trans_id,1) = error_a.block(trans_id,0,error_a.rows()-trans_id,1);

    vector<Type*> Ha_order_big;
    for(Type *var: state->_variables)
    {
      if(var == state->transform_map_to_vio)
      {
        continue;
      }
      Ha_order_big.push_back(var);
    }

    Eigen::MatrixXd P_AA = StateHelper::get_marginal_covariance(state,Ha_order_big);

    Eigen::MatrixXd P_AA_new = P_AA - M_a * S_inv * M_a.transpose();

    Eigen::MatrixXd P_AT_new = -M_a * Ainv * Ht * tmp;

    Eigen::MatrixXd P_TT_new = tmp;

    Eigen::MatrixXd P_AN_new = -M_a * S_inv * M_n.transpose();

    Eigen::MatrixXd P_TN_new = -tmp * Ht.transpose() * Ainv * M_n.transpose();

    //* relative transformation should be the last variable in the state by now;
    assert(state->_Cov.rows()-state->transform_map_to_vio->size() == state->transform_map_to_vio->id());
    assert(active_size == state->transform_map_to_vio->id());
    assert(P_AA.rows() == active_size);
    //*update cov

    state->_Cov.block(0,0,P_AA.rows(),P_AA.cols()) = P_AA_new;

    state->_Cov.block(active_size,active_size,trans_size,trans_size) = P_TT_new;

    state->_Cov.block(0,active_size,active_size,trans_size) = P_AT_new;

    state->_Cov.block(active_size,0,trans_size,active_size) = P_AT_new.transpose();

    state->_Cross_Cov_AN.block(0,0,active_size,nuisance_size) = P_AN_new;
   
    state->_Cross_Cov_AN.block(active_size,0,trans_size,nuisance_size) = P_TN_new;

    Eigen::VectorXd diags = state->_Cov.diagonal();
      bool found_neg = false;
      for(int i=0; i<diags.rows(); i++) {
          if(diags(i) < 0.0) {
              printf(RED "StateHelper::EKFUpdate() - diagonal at %d is %.2f\n" RESET,i,diags(i));
              found_neg = true;
          }
      }
      assert(!found_neg);



    //==========================================================
    //==========================================================

   
    UpdateVarInvariant(state, error);

        
}
    
    






void StateHelper::SKFUpdate(State *state, const std::vector<Type *> &Hx_order, const std::vector<Type *> &Hn_order,
                            const Eigen::MatrixXd &Hx, const Eigen::MatrixXd &H_nuisance,
                            const Eigen::VectorXd &res,  Eigen::MatrixXd &R, bool iterative) {
    //==========================================================
    //==========================================================
    // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
    cout<<"in StateHelper::SKFUpdate"<<endl;
    assert(res.rows() == R.rows());
    assert(Hx.rows() == res.rows());
    assert(Hx.rows()==H_nuisance.rows());
    int nuisance_size=state->_Cov_nuisance.rows();
    int active_size=state->_Cov.rows();
    Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(active_size+nuisance_size, res.rows());
    cout<<"active and nuisnace size:"<<active_size<<" "<<nuisance_size<<endl;
    Eigen::MatrixXd M_a_last = Eigen::MatrixXd::Zero(active_size+nuisance_size, res.rows());
    Eigen::MatrixXd Hx_last;
    Eigen::MatrixXd Hn_last;

    // Get the location in small jacobian for each measuring variable
    int current_it = 0;
    std::vector<int> Hx_id;
    for (Type *meas_var: Hx_order) {
        Hx_id.push_back(current_it);
        current_it += meas_var->size();
    }
    int state_size=current_it;

    current_it=0;
    std::vector<int> Hn_id;
    for (Type *meas_n_var: Hn_order)
    {
        Hn_id.push_back(current_it);
        current_it+=meas_n_var->size();
    }

    //==========================================================
    //==========================================================
    // For each active variable find its M = P*H^T

    //[ Pxx   Pxn][ Hx^T]  -->[ Pxx*Hx^T+Pxn*Hn^T ] = [ L_active ]
    //[ Pxn^T Pnn][ Hn^T]     [Pxn^T*Hx^T+Pnn*Hn^T]   [L_nuisance]

    //K=M*S^-1 --> [ L_active ] * S^-1
    //             [L_nuisance]
    //1: Pxx*Hx^T
    cout<<"1"<<endl;

    for (Type *var: state->_variables) {
        // Sum up effect of each subjacobian = K_i= \sum_m (P_im Hm^T)
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
        Eigen::MatrixXd M_i_last = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for (size_t i = 0; i < Hx_order.size(); i++) {
            Type *meas_var = Hx_order[i];
            M_i.noalias() += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                             Hx.block(0, Hx_id[i], Hx.rows(), meas_var->size()).transpose();
            // if(state->iter)
            // {
            //     M_i_last.noalias() += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
            //                  Hx_last.block(0, Hx_id[i], Hx.rows(), meas_var->size()).transpose();
            // }
            
        }
        M_a.block(var->id(), 0, var->size(), res.rows()).noalias() += M_i;
        // if(state->iter)
        // {
        //      M_a_last.block(var->id(), 0, var->size(), res.rows()).noalias() += M_i_last;
        // }
    }
    //2: Pxn*Hn^T
    cout<<"2"<<endl;
    for(Type *var: state->_variables){
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(),res.rows());
        Eigen::MatrixXd M_i_last = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for (size_t i=0; i < Hn_order.size();i++){
            Type *meas_n_var = Hn_order[i];
            M_i.noalias()+=state->_Cross_Cov_AN.block(var->id(),meas_n_var->id(),var->size(),meas_n_var->size()) *
                           H_nuisance.block(0,Hn_id[i],H_nuisance.rows(),meas_n_var->size()).transpose();
            // if(state->iter)
            // {
            //     M_i_last.noalias() += state->_Cross_Cov_AN.block(var->id(),meas_n_var->id(),var->size(),meas_n_var->size()) *
            //                Hn_last.block(0,Hn_id[i],H_nuisance.rows(),meas_n_var->size()).transpose();
            // }
        }
        M_a.block(var->id(),0,var->size(),res.rows()).noalias() += M_i;
        // if(state->iter)
        // {
        //      M_a_last.block(var->id(), 0, var->size(), res.rows()).noalias() += M_i_last;
        // }
    }
    cout<<"Now we finish the L_active"<<endl;
    //Now we finish the L_active.

    //3: Pxn^T*Hx^T
    for(Type *var: state->_nuisance_variables){
        Eigen::MatrixXd M_i= Eigen::MatrixXd::Zero(var->size(),res.rows());
        Eigen::MatrixXd M_i_last = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for (size_t i=0; i<Hx_order.size();i++){
            Type *meas_var = Hx_order[i];
            M_i.noalias()+=state->_Cross_Cov_AN.block(meas_var->id(),var->id(),meas_var->size(),var->size()).transpose()*
                    Hx.block(0,Hx_id[i],Hx.rows(),meas_var->size()).transpose();
            // if(state->iter)
            // {
            //     M_i_last.noalias() += state->_Cross_Cov_AN.block(meas_var->id(),var->id(),meas_var->size(),var->size()).transpose()*
            //         Hx_last.block(0,Hx_id[i],Hx.rows(),meas_var->size()).transpose();
            // }
        }
        M_a.block(active_size+var->id(),0,var->size(),res.rows()).noalias() += M_i;
        // if(state->iter)
        // {
        //      M_a_last.block(active_size+var->id(), 0, var->size(), res.rows()).noalias() += M_i_last;
        // }
    }

    //4: Pnn*Hn^T
    for(Type *var: state->_nuisance_variables){
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(),res.rows());
        Eigen::MatrixXd M_i_last = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for(size_t i=0; i<Hn_order.size();i++){
            Type *meas_n_var=Hn_order[i];
            M_i.noalias()+=state->_Cov_nuisance.block(var->id(),meas_n_var->id(),var->size(),meas_n_var->size()) *
                           H_nuisance.block(0,Hn_id[i],H_nuisance.rows(),meas_n_var->size()).transpose();
            // if(state->iter)
            // {
            //     M_i_last.noalias() += state->_Cov_nuisance.block(var->id(),meas_n_var->id(),var->size(),meas_n_var->size()) *
            //                Hn_last.block(0,Hn_id[i],H_nuisance.rows(),meas_n_var->size()).transpose();
            // }
        }
        M_a.block(active_size+var->id(),0,var->size(),res.rows()).noalias() += M_i;
        // if(state->iter)
        // {
        //      M_a_last.block(active_size+var->id(), 0, var->size(), res.rows()).noalias() += M_i_last;
        // }
    }
    cout<<"Now we finish the L_nuisance"<<endl;
    //Now we finish the L_nuisance.



    //==========================================================
    //==========================================================
    // Get covariance of the involved terms
    Eigen::MatrixXd Pxx_small = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd Pxn_small = StateHelper::get_marginal_cross_covariance(state,Hx_order,Hn_order);
    Eigen::MatrixXd Pnn_small = StateHelper::get_marginal_nuisance_covariance(state,Hn_order);
    if(H_nuisance.cols()>0)
    { 
        cout<<"Pxx_small max: "<<Pxx_small.maxCoeff()<<" Pxn_small max: "<<Pxn_small.maxCoeff()<<" Pnn_small max: "<<Pnn_small.maxCoeff()<<endl;
        cout<<"Hx max: "<<Hx.maxCoeff()<<" Hn max: "<<H_nuisance.maxCoeff()<<endl;
    }
    else
    {
    
        cout<<"Pxx_small max: "<<Pxx_small.maxCoeff()<<endl;
        cout<<"Hx max: "<<Hx.maxCoeff()<<endl;
    }
    
        
    cout<<"Pxx_small size:"<<Pxx_small.rows()<<"*"<<Pxx_small.cols()<<endl;
    cout<<"Pxn_small size:"<<Pxn_small.rows()<<"*"<<Pxn_small.cols()<<endl;
    cout<<"Pnn_small size:"<<Pnn_small.rows()<<"*"<<Pnn_small.cols()<<endl;
    cout<<"Hx size:"<<Hx.rows()<<"*"<<Hx.cols()<<endl;
    cout<<"Hn size:"<<H_nuisance.rows()<<"*"<<H_nuisance.cols()<<endl;
    // double h=state->_options.opt_thred*res.rows();
    // double h=1000;

    
    // Residual covariance S = H*Cov*H' + R
    Eigen::MatrixXd S(R.rows(), R.rows());
    Eigen::MatrixXd S_last(R.rows(), R.rows());
    Eigen::MatrixXd A(R.rows(),R.rows());

    cout<<"S size:"<<S.rows()<<"*"<<S.cols()<<endl;
    S = Hx * Pxx_small * Hx.transpose();
    cout<<"1"<<endl;
    S += Hx * Pxn_small * H_nuisance.transpose();
    cout<<"2"<<endl;
    S += (Hx * Pxn_small * H_nuisance.transpose()).transpose();
    cout<<"3"<<endl;
    S += H_nuisance * Pnn_small * H_nuisance.transpose();
    cout<<"4"<<endl;
    S += R;
    cout<<"Now we finish the S"<<endl;
   

    // Invert our S (should we use a more stable method here??)
    Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
    S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
    cout<<"Sinv"<<endl;

    Eigen::MatrixXd L_active=M_a.block(0,0,active_size,M_a.cols());
    Eigen::MatrixXd L_nuisance=M_a.block(active_size,0,nuisance_size,M_a.cols());
    
    cout<<"L_active and L_nuisance"<<endl;

    Eigen::MatrixXd K_active=L_active * Sinv.selfadjointView<Eigen::Upper>();
    cout<<"K_active"<<endl;
    
    assert(iterative == false);
    if(!iterative)
    {
        // Update Covariance
        // if(state->iter==false)
        // {
            state->_Cov.triangularView<Eigen::Upper>() -= K_active*L_active.transpose();
            cout<<"Cov"<<endl;
            state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
            cout<<"Cov"<<endl;
            cout<<"K_active size:"<<K_active.rows()<<"*"<<K_active.cols()<<endl;
            cout<<"L_nuisance size:"<<L_nuisance.rows()<<"*"<<L_nuisance.cols()<<endl;
            cout<<"state->_Cross_Cov_AN size:"<<state->_Cross_Cov_AN.rows()<<"*"<<state->_Cross_Cov_AN.cols()<<endl;
            state->_Cross_Cov_AN -= K_active*L_nuisance.transpose();
            cout<<"Now we finish the Cov and Cross_Cov update"<<endl;
        // }
        
        

        // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
        Eigen::VectorXd diags = state->_Cov.diagonal();
        bool found_neg = false;
        for(int i=0; i<diags.rows(); i++) {
            if(diags(i) < 0.0) {
                printf(RED "StateHelper::EKFUpdate() - diagonal at %d is %.2f\n" RESET,i,diags(i));
                found_neg = true;
            }
        }
        assert(!found_neg);
    
        // Calculate our delta and update all our active states
        // and keep nuisance part untouched.
        cout<<"res size: "<<res.rows()<<"*"<<res.cols()<<endl;

        Eigen::MatrixXd delta_res = Eigen::MatrixXd::Zero(res.rows(),1);
        if(state->iter!=false)// we have relinearized value for map update;
        {
            //x_est-x_lin
            Eigen::VectorXd x_error=Eigen::VectorXd::Zero(state_size);
            int id=0;
            bool mark=0;
            for (Type *meas_var: Hx_order) {
                if(meas_var->size()==6)
                {
                    // PoseJPL* var=dynamic_cast<PoseJPL*>(meas_var);
                    // Eigen::Matrix3d R_est=var->Rot();
                    // Eigen::Matrix3d R_lin=var->Rot_linp();
                    // Eigen::Vector3d R_error=Eigen::Vector3d::Zero();
                    // Eigen::Vector3d p_error=var->pos_linp()-R_lin*R_est.transpose()*var->pos();
                    // Eigen::VectorXd pos_error=Eigen::VectorXd::Zero(6);
                    // pos_error<<R_error,p_error;
                    // x_error.block(id,0,meas_var->size(),1)=pos_error;
                    if(meas_var->id()==state->transform_map_to_vio->id())
                    {
                      PoseJPL* var=dynamic_cast<PoseJPL*>(meas_var);
                      Eigen::Matrix3d R_est=state->_imu->Rot().transpose();
                      Eigen::Matrix3d R_lin=state->_imu->Rot_linp().transpose();
                      Eigen::Vector4d q_est=var->q()->value();
                      Eigen::Vector4d q_lin=var->q()->linp();
                      Eigen::Vector3d q_error = compute_error(q_est,q_lin);
                      Eigen::Matrix3d R_error = R_est*R_lin.transpose();
                      Eigen::Vector3d p_error = var->pos()-R_est*R_lin.transpose()*var->pos_linp();
                      Eigen::Matrix4d Pos_error = Eigen::Matrix4d::Identity();
                      Pos_error.block(0,0,3,3) = R_error;
                      Pos_error.block(0,3,3,1) = p_error;
                      // Eigen::VectorXd pos_error=Eigen::VectorXd::Zero(6);
                      // pos_error<<q_error,p_error;
                      Eigen::VectorXd pos_error=log_se3(Pos_error);
                      x_error.block(id,0,meas_var->size(),1)=pos_error;
                      // cout<<"transform_error: "<<endl<<pos_error.transpose()<<endl;
                    }
                    else
                    {
                      PoseJPL* var=dynamic_cast<PoseJPL*>(meas_var);
                      Eigen::Matrix3d R_est=var->Rot().transpose();
                      Eigen::Matrix3d R_lin=var->Rot_linp().transpose();
                      Eigen::Vector4d q_est=rot_2_quat(R_est);
                      Eigen::Vector4d q_lin=rot_2_quat(R_lin);
                      Eigen::Vector3d q_error = compute_error(q_est,q_lin);
                      Eigen::Matrix3d R_error=R_est*R_lin.transpose();
                      Eigen::Vector3d p_error = var->pos()-R_est*R_lin.transpose()*var->pos_linp();
                      Eigen::Matrix4d Pos_error = Eigen::Matrix4d::Identity();
                      Pos_error.block(0,0,3,3) = R_error;
                      Pos_error.block(0,3,3,1) = p_error;
                      // Eigen::VectorXd pos_error=Eigen::VectorXd::Zero(6);
                      // pos_error<<q_error,p_error;
                      Eigen::VectorXd pos_error=log_se3(Pos_error);
                      x_error.block(id,0,meas_var->size(),1)=pos_error;
                      // cout<<"pos_error: "<<endl<<pos_error.transpose()<<endl;
                    }
                }
                else
                {
                    x_error.block(id,0,meas_var->size(),1)=meas_var->value()-meas_var->linp();
                }
                id+=meas_var->size();
            }
            // assert(mark==1);
            assert(id==state_size);
            //nuisnace part value are not update, so its value() and linp() are always the same, so x_error is zero
            //so here we dont need to consider the nuisance part.
            delta_res=Hx*x_error;
            
        }

        Eigen::VectorXd dx = K_active * (res-delta_res);
        // cout<<"dx: "<<endl<<dx.transpose()<<endl;
        UpdateVarInvariant(state, dx);
        if(state->set_transform)
        {
          // cout<<"dx: "<<endl<<dx<<endl;
          // cout<<"K_active: "<<K_active<<endl<<"res: "<<res<<endl;
        }
        
        // sleep(5);
        // cout<<"imu state position: "<<state->_imu->pos().transpose()<<" imu clone state postion: "<<state->_clones_IMU[state->_timestamp]->pos().transpose()<<endl;
        // cout<<"imu state orientation: "<<state->_imu->quat().transpose()<<" imu clone state orientation: "<<state->_clones_IMU[state->_timestamp]->quat().transpose()<<endl;
        // cout<<"state timestamp: "<<to_string(state->_timestamp)<<" imu orientation: "<<state->_imu->quat().transpose()<<" imu position: "<<state->_imu->pos().transpose()<<endl;
        
    }
    
    

}


void StateHelper::EKFMAPUpdate(State *state, const std::vector<Type *> &Hx_order, const std::vector<Type *> &Hn_order,
                            const Eigen::MatrixXd &Hx, const Eigen::MatrixXd &H_nuisance,
                            const Eigen::VectorXd &res, const Eigen::MatrixXd &R, bool iterative) {
    //==========================================================
    //==========================================================
    // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
    cout<<"in StateHelper::SKFUpdate"<<endl;
    assert(res.rows() == R.rows());
    assert(Hx.rows() == res.rows());
    assert(Hx.rows()==H_nuisance.rows());
    int nuisance_size=state->_Cov_nuisance.rows();
    int active_size=state->_Cov.rows();
    Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(active_size+nuisance_size, res.rows());
    cout<<"active and nuisnace size:"<<active_size<<" "<<nuisance_size<<endl;



    // Get the location in small jacobian for each measuring variable
    int current_it = 0;
    std::vector<int> Hx_id;
    for (Type *meas_var: Hx_order) {
        Hx_id.push_back(current_it);
        current_it += meas_var->size();
    }

    current_it=0;
    std::vector<int> Hn_id;
    for (Type *meas_n_var: Hn_order)
    {
        Hn_id.push_back(current_it);
        current_it+=meas_n_var->size();
    }

    //==========================================================
    //==========================================================
    // For each active variable find its M = P*H^T

    //[ Pxx   Pxn][ Hx^T]  -->[ Pxx*Hx^T+Pxn*Hn^T ] = [ L_active ]
    //[ Pxn^T Pnn][ Hn^T]     [Pxn^T*Hx^T+Pnn*Hn^T]   [L_nuisance]

    //K=M*S^-1 --> [ L_active ] * S^-1
    //             [L_nuisance]
    //1: Pxx*Hx^T
    cout<<"1"<<endl;

    for (Type *var: state->_variables) {
        // Sum up effect of each subjacobian = K_i= \sum_m (P_im Hm^T)
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for (size_t i = 0; i < Hx_order.size(); i++) {
            Type *meas_var = Hx_order[i];
            M_i.noalias() += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                             Hx.block(0, Hx_id[i], Hx.rows(), meas_var->size()).transpose();
        }
        M_a.block(var->id(), 0, var->size(), res.rows()).noalias() += M_i;
    }
    //2: Pxn*Hn^T
    cout<<"2"<<endl;
    for(Type *var: state->_variables){
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(),res.rows());
        for (size_t i=0; i < Hn_order.size();i++){
            Type *meas_n_var = Hn_order[i]; 
            M_i.noalias()+=state->_Cross_Cov_AN.block(var->id(),meas_n_var->id(),var->size(),meas_n_var->size()) *
                           H_nuisance.block(0,Hn_id[i],H_nuisance.rows(),meas_n_var->size()).transpose();
        }
        M_a.block(var->id(),0,var->size(),res.rows()).noalias() += M_i;
    }
    cout<<"Now we finish the L_active"<<endl;
    //Now we finish the L_active.

    //3: Pxn^T*Hx^T
    for(Type *var: state->_nuisance_variables){
        Eigen::MatrixXd M_i= Eigen::MatrixXd::Zero(var->size(),res.rows());
        for (size_t i=0; i<Hx_order.size();i++){
            Type *meas_var = Hx_order[i];
            M_i.noalias()+=state->_Cross_Cov_AN.block(meas_var->id(),var->id(),meas_var->size(),var->size()).transpose()*
                    Hx.block(0,Hx_id[i],Hx.rows(),meas_var->size()).transpose();
        }
        M_a.block(active_size+var->id(),0,var->size(),res.rows()).noalias() += M_i;
    }

    //4: Pnn*Hn^T
    for(Type *var: state->_nuisance_variables){
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(),res.rows());
        for(size_t i=0; i<Hn_order.size();i++){
            Type *meas_n_var=Hn_order[i];
            M_i.noalias()+=state->_Cov_nuisance.block(var->id(),meas_n_var->id(),var->size(),meas_n_var->size()) *
                           H_nuisance.block(0,Hn_id[i],H_nuisance.rows(),meas_n_var->size()).transpose();
        }
        M_a.block(active_size+var->id(),0,var->size(),res.rows()).noalias() += M_i;
    }
    cout<<"Now we finish the L_nuisance"<<endl;
    //Now we finish the L_nuisance.



    //==========================================================
    //==========================================================
    // Get covariance of the involved terms
    Eigen::MatrixXd Pxx_small = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd Pxn_small = StateHelper::get_marginal_cross_covariance(state,Hx_order,Hn_order);
    Eigen::MatrixXd Pnn_small = StateHelper::get_marginal_nuisance_covariance(state,Hn_order);
    if(H_nuisance.cols()>0)
    { 
        cout<<"Pxx_small max: "<<Pxx_small.maxCoeff()<<" Pxn_small max: "<<Pxn_small.maxCoeff()<<" Pnn_small max: "<<Pnn_small.maxCoeff()<<endl;
        cout<<"Hx max: "<<Hx.maxCoeff()<<" Hn max: "<<H_nuisance.maxCoeff()<<endl;
    }
    else
    {
    
        cout<<"Pxx_small max: "<<Pxx_small.maxCoeff()<<endl;
        cout<<"Hx max: "<<Hx.maxCoeff()<<endl;
    }
    
        
    cout<<"Pxx_small size:"<<Pxx_small.rows()<<"*"<<Pxx_small.cols()<<endl;
    cout<<"Pxn_small size:"<<Pxn_small.rows()<<"*"<<Pxn_small.cols()<<endl;
    cout<<"Pnn_small size:"<<Pnn_small.rows()<<"*"<<Pnn_small.cols()<<endl;
    cout<<"Hx size:"<<Hx.rows()<<"*"<<Hx.cols()<<endl;
    cout<<"Hn size:"<<H_nuisance.rows()<<"*"<<H_nuisance.cols()<<endl;
    // Residual covariance S = H*Cov*H' + R
    Eigen::MatrixXd S(R.rows(), R.rows());
    cout<<"S size:"<<S.rows()<<"*"<<S.cols()<<endl;
    S = Hx * Pxx_small * Hx.transpose();
    cout<<"1"<<endl;
    S += Hx * Pxn_small * H_nuisance.transpose();
    cout<<"2"<<endl;
    S += (Hx * Pxn_small * H_nuisance.transpose()).transpose();
    cout<<"3"<<endl;
    S += H_nuisance * Pnn_small * H_nuisance.transpose();
    cout<<"4"<<endl;
    S += R;
    cout<<"Now we finish the S"<<endl;
    
    //Eigen::MatrixXd S = H * P_small * H.transpose() + R;


    // Invert our S (should we use a more stable method here??)
    Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
    S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
    // Sinv=S.inverse();
    cout<<"Sinv"<<endl;

    Eigen::VectorXd diags_Sinv = Sinv.diagonal();
    double max_Sinv=diags_Sinv.maxCoeff();
    cout<<"max_Sinv: "<<max_Sinv<<endl;
    bool found_neg_Sinv = false;
    for(int i=0; i<diags_Sinv.rows(); i++) {
        if(diags_Sinv(i) < 0.0) {
            printf(RED "StateHelper::EKFUpdate() Sinv - diagonal at %d is %.2f\n" RESET,i,diags_Sinv(i));
            found_neg_Sinv = true;
        }
    }

    Eigen::MatrixXd L_active=M_a.block(0,0,active_size,M_a.cols());
    Eigen::MatrixXd L_nuisance=M_a.block(active_size,0,nuisance_size,M_a.cols());
    cout<<"L_active and L_nuisance"<<endl;

    // Eigen::MatrixXd K_active=L_active * Sinv.selfadjointView<Eigen::Upper>();
     Eigen::MatrixXd K_active=L_active * Sinv;
    cout<<"K_active size: "<<K_active.rows()<<" "<<K_active.cols() <<endl;
    Eigen::MatrixXd K_nuisance=L_nuisance * Sinv;
    // Eigen::MatrixXd K_nuisance=L_nuisance * Sinv.selfadjointView<Eigen::Upper>();
    cout<<"K_nuisance size: "<<K_nuisance.rows()<<" "<<K_nuisance.cols() <<endl;
    


    //Eigen::MatrixXd K = M_a * Sinv.selfadjointView<Eigen::Upper>();
//    Eigen::MatrixXd KHP=Eigen::MatrixXd::Zero(state->_Cov.rows(),state->_Cov.cols());
//    KHP.block(0,0,active_size,active_size)=K_active*L_active.transpose();
//    KHP.block(0,active_size,active_size,nuisance_size)=K_active*L_nuisance.transpose();
//    KHP.block(active_size,0,nuisance_size,active_size)=KHP.block(0,active_size,active_size,nuisance_size).transpose();
   
    // {
    //     Eigen::VectorXd diags = state->_Cov.diagonal();
    //     Eigen::VectorXd diags_L=(L_active*L_active.transpose()).diagonal();
    //     cout<<"diags_L.max: "<<diags_L.maxCoeff()<<endl;
    // bool found_neg = false;
    // for(int i=0; i<diags.rows(); i++) {
    //     if(diags(i)-diags_L(i) < 0.0) {
    //         printf(RED "StateHelper::EKFUpdate() diags-diags_L - diagonal at %d is %.2f\n" RESET,i,diags(i));
    //         found_neg = true;
    //     }
    // }
    // }



    if(!iterative)
    {
        // Update Covariance
        state->_Cov.triangularView<Eigen::Upper>() -= K_active*L_active.transpose();
        // state->_Cov -= K_active*L_active.transpose();
        cout<<"Cov"<<endl;
        state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
        // MatrixXd Cov = 0.5*(state->_Cov+state->_Cov.transpose());
        // state->_Cov=Cov;

        cout<<"Cov"<<endl;
        cout<<"K_active size:"<<K_active.rows()<<"*"<<K_active.cols()<<endl;
        cout<<"L_nuisance size:"<<L_nuisance.rows()<<"*"<<L_nuisance.cols()<<endl;
        cout<<"state->_Cross_Cov_AN size:"<<state->_Cross_Cov_AN.rows()<<"*"<<state->_Cross_Cov_AN.cols()<<endl;
        state->_Cross_Cov_AN -= K_active*L_nuisance.transpose();
        cout<<"Now we finish the Cov and Cross_Cov update"<<endl;
        // state->_Cov_nuisance -= K_nuisance*L_nuisance.transpose();
        // Cov = 0.5*(state->_Cov_nuisance+state->_Cov_nuisance.transpose());
        // state->_Cov_nuisance=Cov;
        state->_Cov_nuisance.triangularView<Eigen::Upper>() -= K_nuisance*L_nuisance.transpose();
        state->_Cov_nuisance = state->_Cov_nuisance.selfadjointView<Eigen::Upper>();

        // if(state->set_transform)
        // {
        //     MatrixXd cov_trans=state->_Cov.block(state->transform_vio_to_map->id(),state->transform_vio_to_map->id(),6,6);
        
        // EigenSolver<Matrix<double,6,6>> es(cov_trans);
        
        // MatrixXd D = es.pseudoEigenvalueMatrix();
        // cout<<D<<endl;
        // }
        

        // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
        Eigen::VectorXd diags = state->_Cov.diagonal();
        bool found_neg = false;
        for(int i=0; i<diags.rows(); i++) {
            if(diags(i) < 0.0) {
                printf(RED "StateHelper::EKFUpdate() - diagonal at %d is %.2f\n" RESET,i,diags(i));
                found_neg = true;
            }
        }
        assert(!found_neg);

        Eigen::VectorXd diags_n = state->_Cov_nuisance.diagonal();
        found_neg = false;
        for(int i=0; i<diags_n.rows(); i++) {
            if(diags_n(i) < 0.0) {
                printf(RED "StateHelper::EKFUpdate() - nuisance diagonal at %d is %.2f\n" RESET,i,diags(i));
                found_neg = true;
            }
        }
        assert(!found_neg);
    
        // Calculate our delta and update all our active states
        // and keep nuisance part untouched.
        Eigen::VectorXd dx = K_active * res;
        for (size_t i = 0; i < state->_variables.size(); i++) {
            //if not literative, the linearized point should be equal to the _value
            state->_variables.at(i)->update_linp(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
            state->_variables.at(i)->update(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
        }
        cout<<"Now we finish the state->variables update"<<endl;


        Eigen::VectorXd dx_n=K_nuisance*res;
        for(size_t i=0;i<state->_nuisance_variables.size();i++){
            state->_nuisance_variables.at(i)->update_linp(dx_n.block(state->_nuisance_variables.at(i)->id(),0,state->_nuisance_variables.at(i)->size(), 1));
            state->_nuisance_variables.at(i)->update(dx_n.block(state->_nuisance_variables.at(i)->id(),0,state->_nuisance_variables.at(i)->size(), 1));
        }
        cout<<"Now we finish the state->nuisance_variables update"<<endl;
    }
    else //we should update linear point of state_variable instead of the _value
    {
        Eigen::VectorXd dx = K_active * res;
        for (size_t i = 0; i < state->_variables.size(); i++) {
        state->_variables.at(i)->update_linp(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
        }
        Eigen::VectorXd dx_n=K_nuisance*res;
        for(size_t i=0;i<state->_nuisance_variables.size();i++){
        state->_nuisance_variables.at(i)->update_linp(dx_n.block(state->_nuisance_variables.at(i)->id(),0,state->_nuisance_variables.at(i)->size(), 1));
            
        }
    }
    

}




Eigen::MatrixXd StateHelper::get_marginal_covariance(State *state, const std::vector<Type *> &small_variables) {

    // Calculate the marginal covariance size we need to make our matrix
    int cov_size = 0;
    for (size_t i = 0; i < small_variables.size(); i++) {
        cov_size += small_variables[i]->size();
    }

    // Construct our return covariance
    Eigen::MatrixXd Small_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

    // For each variable, lets copy over all other variable cross terms
    // Note: this copies over itself to when i_index=k_index
    // cout<<"small_variable size:"<<small_variables.size()<<endl;
    // cout<<"cov_size: "<<cov_size<<endl;
    // cout<<"state->Cov size"<<state->_Cov.rows()<<endl;
    int i_index = 0;
    for (size_t i = 0; i < small_variables.size(); i++) {
        int k_index = 0;
        for (size_t k = 0; k < small_variables.size(); k++) {
            // cout<<"small_variables[i]->id(): "<<small_variables[i]->id()<<" size:"<<small_variables[i]->size()<<endl;
            // cout<<"small_variables[k]->id(): "<<small_variables[k]->id()<<" size:"<<small_variables[k]->size()<<endl;
            Small_cov.block(i_index, k_index, small_variables[i]->size(), small_variables[k]->size()) =
                    state->_Cov.block(small_variables[i]->id(), small_variables[k]->id(), small_variables[i]->size(), small_variables[k]->size());
            k_index += small_variables[k]->size();
        }
        i_index += small_variables[i]->size();
    }

    // Return the covariance
    //Small_cov = 0.5*(Small_cov+Small_cov.transpose());
    return Small_cov;
}

Eigen::MatrixXd StateHelper::get_marginal_nuisance_covariance(State *state,
                                                              const std::vector<Type *> &small_variables) {
    // Calculate the marginal covariance size we need to make our matrix
    int cov_size = 0;
    for (size_t i = 0; i < small_variables.size(); i++) {
        cov_size += small_variables[i]->size();
    }

    // Construct our return covariance
    Eigen::MatrixXd Small_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

    // For each variable, lets copy over all other variable cross terms
    // Note: this copies over itself to when i_index=k_index
    // cout<<"small_variable size:"<<small_variables.size()<<endl;
    // cout<<"cov_size: "<<cov_size<<endl;
    // cout<<"state->Cov size"<<state->_Cov.rows()<<endl;
    int i_index = 0;
    for (size_t i = 0; i < small_variables.size(); i++) {
        int k_index = 0;
        for (size_t k = 0; k < small_variables.size(); k++) {
            // cout<<"small_variables[i]->id(): "<<small_variables[i]->id()<<" size:"<<small_variables[i]->size()<<endl;
            // cout<<"small_variables[k]->id(): "<<small_variables[k]->id()<<" size:"<<small_variables[k]->size()<<endl;
            Small_cov.block(i_index, k_index, small_variables[i]->size(), small_variables[k]->size()) =
                    state->_Cov_nuisance.block(small_variables[i]->id(), small_variables[k]->id(), small_variables[i]->size(), small_variables[k]->size());
            k_index += small_variables[k]->size();
        }
        i_index += small_variables[i]->size();
    }

    // Return the covariance
    //Small_cov = 0.5*(Small_cov+Small_cov.transpose());
    return Small_cov;
}

Eigen::MatrixXd StateHelper::get_marginal_cross_covariance(State *state, const std::vector<Type *> &x_variables,
                                                           const std::vector<Type *> &n_variables) {
    // Calculate the marginal covariance size we need to make our matrix
    int cov_size_rows = 0;
    for (size_t i = 0; i < x_variables.size(); i++) {
        cov_size_rows += x_variables[i]->size();
    }
    int cov_size_cols = 0;
    for (size_t i = 0; i < n_variables.size(); i++) {
        cov_size_cols += n_variables[i]->size();
    }

    // Construct our return covariance
    Eigen::MatrixXd Small_cov = Eigen::MatrixXd::Zero(cov_size_rows, cov_size_cols);

    // For each variable, lets copy over all other variable cross terms
    // Note: this copies over itself to when i_index=k_index
    int i_index = 0;
    for (size_t i = 0; i < x_variables.size(); i++) {
        int k_index = 0;
        for (size_t k = 0; k < n_variables.size(); k++) {
            Small_cov.block(i_index, k_index, x_variables[i]->size(), n_variables[k]->size()) =
                    state->_Cross_Cov_AN.block(x_variables[i]->id(), n_variables[k]->id(), x_variables[i]->size(), n_variables[k]->size());
            k_index += n_variables[k]->size();
        }
        i_index += x_variables[i]->size();
    }

    // Return the covariance
    //Small_cov = 0.5*(Small_cov+Small_cov.transpose());
    return Small_cov;
}

Eigen::MatrixXd StateHelper::get_full_covariance(State *state) {

    // Size of the covariance is the active
    int cov_size = (int)state->_Cov.rows();

    // Construct our return covariance
    Eigen::MatrixXd full_cov = Eigen::MatrixXd::Zero(cov_size, cov_size);

    // Copy in the active state elements
    full_cov.block(0,0,state->_Cov.rows(),state->_Cov.rows()) = state->_Cov;

    // Return the covariance
    return full_cov;

}




void StateHelper::marginalize(State *state, Type *marg) {

    // Check if the current state has the element we want to marginalize
    if (std::find(state->_variables.begin(), state->_variables.end(), marg) == state->_variables.end()) {
        printf(RED "StateHelper::marginalize() - Called on variable that is not in the state\n" RESET);
        printf(RED "StateHelper::marginalize() - Marginalization, does NOT work on sub-variables yet...\n" RESET);
        std::exit(EXIT_FAILURE);
    }
    cout<<"before assert"<<endl;
    assert(state->_Cov.rows()==state->_Cross_Cov_AN.rows());

    //Generic covariance has this form for x_1, x_m, x_2. If we want to remove x_m:
    //
    //  P_(x_1,x_1) P(x_1,x_m) P(x_1,x_2)
    //  P_(x_m,x_1) P(x_m,x_m) P(x_m,x_2)
    //  P_(x_2,x_1) P(x_2,x_m) P(x_2,x_2)
    //
    //  to
    //
    //  P_(x_1,x_1) P(x_1,x_2)
    //  P_(x_2,x_1) P(x_2,x_2)
    //
    // i.e. x_1 goes from 0 to marg_id, x_2 goes from marg_id+marg_size to Cov.rows() in the original covariance

    int marg_size = marg->size();
    int marg_id = marg->id();
    int x2_size = (int)state->_Cov.rows() - marg_id - marg_size;

    Eigen::MatrixXd Cov_new(state->_Cov.rows() - marg_size, state->_Cov.rows() - marg_size);

    //P_(x_1,x_1)
    Cov_new.block(0, 0, marg_id, marg_id) = state->_Cov.block(0, 0, marg_id, marg_id);

    //P_(x_1,x_2)
    Cov_new.block(0, marg_id, marg_id, x2_size) = state->_Cov.block(0, marg_id + marg_size, marg_id, x2_size);

    //P_(x_2,x_1)
    Cov_new.block(marg_id, 0, x2_size, marg_id) = Cov_new.block(0, marg_id, marg_id, x2_size).transpose();

    //P(x_2,x_2)
    Cov_new.block(marg_id, marg_id, x2_size, x2_size) = state->_Cov.block(marg_id + marg_size, marg_id + marg_size, x2_size, x2_size);


    // Now set new covariance
    state->_Cov = Cov_new;
    //state->Cov() = 0.5*(Cov_new+Cov_new.transpose());
    assert(state->_Cov.rows() == Cov_new.rows());

    // Margin cross_cov part
//    if(!state->_nuisance_variables.empty())
    {
        cout<<"before marginalize of Cross_Cov_AN"<<endl;
        Eigen::MatrixXd Cross_Cov_new(state->_Cross_Cov_AN.rows()-marg_size,state->_Cross_Cov_AN.cols());

        cout<<"1"<<endl;

        int n_size=state->_Cross_Cov_AN.cols();
        Cross_Cov_new.block(0,0,marg_id,n_size)=state->_Cross_Cov_AN.block(0,0,marg_id,n_size);
        cout<<"2"<<endl;
        Cross_Cov_new.block(marg_id,0,x2_size,n_size)=state->_Cross_Cov_AN.block(marg_id+marg_size,0,x2_size,n_size);
        cout<<"3"<<endl;
        state->_Cross_Cov_AN = Cross_Cov_new;
        cout<<"4"<<endl;
        assert(state->_Cross_Cov_AN.rows() == Cross_Cov_new.rows());
        cout<<"Now we finish marginalize of Cross_Cov_AN "<<endl;
    }

    // Now we keep the remaining variables and update their ordering
    // Note: DOES NOT SUPPORT MARGINALIZING SUBVARIABLES YET!!!!!!!
    std::vector<Type *> remaining_variables;
    for (size_t i = 0; i < state->_variables.size(); i++) {
        // Only keep non-marginal states
        if (state->_variables.at(i) != marg) {
            if (state->_variables.at(i)->id() > marg_id) {
                // If the variable is "beyond" the marginal one in ordering, need to "move it forward"
                state->_variables.at(i)->set_local_id(state->_variables.at(i)->id() - marg_size);
            }
            remaining_variables.push_back(state->_variables.at(i));
        }
    }
    // //we also need to reset the local id of transfor_vio_to_map, as this variable is used in "get_feature_jacobian_kf"
    // //by zzq
    // if(state->set_transform && state->transform_vio_to_map->id() > marg_id)
    // {
    //      state->transform_vio_to_map->set_local_id(state->transform_vio_to_map->id()-marg_size);
    // }

    // Delete the old state variable to free up its memory
    delete marg;

    // Now set variables as the remaining ones
    state->_variables = remaining_variables;

}


Type* StateHelper::clone(State *state, Type *variable_to_clone) {

    //Get total size of new cloned variables, and the old covariance size
    int total_size = variable_to_clone->size(); //here variable_to_clone is imu pose.
    int old_size = (int)state->_Cov.rows();
    int new_loc = (int)state->_Cov.rows();
    int nuisance_size=(int)state->_Cov_nuisance.rows();


    // Resize both our covariance to the new size
    state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size + total_size, old_size + total_size));
    state->_Cross_Cov_AN.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size+total_size,nuisance_size));

    // What is the new state, and variable we inserted
    const std::vector<Type*> new_variables = state->_variables;
    Type *new_clone = nullptr;

    // Loop through all variables, and find the variable that we are going to clone
    for (size_t k = 0; k < state->_variables.size(); k++) {

        // Skip this if it is not the same
        Type *type_check = state->_variables.at(k)->check_if_same_variable(variable_to_clone);
        if (type_check == nullptr)
            continue;

        // So we will clone this one
        int old_loc = type_check->id();

        // Copy the covariance elements
        state->_Cov.block(new_loc, new_loc, total_size, total_size) = state->_Cov.block(old_loc, old_loc, total_size, total_size);
        state->_Cov.block(0, new_loc, old_size, total_size) = state->_Cov.block(0, old_loc, old_size, total_size);
        state->_Cov.block(new_loc, 0, total_size, old_size) = state->_Cov.block(old_loc, 0, total_size, old_size);
        state->_Cross_Cov_AN.block(new_loc,0,total_size,nuisance_size)=state->_Cross_Cov_AN.block(old_loc,0,total_size,nuisance_size);

        // Create clone from the type being cloned
        new_clone = type_check->clone();
        new_clone->set_local_id(new_loc);

        // Add to variable list
        state->_variables.push_back(new_clone);
        break;

    }

    // Check if the current state has this variable
    if (new_clone == nullptr) {
        printf(RED "StateHelper::clone() - Called on variable is not in the state\n" RESET);
        printf(RED "StateHelper::clone() - Ensure that the variable specified is a variable, or sub-variable..\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    return new_clone;

}




bool StateHelper::initialize(State *state, Type *new_variable, const std::vector<Type *> &H_order, Eigen::MatrixXd &H_R,
                             Eigen::MatrixXd &H_L, Eigen::MatrixXd &R, Eigen::VectorXd &res, double chi_2_mult) {

    // Check that this new variable is not already initialized
    if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
        std::cerr << "StateHelper::initialize() - Called on variable that is already in the state" << std::endl;
        std::cerr << "StateHelper::initialize() - Found this variable at " << new_variable->id() << " in covariance" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Check that we have isotropic noise (i.e. is diagonal and all the same value)
    // TODO: can we simplify this so it doesn't take as much time?
    assert(R.rows()==R.cols());
    assert(R.rows()>0);
    for(int r=0; r<R.rows(); r++) {
        for(int c=0; c<R.cols(); c++) {
            if(r==c && R(0,0) != R(r,c)) {
                printf(RED "StateHelper::initialize() - Your noise is not isotropic!\n" RESET);
                printf(RED "StateHelper::initialize() - Found a value of %.2f verses value of %.2f\n" RESET, R(r,c), R(0,0));
                std::exit(EXIT_FAILURE);
            } else if(r!=c && R(r,c) != 0.0) {
                printf(RED "StateHelper::initialize() - Your noise is not diagonal!\n" RESET);
                printf(RED "StateHelper::initialize() - Found a value of %.2f at row %d and column %d\n" RESET, R(r,c), r, c);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    //==========================================================
    //==========================================================
    // First we perform QR givens to seperate the system
    // The top will be a system that depends on the new state, while the bottom does not
    size_t new_var_size = new_variable->size();
    assert((int)new_var_size == H_L.cols());

    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_L.cols(); ++n) {
        for (int m = (int) H_L.rows() - 1; m > n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_L(m - 1, n), H_L(m, n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_L.block(m - 1, n, 2, H_L.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (H_R.block(m - 1, 0, 2, H_R.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
        }
    }

    // Separate into initializing and updating portions
    // 1. Invertible initializing system
    //corresponding to H_x1,H_f1,r_1,n_1 in @ref pages
    Eigen::MatrixXd Hxinit = H_R.block(0, 0, new_var_size, H_R.cols());
    Eigen::MatrixXd H_finit = H_L.block(0, 0, new_var_size, new_var_size);
    Eigen::VectorXd resinit = res.block(0, 0, new_var_size, 1);
    Eigen::MatrixXd Rinit = R.block(0, 0, new_var_size, new_var_size);

    // 2. Nullspace projected updating system
    //corresponding to H_x2,r_2,n_2 in @ref pages
    Eigen::MatrixXd Hup = H_R.block(new_var_size, 0, H_R.rows() - new_var_size, H_R.cols());
    Eigen::VectorXd resup = res.block(new_var_size, 0, res.rows() - new_var_size, 1);
    Eigen::MatrixXd Rup = R.block(new_var_size, new_var_size, R.rows() - new_var_size, R.rows() - new_var_size);

    //==========================================================
    //==========================================================

    // Do mahalanobis distance testing
    Eigen::MatrixXd P_up = get_marginal_covariance(state, H_order);
    assert(Rup.rows() == Hup.rows());
    assert(Hup.cols() == P_up.cols());
    Eigen::MatrixXd S = Hup*P_up*Hup.transpose()+Rup;
    double chi2 = resup.dot(S.llt().solve(resup));

    // Get what our threshold should be
    boost::math::chi_squared chi_squared_dist(res.rows());
    double chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
    if (chi2 > chi_2_mult*chi2_check) {
        return false;
    }

    //==========================================================
    //==========================================================
    // Finally, initialize it in our state
    StateHelper::initialize_invertible(state, new_variable, H_order, Hxinit, H_finit, Rinit, resinit);

    // Update with updating portion
    if (Hup.rows() > 0) {
        StateHelper::EKFUpdate(state, H_order, Hup, resup, Rup);
    }
    return true;

}

bool StateHelper::initialize_skf(State *state, Type *new_variable, const std::vector<Type *> &Hx_order,
                                 const std::vector<Type *> &Hn_order, Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_n,
                                 Eigen::MatrixXd &H_f, Eigen::MatrixXd &R, Eigen::VectorXd &res, double chi_2_mult) {
    // Check that this new variable is not already initialized
    if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
        std::cerr << "StateHelper::initialize() - Called on variable that is already in the state" << std::endl;
        std::cerr << "StateHelper::initialize() - Found this variable at " << new_variable->id() << " in covariance" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Check that we have isotropic noise (i.e. is diagonal and all the same value)
    // TODO: can we simplify this so it doesn't take as much time?
    assert(R.rows()==R.cols());
    assert(R.rows()>0);
    for(int r=0; r<R.rows(); r++) {
        for(int c=0; c<R.cols(); c++) {
            if(r==c && R(0,0) != R(r,c)) {
                printf(RED "StateHelper::initialize() - Your noise is not isotropic!\n" RESET);
                printf(RED "StateHelper::initialize() - Found a value of %.2f verses value of %.2f\n" RESET, R(r,c), R(0,0));
                std::exit(EXIT_FAILURE);
            } else if(r!=c && R(r,c) != 0.0) {
                printf(RED "StateHelper::initialize() - Your noise is not diagonal!\n" RESET);
                printf(RED "StateHelper::initialize() - Found a value of %.2f at row %d and column %d\n" RESET, R(r,c), r, c);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    //==========================================================
    //==========================================================
    // First we perform QR givens to seperate the system
    // The top will be a system that depends on the new state, while the bottom does not
    size_t new_var_size = new_variable->size();
    cout<<"in initializa_skf, before assert"<<endl;
    assert((int)new_var_size == H_f.cols());
    assert(H_x.rows()==H_n.rows());
    cout<<"in initializa_skf, after assert"<<endl;
    //we put Hx and Hn into a big H_big
    Eigen::MatrixXd H_big=Eigen::MatrixXd::Zero(H_x.rows(),H_x.cols()+H_n.cols());
    H_big.block(0,0,H_x.rows(),H_x.cols())=H_x;
    H_big.block(0,H_x.cols(),H_n.rows(),H_n.cols())=H_n;

   cout<<"after H_big"<<endl;

    Eigen::JacobiRotation<double> tempHo_GR;
    for (int n = 0; n < H_f.cols(); ++n) {
        for (int m = (int) H_f.rows() - 1; m > n; m--) {
            // Givens matrix G
            tempHo_GR.makeGivens(H_f(m - 1, n), H_f(m, n));
            // Multiply G to the corresponding lines (m-1,m) in each matrix
            // Note: we only apply G to the nonzero cols [n:Ho.cols()-n-1], while
            //       it is equivalent to applying G to the entire cols [0:Ho.cols()-1].
            (H_f.block(m - 1, n, 2, H_f.cols() - n)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (res.block(m - 1, 0, 2, 1)).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
            (H_big.block(m - 1, 0, 2, H_big.cols())).applyOnTheLeft(0, 1, tempHo_GR.adjoint());
        }
    }
    cout<<"after nullspace projection"<<endl;

    // Separate into initializing and updating portions
    //res=H_big*x+H_f*f+n  --> [r1]=[Hx1 Hn1]*[x ]+[Hf1]*f + [n1]
    //                         [r2]=[Hx2 Hn2]*[xn] [ 0 ]     [n2]

    // 1. Invertible initializing system
    //corresponding to H_x1,H_n1, H_f1,r_1,n_1
    Eigen::MatrixXd Hxinit = H_big.block(0, 0, new_var_size, H_x.cols());
    Eigen::MatrixXd Hninit = H_big.block(0,H_x.cols(),new_var_size,H_n.cols());
    Eigen::MatrixXd H_finit = H_f.block(0, 0, new_var_size, new_var_size);
    Eigen::VectorXd resinit = res.block(0, 0, new_var_size, 1);
    Eigen::MatrixXd Rinit = R.block(0, 0, new_var_size, new_var_size);
    cout<<"after Hxinit"<<endl;

    // 2. Nullspace projected updating system
    //corresponding to H_x2,H_n2,r_2,n_2
    Eigen::MatrixXd Hxup = H_big.block(new_var_size, 0, H_big.rows() - new_var_size, H_x.cols());
    Eigen::MatrixXd Hnup = H_big.block(new_var_size, H_x.cols(), H_big.rows() - new_var_size, H_n.cols());
    Eigen::VectorXd resup = res.block(new_var_size, 0, res.rows() - new_var_size, 1);
    Eigen::MatrixXd Rup = R.block(new_var_size, new_var_size, R.rows() - new_var_size, R.rows() - new_var_size);
     cout<<"after Hxup"<<endl;
    //==========================================================
    //==========================================================

    // Do mahalanobis distance testing
    Eigen::MatrixXd Px_up = get_marginal_covariance(state, Hx_order);
    cout<<"get Pxup"<<endl;
    Eigen::MatrixXd Pn_up = get_marginal_nuisance_covariance(state,Hn_order);
    cout<<"get Pnup"<<endl;
    Eigen::MatrixXd Pxn_up = get_marginal_cross_covariance(state,Hx_order,Hn_order);
    cout<<"get Pxnup"<<endl;
    assert(Rup.rows() == Hxup.rows());
    assert(Rup.rows()==Hnup.rows());
    assert(Hnup.cols()==Pn_up.cols());
    assert(Hxup.cols() == Px_up.cols());
    assert(Pxn_up.rows()==Hxup.cols());
    assert(Pxn_up.cols()==Hnup.cols());
    Eigen::MatrixXd S(Rup.rows(),Rup.rows());
    cout<<"before S"<<endl;

    // Residual covariance S = H*Cov*H' + R
    S  = Hxup * Px_up * Hxup.transpose();
    S += Hxup * Pxn_up * Hnup.transpose();
    S += (Hxup * Pxn_up * Hnup.transpose()).transpose();
    S += Hnup * Pn_up * Hnup.transpose();
    S += Rup;
    cout<<"after S"<<endl;
    double chi2 = resup.dot(S.llt().solve(resup));



    // Get what our threshold should be
    boost::math::chi_squared chi_squared_dist(res.rows());
    double chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
    if (chi2 > chi_2_mult*chi2_check) {
        return false;
    }

    //==========================================================
    //==========================================================
    // Finally, initialize it in our state
    cout<<"before initializa_invertible_with_nuisance"<<endl;
    StateHelper::initialize_invertible_with_nuisance(state, new_variable, Hx_order, Hn_order, Hxinit,Hninit, H_finit, Rinit, resinit);
    cout<<"after initializa_invertible_with_nuisance"<<endl;
    // Update with updating portion
    if (Hxup.rows() > 0) {
        if(!state->_options.full_ekf)
            StateHelper::SKFUpdate(state,Hx_order, Hn_order, Hxup, Hnup, resup, Rup,false);
        else
            StateHelper::EKFMAPUpdate(state,Hx_order, Hn_order, Hxup, Hnup, resup, Rup,false);
    }
    return true;
}


void StateHelper::initialize_invertible(State *state, Type *new_variable, const std::vector<Type *> &H_order, const Eigen::MatrixXd &H_R,
                                        const Eigen::MatrixXd &H_L, const Eigen::MatrixXd &R, const Eigen::VectorXd &res) {

    // Check that this new variable is not already initialized
    if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
        std::cerr << "StateHelper::initialize_invertible() - Called on variable that is already in the state" << std::endl;
        std::cerr << "StateHelper::initialize_invertible() - Found this variable at " << new_variable->id() << " in covariance" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Check that we have isotropic noise (i.e. is diagonal and all the same value)
    // TODO: can we simplify this so it doesn't take as much time?
    assert(R.rows()==R.cols());
    assert(R.rows()>0);
    for(int r=0; r<R.rows(); r++) {
        for(int c=0; c<R.cols(); c++) {
            if(r==c && R(0,0) != R(r,c)) {
                printf(RED "StateHelper::initialize_invertible() - Your noise is not isotropic!\n" RESET);
                printf(RED "StateHelper::initialize_invertible() - Found a value of %.2f verses value of %.2f\n" RESET, R(r,c), R(0,0));
                std::exit(EXIT_FAILURE);
            } else if(r!=c && R(r,c) != 0.0) {
                printf(RED "StateHelper::initialize_invertible() - Your noise is not diagonal!\n" RESET);
                printf(RED "StateHelper::initialize_invertible() - Found a value of %.2f at row %d and column %d\n" RESET, R(r,c), r, c);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    //==========================================================
    //==========================================================
    // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
    assert(res.rows() == R.rows());
    assert(H_L.rows() == res.rows());
    assert(H_L.rows() == H_R.rows());
    Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());

    // Get the location in small jacobian for each measuring variable
    int current_it = 0;
    std::vector<int> H_id;
    for (Type *meas_var: H_order) {
        H_id.push_back(current_it);
        current_it += meas_var->size();
    }

    //==========================================================
    //==========================================================
    // For each active variable find its M = P*H^T
    for (Type *var: state->_variables) {
        // Sum up effect of each subjacobian= K_i= \sum_m (P_im Hm^T)
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for (size_t i = 0; i < H_order.size(); i++) {
            Type *meas_var = H_order[i];
            M_i += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                   H_R.block(0, H_id[i], H_R.rows(), meas_var->size()).transpose();
        }
        M_a.block(var->id(), 0, var->size(), res.rows()) = M_i;
    }


    //==========================================================
    //==========================================================
    // Get covariance of this small jacobian
    Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);

    // M = H_R*Cov*H_R' + R
    Eigen::MatrixXd M(H_R.rows(), H_R.rows());
    M.triangularView<Eigen::Upper>() = H_R * P_small * H_R.transpose();
    M.triangularView<Eigen::Upper>() += R;

    // Covariance of the variable/landmark that will be initialized
    assert(H_L.rows()==H_L.cols());
    assert(H_L.rows() == new_variable->size());
    Eigen::MatrixXd H_Linv = H_L.inverse();
    Eigen::MatrixXd P_LL = H_Linv * M.selfadjointView<Eigen::Upper>() * H_Linv.transpose();

    // Augment the covariance matrix
    size_t oldSize = state->_Cov.rows();
    size_t oldSize_n =state->_Cov_nuisance.rows();
    state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(oldSize + new_variable->size(), oldSize + new_variable->size()));
    state->_Cross_Cov_AN.conservativeResizeLike(Eigen::MatrixXd::Zero(oldSize+new_variable->size(),oldSize_n));

    state->_Cov.block(0, oldSize, oldSize, new_variable->size()).noalias() = -M_a * H_Linv.transpose();
    state->_Cov.block(oldSize, 0, new_variable->size(), oldSize) = state->_Cov.block(0, oldSize, oldSize, new_variable->size()).transpose();
    state->_Cov.block(oldSize, oldSize, new_variable->size(), new_variable->size()) = P_LL;


    // Update the variable that will be initialized (invertible systems can only update the new variable).
    // However this update should be almost zero if we already used a conditional Gauss-Newton to solve for the initial estimate
    //for invariant error formulate, \hat{f}-\hat{R}R^{-1}f =\tilde{f}  --> f \approx \hat{f} - \tilde{f}
    // cout<<"initialize_invertable, landmark is "<<new_variable->value().transpose()<<endl;
    // cout<<"update value: "<<(-H_Linv * res).transpose()<<endl;
    new_variable->update(H_Linv * res);
    // cout<<"after update, landmark is "<<new_variable->value().transpose()<<endl;
    // Now collect results, and add it to the state variables
    new_variable->set_local_id(oldSize);
    state->_variables.push_back(new_variable);
    //std::cout << new_variable->id() <<  " init dx = " << (H_Linv * res).transpose() << std::endl;

}

void StateHelper::initialize_invertible_with_nuisance(State *state, Type *new_variable,
                                                      const std::vector<Type *> &Hx_order,
                                                      const std::vector<Type *> &Hn_order, const Eigen::MatrixXd &H_x,
                                                      const Eigen::MatrixXd &H_n, const Eigen::MatrixXd &H_f,
                                                      const Eigen::MatrixXd &R, const Eigen::VectorXd &res) {
    // Check that this new variable is not already initialized
    if (std::find(state->_variables.begin(), state->_variables.end(), new_variable) != state->_variables.end()) {
        std::cerr << "StateHelper::initialize_invertible() - Called on variable that is already in the state" << std::endl;
        std::cerr << "StateHelper::initialize_invertible() - Found this variable at " << new_variable->id() << " in covariance" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Check that we have isotropic noise (i.e. is diagonal and all the same value)
    // TODO: can we simplify this so it doesn't take as much time?
    assert(R.rows()==R.cols());
    assert(R.rows()>0);
    for(int r=0; r<R.rows(); r++) {
        for(int c=0; c<R.cols(); c++) {
            if(r==c && R(0,0) != R(r,c)) {
                printf(RED "StateHelper::initialize_invertible() - Your noise is not isotropic!\n" RESET);
                printf(RED "StateHelper::initialize_invertible() - Found a value of %.2f verses value of %.2f\n" RESET, R(r,c), R(0,0));
                std::exit(EXIT_FAILURE);
            } else if(r!=c && R(r,c) != 0.0) {
                printf(RED "StateHelper::initialize_invertible() - Your noise is not diagonal!\n" RESET);
                printf(RED "StateHelper::initialize_invertible() - Found a value of %.2f at row %d and column %d\n" RESET, R(r,c), r, c);
                std::exit(EXIT_FAILURE);
            }
        }
    }

    //==========================================================
    //==========================================================
    // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
    assert(res.rows() == R.rows());
    assert(H_x.rows() == res.rows());
    assert(H_f.rows() == H_x.rows());
    assert(H_n.rows() == H_f.rows());
    Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());
    Eigen::MatrixXd M_b = Eigen::MatrixXd::Zero(state->_Cov.rows(),res.rows());
    Eigen::MatrixXd M_c = Eigen::MatrixXd::Zero(state->_Cov_nuisance.rows(),res.rows());
    Eigen::MatrixXd M_d = Eigen::MatrixXd::Zero(state->_Cov_nuisance.rows(),res.rows());


    // Get the location in small jacobian for each measuring variable
    int current_it = 0;
    std::vector<int> Hx_id;
    for (Type *meas_var: Hx_order) {
        Hx_id.push_back(current_it);
        current_it += meas_var->size();
    }

    current_it = 0;
    std::vector<int> Hn_id;
    for (Type *meas_var: Hn_order){
        Hn_id.push_back(current_it);
        current_it += meas_var->size();
    }

    //==========================================================
    //==========================================================
    // For each active variable find its M = P*H^T   //Pxx*Hx^T
    for (Type *var: state->_variables) {
        // Sum up effect of each subjacobian= K_i= \sum_m (P_im Hm^T)
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for (size_t i = 0; i < Hx_order.size(); i++) {
            Type *meas_var = Hx_order[i];
            M_i += state->_Cov.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                   H_x.block(0, Hx_id[i], H_x.rows(), meas_var->size()).transpose();
        }
        M_a.block(var->id(), 0, var->size(), res.rows()) = M_i;
    }
    cout<<"M_a"<<endl;
    // For each active variable find its M = P*H^T   //Pxn*Hn^T
    for (Type *var: state->_variables){
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for(size_t i=0; i < Hn_order.size();i++){
            Type *meas_var = Hn_order[i];
            M_i+= state->_Cross_Cov_AN.block(var->id(),meas_var->id(),var->size(),meas_var->size())*
                    H_n.block(0,Hn_id[i],H_n.rows(),meas_var->size()).transpose();
        }
        M_b.block(var->id(),0,var->size(),res.rows()) = M_i;
    }
    cout<<"M_b"<<endl;
    //Pnn*Hn^T
    for (Type *var: state->_nuisance_variables) {
        // Sum up effect of each subjacobian= K_i= \sum_m (P_im Hm^T)
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for (size_t i = 0; i < Hn_order.size(); i++) {
            Type *meas_var = Hn_order[i];
            M_i += state->_Cov_nuisance.block(var->id(), meas_var->id(), var->size(), meas_var->size()) *
                   H_n.block(0, Hn_id[i], H_n.rows(), meas_var->size()).transpose();
        }
        M_c.block(var->id(), 0, var->size(), res.rows()) = M_i;
    }
    cout<<"M_c"<<endl;
    //Pnx*Hx^T
    for (Type *var: state->_nuisance_variables) {
        // Sum up effect of each subjacobian= K_i= \sum_m (P_im Hm^T)
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(), res.rows());
        for (size_t i = 0; i < Hx_order.size(); i++) {
            Type *meas_var = Hx_order[i];
            M_i += state->_Cross_Cov_AN.block(meas_var->id(), var->id(), meas_var->size(),var->size()).transpose() *
                   H_x.block(0, Hx_id[i], H_n.rows(), meas_var->size()).transpose();
        }
        M_d.block(var->id(), 0, var->size(), res.rows()) = M_i;
    }
    cout<<"M_d"<<endl;
    //==========================================================
    //==========================================================
    // Get covariance of this small jacobian
    Eigen::MatrixXd Px_small = StateHelper::get_marginal_covariance(state, Hx_order);
    Eigen::MatrixXd Pn_small = StateHelper::get_marginal_nuisance_covariance(state,Hn_order);
    Eigen::MatrixXd Pxn_small= StateHelper::get_marginal_cross_covariance(state,Hx_order,Hn_order);

    // M = H_R*Cov*H_R' + R
    Eigen::MatrixXd M(H_x.rows(), H_x.rows());
    M = H_x * Px_small * H_x.transpose();
    M += R;
    M += H_n * Pn_small * H_n.transpose();
    M += H_x*Pxn_small*H_n.transpose();
    M += H_n*Pxn_small.transpose()*H_x.transpose();

    // Covariance of the variable/landmark that will be initialized
    assert(H_f.rows()==H_f.cols());
    assert(H_f.rows() == new_variable->size());
    Eigen::MatrixXd H_finv = H_f.inverse();
    Eigen::MatrixXd P_ff = H_finv * M.selfadjointView<Eigen::Upper>() * H_finv.transpose();

    // Augment the covariance matrix
    cout<<"before augment"<<endl;
    size_t oldSize = state->_Cov.rows();
    size_t oldSize_n=state->_Cross_Cov_AN.cols();
    state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(oldSize + new_variable->size(), oldSize + new_variable->size()));
    state->_Cross_Cov_AN.conservativeResizeLike(Eigen::MatrixXd::Zero(oldSize+new_variable->size(),oldSize_n));
    cout<<"1"<<endl;
    state->_Cov.block(0, oldSize, oldSize, new_variable->size()).noalias() = -(M_a + M_b )* H_finv.transpose();
    cout<<"2"<<endl;
    state->_Cov.block(oldSize, 0, new_variable->size(), oldSize) = state->_Cov.block(0, oldSize, oldSize, new_variable->size()).transpose();
    cout<<"3"<<endl;
    state->_Cov.block(oldSize, oldSize, new_variable->size(), new_variable->size()) = P_ff;
    cout<<"4"<<endl;
    state->_Cross_Cov_AN.block(oldSize,0,new_variable->size(),oldSize_n)=-H_finv*(M_c.transpose()+M_d.transpose());
    cout<<"after augment"<<endl;

    // Update the variable that will be initialized (invertible systems can only update the new variable).
    // However this update should be almost zero if we already used a conditional Gauss-Newton to solve for the initial estimate
    new_variable->update(H_finv * res);

    // Now collect results, and add it to the state variables
    new_variable->set_local_id(oldSize);
    state->_variables.push_back(new_variable);
    //std::cout << new_variable->id() <<  " init dx = " << (H_Linv * res).transpose() << std::endl;

}


void StateHelper::augment_clone(State *state, Eigen::Matrix<double, 3, 1> last_w) {

    // Call on our marginalizer to clone, it will add it to our vector of types
    // NOTE: this will clone the clone pose to the END of the covariance...
    //clone function will add the cloned pose to the state->variables, and augment the covariance. by zzq
    Type *posetemp = StateHelper::clone(state, state->_imu->pose());

    // Cast to a JPL pose type
    //将基类转换成子类，为了数据类型的安全，需要用dynamic_cast. by zzq
    PoseJPL *pose = dynamic_cast<PoseJPL*>(posetemp);

    // Check that it was a valid cast
    if (pose == nullptr) {
        printf(RED "INVALID OBJECT RETURNED FROM STATEHELPER CLONE, EXITING!#!@#!@#\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // Append the new clone to our clone vector
    state->_clones_IMU.insert({state->_timestamp, pose});

    // If we are doing time calibration, then our clones are a function of the time offset
    // Logic is based on Mingyang Li and Anastasios I. Mourikis paper:
    // http://journals.sagepub.com/doi/pdf/10.1177/0278364913515286
    // NOTE: for RIMSCKF, the defination of error of pose is not the standard error,
    // so the Jacobian would be different from MSCKF. the error of timeoffset is still a standard error.
    // TODO: the performance seems not good????
    if (state->_options.do_calib_camera_timeoffset) {
        // Jacobian to augment by
        Eigen::Matrix<double, 6, 1> dnc_dt = Eigen::MatrixXd::Zero(6, 1);
        dnc_dt.block(0, 0, 3, 1) = pose->Rot().transpose()*last_w;
        // dnc_dt.block(3, 0, 3, 1) = skew_x(pose->Rot().transpose()*last_w)*pose->pos() - state->_imu->vel();
        dnc_dt.block(3, 0, 3, 1) = - skew_x(pose->Rot().transpose()*last_w)*pose->pos() + state->_imu->vel();
        // Augment covariance with time offset Jacobian
        state->_Cov.block(0, pose->id(), state->_Cov.rows(), 6) +=
                state->_Cov.block(0, state->_calib_dt_CAMtoIMU->id(), state->_Cov.rows(), 1) * dnc_dt.transpose();
        state->_Cov.block(pose->id(), 0, 6, state->_Cov.rows()) +=
                dnc_dt * state->_Cov.block(state->_calib_dt_CAMtoIMU->id(), 0, 1, state->_Cov.rows());
    }

    //     // If we are doing time calibration, then our clones are a function of the time offset
    // // Logic is based on Mingyang Li and Anastasios I. Mourikis paper:
    // // http://journals.sagepub.com/doi/pdf/10.1177/0278364913515286
    // if (state->_options.do_calib_camera_timeoffset) {
    //     // Jacobian to augment by
    //     Eigen::Matrix<double, 6, 1> dnc_dt = Eigen::MatrixXd::Zero(6, 1);
    //     dnc_dt.block(0, 0, 3, 1) = last_w;
    //     dnc_dt.block(3, 0, 3, 1) = state->_imu->vel();
    //     // Augment covariance with time offset Jacobian
    //     state->_Cov.block(0, pose->id(), state->_Cov.rows(), 6) +=
    //             state->_Cov.block(0, state->_calib_dt_CAMtoIMU->id(), state->_Cov.rows(), 1) * dnc_dt.transpose();
    //     state->_Cov.block(pose->id(), 0, 6, state->_Cov.rows()) +=
    //             dnc_dt * state->_Cov.block(state->_calib_dt_CAMtoIMU->id(), 0, 1, state->_Cov.rows());
    // }
}

void StateHelper::add_kf_nuisance(State *state,Keyframe *kf)
{


    if(state->_clones_Keyframe.find(kf->time_stamp)!=state->_clones_Keyframe.end())
    {
        return;
    }

    Type* variable_to_clone=kf->_Pose_KFinWorld;


    //Get total size of new cloned variables, and the old covariance size
    int total_size = variable_to_clone->size(); //here variable_to_clone is keyframe pose.
    int old_size = (int)state->_Cov_nuisance.rows();
    int new_loc = (int)state->_Cov_nuisance.rows();
    int active_size=(int)state->_Cov.rows();


    // What is the new state, and variable we inserted
    const std::vector<Type*> new_variables = state->_variables;
    Type *new_clone = nullptr;

    new_clone=variable_to_clone->clone();
    new_clone->set_local_id(new_loc);

    PoseJPL *pose = dynamic_cast<PoseJPL*>(new_clone);

    state->_nuisance_variables.push_back(pose);

    state->_clones_Keyframe.insert({kf->time_stamp,pose});

    if(state->_options.use_schmidt)
    {
        //TODO: here we don't handle the cross-cov,assume it to be zeros
        //TODO: besides, kf->P_inv should be the the kfpose_error cov in world frame, 
        // Resize both our covariance to the new size
        state->_Cov_nuisance.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size + total_size, old_size + total_size));
        state->_Cross_Cov_AN.conservativeResizeLike(Eigen::MatrixXd::Zero(active_size,old_size+total_size));
        
        MatrixXd P=MatrixXd::Identity(total_size,total_size)*state->_options.keyframe_cov_pos;
        P.block(0,0,3,3)=MatrixXd::Identity(3,3)*state->_options.keyframe_cov_ori;
        Matrix<double,6,6> J = MatrixXd::Zero(6,6);
        J.block(0,0,3,3) = -Matrix3d::Identity();
        J.block(3,0,3,3) = -skew_x(pose->pos()) ;
        J.block(3,3,3,3) = -Matrix3d::Identity();
        P = J*P*J.transpose();
        kf->P_inv=P.inverse();
        kf->P=P;
        state->_Cov_nuisance.block(new_loc,new_loc,total_size,total_size)=P;
    }
   

   

}


bool StateHelper::add_map_transformation(State *state,Matrix<double,7,1> pose_map_to_vio) {


    
    PoseJPL* pose_tranform = new PoseJPL();

    //Get total size of new cloned variables, and the old covariance size
    int old_size = (int)state->_Cov.rows();
    int new_loc = (int)state->_Cov.rows();
    int nuisance_size=(int)state->_Cov_nuisance.rows();
    // state->transform_vio_to_map->set_local_id(new_loc);
    // state->transform_vio_to_map->set_value(pose_vio_to_map);
    // int variable_size=state->transform_vio_to_map->size();
    int variable_size=pose_tranform->size();
    cout<<"old_size: "<<old_size<<" variable_size:"<<variable_size<<endl;

    //augement cov and cross_cov
    state->_Cov.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size+variable_size,old_size+variable_size));
    //set the initial transformation cov a big value.


    // cout<<"pnp relative pose: "<<pose_map_to_vio.transpose()<<endl;
    // cout<<"error_p: "<<pose_map_to_vio.block(4,0,3,1).transpose()<<endl;
    // Quaterniond q_hamilton_true(0.707107,0,-0.707107,0);

    // Matrix<double,4,1> q_true = rot_2_quat(q_hamilton_true.toRotationMatrix().transpose());
    // // q_true<<0,0,0,1;
    // Vector3d error_q = compute_error(q_true,pose_map_to_vio.block(0,0,4,1));
    // Vector3d error_p = pose_map_to_vio.block(4,0,3,1);
    // cout<<"error_q: "<<error_q.transpose()<<endl;
    // double x = max(abs(pose_map_to_vio.block(4,0,3,1).maxCoeff()),abs(pose_map_to_vio.block(4,0,3,1).minCoeff()));
    // double y = max(abs(error_q.maxCoeff()),abs(error_q.minCoeff()));
    // double m = max(x,y);
    // cout<<"m: "<<m<<endl;
    // Eigen::Vector3d trans_map_in_vins = pose_map_to_vio.block(4,0,3,1);
    // Eigen::Quaterniond q(pose_map_to_vio(3,0),pose_map_to_vio(0,0),pose_map_to_vio(1,0),pose_map_to_vio(2,0));
    // // sleep(3); 
    // MatrixXd P=MatrixXd::Identity(variable_size,variable_size);
    // P(0,0) = error_q(0)*error_q(0);
    // P(1,1) = error_q(1)*error_q(1);
    // P(2,2) = error_q(2)*error_q(2);
    // P(3,3) = error_p(0)*error_p(0);
    // P(4,4) = error_p(1)*error_p(1);
    // P(5,5) = error_p(2)*error_p(2);
    // P.block(0,3,3,3) = MatrixXd::Identity(3,3) * m*m;
    //  Matrix<double,6,6> J = MatrixXd::Zero(6,6);
    // J.block(0,0,3,3) = -q.toRotationMatrix();
    // J.block(3,0,3,3) = -skew_x(trans_map_in_vins) * q.toRotationMatrix();
    // J.block(3,3,3,3) = -Matrix3d::Identity();
    // P = J*P*J.transpose();
    MatrixXd P=MatrixXd::Identity(variable_size,variable_size) * state->_options.transform_cov;
   
    state->_Cov.block(new_loc,new_loc,variable_size,variable_size)=P;

   

    //the cross cov of nuisance is zero
    state->_Cross_Cov_AN.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size+variable_size,nuisance_size));


    pose_tranform->set_local_id(new_loc);
    pose_tranform->set_value(pose_map_to_vio);
    pose_tranform->set_linp(pose_map_to_vio);
    pose_tranform->set_fej(pose_map_to_vio);
    state->_variables.push_back(pose_tranform);
    state->transform_map_to_vio=pose_tranform;
    // Type *new_clone = nullptr;

    // new_clone=state->transform_vio_to_map->clone();
    // new_clone->set_local_id(new_loc);

    cout<<"transformation id: "<<pose_tranform->id()<<" "<<state->transform_map_to_vio->id()<<new_loc<<endl;
    cout<<"Cov size:"<<state->_Cov.rows()<<endl;


    // state->_variables.push_back(new_clone);

    return true;
}


bool StateHelper::reset_map_transformation(State *state,Matrix<double,7,1> pose_vio_to_map) {


    state->transform_vio_to_map->set_value(pose_vio_to_map);
    int local_id=state->transform_vio_to_map->id();
    int variable_size=state->transform_vio_to_map->size();
    int nuisance_size=(int)state->_Cov_nuisance.rows();
    MatrixXd P=MatrixXd::Identity(variable_size,variable_size)*state->_options.transform_cov;
    MatrixXd cov_transform=state->_Cov.block(local_id,local_id,variable_size,variable_size);
    double min=cov_transform.diagonal().minCoeff();
    double scale=state->_options.transform_cov;
    scale=scale/min;
    scale=sqrt(scale);
    cout<<"min: "<<min<<" scale: "<<scale<<endl;
    state->_Cov.block(local_id,local_id,variable_size,variable_size)=P;
    // state->_Cov.block(0,local_id,local_id,variable_size)=state->_Cov.block(0,local_id,local_id,variable_size)*scale;
    // state->_Cov.block(local_id,0,variable_size,local_id)=state->_Cov.block(local_id,0,variable_size,local_id)*scale;
    // state->_Cross_Cov_AN.block(local_id,0,variable_size,nuisance_size)=state->_Cross_Cov_AN.block(local_id,0,variable_size,nuisance_size)*scale;
    // state->_Cov.block(0,local_id,local_id,variable_size)=MatrixXd::Zero(local_id,variable_size);
    // state->_Cov.block(local_id,0,variable_size,local_id)=MatrixXd::Zero(variable_size,local_id);
    // state->_Cross_Cov_AN.block(local_id,0,variable_size,nuisance_size)=MatrixXd::Zero(variable_size,nuisance_size);
    cout<<"new set transform: "<<state->transform_vio_to_map->quat()<<" "<<state->transform_vio_to_map->pos()<<endl;
    return true;
}
