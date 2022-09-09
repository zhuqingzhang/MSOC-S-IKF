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


using namespace ov_core;
using namespace ov_msckf;





void StateHelper::EKFPropagation(State *state, const std::vector<Type*> &order_NEW, const std::vector<Type*> &order_OLD,
                                 const Eigen::MatrixXd &Phi, const Eigen::MatrixXd &Q) {

    // We need at least one old and new variable
    if (order_NEW.empty() || order_OLD.empty()) {
        printf(RED "StateHelper::EKFPropagation() - Called with empty variable arrays!\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // Loop through our Phi order and ensure that they are continuous in memory
    int size_order_NEW = order_NEW.at(0)->size();
    for(size_t i=0; i<order_NEW.size()-1; i++) {
        if(order_NEW.at(i)->id()+order_NEW.at(i)->size()!=order_NEW.at(i+1)->id()) {
            printf(RED "StateHelper::EKFPropagation() - Called with non-contiguous state elements!\n" RESET);
            printf(RED "StateHelper::EKFPropagation() - This code only support a state transition which is in the same order as the state\n" RESET);
            std::exit(EXIT_FAILURE);
        }
        size_order_NEW += order_NEW.at(i+1)->size();
    }

    // Size of the old phi matrix
    int size_order_OLD = order_OLD.at(0)->size();
    for(size_t i=0; i<order_OLD.size()-1; i++) {
        size_order_OLD += order_OLD.at(i+1)->size();
    }

    // Assert that we have correct sizes
    assert(size_order_NEW==Phi.rows());
    assert(size_order_OLD==Phi.cols());
    assert(size_order_NEW==Q.cols());
    assert(size_order_NEW==Q.rows());

    // Get the location in small phi for each measuring variable
    int current_it = 0;
    std::vector<int> Phi_id;
    for (Type *var: order_OLD) {
        Phi_id.push_back(current_it);
        current_it += var->size();
    }

    // Loop through all our old states and get the state transition times it
    // Cov_PhiT = [ Pxx ] [ Phi ]'
    Eigen::MatrixXd Cov_PhiT = Eigen::MatrixXd::Zero(state->_Cov.rows(), Phi.rows());
    for (size_t i=0; i<order_OLD.size(); i++) {
        Type *var = order_OLD.at(i);
        Cov_PhiT.noalias() += state->_Cov.block(0, var->id(), state->_Cov.rows(), var->size())
                              * Phi.block(0, Phi_id[i], Phi.rows(), var->size()).transpose();
    }

    //Phi_CovAN =[Phi] [P_AN]
    Eigen::MatrixXd Phi_CovAN = Eigen::MatrixXd::Zero(Phi.rows(),state->_Cov_nuisance.cols());
    if(state->_nuisance_variables.size()>0)
    {
        cout<<"in ekf_propagation Phi_CovAN"<<endl;
        for(size_t i=0; i<order_OLD.size();i++){
            Type *var = order_OLD.at(i);
            Phi_CovAN.noalias() +=Phi.block(0,Phi_id[i],Phi.rows(),var->size())*
                                  state->_Cross_Cov_AN.block(var->id(),0,var->size(),state->_Cov_nuisance.cols());
        }
    }


    // Get Phi_NEW*Covariance*Phi_NEW^t + Q
    Eigen::MatrixXd Phi_Cov_PhiT = Q.selfadjointView<Eigen::Upper>();
    for (size_t i=0; i<order_OLD.size(); i++) {
        Type *var = order_OLD.at(i);
        Phi_Cov_PhiT.noalias() += Phi.block(0, Phi_id[i], Phi.rows(), var->size())
                                  * Cov_PhiT.block(var->id(), 0, var->size(), Phi.rows());
    }

    // We are good to go!
    int start_id = order_NEW.at(0)->id();
    int phi_size = Phi.rows();
    int total_size = state->_Cov.rows();
    state->_Cov.block(start_id,0,phi_size,total_size) = Cov_PhiT.transpose();
    state->_Cov.block(0,start_id,total_size,phi_size) = Cov_PhiT;
    state->_Cov.block(start_id,start_id,phi_size,phi_size) = Phi_Cov_PhiT;

    if(state->_nuisance_variables.size()>0)
    {
        cout<<"in ekf_propagation Cross_cov_AN"<<endl;
        state->_Cross_Cov_AN.block(start_id,0,phi_size,state->_Cov_nuisance.cols())=Phi_CovAN;
    }

    // We should check if we are not positive semi-definitate (i.e. negative diagionals is not s.p.d)
    Eigen::VectorXd diags = state->_Cov.diagonal();
    bool found_neg = false;
    for(int i=0; i<diags.rows(); i++) {
        if(diags(i) < 0.0) {
            printf(RED "StateHelper::EKFPropagation() - diagonal at %d is %.2f\n" RESET,i,diags(i));
            found_neg = true;
        }
    }
    assert(!found_neg);

}


void StateHelper::EKFUpdate(State *state, const std::vector<Type *> &H_order, const Eigen::MatrixXd &H,
                            const Eigen::VectorXd &res, const Eigen::MatrixXd &R) {

    //==========================================================
    //==========================================================
    // Part of the Kalman Gain K = (P*H^T)*S^{-1} = M*S^{-1}
    assert(res.rows() == R.rows());
    assert(H.rows() == res.rows());
    Eigen::MatrixXd M_a = Eigen::MatrixXd::Zero(state->_Cov.rows(), res.rows());
    cout<<"0"<<endl;
    // Get the location in small jacobian for each measuring variable
    int current_it = 0;
    std::vector<int> H_id;
    for (Type *meas_var: H_order) {
        H_id.push_back(current_it);
        current_it += meas_var->size();
    }
    cout<<"1"<<endl;

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
    cout<<"2"<<endl;

    //==========================================================
    //==========================================================
    // Get covariance of the involved terms
    Eigen::MatrixXd P_small = StateHelper::get_marginal_covariance(state, H_order);
    cout<<"3"<<endl;
    // Residual covariance S = H*Cov*H' + R
    Eigen::MatrixXd S(R.rows(), R.rows());
    S.triangularView<Eigen::Upper>() = H * P_small * H.transpose();
    S.triangularView<Eigen::Upper>() += R;
    //Eigen::MatrixXd S = H * P_small * H.transpose() + R;
    cout<<"4"<<endl;
    // Invert our S (should we use a more stable method here??)
    Eigen::MatrixXd Sinv = Eigen::MatrixXd::Identity(R.rows(), R.rows());
    S.selfadjointView<Eigen::Upper>().llt().solveInPlace(Sinv);
    Eigen::MatrixXd K = M_a * Sinv.selfadjointView<Eigen::Upper>();
    //Eigen::MatrixXd K = M_a * S.inverse();
    cout<<"5"<<endl;
    // Update Covariance
    state->_Cov.triangularView<Eigen::Upper>() -= K * M_a.transpose();
    state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
    //Cov -= K * M_a.transpose();
    //Cov = 0.5*(Cov+Cov.transpose());
    cout<<"6"<<endl;
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
    // for(auto feat:state->_features_SLAM)
    // {
    //     cout<<"landmark id: "<<feat.second->_featid<<" before update: "<<feat.second->get_xyz(false).transpose()<<endl;        
    // }

    Eigen::VectorXd dx = K*res;
    for (size_t i = 0; i < state->_variables.size(); i++) {
        state->_variables.at(i)->update_linp(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
        state->_variables.at(i)->update(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
    }
    cout<<"7"<<endl;
    // for(auto feat:state->_features_SLAM)
    // {
    //     cout<<"landmark id: "<<feat.second->_featid<<" after update: "<<feat.second->get_xyz(false).transpose()<<endl;        
    // }
    // cout<<"imu state position: "<<state->_imu->pos().transpose()<<" imu clone state postion: "<<state->_clones_IMU[state->_timestamp]->pos().transpose()<<endl;
    // cout<<"imu state orientation: "<<state->_imu->quat().transpose()<<" imu clone state orientation: "<<state->_clones_IMU[state->_timestamp]->quat().transpose()<<endl;
    // cout<<"state timestamp: "<<to_string(state->_timestamp)<<" imu orientation: "<<state->_imu->quat().transpose()<<" imu position: "<<state->_imu->pos().transpose()<<endl;

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

            
        }
        M_a.block(var->id(), 0, var->size(), res.rows()).noalias() += M_i;

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

        }
        M_a.block(var->id(),0,var->size(),res.rows()).noalias() += M_i;

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

        }
        M_a.block(active_size+var->id(),0,var->size(),res.rows()).noalias() += M_i;

    }

    //4: Pnn*Hn^T
    for(Type *var: state->_nuisance_variables){
        Eigen::MatrixXd M_i = Eigen::MatrixXd::Zero(var->size(),res.rows());
        Eigen::MatrixXd M_i_last = Eigen::MatrixXd::Zero(var->size(), res.rows());
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

    Eigen::MatrixXd K_active=L_active * Sinv.selfadjointView<Eigen::Upper>();
    cout<<"K_active"<<endl;
   

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
        
        Eigen::MatrixXd delta_res = Eigen::MatrixXd::Zero(res.rows(),1);
        if(state->iter!=false)// we have relinearized value;
        {
            //x_est-x_lin
            Eigen::VectorXd x_error=Eigen::VectorXd::Zero(state_size);
            int id=0;
            for (Type *meas_var: Hx_order) {
                if(meas_var->size()==6)
                {
                    PoseJPL* var=dynamic_cast<PoseJPL*>(meas_var);
                    Eigen::Vector4d q_est=var->q()->value();
                    Eigen::Vector4d q_lin=var->q()->linp();
                    Eigen::Vector3d q_error=compute_error(q_est,q_lin);
                    Eigen::Vector3d p_error=var->p()->value()-var->p()->linp();
                    Eigen::VectorXd pos_error=Eigen::VectorXd::Zero(6);
                    // if(meas_var->id()==state->transform_vio_to_map->id()&& state->_options.trans_fej)
                    // {
                    //     q_lin=var->q()->fej();
                    //     q_error=compute_error(q_est,q_lin);
                    //     pos_error=var->p()->value()-var->p()->fej();
                    // }
                    pos_error<<q_error,p_error;
                    x_error.block(id,0,meas_var->size(),1)=pos_error;
                }
                else
                {
                    x_error.block(id,0,meas_var->size(),1)=meas_var->value()-meas_var->linp();
                }
                id+=meas_var->size();
            }
            assert(id==state_size);
            //nuisnace part value are not update, so its value() and linp() are always the same, so x_error is zero
            //so here we dont need to consider the nuisance part.
            delta_res=Hx*x_error;
        }

        Eigen::VectorXd dx = K_active * (res-delta_res);
        double error=(res-delta_res).norm();
        assert(state->of_points.is_open());
        
        if(state->have_match)
        {   
            state->of_points<<to_string(state->_timestamp)<<" "<<to_string(error)<<" ";
            state->of_points<<to_string(1)<<endl;
        }
        
        for (size_t i = 0; i < state->_variables.size(); i++) {
            //if not literative, the linearized point should be equal to the _value
            state->_variables.at(i)->update_linp(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
            state->_variables.at(i)->update(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
        }
        cout<<"Now we finish the state->variables update"<<endl;
        // cout<<"imu state: "<<state->_imu->pos().transpose()<<"imu clone state: "<<state->_clones_IMU[state->_timestamp]->pos().transpose()<<endl;
    }
    else //we should update linear point of state_variable instead of the _value. this part is for IEKF
    {
        cout<<"in iterative"<<endl;
        VectorXd error=MatrixXd::Zero(Hx.cols(),1);
        int id=0;
         for (Type *meas_var: Hx_order) {
                if(meas_var->size()==6)
                {
                    PoseJPL* var=dynamic_cast<PoseJPL*>(meas_var);
                    Eigen::Vector4d q_est=var->q()->value();
                    Eigen::Vector4d q_lin=var->q()->linp();
                    Eigen::Vector3d q_error=compute_error(q_est,q_lin);
                    Eigen::Vector3d p_error=var->p()->value()-var->p()->linp();
                    Eigen::VectorXd pos_error=Eigen::VectorXd::Zero(6);
                    pos_error<<q_error,p_error;
                    error.block(id,0,meas_var->size(),1)=pos_error;
                }
                else
                {
                    error.block(id,0,meas_var->size(),1)=meas_var->value()-meas_var->linp();
                }
                id+=meas_var->size();
            }
        
        assert(id==Hx.cols()); 
        cout<<"error norm: "<<error.norm()<<endl;
        // sleep(1);  
        double a=1;
        
        Eigen::VectorXd dx = a*K_active * (res-Hx*error);
        int state_id=state->_clones_IMU.at(state->_timestamp)->id();
        double norm=dx.block(state_id,0,6,1).norm();

        if(state->iter_count<state->_options.iter_num-1)
        {
            MatrixXd e= MatrixXd::Zero(active_size,1);
            // cout<<"1"<<endl;
            state->_variables_last.clear();
            for (size_t i = 0; i < state->_variables.size(); i++) {
            MatrixXd last_linp=state->_variables.at(i)->linp();
            // cout<<"last_linp: "<<last_linp.transpose()<<endl;
            state->_variables_last.push_back(last_linp);
            state->_variables.at(i)->update_linp(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
           
            }
 
            state->iter_count++;
           
        }
        else if(state->iter_count==state->_options.iter_num-1)
        {
            
            state->_Cov.triangularView<Eigen::Upper>() -=K_active*L_active.transpose();
           
            state->_Cov = state->_Cov.selfadjointView<Eigen::Upper>();
           
            state->_Cross_Cov_AN -= K_active*L_nuisance.transpose();


            Eigen::VectorXd diags = state->_Cov.diagonal();
            bool found_neg = false;
            for(int i=0; i<diags.rows(); i++) {
                if(diags(i) < 0.0) {
                    printf(RED "StateHelper::EKFUpdate() - diagonal at %d is %.2f\n" RESET,i,diags(i));
                    found_neg = true;
                }
            }
            assert(!found_neg);

            for (size_t i = 0; i < state->_variables.size(); i++) {
            //if not literative, the linearized point should be equal to the _value
            state->_variables.at(i)->update_linp(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
            state->_variables.at(i)->update(dx.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
            }
            state->iter_count++;
        }

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
    int trans_size = state->transform_vio_to_map->size();
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
      if(var == state->transform_vio_to_map)
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
        M_a.block(var->id()-state->transform_vio_to_map->size(),0,var->size(),res.rows()).noalias() += M_i;
      }
      else
      {
        M_a.block(var->id(),0,var->size(),res.rows()).noalias() += M_i;
      }
      

    }
    assert(skip_trans);
    cout<<"2"<<endl;

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
        M_n.block(var->id(),0,var->size(),res.rows()).noalias() += M_i;
        // if(state->iter)
        // {
        //      M_a_last.block(active_size+var->id(), 0, var->size(), res.rows()).noalias() += M_i_last;
        // }
    }
    cout<<"2.5"<<endl;
    Eigen::VectorXd error_a = M_a * S_inv * res;
    cout<<"error_a: "<<error_a.transpose()<<endl;
    Eigen::VectorXd error_t = tmp * Ht.transpose()*Ainv *res;
    // error_t = Eigen::VectorXd::Zero(6);
    cout<<"error_t: "<<error_t.transpose()<<endl;
    // cout<<"tmp * Ht.transpoe(): "<<endl<<tmp * Ht.transpose()<<endl;
    // cout<<"tmp * Ht.transpoe()*Ainv: "<<endl<<tmp * Ht.transpose()*Ainv<<endl;
    cout<<"tmp: "<<tmp<<endl;
    // cout<<"Ht: "<<Ht<<endl;
    cout<<"Ainv: "<<Ainv<<endl;
    cout<<"res: "<<res<<endl;
    cout<<"3"<<endl;
    Eigen::VectorXd error = Eigen::VectorXd::Zero(state->_Cov.rows());
    cout<<"4"<<endl;
    assert(error.rows() == error_a.rows()+error_t.rows());
    int trans_id = state->transform_vio_to_map->id();
    error.block(0,0,trans_id,1) = error_a.block(0,0,trans_id,1);
    error.block(trans_id,0,trans_size,1) = error_t;
    error.block(trans_id+trans_size,0,error_a.rows()-trans_id,1) = error_a.block(trans_id,0,error_a.rows()-trans_id,1);
    cout<<"error: "<<error.transpose()<<endl;
    cout<<"5"<<endl;
    vector<Type*> Ha_order_big;
    for(Type *var: state->_variables)
    {
      if(var == state->transform_vio_to_map)
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
    assert(state->_Cov.rows()-state->transform_vio_to_map->size() == state->transform_vio_to_map->id());
    assert(active_size == state->transform_vio_to_map->id());
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

    
   
    for (size_t i = 0; i < state->_variables.size(); i++) {
            //if not literative, the linearized point should be equal to the _value
      state->_variables.at(i)->update_linp(error.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
      state->_variables.at(i)->update(error.block(state->_variables.at(i)->id(), 0, state->_variables.at(i)->size(), 1));
    }
    cout<<"9"<<endl;
  
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

    S = Hx * Pxx_small * Hx.transpose();

    S += Hx * Pxn_small * H_nuisance.transpose();

    S += (Hx * Pxn_small * H_nuisance.transpose()).transpose();

    S += H_nuisance * Pnn_small * H_nuisance.transpose();

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

    // Eigen::MatrixXd K_active=L_active * Sinv.selfadjointView<Eigen::Upper>();
     Eigen::MatrixXd K_active=L_active * Sinv;

    Eigen::MatrixXd K_nuisance=L_nuisance * Sinv;
  



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
        
        state->_Cov_nuisance.triangularView<Eigen::Upper>() -= K_nuisance*L_nuisance.transpose();
        state->_Cov_nuisance = state->_Cov_nuisance.selfadjointView<Eigen::Upper>();

       

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
   
    int i_index = 0;
    for (size_t i = 0; i < small_variables.size(); i++) {
        int k_index = 0;
        for (size_t k = 0; k < small_variables.size(); k++) {
            Small_cov.block(i_index, k_index, small_variables[i]->size(), small_variables[k]->size()) =
            state->_Cov.block(small_variables[i]->id(), small_variables[k]->id(), small_variables[i]->size(), small_variables[k]->size());
            k_index += small_variables[k]->size();
        }
        i_index += small_variables[i]->size();
    }

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
    cout<<"initialize_invertable, landmark is "<<new_variable->value().transpose()<<endl;
    cout<<"update value: "<<(H_Linv * res).transpose()<<endl;
    new_variable->update(H_Linv * res);
    cout<<"after update, landmark is "<<new_variable->value().transpose()<<endl;

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
    if (state->_options.do_calib_camera_timeoffset) {
        // Jacobian to augment by
        Eigen::Matrix<double, 6, 1> dnc_dt = Eigen::MatrixXd::Zero(6, 1);
        dnc_dt.block(0, 0, 3, 1) = last_w;
        dnc_dt.block(3, 0, 3, 1) = state->_imu->vel();
        // Augment covariance with time offset Jacobian
        state->_Cov.block(0, pose->id(), state->_Cov.rows(), 6) +=
                state->_Cov.block(0, state->_calib_dt_CAMtoIMU->id(), state->_Cov.rows(), 1) * dnc_dt.transpose();
        state->_Cov.block(pose->id(), 0, 6, state->_Cov.rows()) +=
                dnc_dt * state->_Cov.block(state->_calib_dt_CAMtoIMU->id(), 0, 1, state->_Cov.rows());
    }


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


    // Resize both our covariance to the new size
    state->_Cov_nuisance.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size + total_size, old_size + total_size));
    state->_Cross_Cov_AN.conservativeResizeLike(Eigen::MatrixXd::Zero(active_size,old_size+total_size));


    // What is the new state, and variable we inserted
    const std::vector<Type*> new_variables = state->_variables;
    Type *new_clone = nullptr;

    new_clone=variable_to_clone->clone();
    new_clone->set_local_id(new_loc);


    //TODO: here we don't handle the cross-cov,assume it to be zeros
    //TODO: besides, kf->P_inv should be the the kfpose_error cov in world frame,
    
    MatrixXd P=MatrixXd::Identity(total_size,total_size)*state->_options.keyframe_cov_pos;
    P.block(0,0,3,3)=MatrixXd::Identity(3,3)*state->_options.keyframe_cov_ori;
    kf->P_inv=P.inverse();
    kf->P=P;
    state->_Cov_nuisance.block(new_loc,new_loc,total_size,total_size)=P;

    PoseJPL *pose = dynamic_cast<PoseJPL*>(new_clone);

    state->_nuisance_variables.push_back(pose);

    

    state->_clones_Keyframe.insert({kf->time_stamp,pose});

}


bool StateHelper::add_map_transformation(State *state,Matrix<double,7,1> pose_vio_to_map) {


    
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
    cout<<"pnp relative pose: "<<pose_vio_to_map.transpose()<<endl;
    cout<<"error_p: "<<pose_vio_to_map.block(4,0,3,1).transpose()<<endl;
    Matrix<double,4,1> q_true = Matrix<double,4,1>::Zero();
    q_true<<0,0,0,1;
    Vector3d error_q = compute_error(q_true,pose_vio_to_map.block(0,0,4,1));
    cout<<"error_q: "<<error_q.transpose()<<endl;
    double x = max(abs(pose_vio_to_map.block(4,0,3,1).maxCoeff()),abs(pose_vio_to_map.block(4,0,3,1).minCoeff()));
    double y = max(abs(error_q.maxCoeff()),abs(error_q.minCoeff()));
    double m =max(x,y);
    cout<<"m: "<<m<<endl;
    // sleep(3); 
    MatrixXd P=MatrixXd::Identity(variable_size,variable_size);
    P.block(0,0,3,3) = MatrixXd::Identity(3,3) * m*m;
    P.block(0,3,3,3) = MatrixXd::Identity(3,3) * m*m;

    P=MatrixXd::Identity(variable_size,variable_size)*state->_options.transform_cov;

    state->_Cov.block(new_loc,new_loc,variable_size,variable_size)=P;

    
    //the cross cov of nuisance is zero
    state->_Cross_Cov_AN.conservativeResizeLike(Eigen::MatrixXd::Zero(old_size+variable_size,nuisance_size));


    pose_tranform->set_local_id(new_loc);
    pose_tranform->set_value(pose_vio_to_map);
    pose_tranform->set_linp(pose_vio_to_map);
    pose_tranform->set_fej(pose_vio_to_map);
    state->_variables.push_back(pose_tranform);
    state->transform_vio_to_map=pose_tranform;
    // Type *new_clone = nullptr;

    // new_clone=state->transform_vio_to_map->clone();
    // new_clone->set_local_id(new_loc);

    cout<<"transformation id: "<<pose_tranform->id()<<" "<<state->transform_vio_to_map->id()<<new_loc<<endl;
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
    cout<<"new set transform: "<<state->transform_vio_to_map->quat()<<" "<<state->transform_vio_to_map->pos()<<endl;
    return true;
}
