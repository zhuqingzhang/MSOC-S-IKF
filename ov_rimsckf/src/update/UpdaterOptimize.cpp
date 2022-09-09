
#include "UpdaterOptimize.h"

using namespace ov_rimsckf;


void UpdaterOptimize::Optimize_initial_transform_with_cov(State *state, Keyframe* loop_kf, Eigen::Vector3d& p_loopInCur, Eigen::Matrix3d& R_loopToCur, Eigen::MatrixXd& Cov_loopToCur)
{
      
      double rotate_kftoC[3];
      ceres::RotationMatrixToAngleAxis(R_loopToCur.data(),rotate_kftoC);
      Eigen::Quaterniond q_kftoC(R_loopToCur); 


      double transform[6];  
      transform[0]=rotate_kftoC[0];
      transform[1]=rotate_kftoC[1];
      transform[2]=rotate_kftoC[2];
      transform[3]=p_loopInCur(0);
      transform[4]=p_loopInCur(1);
      transform[5]=p_loopInCur(2);

      PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
      Matrix3d R_ItoC=extrinsics->Rot();
      Vector3d p_IinC=extrinsics->pos(); 

      PoseJPL *clone_Cur = state->_clones_IMU.at(state->_timestamp);
      Matrix3d R_LtoI = clone_Cur->Rot();
      Vector3d p_IinL = clone_Cur->pos();

      Matrix3d R_LtoC = R_ItoC*R_LtoI;
      Vector3d p_LinC = p_IinC-R_LtoC*p_IinL;

      double rotate_ltoc[3];
      ceres::RotationMatrixToAngleAxis(R_LtoC.data(),rotate_ltoc);
      Matrix<double,6,1> current_state=Matrix<double,6,1>::Zero();
      current_state<<rotate_ltoc[0],rotate_ltoc[1],rotate_ltoc[2],p_LinC(0),p_LinC(1),p_LinC(2);
      // current_state[1]=rotate_gtoc[1];
      // current_state[2]=rotate_gtoc[2];
      // current_state[3]=p_GinC(0);
      // current_state[4]=p_GinC(1);
      // current_state[5]=p_GinC(2);

      Vec* distortion = state->_cam_intrinsics.at(0);
      Matrix<double,8,1> intrinsics=distortion->value();
      bool fisheye=state->_cam_intrinsics_model.at(0);
      cout<<"tranform before optimize: ";
      for(auto t:transform)
      {
           cout<<t<<" ";
      }
      cout<<endl;
      cout<<"transform before optimize: "<<q_kftoC.coeffs().transpose()<<"|"<<p_loopInCur.transpose()<<endl;
      double transform_rot[3];
      double transform_pretrans[3];
      for(int i=0;i<3;i++)
      {
            transform_rot[i]=transform[i];
            transform_pretrans[i]=transform[i+3];
      }

      // cout<<"tranform before optimize: "<<endl<<state->transform_vio_to_map->quat()<<" | "<<state->transform_vio_to_map->pos()<<endl;
      ceres::Problem _Problem;
      double ts=state->_timestamp_approx;
      Eigen::Matrix3d R_KFtoW=loop_kf->_Pose_KFinWorld->Rot();
      Eigen::Vector3d p_KFinW=loop_kf->_Pose_KFinWorld->pos();
     
      for(int j=0;j<loop_kf->matched_point_2d_uv_map.at(ts).size();j++)
      {

            cv::Point2f uv=loop_kf->matched_point_2d_uv_map.at(ts).at(j);
            cv::Point3f uv_3d_kf=loop_kf->point_3d_map.at(ts).at(j);
            
            Eigen::Vector3d p_FinKF(double(uv_3d_kf.x),double(uv_3d_kf.y),double(uv_3d_kf.z));
            double x=uv.x;
            double y=uv.y;
            double point[3];
            point[0] = p_FinKF[0];
            point[1] = p_FinKF[1];
            point[2] = p_FinKF[2];


            Eigen::Vector3d p_FinW= R_KFtoW * p_FinKF + p_KFinW;
            ceres::CostFunction *cost_function;
            cost_function=new ceres::AutoDiffCostFunction<ObservationErrorInitialTransform,2,6>(new ObservationErrorInitialTransform(x,y,p_FinKF,intrinsics,fisheye));
                  ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
            _Problem.AddResidualBlock(cost_function, loss_function, transform);
            // cost_function=new ceres::AutoDiffCostFunction<ObservationErrorInitialTransform2,2,6,3>(new ObservationErrorInitialTransform2(x,y,intrinsics,fisheye));
            //       ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
            // _Problem.AddResidualBlock(cost_function, loss_function, transform,point);
      }
      

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
      options.minimizer_progress_to_stdout = true;
      options.max_num_iterations=30;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &_Problem, &summary); 

      ceres::Covariance::Options options_cov;
      ceres::Covariance covariance(options_cov);

      vector<pair<const double*, const double*>> covariance_blocks;
      covariance_blocks.push_back(make_pair(transform,transform));

      CHECK(covariance.Compute(covariance_blocks, &_Problem));

      double covariance_transform[6*6];
      covariance.GetCovarianceBlock(transform,transform,covariance_transform);
      Eigen::Matrix<double,6,6> Covariance_Transform = Eigen::Matrix<double,6,6>::Identity();
      for(int i=0;i<6;i++)
      {
        for(int j=0;j<6;j++)
        {
          Covariance_Transform(i,j) = covariance_transform[i*6+j];
        }
        
      }

      Matrix<double,3,3> R_KFtoCur_optimized;
      rotate_kftoC[0]=transform[0];
      rotate_kftoC[1]=transform[1];
      rotate_kftoC[2]=transform[2];
      ceres::AngleAxisToRotationMatrix(rotate_kftoC,R_KFtoCur_optimized.data());
      Vector3d p_KFinCur_optimized(transform[3],transform[4],transform[5]);
      Quaterniond q_KFtoCur_optimized(R_KFtoCur_optimized);
      cout<<"after optimiation: "<<q_KFtoCur_optimized.coeffs().transpose()<<" | "<<p_KFinCur_optimized.transpose()<<endl;
      cout<<"covariance: "<<endl<<Covariance_Transform<<std::endl;

      R_loopToCur = R_KFtoCur_optimized;
      p_loopInCur = p_KFinCur_optimized;
      Cov_loopToCur = 1 * Covariance_Transform;

      
      // sleep(10);
}


