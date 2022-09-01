
#include "UpdaterOptimize.h"

using namespace ov_rimsckf;

void UpdaterOptimize::Optimize_with_ceres(State *state, std::vector<Feature*>& feature_vec)
{

      
      //get the transform between the VIO(G) to World(W)
      Eigen::Matrix<double,3,3> R_GtoW = state->transform_vio_to_map->Rot();
      Eigen::Matrix<double,3,1> p_GinW = state->transform_vio_to_map->pos();
      // Eigen::Matrix<double,3,3> R_GtoW = Matrix3d::Identity();
      // Eigen::Matrix<double,3,1> p_GinW = Vector3d::Zero();
      Eigen::Matrix<double,3,3> R_WtoG = R_GtoW.transpose();
      Eigen::Matrix<double,3,1> p_WinG = -R_WtoG*p_GinW;
      double rotate_wtog[3];
      ceres::RotationMatrixToAngleAxis(R_WtoG.data(),rotate_wtog);
      double transform[6];  
      transform[0]=rotate_wtog[0];
      transform[1]=rotate_wtog[1];
      transform[2]=rotate_wtog[2];
      transform[3]=p_WinG(0);
      transform[4]=p_WinG(1);
      transform[5]=p_WinG(2);

      PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
      Matrix3d R_ItoC=extrinsics->Rot();
      Vector3d p_IinC=extrinsics->pos(); 

      PoseJPL *clone_Cur = state->_clones_IMU.at(state->_timestamp);
      Matrix3d R_GtoI = clone_Cur->Rot();
      Vector3d p_IinG = clone_Cur->pos();

      Matrix3d R_GtoC = R_ItoC*R_GtoI;
      Vector3d p_GinC = p_IinC-R_GtoC*p_IinG;

      double rotate_gtoc[3];
      ceres::RotationMatrixToAngleAxis(R_GtoC.data(),rotate_gtoc);
      Matrix<double,6,1> current_state=Matrix<double,6,1>::Zero();
      current_state<<rotate_gtoc[0],rotate_gtoc[1],rotate_gtoc[2],p_GinC(0),p_GinC(1),p_GinC(2);
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
      cout<<"state transform before optimize: "<<state->transform_vio_to_map->quat().transpose()<<"|"<<state->transform_vio_to_map->pos().transpose()<<endl;
      double transform_rot[3];
      double transform_pretrans[3];
      for(int i=0;i<3;i++)
      {
            transform_rot[i]=transform[i];
            transform_pretrans[i]=transform[i+3];
      }

      // cout<<"tranform before optimize: "<<endl<<state->transform_vio_to_map->quat()<<" | "<<state->transform_vio_to_map->pos()<<endl;
      ceres::Problem _Problem;
      for(int i=0;i<feature_vec.size();i++)
      {
            Feature* feat=feature_vec[i];
            for (auto const& pair : feat->keyframe_matched_obs)
            {
                  for(auto const& match : pair.second )
                  {
                        double kf_id=match.first;
                        size_t kf_feature_id=match.second;
                        Keyframe* keyframe=state->get_kfdataBase()->get_keyframe(kf_id);
                        Eigen::Matrix<double,9,1> kf_db = keyframe->_intrinsics;
                        Eigen::Matrix<double,8,1> kf_d = kf_db.block(0,0,8,1);
                        bool is_fisheye= int(kf_db(8,0));
                        //get the kf position in world frame.
                        PoseJPL* kf_w=state->_clones_Keyframe[kf_id];
                        Eigen::Matrix<double,3,3> R_KFtoW = kf_w->Rot();
                        Eigen::Matrix<double,3,1> p_KFinW = kf_w->pos();

                        double ts=state->_timestamp_approx; 
                        cv::Point3f p_fink=keyframe->point_3d_map.at(ts)[kf_feature_id];
                        Vector3d p_FinKF(double(p_fink.x),double(p_fink.y),double(p_fink.z));

                         cv::Point2f uv_ob=keyframe->matched_point_2d_uv_map.at(ts)[kf_feature_id];
                        double x=uv_ob.x;
                        double y=uv_ob.y;
                        Vector2d uv_OB(x,y);
                        double uv_x=feat->uvs[0][0](0);
                        double uv_y=feat->uvs[0][0](1);
                        Vector2d uv_feat(uv_x,uv_y);
                        assert(uv_OB==uv_feat); //check if feature link is correct

                        //transform the feature to map reference;
                        //hk
                        Eigen::Vector3d p_FinW =  R_KFtoW * p_FinKF + p_KFinW;

                        
                        ceres::CostFunction *cost_function;
                        // cost_function=ObservationError::Create(uv_x,uv_y,p_FinW.data(),intrinsics.data(),fisheye,current_state);
                        // cout<<"p_FinW: "<<p_FinW.transpose()<<endl;
                        cost_function=new ceres::AutoDiffCostFunction<ObservationError,2,3,3>(new ObservationError(uv_x,uv_y,p_FinW,intrinsics,fisheye,current_state));
                        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

                        _Problem.AddResidualBlock(cost_function, loss_function, transform_rot,transform_pretrans);
                        // _Problem.SetParameterBlockConstant(transform_pretrans);



      
                  }  
            }
      } 

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
      options.minimizer_progress_to_stdout = true;
      options.max_num_iterations=30;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &_Problem, &summary); 

      Matrix<double,3,3> R_WtoG_optimized;
      rotate_wtog[0]=transform_rot[0];
      rotate_wtog[1]=transform_rot[1];
      rotate_wtog[2]=transform_rot[2];
      ceres::AngleAxisToRotationMatrix(rotate_wtog,R_WtoG_optimized.data());
      Vector3d p_WinG_optimized(transform_pretrans[0],transform_pretrans[1],transform_pretrans[2]);
      Matrix3d R_GtoW_optimized=R_WtoG_optimized.transpose();
      Vector3d p_GinW_optimized=-R_GtoW_optimized*p_WinG_optimized;
      Vector4d q=rot_2_quat(R_GtoW_optimized);
      Matrix<double,7,1> transform_new=Matrix<double,7,1>::Zero();
      transform_new<<q(0),q(1),q(2),q(3),p_GinW_optimized(0),p_GinW_optimized(1),p_GinW_optimized(2);
      state->transform_vio_to_map->set_linp(transform_new);
      std::cout << summary.FullReport() << "\n";

      cout<<"tranform after optimize: ";
      for(auto t:transform_rot)
      {
            cout<<t<<" ";
      }
      for(auto t:transform_pretrans)
      {
            cout<<t<<" ";
      }

      cout<<endl;
      cout<<"state transform after optimize: "<<state->transform_vio_to_map->quat().transpose()<<"|"<<state->transform_vio_to_map->pos().transpose()<<endl;
      
      


}

void UpdaterOptimize::Optimize_initial_transform(State *state, std::vector<Keyframe*>& loop_kfs)
{
      
      
      Eigen::Matrix<double,3,3> R_GtoL = state->transform_map_to_vio->Rot();
      Eigen::Matrix<double,3,1> p_GinL = state->transform_map_to_vio->pos();
      // Eigen::Matrix<double,3,3> R_GtoW = Matrix3d::Identity();
      // Eigen::Matrix<double,3,1> p_GinW = Vector3d::Zero();
      // Eigen::Matrix<double,3,3> R_GtoL = R_LtoG.transpose();
      // Eigen::Matrix<double,3,1> p_GinL =-R_GtoL*p_LinG;
      double rotate_gtol[3];
      ceres::RotationMatrixToAngleAxis(R_GtoL.data(),rotate_gtol);
      double transform[6];  
      transform[0]=rotate_gtol[0];
      transform[1]=rotate_gtol[1];
      transform[2]=rotate_gtol[2];
      transform[3]=p_GinL(0);
      transform[4]=p_GinL(1);
      transform[5]=p_GinL(2);

      PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
      Matrix3d R_ItoC=extrinsics->Rot();
      Vector3d p_IinC=extrinsics->pos(); 

      PoseJPL *clone_Cur = state->_clones_IMU.at(state->_timestamp);
      Matrix3d R_GtoI = clone_Cur->Rot();
      Vector3d p_IinG = clone_Cur->pos();

      Matrix3d R_GtoC = R_ItoC*R_GtoI;
      Vector3d p_GinC = p_IinC-R_GtoC*p_IinG;

      double rotate_gtoc[3];
      ceres::RotationMatrixToAngleAxis(R_GtoC.data(),rotate_gtoc);
      Matrix<double,6,1> current_state=Matrix<double,6,1>::Zero();
      current_state<<rotate_gtoc[0],rotate_gtoc[1],rotate_gtoc[2],p_GinC(0),p_GinC(1),p_GinC(2);
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
      cout<<"state transform before optimize: "<<state->transform_vio_to_map->quat().transpose()<<"|"<<state->transform_vio_to_map->pos().transpose()<<endl;
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
      for(int i=0;i<loop_kfs.size();i++)
      {
            Keyframe* keyframe = loop_kfs[i];
            for(int j=0;j<keyframe->matched_point_2d_uv_map.at(ts).size();j++)
            {

                  cv::Point2f uv=keyframe->matched_point_2d_uv_map.at(ts).at(j);
                  cv::Point3f uv_3d_kf=keyframe->point_3d_map.at(ts).at(j);
                  Eigen::Matrix3d R_KFtoW=keyframe->_Pose_KFinWorld->Rot();
                  Eigen::Vector3d p_KFinW=keyframe->_Pose_KFinWorld->pos();
                  Eigen::Vector3d p_FinKF(double(uv_3d_kf.x),double(uv_3d_kf.y),double(uv_3d_kf.z));
                  double x=uv.x;
                  double y=uv.y;


                  Eigen::Vector3d p_FinW= R_KFtoW * p_FinKF + p_KFinW;
                  ceres::CostFunction *cost_function;
                  cost_function=new ceres::AutoDiffCostFunction<ObservationError,2,3,3>(new ObservationError(x,y,p_FinW,intrinsics,fisheye,current_state));
                        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
                  _Problem.AddResidualBlock(cost_function, loss_function, transform_rot,transform_pretrans);
            }
      }

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
      options.minimizer_progress_to_stdout = true;
      options.max_num_iterations=30;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &_Problem, &summary); 

      Matrix<double,3,3> R_GtoL_optimized;
      rotate_gtol[0]=transform_rot[0];
      rotate_gtol[1]=transform_rot[1];
      rotate_gtol[2]=transform_rot[2];
      ceres::AngleAxisToRotationMatrix(rotate_gtol,R_GtoL_optimized.data());
      Vector3d p_GinL_optimized(transform_pretrans[0],transform_pretrans[1],transform_pretrans[2]);
      Vector4d q=rot_2_quat(R_GtoL_optimized);
      Matrix<double,7,1> transform_new=Matrix<double,7,1>::Zero();
      transform_new<<q(0),q(1),q(2),q(3),p_GinL_optimized(0),p_GinL_optimized(1),p_GinL_optimized(2);
      state->transform_map_to_vio->set_fej(transform_new);
      state->transform_map_to_vio->set_value(transform_new);
      state->transform_map_to_vio->set_linp(transform_new);
      
      std::cout << summary.FullReport() << "\n";

      cout<<"tranform after optimize: ";
      for(auto t:transform_rot)
      {
            cout<<t<<" ";
      }
      for(auto t:transform_pretrans)
      {
            cout<<t<<" ";
      }

      cout<<endl;
      cout<<"state transform after optimize: "<<state->transform_map_to_vio->quat().transpose()<<"|"<<state->transform_map_to_vio->pos().transpose()<<endl;
      // sleep(10);
      
}

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


void UpdaterOptimize::Optimize_with_ceres_2(State *state, std::vector<Feature*>& feature_vec)
{

      
      //get the transform between the VIO(G) to World(W)
      Eigen::Matrix<double,3,3> R_GtoW = state->transform_vio_to_map->Rot();
      Eigen::Matrix<double,3,1> p_GinW = state->transform_vio_to_map->pos();
      // Eigen::Matrix<double,3,3> R_GtoW = Matrix3d::Identity();
      // Eigen::Matrix<double,3,1> p_GinW = Vector3d::Zero();
      Eigen::Matrix<double,3,3> R_WtoG = R_GtoW.transpose();
      Eigen::Matrix<double,3,1> p_WinG = -R_WtoG*p_GinW;
      double rotate_wtog[3];
      ceres::RotationMatrixToAngleAxis(R_WtoG.data(),rotate_wtog);
      double transform[6];  
      transform[0]=rotate_wtog[0];
      transform[1]=rotate_wtog[1];
      transform[2]=rotate_wtog[2];
      transform[3]=p_WinG(0);
      transform[4]=p_WinG(1);
      transform[5]=p_WinG(2);

      PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
      Matrix3d R_ItoC=extrinsics->Rot();
      Vector3d p_IinC=extrinsics->pos(); 

      PoseJPL *clone_Cur = state->_clones_IMU.at(state->_timestamp);
      Matrix3d R_GtoI = clone_Cur->Rot();
      Vector3d p_IinG = clone_Cur->pos();

      Matrix3d R_GtoC = R_ItoC*R_GtoI;
      Vector3d p_GinC = p_IinC-R_GtoC*p_IinG;

      double rotate_gtoc[3];
      ceres::RotationMatrixToAngleAxis(R_GtoC.data(),rotate_gtoc);
      Matrix<double,6,1> current_state=Matrix<double,6,1>::Zero();
      current_state<<rotate_gtoc[0],rotate_gtoc[1],rotate_gtoc[2],p_GinC(0),p_GinC(1),p_GinC(2);
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
      cout<<"state transform before optimize: "<<state->transform_vio_to_map->quat().transpose()<<"|"<<state->transform_vio_to_map->pos().transpose()<<endl;
      double transform_rot[3];
      double transform_pretrans[3];
      for(int i=0;i<3;i++)
      {
            transform_rot[i]=transform[i];
            transform_pretrans[i]=transform[i+3];
      }

      // cout<<"tranform before optimize: "<<endl<<state->transform_vio_to_map->quat()<<" | "<<state->transform_vio_to_map->pos()<<endl;
      double* points;
      points=new double[feature_vec.size()*3*10];
      vector<int> index;
      int num=0;
      // cout<<"points before optimize: "<<endl;
      for(int i=0;i<feature_vec.size();i++) { 
       Feature* feat=feature_vec[i];
            for (auto const& pair : feat->keyframe_matched_obs)
            {
                  for(auto const& match : pair.second )
                  {
                        double kf_id=match.first;
                        size_t kf_feature_id=match.second;
                        Keyframe* keyframe=state->get_kfdataBase()->get_keyframe(kf_id);
                        double ts=state->_timestamp_approx; 
                        cv::Point3f p_fink=keyframe->point_3d_map.at(ts)[kf_feature_id];
                        cv::Point3f p_fink_linp=keyframe->point_3d_linp_map.at(ts)[kf_feature_id];
                        assert(p_fink==p_fink_linp);
                        Vector3d p_FinKF(double(p_fink.x),double(p_fink.y),double(p_fink.z)); 
                        
                        points[3*num+0]=p_FinKF(0);
                        points[3*num+1]=p_FinKF(1);
                        points[3*num+2]=p_FinKF(2);
                        index.push_back(num);
                        num++;
                        // cout<<"("<<p_fink.x<<" "<<p_fink.y<<" "<<p_fink.z<<") ";
                  }
            }
      }
      // cout<<endl;
      num=0;
      ceres::Problem _Problem;
      for(int i=0;i<feature_vec.size();i++)
      {
            Feature* feat=feature_vec[i];
            for (auto const& pair : feat->keyframe_matched_obs)
            {
                  for(auto const& match : pair.second )
                  {
                        double kf_id=match.first;
                        size_t kf_feature_id=match.second;
                        Keyframe* keyframe=state->get_kfdataBase()->get_keyframe(kf_id);
                        Eigen::Matrix<double,9,1> kf_db = keyframe->_intrinsics;
                        Eigen::Matrix<double,8,1> kf_d = kf_db.block(0,0,8,1);
                        bool is_fisheye= int(kf_db(8,0));
                        //get the kf position in world frame.
                        PoseJPL* kf_w=state->_clones_Keyframe[kf_id];
                        Eigen::Matrix<double,3,3> R_KFtoW = kf_w->Rot();
                        Eigen::Matrix<double,3,1> p_KFinW = kf_w->pos();
                        double rot_kftow[3];
                        ceres::RotationMatrixToAngleAxis(R_KFtoW.data(),rot_kftow);
                        Matrix<double,6,1> kf_state=Matrix<double,6,1>::Zero();
                        kf_state<<rot_kftow[0],rot_kftow[1],rot_kftow[2],p_KFinW(0,0),p_KFinW(1,0),p_KFinW(2,0);


                        double ts=state->_timestamp_approx; 

                         cv::Point2f uv_ob=keyframe->matched_point_2d_uv_map.at(ts)[kf_feature_id];
                        double x=uv_ob.x;
                        double y=uv_ob.y;
                        Vector2d uv_OB(x,y);
                        double uv_x=feat->uvs[0][0](0);
                        double uv_y=feat->uvs[0][0](1);
                        Vector2d uv_feat(uv_x,uv_y);
                        assert(uv_OB==uv_feat); //check if feature link is correct

                        double* point=points+3*num;
                        num++;
                        
                        
                        ceres::CostFunction *cost_function;
                        // cost_function=ObservationError::Create(uv_x,uv_y,p_FinW.data(),intrinsics.data(),fisheye,current_state);
                        // cout<<"p_FinW: "<<p_FinW.transpose()<<endl;
                        cost_function=new ceres::AutoDiffCostFunction<ObservationError2,2,3,3,3>(new ObservationError2(uv_x,uv_y,kf_state,intrinsics,fisheye,current_state));
                        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

                        _Problem.AddResidualBlock(cost_function, loss_function, transform_rot,transform_pretrans, point);
                        // _Problem.SetParameterBlockConstant(transform_pretrans);
                        // _Problem.SetParameterBlockConstant(point);
                        
                  


      
                  }  
            }
      } 

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
      options.minimizer_progress_to_stdout = true;
      options.max_num_iterations=30;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &_Problem, &summary); 

      Matrix<double,3,3> R_WtoG_optimized;
      rotate_wtog[0]=transform_rot[0];
      rotate_wtog[1]=transform_rot[1];
      rotate_wtog[2]=transform_rot[2];
      ceres::AngleAxisToRotationMatrix(rotate_wtog,R_WtoG_optimized.data());
      Vector3d p_WinG_optimized(transform_pretrans[0],transform_pretrans[1],transform_pretrans[2]);
      Matrix3d R_GtoW_optimized=R_WtoG_optimized.transpose();
      Vector3d p_GinW_optimized=-R_GtoW_optimized*p_WinG_optimized;
      Vector4d q=rot_2_quat(R_GtoW_optimized);
      Matrix<double,7,1> transform_new=Matrix<double,7,1>::Zero();
      transform_new<<q(0),q(1),q(2),q(3),p_GinW_optimized(0),p_GinW_optimized(1),p_GinW_optimized(2);
      state->transform_vio_to_map->set_linp(transform_new);
      
      num=0;
      // cout<<"points after optimize: "<<endl;
      for(int i=0;i<feature_vec.size();i++) { 
       Feature* feat=feature_vec[i];
            for (auto const& pair : feat->keyframe_matched_obs)
            {
                  for(auto const& match : pair.second )
                  {
                        double kf_id=match.first;
                        size_t kf_feature_id=match.second;
                        Keyframe* keyframe=state->get_kfdataBase()->get_keyframe(kf_id);
                        double ts=state->_timestamp_approx; 
                        cv::Point3f pt;
                        pt.x=points[3*num];
                        pt.y=points[3*num+1];
                        pt.z=points[3*num+2];
                        if(pt.z>0)
                           keyframe->point_3d_linp_map.at(ts)[kf_feature_id]=pt;
                        
                        num++;
                        // cout<<"("<<pt.x<<" "<<pt.y<<" "<<pt.z<<") ";
                  }
            }
      }
      cout<<endl;

      
     
      cout<<"tranform after optimize: ";
      for(auto t:transform_rot)
      {
            cout<<t<<" ";
      }
      for(auto t:transform_pretrans)
      {
            cout<<t<<" ";
      }

      std::cout << summary.BriefReport() << "\n";

      // cout<<endl;
      // cout<<"state transform after optimize: "<<state->transform_vio_to_map->quat().transpose()<<"|"<<state->transform_vio_to_map->pos().transpose()<<endl;

}

void UpdaterOptimize::Optimize_with_ceres_3(State *state, std::vector<Feature*>& feature_vec)
{

      
      //get the transform between the VIO(G) to World(W)
      Eigen::Matrix<double,3,3> R_GtoW = state->transform_vio_to_map->Rot();
      Eigen::Matrix<double,3,1> p_GinW = state->transform_vio_to_map->pos();
      // Eigen::Matrix<double,3,3> R_GtoW = Matrix3d::Identity();
      // Eigen::Matrix<double,3,1> p_GinW = Vector3d::Zero();
      Eigen::Matrix<double,3,3> R_WtoG = R_GtoW.transpose();
      Eigen::Matrix<double,3,1> p_WinG = -R_WtoG*p_GinW;
      double rotate_wtog[3];
      ceres::RotationMatrixToAngleAxis(R_WtoG.data(),rotate_wtog);
      double transform[6];  
      transform[0]=rotate_wtog[0];
      transform[1]=rotate_wtog[1];
      transform[2]=rotate_wtog[2];
      transform[3]=p_WinG(0);
      transform[4]=p_WinG(1);
      transform[5]=p_WinG(2);

      PoseJPL *extrinsics = state->_calib_IMUtoCAM.at(0);
      Matrix3d R_ItoC=extrinsics->Rot();
      Vector3d p_IinC=extrinsics->pos(); 

      PoseJPL *clone_Cur = state->_clones_IMU.at(state->_timestamp);
      Matrix3d R_GtoI = clone_Cur->Rot();
      Vector3d p_IinG = clone_Cur->pos();

      Matrix3d R_GtoC = R_ItoC*R_GtoI;
      Vector3d p_GinC = p_IinC-R_GtoC*p_IinG;

      double rotate_gtoc[3];
      ceres::RotationMatrixToAngleAxis(R_GtoC.data(),rotate_gtoc);
      Matrix<double,6,1> current_state=Matrix<double,6,1>::Zero();
      current_state<<rotate_gtoc[0],rotate_gtoc[1],rotate_gtoc[2],p_GinC(0),p_GinC(1),p_GinC(2);
      // double current_state[6];
      // current_state[0]=rotate_gtoc[0];
      // current_state[1]=rotate_gtoc[1];
      // current_state[2]=rotate_gtoc[2];
      // current_state[3]=p_GinC(0);
      // current_state[4]=p_GinC(1);
      // current_state[5]=p_GinC(2);
      // cout<<"current state before optimize: "<<endl;
      // for(auto c:current_state)
      // {
      //    cout<<c<<" ";
      // }
      // cout<<endl;

      Vec* distortion = state->_cam_intrinsics.at(0);
      Matrix<double,8,1> intrinsics=distortion->value();
      bool fisheye=state->_cam_intrinsics_model.at(0);
      cout<<"tranform before optimize: ";
      for(auto t:transform)
      {
           cout<<t<<" ";
      }
      cout<<endl;
      cout<<"state transform before optimize: "<<state->transform_vio_to_map->quat().transpose()<<"|"<<state->transform_vio_to_map->pos().transpose()<<endl;
      double transform_rot[3];
      double transform_pretrans[3];
      for(int i=0;i<3;i++)
      {
            transform_rot[i]=transform[i];
            transform_pretrans[i]=transform[i+3];
      }
      assert(state->transform_vio_to_map->quat()==state->transform_vio_to_map->quat_linp());
      assert(state->_clones_IMU.at(state->_timestamp)->quat_linp()==state->_clones_IMU.at(state->_timestamp)->quat());

      // cout<<"tranform before optimize: "<<endl<<state->transform_vio_to_map->quat()<<" | "<<state->transform_vio_to_map->pos()<<endl;
      double* points;
      points=new double[feature_vec.size()*3*10];
      vector<int> index;
      int num=0;
      // cout<<"points before optimize: "<<endl;
      for(int i=0;i<feature_vec.size();i++) { 
       Feature* feat=feature_vec[i];
            for (auto const& pair : feat->keyframe_matched_obs)
            {
                  for(auto const& match : pair.second )
                  {
                        double kf_id=match.first;
                        size_t kf_feature_id=match.second;
                        Keyframe* keyframe=state->get_kfdataBase()->get_keyframe(kf_id);
                        double ts=state->_timestamp_approx; 
                        cv::Point3f p_fink=keyframe->point_3d_map.at(ts)[kf_feature_id];
                        Vector3d p_FinKF(double(p_fink.x),double(p_fink.y),double(p_fink.z)); 
                        
                        points[3*num+0]=p_FinKF(0);
                        points[3*num+1]=p_FinKF(1);
                        points[3*num+2]=p_FinKF(2);
                        index.push_back(num);
                        num++;
                        // cout<<"("<<p_fink.x<<" "<<p_fink.y<<" "<<p_fink.z<<") ";
                  }
            }
      }
      // cout<<endl;
      num=0;
      ceres::Problem _Problem;
      for(int i=0;i<feature_vec.size();i++)
      {
            Feature* feat=feature_vec[i];
            for (auto const& pair : feat->keyframe_matched_obs)
            {
                  for(auto const& match : pair.second )
                  {
                        double kf_id=match.first;
                        size_t kf_feature_id=match.second;
                        Keyframe* keyframe=state->get_kfdataBase()->get_keyframe(kf_id);
                        Eigen::Matrix<double,9,1> kf_db = keyframe->_intrinsics;
                        Eigen::Matrix<double,8,1> kf_d = kf_db.block(0,0,8,1);
                        bool is_fisheye= int(kf_db(8,0));
                        //get the kf position in world frame.
                        PoseJPL* kf_w=state->_clones_Keyframe[kf_id];
                        Eigen::Matrix<double,3,3> R_KFtoW = kf_w->Rot();
                        Eigen::Matrix<double,3,1> p_KFinW = kf_w->pos();
                        double rot_kftow[3];
                        ceres::RotationMatrixToAngleAxis(R_KFtoW.data(),rot_kftow);
                        Matrix<double,6,1> kf_state=Matrix<double,6,1>::Zero();
                        kf_state<<rot_kftow[0],rot_kftow[1],rot_kftow[2],p_KFinW(0,0),p_KFinW(1,0),p_KFinW(2,0);


                        double ts=state->_timestamp_approx; 

                         cv::Point2f uv_ob=keyframe->matched_point_2d_uv_map.at(ts)[kf_feature_id];
                        double x=uv_ob.x;
                        double y=uv_ob.y;
                        Vector2d uv_OB(x,y);
                        double uv_x=feat->uvs[0][0](0);
                        double uv_y=feat->uvs[0][0](1);
                        Eigen::Matrix<double,2,1> uv_kf;
                        uv_kf << (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].x, (double)keyframe->point_2d_uv_map.at(ts)[kf_feature_id].y;
                        Vector2d uv_feat(uv_x,uv_y);
                        assert(uv_OB==uv_feat); //check if feature link is correct

                        double* point=points+3*num;
                        num++;
                        
                        
                        ceres::CostFunction *cost_function;
                        // cost_function=ObservationError::Create(uv_x,uv_y,p_FinW.data(),intrinsics.data(),fisheye,current_state);
                        // cout<<"p_FinW: "<<p_FinW.transpose()<<endl;
                        cost_function=new ceres::AutoDiffCostFunction<ObservationError3,2,3,3,3>(new ObservationError3(uv_x,uv_y,uv_kf(0),uv_kf(1), kf_state,intrinsics,fisheye,current_state));
                        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

                        _Problem.AddResidualBlock(cost_function, loss_function, transform_rot,transform_pretrans, point);
                        // _Problem.SetParameterBlockConstant(point);

                        ceres::CostFunction *cost_function_2;
                        cost_function_2=new ceres::AutoDiffCostFunction<ObservationError4,2,3>(new ObservationError4(uv_kf(0),uv_kf(1),intrinsics,is_fisheye));
                        _Problem.AddResidualBlock(cost_function_2, NULL, point);
                        // _Problem.SetParameterBlockConstant(point);
                        // _Problem.SetParameterBlockConstant(transform_rot);
                        // _Problem.SetParameterBlockConstant(transform_pretrans);
                        // _Problem.SetParameterBlockConstant(current_state);
      
                  }  
            }
      } 

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
      options.minimizer_progress_to_stdout = true;
      options.max_num_iterations=30;
      ceres::Solver::Summary summary;
      ceres::Solve(options, &_Problem, &summary); 

      Matrix<double,3,3> R_WtoG_optimized;
      rotate_wtog[0]=transform_rot[0];
      rotate_wtog[1]=transform_rot[1];
      rotate_wtog[2]=transform_rot[2];
      ceres::AngleAxisToRotationMatrix(rotate_wtog,R_WtoG_optimized.data());
      Vector3d p_WinG_optimized(transform_pretrans[0],transform_pretrans[1],transform_pretrans[2]);
      Matrix3d R_GtoW_optimized=R_WtoG_optimized.transpose();
      Vector3d p_GinW_optimized=-R_GtoW_optimized*p_WinG_optimized;
      Vector4d q=rot_2_quat(R_GtoW_optimized);
      Matrix<double,7,1> transform_new=Matrix<double,7,1>::Zero();
      transform_new<<q(0),q(1),q(2),q(3),p_GinW_optimized(0),p_GinW_optimized(1),p_GinW_optimized(2);
      state->transform_vio_to_map->set_linp(transform_new);

      // Matrix<double,3,3> R_GtoC_optimized;
      // ceres::AngleAxisToRotationMatrix(current_state,R_GtoC_optimized.data());
      // Vector3d p_GinC_optimized(current_state[3],current_state[4],current_state[5]);
      // Matrix3d R_ItoG_optimized=R_GtoC_optimized.transpose()*R_ItoC;
      // Vector3d p_IinG_optimized=R_GtoC_optimized.transpose()*(p_IinC-p_GinC_optimized);
      // Vector4d q_I=rot_2_quat(R_ItoG_optimized);
      // Matrix<double,7,1> current_new=Matrix<double,7,1>::Zero();
      // current_new<<q_I(0),q_I(1),q_I(2),q_I(3),p_IinG_optimized(0),p_IinG_optimized(1),p_IinG_optimized(2);
      // state->_clones_IMU.at(state->_timestamp)->set_linp(current_new);
      
      num=0;
      // cout<<"points after optimize: "<<endl;
      for(int i=0;i<feature_vec.size();i++) { 
       Feature* feat=feature_vec[i];
            for (auto const& pair : feat->keyframe_matched_obs)
            {
                  for(auto const& match : pair.second )
                  {
                        double kf_id=match.first;
                        size_t kf_feature_id=match.second;
                        Keyframe* keyframe=state->get_kfdataBase()->get_keyframe(kf_id);
                        double ts=state->_timestamp_approx; 
                        cv::Point3f pt;
                        pt.x=points[3*num];
                        pt.y=points[3*num+1];
                        pt.z=points[3*num+2];
                        if(pt.z>0)
                           keyframe->point_3d_map.at(ts)[kf_feature_id]=pt;
                        
                        num++;
                        // cout<<"("<<pt.x<<" "<<pt.y<<" "<<pt.z<<") ";
                  }
            }
      }
      // cout<<endl;
     
      cout<<"tranform after optimize: ";
      for(auto t:transform_rot)
      {
            cout<<t<<" ";
      }
      for(auto t:transform_pretrans)
      {
            cout<<t<<" ";
      }

      std::cout << summary.BriefReport() << "\n";

      // cout<<endl;
      // cout<<"state transform after optimize: "<<state->transform_vio_to_map->quat().transpose()<<"|"<<state->transform_vio_to_map->pos().transpose()<<endl;
       
      


}


