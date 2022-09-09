#ifndef OV_MSCKF_UPDATER_OPTIMIZE_H
#define OV_MSCKF_UPDATER_OPTIMIZE_H

#include <Eigen/Eigen>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "state/State.h"
#include "feat/Feature.h"
#include "match/Keyframe.h"
#include "utils/quat_ops.h"

namespace ov_msckf{


      class ObservationErrorInitialTransform{
            public:

            ObservationErrorInitialTransform(double observation_x, double observation_y ,Eigen::Vector3d point,
                             Eigen::VectorXd intrinsics, bool fisheye ):_observation_x(observation_x),
                             _observation_y(observation_y),_fisheye(fisheye){
                                   
                                   _intrinsic=intrinsics;
                                   _point=point;
                              //      cout<<"_point: "<<_point[0]<<" "<<_point[1]<<" "<<_point[2]<<endl;
                             }
            
            template<typename T>
            bool operator()(const T *const trans_kftocur, T *residuals) const
            {
                   T pt_w[3];
                   pt_w[0]=T(_point(0));
                   pt_w[1]=T(_point(1));
                   pt_w[2]=T(_point(2));
                  //  cout<<"*******pt_w: "<<pt_w[0]<<" "<<pt_w[1]<<" "<<pt_w[2]<<endl;
                  
                  T prediction[2];

                  T pt_cur[3];
                  ceres::AngleAxisRotatePoint(trans_kftocur,pt_w,pt_cur);

                  

                  // pt_vins[0]=rot[0];
                  // pt_vins[1]=rot[1];
                  // pt_vins[2]=rot[2];               
                  // cout<<"pt_w: "<<pt_w[0]<<" "<<pt_w[1]<<" "<<pt_w[2]<<" pt_vins:"<<pt_vins[0]<<" "<<pt_vins[1]<<" "<<pt_vins[2]<<endl;
                  pt_cur[0]+=trans_kftocur[3];
                  pt_cur[1]+=trans_kftocur[4];
                  pt_cur[2]+=trans_kftocur[5];

                  T pt_n[2];
                  pt_n[0]=pt_cur[0]/pt_cur[2];
                  pt_n[1]=pt_cur[1]/pt_cur[2];
                  
                  //distortion
                  T fx=T(_intrinsic(0,0));
                  T fy=T(_intrinsic(1,0));
                  T cx=T(_intrinsic(2,0));
                  T cy=T(_intrinsic(3,0));
                  T k1=T(_intrinsic(4,0));
                  T k2=T(_intrinsic(5,0));
                  T k3=T(_intrinsic(6,0));
                  T k4=T(_intrinsic(7,0));

                  // prediction[0]=fx*pt_n[0]+cx;
                  // prediction[1]=fy*pt_n[1]+cy;
                  if(_fisheye)
                  {
                        // Calculate distorted coordinates for fisheye
                        T r = ceres::sqrt(pt_n[0]*pt_n[0]+pt_n[1]*pt_n[1]);
                        T theta = ceres::atan(r);
                        T theta_d = theta+k1*ceres::pow(theta,3)+k2*ceres::pow(theta,5)+k3*ceres::pow(theta,7)+k4*ceres::pow(theta,9);

                        // Handle when r is small (meaning our xy is near the camera center)
                        T inv_r = (r > T(1e-8))? T(1.0)/r : T(1.0);
                        T cdist = (r > T(1e-8))? theta_d * inv_r : T(1.0);

                        // Calculate distorted coordinates for fisheye
                        T x1 = pt_n[0]*cdist;
                        T y1 = pt_n[1]*cdist;
                        prediction[0] = fx*x1 + cx;
                        prediction[1] = fy*y1 + cy;
                  }
                  else
                  {
                        T r = ceres::sqrt(pt_n[0]*pt_n[0]+pt_n[1]*pt_n[1]);
                        T r_2 = r*r;
                        T r_4 = r_2*r_2;
                        T x1 = pt_n[0]*(T(1.0)+k1*r_2+k2*r_4)+T(2.0)*k3*pt_n[0]*pt_n[1]+k4*(r_2+T(2.0)*pt_n[0]*pt_n[0]);
                        T y1 = pt_n[1]*(T(1.0)+k1*r_2+k2*r_4)+k3*(r_2+T(2.0)*pt_n[1]*pt_n[1])+T(2.0)*k4*pt_n[0]*pt_n[1];
                        prediction[0] = fx*x1 + cx;
                        prediction[1] = fy*y1 + cy;
                  }
                  
                  // ProjectionWithDistortion(transform, pt_w,prediction);

                  residuals[0]=prediction[0]-T(_observation_x);
                  residuals[1]=prediction[1]-T(_observation_y);
                  // cout<<"observation: ("<<_observation_x<<","<<_observation_y<<"); predict: ("<<prediction[0]<<","<<prediction[1]<<")"<<endl;
                   return true;
                  
            }
            
                  

      //       static ceres::CostFunction *Create( double observed_x, double observed_y,double* point,
      //                        double* intrinsics, bool fisheye, double* current_state) {
      //   return (new ceres::AutoDiffCostFunction<ObservationError, 2,3,3>(
      //       new ObservationError(observed_x, observed_y,point,intrinsics,fisheye,current_state)));
      //       }

            private:
            double _observation_x;
            double _observation_y;
            Eigen::VectorXd _intrinsic;//fx,fy,cx,cy,k1,k2,k3,k4
            bool _fisheye;
            Eigen::Vector3d _point; //point3d in world frame


      };


      class UpdaterOptimize{

            public:

            UpdaterOptimize(){}
            
           
            void Optimize_initial_transform_with_cov(State *state, Keyframe* loop_kf, Eigen::Vector3d& p_loopInCur, Eigen::Matrix3d& R_loopToCur, Eigen::MatrixXd& Cov_loopToCur);

          
      };
}



#endif //OV_MSCKF_UPDATER_OPTIMIZE_H