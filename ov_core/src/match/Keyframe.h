//
// Created by zzq on 2020/10/10.
//

#ifndef SRC_KEYFRAME_H
#define SRC_KEYFRAME_H

#include "feat/Feature.h"
#include "types/Type.h"
#include "types/IMU.h"
#include "types/Vec.h"
#include "types/PoseJPL.h"
#include "types/Landmark.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <unordered_map>

using namespace Eigen;
using namespace DVision;
using namespace ov_type;

extern string BRIEF_PATTERN_FILE;


namespace ov_core{


    class KF_BriefExtractor
    {
    public:
        virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
        KF_BriefExtractor(const std::string &pattern_file);

        DVision::BRIEF m_brief;
    };

    class Keyframe {
    public:
        Keyframe(double _time_stamp, size_t _index, cv::Mat& img,size_t camid,vector<cv::Point2f> &_point_2d_uv,
                 vector<cv::Point2f> &_point_2d_norm,vector<size_t> &_point_id,Eigen::VectorXd intrinsics);
//        Keyframe(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
//                 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
//                 vector<size_t> &_point_id, int _sequence);
        Keyframe(double _time_stamp, int _index, Eigen::Matrix<double,7,1> &pos1, Eigen::Matrix<double,7,1> &pos2,
                 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
                 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm,
                 vector<BRIEF::bitset> &_brief_descriptors,Eigen::VectorXd intrinsic);
        Keyframe(double _time_stamp,int _index,size_t camid, cv::Mat& img, VectorXd intrinsics);
        Keyframe()
        {
            _Pose_KFinWorld =new PoseJPL();
            _Pose_KFinVIO= new PoseJPL();
            _Pose_VIOtoWorld=new PoseJPL();
        }
        bool findConnection(Keyframe* old_kf);

        void get_matches(Keyframe* loop_kf, vector<cv::Point2f>& matched_loop_kf_uv, vector<cv::Point2f>& matched_loop_kf_uv_norm,
                        vector<cv::Point3f>& matched_loop_kf_3dpt, vector<cv::Point2f>& matched_query_kf_uv, vector<cv::Point2f>& matched_query_kf_uv_norm);
//        void computeWindowBRIEFPoint();
        //void computeBRIEFPoint();
        //void extractBrief();
        int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);
        bool searchInAera(const BRIEF::bitset window_descriptor,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm,
                          cv::Point2f &best_match,
                          cv::Point2f &best_match_norm, size_t &id);
        void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                              std::vector<cv::Point2f> &matched_2d_old_norm,
                              std::vector<uchar> &status,
                              const std::vector<BRIEF::bitset> &descriptors_old,
                              const std::vector<cv::KeyPoint> &keypoints_old,
                              const std::vector<cv::KeyPoint> &keypoints_old_norm,
                              std::vector<size_t> &matched_id_old);
        bool FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                    const std::vector<cv::Point2f> &matched_2d_old_norm,
                                    vector<uchar> &status, Eigen::VectorXd intrinsics_old_kf,Eigen::MatrixXd &Fund);
        bool RecoverRelativePose(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                 const std::vector<cv::Point2f> &matched_2d_old_norm,Eigen::Matrix3d &K,vector<uchar> &status,
                                 Eigen::Matrix3d &R_cur_to_old,Eigen::Vector3d &p_cur_in_old);

        bool PnPRANSAC(const vector<cv::Point2f> &matched_2d_cur,
                       const std::vector<cv::Point3f> &matched_3d_kf,
                       Eigen::Vector3d &PnP_p_loopIncur, Eigen::Matrix3d &PnP_R_loopTocur,int thred);
        void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
        void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
        void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
        void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
        void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

        Eigen::Vector3d getLoopRelativeT();
        double getLoopRelativeYaw();
        Eigen::Vector4d getLoopRelativeQ();



        double time_stamp;
        string image_name;
        int index=-1;
        int local_index;
        PoseJPL *_Pose_KFinVIO;
        PoseJPL *_Pose_KFinWorld;
        PoseJPL *_Pose_VIOtoWorld;
        Eigen::VectorXd _intrinsics;

        Eigen::Matrix<double,6,6> P_inv;
        Eigen::Matrix<double,6,6> P;


        Eigen::Vector3d vio_T_w_i;  //keyframe location in the current vio system reference
        Eigen::Matrix3d vio_R_w_i;
        Eigen::Vector3d T_w_i;    //kerframe location in the world reference
        Eigen::Matrix3d R_w_i;
        Eigen::Vector3d origin_vio_T;
        Eigen::Matrix3d origin_vio_R;
        cv::Mat image;
        cv::Mat thumbnail;


        vector<cv::Point3f> point_3d;
        vector<cv::Point2f> point_2d_uv;
        vector<cv::Point2f> point_2d_norm;
        vector<size_t> point_id;
        
        //matched_imaged timestamp, matched feature 3d point in this keyframe
        map<double,vector<cv::Point3f>> point_3d_map;
        map<double,vector<cv::Point2f>> point_2d_uv_map;
        map<double,vector<cv::Point2f>> point_2d_uv_norm_map;
        map<double,vector<size_t>> point_id_map;
        map<double,vector<cv::Point3f>> point_3d_linp_map;


        vector<cv::Point2f> matched_point_2d_uv;
        vector<cv::Point2f> matched_point_2d_norm;
        vector<size_t> matched_id;

        map<double,vector<cv::Point2f>> matched_point_2d_uv_map;
        map<double,vector<cv::Point2f>> matched_point_2d_norm_map;
        map<double,vector<size_t>> matched_id_map;

        //the final match id
        vector<size_t> loop_feature_id;//local id in *this keyframe
        vector<size_t> match_feature_id;//global id in feature database

        vector<cv::KeyPoint> keypoints;
        vector<cv::KeyPoint> keypoints_norm;
        vector<Vector3d> keypoints_3d;
        vector<cv::KeyPoint> window_keypoints;
        vector<BRIEF::bitset> brief_descriptors;
        vector<BRIEF::bitset> window_brief_descriptors;
        bool has_fast_point;
        int sequence;
        bool to_delete=false;
        size_t cam_id;

        vector<size_t> _matched_id_cur;
        vector<size_t> _matched_id_old;


        bool has_loop;
        int loop_index=-1;
        vector<int> loop_image_id_vec;

        double loop_img_timestamp=-1;
        vector<double> loop_img_timestamp_vec;
        Eigen::Matrix<double, 8, 1 > loop_info; //*this keyframe相对于loop keyframe的位姿 t,q(w,x,y,z),yaw (from cur to loop)
    };
}



#endif //SRC_KEYFRAME_H
