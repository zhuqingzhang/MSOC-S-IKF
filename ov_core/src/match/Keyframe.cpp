//
// Created by zzq on 2020/10/10.
//

#include "Keyframe.h"
using namespace ov_core;

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

Keyframe::Keyframe(double _time_stamp, size_t _index, cv::Mat& img,size_t camid,vector<cv::Point2f> &_point_2d_uv,
                   vector<cv::Point2f> &_point_2d_norm,vector<size_t> &_point_id,Eigen::VectorXd intrinsics): time_stamp(_time_stamp),index(_index)
{
                       cam_id=camid;
    _intrinsics=intrinsics;
    image=img.clone();
    point_2d_uv = _point_2d_uv;
    point_2d_norm = _point_2d_norm;
    point_id=_point_id;
    loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
    has_loop=false;
    loop_index = -1;
  //  computeWindowBRIEFPoint();

}

Keyframe::Keyframe(double _time_stamp, int _index, std::size_t camid, cv::Mat &img, VectorXd intrinsics) {
    cam_id=camid;
    _intrinsics=intrinsics;
    image=img.clone();
    time_stamp=_time_stamp;
    index=_index;

}

// create Keyframe online（without descriptor at first)
//先对已有的2d点point_2d_uv提取描述子，然后再在图片上提取新的fast特征点keypoint并计算keypoint的描述子
//Keyframe::Keyframe(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
//                   vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
//                   vector<size_t> &_point_id, int _sequence)
//{
//    time_stamp = _time_stamp;
//    index = _index;
//    vio_T_w_i = _vio_T_w_i;
//    vio_R_w_i = _vio_R_w_i;
//    T_w_i = vio_T_w_i;
//    R_w_i = vio_R_w_i;
//    origin_vio_T = vio_T_w_i;
//    origin_vio_R = vio_R_w_i;
//    image = _image.clone();
//    cv::resize(image, thumbnail, cv::Size(80, 60));
//    point_3d = _point_3d;
//    point_2d_uv = _point_2d_uv;
//    point_2d_norm = _point_2d_norm;
//    point_id = _point_id;
//    has_loop = false;
//    loop_index = -1;
//    has_fast_point = false;
//    loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
//    sequence = _sequence;
//    computeWindowBRIEFPoint();
//   // computeBRIEFPoint();
////    if(!DEBUG_IMAGE)
////        image.release();
//}

// load previous Keyframe
Keyframe::Keyframe(double _time_stamp, int _index, Eigen::Matrix<double,7,1> &pos1, Eigen::Matrix<double,7,1> &pos2,
                   cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
                   vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm,vector<BRIEF::bitset> &_brief_descriptors,
                   Eigen::VectorXd intrinsics)
{
    time_stamp = _time_stamp;
    index = _index;
    cam_id=-1;
    _intrinsics=intrinsics;

    //for keyframe load from map, we assume VIO frame is World frame
    _Pose_KFinVIO= new PoseJPL();
    _Pose_KFinVIO->set_value(pos2);
    _Pose_KFinVIO->set_fej(pos2);
    _Pose_KFinWorld =new PoseJPL();
    _Pose_KFinWorld->set_value(pos2);
    _Pose_KFinWorld->set_fej(pos2);
    Eigen::Matrix<double,7,1> relative_pose;
    relative_pose<<0,0,0,1,0,0,0;
    _Pose_VIOtoWorld=new PoseJPL();
    _Pose_VIOtoWorld->set_value(relative_pose);
    _Pose_VIOtoWorld->set_value(relative_pose);


//    if (DEBUG_IMAGE)
//    {
//        image = _image.clone();
//        cv::resize(image, thumbnail, cv::Size(80, 60));
//    }
    if (_loop_index != -1)
        has_loop = true;
    else
        has_loop = false;
    loop_index = _loop_index;
    loop_info = _loop_info;
    has_fast_point = false;
    sequence = 0;
    keypoints = _keypoints;
    keypoints_norm = _keypoints_norm;

    brief_descriptors = _brief_descriptors;
}


//void Keyframe::computeWindowBRIEFPoint()
//{
//    BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
//    for(int i = 0; i < (int)point_2d_uv.size(); i++)
//    {
//        cv::KeyPoint key;
//        key.pt = point_2d_uv[i];
//        window_keypoints.push_back(key);
//    }
//    extractor(image, window_keypoints, window_brief_descriptors);
//}

//void Keyframe::computeBRIEFPoint()
//{
//    BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());
//    const int fast_th = 20; // corner detector response threshold
//    if(1)
//        cv::FAST(image, keypoints, fast_th, true);
//    else
//    {
//        vector<cv::Point2f> tmp_pts;
//        cv::goodFeaturesToTrack(image, tmp_pts, 500, 0.01, 10);
//        for(int i = 0; i < (int)tmp_pts.size(); i++)
//        {
//            cv::KeyPoint key;
//            key.pt = tmp_pts[i];
//            keypoints.push_back(key);
//        }
//    }
//    extractor(image, keypoints, brief_descriptors);
//    for (int i = 0; i < (int)keypoints.size(); i++)
//    {
//        Eigen::Vector3d tmp_p;
//        m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
//        cv::KeyPoint tmp_norm;
//        tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());
//        keypoints_norm.push_back(tmp_norm);
//    }
//}

void KF_BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
    m_brief.compute(im, keys, descriptors);
}

void Keyframe::get_matches(Keyframe* loop_kf, vector<cv::Point2f>& matched_loop_kf_uv, vector<cv::Point2f>& matched_loop_kf_uv_norm,
                        vector<cv::Point3f>& matched_loop_kf_3dpt, vector<cv::Point2f>& matched_query_kf_uv, vector<cv::Point2f>& matched_query_kf_uv_norm)
{
  for(int i = 0; i < (int)brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        cv::Point2f best_pt;
        int bestDist = 128;
        int bestIndex = -1;
        for(int j = 0; j < (int)loop_kf->brief_descriptors.size(); j++)
        {

            int dis = HammingDis(brief_descriptors[i], loop_kf->brief_descriptors[j]);
            if(dis < bestDist)
            {
                bestDist = dis;
                bestIndex = j;
            }
        }
        //printf("best dist %d", bestDist);
        if (bestIndex != -1 && bestDist < 50)
        {
          matched_loop_kf_uv.push_back(loop_kf->keypoints[bestIndex].pt);
          matched_loop_kf_uv_norm.push_back(loop_kf->keypoints_norm[bestIndex].pt);
          cv::Point3f pt_3d = cv::Point3f(loop_kf->keypoints_3d[bestIndex](0),loop_kf->keypoints_3d[bestIndex](0),loop_kf->keypoints_3d[bestIndex](0));
          matched_loop_kf_3dpt.push_back(pt_3d);
          matched_query_kf_uv.push_back(keypoints[i].pt);
          matched_query_kf_uv_norm.push_back(keypoints_norm[i].pt);
        }
  
    }
}


bool Keyframe::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::KeyPoint> &keypoints_old,
                            const std::vector<cv::KeyPoint> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm, size_t &id)
{
//   cout<<"in search in area"<<endl;
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {

        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }
    //printf("best dist %d", bestDist);
    if (bestIndex != -1 && bestDist < 50)
    {
        best_match = keypoints_old[bestIndex].pt;
        best_match_norm = keypoints_old_norm[bestIndex].pt;
        id=bestIndex;
        return true;
    }
    else
        return false;
}

void Keyframe::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                                std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::KeyPoint> &keypoints_old,
                                const std::vector<cv::KeyPoint> &keypoints_old_norm,
                                std::vector<size_t> &matched_id_old)
{
    for(int i = 0; i < (int)window_brief_descriptors.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        size_t id=-1;
        if (searchInAera(window_brief_descriptors[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm,id))
        {
            status.push_back(1);
            //cout<<"searchinarea_true"<<endl;
        }
        else
        {
            status.push_back(0);
            //cout<<"searchinarea_false"<<endl;
        }

        matched_2d_old.push_back(pt);
        matched_2d_old_norm.push_back(pt_norm);
        matched_id_old.push_back(id);
    }

}


bool Keyframe::FundmantalMatrixRANSAC(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                      const std::vector<cv::Point2f> &matched_2d_old_norm,
                                      vector<uchar> &status,Eigen::VectorXd intrinsics_old_kf,Eigen::MatrixXd &Fund)
{
    int n = (int)matched_2d_cur_norm.size();
    for (int i = 0; i < n; i++)
        status.push_back(0);
    if (n >= 8)
    {
//        double fx_cur = _intrinsics(0);
//        double fy_cur = _intrinsics(1);
//        double cx_cur = _intrinsics(2);
//        double cy_cur = _intrinsics(3);
//        double fx_old = intrinsics_old_kf(0);
//        double fy_old = intrinsics_old_kf(1);
//        double cx_old = intrinsics_old_kf(2);
//        double cy_old = intrinsics_old_kf(3);
//        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
//        for (int i = 0; i < (int)matched_2d_cur_norm.size(); i++)
//        {
//
//            double tmp_x, tmp_y;
//            tmp_x = fx_cur * matched_2d_cur_norm[i].x + cx_cur;
//            tmp_y = fy_cur * matched_2d_cur_norm[i].y + cy_cur;
//            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);
//
//
//            tmp_x = fx_old * matched_2d_old_norm[i].x + cx_old;
//            tmp_y = fy_old * matched_2d_old_norm[i].y + cy_old;
//            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
//        }
        cv::Mat F;
        cout<<"before findFundamentalMat"<<endl;
        F=cv::findFundamentalMat(matched_2d_cur_norm, matched_2d_old_norm, cv::FM_RANSAC, 3.0, 0.99, status);
        cout<<"after findFundamentalMat"<<endl;
        cv::cv2eigen(F,Fund);
        return true;
    }
    return false;
}

bool Keyframe::RecoverRelativePose(const std::vector<cv::Point2f> &matched_2d_cur_norm,
                                   const std::vector<cv::Point2f> &matched_2d_old_norm, Eigen::Matrix3d &K,
                                   vector<uchar> &status, Eigen::Matrix3d &R_cur_to_old,
                                   Eigen::Vector3d &p_cur_in_old)
{
    int n = (int)matched_2d_cur_norm.size();
    for (int i = 0; i < n; i++)
        status.push_back(0);
    if (n >= 8) {
        double fx_cur = K(0, 0);
        double fy_cur = K(1, 1);
        double cx_cur = K(0, 2);
        double cy_cur = K(1, 2);
        double fx_old = K(0, 0);
        double fy_old = K(1, 1);
        double cx_old = K(0, 2);
        double cy_old = K(1, 2);
        vector<cv::Point2f> tmp_cur(n), tmp_old(n);
        for (int i = 0; i < (int) matched_2d_cur_norm.size(); i++) {

            double tmp_x, tmp_y;
            tmp_x = fx_cur * matched_2d_cur_norm[i].x + cx_cur;
            tmp_y = fy_cur * matched_2d_cur_norm[i].y + cy_cur;
            tmp_cur[i] = cv::Point2f(tmp_x, tmp_y);


            tmp_x = fx_old * matched_2d_old_norm[i].x + cx_old;
            tmp_y = fy_old * matched_2d_old_norm[i].y + cy_old;
            tmp_old[i] = cv::Point2f(tmp_x, tmp_y);
        }
        cv::Mat K_cv;
        cv::eigen2cv(K,K_cv);
        cv::Mat ess;
        cout<<"before findEssentialMat"<<endl;
        status.clear();
        ess=cv::findEssentialMat(tmp_cur,tmp_old,K_cv,cv::RANSAC,0.99,1,status);
        cout<<"status size: "<<status.size()<<" tmp_cur size:"<<tmp_cur.size()<<endl;
        cout<<"after findEssentialMat"<<endl;
        cv::Mat R, t;
        status.clear();
        cv::recoverPose(ess, tmp_cur, tmp_old, K_cv, R, t, status);
        cout<<"status size: "<<status.size()<<" tmp_cur size:"<<tmp_cur.size()<<endl;
        cout<<"after recoverPose"<<endl;
        cv::cv2eigen(R, R_cur_to_old);
        cv::cv2eigen(t, p_cur_in_old);
        return true;
    }
    return false;
}

bool Keyframe::PnPRANSAC(const vector<cv::Point2f> &matched_2d_cur,
                         const std::vector<cv::Point3f> &matched_3d_kf,
                         Eigen::Vector3d &PnP_p_loopIncur, Eigen::Matrix3d &PnP_R_loopTocur,int thred)
{
    //for (int i = 0; i < matched_3d.size(); i++)
    //	printf("3d x: %f, y: %f, z: %f\n",matched_3d[i].x, matched_3d[i].y, matched_3d[i].z );
    //printf("match size %d \n", matched_3d.size());

    cv::Mat r, rvec, t, D, tmp_r;

    cv::Mat inliers;

    bool flag=false;
    cout<<"PnPRansac_in"<<endl;
    cout<<"match_3d_kf.size: "<<matched_3d_kf.size()<<endl;
    // cv::Mat K = (cv::Mat_<double>(3, 3) << _intrinsics(0), 0, _intrinsics(2), 0, _intrinsics(1), _intrinsics(3), 0, 0, 1.0);
    // D=(cv::Mat_<double>(4,1)<<_intrinsics(4),_intrinsics(5),_intrinsics(6),_intrinsics(7));
     cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    if(matched_3d_kf.size()>=20)
    {
    //    flag=solvePnPRansac(matched_3d_kf, matched_2d_cur, K, D, rvec, t, false, 100, 0.1, 0.9, inliers);
       flag=solvePnP(matched_3d_kf,matched_2d_cur,K,D,rvec,t,false,cv::SOLVEPNP_EPNP);
        cout<<"PnPRansac_out with flag="<<flag<<endl;
        cout<<"t: "<<t<<endl;
    }
    else
    {
        return false;
    }
    
    // if(flag)
    // {
    //     flag=solvePnP(matched_3d_kf,matched_2d_cur,K,D,rvec,t,cv::SOLVEPNP_ITERATIVE);
    //     cout<<"t: "<<t<<endl;
    //     // sleep(5);
    // }
    
    



//    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
//        status.push_back(0);
//
//    for( int i = 0; i < inliers.rows; i++)
//    {
//        int n = inliers.at<int>(i);
//        status[n] = 1;
//    }
VectorXd cam_d=_intrinsics;
    if(flag)
    {
      cv::Rodrigues(rvec, r);
      Matrix3d R_pnp, R_w_c_old;
      cv::cv2eigen(r, PnP_R_loopTocur);
      cv::cv2eigen(t, PnP_p_loopIncur);
      cout<<"PnP_p_loopIncur: "<<PnP_p_loopIncur<<endl;
    //   for(int i=0;i<matched_3d_kf.size();i++)
    //   {
    //       Vector3d kf_3d(double(matched_3d_kf[i].x),double(matched_3d_kf[i].y),double(matched_3d_kf[i].z));
    //       Vector2d cur_2d(double(matched_2d_cur[i].x),double(matched_2d_cur[i].y));
    //       Vector3d cur_3d= PnP_R_loopTocur*kf_3d+PnP_p_loopIncur;
    //       Vector2d uv_norm=Vector2d::Zero();
    //       uv_norm<<cur_3d[0]/cur_3d[2],cur_3d[1]/cur_3d[2];
    //       Vector2d uv_dist=Vector2d::Zero();
    //       double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
    //       double r_2 = r*r;
    //       double r_4 = r_2*r_2;
    //       double x1 = uv_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm(0)*uv_norm(1)+cam_d(7)*(r_2+2*uv_norm(0)*uv_norm(0));
    //       double y1 = uv_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm(1)*uv_norm(1))+2*cam_d(7)*uv_norm(0)*uv_norm(1);
    //       uv_dist(0) = cam_d(0)*x1 + cam_d(2);
    //       uv_dist(1) = cam_d(1)*y1 + cam_d(3);

    //       uv_norm=Vector2d::Zero();
    //       uv_norm<<kf_3d[0]/kf_3d[2],kf_3d[1]/kf_3d[2];

    //       Vector2d uv_dist_kf=Vector2d::Zero();
    //       r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
    //       r_2 = r*r;
    //       r_4 = r_2*r_2;
    //       x1 = uv_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm(0)*uv_norm(1)+cam_d(7)*(r_2+2*uv_norm(0)*uv_norm(0));
    //       y1 = uv_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm(1)*uv_norm(1))+2*cam_d(7)*uv_norm(0)*uv_norm(1);
    //       uv_dist_kf(0) = cam_d(0)*x1 + cam_d(2);
    //       uv_dist_kf(1) = cam_d(1)*y1 + cam_d(3);


          
          
          
    //       cout<<"3d kf: "<<matched_3d_kf[i]<<"2d project kf: "<<uv_dist_kf.transpose()<<" 2d cur:"<<matched_2d_cur[i]<<" project: "<<uv_dist.transpose()<<endl;
    //   }
      return true;
    }
    else
    {
        return false;
    }
    
    


}


bool Keyframe::findConnection(Keyframe* old_kf)
{
//    TicToc tmp_t;
    //printf("find Connection\n");
    vector<cv::Point2f> matched_2d_cur, matched_2d_old;
    vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
//    vector<cv::Point3f> matched_3d;
    vector<size_t> matched_id_cur,matched_id_old;
    vector<uchar> status;

//    matched_3d = point_3d;
    matched_2d_cur = point_2d_uv;
    matched_2d_cur_norm = point_2d_norm;
    matched_id_cur = point_id;

//    TicToc t_match;
#if 0
    if (DEBUG_IMAGE)
	    {
	        cv::Mat gray_img, loop_match_img;
	        cv::Mat old_img = old_kf->image;
	        cv::hconcat(image, old_img, gray_img);
	        cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)point_2d_uv.size(); i++)
	        {
	            cv::Point2f cur_pt = point_2d_uv[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)old_kf->keypoints.size(); i++)
	        {
	            cv::Point2f old_pt = old_kf->keypoints[i].pt;
	            old_pt.x += COL;
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        ostringstream path;
	        path << "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "0raw_point.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
    //printf("search by des\n");
    cout<<"before searchByBriedfdes"<<endl;
    searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, old_kf->brief_descriptors, old_kf->keypoints, old_kf->keypoints_norm, matched_id_old);
    cout<<"after searchByBriedfdes"<<endl;
    reduceVector(matched_2d_cur, status);
    reduceVector(matched_2d_old, status);
    reduceVector(matched_2d_cur_norm, status);
    reduceVector(matched_2d_old_norm, status);
//    reduceVector(matched_3d, status);
    reduceVector(matched_id_cur, status);
    reduceVector(matched_id_old, status);
    //printf("search by des finish\n");
    cout<<"matched_2d_cur size:"<<matched_2d_cur.size()<<endl;
    cout<<"matched_2d_old size:"<<matched_2d_old.size()<<endl;
    cout<<"matched_2d_cur_norm size:"<<matched_2d_cur_norm.size()<<endl;
    cout<<"matched_2d_old_norm size:"<<matched_2d_old_norm.size()<<endl;
    cout<<"matched_id_cur size:"<<matched_id_cur.size()<<endl;
    cout<<"matched_id_old size:"<<matched_id_old.size()<<endl;

#if 0
    if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap);
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path, path1, path2;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	        /*
	        path1 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_1.jpg";
	        cv::imwrite( path1.str().c_str(), image);
	        path2 <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "1descriptor_match_2.jpg";
	        cv::imwrite( path2.str().c_str(), old_img);
	        */

	    }
#endif
    status.clear();
    Eigen::MatrixXd Fund;
    Eigen::Matrix3d R_cur_to_old;
    Eigen::Vector3d p_cur_in_old;


    Eigen::VectorXd intrinsics_old_kf=old_kf->_intrinsics.block(0,0,8,1);
    cout<<"before FundamantalrixRANSAC"<<endl;
    if(FundmantalMatrixRANSAC(matched_2d_cur_norm, matched_2d_old_norm, status,intrinsics_old_kf,Fund))
    {
        reduceVector(matched_2d_cur, status);
        reduceVector(matched_2d_old, status);
        reduceVector(matched_2d_cur_norm, status);
        reduceVector(matched_2d_old_norm, status);
//    reduceVector(matched_3d, status);
        reduceVector(matched_id_cur, status);
        reduceVector(matched_id_old, status);

        //we now only support recovery the relative pose with same instrinsics
        cout<<"before assert"<<endl;
        cout<<_intrinsics<<endl;
        cout<<intrinsics_old_kf<<endl;
        assert(_intrinsics==intrinsics_old_kf);
        cout<<"after assert"<<endl;
        Eigen::Matrix3d K;
        K<<_intrinsics(0),0,_intrinsics(2),0,_intrinsics(1),_intrinsics(3),0,0,1;
        cout<<"get K:"<<K<<endl;
//        Eigen::Matrix3d Essential=K.transpose()*Fund*K;

        bool recover;
        cout<<"before RecoverRelativePose"<<endl;
        status.clear();
        recover=RecoverRelativePose(matched_2d_cur_norm,matched_2d_old_norm,K,status,R_cur_to_old,p_cur_in_old);
        if(recover)
        {
            reduceVector(matched_2d_cur, status);
            reduceVector(matched_2d_old, status);
            reduceVector(matched_2d_cur_norm, status);
            reduceVector(matched_2d_old_norm, status);
//    reduceVector(matched_3d, status);
            reduceVector(matched_id_cur, status);
            reduceVector(matched_id_old, status);
        }
        else
        {
            cout<<"unable to recover relative pose"<<endl;
            return false;
        }

    }


#if 0
    if (DEBUG_IMAGE)
	    {
			int gap = 10;
        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
            cv::Mat gray_img, loop_match_img;
            cv::Mat old_img = old_kf->image;
            cv::hconcat(image, gap_image, gap_image);
            cv::hconcat(gap_image, old_img, gray_img);
            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
	        for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f cur_pt = matched_2d_cur[i];
	            cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for(int i = 0; i< (int)matched_2d_old.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x += (COL + gap);
	            cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
	        }
	        for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	        {
	            cv::Point2f old_pt = matched_2d_old[i];
	            old_pt.x +=  (COL + gap) ;
	            cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 1, 8, 0);
	        }

	        ostringstream path;
	        path <<  "/home/tony-ws1/raw_data/loop_image/"
	                << index << "-"
	                << old_kf->index << "-" << "2fundamental_match.jpg";
	        cv::imwrite( path.str().c_str(), loop_match_img);
	    }
#endif
#if 0
//    Eigen::Vector3d PnP_T_old;
//    Eigen::Matrix3d PnP_R_old;
//    Eigen::Vector3d relative_t;
//    Quaterniond relative_q;
//    double relative_yaw;
//    if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
//    {
//        status.clear();
//        PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);
//        reduceVector(matched_2d_cur, status);
//        reduceVector(matched_2d_old, status);
//        reduceVector(matched_2d_cur_norm, status);
//        reduceVector(matched_2d_old_norm, status);
//        reduceVector(matched_3d, status);
//        reduceVector(matched_id, status);
//#if 1
//        if (DEBUG_IMAGE)
//        {
//            int gap = 10;
//            cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
//            cv::Mat gray_img, loop_match_img;
//            cv::Mat old_img = old_kf->image;
//            cv::hconcat(image, gap_image, gap_image);
//            cv::hconcat(gap_image, old_img, gray_img);
//            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);
//            for(int i = 0; i< (int)matched_2d_cur.size(); i++)
//            {
//                cv::Point2f cur_pt = matched_2d_cur[i];
//                cv::circle(loop_match_img, cur_pt, 5, cv::Scalar(0, 255, 0));
//            }
//            for(int i = 0; i< (int)matched_2d_old.size(); i++)
//            {
//                cv::Point2f old_pt = matched_2d_old[i];
//                old_pt.x += (COL + gap);
//                cv::circle(loop_match_img, old_pt, 5, cv::Scalar(0, 255, 0));
//            }
//            for (int i = 0; i< (int)matched_2d_cur.size(); i++)
//            {
//                cv::Point2f old_pt = matched_2d_old[i];
//                old_pt.x += (COL + gap) ;
//                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 0), 2, 8, 0);
//            }
//            cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));
//            putText(notation, "current frame: " + to_string(index) + "  sequence: " + to_string(sequence), cv::Point2f(20, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
//
//            putText(notation, "previous frame: " + to_string(old_kf->index) + "  sequence: " + to_string(old_kf->sequence), cv::Point2f(20 + COL + gap, 30), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255), 3);
//            cv::vconcat(notation, loop_match_img, loop_match_img);
//
//            /*
//            ostringstream path;
//            path <<  "/home/tony-ws1/raw_data/loop_image/"
//                    << index << "-"
//                    << old_kf->index << "-" << "3pnp_match.jpg";
//            cv::imwrite( path.str().c_str(), loop_match_img);
//            */
//            if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
//            {
//                /*
//                cv::imshow("loop connection",loop_match_img);
//                cv::waitKey(10);
//                */
//                cv::Mat thumbimage;
//                cv::resize(loop_match_img, thumbimage, cv::Size(loop_match_img.cols / 2, loop_match_img.rows / 2));
//                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", thumbimage).toImageMsg();
//                msg->header.stamp = ros::Time(time_stamp);
//                pub_match_img.publish(msg);
//            }
//        }
//#endif
//    }
#endif
    cout<<"matched_2d_cur.size :"<<(int)matched_2d_cur.size()<<endl;
    if ((int)matched_2d_cur.size() > 20)
    {
        double relative_yaw=R_cur_to_old.eulerAngles(2,1,0)(0);
         //printf("PNP relative\n");
        //cout << "pnp relative_t " << relative_t.transpose() << endl;
        //cout << "pnp relative_yaw " << relative_yaw << endl;
        if (abs(relative_yaw) < 30.0 && p_cur_in_old.norm() < 20.0)
        {
            cout<<"in if"<<endl;

            Eigen::Vector4d relative_q;
            relative_q=ov_core::rot_2_quat(R_cur_to_old); //in the form of JPL
            has_loop = true;
            loop_index = old_kf->index;
            //in loop_info, we also record the relative_q in the form of JPL (x,y,z,w)
            loop_info << p_cur_in_old.x(), p_cur_in_old.y(), p_cur_in_old.z(),
                    relative_q(0), relative_q(1), relative_q(2), relative_q(3),
                    relative_yaw;
            cout<<"before assert"<<endl;
            assert(matched_id_cur.size()==matched_id_old.size());
            cout<<"after assert"<<endl;

            _matched_id_cur=matched_id_cur;
            _matched_id_old=matched_id_old;
            //cout << "pnp relative_t " << relative_t.transpose() << endl;
            //cout << "pnp relative_q " << relative_q.w() << " " << relative_q.vec().transpose() << endl;
            return true;
        }
    }
    //printf("loop final use num %d %lf--------------- \n", (int)matched_2d_cur.size(), t_match.toc());
    return false;
}


int Keyframe::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

void Keyframe::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void Keyframe::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void Keyframe::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void Keyframe::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    vio_T_w_i = _T_w_i;
    vio_R_w_i = _R_w_i;
    T_w_i = vio_T_w_i;
    R_w_i = vio_R_w_i;
}

Eigen::Vector3d Keyframe::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Vector4d Keyframe::getLoopRelativeQ()
{
    return Eigen::Vector4d(loop_info(4), loop_info(5), loop_info(6), loop_info(3));
}

double Keyframe::getLoopRelativeYaw()
{
    return loop_info(7);
}

void Keyframe::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
    if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
    {
        //printf("update loop info\n");
        loop_info = _loop_info;
    }
}

KF_BriefExtractor::KF_BriefExtractor(const std::string &pattern_file)
{
    // The DVision::BRIEF extractor computes a random pattern by default when
    // the object is created.
    // We load the pattern that we used to build the vocabulary, to make
    // the descriptors compatible with the predefined vocabulary

    // loads the pattern
    cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
    if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

    vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;

    m_brief.importPairs(x1, y1, x2, y2);
}
