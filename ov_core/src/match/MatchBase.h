//
// Created by zzq on 2020/10/11.
//

#ifndef SRC_MATCHBASE_H
#define SRC_MATCHBASE_H

#include "match/KeyframeDatabase.h"
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "ThirdParty/DBoW/TemplatedDatabase.h"
#include "ThirdParty/DBoW/TemplatedVocabulary.h"
#include "track/Grider_FAST.h"
#include "match/MatchBaseOptions.h"
#include <queue>
#include <fstream>
#include <sstream>
#include <string>
#include "types/PoseJPL.h"
#include <Eigen/Geometry>

using namespace ov_core;
using namespace DBoW2;





namespace ov_core{

    class MapFeatureType
    {
      public:
       enum Type
      {
        Brief,
        ORB,
        UNKNOWN
      };

      static inline Type from_string(const std::string& feat_type) {
            if(feat_type=="Brief") return Brief;
            if(feat_type=="ORB") return ORB;
            return UNKNOWN;
      }
    };
    
   

    class MatchBase {

    public:
    MatchBase(ov_core::MatchBaseOptions &options): kfdatabase(new KeyframeDatabase()){
      _options = options;

      cv::Matx33d tempK;
      tempK(0, 0) = _options.intrinsics(0);
      tempK(0, 1) = 0;
      tempK(0, 2) = _options.intrinsics(2);
      tempK(1, 0) = 0;
      tempK(1, 1) = _options.intrinsics(1);
      tempK(1, 2) = _options.intrinsics(3);
      tempK(2, 0) = 0;
      tempK(2, 1) = 0;
      tempK(2, 2) = 1;
      camera_k_OPENCV = tempK;
      // Distortion parameters
      cv::Vec4d tempD;
      tempD(0) = _options.intrinsics(4);
      tempD(1) = _options.intrinsics(5);
      tempD(2) = _options.intrinsics(6);
      tempD(3) = _options.intrinsics(7);
      camera_d_OPENCV = tempD;

      _is_fisheye = _options.is_fisheye;
        
    }


    void feed_image(ov_core::Keyframe& img)
    {
      std::unique_lock<std::mutex> lck(img_buffer_mutex);
      img_buffer.push(img);

      return;
    }

    void DetectAndMatch();

    virtual void loadVocabulary(string voc_file)=0;

    virtual void loadPriorMap(string map_file)=0;

    virtual void ExtractFeatureAndDescriptor(Keyframe& kf)=0;

    virtual bool DetectLoop(Keyframe& kf)=0;

    virtual void MatchingWithLoop(Keyframe& kf)=0;


    cv::Point2f undistort_point(cv::Point2f pt_in)
    {
      if(_is_fisheye)
      {
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = pt_in.x;
        mat.at<float>(0, 1) = pt_in.y;
        mat = mat.reshape(2); // Nx1, 2-channel
        // Undistort it!
        cv::fisheye::undistortPoints(mat, mat, camera_k_OPENCV, camera_d_OPENCV);
        // Construct our return vector
        cv::Point2f pt_out;
        mat = mat.reshape(1); // Nx2, 1-channel
        pt_out.x = mat.at<float>(0, 0);
        pt_out.y = mat.at<float>(0, 1);
        return pt_out;
      }
      else
      {
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = pt_in.x;
        mat.at<float>(0, 1) = pt_in.y;
        mat = mat.reshape(2); // Nx1, 2-channel
        // Undistort it!
        cv::undistortPoints(mat, mat, camera_k_OPENCV, camera_d_OPENCV);
        // Construct our return vector
        cv::Point2f pt_out;
        mat = mat.reshape(1); // Nx2, 1-channel
        pt_out.x = mat.at<float>(0, 0);
        pt_out.y = mat.at<float>(0, 1);
        return pt_out;
      }
    }

    

    

    KeyframeDatabase* get_interval_kfdatabase()
    {
      return kfdatabase;
    }

    bool finish_matching_thread(double ts)
    {
      if(match_map.find(ts)!=match_map.end())
        return true;
      return false;
    }

    vector<int> get_matchkf_ids(double ts)
    {
      vector<int> res;
      res.clear();
      if(match_map.find(ts)!=match_map.end())
      {
        return match_map.at(ts);
      }
      else
      {
        return res;
      }
    }

    vector<Keyframe*> get_matchkfs(double ts)
    {
      vector<Keyframe*> res;
      res.clear();
      if(match_map.find(ts)!=match_map.end())
      {
        vector<int> match_ids;
        match_ids = match_map.at(ts);
        for(int i=0; i<match_ids.size(); i++)
        {
          Keyframe* kf = kfdatabase->get_keyframe(match_ids[i]);
          assert(kf!=nullptr);
          res.push_back(kf);
        }
        return res;
      }
      else
      {
        return res;
      }
    }
    
    MatchBaseOptions _options;



    protected:
    
    std::mutex img_buffer_mutex; 
    std::queue<ov_core::Keyframe> img_buffer;
    KeyframeDatabase *kfdatabase;
    //* current image ts v.s. matched map kf ids
    std::map<double,vector<int>> match_map;
    
    /// Camera intrinsics in OpenCV format
    cv::Matx33d camera_k_OPENCV;
    /// Camera distortion in OpenCV format
    cv::Vec4d camera_d_OPENCV;
    ///
    bool _is_fisheye;




    };

}



#endif //SRC_MATCHBASE_H
