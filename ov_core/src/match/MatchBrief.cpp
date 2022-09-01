//
// Created by zzq on 2020/10/11.
//

#include "MatchBrief.h"

using namespace ov_core;

void MatchBrief::loadVocabulary(string voc_file)
{
  cout<<"loading vocabulary..."<<endl;

  voc = new BriefVocabulary(voc_file);

  _db.setVocabulary(*voc,false,0);

  cout<<"done!"<<endl;
}

//* the map should have following format
/*
  image_ts  tx ty tz qx qy qz qw num_features
  u v x y z...
*/
//* for each image, its brief descriptor is stored by .dat, as in VINS_FUSION
void MatchBrief::loadPriorMap(string map_file)
{
  cout<<"load map..."<<endl;
  
  int count_kf = 0;
  int num_feature=0;
  int line_count=0;
  double timestamp;
  Eigen::Matrix<double,7,1> q_kfinw;
  ifstream fi_map;
  fi_map.open(map_file.data());
  assert(fi_map.is_open());

  string line,res;
  while(getline(fi_map,line))
  {

    if(line_count%2==0)
    {
      stringstream ss(line);

      ss>>res;
      timestamp = stod(res);

      ss>>res;
      double tx=stod(res);
      ss>>res;
      double ty=stod(res);
      ss>>res;
      double tz=stod(res);
      ss>>res;
      double qx=stod(res);
      ss>>res;
      double qy=stod(res);
      ss>>res;
      double qz=stod(res);
      ss>>res;
      double qw=stod(res);

      Eigen::Quaterniond q1(qw,qx,qy,qz);
      Eigen::Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
      q_kfinw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx,ty,tz;

      ss>>res;
      num_feature = stoi(res);

    }
    if(line_count%2==1)
    {
      stringstream ss(line);
      Keyframe* kf = new Keyframe();
      kf->time_stamp = timestamp;
      kf->index = count_kf;
      kf->_Pose_KFinWorld->set_value(q_kfinw);

      string brief_filepath = _options.map_path + "/" + to_string(count_kf) + "_briefdes.dat";
      ifstream brief_file(brief_filepath,std::ios::binary);

      for(int i=0;i<num_feature;i++)
      {
        cv::KeyPoint kp,kp_norm;
        ss>>res;
        kp.pt.x = stof(res);
        ss>>res;
        kp.pt.y = stof(res);
        ss>>res;
        double x = stof(res);
        ss>>res;
        double y = stof(res);
        ss>>res;
        double z = stof(res);
        kp_norm.pt.x = x/z;
        kp_norm.pt.y = y/z;
        kf->keypoints.push_back(kp);
        kf->keypoints_norm.push_back(kp_norm);
        kf->keypoints_3d.push_back(Eigen::Vector3d(x,y,z));

        BRIEF::bitset tmp_des;
        brief_file >> tmp_des;
        kf->brief_descriptors.push_back(tmp_des);
      }

      if(_options.show_image)
      {
        string image_name = _options.map_path + "/" + to_string(count_kf) + "_image.png";
        cv::Mat image = cv::imread(image_name.c_str(),0);
        kf->image = image.clone();
      }

      brief_file.close();
      count_kf++;

      kfdatabase->update_kf(kf);
      _db.add(kf->brief_descriptors);
    }
    line_count++;
  }
  cout<<"kfdatabase size: "<<kfdatabase->size()<<endl;
  cout<<"done! The keyframe number of map is "<<count_kf<<endl;
}

void MatchBrief::ExtractFeatureAndDescriptor(Keyframe& kf)
{
  cv::Mat img;
  cv::equalizeHist(kf.image, img);

  std::vector<cv::KeyPoint> pts0_ext;
  Grider_FAST::perform_griding(img, pts0_ext, 500, 5, 3, 15, true);

  std::vector<cv::KeyPoint> pts0_ext_norm;
  pts0_ext_norm.resize(pts0_ext.size());
  //TODO: How to parallel to improving process speed?
  // parallel_for_(cv::Range(0, pts0_ext.size()), LambdaBody([&,this](const cv::Range& range) {
  //   for (int r = range.start; r < range.end; r++) {
  //       cv::Point2f pt = pts0_ext[r].pt;
  //       cv::Point2f pt_norm = this->undistort_point(pt);     
  //       cv::KeyPoint kp;
  //       kp.pt = pt_norm; 
  //       pts0_ext_norm.at(r) = kp;
  //   }
  // }
  // ));
  for(int i=0;i<pts0_ext.size();i++)
  {
    cv::Point2f pt = pts0_ext[i].pt;
    cv::Point2f pt_norm = undistort_point(pt);     
    cv::KeyPoint kp;
    kp.pt = pt_norm; 
    pts0_ext_norm.at(i) = kp;
  }

  kf.keypoints = pts0_ext;
  kf.keypoints_norm = pts0_ext_norm;

  vector<BRIEF::bitset> descriptors;
  extractor(img,pts0_ext,descriptors);
  kf.brief_descriptors = descriptors;

}


bool MatchBrief::DetectLoop(Keyframe& kf)
{
    
    //first query; then add this frame into database!
    QueryResults ret;
    _db.query(kf.brief_descriptors, ret, 4, kfdatabase->size());
    
    cv::Mat compressed_image;
    if (_options.show_image)
    {
        int feature_num = kf.keypoints.size();
        cv::resize(kf.image, compressed_image, cv::Size(376, 240));
        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
        
    }

    //printf("add feature time: %f", t_add.toc());
    // ret[0] is the nearest neighbour's score. threshold change with neighour score
    bool find_loop = false;

    std::cout<<"loop image size: "<<ret.size()<<" with score: ";
    for(int i=0;i<ret.size();i++)
    {
      cout<<ret[i].Score<<" ";
    }
    cout<<endl;
    if(ret.size()==0)
      return false;

    cv::Mat loop_result;
    if (_options.show_image)
    {
        loop_result = compressed_image.clone();
        if (ret.size() > 0)
            putText(loop_result, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
    }
    // visual loop result 
    if (_options.show_image)
    {
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            int tmp_index = ret[i].Id;
            Keyframe* tmp = kfdatabase->get_keyframe(tmp_index);
            cv::Mat tmp_image = tmp->image.clone();
            putText(tmp_image, "index:  " + to_string(tmp_index) + "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
            cv::hconcat(loop_result, tmp_image, loop_result);
            cv::imshow("loop_result", loop_result);
            cv::waitKey(20);
        }

    }
    // a good match with its nerghbour
    if (ret.size() >= 1 &&ret[0].Score > 0.05)
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            //if (ret[i].Score > ret[0].Score * 0.3)
            if (ret[i].Score > 0.015)
            {          
                find_loop = true;
                int tmp_index = ret[i].Id;
                if (_options.show_image )
                {
                    int tmp_index = ret[i].Id;
                    Keyframe* tmp = kfdatabase->get_keyframe(tmp_index);
                    cv::Mat tmp_image = tmp->image.clone();
                    putText(tmp_image, "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
                    cv::hconcat(loop_result, tmp_image, loop_result);
                    cv::imshow("loop_result", loop_result);
                    cv::waitKey(20);
                }
            }
            
        }
    
    vector<int> loop_index;
    if (find_loop)
    {
        
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if(ret[i].Score>0.05)
            {
              loop_index.push_back(ret[i].Id);
            }                
        }

        kf.loop_image_id_vec=loop_index;

        return true;
    }
    loop_index.clear();
    match_map.insert({kf.time_stamp,loop_index});

    return false;
    

}

void MatchBrief::MatchingWithLoop(Keyframe& kf)
{
  std::cout <<"In matching with loop"<< std::endl;
  
  for(int i=0;i<kf.loop_image_id_vec.size();i++)
  {
    Keyframe* loop_kf = kfdatabase->get_keyframe(kf.loop_image_id_vec[i]);
    vector<cv::Point2f> matched_loop_kf_uv, matched_loop_kf_uv_norm;
    vector<cv::Point3f> matched_loop_kf_3dpt;
    vector<cv::Point2f> matched_query_kf_uv, matched_query_kf_uv_norm;

    kf.get_matches(loop_kf,matched_loop_kf_uv,matched_loop_kf_uv_norm,matched_loop_kf_3dpt, matched_query_kf_uv, matched_query_kf_uv_norm);

    assert(matched_loop_kf_3dpt.size()==matched_loop_kf_uv.size());
    assert(matched_loop_kf_3dpt.size()==matched_loop_kf_uv_norm.size());
    assert(matched_query_kf_uv_norm.size()==matched_query_kf_uv.size());
    assert(matched_query_kf_uv_norm.size()==matched_loop_kf_3dpt.size());

    if(matched_query_kf_uv.size() < 10)
        continue; //TODO:

    // Do RANSAC outlier rejection (note since we normalized the max pixel error is now in the normalized cords)
    std::vector<uchar> mask_rsc;
    double max_focallength = std::max(camera_k_OPENCV(0,0),camera_k_OPENCV(1,1));
    cv::findFundamentalMat(matched_query_kf_uv_norm, matched_loop_kf_uv_norm, cv::FM_RANSAC, 1/max_focallength, 0.9, mask_rsc);


    int j = 0;
    for (int k = 0; k < int(matched_query_kf_uv.size()); k++)
    {
      if (mask_rsc[k] == 1)
        {
          matched_loop_kf_uv[j] = matched_loop_kf_uv[k];
          matched_loop_kf_uv_norm[j] = matched_loop_kf_uv_norm[k];
          matched_loop_kf_3dpt[j] = matched_loop_kf_3dpt[k];
          matched_query_kf_uv[j] = matched_query_kf_uv[k];
          matched_query_kf_uv_norm[j] = matched_loop_kf_uv_norm[k];
          j++;
        }
    }
    matched_loop_kf_uv.resize(j);
    matched_loop_kf_uv_norm.resize(j);
    matched_loop_kf_3dpt.resize(j);
    matched_query_kf_uv.resize(j);
    matched_query_kf_uv_norm.resize(j);
    
    cout<<"for loop kf "<<i<<": "<<matched_loop_kf_3dpt.size()<<" matches"<<endl;

    loop_kf->matched_point_2d_uv_map.insert({kf.time_stamp,matched_query_kf_uv});
    loop_kf->matched_point_2d_norm_map.insert({kf.time_stamp,matched_query_kf_uv_norm});
    loop_kf->point_2d_uv_map.insert({kf.time_stamp,matched_loop_kf_uv});
    loop_kf->point_2d_uv_norm_map.insert({kf.time_stamp,matched_loop_kf_uv_norm});
    loop_kf->point_3d_map.insert({kf.time_stamp,matched_loop_kf_3dpt});
    loop_kf->point_3d_linp_map.insert({kf.time_stamp,matched_loop_kf_3dpt});
    loop_kf->loop_img_timestamp_vec.push_back(kf.time_stamp);

  }

  //we finish all the matching works, and the match information into match_map.
  match_map.insert({kf.time_stamp,kf.loop_image_id_vec});
}



BriefExtractor::BriefExtractor(const std::string &pattern_file)
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

void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  m_brief.compute(im, keys, descriptors);
}
