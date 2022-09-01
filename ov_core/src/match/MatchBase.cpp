//
// Created by zzq on 2020/10/11.
//

#include "MatchBase.h"

void MatchBase::DetectAndMatch()
{
  while(1)
  {
    Keyframe kf;
    {
      std::unique_lock<std::mutex> lck(img_buffer_mutex);
      if(!img_buffer.empty())
      {
        kf = img_buffer.front();
        img_buffer.pop();
      }
    }

    if(kf.index!=-1)
    {
      ExtractFeatureAndDescriptor(kf);

      if(DetectLoop(kf))
      {
        MatchingWithLoop(kf);
      }
    }
  }
}
/*
int MatchBase::detectLoop(Keyframe *keyframe, size_t frame_index) {
    //    cv::Mat compressed_image;
//
//    if (DEBUG_IMAGE)
//    {
//        int feature_num = keyframe->keypoints.size();
//        cv::resize(keyframe->image, compressed_image, cv::Size(376, 240));
//        putText(compressed_image, "feature_num:" + to_string(feature_num), cv::Point2f(10, 10), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
//        image_pool[frame_index] = compressed_image;
//    }
//    TicToc tmp_t;
    //first query; then add this frame into database!
    QueryResults ret;
//    TicToc t_query;
    db->query(keyframe->brief_descriptors, ret, 4);
    //printf("query time: %f", t_query.toc());
    //cout << "Searching for Image " << frame_index << ". " << ret << endl;

//    TicToc t_add;

    ///there is no need to add this keyframe descriptor into db,
    ///as we don't really add current keyframe into keyframe database
    //db.add(keyframe->brief_descriptors);
    //printf("add feature time: %f", t_add.toc());
    // ret[0] is the nearest neighbour's score. threshold change with neighour score
    bool find_loop = false;
    cv::Mat loop_result;
//    if (DEBUG_IMAGE)
//    {
//        loop_result = compressed_image.clone();
//        if (ret.size() > 0)
//            putText(loop_result, "neighbour score:" + to_string(ret[0].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
//    }
//    // visual loop result
//    if (DEBUG_IMAGE)
//    {
//        for (unsigned int i = 0; i < ret.size(); i++)
//        {
//            int tmp_index = ret[i].Id;
//            auto it = image_pool.find(tmp_index);
//            cv::Mat tmp_image = (it->second).clone();
//            putText(tmp_image, "index:  " + to_string(tmp_index) + "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255));
//            cv::hconcat(loop_result, tmp_image, loop_result);
//        }
//    }
    // a good match with its neighbour
    for(int i=0;i<ret.size();i++)
    {
        cout<<"ret"<<i<<"_score: "<<ret[i].Score<<endl;
    }
    if (ret.size() >= 1 &&ret[0].Score > 0.05)
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            //if (ret[i].Score > ret[0].Score * 0.3)
            if (ret[i].Score > 0.015)
            {
                find_loop = true;
                int tmp_index = ret[i].Id;
//                if (DEBUG_IMAGE && 0)
//                {
//                    auto it = image_pool.find(tmp_index);
//                    cv::Mat tmp_image = (it->second).clone();
//                    putText(tmp_image, "loop score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255));
//                    cv::hconcat(loop_result, tmp_image, loop_result);
//                }
            }

        }

    // if (DEBUG_IMAGE)
    // {
    //     cv::imshow("loop_result", loop_result);
    //     cv::waitKey(20);
    // }


    //select the oldest keyframe
    if (find_loop)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
        return min_index;
    }
    else
        return -1;
}
*/