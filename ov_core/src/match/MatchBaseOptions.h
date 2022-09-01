//
// Created by zzq on 2021/9/11.
//
#ifndef OV_CORE_MATCHBASE_OPTIONS_H
#define OV_CORE_MATCHBASE_OPTIONS_H

#include <string>
#include <vector>
#include <Eigen/Eigen>
using namespace std;
namespace ov_core {


    /**
     * @brief Struct which stores all our filter options
     */
    struct MatchBaseOptions {

      string voc_file = "voc_file";

      string map_feature_type = "Brief";

      string brief_pattern_filename = "brief_pattern_filename";

      string map_file = "map";

      string map_path = "path";

      bool show_image = false;

      Eigen::MatrixXd intrinsics;

      bool is_fisheye = false;
        
    };


}

#endif //OV_CORE_MATCHBASE_OPTIONS_H
