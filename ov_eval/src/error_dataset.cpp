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


#include <string>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>


#include "calc/ResultTrajectory.h"
#include "utils/Loader.h"
#include "utils/Colors.h"

#ifdef HAVE_PYTHONLIBS

// import the c++ wrapper for matplot lib
// https://github.com/lava/matplotlib-cpp
// sudo apt-get install python-matplotlib python-numpy python2.7-dev
#include "plot/matplotlibcpp.h"

#endif

void save_rpe_2d_data(std::map<double,std::pair<ov_eval::Statistics,ov_eval::Statistics>>& rpe_2d_dataset,string name)
{
  
   
   for(auto &seg : rpe_2d_dataset) {
      ofstream fo;
      string output_file = "/tmp/rpe_2d_data"+name+"_seg"+to_string(int(seg.first))+".txt";
      if(boost::filesystem::exists(output_file))
      {
            boost::filesystem::remove(output_file);
            cout<<"exist output_file, remove."<<endl;
      }

      fo.open(output_file,ofstream::out|ofstream::app);
      seg.second.first.calculate();
      seg.second.second.calculate();
      double size = seg.second.second.values.size();
      for(int i=0;i<size;i++)
      {
        fo<<to_string(seg.second.second.values.at(i))<<endl;
      }
      fo.close();          
   }
  
}

void save_rpe_3d_data(std::map<double,std::pair<ov_eval::Statistics,ov_eval::Statistics>>& rpe_3d_dataset,string name)
{
  
   
   for(auto &seg : rpe_3d_dataset) {
      ofstream fo;
      string output_file = "/tmp/rpe_3d_data"+name+"_seg"+to_string(int(seg.first))+".txt";
      if(boost::filesystem::exists(output_file))
      {
            boost::filesystem::remove(output_file);
            cout<<"exist output_file, remove."<<endl;
      }

      fo.open(output_file,ofstream::out|ofstream::app);
      seg.second.first.calculate();
      seg.second.second.calculate();
      double size = seg.second.second.values.size();
      for(int i=0;i<size;i++)
      {
        fo<<to_string(seg.second.second.values.at(i))<<endl;
      }
      fo.close();          
   }
  
}


int main(int argc, char **argv) {

    // Ensure we have a path
    if(argc < 4) {
        printf(RED "ERROR: Please specify a align mode, folder, and algorithms\n" RESET);
        printf(RED "ERROR: ./error_dataset <align_mode> <file_gt.txt> <folder_algorithms>\n" RESET);
        printf(RED "ERROR: rosrun ov_eval error_dataset <align_mode> <file_gt.txt> <folder_algorithms>\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // Load it!
    boost::filesystem::path path_gt(argv[2]);
    std::vector<double> times;
    std::vector<double> times_for_transform;
    std::vector<Eigen::Matrix<double,7,1>> poses;
    std::vector<Eigen::Matrix3d> cov_ori, cov_pos;
    std::vector<Eigen::Matrix<double,7,1>> poses_for_transform;
    std::vector<Eigen::Matrix3d> cov_ori_for_transform, cov_pos_for_transform;
    ov_eval::Loader::load_data(argv[2], times, poses, cov_ori, cov_pos);
    
    string gt_path = argv[2];
    string gt_name = path_gt.stem().string();
    
    std::cout<<"gt_name: "<<gt_name<<endl;
    string gt_traj_path;
    boost::filesystem::path path_traj_gt;
    bool is_trans=false;
    if(gt_name=="transform") 
    {
      gt_traj_path = path_gt.branch_path().string()+"/traj.txt";
      cout<<"gt_traj_path: "<<gt_traj_path<<endl;
      path_traj_gt = boost::filesystem::path(gt_traj_path);
      ov_eval::Loader::load_data(gt_traj_path, times_for_transform, poses_for_transform, cov_ori_for_transform, cov_pos_for_transform);
      is_trans=true;
    }
    

    // Print its length and stats
    double length = ov_eval::Loader::get_total_length(poses);
    printf("[COMP]: %d poses in %s => length of %.2f meters\n",(int)times.size(),path_gt.stem().string().c_str(),length);


    // Get the algorithms we will process
    // Also create empty statistic objects for each of our datasets
    std::string path_algos(argv[3]);
    std::vector<boost::filesystem::path> path_algorithms;
    for(const auto& entry : boost::filesystem::directory_iterator(path_algos)) {
        if(boost::filesystem::is_directory(entry)) {
            path_algorithms.push_back(entry.path());

        }
    }
    std::sort(path_algorithms.begin(), path_algorithms.end());


    //===============================================================================
    //===============================================================================
    //===============================================================================


    // Relative pose error segment lengths
    //std::vector<double> segments = {8.0, 16.0, 24.0, 32.0, 40.0};
    // std::vector<double> segments = {7.0, 14.0, 21.0, 28.0, 35.0};
    std::vector<double> segments = {100.0, 200.0, 500.0};


    //===============================================================================
    //===============================================================================
    //===============================================================================


    // Loop through each algorithm type
    for(size_t i=0; i<path_algorithms.size(); i++) {

        // Debug print
        printf("======================================\n");
        printf("[COMP]: processing %s algorithm\n", path_algorithms.at(i).stem().c_str());

        // Get the list of datasets this algorithm records
        std::map<std::string,boost::filesystem::path> path_algo_datasets;
        for(auto& entry : boost::filesystem::directory_iterator(path_algorithms.at(i))) {
            if(boost::filesystem::is_directory(entry)) {
                path_algo_datasets.insert({entry.path().stem().string(),entry.path()});
            }
        }

        // Check if we have runs for our dataset
        if(path_algo_datasets.find(path_gt.stem().string())==path_algo_datasets.end()) {
            printf(RED "[COMP]: %s dataset does not have any runs for %s!!!!!\n" RESET,path_algorithms.at(i).stem().c_str(),path_gt.stem().c_str());
            continue;
        }

        
        if(is_trans)
        {
          if(path_algo_datasets.find(path_traj_gt.stem().string())==path_algo_datasets.end()) {
            printf(RED "[COMP]: %s dataset does not have any runs for %s to compute transform nees!!!!!\n" RESET,path_algorithms.at(i).stem().c_str(),path_traj_gt.stem().c_str());
            continue;
          }
        }


        // Errors for this specific dataset (i.e. our averages over the total runs)
        ov_eval::Statistics ate_dataset_ori, ate_dataset_pos;
        ov_eval::Statistics ate_2d_dataset_ori, ate_2d_dataset_pos;
        std::map<double,std::pair<ov_eval::Statistics,ov_eval::Statistics>> rpe_dataset;
        std::map<double,std::pair<ov_eval::Statistics,ov_eval::Statistics>> rpe_2d_dataset;
        for(const auto& len : segments) {
            rpe_dataset.insert({len,{ov_eval::Statistics(),ov_eval::Statistics()}});
            rpe_2d_dataset.insert({len,{ov_eval::Statistics(),ov_eval::Statistics()}});
        }
        std::map<double,std::pair<ov_eval::Statistics,ov_eval::Statistics>> rmse_dataset;
        std::map<double,std::pair<ov_eval::Statistics,ov_eval::Statistics>> rmse_2d_dataset;
        std::map<double,std::pair<ov_eval::Statistics,ov_eval::Statistics>> nees_dataset;
        std::map<double,std::pair<ov_eval::Statistics,ov_eval::Statistics>> nees_inv_dataset;

        // Loop though the different runs for this dataset
        std::vector<std::string> file_paths;
       
        for(auto& entry : boost::filesystem::directory_iterator(path_algo_datasets.at(path_gt.stem().string()))) {
            if(entry.path().extension() != ".txt")
                continue;
            file_paths.push_back(entry.path().string());
        }
        std::sort(file_paths.begin(), file_paths.end());

        // Check if we have runs
        if(file_paths.empty()) {
            printf(RED "\tERROR: No runs found for %s, is the folder structure right??\n" RESET, path_algorithms.at(i).stem().c_str());
            continue;
        }
        
        std::vector<std::string> file_traj_paths;
        if(is_trans)
        {
          for(auto& entry : boost::filesystem::directory_iterator(path_algo_datasets.at(path_traj_gt.stem().string()))) {
              if(entry.path().extension() != ".txt")
                  continue;
              file_traj_paths.push_back(entry.path().string());
              // cout<<"traj_path: "<<entry.path().string()<<endl;
          }
          std::sort(file_traj_paths.begin(), file_traj_paths.end());

          // Check if we have runs
          if(file_traj_paths.empty()) {
              printf(RED "\tERROR: No runs found for %s to compute nees for transform, is the folder structure right??\n" RESET, path_algorithms.at(i).stem().c_str());
              continue;
          }
          assert(file_paths.size()==file_traj_paths.size());
        }

        // Loop though the different runs for this dataset
        for(int file_count=0;file_count<file_paths.size();file_count++) {

            // Create our trajectory object
            std::string path_gttxt = path_gt.string();
            ov_eval::ResultTrajectory traj(file_paths[file_count], path_gttxt, argv[1]);
            

            // Calculate ATE error for this dataset
            ov_eval::Statistics error_ori, error_pos;
            traj.calculate_ate(error_ori, error_pos);
            ate_dataset_ori.values.push_back(error_ori.rmse);
            ate_dataset_pos.values.push_back(error_pos.rmse);
            // cout<<"error_ori.values.size: "<<error_ori.values.size()<<endl;
            for(size_t j=0; j<error_ori.values.size(); j++) {
              // cout<<"error_ori.values.at(j) "<<error_ori.values.at(j)<<endl;
                rmse_dataset[error_ori.timestamps.at(j)].first.values.push_back(error_ori.values.at(j));
                rmse_dataset[error_pos.timestamps.at(j)].second.values.push_back(error_pos.values.at(j));
                assert(error_ori.timestamps.at(j)==error_pos.timestamps.at(j));
            }

            // Calculate ATE 2D error for this dataset
            ov_eval::Statistics error_ori_2d, error_pos_2d;
            traj.calculate_ate_2d(error_ori_2d, error_pos_2d);
            ate_2d_dataset_ori.values.push_back(error_ori_2d.rmse);
            ate_2d_dataset_pos.values.push_back(error_pos_2d.rmse);
            for(size_t j=0; j<error_ori_2d.values.size(); j++) {
                rmse_2d_dataset[error_ori_2d.timestamps.at(j)].first.values.push_back(error_ori_2d.values.at(j));
                rmse_2d_dataset[error_pos_2d.timestamps.at(j)].second.values.push_back(error_pos_2d.values.at(j));
                assert(error_ori_2d.timestamps.at(j)==error_pos_2d.timestamps.at(j));
            }
            
            

            // NEES error for this dataset
            ov_eval::Statistics nees_ori, nees_pos;
            ov_eval::Statistics nees_inv_ori, nees_inv_pos;
            if(!is_trans)
            {
              traj.calculate_nees(nees_ori, nees_pos);
              traj.calculate_inv_nees(nees_inv_ori,nees_inv_pos);
            }
            else
            {
              std::string path_traj_gttxt = path_traj_gt.string();
              ov_eval::ResultTrajectory traj_vins(file_traj_paths[file_count], path_traj_gttxt, argv[1]);
              assert(traj_vins.est_times.size()==traj_vins.gt_times.size());
              double first_ts_of_trans = traj.est_times[0];
              int index=-1;
              for(int time_id=0;time_id<traj_vins.est_times.size();time_id++)
              {
                  if(traj_vins.est_times[time_id]==first_ts_of_trans)
                  {
                    index=time_id;
                    break;
                  }
              }
              assert(index!=-1);
              std::vector<double> est_time_reduced;
              std::vector<Eigen::Matrix<double,7,1>> est_poses_reduced;
              std::vector<Eigen::Matrix<double,7,1>> gt_poses_reduced;
              for(int time_id=index; time_id<traj_vins.est_times.size(); time_id++)
              {
                est_time_reduced.push_back(traj_vins.est_times[time_id]);
                est_poses_reduced.push_back(traj_vins.est_poses[time_id]);
                gt_poses_reduced.push_back(traj_vins.gt_poses_aignedtoEST[time_id]);
              }
              assert(est_time_reduced.size()==traj.est_times.size());
              assert(est_poses_reduced.size()==traj.est_poses.size());
              assert(gt_poses_reduced.size()==traj.gt_poses.size());

              traj.calculate_nees(nees_ori,nees_pos);
              traj.calculate_invariant_transform_nees(est_poses_reduced,gt_poses_reduced,nees_inv_ori,nees_inv_pos);
            }

            for(size_t j=0; j<nees_ori.values.size(); j++) {
                  nees_dataset[nees_ori.timestamps.at(j)].first.values.push_back(nees_ori.values.at(j));
                  nees_dataset[nees_ori.timestamps.at(j)].second.values.push_back(nees_pos.values.at(j));
                  nees_inv_dataset[nees_inv_ori.timestamps.at(j)].first.values.push_back(nees_inv_ori.values.at(j));
                  nees_inv_dataset[nees_inv_ori.timestamps.at(j)].second.values.push_back(nees_inv_pos.values.at(j));
                  assert(nees_ori.timestamps.at(j)==nees_pos.timestamps.at(j));
                  assert(nees_inv_ori.timestamps.at(j)==nees_inv_pos.timestamps.at(j));
            }
            


            // Calculate RPE error for this dataset
            std::map<double,std::pair<ov_eval::Statistics,ov_eval::Statistics>> error_rpe;
            traj.calculate_rpe(segments, error_rpe);
            for(const auto& elm : error_rpe) {
                rpe_dataset.at(elm.first).first.values.insert(rpe_dataset.at(elm.first).first.values.end(),elm.second.first.values.begin(),elm.second.first.values.end());
                rpe_dataset.at(elm.first).first.timestamps.insert(rpe_dataset.at(elm.first).first.timestamps.end(),elm.second.first.timestamps.begin(),elm.second.first.timestamps.end());
                rpe_dataset.at(elm.first).second.values.insert(rpe_dataset.at(elm.first).second.values.end(),elm.second.second.values.begin(),elm.second.second.values.end());
                rpe_dataset.at(elm.first).second.timestamps.insert(rpe_dataset.at(elm.first).second.timestamps.end(),elm.second.second.timestamps.begin(),elm.second.second.timestamps.end());
            }

            // Calculate RPE 2d error for this dataset
            std::map<double,std::pair<ov_eval::Statistics,ov_eval::Statistics>> error_rpe_2d;
            traj.calculate_rpe_2d(segments, error_rpe_2d);
            for(const auto& elm : error_rpe_2d) {
                rpe_2d_dataset.at(elm.first).first.values.insert(rpe_2d_dataset.at(elm.first).first.values.end(),elm.second.first.values.begin(),elm.second.first.values.end());
                rpe_2d_dataset.at(elm.first).first.timestamps.insert(rpe_2d_dataset.at(elm.first).first.timestamps.end(),elm.second.first.timestamps.begin(),elm.second.first.timestamps.end());
                rpe_2d_dataset.at(elm.first).second.values.insert(rpe_2d_dataset.at(elm.first).second.values.end(),elm.second.second.values.begin(),elm.second.second.values.end());
                rpe_2d_dataset.at(elm.first).second.timestamps.insert(rpe_2d_dataset.at(elm.first).second.timestamps.end(),elm.second.second.timestamps.begin(),elm.second.second.timestamps.end());
            }

        }

        // Compute our mean ATE score
        ate_dataset_ori.calculate();
        ate_dataset_pos.calculate();
        ate_2d_dataset_ori.calculate();
        ate_2d_dataset_pos.calculate();

        // Print stats for this specific dataset
        std::string prefix = (ate_dataset_ori.mean > 10 || ate_dataset_pos.mean > 10)? RED : "";
        printf("%s\tATE: mean_ori = %.3f | mean_pos = %.3f (%d runs)\n" RESET,prefix.c_str(),ate_dataset_ori.mean,ate_dataset_pos.mean,(int)ate_dataset_ori.values.size());
        printf("\tATE: std_ori  = %.5f | std_pos  = %.5f\n",ate_dataset_ori.std,ate_dataset_pos.std);
        printf("\tATE 2D: mean_ori = %.3f | mean_pos = %.3f (%d runs)\n",ate_2d_dataset_ori.mean,ate_2d_dataset_pos.mean,(int)ate_2d_dataset_ori.values.size());
        printf("\tATE 2D: std_ori  = %.5f | std_pos  = %.5f\n",ate_2d_dataset_ori.std,ate_2d_dataset_pos.std);
        for(auto &seg : rpe_dataset) {
            seg.second.first.calculate();
            seg.second.second.calculate();
            printf("\tRPE: seg %d - mean_ori = %.3f | mean_pos = %.3f (%d samples)\n",(int)seg.first,seg.second.first.mean,seg.second.second.mean,(int)seg.second.second.values.size());
            printf("              - std_ori  = %.3f | std_pos  = %.3f\n",seg.second.first.std,seg.second.second.std);
            printf("              - min_ori  = %.3f | min_pos  = %.3f\n",seg.second.first.min,seg.second.second.min);
            printf("              - max_ori  = %.3f | max_pos  = %.3f\n",seg.second.first.max,seg.second.second.max);

        }
        save_rpe_3d_data(rpe_dataset,path_algorithms.at(i).stem().c_str());

        for(auto &seg : rpe_2d_dataset) {
            seg.second.first.calculate();
            seg.second.second.calculate();
            printf("\tRPE 2D: seg %d - mean_ori = %.3f | mean_pos = %.3f (%d samples)\n",(int)seg.first,seg.second.first.mean,seg.second.second.mean,(int)seg.second.second.values.size());
            printf("                 - std_ori  = %.3f | std_pos  = %.3f\n",seg.second.first.std,seg.second.second.std);
            printf("                 - min_ori  = %.3f | min_pos  = %.3f\n",seg.second.first.min,seg.second.second.min);
            printf("                 - max_ori  = %.3f | max_pos  = %.3f\n",seg.second.first.max,seg.second.second.max);

        }

        save_rpe_2d_data(rpe_2d_dataset,path_algorithms.at(i).stem().c_str());


        // RMSE: Convert into the right format (only use times where all runs have an error)
        ov_eval::Statistics rmse_ori, rmse_pos;
        for(auto &elm : rmse_dataset) {
          // cout<<"elm.second.first.values.size(): "<<elm.second.first.values.size()<<" file_paths.size():" <<file_paths.size()<<endl;
            if(elm.second.first.values.size()==file_paths.size()) {
                elm.second.first.calculate();
                elm.second.second.calculate();
                rmse_ori.timestamps.push_back(elm.first);
                rmse_ori.values.push_back(elm.second.first.rmse);
                rmse_pos.timestamps.push_back(elm.first);
                rmse_pos.values.push_back(elm.second.second.rmse);
            }
        }
        rmse_ori.calculate();
        rmse_pos.calculate();
        printf("\tRMSE: mean_ori = %.3f | mean_pos = %.3f\n",rmse_ori.mean,rmse_pos.mean);

        // RMSE: Convert into the right format (only use times where all runs have an error)
        ov_eval::Statistics rmse_2d_ori, rmse_2d_pos;
        for(auto &elm : rmse_2d_dataset) {
            if(elm.second.first.values.size()==file_paths.size()) {
                elm.second.first.calculate();
                elm.second.second.calculate();
                rmse_2d_ori.timestamps.push_back(elm.first);
                rmse_2d_ori.values.push_back(elm.second.first.rmse);
                rmse_2d_pos.timestamps.push_back(elm.first);
                rmse_2d_pos.values.push_back(elm.second.second.rmse);
            }
        }
        rmse_2d_ori.calculate();
        rmse_2d_pos.calculate();
        printf("\tRMSE 2D: mean_ori = %.3f | mean_pos = %.3f\n",rmse_2d_ori.mean,rmse_2d_pos.mean);

        // NEES: Convert into the right format (only use times where all runs have an error)
        ov_eval::Statistics nees_ori, nees_pos;
        ov_eval::Statistics nees_inv_ori, nees_inv_pos;
        for(auto &elm : nees_dataset) {
            if(elm.second.first.values.size()==file_paths.size()) {
                elm.second.first.calculate();
                elm.second.second.calculate();
                nees_ori.timestamps.push_back(elm.first);
                nees_ori.values.push_back(elm.second.first.mean);
                nees_pos.timestamps.push_back(elm.first);
                nees_pos.values.push_back(elm.second.second.mean);
            }
        }
        nees_ori.calculate();
        nees_pos.calculate();
        printf("\tNEES: mean_ori = %.8f | mean_pos = %.8f\n",nees_ori.mean/3.0,nees_pos.mean/3.0);

        for(auto &elm : nees_inv_dataset) {
            if(elm.second.first.values.size()==file_paths.size()) {
                elm.second.first.calculate();
                elm.second.second.calculate();
                nees_inv_ori.timestamps.push_back(elm.first);
                nees_inv_ori.values.push_back(elm.second.first.mean);
                nees_inv_pos.timestamps.push_back(elm.first);
                nees_inv_pos.values.push_back(elm.second.second.mean);
            }
        }
        nees_inv_ori.calculate();
        nees_inv_pos.calculate();
        printf("\tNEES_inv: mean_ori = %.8f | mean_pos = %.8f\n",nees_inv_ori.mean/3.0,nees_inv_pos.mean/3.0);


#ifdef HAVE_PYTHONLIBS

        //=====================================================
        // RMSE plot at each timestep
        matplotlibcpp::figure_size(1000, 600);

        // Zero our time arrays
        double starttime1 = (rmse_ori.timestamps.empty())? 0 : rmse_ori.timestamps.at(0);
        double endtime1 = (rmse_ori.timestamps.empty())? 0 : rmse_ori.timestamps.at(rmse_ori.timestamps.size()-1);
        for(size_t j=0; j<rmse_ori.timestamps.size(); j++) {
            rmse_ori.timestamps.at(j) -= starttime1;
            rmse_pos.timestamps.at(j) -= starttime1;
        }

        // Update the title and axis labels
        matplotlibcpp::subplot(2,1,1);
        matplotlibcpp::title("Root Mean Squared Error - "+path_algorithms.at(i).stem().string());
        matplotlibcpp::ylabel("Error Orientation (deg)");
        matplotlibcpp::plot(rmse_ori.timestamps, rmse_ori.values);
        matplotlibcpp::xlim(0.0,endtime1-starttime1);
        matplotlibcpp::subplot(2,1,2);
        matplotlibcpp::ylabel("Error Position (m)");
        matplotlibcpp::xlabel("dataset time (s)");
        matplotlibcpp::plot(rmse_pos.timestamps, rmse_pos.values);
        matplotlibcpp::xlim(0.0,endtime1-starttime1);

        // Display to the user
        matplotlibcpp::tight_layout();
        matplotlibcpp::show(false);

        //=====================================================

        if(!nees_ori.values.empty() && !nees_pos.values.empty()) {
            // NEES plot at each timestep
            matplotlibcpp::figure_size(1000, 600);

            // Zero our time arrays
            double starttime2 = (nees_ori.timestamps.empty())? 0 : nees_ori.timestamps.at(0);
            double endtime2 = (nees_ori.timestamps.empty())? 0 : nees_ori.timestamps.at(nees_ori.timestamps.size()-1);
            for(size_t j=0; j<nees_ori.timestamps.size(); j++) {
                nees_ori.timestamps.at(j) -= starttime2;
                nees_pos.timestamps.at(j) -= starttime2;
            }

            // Update the title and axis labels
            matplotlibcpp::subplot(2,1,1);
            matplotlibcpp::title("Normalized Estimation Error Squared - "+path_algorithms.at(i).stem().string());
            matplotlibcpp::ylabel("NEES Orientation");
            matplotlibcpp::plot(nees_ori.timestamps, nees_ori.values);
            matplotlibcpp::xlim(0.0,endtime2-starttime2);
            matplotlibcpp::subplot(2,1,2);
            matplotlibcpp::ylabel("NEES Position");
            matplotlibcpp::xlabel("dataset time (s)");
            matplotlibcpp::plot(nees_pos.timestamps, nees_pos.values);
            matplotlibcpp::xlim(0.0,endtime2-starttime2);

            // Display to the user
            matplotlibcpp::tight_layout();
            matplotlibcpp::show(false);
        }

#endif


    }

    // Final line for our printed stats
    printf("============================================\n");


#ifdef HAVE_PYTHONLIBS

    // Wait till the user kills this node
    matplotlibcpp::show(true);

#endif

    // Done!
    return EXIT_SUCCESS;

}


