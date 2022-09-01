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
#include "utils/Colors.h"

#ifdef HAVE_PYTHONLIBS

// import the c++ wrapper for matplot lib
// https://github.com/lava/matplotlib-cpp
// sudo apt-get install python-matplotlib python-numpy python2.7-dev
#include "plot/matplotlibcpp.h"

// Will plot three error values in three sub-plots in our current figure
void plot_3errors(ov_eval::Statistics sx, ov_eval::Statistics sy, ov_eval::Statistics sz) {

    // Parameters that define the line styles
    std::map<std::string, std::string> params_value, params_bound;
    params_value.insert({"label","error"});
    params_value.insert({"linestyle","-"});
    params_value.insert({"color","blue"});
    params_bound.insert({"label","3 sigma bound"});
    params_bound.insert({"linestyle","--"});
    params_bound.insert({"color","red"});

    // Plot our error value
    matplotlibcpp::subplot(3,1,1);
    matplotlibcpp::plot(sx.timestamps, sx.values, params_value);
    if(!sx.values_bound.empty()) {
        matplotlibcpp::plot(sx.timestamps, sx.values_bound, params_bound);
        for(size_t i=0; i<sx.timestamps.size(); i++) {
            sx.values_bound.at(i) *= -1;
        }
        matplotlibcpp::plot(sx.timestamps, sx.values_bound, "r--");
    }

    // Plot our error value
    matplotlibcpp::subplot(3,1,2);
    matplotlibcpp::plot(sy.timestamps, sy.values, params_value);
    if(!sy.values_bound.empty()) {
        matplotlibcpp::plot(sy.timestamps, sy.values_bound, params_bound);
        for(size_t i=0; i<sy.timestamps.size(); i++) {
            sy.values_bound.at(i) *= -1;
        }
        matplotlibcpp::plot(sy.timestamps, sy.values_bound, "r--");
    }

    // Plot our error value
    matplotlibcpp::subplot(3,1,3);
    matplotlibcpp::plot(sz.timestamps, sz.values, params_value);
    if(!sz.values_bound.empty()) {
        matplotlibcpp::plot(sz.timestamps, sz.values_bound, params_bound);
        for(size_t i=0; i<sz.timestamps.size(); i++) {
            sz.values_bound.at(i) *= -1;
        }
        matplotlibcpp::plot(sz.timestamps, sz.values_bound, "r--");
    }

}

#endif

void save_nees(ov_eval::Statistics pos, ov_eval::Statistics ori)
{
  ofstream fo;
  string output_file = "/tmp/invariant_nees.txt";
  if(boost::filesystem::exists(output_file))
   {
         boost::filesystem::remove(output_file);
         cout<<"exist output_file, remove."<<endl;
   }

   fo.open(output_file,ofstream::out|ofstream::app);
   assert(pos.timestamps.size()==pos.values.size());
   assert(pos.values.size()==ori.values.size());

   for(int i=0;i<pos.timestamps.size();i++) 
   {
     fo<<to_string(pos.timestamps[i])<<" "<<to_string(pos.values[i])<<" "<<to_string(ori.timestamps[i])<<" "<<to_string(ori.values[i])<<endl;
   }
   fo.close();
}

void save_3error(ov_eval::Statistics px, ov_eval::Statistics py, ov_eval::Statistics pz, ov_eval::Statistics ox,ov_eval::Statistics oy, ov_eval::Statistics oz)
{
   ofstream fo;
    string output_file = "/tmp/invariant_3error.txt";
  if(boost::filesystem::exists(output_file))
   {
         boost::filesystem::remove(output_file);
         cout<<"exist output_file, remove."<<endl;
   }

   fo.open(output_file,ofstream::out|ofstream::app);
   assert(px.timestamps.size()==px.values_bound.size());
   assert(px.timestamps.size()== py.timestamps.size());
   assert(pz.timestamps.size()==px.timestamps.size());
   assert(px.timestamps.size()== ox.timestamps.size());
   assert(px.timestamps.size()== oy.timestamps.size());
   assert(px.timestamps.size()== oz.timestamps.size());


   for(int i=0;i<px.timestamps.size();i++) 
   {
     fo<<to_string(px.timestamps[i])<<" "<<to_string(px.values[i])<<" "<<to_string(px.values_bound[i])<<" "
      <<to_string(py.timestamps[i])<<" "<<to_string(py.values[i])<<" "<<to_string(py.values_bound[i])<<" "
      <<to_string(pz.timestamps[i])<<" "<<to_string(pz.values[i])<<" "<<to_string(pz.values_bound[i])<<" "
      <<to_string(ox.timestamps[i])<<" "<<to_string(ox.values[i])<<" "<<to_string(ox.values_bound[i])<<" "
      <<to_string(oy.timestamps[i])<<" "<<to_string(oy.values[i])<<" "<<to_string(oy.values_bound[i])<<" "
      <<to_string(oz.timestamps[i])<<" "<<to_string(oz.values[i])<<" "<<to_string(oz.values_bound[i])<<endl;
      
   }
   fo.close();
}


int main(int argc, char **argv) {

    // Ensure we have a path
    if(argc < 4) {
        printf(RED "ERROR: Please specify a align mode, groudtruth, and algorithm run file\n" RESET);
        printf(RED "ERROR: ./error_singlerun <align_mode> <file_gt.txt> <file_est.txt>\n" RESET);
        printf(RED "ERROR: rosrun ov_eval error_singlerun <align_mode> <file_gt.txt> <file_est.txt>\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // Load it!
    boost::filesystem::path path_gt(argv[2]);
    std::vector<double> times;
    std::vector<double> times_for_transform;
    std::vector<Eigen::Matrix<double,7,1>> poses;
    std::vector<Eigen::Matrix<double,7,1>> poses_for_transform;
    std::vector<Eigen::Matrix3d> cov_ori, cov_pos;
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

    // Create our trajectory object
    ov_eval::ResultTrajectory traj(argv[3], argv[2], argv[1]);
    boost::filesystem::path path_traj(argv[3]);


      ov_eval::Statistics nees_ori, nees_pos;
      ov_eval::Statistics nees_inv_ori, nees_inv_pos;
      ov_eval::Statistics posx, posy, posz;
      ov_eval::Statistics orix, oriy, oriz;
      ov_eval::Statistics roll, pitch, yaw;
    if(!is_trans)
    {
      
      traj.calculate_nees(nees_ori, nees_pos);
      traj.calculate_inv_nees(nees_inv_ori,nees_inv_pos);
      save_nees(nees_inv_pos,nees_inv_ori);

      // ov_eval::Statistics posx, posy, posz;
      // ov_eval::Statistics orix, oriy, oriz;
      // ov_eval::Statistics roll, pitch, yaw;
      traj.calculate_invariant_error(posx,posy,posz,orix,oriy,oriz,roll,pitch,yaw);
      save_3error(posx,posy,posz,orix,oriy,oriz);
    }
    else
    {
      std::string path_traj_gttxt = path_traj_gt.string();
      int size = path_traj.branch_path().string().size();
      std::string path_traj_est = path_traj.branch_path().string().substr(0,size-9)+"traj/"+path_traj.stem().string()+".txt";
      cout<<"traj path: "<<path_traj_est<<endl;
      ov_eval::ResultTrajectory traj_vins(path_traj_est, path_traj_gttxt, argv[1]);
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
      save_nees(nees_inv_pos,nees_inv_ori);


      traj.calculate_invariant_transform_error(est_poses_reduced,gt_poses_reduced,posx,posy,posz,orix,oriy,oriz);
      save_3error(posx,posy,posz,orix,oriy,oriz);

    }



    //===========================================================
    // Absolute trajectory error
    //===========================================================

    // Calculate
    ov_eval::Statistics error_ori, error_pos;
    traj.calculate_ate(error_ori, error_pos);

    // Print it
    printf("======================================\n");
    printf("Absolute Trajectory Error\n");
    printf("======================================\n");
    printf("rmse_ori = %.3f | rmse_pos = %.3f\n",error_ori.rmse,error_pos.rmse);
    printf("mean_ori = %.3f | mean_pos = %.3f\n",error_ori.mean,error_pos.mean);
    printf("min_ori  = %.3f | min_pos  = %.3f\n",error_ori.min,error_pos.min);
    printf("max_ori  = %.3f | max_pos  = %.3f\n",error_ori.max,error_pos.max);
    printf("std_ori  = %.3f | std_pos  = %.3f\n",error_ori.std,error_pos.std);

    //===========================================================
    // Relative pose error
    //===========================================================

    // Calculate
    std::vector<double> segments = {8.0, 16.0, 24.0, 32.0, 40.0};
    std::map<double,std::pair<ov_eval::Statistics,ov_eval::Statistics>> error_rpe;
    traj.calculate_rpe(segments, error_rpe);

    // Print it
    printf("======================================\n");
    printf("Relative Pose Error\n");
    printf("======================================\n");
    for(const auto &seg : error_rpe) {
        printf("seg %d - median_ori = %.3f | median_pos = %.3f (%d samples)\n",(int)seg.first,seg.second.first.median,seg.second.second.median,(int)seg.second.second.values.size());
        //printf("seg %d - std_ori  = %.3f | std_pos  = %.3f\n",(int)seg.first,seg.second.first.std,seg.second.second.std);
    }


#ifdef HAVE_PYTHONLIBS

    // Parameters
    std::map<std::string, std::string> params_rpe;
    params_rpe.insert({"notch","true"});
    params_rpe.insert({"sym",""});

    // Plot this figure
    matplotlibcpp::figure_size(800, 600);

    // Plot each RPE next to each other
    double ct = 1;
    double width = 0.50;
    std::vector<double> xticks;
    std::vector<std::string> labels;
    for(const auto &seg : error_rpe) {
        xticks.push_back(ct);
        labels.push_back(std::to_string((int)seg.first));
        matplotlibcpp::boxplot(seg.second.first.values, ct++, width, "blue", "-", params_rpe);
    }

    // Display to the user
    matplotlibcpp::xlim(0.5,ct-0.5);
    matplotlibcpp::xticks(xticks,labels);
    matplotlibcpp::title("Relative Orientation Error");
    matplotlibcpp::ylabel("orientation error (deg)");
    matplotlibcpp::xlabel("sub-segment lengths (m)");
    matplotlibcpp::show(false);

    // Plot this figure
    matplotlibcpp::figure_size(800, 600);

    // Plot each RPE next to each other
    ct = 1;
    for(const auto &seg : error_rpe) {
        matplotlibcpp::boxplot(seg.second.second.values, ct++, width, "blue", "-", params_rpe);
    }

    // Display to the user
    matplotlibcpp::xlim(0.5,ct-0.5);
    matplotlibcpp::xticks(xticks,labels);
    matplotlibcpp::title("Relative Position Error");
    matplotlibcpp::ylabel("translation error (m)");
    matplotlibcpp::xlabel("sub-segment lengths (m)");
    matplotlibcpp::show(false);

#endif

    //===========================================================
    // Normalized Estimation Error Squared
    //===========================================================

    // Calculate
    // ov_eval::Statistics nees_ori, nees_pos;
    // traj.calculate_inv_nees(nees_ori, nees_pos);
    // save_nees(nees_pos,nees_ori);
    // Print it
    printf("======================================\n");
    printf("Normalized Estimation Error Squared\n");
    printf("======================================\n");
    printf("mean_ori = %.3f | mean_pos = %.3f\n",nees_inv_ori.mean/3.0,nees_inv_pos.mean/3.0);
    printf("min_ori  = %.3f | min_pos  = %.3f\n",nees_inv_ori.min/3.0,nees_inv_pos.min/3.0);
    printf("max_ori  = %.3f | max_pos  = %.3f\n",nees_inv_ori.max/3.0,nees_inv_pos.max/3.0);
    printf("std_ori  = %.3f | std_pos  = %.3f\n",nees_inv_ori.std/3.0,nees_inv_pos.std/3.0);
    printf("======================================\n");



#ifdef HAVE_PYTHONLIBS

    if(!nees_inv_ori.values.empty() && !nees_inv_pos.values.empty()) {
        // Zero our time arrays
        double starttime1 = (nees_inv_ori.timestamps.empty())? 0 : nees_inv_ori.timestamps.at(0);
        double endtime1 = (nees_inv_ori.timestamps.empty())? 0 : nees_inv_ori.timestamps.at(nees_inv_ori.timestamps.size()-1);
        for(size_t i=0; i<nees_inv_ori.timestamps.size(); i++) {
            nees_inv_ori.timestamps.at(i) -= starttime1;
            nees_inv_pos.timestamps.at(i) -= starttime1;
        }

        // Plot this figure
        matplotlibcpp::figure_size(1000, 600);

        // Parameters that define the line styles
        std::map<std::string, std::string> params_neesp, params_neeso;
        params_neesp.insert({"label","nees position"});
        params_neesp.insert({"linestyle","-"});
        params_neesp.insert({"color","blue"});
        params_neeso.insert({"label","nees orientation"});
        params_neeso.insert({"linestyle","-"});
        params_neeso.insert({"color","blue"});


        // Update the title and axis labels
        matplotlibcpp::subplot(2,1,1);
        matplotlibcpp::title("Normalized Estimation Error Squared");
        matplotlibcpp::ylabel("NEES Orientation");
        matplotlibcpp::plot(nees_inv_ori.timestamps, nees_inv_ori.values, params_neeso);
        matplotlibcpp::xlim(0.0,endtime1-starttime1);
        matplotlibcpp::subplot(2,1,2);
        matplotlibcpp::ylabel("NEES Position");
        matplotlibcpp::xlabel("dataset time (s)");
        matplotlibcpp::plot(nees_inv_pos.timestamps, nees_inv_pos.values, params_neesp);
        matplotlibcpp::xlim(0.0,endtime1-starttime1);

        // Display to the user
        matplotlibcpp::tight_layout();
        matplotlibcpp::show(false);
    }

#endif


    //===========================================================
    // Plot the error if we have matplotlib to plot!
    //===========================================================

    // Calculate
    // ov_eval::Statistics posx, posy, posz;
    // ov_eval::Statistics orix, oriy, oriz;
    // ov_eval::Statistics roll, pitch, yaw;
    // traj.calculate_invariant_error(posx,posy,posz,orix,oriy,oriz,roll,pitch,yaw);
    // save_3error(posx,posy,posz,orix,oriy,oriz);

    // Zero our time arrays
    double starttime2 = (posx.timestamps.empty())? 0 : posx.timestamps.at(0);
    double endtime2 = (posx.timestamps.empty())? 0 : posx.timestamps.at(posx.timestamps.size()-1);
    for(size_t i=0; i<posx.timestamps.size(); i++) {
        posx.timestamps.at(i) -= starttime2;
        posy.timestamps.at(i) -= starttime2;
        posz.timestamps.at(i) -= starttime2;
        orix.timestamps.at(i) -= starttime2;
        oriy.timestamps.at(i) -= starttime2;
        oriz.timestamps.at(i) -= starttime2;
        // roll.timestamps.at(i) -= starttime2;
        // pitch.timestamps.at(i) -= starttime2;
        // yaw.timestamps.at(i) -= starttime2;
    }
      cout<<"posx size: "<<posx.timestamps.size()<<" "<<posx.values.size()<<std::endl;
    cout<<"start time: "<<starttime2<<" end time: "<<endtime2<<endl;




#ifdef HAVE_PYTHONLIBS

    //=====================================================
    // Plot this figure
    matplotlibcpp::figure_size(1000, 600);
    plot_3errors(posx,posy,posz);

    // Update the title and axis labels
    matplotlibcpp::subplot(3,1,1);
    matplotlibcpp::title("IMU Position Error");
    matplotlibcpp::ylabel("x-error (m)");
    matplotlibcpp::xlim(0.0,endtime2-starttime2);
    matplotlibcpp::subplot(3,1,2);
    matplotlibcpp::ylabel("y-error (m)");
    matplotlibcpp::xlim(0.0,endtime2-starttime2);
    matplotlibcpp::subplot(3,1,3);
    matplotlibcpp::ylabel("z-error (m)");
    matplotlibcpp::xlabel("dataset time (s)");
    matplotlibcpp::xlim(0.0,endtime2-starttime2);

    // Display to the user
    matplotlibcpp::tight_layout();
    matplotlibcpp::show(false);

    //=====================================================
    // Plot this figure
    matplotlibcpp::figure_size(1000, 600);
    plot_3errors(orix,oriy,oriz);

    // Update the title and axis labels
    matplotlibcpp::subplot(3,1,1);
    matplotlibcpp::title("IMU Orientation Error");
    matplotlibcpp::ylabel("x-error (deg)");
    matplotlibcpp::xlim(0.0,endtime2-starttime2);
    matplotlibcpp::subplot(3,1,2);
    matplotlibcpp::ylabel("y-error (deg)");
    matplotlibcpp::xlim(0.0,endtime2-starttime2);
    matplotlibcpp::subplot(3,1,3);
    matplotlibcpp::ylabel("z-error (deg)");
    matplotlibcpp::xlabel("dataset time (s)");
    matplotlibcpp::xlim(0.0,endtime2-starttime2);

    // Display to the user
    matplotlibcpp::tight_layout();
    matplotlibcpp::show(true);

    //=====================================================
    // // Plot this figure
    // matplotlibcpp::figure_size(1000, 600);
    // plot_3errors(roll,pitch,yaw);

    // // Update the title and axis labels
    // matplotlibcpp::subplot(3,1,1);
    // matplotlibcpp::title("Global Orientation RPY Error");
    // matplotlibcpp::ylabel("roll error (deg)");
    // matplotlibcpp::xlim(0.0,endtime2-starttime2);
    // matplotlibcpp::subplot(3,1,2);
    // matplotlibcpp::ylabel("pitch error (deg)");
    // matplotlibcpp::xlim(0.0,endtime2-starttime2);
    // matplotlibcpp::subplot(3,1,3);
    // matplotlibcpp::ylabel("yaw error (deg)");
    // matplotlibcpp::xlabel("dataset time (s)");
    // matplotlibcpp::xlim(0.0,endtime2-starttime2);

    // // Display to the user
    // matplotlibcpp::tight_layout();
    // matplotlibcpp::show(true);


#endif


    // Done!
    return EXIT_SUCCESS;

}


