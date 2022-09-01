#include <string>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>

#include "calc/ResultTrajectory.h"
#include "utils/Colors.h"
#include "utils/Math.h"

#ifdef HAVE_PYTHONLIBS

// import the c++ wrapper for matplot lib
// https://github.com/lava/matplotlib-cpp
// sudo apt-get install python-matplotlib python-numpy python2.7-dev
#include "plot/matplotlibcpp.h"

using namespace ov_eval;

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


int main(int argc, char **argv) {

    // Ensure we have a path
//     if(argc < 4) {
//         printf(RED "ERROR: Please specify a align mode, groudtruth, and algorithm run file\n" RESET);
//         printf(RED "ERROR: ./error_singlerun <align_mode> <file_gt.txt> <file_est.txt>\n" RESET);
//         printf(RED "ERROR: rosrun ov_eval error_singlerun <align_mode> <file_gt.txt> <file_est.txt>\n" RESET);
//         std::exit(EXIT_FAILURE);
//     }
    if(argc>2)
    {
          printf(RED "ERROR: ./plot_3error <file_est.txt>\n" RESET);
    }

    // Load it!
    boost::filesystem::path path_gt(argv[1]);
    std::vector<double> times;
    std::vector<Eigen::Matrix<double,7,1>> poses;
    std::vector<Eigen::Matrix3d> cov_ori, cov_pos;
    ov_eval::Loader::load_data(argv[1], times, poses, cov_ori, cov_pos);
    // Print its length and stats
//     double length = ov_eval::Loader::get_total_length(poses);
//     printf("[COMP]: %d poses in %s => length of %.2f meters\n",(int)times.size(),path_gt.stem().string().c_str(),length);
     
    ov_eval::Statistics posx, posy, posz;
    ov_eval::Statistics roll, pitch, yaw;

    double mean_x,mean_y,mean_z;
    double mean_roll,mean_pitch,mean_yaw;

    for(int i=0;i<poses.size();i++)
    {
          posx.values.push_back(poses[i](0,0));
          posx.timestamps.push_back(times.at(i));
          posy.values.push_back(poses[i](1,0));
          posy.timestamps.push_back(times.at(i));
          posz.values.push_back(poses[i](2,0));
          posz.timestamps.push_back(times.at(i));
          Eigen::Vector3d ypr_est_ItoG = Math::rot2rpy(Math::quat_2_Rot(poses.at(i).block(3,0,4,1)).transpose());
          for(size_t idx=0; idx<3; idx++) {
            while(ypr_est_ItoG(idx)<-M_PI) {
                ypr_est_ItoG(idx) += 2*M_PI;
            }
            while(ypr_est_ItoG(idx)>M_PI) {
                ypr_est_ItoG(idx) -= 2*M_PI;
            }
            roll.timestamps.push_back(times.at(i));
            roll.values.push_back(180.0/M_PI*ypr_est_ItoG(0));
            pitch.timestamps.push_back(times.at(i));
            pitch.values.push_back(180.0/M_PI*ypr_est_ItoG(1));
            yaw.timestamps.push_back(times.at(i));
            yaw.values.push_back(180.0/M_PI*ypr_est_ItoG(2));
        }
    }
    posx.calculate();
    posy.calculate();
    posz.calculate();
    roll.calculate();
    pitch.calculate();
    yaw.calculate();

    for(int i=0;i<poses.size();i++)
    {
          posx.values_bound.push_back(abs(posx.values.at(i)-posx.mean));
          posy.values_bound.push_back(abs(posy.values.at(i)-posy.mean));
          posz.values_bound.push_back(abs(posz.values.at(i)-posz.mean));
          roll.values_bound.push_back(abs(roll.values.at(i)-roll.mean));
          pitch.values_bound.push_back(abs(pitch.values.at(i)-pitch.mean));
          yaw.values_bound.push_back(abs(yaw.values.at(i)-yaw.mean));

    }






    // Zero our time arrays
    double starttime2 = (posx.timestamps.empty())? 0 : posx.timestamps.at(0);
    double endtime2 = (posx.timestamps.empty())? 0 : posx.timestamps.at(posx.timestamps.size()-1);
    for(size_t i=0; i<posx.timestamps.size(); i++) {
        posx.timestamps.at(i) -= starttime2;
        posy.timestamps.at(i) -= starttime2;
        posz.timestamps.at(i) -= starttime2;
        roll.timestamps.at(i) -= starttime2;
        pitch.timestamps.at(i) -= starttime2;
        yaw.timestamps.at(i) -= starttime2;
    }
    cout<<"posx  value size: "<<posx.values.size()<<" bound size: "<<posx.values_bound.size()<<" time size: "<<posx.timestamps.size()<<endl;
    cout<<"posy  value size: "<<posy.values.size()<<" bound size: "<<posy.values_bound.size()<<" time size: "<<posy.timestamps.size()<<endl;
    cout<<"posz  value size: "<<posz.values.size()<<" bound size: "<<posz.values_bound.size()<<" time size: "<<posz.timestamps.size()<<endl;
    cout<<"roll  value size: "<<roll.values.size()<<" bound size: "<<roll.values_bound.size()<<" time size: "<<roll.timestamps.size()<<endl;
    cout<<"pitch  value size: "<<pitch.values.size()<<" bound size: "<<pitch.values_bound.size()<<" time size: "<<pitch.timestamps.size()<<endl;
    cout<<"yaw value size: "<<yaw.values.size()<<" bound size: "<<yaw.values_bound.size()<<" time size: "<<yaw.timestamps.size()<<endl;
    cout<<"finish compute"<<endl;



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
    matplotlibcpp::show(true);

      cout<<"finish plot pos"<<endl;

//     //=====================================================
//     // Plot this figure
//     matplotlibcpp::figure_size(1000, 600);
//     plot_3errors(orix,oriy,oriz);

//     // Update the title and axis labels
//     matplotlibcpp::subplot(3,1,1);
//     matplotlibcpp::title("IMU Orientation Error");
//     matplotlibcpp::ylabel("x-error (deg)");
//     matplotlibcpp::xlim(0.0,endtime2-starttime2);
//     matplotlibcpp::subplot(3,1,2);
//     matplotlibcpp::ylabel("y-error (deg)");
//     matplotlibcpp::xlim(0.0,endtime2-starttime2);
//     matplotlibcpp::subplot(3,1,3);
//     matplotlibcpp::ylabel("z-error (deg)");
//     matplotlibcpp::xlabel("dataset time (s)");
//     matplotlibcpp::xlim(0.0,endtime2-starttime2);

//     // Display to the user
//     matplotlibcpp::tight_layout();
//     matplotlibcpp::show(false);

    //=====================================================
    // Plot this figure
//     matplotlibcpp::figure_size(1000, 600);
//     plot_3errors(roll,pitch,yaw);

//     // Update the title and axis labels
//     matplotlibcpp::subplot(3,1,1);
//     matplotlibcpp::title("Global Orientation RPY Error");
//     matplotlibcpp::ylabel("roll error (deg)");
//     matplotlibcpp::xlim(0.0,endtime2-starttime2);
//     matplotlibcpp::subplot(3,1,2);
//     matplotlibcpp::ylabel("pitch error (deg)");
//     matplotlibcpp::xlim(0.0,endtime2-starttime2);
//     matplotlibcpp::subplot(3,1,3);
//     matplotlibcpp::ylabel("yaw error (deg)");
//     matplotlibcpp::xlabel("dataset time (s)");
//     matplotlibcpp::xlim(0.0,endtime2-starttime2);

//     // Display to the user
//     matplotlibcpp::tight_layout();
//     matplotlibcpp::show(true);

      cout<<"finish plot RPY"<<endl;


#endif


    // Done!
    return EXIT_SUCCESS;

}
