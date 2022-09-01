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
#include "Simulator.h"


using namespace ov_rimsckf;




Simulator::Simulator(VioManagerOptions& params_) {


    //===============================================================
    //===============================================================

    // Nice startup message
    printf("=======================================\n");
    printf("VISUAL-INERTIAL SIMULATOR STARTING\n");
    printf("=======================================\n");

    // Store a copy of our params
    this->params = params_;
    params.print_estimator();
    params.print_noise();
    params.print_state();
    params.print_trackers();
    params.print_simulation();

    // Check that the max cameras matches the size of our cam matrices
    if(params.state_options.num_cameras != (int)params.camera_fisheye.size()) {
        printf(RED "[SIM]: camera calib size does not match max cameras...\n" RESET);
        printf(RED "[SIM]: got %d but expected %d max cameras\n" RESET, (int)params.camera_fisheye.size(), params.state_options.num_cameras);
        std::exit(EXIT_FAILURE);
    }

    // Load the groundtruth trajectory and its spline
    load_data(params.sim_traj_path);
    spline.feed_trajectory(traj_data);

    load_data_turb(params.sim_traj_path_turb);
    spline_turb.feed_trajectory(traj_data_turb);

    // Set all our timestamps as starting from the minimum spline time
    timestamp = spline.get_start_time();
    timestamp_last_imu = spline.get_start_time();
    timestamp_last_cam = spline.get_start_time();

    // Get the pose at the current timestep
    Eigen::Matrix3d R_GtoI_init;
    Eigen::Vector3d p_IinG_init;
    bool success_pose_init = spline.get_pose(timestamp, R_GtoI_init, p_IinG_init);
    if(!success_pose_init) {
        printf(RED "[SIM]: unable to find the first pose in the spline...\n" RESET);
        std::exit(EXIT_FAILURE);
    }

    // Find the timestamp that we move enough to be considered "moved"
    double distance = 0.0;
    double distancethreshold = params.sim_distance_threshold;
    while(true) {

        // Get the pose at the current timestep
        Eigen::Matrix3d R_GtoI;
        Eigen::Vector3d p_IinG;
        bool success_pose = spline.get_pose(timestamp, R_GtoI, p_IinG);

        // Check if it fails
        if(!success_pose) {
            printf(RED "[SIM]: unable to find jolt in the groundtruth data to initialize at\n" RESET);
            std::exit(EXIT_FAILURE);
        }

        // Append to our scalar distance
        distance += (p_IinG-p_IinG_init).norm();
        p_IinG_init = p_IinG;

        // Now check if we have an acceleration, else move forward in time
        if(distance > distancethreshold) {
            break;
        } else {
            timestamp += 1.0/params.sim_freq_cam;
            timestamp_last_imu += 1.0/params.sim_freq_cam;
            timestamp_last_cam += 1.0/params.sim_freq_cam;
        }

    }
    printf("[SIM]: moved %.3f seconds into the dataset where it starts moving\n",timestamp-spline.get_start_time());

    // Append the current true bias to our history
    hist_true_bias_time.push_back(timestamp_last_imu-1.0/params.sim_freq_imu);
    hist_true_bias_accel.push_back(true_bias_accel);
    hist_true_bias_gyro.push_back(true_bias_gyro);
    hist_true_bias_time.push_back(timestamp_last_imu);
    hist_true_bias_accel.push_back(true_bias_accel);
    hist_true_bias_gyro.push_back(true_bias_gyro);

    // Our simulation is running
    is_running = true;

    //===============================================================
    //===============================================================

    // Load the seeds for the random number generators
    gen_state_init = std::mt19937(params.sim_seed_state_init);
    gen_state_init.seed(params.sim_seed_state_init);
    gen_state_perturb = std::mt19937(params.sim_seed_preturb);
    gen_state_perturb.seed(params.sim_seed_preturb);
    gen_meas_imu = std::mt19937(params.sim_seed_measurements);
    gen_meas_imu.seed(params.sim_seed_measurements);

    // Create generator for our camera
    for(int i=0; i<params.state_options.num_cameras; i++) {
        gen_meas_cams.push_back(std::mt19937(params.sim_seed_measurements));
        gen_meas_cams.at(i).seed(params.sim_seed_measurements);
    }


    //===============================================================
    //===============================================================

    // One std generator
    std::normal_distribution<double> w(0,1);

    // Perturb all calibration if we should
    if(params.sim_do_perturbation) {

        // cam imu offset
        params_.calib_camimu_dt += 0.01*w(gen_state_perturb);
        printf("[SIM]: the perturbed calib_camimu_dt %lf\n",params_.calib_camimu_dt);

        // camera intrinsics and extrinsics
        for(int i=0; i<params_.state_options.num_cameras; i++) {

            // Camera intrinsic properties (k1, k2, p1, p2)
            // for(int r=0; r<4; r++) {
            //     params_.camera_intrinsics.at(i)(r) += 1.0*w(gen_state_perturb);
            // }

            // // Camera intrinsic properties (r1, r2, r3, r4)
            // for(int r=4; r<8; r++) {
            //     params_.camera_intrinsics.at(i)(r) += 0.005*w(gen_state_perturb);
            // }

            // Our camera extrinsics transform (position)
          //   for(int r=4; r<7; r++) {
          //       params_.camera_extrinsics.at(i)(r) += 0.01*w(gen_state_perturb);
          //   }

          //   // Our camera extrinsics transform (orientation)
          //   Eigen::Vector3d w_vec;
          //   w_vec << 0.001*w(gen_state_perturb), 0.001*w(gen_state_perturb), 0.001*w(gen_state_perturb);
          //  params_.camera_extrinsics.at(i).block(0,0,4,1) =
          //          rot_2_quat(exp_so3(w_vec)*quat_2_Rot(params_.camera_extrinsics.at(i).block(0,0,4,1)));

        }
        
    }

    //===============================================================
    //===============================================================


    // We will create synthetic camera frames and ensure that each has enough features
    //double dt = 0.25/freq_cam;
    double dt = 0.25;
    size_t mapsize = featmap.size();
    printf("[SIM]: Generating map features at %d rate\n",(int)(1.0/dt));

    // Loop through each camera
    // NOTE: we loop through cameras here so that the feature map for camera 1 will always be the same
    // NOTE: thus when we add more cameras the first camera should get the same measurements
    for(int i=0; i<params.state_options.num_cameras; i++) {

        // Reset the start time
        double time_synth = spline.get_start_time();

        // Loop through each pose and generate our feature map in them!!!!
        while(true) {

            // Get the pose at the current timestep
            Eigen::Matrix3d R_GtoI;
            Eigen::Vector3d p_IinG;
            bool success_pose = spline.get_pose(time_synth, R_GtoI, p_IinG);

            // We have finished generating features
            if(!success_pose)
                break;

            // Get the uv features for this frame
            //project all points in featmap into the current frame, to see how many features could be observed by cuurent frame.
            std::vector<std::pair<size_t,Eigen::VectorXf>> uvs = project_pointcloud(R_GtoI, p_IinG, i, featmap);
            // If we do not have enough, generate more
            if((int)uvs.size() < params.num_pts) {
                generate_points(R_GtoI, p_IinG, i, featmap, params.num_pts-(int)uvs.size());
            }

            // Move forward in time
            time_synth += dt;
        }

        // Debug print
        printf("[SIM]: Generated %d map features in total over %d frames (camera %d)\n",(int)(featmap.size()-mapsize),(int)((time_synth-spline.get_start_time())/dt),i);
        mapsize = featmap.size();

    }
    
    string sim_points= "/tmp/sim_points.txt";
    of_point.open(sim_points.data());
    assert(of_point.is_open());
    string sim_gyro_bias="/tmp/sim_gbias_true.txt";
    of_gyro_bias.open(sim_gyro_bias.data());
    assert(of_gyro_bias.is_open());
    string sim_acc_bias="/tmp/sim_abias_true.txt";
    of_acc_bias.open(sim_acc_bias.data());
    assert(of_acc_bias.is_open());


    // Nice sleep so the user can look at the printout
    sleep(3);

}





bool Simulator::get_state(double desired_time, Eigen::Matrix<double,17,1> &imustate) {

    // Set to default state
    imustate.setZero();
    imustate(4) = 1;

    // Current state values
    Eigen::Matrix3d R_GtoI;
    Eigen::Vector3d p_IinG, w_IinI, v_IinG;

    // Get the pose, velocity, and acceleration
    bool success_vel = spline.get_velocity(desired_time, R_GtoI, p_IinG, w_IinI, v_IinG);

    // Find the bounding bias values
    bool success_bias = false;
    size_t id_loc = 0;
    for(size_t i=0; i<hist_true_bias_time.size()-1; i++) {
        if(hist_true_bias_time.at(i) < desired_time && hist_true_bias_time.at(i+1) >= desired_time) {
            id_loc = i;
            success_bias = true;
            break;
        }
    }

    // If failed, then that means we don't have any more spline or bias
    if(!success_vel || !success_bias) {
        return false;
    }

    // Interpolate our biases (they will be at every IMU timestep)
    double lambda = (desired_time-hist_true_bias_time.at(id_loc))/(hist_true_bias_time.at(id_loc+1)-hist_true_bias_time.at(id_loc));
    Eigen::Vector3d true_bg_interp = (1-lambda)*hist_true_bias_gyro.at(id_loc) + lambda*hist_true_bias_gyro.at(id_loc+1);
    Eigen::Vector3d true_ba_interp = (1-lambda)*hist_true_bias_accel.at(id_loc) + lambda*hist_true_bias_accel.at(id_loc+1);

    // Finally lets create the current state
    imustate(0,0) = desired_time;
    imustate.block(1,0,4,1) = rot_2_quat(R_GtoI);
    imustate.block(5,0,3,1) = p_IinG;
    imustate.block(8,0,3,1) = v_IinG;
    imustate.block(11,0,3,1) = true_bg_interp;
    imustate.block(14,0,3,1) = true_ba_interp;
    return true;

}




bool Simulator::get_next_imu(double &time_imu, Eigen::Vector3d &wm, Eigen::Vector3d &am) {

    // Return if the camera measurement should go before us
    if(timestamp_last_cam+1.0/params.sim_freq_cam < timestamp_last_imu+1.0/params.sim_freq_imu)
        return false;

    // Else lets do a new measurement!!!
    timestamp_last_imu += 1.0/params.sim_freq_imu;
    timestamp = timestamp_last_imu;
    time_imu = timestamp_last_imu;

    // Current state values
    Eigen::Matrix3d R_GtoI;
    Eigen::Vector3d p_IinG, w_IinI, v_IinG, alpha_IinI, a_IinG;

    // Get the pose, velocity, and acceleration
    // NOTE: we get the acceleration between our two IMU
    // NOTE: this is because we are using a constant measurement model for integration
    //bool success_accel = spline.get_acceleration(timestamp+0.5/freq_imu, R_GtoI, p_IinG, w_IinI, v_IinG, alpha_IinI, a_IinG);
    bool success_accel = spline.get_acceleration(timestamp, R_GtoI, p_IinG, w_IinI, v_IinG, alpha_IinI, a_IinG);

    // If failed, then that means we don't have any more spline
    // Thus we should stop the simulation
    if(!success_accel) {
        is_running = false;
        return false;
    }

    // Transform omega and linear acceleration into imu frame
    Eigen::Vector3d omega_inI = w_IinI;
    Eigen::Vector3d accel_inI = R_GtoI*(a_IinG+params.gravity);

    // Now add noise to these measurements
    double dt = 1.0/params.sim_freq_imu;
    std::normal_distribution<double> w(0,1);
    wm(0) = omega_inI(0) + true_bias_gyro(0) + params.imu_noises.sigma_w/params.imu_noises.noise_multipler/std::sqrt(dt)*w(gen_meas_imu);
    wm(1) = omega_inI(1) + true_bias_gyro(1) + params.imu_noises.sigma_w/params.imu_noises.noise_multipler/std::sqrt(dt)*w(gen_meas_imu);
    wm(2) = omega_inI(2) + true_bias_gyro(2) + params.imu_noises.sigma_w/params.imu_noises.noise_multipler/std::sqrt(dt)*w(gen_meas_imu);
    am(0) = accel_inI(0) + true_bias_accel(0) + params.imu_noises.sigma_a/params.imu_noises.noise_multipler/std::sqrt(dt)*w(gen_meas_imu);
    am(1) = accel_inI(1) + true_bias_accel(1) + params.imu_noises.sigma_a/params.imu_noises.noise_multipler/std::sqrt(dt)*w(gen_meas_imu);
    am(2) = accel_inI(2) + true_bias_accel(2) + params.imu_noises.sigma_a/params.imu_noises.noise_multipler/std::sqrt(dt)*w(gen_meas_imu);

    // Move the biases forward in time
    true_bias_gyro(0) += params.imu_noises.sigma_wb/params.imu_noises.noise_multipler*std::sqrt(dt)*w(gen_meas_imu);
    true_bias_gyro(1) += params.imu_noises.sigma_wb/params.imu_noises.noise_multipler*std::sqrt(dt)*w(gen_meas_imu);
    true_bias_gyro(2) += params.imu_noises.sigma_wb/params.imu_noises.noise_multipler*std::sqrt(dt)*w(gen_meas_imu);
    true_bias_accel(0) += params.imu_noises.sigma_ab/params.imu_noises.noise_multipler*std::sqrt(dt)*w(gen_meas_imu);
    true_bias_accel(1) += params.imu_noises.sigma_ab/params.imu_noises.noise_multipler*std::sqrt(dt)*w(gen_meas_imu);
    true_bias_accel(2) += params.imu_noises.sigma_ab/params.imu_noises.noise_multipler*std::sqrt(dt)*w(gen_meas_imu);

    // Append the current true bias to our history
    hist_true_bias_time.push_back(timestamp_last_imu);
    hist_true_bias_gyro.push_back(true_bias_gyro);
    hist_true_bias_accel.push_back(true_bias_accel);

    of_gyro_bias<<to_string(timestamp_last_imu)<<" "<<to_string(true_bias_gyro(0))
                                    <<" "<<to_string(true_bias_gyro(1))
                                    <<" "<<to_string(true_bias_gyro(2))
                                    <<" "<<0
                                    <<" "<<0
                                    <<" "<<0
                                    <<" "<<1<<endl;
    of_acc_bias<<to_string(timestamp_last_imu)<<" "<<to_string(true_bias_accel(0))
                                    <<" "<<to_string(true_bias_accel(1))
                                    <<" "<<to_string(true_bias_accel(2))
                                    <<" "<<0
                                    <<" "<<0
                                    <<" "<<0
                                    <<" "<<1<<endl;

    // Return success
    return true;

}



bool Simulator::get_next_cam(double &time_cam, std::vector<int> &camids, std::vector<std::vector<std::pair<size_t,Eigen::VectorXf>>> &feats) {

    // Return if the imu measurement should go before us
    if(timestamp_last_imu+1.0/params.sim_freq_imu < timestamp_last_cam+1.0/params.sim_freq_cam)
        return false;

    // Else lets do a new measurement!!!
    timestamp_last_cam += 1.0/params.sim_freq_cam;
    timestamp = timestamp_last_cam;
    time_cam = timestamp_last_cam-params.calib_camimu_dt;

    // Get the pose at the current timestep
    Eigen::Matrix3d R_GtoI;
    Eigen::Vector3d p_IinG;
    bool success_pose = spline.get_pose(timestamp, R_GtoI, p_IinG);

    // We have finished generating measurements
    if(!success_pose) {
        is_running = false;
        return false;
    }

    // Loop through each camera
    for(int i=0; i<params.state_options.num_cameras; i++) {

        // Get the uv features for this frame
        std::vector<std::pair<size_t,Eigen::VectorXf>> uvs = project_pointcloud(R_GtoI, p_IinG, i, featmap);

        // If we do not have enough, generate more
        if((int)uvs.size() < params.num_pts) {
            printf(YELLOW "[SIM]: cam %d was unable to generate enough features (%d < %d projections)\n" RESET,(int)i,(int)uvs.size(),params.num_pts);
        }

        // If greater than only select the first set
        if((int)uvs.size() > params.num_pts) {
            uvs.erase(uvs.begin()+params.num_pts, uvs.end());
        }

        // Append the map size so all cameras have unique features in them (but the same map)
        // Only do this if we are not enforcing stereo constraints between all our cameras
        for (size_t f=0; f<uvs.size() && !params.use_stereo; f++) {
            uvs.at(f).first += i*featmap.size();
        }

        // Loop through and add noise to each uv measurement
        std::normal_distribution<double> w(0,1);
        for(size_t j=0; j<uvs.size(); j++) {
            uvs.at(j).second(0) += params.msckf_options.sigma_pix/params.msckf_options.meas_noise_multipler*w(gen_meas_cams.at(i));
            uvs.at(j).second(1) += params.msckf_options.sigma_pix/params.msckf_options.meas_noise_multipler*w(gen_meas_cams.at(i));
        }

        // Push back for this camera
        feats.push_back(uvs);
        camids.push_back(i);

    }


    // Return success
    return true;

}




void Simulator::load_data(std::string path_traj) {

    // Try to open our groundtruth file
    std::ifstream file;
    file.open(path_traj);
    if (!file) {
        printf(RED "ERROR: Unable to open simulation trajectory file...\n" RESET);
        printf(RED "ERROR: %s\n" RESET,path_traj.c_str());
        std::exit(EXIT_FAILURE);
    }

    // Debug print
    std::string base_filename = path_traj.substr(path_traj.find_last_of("/\\") + 1);
    printf("[SIM]: loaded trajectory %s\n",base_filename.c_str());

    // Loop through each line of this file
    std::string current_line;
    while(std::getline(file, current_line)) {

        // Skip if we start with a comment
        if(!current_line.find("#"))
            continue;

        // Loop variables
        int i = 0;
        std::istringstream s(current_line);
        std::string field;
        Eigen::Matrix<double,8,1> data;

        // Loop through this line (timestamp(s) tx ty tz qx qy qz qw)
        while(std::getline(s,field,' ')) {
            // Skip if empty
            if(field.empty() || i >= data.rows())
                continue;
            // save the data to our vector
            data(i) = std::atof(field.c_str());
            i++;
        }

        // Only a valid line if we have all the parameters
        if(i > 7) {
            traj_data.push_back(data);
            //std::cout << std::setprecision(15) << data.transpose() << std::endl;
        }

    }

    // Finally close the file
    file.close();

    // Error if we don't have any data
    if (traj_data.empty()) {
        printf(RED "ERROR: Could not parse any data from the file!!\n" RESET);
        printf(RED "ERROR: %s\n" RESET,path_traj.c_str());
        std::exit(EXIT_FAILURE);
    }

}

void Simulator::load_data_turb(std::string path_traj) {

    // Try to open our groundtruth file
    std::ifstream file;
    file.open(path_traj);
    if (!file) {
        printf(RED "ERROR: Unable to open simulation trajectory file...\n" RESET);
        printf(RED "ERROR: %s\n" RESET,path_traj.c_str());
        std::exit(EXIT_FAILURE);
    }

    // Debug print
    std::string base_filename = path_traj.substr(path_traj.find_last_of("/\\") + 1);
    printf("[SIM]: loaded trajectory %s\n",base_filename.c_str());

    // Loop through each line of this file
    std::string current_line;
    while(std::getline(file, current_line)) {

        // Skip if we start with a comment
        if(!current_line.find("#"))
            continue;

        // Loop variables
        int i = 0;
        std::istringstream s(current_line);
        std::string field;
        Eigen::Matrix<double,8,1> data;

        // Loop through this line (timestamp(s) tx ty tz qx qy qz qw)
        while(std::getline(s,field,' ')) {
            // Skip if empty
            if(field.empty() || i >= data.rows())
                continue;
            // save the data to our vector
            data(i) = std::atof(field.c_str());
            i++;
        }

        // Only a valid line if we have all the parameters
        if(i > 7) {
            traj_data_turb.push_back(data);
            //std::cout << std::setprecision(15) << data.transpose() << std::endl;
        }

    }

    // Finally close the file
    file.close();

    // Error if we don't have any data
    if (traj_data_turb.empty()) {
        printf(RED "ERROR: Could not parse any data from the file!!\n" RESET);
        printf(RED "ERROR: %s\n" RESET,path_traj.c_str());
        std::exit(EXIT_FAILURE);
    }

}







std::vector<std::pair<size_t,Eigen::VectorXf>> Simulator::project_pointcloud(const Eigen::Matrix3d &R_GtoI, const Eigen::Vector3d &p_IinG,
                                                                             int camid, const std::unordered_map<size_t,Eigen::Vector3d> &feats) {

    // Assert we have good camera
    assert(camid < params.state_options.num_cameras);
    assert((int)params.camera_fisheye.size() == params.state_options.num_cameras);
    assert((int)params.camera_wh.size() == params.state_options.num_cameras);
    assert((int)params.camera_intrinsics.size() == params.state_options.num_cameras);
    assert((int)params.camera_extrinsics.size() == params.state_options.num_cameras);

    // Grab our extrinsic and intrinsic values
    Eigen::Matrix<double,3,3> R_ItoC = quat_2_Rot(params.camera_extrinsics.at(camid).block(0,0,4,1));
    Eigen::Matrix<double,3,1> p_IinC = params.camera_extrinsics.at(camid).block(4,0,3,1);
    Eigen::Matrix<double,8,1> cam_d = params.camera_intrinsics.at(camid);

    // Our projected uv true measurements
    std::vector<std::pair<size_t,Eigen::VectorXf>> uvs;

    // Loop through our map
    for(const auto &feat : feats) {

        // Transform feature into current camera frame
        Eigen::Vector3d p_FinI = R_GtoI*(feat.second-p_IinG);
        Eigen::Vector3d p_FinC = R_ItoC*p_FinI+p_IinC;

        // Skip cloud if too far away
        if(p_FinC(2) > 15 || p_FinC(2) < 0.5)
            continue;

        // Project to normalized coordinates
        Eigen::Vector2f uv_norm;
        uv_norm << p_FinC(0)/p_FinC(2),p_FinC(1)/p_FinC(2);

        // Distort the normalized coordinates (false=radtan, true=fisheye)
        Eigen::Vector2f uv_dist;

        // Calculate distortion uv and jacobian
        if(params.camera_fisheye.at(camid)) {

            // Calculate distorted coordinates for fisheye
            double r = sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
            double theta = std::atan(r);
            double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

            // Handle when r is small (meaning our xy is near the camera center)
            double inv_r = r > 1e-8 ? 1.0/r : 1;
            double cdist = r > 1e-8 ? theta_d * inv_r : 1;

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm(0)*cdist;
            double y1 = uv_norm(1)*cdist;
            uv_dist(0) = cam_d(0)*x1 + cam_d(2);
            uv_dist(1) = cam_d(1)*y1 + cam_d(3);

        } else {

            // Calculate distorted coordinates for radial
            double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
            double r_2 = r*r;
            double r_4 = r_2*r_2;
            double x1 = uv_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm(0)*uv_norm(1)+cam_d(7)*(r_2+2*uv_norm(0)*uv_norm(0));
            double y1 = uv_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm(1)*uv_norm(1))+2*cam_d(7)*uv_norm(0)*uv_norm(1);
            uv_dist(0) = cam_d(0)*x1 + cam_d(2);
            uv_dist(1) = cam_d(1)*y1 + cam_d(3);

        }

        // Check that it is inside our bounds
        if(uv_dist(0) < 0 || uv_dist(0) > params.camera_wh.at(camid).first || uv_dist(1) < 0 || uv_dist(1) > params.camera_wh.at(camid).second) {
            continue;
        }

        // Else we can add this as a good projection
        uvs.push_back({feat.first, uv_dist});

    }

    // Return our projections
    return uvs;

}


void Simulator::project_pointcloud_map(const Eigen::Matrix3d &R_CtoG, const Eigen::Vector3d &p_CinG,
                                       const Eigen::Matrix3d &R_CtoG_turb, const Eigen::Vector3d &p_CinG_turb,
                                       const Eigen::Matrix3d &R_kftoG, const Eigen::Vector3d &p_kfinG,
                                       const Eigen::Matrix3d &R_kftoG_turb, const Eigen::Vector3d &p_kfinG_turb,
                                       int camid, const std::unordered_map<size_t,Eigen::Vector3d> &feats,
                                       vector<Eigen::VectorXf> &uvs_cur,
                                       vector<Eigen::VectorXf> &uvs_kf, vector<Eigen::VectorXd> &pts3d_kf) {

    // Assert we have good camera
    assert(camid < params.state_options.num_cameras);
    assert((int)params.camera_fisheye.size() == params.state_options.num_cameras);
    assert((int)params.camera_wh.size() == params.state_options.num_cameras);
    assert((int)params.camera_intrinsics.size() == params.state_options.num_cameras);
    assert((int)params.camera_extrinsics.size() == params.state_options.num_cameras);

    // Grab our extrinsic and intrinsic values
    Eigen::Matrix<double,3,3> R_ItoC = quat_2_Rot(params.camera_extrinsics.at(camid).block(0,0,4,1));
    Eigen::Matrix<double,3,1> p_IinC = params.camera_extrinsics.at(camid).block(4,0,3,1);
    Eigen::Matrix<double,8,1> cam_d = params.camera_intrinsics.at(camid);

    // Our projected uv true measurements
    std::vector<std::pair<size_t,Eigen::VectorXf>> uvs;

    // Loop through our map
    for(const auto &feat : feats) {

        // Transform feature into current camera frame
       
//        Eigen::Vector3d p_FinC = R_CtoG.transpose()*(feat.second-p_CinG);
        Eigen::Vector3d p_FinC = R_CtoG_turb.transpose()*(featmatchingturb.at(feat.first)-p_CinG_turb);

        // Skip cloud if too far away
        if(p_FinC(2) > 50 || p_FinC(2) < 0.5)
            continue;

        // Project to normalized coordinates
        Eigen::Vector2f uv_norm;
        uv_norm << p_FinC(0)/p_FinC(2),p_FinC(1)/p_FinC(2);

        // Distort the normalized coordinates (false=radtan, true=fisheye)
        Eigen::Vector2f uv_dist;

        // Calculate distortion uv and jacobian
        if(params.camera_fisheye.at(camid)) {

            // Calculate distorted coordinates for fisheye
            double r = sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
            double theta = std::atan(r);
            double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

            // Handle when r is small (meaning our xy is near the camera center)
            double inv_r = r > 1e-8 ? 1.0/r : 1;
            double cdist = r > 1e-8 ? theta_d * inv_r : 1;

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm(0)*cdist;
            double y1 = uv_norm(1)*cdist;
            uv_dist(0) = cam_d(0)*x1 + cam_d(2);
            uv_dist(1) = cam_d(1)*y1 + cam_d(3);

        } else {

            // Calculate distorted coordinates for radial
            double r = std::sqrt(uv_norm(0)*uv_norm(0)+uv_norm(1)*uv_norm(1));
            double r_2 = r*r;
            double r_4 = r_2*r_2;
            double x1 = uv_norm(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm(0)*uv_norm(1)+cam_d(7)*(r_2+2*uv_norm(0)*uv_norm(0));
            double y1 = uv_norm(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm(1)*uv_norm(1))+2*cam_d(7)*uv_norm(0)*uv_norm(1);
            uv_dist(0) = cam_d(0)*x1 + cam_d(2);
            uv_dist(1) = cam_d(1)*y1 + cam_d(3);

        }

        // Check that it is inside our bounds
        if(uv_dist(0) < 0 || uv_dist(0) > params.camera_wh.at(camid).first || uv_dist(1) < 0 || uv_dist(1) > params.camera_wh.at(camid).second) {
            continue;
        }


//        Eigen::Vector3d p_Finkf = R_kftoG.transpose()*(feat.second-p_kfinG);
        Eigen::Vector3d p_Finkf = R_kftoG_turb.transpose()*(featmatchingturb.at(feat.first)-p_kfinG_turb);
         // Skip cloud if too far away
        if(p_Finkf(2) > 50 || p_Finkf(2) < 0.5)
            continue;

        // Project to normalized coordinates
        Eigen::Vector2f uv_norm_kf;
        uv_norm_kf << p_Finkf(0)/p_Finkf(2),p_Finkf(1)/p_Finkf(2);

        // Distort the normalized coordinates (false=radtan, true=fisheye)
        Eigen::Vector2f uv_dist_kf;

        // Calculate distortion uv and jacobian
        if(params.camera_fisheye.at(camid)) {

            // Calculate distorted coordinates for fisheye
            double r = sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
            double theta = std::atan(r);
            double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

            // Handle when r is small (meaning our xy is near the camera center)
            double inv_r = r > 1e-8 ? 1.0/r : 1;
            double cdist = r > 1e-8 ? theta_d * inv_r : 1;

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm_kf(0)*cdist;
            double y1 = uv_norm_kf(1)*cdist;
            uv_dist_kf(0) = cam_d(0)*x1 + cam_d(2);
            uv_dist_kf(1) = cam_d(1)*y1 + cam_d(3);

        } else {

            // Calculate distorted coordinates for radial
            double r = std::sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
            double r_2 = r*r;
            double r_4 = r_2*r_2;
            double x1 = uv_norm_kf(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm_kf(0)*uv_norm_kf(1)+cam_d(7)*(r_2+2*uv_norm_kf(0)*uv_norm_kf(0));
            double y1 = uv_norm_kf(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm_kf(1)*uv_norm_kf(1))+2*cam_d(7)*uv_norm_kf(0)*uv_norm_kf(1);
            uv_dist_kf(0) = cam_d(0)*x1 + cam_d(2);
            uv_dist_kf(1) = cam_d(1)*y1 + cam_d(3);

        }

        // Check that it is inside our bounds
        if(uv_dist_kf(0) < 0 || uv_dist_kf(0) > params.camera_wh.at(camid).first || uv_dist_kf(1) < 0 || uv_dist_kf(1) > params.camera_wh.at(camid).second) {
            continue;
        }



        // Else we can add this as a good projection
        std::normal_distribution<double> w(0,1);
        

        // uv_dist(0) += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
        // uv_dist(1) += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
        uv_dist(0) = uvsmatching[feat.first](0);
        uv_dist(1) = uvsmatching[feat.first](1);
//        uv_dist_kf(0) += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
//        uv_dist_kf(1) += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
        
        uvs_cur.push_back(uv_dist);
        uvs_kf.push_back(uv_dist_kf);
        pts3d_kf.push_back(p_Finkf);

    }

    // Return our projections
    return ;

}




void Simulator::generate_points(const Eigen::Matrix3d &R_GtoI, const Eigen::Vector3d &p_IinG,
                                int camid, std::unordered_map<size_t,Eigen::Vector3d> &feats, int numpts) {

    // Assert we have good camera
    assert(camid < params.state_options.num_cameras);
    assert((int)params.camera_fisheye.size() == params.state_options.num_cameras);
    assert((int)params.camera_wh.size() == params.state_options.num_cameras);
    assert((int)params.camera_intrinsics.size() == params.state_options.num_cameras);
    assert((int)params.camera_extrinsics.size() == params.state_options.num_cameras);

    // Grab our extrinsic and intrinsic values
    Eigen::Matrix<double,3,3> R_ItoC = quat_2_Rot(params.camera_extrinsics.at(camid).block(0,0,4,1));
    Eigen::Matrix<double,3,1> p_IinC = params.camera_extrinsics.at(camid).block(4,0,3,1);
    Eigen::Matrix<double,8,1> cam_d = params.camera_intrinsics.at(camid);

    // Convert to opencv format since we will use their undistort functions
    cv::Matx33d camK;
    camK(0, 0) = cam_d(0);
    camK(0, 1) = 0;
    camK(0, 2) = cam_d(2);
    camK(1, 0) = 0;
    camK(1, 1) = cam_d(1);
    camK(1, 2) = cam_d(3);
    camK(2, 0) = 0;
    camK(2, 1) = 0;
    camK(2, 2) = 1;
    cv::Vec4d camD;
    camD(0) = cam_d(4);
    camD(1) = cam_d(5);
    camD(2) = cam_d(6);
    camD(3) = cam_d(7);


    // Generate the desired number of features
    for(int i=0; i<numpts; i++) {

        // Uniformly randomly generate within our fov
        std::uniform_real_distribution<double> gen_u(0,params.camera_wh.at(camid).first);
        std::uniform_real_distribution<double> gen_v(0,params.camera_wh.at(camid).second);
        double u_dist = gen_u(gen_state_init);
        double v_dist = gen_v(gen_state_init);

        // Convert to opencv format
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = u_dist;
        mat.at<float>(0, 1) = v_dist;
        mat = mat.reshape(2); // Nx1, 2-channel

        // Undistort this point to our normalized coordinates (false=radtan, true=fisheye)
        if(params.camera_fisheye.at(camid)) {
            cv::fisheye::undistortPoints(mat, mat, camK, camD);
        } else {
            cv::undistortPoints(mat, mat, camK, camD);
        }

        // Construct our return vector
        Eigen::Vector3d uv_norm;
        mat = mat.reshape(1); // Nx2, 1-channel
        uv_norm(0) = mat.at<float>(0, 0);
        uv_norm(1) = mat.at<float>(0, 1);
        uv_norm(2) = 1;

        // Generate a random depth
        // TODO: we should probably have this as a simulation parameter
        std::uniform_real_distribution<double> gen_depth(5,10);
        double depth = gen_depth(gen_state_init);

        // Get the 3d point
        Eigen::Vector3d p_FinC;
        p_FinC = depth*uv_norm;

        // Move to the global frame of reference
        Eigen::Vector3d p_FinI = R_ItoC.transpose()*(p_FinC-p_IinC);
        Eigen::Vector3d p_FinG = R_GtoI.transpose()*p_FinI+p_IinG;

        // Append this as a new feature
        featmap.insert({id_map,p_FinG});
        id_map++;

    }


}


bool Simulator::generate_points_map(const Eigen::Matrix3d &R_CtoG, const Eigen::Vector3d &p_CinG,
                                    const Eigen::Matrix3d &R_CtoG_turb, const Eigen::Vector3d &p_CinG_turb,
                                    const Eigen::Matrix3d &R_kftoG, const Eigen::Vector3d &p_kfinG,
                                    const Eigen::Matrix3d &R_kftoG_turb, const Eigen::Vector3d &p_kfinG_turb,
                                    int camid, const std::unordered_map<size_t,Eigen::Vector3d> &feats,
                                    int numpts, vector<Eigen::VectorXf> &uvs_cur,
                                    vector<Eigen::VectorXf> &uvs_kf, vector<Eigen::VectorXd> &pts3d_kf) {

    // Assert we have good camera
    assert(camid < params.state_options.num_cameras);
    assert((int)params.camera_fisheye.size() == params.state_options.num_cameras);
    assert((int)params.camera_wh.size() == params.state_options.num_cameras);
    assert((int)params.camera_intrinsics.size() == params.state_options.num_cameras);
    assert((int)params.camera_extrinsics.size() == params.state_options.num_cameras);

    // Grab our extrinsic and intrinsic values
    Eigen::Matrix<double,3,3> R_ItoC = quat_2_Rot(params.camera_extrinsics.at(camid).block(0,0,4,1));
    Eigen::Matrix<double,3,1> p_IinC = params.camera_extrinsics.at(camid).block(4,0,3,1);
    Eigen::Matrix<double,8,1> cam_d = params.camera_intrinsics.at(camid);

    // Convert to opencv format since we will use their undistort functions
    cv::Matx33d camK;
    camK(0, 0) = cam_d(0);
    camK(0, 1) = 0;
    camK(0, 2) = cam_d(2);
    camK(1, 0) = 0;
    camK(1, 1) = cam_d(1);
    camK(1, 2) = cam_d(3);
    camK(2, 0) = 0;
    camK(2, 1) = 0;
    camK(2, 2) = 1;
    cv::Vec4d camD;
    camD(0) = cam_d(4);
    camD(1) = cam_d(5);
    camD(2) = cam_d(6);
    camD(3) = cam_d(7);


    int count=0;
    int mark=0;
    while(count<numpts)
    {
        if(count==0)
        {
           mark++;
        }
        if(mark>20)
            return false;
        std::uniform_real_distribution<double> gen_u(0,params.camera_wh.at(camid).first);
        std::uniform_real_distribution<double> gen_v(0,params.camera_wh.at(camid).second);
        double u_dist = gen_u(gen_state_init);
        double v_dist = gen_v(gen_state_init);

        // Convert to opencv format
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = u_dist;
        mat.at<float>(0, 1) = v_dist;
        mat = mat.reshape(2); // Nx1, 2-channel

        if(params.camera_fisheye.at(camid)) {
            cv::fisheye::undistortPoints(mat, mat, camK, camD);
        } else {
            cv::undistortPoints(mat, mat, camK, camD);
        }

        // Construct our return vector
        Eigen::Vector3d uv_norm;
        mat = mat.reshape(1); // Nx2, 1-channel
        uv_norm(0) = mat.at<float>(0, 0);
        uv_norm(1) = mat.at<float>(0, 1);
        uv_norm(2) = 1;
        cout<<"count: "<<count<<endl;
        std::uniform_real_distribution<double> gen_depth(5,40);
        double depth = gen_depth(gen_state_init);
        cout<<"depth: "<<depth<<endl;

        Eigen::Vector3d p_FinC;
        p_FinC = depth*uv_norm;

        std::normal_distribution<double> w_pt(0,0.5);

        Eigen::Vector3d p_FinG=R_CtoG*p_FinC+p_CinG;

        Vector3d d_cur_=R_CtoG*uv_norm;
        Vector3d p=p_CinG+d_cur_*depth;
        std::cout <<"ground truth: "<< p_FinG.transpose() <<std::endl;
        cout<<"p: "<<p.transpose()<<endl;
        d_cur_.normalize();
        p=p_CinG+d_cur_*depth*uv_norm.norm();
        cout<<"p: "<<p.transpose()<<endl;
//        sleep(1);

        Eigen::Vector3d p_Finkf = R_kftoG.transpose()*(p_FinG-p_kfinG);
//        Eigen::Vector3d p_Finkf_turb = R_kftoG.transpose()*(p_FinG_turb-p_kfinG);
         // Skip cloud if too far away
        if(p_Finkf(2) > 50 || p_Finkf(2) < 0.5)
            continue;

        // Project to normalized coordinates
        Eigen::Vector2f uv_norm_kf;
        uv_norm_kf << p_Finkf(0)/p_Finkf(2),p_Finkf(1)/p_Finkf(2);

        // Distort the normalized coordinates (false=radtan, true=fisheye)
        Eigen::Vector2f uv_dist_kf;

        // Calculate distortion uv and jacobian
        if(params.camera_fisheye.at(camid)) {

            // Calculate distorted coordinates for fisheye
            double r = sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
            double theta = std::atan(r);
            double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

            // Handle when r is small (meaning our xy is near the camera center)
            double inv_r = r > 1e-8 ? 1.0/r : 1;
            double cdist = r > 1e-8 ? theta_d * inv_r : 1;

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm_kf(0)*cdist;
            double y1 = uv_norm_kf(1)*cdist;
            uv_dist_kf(0) = cam_d(0)*x1 + cam_d(2);
            uv_dist_kf(1) = cam_d(1)*y1 + cam_d(3);

        } else {

            // Calculate distorted coordinates for radial
            double r = std::sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
            double r_2 = r*r;
            double r_4 = r_2*r_2;
            double x1 = uv_norm_kf(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm_kf(0)*uv_norm_kf(1)+cam_d(7)*(r_2+2*uv_norm_kf(0)*uv_norm_kf(0));
            double y1 = uv_norm_kf(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm_kf(1)*uv_norm_kf(1))+2*cam_d(7)*uv_norm_kf(0)*uv_norm_kf(1);
            uv_dist_kf(0) = cam_d(0)*x1 + cam_d(2);
            uv_dist_kf(1) = cam_d(1)*y1 + cam_d(3);

        }

        // Check that it is inside our bounds
        if(uv_dist_kf(0) < 0 || uv_dist_kf(0) > params.camera_wh.at(camid).first || uv_dist_kf(1) < 0 || uv_dist_kf(1) > params.camera_wh.at(camid).second) {
            continue;
        }

        
        // Else we can add this as a good projection
        std::normal_distribution<double> w(0,1);
        cout<<"u v cur before turb: "<<u_dist<<" "<<v_dist<<endl;
        cout<<"u v kf before turb: "<<uv_dist_kf(0)<<" "<<uv_dist_kf(1)<<endl;
        u_dist += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
        v_dist += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
        uv_dist_kf(0) += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
        uv_dist_kf(1) += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
        Vector2f uv_dist;
        uv_dist<<u_dist,v_dist;

        if(uv_dist_kf(0) < 0 || uv_dist_kf(0) > params.camera_wh.at(camid).first || uv_dist_kf(1) < 0 || uv_dist_kf(1) > params.camera_wh.at(camid).second) {
            continue;
        }
        if(uv_dist(0) < 0 || uv_dist(0) > params.camera_wh.at(camid).first || uv_dist(1) < 0 || uv_dist(1) > params.camera_wh.at(camid).second) {
            continue;
        }
        cout<<"u v cur after turb: "<<u_dist<<" "<<v_dist<<endl;
        cout<<"u v kf after turb: "<<uv_dist_kf(0)<<" "<<uv_dist_kf(1)<<endl;



        /*****triangulate point through noisy uv observation******/
        // Convert to opencv format
        cv::Mat mat_cur(1, 2, CV_32F);
        mat_cur.at<float>(0, 0) = uv_dist(0);
        mat_cur.at<float>(0, 1) = uv_dist(1);
        mat_cur = mat_cur.reshape(2); // Nx1, 2-channel

        if(params.camera_fisheye.at(camid)) {
            cv::fisheye::undistortPoints(mat_cur, mat_cur, camK, camD);
        } else {
            cv::undistortPoints(mat_cur, mat_cur, camK, camD);
        }
        // Construct our return vector
        Eigen::Vector3d uv_norm_turb;
        mat_cur = mat_cur.reshape(1); // Nx2, 1-channel
        uv_norm_turb(0) = mat_cur.at<float>(0, 0);
        uv_norm_turb(1) = mat_cur.at<float>(0, 1);
        uv_norm_turb(2) = 1;

        cv::Mat mat_kf(1, 2, CV_32F);
        mat_kf.at<float>(0, 0) = uv_dist_kf(0);
        mat_kf.at<float>(0, 1) = uv_dist_kf(1);
        mat_kf = mat_kf.reshape(2); // Nx1, 2-channel

        if(params.camera_fisheye.at(camid)) {
            cv::fisheye::undistortPoints(mat_kf, mat_kf, camK, camD);
        } else {
            cv::undistortPoints(mat_kf, mat_kf, camK, camD);
        }
        // Construct our return vector
        Eigen::Vector3d uv_norm_kf_turb;
        mat_kf = mat_kf.reshape(1); // Nx2, 1-channel
        uv_norm_kf_turb(0) = mat_kf.at<float>(0, 0);
        uv_norm_kf_turb(1) = mat_kf.at<float>(0, 1);
        uv_norm_kf_turb(2) = 1;

//        Vector3d d_cur=R_CtoG_turb*uv_norm_turb;
//        d_cur.normalize();
//        Vector3d d_kf=R_kftoG_turb*uv_norm_kf_turb;
//        d_kf.normalize();
//
//        Vector3d n=d_cur.cross(d_kf);
//        Vector3d n2=d_kf.cross(n);
//        Vector3d n1=d_cur.cross(n);
//        Vector3d trans_error_1=p_kfinG_turb-p_CinG_turb;
//        Vector3d trans_error_2=p_CinG_turb-p_kfinG_turb;
//        Vector3d c1=p_CinG_turb+trans_error_1.dot(n2)/(d_cur.dot(n2))*d_cur;
//        Vector3d c2=p_kfinG_turb+trans_error_2.dot(n1)/(d_kf.dot(n1))*d_kf;
//        Eigen::Vector3d pt3d_turb_=(c1+c2)*0.5;
//
//        std::cout <<"ground truth: \n"<< p_FinG.transpose() <<std::endl;
//        std::cout <<"your result: \n"<< pt3d_turb_.transpose() <<std::endl;
//
//        Vector3d d_cur_=R_CtoG*uv_norm;
//        d_cur_.normalize();
//        Vector3d uv_norm_kf_gt;
//        uv_norm_kf_gt(0)=uv_norm_kf(0);
//        uv_norm_kf_gt(1)=uv_norm_kf(1);
//        uv_norm_kf_gt(2)=1;
//
//        Vector3d d_kf_=R_kftoG*uv_norm_kf_gt;
//        d_kf_.normalize();
//
//        Vector3d n_=d_cur_.cross(d_kf_);
//
//        Vector3d n2_=d_kf_.cross(n_);
//
//        Vector3d n1_=d_cur_.cross(n_);
//
//        Vector3d trans_error_1_=p_kfinG-p_CinG;
//        Vector3d trans_error_2_=p_CinG-p_kfinG;
//        Vector3d c1_=p_CinG+trans_error_1_.dot(n2_)/(d_cur_.dot(n2_))*d_cur_;
//        Vector3d c2_=p_kfinG+trans_error_2_.dot(n1_)/(d_kf_.dot(n1_))*d_kf_;
//        Eigen::Vector3d pt3d_=(c1_+c2_)*0.5;
//
//        Vector3d p=p_CinG+d_cur_*depth*uv_norm.norm();
//
//        std::cout <<"ground truth: \n"<< p_FinG.transpose() <<std::endl;
//        cout<<"p: "<<p.transpose()<<endl;
//        std::cout <<"your result: \n"<< pt3d_.transpose() <<std::endl;


        Matrix3d R_GtoC_turb=R_CtoG_turb.transpose();
        Vector3d p_GinC_turb=-R_GtoC_turb*p_CinG_turb;
        Matrix3d R_Gtokf_turb=R_kftoG_turb.transpose();
        Vector3d p_Ginkf_turb=-R_Gtokf_turb*p_kfinG_turb;
        Matrix<double,3,4> T_GtoC_turb=MatrixXd::Zero(3,4);
        T_GtoC_turb.block(0,0,3,3)=R_GtoC_turb;
        T_GtoC_turb.block(0,3,3,1)=p_GinC_turb;
        Matrix<double,3,4> T_Gtokf_turb=MatrixXd::Zero(3,4);
        T_Gtokf_turb.block(0,0,3,3)=R_Gtokf_turb;
        T_Gtokf_turb.block(0,3,3,1)=p_Ginkf_turb;

        Matrix<double,4,4> A=MatrixXd::Zero(4,4);
        A.block(0,0,1,4)=uv_norm_turb(0)*T_GtoC_turb.block(2,0,1,4)-T_GtoC_turb.block(0,0,1,4);
        A.block(1,0,1,4)=uv_norm_turb(1)*T_GtoC_turb.block(2,0,1,4)-T_GtoC_turb.block(1,0,1,4);
        A.block(2,0,1,4)=uv_norm_kf_turb(0)*T_Gtokf_turb.block(2,0,1,4)-T_Gtokf_turb.block(0,0,1,4);
        A.block(3,0,1,4)=uv_norm_kf_turb(1)*T_Gtokf_turb.block(2,0,1,4)-T_Gtokf_turb.block(1,0,1,4);

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A.transpose()*A, Eigen::ComputeThinU | Eigen::ComputeThinV );
        Eigen::Matrix4d V = svd.matrixV(), U = svd.matrixU();
//        cout<<"V: "<<endl<<V<<endl;
//        cout<<"U: "<<endl<<U<<endl;
        Eigen::MatrixXd Singular=svd.singularValues();
//        cout<<"S: "<<endl<<Singular<<endl;
        Eigen::Vector3d pt3d_turb;
        pt3d_turb=V.col(3).head<3>()/V(3,3);
        cout<<"singular: "<<Singular.transpose()<<endl;
        cout<<"count: "<<count<<endl;
        if(Singular(3)<Singular(2)*1e-2)  //make sure sigma4<<sigma3
        {
            cout<<"the triangle with turb is valid"<<endl;
            std::cout <<"ground truth: \n"<< p_FinG.transpose() <<std::endl;
            std::cout <<"your result: \n"<< pt3d_turb.transpose() <<std::endl;
//            sleep(1);
        }
        else
        {
            continue;
        }
        Vector3d p_Fincur_turb;
        p_Fincur_turb=R_GtoC_turb*pt3d_turb+p_GinC_turb;
        cout<<"p_FinCur_turb: "<<p_Fincur_turb.transpose()<<endl;
//        assert(p_Fincur_turb(2)>0);
        if(p_Fincur_turb(2) > 50 || p_Fincur_turb(2) < 0.5)
            continue;
        Vector3d p_Finkf_turb;
        p_Finkf_turb=R_Gtokf_turb*pt3d_turb+p_Ginkf_turb;
        cout<<"p_Finkf_turb: "<<p_Finkf_turb.transpose()<<endl;
        if(p_Finkf_turb(2) > 50 || p_Finkf_turb(2) < 0.5)
            continue;



        /******reproject to image*********/
        Eigen::Vector2f uv_norm_cur_turb;
        uv_norm_cur_turb << p_Fincur_turb(0)/p_Fincur_turb(2),p_Fincur_turb(1)/p_Fincur_turb(2);

        // Distort the normalized coordinates (false=radtan, true=fisheye)
        Eigen::Vector2f uv_cur_dist_turb;

        // Calculate distortion uv and jacobian
        if(params.camera_fisheye.at(camid)) {

            // Calculate distorted coordinates for fisheye
            double r = sqrt(uv_norm_cur_turb(0)*uv_norm_cur_turb(0)+uv_norm_cur_turb(1)*uv_norm_cur_turb(1));
            double theta = std::atan(r);
            double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

            // Handle when r is small (meaning our xy is near the camera center)
            double inv_r = r > 1e-8 ? 1.0/r : 1;
            double cdist = r > 1e-8 ? theta_d * inv_r : 1;

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm_cur_turb(0)*cdist;
            double y1 = uv_norm_cur_turb(1)*cdist;
            uv_cur_dist_turb(0) = cam_d(0)*x1 + cam_d(2);
            uv_cur_dist_turb(1) = cam_d(1)*y1 + cam_d(3);

        } else {

            // Calculate distorted coordinates for radial
            double r = std::sqrt(uv_norm_cur_turb(0)*uv_norm_cur_turb(0)+uv_norm_cur_turb(1)*uv_norm_cur_turb(1));
            double r_2 = r*r;
            double r_4 = r_2*r_2;
            double x1 = uv_norm_cur_turb(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm_cur_turb(0)*uv_norm_cur_turb(1)+cam_d(7)*(r_2+2*uv_norm_cur_turb(0)*uv_norm_cur_turb(0));
            double y1 = uv_norm_cur_turb(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm_cur_turb(1)*uv_norm_cur_turb(1))+2*cam_d(7)*uv_norm_cur_turb(0)*uv_norm_cur_turb(1);
            uv_cur_dist_turb(0) = cam_d(0)*x1 + cam_d(2);
            uv_cur_dist_turb(1) = cam_d(1)*y1 + cam_d(3);

        }
        cout<<"uv cur reproject: "<<uv_cur_dist_turb(0)<<" "<<uv_cur_dist_turb(1)<<endl;

        Eigen::Vector2f uv_norm_kf_turb_;
        uv_norm_kf_turb_ << p_Finkf_turb(0)/p_Finkf_turb(2),p_Finkf_turb(1)/p_Finkf_turb(2);

        // Distort the normalized coordinates (false=radtan, true=fisheye)
        Eigen::Vector2f uv_dist_kf_turb;

        // Calculate distortion uv and jacobian
        if(params.camera_fisheye.at(camid)) {

            // Calculate distorted coordinates for fisheye
            double r = sqrt(uv_norm_kf_turb_(0)*uv_norm_kf_turb_(0)+uv_norm_kf_turb_(1)*uv_norm_kf_turb_(1));
            double theta = std::atan(r);
            double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

            // Handle when r is small (meaning our xy is near the camera center)
            double inv_r = r > 1e-8 ? 1.0/r : 1;
            double cdist = r > 1e-8 ? theta_d * inv_r : 1;

            // Calculate distorted coordinates for fisheye
            double x1 = uv_norm_kf_turb_(0)*cdist;
            double y1 = uv_norm_kf_turb_(1)*cdist;
            uv_dist_kf_turb(0) = cam_d(0)*x1 + cam_d(2);
            uv_dist_kf_turb(1) = cam_d(1)*y1 + cam_d(3);

        } else {

            // Calculate distorted coordinates for radial
            double r = std::sqrt(uv_norm_kf_turb_(0)*uv_norm_kf_turb_(0)+uv_norm_kf_turb_(1)*uv_norm_kf_turb_(1));
            double r_2 = r*r;
            double r_4 = r_2*r_2;
            double x1 = uv_norm_kf_turb_(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm_kf_turb_(0)*uv_norm_kf_turb_(1)+cam_d(7)*(r_2+2*uv_norm_kf_turb_(0)*uv_norm_kf_turb_(0));
            double y1 = uv_norm_kf_turb_(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm_kf_turb_(1)*uv_norm_kf_turb_(1))+2*cam_d(7)*uv_norm_kf_turb_(0)*uv_norm_kf_turb_(1);
            uv_dist_kf_turb(0) = cam_d(0)*x1 + cam_d(2);
            uv_dist_kf_turb(1) = cam_d(1)*y1 + cam_d(3);

        }
        cout<<"uv kf reproject: "<<uv_dist_kf_turb(0)<<" "<<uv_dist_kf_turb(1)<<endl;


        /*******tri with gt pose******/

//        Matrix3d R_GtoC=R_CtoG.transpose();
//        Vector3d p_GinC=-R_GtoC*p_CinG;
//        Matrix3d R_Gtokf=R_kftoG.transpose();
//        Vector3d p_Ginkf=-R_Gtokf*p_kfinG;
//        Matrix<double,3,4> T_GtoC=MatrixXd::Zero(3,4);
//        T_GtoC.block(0,0,3,3)=R_GtoC;
//        T_GtoC.block(0,3,3,1)=p_GinC;
//        Matrix<double,3,4> T_Gtokf=MatrixXd::Zero(3,4);
//        T_Gtokf.block(0,0,3,3)=R_Gtokf;
//        T_Gtokf.block(0,3,3,1)=p_Ginkf;
//
//        Matrix<double,4,4> D=MatrixXd::Zero(4,4);
//        D.block(0,0,1,4)=uv_norm(0)*T_GtoC.block(2,0,1,4)-T_GtoC.block(0,0,1,4);
//        D.block(1,0,1,4)=uv_norm(1)*T_GtoC.block(2,0,1,4)-T_GtoC.block(1,0,1,4);
//        D.block(2,0,1,4)=uv_norm_kf(0)*T_Gtokf.block(2,0,1,4)-T_Gtokf.block(0,0,1,4);
//        D.block(3,0,1,4)=uv_norm_kf(1)*T_Gtokf.block(2,0,1,4)-T_Gtokf.block(1,0,1,4);
//
//        Eigen::JacobiSVD<Eigen::MatrixXd> svd_(D.transpose()*D, Eigen::ComputeThinU | Eigen::ComputeThinV );
//        Eigen::Matrix4d V_ = svd_.matrixV(), U_ = svd_.matrixU();
////        cout<<"V: "<<endl<<V<<endl;
////        cout<<"U: "<<endl<<U<<endl;
//        Eigen::MatrixXd Singular_=svd_.singularValues();
////        cout<<"S: "<<endl<<Singular<<endl;
//        Eigen::Vector3d pt3d;
//        pt3d=V_.col(3).head<3>()/V_(3,3);
////        cout<<"singular: "<<Singular_.transpose()<<endl;
////        cout<<"count: "<<count<<endl;
//        if(Singular_(3)<Singular_(2)*1e-3)  //make sure sigma4<<sigma3
//        {
//            cout<<"the triangle without turb is valid"<<endl;
//            std::cout <<"ground truth: \n"<< p_FinG.transpose() <<std::endl;
//            std::cout <<"your result: \n"<< pt3d.transpose() <<std::endl;
////            sleep(1);
//        }
//        else
//        {
//            continue;
//        }
//        Vector3d p_Fincur_;
//        p_Fincur_= R_GtoC*pt3d+p_GinC;
//        cout<<"p_FinCur: "<<p_FinC.transpose()<<endl;
////        assert(p_Fincur_turb(2)>0);
//        if(p_Fincur_(2) > 50 || p_Fincur_(2) < 0.5)
//            continue;
//        Vector3d p_Finkf_;
//        p_Finkf_=R_Gtokf*pt3d+p_Ginkf;
//        cout<<"p_Finkf: "<<p_Finkf.transpose()<<endl;
//        if(p_Finkf_(2) > 50 || p_Finkf_(2) < 0.5)
//            continue;
//
//
//
//        cout<<"p_CinG: "<<p_CinG.transpose()<<endl;
//        cout<<"p_KFinG: "<<p_kfinG.transpose()<<endl;
//        cout<<"p_CinG_turb: "<<p_CinG_turb.transpose()<<endl;
//        cout<<"p_KFinG_turb: "<<p_kfinG_turb.transpose()<<endl;
//
//        cout<<"p_FinCur compute: "<<p_Fincur_.transpose()<<endl;
//        cout<<"p_Finkf  compute: "<<p_Finkf_.transpose()<<endl;

        /*****finish triagualte point****/
        featmatching.insert({id_matching,p_FinG});
        featmatchingturb.insert({id_matching,pt3d_turb});
        uvsmatching.insert({id_matching,uv_dist});
        id_matching++;

        uvs_cur.push_back(uv_dist);
        uvs_kf.push_back(uv_dist_kf);
        pts3d_kf.push_back(p_Finkf_turb);

        count++;
    }
    return true;



}



void Simulator::gen_matching(vector<Matrix3d> &kfs_pose_R,vector<Vector3d> &kfs_pose_t,
                  vector<Matrix3d> &kfs_pose_R_turb,vector<Vector3d> &kfs_pose_t_turb,
                  Matrix3d &R_CtoG,Vector3d &p_CinG,
                  vector<int> &matches_num,vector<string> &kfs_ts_with_gt,
                  vector<string> &res_kfs_ts,
                  vector<vector<Eigen::VectorXf>> &res_uvs_cur,
                  vector<vector<Eigen::VectorXf>> &res_uvs_kf,
                  vector<vector<Eigen::VectorXd>> &res_pts3d_kf)
{
    int kfnums=kfs_ts_with_gt.size();
    int camid=0;

    Eigen::Matrix<double,3,3> R_ItoC = quat_2_Rot(params.camera_extrinsics.at(camid).block(0,0,4,1));
    Eigen::Matrix<double,3,1> p_IinC = params.camera_extrinsics.at(camid).block(4,0,3,1);
    Eigen::Matrix<double,8,1> cam_d = params.camera_intrinsics.at(camid);

    // Convert to opencv format since we will use their undistort functions
    cv::Matx33d camK;
    camK(0, 0) = cam_d(0);
    camK(0, 1) = 0;
    camK(0, 2) = cam_d(2);
    camK(1, 0) = 0;
    camK(1, 1) = cam_d(1);
    camK(1, 2) = cam_d(3);
    camK(2, 0) = 0;
    camK(2, 1) = 0;
    camK(2, 2) = 1;
    cv::Vec4d camD;
    camD(0) = cam_d(4);
    camD(1) = cam_d(5);
    camD(2) = cam_d(6);
    camD(3) = cam_d(7);

    std::normal_distribution<double> w(0,1);
    //generate 300 point in kfs[0]
    int anchor_index=0;
    std::unordered_map<int,vector<int>> map_ptid_kfindex;
    std::unordered_map<int,vector<Vector2f>> map_ptid_kfuv;
    std::unordered_map<int,Vector3d> map_ptid_3dgt;
    std::unordered_map<int,Vector2f> map_ptid_curuv;
    int pt_num=0;
    for(; pt_num<500; ) {
        cout<<"****generate "<<pt_num<<"th point****"<<endl;

        // Uniformly randomly generate within our fov
        std::uniform_real_distribution<double> gen_u(0,params.camera_wh.at(camid).first);
        std::uniform_real_distribution<double> gen_v(0,params.camera_wh.at(camid).second);
        double u_dist = gen_u(gen_state_init);
        double v_dist = gen_v(gen_state_init);


        // Convert to opencv format
        cv::Mat mat(1, 2, CV_32F);
        mat.at<float>(0, 0) = u_dist;
        mat.at<float>(0, 1) = v_dist;
        mat = mat.reshape(2); // Nx1, 2-channel

        // Undistort this point to our normalized coordinates (false=radtan, true=fisheye)
        if(params.camera_fisheye.at(camid)) {
            cv::fisheye::undistortPoints(mat, mat, camK, camD);
        } else {
            cv::undistortPoints(mat, mat, camK, camD);
        }

        // Construct our return vector
        Eigen::Vector3d uv_norm;
        mat = mat.reshape(1); // Nx2, 1-channel
        uv_norm(0) = mat.at<float>(0, 0);
        uv_norm(1) = mat.at<float>(0, 1);
        uv_norm(2) = 1;

        // Generate a random depth
        // TODO: we should probably have this as a simulation parameter
        std::uniform_real_distribution<double> gen_depth(5,40);
        double depth = gen_depth(gen_state_init);

        // Get the 3d point
        Eigen::Vector3d p_FinKFAnchor;
        p_FinKFAnchor = depth*uv_norm;

        //Get the 3d point in the global
        Eigen::Vector3d pt3d_g=kfs_pose_R[anchor_index]*p_FinKFAnchor+kfs_pose_t[anchor_index];

        u_dist += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
        v_dist += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
        Vector2f uv_dist;
        uv_dist<<u_dist,v_dist;

        //project the 3d point into current frame
        Vector3d p_FinCur= R_CtoG.transpose() * (pt3d_g - p_CinG);
        if(p_FinCur(2)>50 || p_FinCur(2)<0.5)
            continue;
        Vector2f uv_cur_norm;
        uv_cur_norm<<p_FinCur(0)/p_FinCur(2),p_FinCur(1)/p_FinCur(2);
        Vector2f uv_dist_cur=reproject_uv_norm(uv_cur_norm,0,cam_d);
        if(uv_dist_cur(0) < 0 || uv_dist_cur(0) > params.camera_wh.at(camid).first || uv_dist_cur(1) < 0 || uv_dist_cur(1) > params.camera_wh.at(camid).second) {
            continue;
        }
        uv_dist_cur(0)+=params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
        uv_dist_cur(1)+=params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
        if(uv_dist_cur(0) < 0 || uv_dist_cur(0) > params.camera_wh.at(camid).first || uv_dist_cur(1) < 0 || uv_dist_cur(1) > params.camera_wh.at(camid).second) {
            continue;
        }

        vector<int> kfindex_match;
        vector<Vector2f> kfuv_match;
        //project the 3d point into other kfs
       for(int i=1;i<kfs_ts_with_gt.size();i++)
       {
           Matrix3d R_kftoG=kfs_pose_R[i];
           Vector3d p_kfinG=kfs_pose_t[i];
           Vector3d p_FinKFi=R_kftoG.transpose() * (pt3d_g-p_kfinG);

           if(p_FinKFi(2) > 50 || p_FinKFi(2) < 0.5)
               continue;

           // Project to normalized coordinates
           Eigen::Vector2f uv_norm_kf;
           uv_norm_kf << p_FinKFi(0)/p_FinKFi(2),p_FinKFi(1)/p_FinKFi(2);

           Vector2f uv_dist_kf= reproject_uv_norm(uv_norm_kf,camid,cam_d);

           // Check that it is inside our bounds
           if(uv_dist_kf(0) < 0 || uv_dist_kf(0) > params.camera_wh.at(camid).first || uv_dist_kf(1) < 0 || uv_dist_kf(1) > params.camera_wh.at(camid).second) {
               continue;
           }

           uv_dist_kf(0) += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));
           uv_dist_kf(1) += params.msckf_options.sigma_pix*w(gen_meas_cams.at(camid));

           if(uv_dist_kf(0) < 0 || uv_dist_kf(0) > params.camera_wh.at(camid).first || uv_dist_kf(1) < 0 || uv_dist_kf(1) > params.camera_wh.at(camid).second) {
               continue;
           }

           kfindex_match.push_back(i);
           kfuv_match.push_back(uv_dist_kf);

       }

       if(kfindex_match.empty())  //this 3dpt cannot project to other kfs
       {
           continue;
       }

       //the 3dpt could be projected to other kfs, we record the information for future triangulate
       vector<int> kfindces;
       vector<Vector2f> kfuvs;
       kfindces.push_back(anchor_index);
       kfuvs.push_back(uv_dist);
       for(int i=0;i<kfindex_match.size();i++)
       {
           kfindces.push_back(kfindex_match[i]);
           kfuvs.push_back(kfuv_match[i]);
       }
       assert(kfindces.size()==kfuvs.size());
       map_ptid_kfindex.insert({pt_num,kfindces});
       map_ptid_kfuv.insert({pt_num,kfuvs});
       map_ptid_3dgt.insert({pt_num,pt3d_g});
       map_ptid_curuv.insert({pt_num,uv_dist_cur});
       cout<<"numkf: "<<kfnums<<" v.s. kfcouldproject: "<<kfindces.size()<<endl;

       pt_num++;
    }


    assert(pt_num==map_ptid_kfindex.size());
    assert(pt_num==map_ptid_kfuv.size());
    assert(pt_num==map_ptid_3dgt.size());
    assert(pt_num==map_ptid_curuv.size());

    std::unordered_map<int,vector<int>> map_ptid_kfindex_success;
    std::unordered_map<int,vector<Vector2f>> map_ptid_kfuv_success;
    std::unordered_map<int,Vector3d> map_ptid_3d_success;
    std::unordered_map<int,Vector2f> map_ptid_curuv_success;
    //now, we could triangulate the point with perturbed information;
    for(int pt_index=0;pt_index<pt_num;pt_index++) //for each pt
    {
        vector<int> kfindces=map_ptid_kfindex[pt_index];
        vector<Vector2f> kfuvs=map_ptid_kfuv[pt_index];
        Vector3d p_FinG=map_ptid_3dgt[pt_index];
        Vector2f cur_uv=map_ptid_curuv[pt_index];

        int size=2*kfindces.size();  //for each kf, it provide 2 meas of the pt;
        MatrixXd A=MatrixXd::Zero(size,4);

        for(int i=0;i<kfindces.size();i++)  //for each kf that could observe the pt
        {
            Matrix3d R_KFtoG=kfs_pose_R_turb[kfindces[i]];
            Vector3d p_KFinG=kfs_pose_t_turb[kfindces[i]];
            Matrix3d R_GtoKF=R_KFtoG.transpose();
            Vector3d p_GinKF=-R_GtoKF*p_KFinG;
            Vector2f uv_dist_kf=kfuvs[i];

            cv::Mat mat(1, 2, CV_32F);
            mat.at<float>(0, 0) = uv_dist_kf(0);
            mat.at<float>(0, 1) = uv_dist_kf(1);
            mat = mat.reshape(2); // Nx1, 2-channel

            // Undistort this point to our normalized coordinates (false=radtan, true=fisheye)
            if(params.camera_fisheye.at(camid)) {
                cv::fisheye::undistortPoints(mat, mat, camK, camD);
            } else {
                cv::undistortPoints(mat, mat, camK, camD);
            }
            // Construct our return vector
            Eigen::Vector3d uv_norm_kf;
            mat = mat.reshape(1); // Nx2, 1-channel
            uv_norm_kf(0) = mat.at<float>(0, 0);
            uv_norm_kf(1) = mat.at<float>(0, 1);
            uv_norm_kf(2) = 1;

            Matrix<double,3,4> T_GtoKF=MatrixXd::Zero(3,4);
            T_GtoKF.block(0,0,3,3)=R_GtoKF;
            T_GtoKF.block(0,3,3,1)=p_GinKF;

            A.block(2*i,0,1,4)=uv_norm_kf(0)*T_GtoKF.block(2,0,1,4)-T_GtoKF.block(0,0,1,4);
            A.block(2*i+1,0,1,4)=uv_norm_kf(1)*T_GtoKF.block(2,0,1,4)-T_GtoKF.block(1,0,1,4);

        }

        //now compute the 3d coordinate of this pt;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A.transpose()*A, Eigen::ComputeThinU | Eigen::ComputeThinV );
        Eigen::Matrix4d V = svd.matrixV(), U = svd.matrixU();
        Eigen::MatrixXd Singular=svd.singularValues();
        Eigen::Vector3d pt3d_turb;
        pt3d_turb=V.col(3).head<3>()/V(3,3);
        cout<<"singular: "<<Singular.transpose()<<endl;
        if(Singular(3)<Singular(2)*1e-2)  //make sure sigma4<<sigma3
        {
            cout<<"the triangle with turb is valid"<<endl;
            std::cout <<"ground truth: \n"<< p_FinG.transpose() <<std::endl;
            std::cout <<"your result: \n"<< pt3d_turb.transpose() <<std::endl;
        }
        else
        {
            continue;
        }

        bool success=true;
        for(int i=0;i<kfindces.size();i++)
        {
            Matrix3d R_KFtoG=kfs_pose_R_turb[kfindces[i]];
            Vector3d p_KFinG=kfs_pose_t_turb[kfindces[i]];
            Matrix3d R_GtoKF=R_KFtoG.transpose();
            Vector3d p_GinKF=-R_GtoKF*p_KFinG;
            Vector2f uv_dist_kf=kfuvs[i];

            Vector3d p_FinKF=R_GtoKF*pt3d_turb+p_GinKF;
            if(p_FinKF(2) > 50 || p_FinKF(2) < 0.5)
            {
                success=false;
                break;
            }
            Vector2f uv_norm_kf;
            uv_norm_kf<<p_FinKF(0)/p_FinKF(2),p_FinKF(1)/p_FinKF(2);
            Vector2f uv_dist_kf_reproject=reproject_uv_norm(uv_norm_kf,0,cam_d);
            if(uv_dist_kf_reproject(0) < 0 || uv_dist_kf_reproject(0) > params.camera_wh.at(camid).first || uv_dist_kf_reproject(1) < 0 || uv_dist_kf_reproject(1) > params.camera_wh.at(camid).second) {
                success=false;
                break;
            }
            cout<<"uv ob: "<<uv_dist_kf.transpose()<<endl;
            cout<<"uv compute: "<<uv_dist_kf_reproject.transpose()<<endl;
        }

        if(!success)
            continue;

        //this point is successfully triangulated,record it
        map_ptid_3d_success.insert({pt_index,pt3d_turb});
        map_ptid_kfindex_success.insert({pt_index,kfindces});
        map_ptid_kfuv_success.insert({pt_index,kfuvs});
        map_ptid_curuv_success.insert({pt_index,cur_uv});
    }

    cout<<"generate point: "<<pt_num<<" success point: "<<map_ptid_3d_success.size()<<endl;
//    sleep(1);

    //project 3d pt into current frame;
    std::unordered_map<int,vector<int>> map_ptid_kfindex_final;
    std::unordered_map<int,vector<Vector2f>> map_ptid_kfuv_final;
    std::unordered_map<int,Vector3d> map_ptid_3d_final;
    std::unordered_map<int,Vector2f> map_ptid_curuv_final;
    for(int i=0;i<map_ptid_3d_success.size();i++)
    {
        Vector3d pt3d=map_ptid_3d_success[i];
        Matrix3d R_GtoC=R_CtoG.transpose();
        Vector3d p_GinC=-R_GtoC*p_CinG;
        Vector3d p_FinC=R_GtoC*pt3d+p_GinC;
        if(p_FinC(2)>50 ||p_FinC(2)<0.5)
            continue;
        Vector2f uv_norm;
        uv_norm<<p_FinC(0)/p_FinC(2),p_FinC(1)/p_FinC(2);
        Vector2f uv_dist=reproject_uv_norm(uv_norm,0,cam_d);
        if(uv_dist(0) < 0 || uv_dist(0) > params.camera_wh.at(camid).first || uv_dist(1) < 0 || uv_dist(1) > params.camera_wh.at(camid).second) {
            continue;
        }
        //success, record;
        map_ptid_3d_final.insert({i,pt3d});
        map_ptid_kfindex_final.insert({i,map_ptid_kfindex_success[i]});
        map_ptid_kfuv_final.insert({i,map_ptid_kfuv_success[i]});
        map_ptid_curuv_final.insert({i,map_ptid_curuv_success[i]});
    }
    cout<<"success project point: "<<map_ptid_3d_final.size()<<endl;
//    sleep(1);

    //for each kf, we record match information
    cout<<"the inital number of kf: "<<kfnums<<endl;
//    vector<int> record_index;
    for(int i=0;i<kfs_ts_with_gt.size();i++)
    {
        //loop all 3d pts, to see if the pt coud be observed by current kf
        vector<VectorXd> pts3d;
        vector<VectorXf> uvskf;
        vector<VectorXf> uvscur;
        vector<VectorXd> pts3dw;
        for(int j=0;j<map_ptid_3d_final.size();j++)
        {
            vector<int> kfindeces=map_ptid_kfindex_final[j];
            int index=-1;
            for(int k=0;k<kfindeces.size();k++)
            {
                if(kfindeces[k]==i)
                {
                    index=k;
                    break;
                }
            }
            if(index==-1)
                continue;
            //current pt could be observed by kf[i];
            Matrix3d R_KFtoG=kfs_pose_R_turb[i];
            Vector3d p_KFinG=kfs_pose_t_turb[i];
            Matrix3d R_GtoKF=R_KFtoG.transpose();
            Vector3d p_GinKF=-R_GtoKF*p_KFinG;
            Vector3d p_FinKF=R_GtoKF*map_ptid_3d_final[j]+p_GinKF;
            pts3d.push_back(p_FinKF);
            uvskf.push_back(map_ptid_kfuv_final[j][index]);
            uvscur.push_back(map_ptid_curuv_final[j]);
            pts3dw.push_back(map_ptid_3d_final[j]);

        }
        if(pts3d.empty())
            continue;

        cout<<"for kf["<<i<<"], it has "<<pts3d.size()<<" features"<<endl;
        cout<<"required match num: "<<matches_num[i]<<endl;
        if(matches_num[i]>pts3d.size())
        {
            res_kfs_ts.push_back(kfs_ts_with_gt[i]);
            res_pts3d_kf.push_back(pts3d);
            res_uvs_cur.push_back(uvscur);
            res_uvs_kf.push_back(uvskf);
        } else{

            res_kfs_ts.push_back(kfs_ts_with_gt[i]);
            uvscur.erase(uvscur.begin()+matches_num[i], uvscur.end());
            uvskf.erase(uvskf.begin()+matches_num[i], uvskf.end());
            pts3d.erase(pts3d.begin()+matches_num[i], pts3d.end());
            pts3dw.erase(pts3dw.begin()+matches_num[i],pts3dw.end());
            res_pts3d_kf.push_back(pts3d);
            res_uvs_cur.push_back(uvscur);
            res_uvs_kf.push_back(uvskf);
        }
        assert(of_point.is_open());
        for(int pt_ind=0;pt_ind<pts3dw.size();pt_ind++)
        {
            of_point<<to_string(pts3dw[pt_ind](0))<<" "<<to_string(pts3dw[pt_ind](1))<<" "<<to_string(pts3dw[pt_ind](2));
            of_point<<endl;
        }


    }
//    sleep(5);
}


Eigen::Vector2f Simulator::reproject_uv_norm(Eigen::Vector2f uv_norm_kf, int camid, Eigen::Matrix<double,8,1> cam_d)
{
    Eigen::Vector2f uv_dist_kf;

    // Calculate distortion uv and jacobian
    if(params.camera_fisheye.at(camid)) {

        // Calculate distorted coordinates for fisheye
        double r = sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
        double theta = std::atan(r);
        double theta_d = theta+cam_d(4)*std::pow(theta,3)+cam_d(5)*std::pow(theta,5)+cam_d(6)*std::pow(theta,7)+cam_d(7)*std::pow(theta,9);

        // Handle when r is small (meaning our xy is near the camera center)
        double inv_r = r > 1e-8 ? 1.0/r : 1;
        double cdist = r > 1e-8 ? theta_d * inv_r : 1;

        // Calculate distorted coordinates for fisheye
        double x1 = uv_norm_kf(0)*cdist;
        double y1 = uv_norm_kf(1)*cdist;
        uv_dist_kf(0) = cam_d(0)*x1 + cam_d(2);
        uv_dist_kf(1) = cam_d(1)*y1 + cam_d(3);

    } else {

        // Calculate distorted coordinates for radial
        double r = std::sqrt(uv_norm_kf(0)*uv_norm_kf(0)+uv_norm_kf(1)*uv_norm_kf(1));
        double r_2 = r*r;
        double r_4 = r_2*r_2;
        double x1 = uv_norm_kf(0)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+2*cam_d(6)*uv_norm_kf(0)*uv_norm_kf(1)+cam_d(7)*(r_2+2*uv_norm_kf(0)*uv_norm_kf(0));
        double y1 = uv_norm_kf(1)*(1+cam_d(4)*r_2+cam_d(5)*r_4)+cam_d(6)*(r_2+2*uv_norm_kf(1)*uv_norm_kf(1))+2*cam_d(7)*uv_norm_kf(0)*uv_norm_kf(1);
        uv_dist_kf(0) = cam_d(0)*x1 + cam_d(2);
        uv_dist_kf(1) = cam_d(1)*y1 + cam_d(3);

    }
    return uv_dist_kf;
}



void  Simulator::triangulate_3dpoint(const Eigen::Matrix3d &R_CtoG,const Eigen::Vector3d &p_CinG,
                          const Eigen::Matrix3d &R_kftoG, const Eigen::Vector3d &p_kfinG,
                          const Eigen::Vector2f &uv_dist,const Eigen::Vector2f &uv_dist_kf,
                          Eigen::Vector3d &pt3d_turb)
{


}

