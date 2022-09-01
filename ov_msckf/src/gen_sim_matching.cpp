#include <csignal>

#include "sim/Simulator.h"
#include "core/VioManager.h"
#include "utils/dataset_reader.h"
#include "utils/parse_cmd.h"
#include "utils/colors.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>

#ifdef ROS_AVAILABLE
#include <ros/ros.h>
#include "core/RosVisualizer.h"
#include "utils/parse_ros.h"
#endif


using namespace ov_msckf;
using namespace std;

Simulator* sim;
VioManager* sys;
#ifdef ROS_AVAILABLE
RosVisualizer* viz;
#endif




// Define the function to be called when ctrl-c (SIGINT) is sent to process
void signal_callback_handler(int signum) {
    std::exit(signum);
}


// Main function
int main(int argc, char** argv)
{

    // Read in our parameters
    VioManagerOptions params;
#ifdef ROS_AVAILABLE
    ros::init(argc, argv, "test_simulation");
    ros::NodeHandle nh("~");
    params = parse_ros_nodehandler(nh);
#else
    params = parse_command_line_arguments(argc, argv);
#endif

    // Create our VIO system
    sim = new Simulator(params);
    sys = new VioManager(params);
#ifdef ROS_AVAILABLE
    viz = new RosVisualizer(nh, sys, sim);
#endif

    // string map_cam_pose_file=params.map_save_path+"gt_cam_20170823.txt";
    // string map_cam_pose_file_turb=params.map_save_path+"gt_cam_turb_ind_0.00025_20170823.txt";
    // string map_cam_pose_file=params.map_save_path+"urban38_cam_origin.txt";
    // string map_cam_pose_file_turb=params.map_save_path+"urban38_cam_origin_turb_ind_0.01_0.00025.txt";
    string map_cam_pose_file=params.map_save_path+"circle_20_traj_cam_map_20loop_horizonal.txt";
    // string map_cam_pose_file_turb=params.map_save_path+"circle_20_traj_cam_map.txt";
    // string map_cam_pose_file_turb=params.map_save_path+"circle_20_traj_cam_map_20loop_horizonal_turb_ind_0.01_0.00025.txt";
    string map_cam_pose_file_turb=params.map_save_path+"circle_20_traj_cam_map_20loop_horizonal.txt";
    string matching_file_name=params.map_save_path + params.pose_graph_filename;
    string save_matching_file_name="/tmp/sim_single_matching.txt";
    string save_multi_matching_file_name="/tmp/sim_multi_matching.txt";
   
    if(params.multi_match)
    {
        cout<<"in generate map multiple"<<endl;
        ifstream fi_matching;
        fi_matching.open(matching_file_name.data());
        assert(fi_matching.is_open());
        ofstream fo_matching_res;
        if (boost::filesystem::exists(save_multi_matching_file_name)) {
            boost::filesystem::remove(save_multi_matching_file_name);
            printf(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
        }
        // Create the directory that we will open the file in
        boost::filesystem::path p(save_multi_matching_file_name);
        boost::filesystem::create_directories(p.parent_path());
        // Open our statistics file!
        fo_matching_res.open(save_multi_matching_file_name, std::ofstream::out | std::ofstream::app);
        assert(fo_matching_res.is_open());

        string str,result;
        int line_num=0;
        double query_ts,kf_ts;
        string query_timestamp,kf_timestamp;
        int match_num=0;
        int match_num_per_img=0;
        int total_line_num_per_img=0;
        vector<string> matching_kfs_ts;
        vector<string> matching_kfs_name;
        vector<int> matching_matches_num;
        while (getline(fi_matching,str))
        {
            if(line_num==0)
            {
                stringstream line(str);
                cout<<str<<endl;
                line>>result; //query_timestamp
                query_timestamp=result;
                line>>result; //match_num_per_img
                match_num_per_img=stoi(result);
                total_line_num_per_img=2*match_num_per_img;
                line_num++;
                matching_kfs_ts.clear();
                matching_kfs_name.clear();
                matching_matches_num.clear();
                continue;
            }
            if(line_num<=total_line_num_per_img)
            {
                if(line_num%2==1)//query_timestamp, keyframe_timestamp, match_number
                {
                /***** for euroc *****/
                //     stringstream line(str);
                //     line>>result;
                // //    cout<<"***"<<result<<endl;
                //     string image_name1=result;
                //     string image_name_front;
                //     string image_name_back;
                //     image_name_front=image_name1.substr(0,10);
                //     image_name_front=image_name_front+".";
                //     image_name_back=image_name1.substr(10,4); //2位小数 forkaist 4 for euroc
                //     string image_name_final1=image_name_front+image_name_back;
                //     cout<<"image_name_final1: "<<image_name_final1<<endl;
                //     query_ts=stod(image_name_final1);
                    
                
                // // //    cout<<"***query_ts"<<to_string(query_ts)<<endl;
                //     line>>result;
                // //    cout<<"==="<<result<<endl;
                //     string image_name2=result;
                //     image_name_front=image_name2.substr(0,10);
                //     image_name_front=image_name_front+".";
                //     image_name_back=image_name2.substr(10,4);
                //     string image_name_final2=image_name_front+image_name_back;
                //     cout<<"image_name_final2: "<<image_name_final2<<endl;
                //     matching_kfs_ts.push_back(image_name_final2);
                //     matching_kfs_name.push_back(image_name2);
                //     kf_ts=stod(image_name_final2);
                // //    cout<<"***kf_ts"<<to_string(kf_ts)<<endl;
                //     line>>result;
                //     match_num=stoi(result);
                //     matching_matches_num.push_back(match_num);



                /**** for YQ ****/
//                 stringstream line(str);
//                 line>>result;
//                 query_timestamp=result;
//                 query_ts=stod(result);
//                 query_ts=round(query_ts*10000)/10000.0;
// //                   cout<<"***query_ts"<<to_string(query_ts)<<endl;
//                 line>>result;
//                 //    cout<<"==="<<result<<endl;
//                 matching_kfs_ts.push_back(result);
//                 kf_ts=stod(result);
//                 kf_ts=floor(kf_ts*10)/10.0;  //保留一位小数
// //                   cout<<"***kf_ts"<<to_string(kf_ts)<<endl;
//                 line>>result;
//                 match_num=stoi(result);
//                 matching_matches_num.push_back(match_num);

                /****** for eight *****/
                stringstream line(str);
                line>>result;
                query_timestamp=result;
                query_ts=stod(result);
                query_ts=round(query_ts*100)/100.0;
                  cout<<"***query_ts"<<to_string(query_ts)<<endl;
                line>>result;
                //    cout<<"==="<<result<<endl;
                matching_kfs_ts.push_back(result);
                kf_ts=stod(result);
                kf_ts=floor(kf_ts*100)/100.0;  //保留两位小数
                  cout<<"***kf_ts"<<to_string(kf_ts)<<endl;
                line>>result;
                match_num=stoi(result);
                matching_matches_num.push_back(match_num);
            
                }
                
                if(line_num==total_line_num_per_img)
                {
                    line_num=0;
                }
                else
                {
                    line_num++;
                }
                
            }
            if(line_num==0)  //this only happened when we traverse all matching information of the current query image
            {
                //now we get all mathcing kf timestamp, and we need to find their gt and current query image gt;

                //frist get current query image gt;
                
                Matrix3d R_GtoI;
                Matrix3d R_GtoI_turb;
                Vector3d p_IinG;
                Vector3d p_IinG_turb;
                bool found=false;
                found=sim->spline.get_pose(query_ts, R_GtoI, p_IinG);
                found=sim->spline_turb.get_pose(query_ts,R_GtoI_turb,p_IinG_turb);
               cout<<"found "<<found<<endl;
               cout<<"current pos: "<<p_IinG.transpose()<<endl<<"currentpos turb: "<<p_IinG_turb.transpose()<<endl;

//                sleep(3);
                // assert(found==true);
                if(found==false)
                   continue;
                Eigen::Matrix<double,3,3> R_ItoC = quat_2_Rot(params.camera_extrinsics.at(0).block(0,0,4,1));
                Eigen::Matrix<double,3,1> p_IinC = params.camera_extrinsics.at(0).block(4,0,3,1);
                Matrix3d R_CtoG=R_GtoI.transpose()*R_ItoC.transpose();
                Vector3d p_CinG=p_IinG-R_CtoG*p_IinC;
                Matrix3d R_CtoG_turb=R_GtoI_turb.transpose()*R_ItoC.transpose();
                Vector3d p_CinG_turb=p_IinG_turb-R_CtoG_turb*p_IinC;


                vector<string> kfs_ts_with_gt;
                vector<int> matches_num;
                vector<Matrix3d> kfs_pose_R;
                vector<Vector3d> kfs_pose_t;
                vector<Matrix3d> kfs_pose_R_turb;
                vector<Vector3d> kfs_pose_t_turb;
                sim->featmatching.clear();
                sim->featmatchingturb.clear();
                sim->id_matching=0;
                sim->uvsmatching.clear();
                //now get the all matching kf gt of the current query image
                for(int i=0;i<match_num_per_img;i++)
                {
                    double t=stod(matching_kfs_ts[i]);
                    // t=floor(t*10)/10.0;  //used for YQ
                    t=floor(t*100)/100.0; //used for eight
                     cout<<"match kfs ts i: "<<to_string(t)<<endl;
                    ifstream fi_kf_gt;
                    ifstream fi_kf_gt_turb;
                    fi_kf_gt.open(map_cam_pose_file.data());
                    fi_kf_gt_turb.open(map_cam_pose_file_turb.data());
                    assert(fi_kf_gt.is_open());
                    assert(fi_kf_gt_turb.is_open());
                    string str2,res2;
                    PoseJPL* kf_cam=new PoseJPL();
                    found=false;
                    while(getline(fi_kf_gt,str2))  //timestamp,tx,ty,tz,qx,qy,qz,qw
                    {
                        /**for Euroc**/
                        // stringstream line2(str2);
                        // line2>>res2;
                        // double ts=stod(res2);
                        // double timestp=ts;
                        // ts=floor(ts*10000)/10000.0; //保留2位小数for kaist 4 for euroc
                        // /**for YQ**/
                        // stringstream line2(str2);
                        // line2>>res2;
                        // double ts=stod(res2);
                        // double timestp=ts;
                        // ts=floor(ts*10)/10.0; //保留一位小数
                        /**for eight**/
                        stringstream line2(str2);
                        line2>>res2;
                        double ts=stod(res2);
                        double timestp=ts;
                        ts=floor(ts*100)/100.0; //保留两位小数
                        if(ts==t)
                        {
                            line2>>res2;
                            //  cout<<"tx read: "<<res1<<" ";
                            float tx=atof(res2.c_str());
                            //  cout<<"tx load: "<<tx<<endl;
                            line2>>res2;
                            //  cout<<"ty read: "<<res1<<" ";
                            float ty=atof(res2.c_str());
                            //  cout<<"ty load: "<<ty<<endl;
                            line2>>res2;
                            //  cout<<"tz read: "<<res1<<" ";
                            float tz=atof(res2.c_str());
                            //  cout<<"tz load: "<<tz<<endl;
                            line2>>res2;
                            //  cout<<"qx read: "<<res1<<" ";
                            float qx=atof(res2.c_str());
                            //  cout<<"qx load: "<< qx <<endl;
                            line2>>res2;
                            //  cout<<"qy read: "<<res1<<" ";
                            float qy=atof(res2.c_str());
                            //  cout<<"qy load: "<<qy<<endl;
                            line2>>res2;
                            //  cout<<"qz read: "<<res1<<" ";
                            float qz=atof(res2.c_str());
                            //  cout<<"qz load: "<<qz<<endl;
                            line2>>res2;
                            //  cout<<"qw read: "<<res1<<" ";
                            float qw=atof(res2.c_str());
                            //  cout<<"qw load: "<<qw<<endl;
                            //here we load the quaternion in the form of Hamilton,we need to tranform it into JPL
                            Quaterniond q1(qw,qx,qy,qz);  
                            Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                            Eigen::Matrix<double,7,1> q_kfInw;
                            q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                            kf_cam->set_value(q_kfInw);
                           cout<<"get kf pose"<<endl;
                            found=true;
                            break;
                        }

                    }
                    fi_kf_gt.close();
                    PoseJPL* kf_cam_turb=new PoseJPL();
                    bool found_2=false;
                    while(getline(fi_kf_gt_turb,str2))  //timestamp,tx,ty,tz,qx,qy,qz,qw
                    {
                         /**for Euroc**/
                        // stringstream line2(str2);
                        // line2>>res2;
                        // double ts=stod(res2);
                        // double timestp=ts;
                        // ts=floor(ts*10000)/10000.0; //保留2位小数for kaist 4 for euroc
                        /**for YQ**/
                        // stringstream line2(str2);
                        // line2>>res2;
                        // double ts=stod(res2);
                        // double timestp=ts;
                        // ts=floor(ts*10)/10.0; //保留一位小数
                        /**for eight**/
                        stringstream line2(str2);
                        line2>>res2;
                        double ts=stod(res2);
                        double timestp=ts;
                        ts=floor(ts*100)/100.0; //保留两位小数
                        if(ts==t)
                        {
                            line2>>res2;
                            //  cout<<"tx read: "<<res1<<" ";
                            float tx=atof(res2.c_str());
                            //  cout<<"tx load: "<<tx<<endl;
                            line2>>res2;
                            //  cout<<"ty read: "<<res1<<" ";
                            float ty=atof(res2.c_str());
                            //  cout<<"ty load: "<<ty<<endl;
                            line2>>res2;
                            //  cout<<"tz read: "<<res1<<" ";
                            float tz=atof(res2.c_str());
                            //  cout<<"tz load: "<<tz<<endl;
                            line2>>res2;
                            //  cout<<"qx read: "<<res1<<" ";
                            float qx=atof(res2.c_str());
                            //  cout<<"qx load: "<< qx <<endl;
                            line2>>res2;
                            //  cout<<"qy read: "<<res1<<" ";
                            float qy=atof(res2.c_str());
                            //  cout<<"qy load: "<<qy<<endl;
                            line2>>res2;
                            //  cout<<"qz read: "<<res1<<" ";
                            float qz=atof(res2.c_str());
                            //  cout<<"qz load: "<<qz<<endl;
                            line2>>res2;
                            //  cout<<"qw read: "<<res1<<" ";
                            float qw=atof(res2.c_str());
                            //  cout<<"qw load: "<<qw<<endl;
                            //here we load the quaternion in the form of Hamilton,we need to tranform it into JPL
                            Quaterniond q1(qw,qx,qy,qz);
                            Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                            Eigen::Matrix<double,7,1> q_kfInw;
                            q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                            kf_cam_turb->set_value(q_kfInw);
                           cout<<"get kf pose"<<endl;
                            found_2=true;
                            break;
                        }

                    }
                    fi_kf_gt_turb.close();
                    if(found==true&&found_2==true)
                    {
                      cout<<"found all kf pose"<<endl;
                        kfs_ts_with_gt.push_back(matching_kfs_ts[i]);  //for YQ and eight
                        // kfs_ts_with_gt.push_back(matching_kfs_name[i]); //for euroc
//                        if(matching_matches_num[i]<20)
//                        {
//
//                            std::uniform_real_distribution<double> gen_depth(20,50);
//                            int num= int(gen_depth(sim->gen_state_init));
//                            matches_num.push_back(num);
//                        }
//                        else
                            matches_num.push_back(matching_matches_num[i]);
                        kfs_pose_R.push_back(kf_cam->Rot());
                        kfs_pose_t.push_back(kf_cam->pos());
                        kfs_pose_R_turb.push_back(kf_cam_turb->Rot());
                        kfs_pose_t_turb.push_back(kf_cam_turb->pos());
                    }

                }

                //if num of kf is one, we skip it, as we cannot triangulate.
                if(kfs_ts_with_gt.size()==0||kfs_ts_with_gt.size()==1)
                {
                    continue;
                }
                assert(kfs_pose_R.size()==kfs_pose_t.size());
                assert(kfs_pose_R.size()==kfs_pose_R_turb.size());
                assert(kfs_pose_R.size()==kfs_pose_t_turb.size());
                assert(kfs_pose_R.size()==matches_num.size());
                assert(kfs_pose_R.size()==kfs_ts_with_gt.size());
                cout<<"num kf: "<<kfs_ts_with_gt.size()<<endl;

//                fo_matching_res<<query_timestamp<<" "<<to_string(kfs_ts_with_gt.size())<<endl;
                vector<string> res_kfs_ts;
                vector<int> res_match_num;
                vector<vector<Eigen::VectorXf>> res_uvs_cur;
                vector<vector<Eigen::VectorXf>> res_uvs_kf;
                vector<vector<Eigen::VectorXd>> res_pts3d_kf;

                sim->gen_matching(kfs_pose_R,kfs_pose_t,kfs_pose_R_turb,kfs_pose_t_turb,
                                  R_CtoG,p_CinG,matches_num,kfs_ts_with_gt,res_kfs_ts,
                                  res_uvs_cur,res_uvs_kf,res_pts3d_kf);

                if(res_kfs_ts.size()>0)
                    fo_matching_res<<query_timestamp<<" "<<to_string(res_kfs_ts.size())<<endl;
                for(int i=0;i<res_kfs_ts.size();i++)
                {
                    fo_matching_res<<query_timestamp<<" "<<res_kfs_ts[i]<<" "<<to_string(res_pts3d_kf[i].size())<<endl;
                    for(int j=0;j<res_pts3d_kf[i].size();j++)
                    {
                        fo_matching_res<<to_string(res_uvs_cur[i][j](0))<<" "<<to_string(res_uvs_cur[i][j](1))<<" ";
                        fo_matching_res<<to_string(res_uvs_kf[i][j](0))<<" "<<to_string(res_uvs_kf[i][j](1))<<" ";
                        fo_matching_res<<to_string(res_pts3d_kf[i][j](0))<<" "<<to_string(res_pts3d_kf[i][j](1))<<" "<<to_string(res_pts3d_kf[i][j](2))<<" ";
                    }
                    fo_matching_res<<endl;
                }





//                for(int i=0;i<kfs_ts_with_gt.size();i++)
//                {
//
//                    match_num=matches_num[i];
////                    fo_matching_res<<query_timestamp<<" "<<kfs_ts_with_gt[i]<<" "<<to_string(match_num)<<endl;
//                    Matrix3d R_kftoG=kfs_pose_R[i];
//                    Vector3d p_kfinG=kfs_pose_t[i];
//                    Matrix3d R_kftoG_turb=kfs_pose_R_turb[i];
//                    Vector3d p_kfinG_turb=kfs_pose_t_turb[i];
////                    cout<<"kf pos: "<<p_kfinG.transpose()<<endl<<"kf pos turb: "<<p_kfinG_turb.transpose()<<endl;
////                    sleep(3);
//                    vector<Eigen::VectorXf> uvs_cur;
//                    vector<Eigen::VectorXf> uvs_kf;
//                    vector<Eigen::VectorXd> pts3d_kf;
//
//
//                    sim->project_pointcloud_map(R_CtoG,p_CinG,R_CtoG_turb,p_CinG_turb,R_kftoG,p_kfinG,
//                            R_kftoG_turb,p_kfinG_turb,0,sim->featmatching,uvs_cur,uvs_kf,pts3d_kf);
//                    assert(uvs_cur.size()==uvs_kf.size());
//                    assert(uvs_cur.size()==pts3d_kf.size());
//                    bool success=false;
//                    if((int)uvs_cur.size()<match_num)
//                    {
//                        int numpts=match_num-(int)uvs_cur.size();
//                        success=sim->generate_points_map(R_CtoG,p_CinG,R_CtoG_turb,p_CinG_turb,R_kftoG,p_kfinG,
//                                                 R_kftoG_turb,p_kfinG_turb,0,sim->featmatching,numpts,uvs_cur,uvs_kf,pts3d_kf);
//                    }
//                    else
//                    {
//                        uvs_cur.erase(uvs_cur.begin()+match_num, uvs_cur.end());
//                        uvs_kf.erase(uvs_kf.begin()+match_num, uvs_kf.end());
//                        pts3d_kf.erase(pts3d_kf.begin()+match_num,pts3d_kf.end());
//                        success=true;
//                    }
//
//
//
//                    if(success==false)
//                        continue;
//                    else {
//
//                        assert(uvs_cur.size()==match_num);
//                        assert(uvs_kf.size()==match_num);
//                        assert(pts3d_kf.size()==match_num);
//                       res_kfs_ts.push_back(kfs_ts_with_gt[i]);
//                       res_match_num.push_back(uvs_cur.size());
//                       res_uvs_cur.push_back(uvs_cur);
//                       res_uvs_kf.push_back(uvs_kf);
//                       res_pts3d_kf.push_back(pts3d_kf);
//
//                    }
//
////                    for(int i=0;i<match_num;i++)
////                    {
////                        fo_matching_res<<to_string(uvs_cur[i](0))<<" "<<to_string(uvs_cur[i](1))<<" ";
////                        fo_matching_res<<to_string(uvs_kf[i](0))<<" "<<to_string(uvs_kf[i](1))<<" ";
////                        fo_matching_res<<to_string(pts3d_kf[i](0))<<" "<<to_string(pts3d_kf[i](1))<<" "<<to_string(pts3d_kf[i](2))<<" ";
////                    }
////                    fo_matching_res<<endl;
//
//                }
//
//                if(res_kfs_ts.size()<kfs_ts_with_gt.size())
//                {
//                    cout<<"res vs ori: "<<res_kfs_ts.size()<<" "<<kfs_ts_with_gt.size()<<endl;
//                    continue;
////                    sleep(2);
//                }
//                if(res_kfs_ts.size()>0)
//                    fo_matching_res<<query_timestamp<<" "<<to_string(res_kfs_ts.size())<<endl;
//                for(int i=0;i<res_kfs_ts.size();i++)
//                {
//                    fo_matching_res<<query_timestamp<<" "<<res_kfs_ts[i]<<" "<<to_string(res_match_num[i])<<endl;
//                    for(int j=0;j<res_match_num[i];j++)
//                    {
//                        fo_matching_res<<to_string(res_uvs_cur[i][j](0))<<" "<<to_string(res_uvs_cur[i][j](1))<<" ";
//                        fo_matching_res<<to_string(res_uvs_kf[i][j](0))<<" "<<to_string(res_uvs_kf[i][j](1))<<" ";
//                        fo_matching_res<<to_string(res_pts3d_kf[i][j](0))<<" "<<to_string(res_pts3d_kf[i][j](1))<<" "<<to_string(res_pts3d_kf[i][j](2))<<" ";
//                    }
//                    fo_matching_res<<endl;
//                }
            }
        }
        fi_matching.close();
        fo_matching_res.close();    
        
    }
    else
    {
        cout<<"in generate map single"<<endl;
        ifstream fi_matching;
        fi_matching.open(matching_file_name.data());
        assert(fi_matching.is_open());
        ofstream fo_matching_res;
        if (boost::filesystem::exists(save_matching_file_name)) {
            boost::filesystem::remove(save_matching_file_name);
            printf(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
        }
        // Create the directory that we will open the file in
        boost::filesystem::path p(save_matching_file_name);
        boost::filesystem::create_directories(p.parent_path());
        // Open our statistics file!
        fo_matching_res.open(save_matching_file_name, std::ofstream::out | std::ofstream::app);
        assert(fo_matching_res.is_open());

        string str,result;
        int line_num=0;
        double query_ts,kf_ts;
        string query_timestamp,kf_timestamp;
        int match_num=0;
        int match_num_per_img=0;
        int total_line_num_per_img=0;
        vector<string> matching_kfs_ts;
        vector<string> matching_kfs_name;
        vector<int> matching_matches_num;
        while (getline(fi_matching,str))
        {
            if(line_num==0)
            {
                stringstream line(str);
                line>>result; //query_timestamp
                cout<<str<<endl;
                query_timestamp=result;
                line>>result; //match_num_per_img
                match_num_per_img=stoi(result);
                total_line_num_per_img=2*match_num_per_img;
                line_num++;
                matching_kfs_ts.clear();
                matching_kfs_name.clear();
                matching_matches_num.clear();
                continue;
            }
            if(line_num<=total_line_num_per_img)
            {
                if(line_num%2==1)//query_timestamp, keyframe_timestamp, match_number
                {
               /***** for euroc *****/
                //     stringstream line(str);
                //     line>>result;
                //    cout<<"***"<<result<<endl;
                //     string image_name1=result;
                //     string image_name_front;
                //     string image_name_back;
                //     image_name_front=image_name1.substr(0,10);
                //     image_name_front=image_name_front+".";
                //     image_name_back=image_name1.substr(10,2); //2位小数 forkaist 4 for euroc
                //     string image_name_final1=image_name_front+image_name_back;
                //     cout<<"image_name_final1: "<<image_name_final1<<endl;
                //     query_ts=stod(image_name_final1);
                    
                
                // // //    cout<<"***query_ts"<<to_string(query_ts)<<endl;
                //     line>>result;
                // //    cout<<"==="<<result<<endl;
                //     string image_name2=result;
                //     image_name_front=image_name2.substr(0,10);
                //     image_name_front=image_name_front+".";
                //     image_name_back=image_name2.substr(10,2);
                //     string image_name_final2=image_name_front+image_name_back;
                //     cout<<"image_name_final2: "<<image_name_final2<<endl;
                //     matching_kfs_ts.push_back(image_name_final2);
                //     matching_kfs_name.push_back(image_name2);
                //     kf_ts=stod(image_name_final2);
                // //    cout<<"***kf_ts"<<to_string(kf_ts)<<endl;
                //     line>>result;
                //     match_num=stoi(result);
                //     matching_matches_num.push_back(match_num);



                /**** for YQ ****/
//                 stringstream line(str);
//                 line>>result;
//                 query_timestamp=result;
//                 query_ts=stod(result);
//                 query_ts=round(query_ts*10000)/10000.0;
// //                   cout<<"***query_ts"<<to_string(query_ts)<<endl;
//                 line>>result;
//                 //    cout<<"==="<<result<<endl;
//                 matching_kfs_ts.push_back(result);
//                 kf_ts=stod(result);
//                 kf_ts=floor(kf_ts*10)/10.0;  //保留一位小数
// //                   cout<<"***kf_ts"<<to_string(kf_ts)<<endl;
//                 line>>result;
//                 match_num=stoi(result);
//                 matching_matches_num.push_back(match_num);

                    /****** for eight *****/
                    stringstream line(str);
                    line>>result;
                    query_timestamp=result;
                    query_ts=stod(result);
                    query_ts=round(query_ts*100)/100.0;
    //                   cout<<"***query_ts"<<to_string(query_ts)<<endl;
                    line>>result;
                    //    cout<<"==="<<result<<endl;
                    matching_kfs_ts.push_back(result);
                    kf_ts=stod(result);
                    kf_ts=floor(kf_ts*100)/100.0;  //保留两位小数
    //                   cout<<"***kf_ts"<<to_string(kf_ts)<<endl;
                    line>>result;
                    match_num=stoi(result);
                    matching_matches_num.push_back(match_num);
                }
                
                if(line_num==total_line_num_per_img)
                {
                    line_num=0;
                }
                else
                {
                    line_num++;
                }
                
            }
            if(line_num==0)  //this only happened when we traverse all matching information of the current query image
            {
                //now we get all mathcing kf timestamp, and we need to find their gt and current query image gt;

                //frist get current query image gt;
                
                Matrix3d R_GtoI;
                Matrix3d R_GtoI_turb;
                Vector3d p_IinG;
                Vector3d p_IinG_turb;
                bool found=false;
                found=sim->spline.get_pose(query_ts, R_GtoI, p_IinG);
                found=sim->spline_turb.get_pose(query_ts,R_GtoI_turb,p_IinG_turb);
                cout<<"found pose: "<<found<<endl;
//                cout<<"current pos: "<<p_IinG.transpose()<<endl<<"currentpos turb: "<<p_IinG_turb.transpose()<<endl;
//                sleep(3);
                // assert(found==true);
                if(found==false)
                   continue;
                Eigen::Matrix<double,3,3> R_ItoC = quat_2_Rot(params.camera_extrinsics.at(0).block(0,0,4,1));
                Eigen::Matrix<double,3,1> p_IinC = params.camera_extrinsics.at(0).block(4,0,3,1);
                Matrix3d R_CtoG=R_GtoI.transpose()*R_ItoC.transpose();
                Vector3d p_CinG=p_IinG-R_CtoG*p_IinC;
                Matrix3d R_CtoG_turb=R_GtoI_turb.transpose()*R_ItoC.transpose();
                Vector3d p_CinG_turb=p_IinG_turb-R_CtoG_turb*p_IinC;


                vector<string> kfs_ts_with_gt;
                vector<int> matches_num;
                vector<Matrix3d> kfs_pose_R;
                vector<Vector3d> kfs_pose_t;
                vector<Matrix3d> kfs_pose_R_turb;
                vector<Vector3d> kfs_pose_t_turb;
                sim->featmatching.clear();
                sim->featmatchingturb.clear();
                sim->id_matching=0;
                sim->uvsmatching.clear();
                //now get the all matching kf gt of the current query image
                for(int i=0;i<match_num_per_img;i++)
                {
                    double t=stod(matching_kfs_ts[i]);
                    // t=floor(t*10)/10.0; //used for YQ
                    t=floor(t*100)/100.0; //used for eight
                    ifstream fi_kf_gt;
                    ifstream fi_kf_gt_turb;
                    fi_kf_gt.open(map_cam_pose_file.data());
                    fi_kf_gt_turb.open(map_cam_pose_file_turb.data());
                    assert(fi_kf_gt.is_open());
                    assert(fi_kf_gt_turb.is_open());
                    string str2,res2;
                    PoseJPL* kf_cam=new PoseJPL();
                    found=false;
                    while(getline(fi_kf_gt,str2))  //timestamp,tx,ty,tz,qx,qy,qz,qw
                    {
                          /**for Euroc**/
                        // stringstream line2(str2);
                        // line2>>res2;
                        // double ts=stod(res2);
                        // double timestp=ts;
                        // ts=floor(ts*100)/100.0; //保留2位小数for kaist 4 for euroc
                        // /**for YQ**/
                        // stringstream line2(str2);
                        // line2>>res2;
                        // double ts=stod(res2);
                        // double timestp=ts;
                        // ts=floor(ts*10)/10.0; //保留一位小数
                        /**for eight**/
                        stringstream line2(str2);
                        line2>>res2;
                        double ts=stod(res2);
                        double timestp=ts;
                        ts=floor(ts*100)/100.0; //保留两位小数
                        if(ts==t)
                        {
                            line2>>res2;
                            //  cout<<"tx read: "<<res1<<" ";
                            float tx=atof(res2.c_str());
                            //  cout<<"tx load: "<<tx<<endl;
                            line2>>res2;
                            //  cout<<"ty read: "<<res1<<" ";
                            float ty=atof(res2.c_str());
                            //  cout<<"ty load: "<<ty<<endl;
                            line2>>res2;
                            //  cout<<"tz read: "<<res1<<" ";
                            float tz=atof(res2.c_str());
                            //  cout<<"tz load: "<<tz<<endl;
                            line2>>res2;
                            //  cout<<"qx read: "<<res1<<" ";
                            float qx=atof(res2.c_str());
                            //  cout<<"qx load: "<< qx <<endl;
                            line2>>res2;
                            //  cout<<"qy read: "<<res1<<" ";
                            float qy=atof(res2.c_str());
                            //  cout<<"qy load: "<<qy<<endl;
                            line2>>res2;
                            //  cout<<"qz read: "<<res1<<" ";
                            float qz=atof(res2.c_str());
                            //  cout<<"qz load: "<<qz<<endl;
                            line2>>res2;
                            //  cout<<"qw read: "<<res1<<" ";
                            float qw=atof(res2.c_str());
                            //  cout<<"qw load: "<<qw<<endl;
                            //here we load the quaternion in the form of Hamilton,we need to tranform it into JPL
                            Quaterniond q1(qw,qx,qy,qz);  
                            Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                            Eigen::Matrix<double,7,1> q_kfInw;
                            q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                            kf_cam->set_value(q_kfInw);
//                            cout<<"get kf pose"<<endl;
                            found=true;
                            break;
                        }

                    }
                    fi_kf_gt.close();
                    PoseJPL* kf_cam_turb=new PoseJPL();
                    bool found_2=false;
                    while(getline(fi_kf_gt_turb,str2))  //timestamp,tx,ty,tz,qx,qy,qz,qw
                    {
                         // /**for Euroc**/
                        // stringstream line2(str2);
                        // line2>>res2;
                        // double ts=stod(res2);
                        // double timestp=ts;
                        // ts=floor(ts*100)/100.0; //保留2位小数for kaist 4 for euroc
                        /**for YQ**/
                        // stringstream line2(str2);
                        // line2>>res2;
                        // double ts=stod(res2);
                        // double timestp=ts;
                        // ts=floor(ts*10)/10.0; //保留一位小数
                        /**for eight**/
                        stringstream line2(str2);
                        line2>>res2;
                        double ts=stod(res2);
                        double timestp=ts;
                        ts=floor(ts*100)/100.0; //保留两位小数
                        if(ts==t)
                        {
                            line2>>res2;
                            //  cout<<"tx read: "<<res1<<" ";
                            float tx=atof(res2.c_str());
                            //  cout<<"tx load: "<<tx<<endl;
                            line2>>res2;
                            //  cout<<"ty read: "<<res1<<" ";
                            float ty=atof(res2.c_str());
                            //  cout<<"ty load: "<<ty<<endl;
                            line2>>res2;
                            //  cout<<"tz read: "<<res1<<" ";
                            float tz=atof(res2.c_str());
                            //  cout<<"tz load: "<<tz<<endl;
                            line2>>res2;
                            //  cout<<"qx read: "<<res1<<" ";
                            float qx=atof(res2.c_str());
                            //  cout<<"qx load: "<< qx <<endl;
                            line2>>res2;
                            //  cout<<"qy read: "<<res1<<" ";
                            float qy=atof(res2.c_str());
                            //  cout<<"qy load: "<<qy<<endl;
                            line2>>res2;
                            //  cout<<"qz read: "<<res1<<" ";
                            float qz=atof(res2.c_str());
                            //  cout<<"qz load: "<<qz<<endl;
                            line2>>res2;
                            //  cout<<"qw read: "<<res1<<" ";
                            float qw=atof(res2.c_str());
                            //  cout<<"qw load: "<<qw<<endl;
                            //here we load the quaternion in the form of Hamilton,we need to tranform it into JPL
                            Quaterniond q1(qw,qx,qy,qz);
                            Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                            Eigen::Matrix<double,7,1> q_kfInw;
                            q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                            kf_cam_turb->set_value(q_kfInw);
//                            cout<<"get kf pose"<<endl;
                            found_2=true;
                            break;
                        }

                    }
                    fi_kf_gt_turb.close();
                    if(found==true&&found_2==true)
                    {
                        kfs_ts_with_gt.push_back(matching_kfs_ts[i]);  //for YQ and eight
                        // kfs_ts_with_gt.push_back(matching_kfs_name[i]); //for euroc
//                        if(matching_matches_num[i]<20)
//                        {
//
//                            std::uniform_real_distribution<double> gen_depth(20,50);
//                            int num= int(gen_depth(sim->gen_state_init));
//                            matches_num.push_back(num);
//                        }
//                        else
                        matches_num.push_back(matching_matches_num[i]);
                        kfs_pose_R.push_back(kf_cam->Rot());
                        kfs_pose_t.push_back(kf_cam->pos());
                        kfs_pose_R_turb.push_back(kf_cam_turb->Rot());
                        kfs_pose_t_turb.push_back(kf_cam_turb->pos());
                    }

                }

                //if num of kf is one, we skip it, as we cannot triangulate.
                if(kfs_ts_with_gt.size()==0||kfs_ts_with_gt.size()==1)
                {
                    continue;
                }
                assert(kfs_pose_R.size()==kfs_pose_t.size());
                assert(kfs_pose_R.size()==kfs_pose_R_turb.size());
                assert(kfs_pose_R.size()==kfs_pose_t_turb.size());
                assert(kfs_pose_R.size()==matches_num.size());
                assert(kfs_pose_R.size()==kfs_ts_with_gt.size());
                cout<<"num kf: "<<kfs_ts_with_gt.size()<<endl;

//                fo_matching_res<<query_timestamp<<" "<<to_string(kfs_ts_with_gt.size())<<endl;
                vector<string> res_kfs_ts;
                vector<int> res_match_num;
                vector<vector<Eigen::VectorXf>> res_uvs_cur;
                vector<vector<Eigen::VectorXf>> res_uvs_kf;
                vector<vector<Eigen::VectorXd>> res_pts3d_kf;

                sim->gen_matching(kfs_pose_R,kfs_pose_t,kfs_pose_R_turb,kfs_pose_t_turb,
                                  R_CtoG,p_CinG,matches_num,kfs_ts_with_gt,res_kfs_ts,
                                  res_uvs_cur,res_uvs_kf,res_pts3d_kf);

                
                if(!res_kfs_ts.empty())
                {
                    cout<<"ref_kfs_ts size:"<< res_kfs_ts.size()<<endl;
                    int index=-1;
                    int max=-1;
                    for(int i=0;i<res_kfs_ts.size();i++)
                    {
                         cout<<res_pts3d_kf[i].size()<<endl;
                         if(int(res_pts3d_kf[i].size())>max)
                         {
                             max=res_pts3d_kf[i].size();
                             index=i;
                         }
                    }
                    cout<<"index: "<<index<<endl;
                    assert(index>=0);
                    int i=index;
                    int upper_limit = min(int(res_pts3d_kf[i].size()),30);
                   fo_matching_res<<query_timestamp<<" "<<res_kfs_ts[index]<<" "<<to_string(upper_limit)<<endl;
                    for(int j=0;j<upper_limit;j++)
                    {
                        fo_matching_res<<to_string(res_uvs_cur[i][j](0))<<" "<<to_string(res_uvs_cur[i][j](1))<<" ";
                        fo_matching_res<<to_string(res_uvs_kf[i][j](0))<<" "<<to_string(res_uvs_kf[i][j](1))<<" ";
                        fo_matching_res<<to_string(res_pts3d_kf[i][j](0))<<" "<<to_string(res_pts3d_kf[i][j](1))<<" "<<to_string(res_pts3d_kf[i][j](2))<<" ";
                    }
                    fo_matching_res<<endl;
                }
                 
            }
        }
        fi_matching.close();
        fo_matching_res.close();    

    }
//     else
//     {   
//         cout<<"in generate map single"<<endl;
//         ifstream fi_matching;
//         fi_matching.open(matching_file_name.data());
//         assert(fi_matching.is_open());
//         ofstream fo_matching_res;
//         if (boost::filesystem::exists(save_matching_file_name)) {
//             boost::filesystem::remove(save_matching_file_name);
//             printf(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
//         }
//         // Create the directory that we will open the file in
//         boost::filesystem::path p(save_matching_file_name);
//         boost::filesystem::create_directories(p.parent_path());
//         // Open our statistics file!
//         fo_matching_res.open(save_matching_file_name, std::ofstream::out | std::ofstream::app);
//         assert(fo_matching_res.is_open());
//         string str,result;
//         int line_num=1;
//         double query_ts,kf_ts;
//         string query_timestamp,kf_timestamp;
//         int match_num=0;
//         while (getline(fi_matching,str))
//         {
//             if(line_num%2==1)
//             {
//                 stringstream line(str);
//                 line>>result;
//                    cout<<"***"<<result<<endl;
                
//                 query_ts=stod(result);
//                 query_timestamp=result;
//                 query_ts=round(query_ts*10000)/10000.0;
//                 // query_ts=floor(query_ts*10)/10.0;
//                    cout<<"***query_ts"<<to_string(query_ts)<<endl;
//                 line>>result;
                
//                 //    cout<<"==="<<result<<endl;
//                 kf_ts=stod(result);
//                 kf_ts=floor(kf_ts*10)/10.0;  //保留一位小数
//                    cout<<"***kf_ts"<<to_string(kf_ts)<<endl;
//                 kf_timestamp=result;
//                 line>>result;
                
//                 match_num=stoi(result);
                
//             }
//             else if(line_num%2==0)
//             {
//                 sim->featmatching.clear();
//                 sim->featmatchingturb.clear();
//                 sim->id_matching=0;
//                 sim->uvsmatching.clear();
//                 //first we need to get the groundtruth pose of query_pose and matching_pose;
//                 PoseJPL* query_cam=new PoseJPL();
                
//                 Matrix3d R_GtoI,R_GtoI_turb;
//                 Vector3d p_IinG,p_IinG_turb;
//                 bool found=false;
//                 found=sim->spline.get_pose(query_ts, R_GtoI, p_IinG);
//                 found=sim->spline_turb.get_pose(query_ts, R_GtoI_turb, p_IinG_turb);
//                 // assert(found==true);
//                 if(found==false)
//                    break;
//                 Eigen::Matrix<double,3,3> R_ItoC = quat_2_Rot(params.camera_extrinsics.at(0).block(0,0,4,1));
//                 Eigen::Matrix<double,3,1> p_IinC = params.camera_extrinsics.at(0).block(4,0,3,1);
//                 Matrix3d R_CtoG=R_GtoI.transpose()*R_ItoC.transpose();
//                 Vector3d p_CinG=p_IinG-R_CtoG*p_IinC;
//                 Matrix3d R_CtoG_turb=R_GtoI_turb.transpose()*R_ItoC.transpose();
//                 Vector3d p_CinG_turb=p_IinG_turb-R_CtoG_turb*p_IinC;

//                 ifstream fi_kf_gt;
//                 ifstream fi_kf_gt_turb;
//                 fi_kf_gt.open(map_cam_pose_file.data());
//                 fi_kf_gt_turb.open(map_cam_pose_file_turb.data());
//                 assert(fi_kf_gt.is_open());
//                 assert(fi_kf_gt_turb.is_open());
//                 string str2,res2;
//                 PoseJPL* kf_cam=new PoseJPL();
//                 found=false;
//                 while(getline(fi_kf_gt,str2))  //timestamp,tx,ty,tz,qx,qy,qz,qw
//                 {
//                     stringstream line2(str2);
//                     // line1>>res1;
//                     // double ts=stod(res1);
//                     // double timestp=ts;
//                     // ts=floor(ts*10000)/10000.0; //保留2位小数for kaist 4 for euroc
//                     // /**for YQ**/
//                     line2>>res2;
//                     double ts=stod(res2);
//                     double timestp=ts;
//                     ts=floor(ts*10)/10.0; //保留一位小数
//                     if(ts==kf_ts)
//                     {
//                         line2>>res2;
//                         //  cout<<"tx read: "<<res1<<" ";
//                         float tx=atof(res2.c_str());
//                         //  cout<<"tx load: "<<tx<<endl;
//                         line2>>res2;
//                         //  cout<<"ty read: "<<res1<<" ";
//                         float ty=atof(res2.c_str());
//                         //  cout<<"ty load: "<<ty<<endl;
//                         line2>>res2;
//                         //  cout<<"tz read: "<<res1<<" ";
//                         float tz=atof(res2.c_str());
//                         //  cout<<"tz load: "<<tz<<endl;
//                         line2>>res2;
//                         //  cout<<"qx read: "<<res1<<" ";
//                         float qx=atof(res2.c_str());
//                         //  cout<<"qx load: "<< qx <<endl;
//                         line2>>res2;
//                         //  cout<<"qy read: "<<res1<<" ";
//                         float qy=atof(res2.c_str());
//                         //  cout<<"qy load: "<<qy<<endl;
//                         line2>>res2;
//                         //  cout<<"qz read: "<<res1<<" ";
//                         float qz=atof(res2.c_str());
//                         //  cout<<"qz load: "<<qz<<endl;
//                         line2>>res2;
//                         //  cout<<"qw read: "<<res1<<" ";
//                         float qw=atof(res2.c_str());
//                         //  cout<<"qw load: "<<qw<<endl;
//                         //here we load the quaternion in the form of Hamilton,we need to tranform it into JPL
//                         Quaterniond q1(qw,qx,qy,qz);  
//                         Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
//                         Eigen::Matrix<double,7,1> q_kfInw;
//                         q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
//                         kf_cam->set_value(q_kfInw);
//                         cout<<"get kf pose"<<endl;
//                         found=true;
//                         break;
//                     }

//                 }
//                 fi_kf_gt.close();
//                 PoseJPL* kf_cam_turb=new PoseJPL();
//                 bool found_2=false;
//                 while(getline(fi_kf_gt_turb,str2))  //timestamp,tx,ty,tz,qx,qy,qz,qw
//                 {
//                     stringstream line2(str2);
//                     // line1>>res1;
//                     // double ts=stod(res1);
//                     // double timestp=ts;
//                     // ts=floor(ts*10000)/10000.0; //保留2位小数for kaist 4 for euroc
//                     // /**for YQ**/
//                     line2>>res2;
//                     double ts=stod(res2);
//                     double timestp=ts;
//                     ts=floor(ts*10)/10.0; //保留一位小数
//                     if(ts==kf_ts)
//                     {
//                         line2>>res2;
//                         //  cout<<"tx read: "<<res1<<" ";
//                         float tx=atof(res2.c_str());
//                         //  cout<<"tx load: "<<tx<<endl;
//                         line2>>res2;
//                         //  cout<<"ty read: "<<res1<<" ";
//                         float ty=atof(res2.c_str());
//                         //  cout<<"ty load: "<<ty<<endl;
//                         line2>>res2;
//                         //  cout<<"tz read: "<<res1<<" ";
//                         float tz=atof(res2.c_str());
//                         //  cout<<"tz load: "<<tz<<endl;
//                         line2>>res2;
//                         //  cout<<"qx read: "<<res1<<" ";
//                         float qx=atof(res2.c_str());
//                         //  cout<<"qx load: "<< qx <<endl;
//                         line2>>res2;
//                         //  cout<<"qy read: "<<res1<<" ";
//                         float qy=atof(res2.c_str());
//                         //  cout<<"qy load: "<<qy<<endl;
//                         line2>>res2;
//                         //  cout<<"qz read: "<<res1<<" ";
//                         float qz=atof(res2.c_str());
//                         //  cout<<"qz load: "<<qz<<endl;
//                         line2>>res2;
//                         //  cout<<"qw read: "<<res1<<" ";
//                         float qw=atof(res2.c_str());
//                         //  cout<<"qw load: "<<qw<<endl;
//                         //here we load the quaternion in the form of Hamilton,we need to tranform it into JPL
//                         Quaterniond q1(qw,qx,qy,qz);
//                         Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
//                         Eigen::Matrix<double,7,1> q_kfInw;
//                         q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
//                         kf_cam_turb->set_value(q_kfInw);
//                         cout<<"get kf pose"<<endl;
//                         found_2=true;
//                         break;
//                     }

//                 }
//                 fi_kf_gt_turb.close();
//                 if(found==false||found_2==false)
//                 {
//                     line_num++;
//                     continue;
//                 }
                
// //                if(match_num<20)
// //                {
// //                    std::uniform_real_distribution<double> gen_depth(20,50);
// //                    match_num= int(gen_depth(sim->gen_state_init));
// //                }
                
// //                fo_matching_res<<query_timestamp<<" "<<kf_timestamp<<" "<<to_string(match_num)<<endl;
//                 Matrix3d R_kftoG=kf_cam->Rot();
//                 Vector3d p_kfinG=kf_cam->pos();
//                 Matrix3d R_kftoG_turb=kf_cam_turb->Rot();
//                 Vector3d p_kfinG_turb=kf_cam_turb->pos();
//                 vector<Eigen::VectorXf> uvs_cur;
//                 vector<Eigen::VectorXf> uvs_kf;
//                 vector<Eigen::VectorXd> pts3d_kf;

                
//                 sim->project_pointcloud_map(R_CtoG,p_CinG,R_CtoG_turb,p_CinG_turb,
//                                             R_kftoG,p_kfinG,R_kftoG_turb,p_kfinG_turb,0,sim->featmatching,uvs_cur,uvs_kf,pts3d_kf);
//                 assert(uvs_cur.size()==uvs_kf.size());
//                 assert(uvs_cur.size()==pts3d_kf.size());
//                 bool success=false;
//                 if((int)uvs_cur.size()<match_num)
//                 {
//                     int numpts=match_num-(int)uvs_cur.size();
//                     success=sim->generate_points_map(R_CtoG,p_CinG,R_CtoG_turb,p_CinG_turb,R_kftoG,p_kfinG,R_kftoG_turb,p_kfinG_turb,0,sim->featmatching,numpts,uvs_cur,uvs_kf,pts3d_kf);
//                 }
//                 else
//                 {
//                     uvs_cur.erase(uvs_cur.begin()+match_num, uvs_cur.end());
//                     uvs_kf.erase(uvs_kf.begin()+match_num, uvs_kf.end());
//                     pts3d_kf.erase(pts3d_kf.begin()+match_num,pts3d_kf.end());
//                     success=true;
//                 }


//                 if(success) {
//                     assert(uvs_cur.size() == match_num);
//                     assert(uvs_kf.size() == match_num);
//                     assert(pts3d_kf.size() == match_num);
//                     fo_matching_res<<query_timestamp<<" "<<kf_timestamp<<" "<<to_string(match_num)<<endl;
//                     for (int i = 0; i < match_num; i++) {
//                         fo_matching_res << to_string(uvs_cur[i](0)) << " " << to_string(uvs_cur[i](1)) << " ";
//                         fo_matching_res << to_string(uvs_kf[i](0)) << " " << to_string(uvs_kf[i](1)) << " ";
//                         fo_matching_res << to_string(pts3d_kf[i](0)) << " " << to_string(pts3d_kf[i](1)) << " "
//                                         << to_string(pts3d_kf[i](2)) << " ";
//                     }
//                     fo_matching_res << endl;
//                 }
//             }
//             line_num++;
//         }
//         fi_matching.close();
//         fo_matching_res.close();
//     }
    cout<<"num of gen matching points: "<<sim->featmatching.size()<<endl;
    ros::Publisher pub_sim_matching;
    pub_sim_matching = nh.advertise<sensor_msgs::PointCloud2>("/ov_msckf/points_matching_map", 2);
    ROS_INFO("Publishing: %s", pub_sim_matching.getTopic().c_str());

    
    
    
    
    
   
    //===================================================================================
    //===================================================================================
    //===================================================================================
// Step through the rosbag


//      if(params.use_prior_map)
//    {
//        if(params.multi_match)
//             sys->loadPoseGraph3(params.map_save_path,params.pose_graph_filename,params.keyframe_pose_filename);
//        else
//            sys->loadPoseGraph2(params.map_save_path,params.pose_graph_filename,params.keyframe_pose_filename);

//    }
//    cout<<"finish load Posegraph "<<endl;

//    int sim_feats_num;
//    sim_feats_num=sim->get_sim_feats_num();
//    sys->trackFEATS->set_sim_feats_num(sim_feats_num);
//    sys->sim_feats_num=sim_feats_num;
//    cout<<"sim feature number: "<<sim_feats_num<<endl;
//     //===================================================================================
//     //===================================================================================
//     //===================================================================================

//     // Get initial state
//     Eigen::Matrix<double, 17, 1> imustate;
//     bool success = sim->get_state(sim->current_timestamp(),imustate);
//     if(!success) {
//         printf(RED "[SIM]: Could not initialize the filter to the first state\n" RESET);
//         printf(RED "[SIM]: Did the simulator load properly???\n" RESET);
//         std::exit(EXIT_FAILURE);
//     }

//     // Since the state time is in the camera frame of reference
//     // Subtract out the imu to camera time offset
//     imustate(0,0) -= sim->get_true_paramters().calib_camimu_dt;

//     // Initialize our filter with the groundtruth
//     sys->initialize_with_gt(imustate);

//     //===================================================================================
//     //===================================================================================
//     //===================================================================================

//     // Buffer our camera image
//     double buffer_timecam = -1;
//     std::vector<int> buffer_camids;
//     std::vector<std::vector<std::pair<size_t,Eigen::VectorXf>>> buffer_feats;

//     // Step through the rosbag
//     signal(SIGINT, signal_callback_handler);
// #ifdef ROS_AVAILABLE
//     while(sim->ok() && ros::ok()) {
// #else
//     while(sim->ok()) {
// #endif
//         std::unordered_map<size_t,Eigen::Vector3d> feats_sim_matching = sim->get_matching_map();

//     // Declare message and sizes
//     sensor_msgs::PointCloud2 cloud_SIM_gen;
//     cloud_SIM_gen.header.frame_id = "global";
//     cloud_SIM_gen.header.stamp = ros::Time::now();
//     cloud_SIM_gen.width  = 3*feats_sim_matching.size();
//     cloud_SIM_gen.height = 1;
//     cloud_SIM_gen.is_bigendian = false;
//     cloud_SIM_gen.is_dense = false; // there may be invalid points

//     // Setup pointcloud fields
//     sensor_msgs::PointCloud2Modifier modifier_SIM_gen(cloud_SIM_gen);
//     modifier_SIM_gen.setPointCloud2FieldsByString(1,"xyz");
//     modifier_SIM_gen.resize(3*feats_sim_matching.size());

//     // Iterators
//     sensor_msgs::PointCloud2Iterator<float> out_x_SIM_gen(cloud_SIM_gen, "x");
//     sensor_msgs::PointCloud2Iterator<float> out_y_SIM_gen(cloud_SIM_gen, "y");
//     sensor_msgs::PointCloud2Iterator<float> out_z_SIM_gen(cloud_SIM_gen, "z");

//     // Fill our iterators
//     for(const auto &pt : feats_sim_matching) {
//         *out_x_SIM_gen = pt.second(0); ++out_x_SIM_gen;
//         *out_y_SIM_gen = pt.second(1); ++out_y_SIM_gen;
//         *out_z_SIM_gen = pt.second(2); ++out_z_SIM_gen;
//     }

//     // Publish
//     pub_sim_matching.publish(cloud_SIM_gen);




//         // IMU: get the next simulated IMU measurement if we have it
//         double time_imu;
//         Eigen::Vector3d wm, am;
//         bool hasimu = sim->get_next_imu(time_imu, wm, am);
//         if(hasimu) {
//             sys->feed_measurement_imu(time_imu, wm, am);
// #ifdef ROS_AVAILABLE
//             viz->visualize_odometry(time_imu);
// #endif
//         }
//         // CAM: get the next simulated camera uv measurements if we have them
//         double time_cam;
//         std::vector<int> camids;
//         std::vector<std::vector<std::pair<size_t,Eigen::VectorXf>>> feats;
//         bool hascam = sim->get_next_cam(time_cam, camids, feats);
//         if(hascam) {
//             if(buffer_timecam != -1) {
//                 sys->feed_measurement_simulation(buffer_timecam, buffer_camids, buffer_feats);
// #ifdef ROS_AVAILABLE
//                 viz->visualize();
// #endif
//             }
//             buffer_timecam = time_cam;
//             buffer_camids = camids;
//             buffer_feats = feats;
//         }

//     }


//     //===================================================================================
//     //===================================================================================
//     //===================================================================================


//     // Final visualization
// #ifdef ROS_AVAILABLE
//     viz->visualize_final();
//     delete viz;
// #endif

//     // Finally delete our system
//     delete sim;
//     delete sys;

    // Done!
    return EXIT_SUCCESS;


}
