#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "core/RosVisualizer.h"
#include "utils/dataset_reader.h"
#include "utils/parse_ros.h"
#include <dirent.h>
#include <unistd.h>

using namespace ov_msckf;
using namespace cv;
using namespace std;

class Feat{
     
     public:
        Feat():success_triang(false){}
    
        ~Feat(){}
     
     //map<image_timstamp, (id,u,v)>
        map<double,Vector3d> uvs;

        map<double,Vector3d> uvs_norm;

        size_t id;

        Vector3d point_3d_W;

        Vector3d point_3d_A;

        double anchor_img_id=-1;

        bool success_triang;

};

VioManager* sys;
// Buffer data
double time_buffer = -1;
cv::Mat img0_buffer, img1_buffer;


double fx=458.654/2;
double fy=457.296/2;
double cx=367.215/2;
double cy=248.375/2;
double k1=-5.6143027800000002e-02;
double k2=1.3952563200000001e-01;
double k3=-1.2155906999999999e-03;
double k4=-9.7281389999999998e-04;

void callback_monocular(const sensor_msgs::ImageConstPtr& msg0);
//void callback_stereo(const sensor_msgs::CompressedImageConstPtr& msg0, const sensor_msgs::CompressedImageConstPtr& msg1);
void callback_stereo(const sensor_msgs::ImageConstPtr& msg0, const sensor_msgs::ImageConstPtr& msg1);

void callback_cam(const sensor_msgs::ImageConstPtr& msg);

bool single_triangulation(Feat* feat, map<double, ov_type::PoseJPL *>& image_poses);

bool single_gaussnewton(Feat* feat, map<double, ov_type::PoseJPL *>& image_poses);

double compute_error(map<double, ov_type::PoseJPL *>& image_poses,
                     Feat* feat, double alpha, double beta, double rho);




vector<string> getFiles(string cate_dir)
{
	vector<string> files;//存放文件名
	DIR *dir;
	struct dirent *ptr;
	char base[1000];
    cout<<"in getFiles"<<endl;
 
	if ((dir=opendir(cate_dir.c_str())) == NULL)
        {
		perror("Open dir error...");
                exit(1);
        }
 
	while ((ptr=readdir(dir)) != NULL)
	{
		if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
		        continue;
		else if(ptr->d_type == 8)    ///file
			//printf("d_name:%s/%s\n",basePath,ptr->d_name);
			files.push_back(ptr->d_name);
		else if(ptr->d_type == 10)    ///link file
			//printf("d_name:%s/%s\n",basePath,ptr->d_name);
			continue;
		else if(ptr->d_type == 4)    ///dir
		{
			files.push_back(ptr->d_name);
			/*
		        memset(base,'\0',sizeof(base));
		        strcpy(base,basePath);
		        strcat(base,"/");
		        strcat(base,ptr->d_nSame);
		        readFileList(base);
			*/
		}
	}
	closedir(dir);
 
	//排序，按从小到大排序
	sort(files.begin(), files.end());
	return files;
}


// Main function
int main(int argc, char** argv)
{
      
    ros::init(argc, argv, "test_kf_match");
    ros::NodeHandle nh("~");
    //Create our VIO system
    VioManagerOptions params = parse_ros_nodehandler(nh);
    cout<<"finish load parameters.............."<<endl;
    string filepath="/home/zzq/Code/map_open_vins_ekf_ros/src/open_vins/triangulate_match/JYM_test_20210421/matches/";
    string filepath_3d="/home/zzq/Code/map_open_vins_ekf_ros/src/open_vins/triangulate_match/JYM_test_20210421/matches_3d/";
    // string filepath_imgname="/home/zzq/Code/map_open_vins_ros/src/open_vins/triangulate_match/leftImageTimestamp.txt";
    vector<string> files;
    files=getFiles(filepath);
    cout<<"files size: "<<files.size()<<endl;
    for(int i=0;i<files.size();i++)
    {
        cout<<files[i]<<endl;
    }   

    // //load each file
    vector<int> error_load;
    for(int file_num=0;file_num<files.size();file_num++)
    {

        size_t feature_num=0;
        //feature database
        
        map<size_t,Feat*> feats_db;
        //every image(timestamp) and it corresponding Feature Vector4d(localid,u,v,globalid)
        map<double,vector<Vector4d>> images_feature;

        // sys = new VioManager(params);

        cout<<"finish initialization!..........."<<endl;
        
        ifstream if_match;
        if_match.open((filepath+files[file_num]).data());
        assert(if_match.is_open());
        cout<<"processing "<<files[file_num]<<"....."<<endl;
         
        ofstream of_3d;
        if (boost::filesystem::exists(filepath_3d+files[file_num])) {
            boost::filesystem::remove(filepath_3d+files[file_num]);
            printf(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
        }
        // Create the directory that we will open the file in
        boost::filesystem::path p(filepath_3d+files[file_num]);
        boost::filesystem::create_directories(p.parent_path());


        //image1 timestamp with vector contains matched images. For each matched images, map the timestamp and matched points
        map<double,vector<map<double,vector<VectorXd>>>> image_matches;
        map<double,PoseJPL*> image_poses;

        map<double,string> image_name_map;

        string str,result;
        int line_number=1;
        ifstream if_pose;
        string pose_file_name="/home/zzq/Code/map_open_vins_ekf_ros/src/open_vins/ov_data/euroc_mav/MH_01_easy_cam.txt";
        ifstream if_imgname;
        double timestamp1,timestamp2;
        int matches_num;
        vector<VectorXd> matched_point;
        string image_name1,image_name2;
        bool success_load=true;
        bool skip=false;
        
        while(getline(if_match,str))
        {
         if(line_number%2==1) //单数行  imagename1,imagename2,match number
         {
             stringstream line(str);
            line>>result; //image name1;
            image_name1=result;  //0000000010.png for example
            string image_name_front;
            string image_name_back;
            image_name_front=image_name1.substr(0,10);
            image_name_front=image_name_front+".";
            image_name_back=image_name1.substr(10,4); //两位小数 for kaist, 4 for euroc
            string image_name_final1=image_name_front+image_name_back;
             //cout<<"image_name_final1: "<<image_name_final1<<endl;
            timestamp1=stod(image_name_final1);

            //  if_imgname.open(filepath_imgname.data());
            //  assert(if_imgname.is_open());
            //  string str_name,res_name;
            //  while(getline(if_imgname,str_name))
            //  {
            //      stringstream line_name(str_name);
            //      line_name>>res_name;  //10 for example
            //      int size_res_name=res_name.size();
            //      for(int count=0;count<10-size_res_name;count++)
            //      {
            //          res_name="0"+res_name;
            //      }
            //      res_name=res_name+".png";
            //      assert(res_name.size()==14);
            //      string image_name=res_name;
            //      if(image_name==image_name1)
            //      {
            //          line_name>>res_name;
            //          timestamp1=stod(res_name);
            //          break;
            //      }
            //  }
            //  if_imgname.close();

              cout<<"timestamp1: "<<to_string(timestamp1)<<" image_name1: "<<image_name1<<endl;
            image_name_map.insert({timestamp1,image_name1});

            line>>result; //image_name2
            image_name2=result;
            image_name_front=image_name2.substr(0,10);
            image_name_front=image_name_front+".";
            image_name_back=image_name2.substr(10,4);
            string image_name_final2=image_name_front+image_name_back;
            // cout<<"image_name_final2: "<<image_name_final2<<endl;
            timestamp2=stod(image_name_final2);
            // if_imgname.open(filepath_imgname.data());
            //  assert(if_imgname.is_open());
            //  //string str_name,res_name;
            //  while(getline(if_imgname,str_name))
            //  {
            //      stringstream line_name(str_name);
            //      line_name>>res_name;  //10 for example
            //      int size_res_name=res_name.size();
            //      for(int count=0;count<10-size_res_name;count++)
            //      {
            //          res_name="0"+res_name;
            //      }
            //      res_name=res_name+".png";
            //      assert(res_name.size()==14);
            //      string image_name=res_name;
            //      if(image_name==image_name2)
            //      {
            //          line_name>>res_name;
            //          timestamp2=stod(res_name);
            //          break;
            //      }
            //  }
            //  if_imgname.close();

               cout<<"timestamp2: "<<to_string(timestamp2)<<" iamge_name2: "<<image_name2<<endl;
            
            image_name_map.insert({timestamp2,image_name2});

            line>>result;
            matches_num=stoi(result);
            vector<map<double,vector<VectorXd>>> matched_image;
           
            if(image_poses.find(timestamp1)==image_poses.end())//load pose
            {
                if_pose.open(pose_file_name.data());
                assert(if_pose.is_open());
                string str1,res1;
                float tx,ty,tz,qx,qy,qz,qw;
                int flag=0;
                while(getline(if_pose,str1))
                {
                    stringstream line1(str1);
                    line1>>res1;
                    double timestamp=stod(res1); //timestamp;
                    timestamp= floor(timestamp*10000)/10000.0; //保留2位小数for kaist,4 for euroc
                    string timestamp_str=to_string(timestamp).substr(0,15); //13forkaist,15foreuroc
                    //    cout<<"timestamp,image_name_final: "<<timestamp_str<<" "<<image_name_final1<<endl;
                    
                    if(timestamp_str==image_name_final1)
                    {
                        line1>>res1;
                        //  cout<<"tx read: "<<res1<<" ";
                        tx=atof(res1.c_str());
                        //  cout<<"tx load: "<<tx<<endl;
                        line1>>res1;
                        //  cout<<"ty read: "<<res1<<" ";
                        ty=atof(res1.c_str());
                        //  cout<<"ty load: "<<ty<<endl;
                        line1>>res1;
                        //  cout<<"tz read: "<<res1<<" ";
                        tz=atof(res1.c_str());
                        //  cout<<"tz load: "<<tz<<endl;
                        line1>>res1;
                        //  cout<<"qx read: "<<res1<<" ";
                        qx=atof(res1.c_str());
                        //  cout<<"qx load: "<< qx <<endl;
                        line1>>res1;
                        //  cout<<"qy read: "<<res1<<" ";
                        qy=atof(res1.c_str());
                        //  cout<<"qy load: "<<qy<<endl;
                        line1>>res1;
                        //  cout<<"qz read: "<<res1<<" ";
                        qz=atof(res1.c_str());
                        //  cout<<"qz load: "<<qz<<endl;
                        line1>>res1;
                        //  cout<<"qw read: "<<res1<<" ";
                        qw=atof(res1.c_str());

                        Quaterniond q1(qw,qx,qy,qz);  
                        Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                        Eigen::Matrix<double,7,1> q_kfInw;
                        q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                        PoseJPL* pose=new PoseJPL();
                        pose->set_value(q_kfInw);
                        image_poses.insert({timestamp1,pose});
                        flag=1;
                        break;
                    }

                    // timestamp = floor(timestamp*10)/10.0;
                    // double approx_ts1=floor(timestamp1*10)/10.0;
                    // double approx_ts1_2=floor(timestamp1*10)+1;
                    // approx_ts1_2=approx_ts1_2/10.0;
                    // if(approx_ts1==timestamp)  //find the closest pose
                    // {
                    //     line1>>res1;
                    //     //  cout<<"tx read: "<<res1<<" ";
                    //     tx=atof(res1.c_str());
                    //     //  cout<<"tx load: "<<tx<<endl;
                    //     line1>>res1;
                    //     //  cout<<"ty read: "<<res1<<" ";
                    //     ty=atof(res1.c_str());
                    //     //  cout<<"ty load: "<<ty<<endl;
                    //     line1>>res1;
                    //     //  cout<<"tz read: "<<res1<<" ";
                    //     tz=atof(res1.c_str());
                    //     //  cout<<"tz load: "<<tz<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qx read: "<<res1<<" ";
                    //     qx=atof(res1.c_str());
                    //     //  cout<<"qx load: "<< qx <<endl;
                    //     line1>>res1;
                    //     //  cout<<"qy read: "<<res1<<" ";
                    //     qy=atof(res1.c_str());
                    //     //  cout<<"qy load: "<<qy<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qz read: "<<res1<<" ";
                    //     qz=atof(res1.c_str());
                    //     //  cout<<"qz load: "<<qz<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qw read: "<<res1<<" ";
                    //     qw=atof(res1.c_str());

                    //     Quaterniond q1(qw,qx,qy,qz);  
                    //     Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                    //     Eigen::Matrix<double,7,1> q_kfInw;
                    //     q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                    //     PoseJPL* pose=new PoseJPL();
                    //     pose->set_value(q_kfInw);
                    //     image_poses.insert({timestamp1,pose});
                    //     flag=1;
                    //     break;
                    // }
                    // else if(approx_ts1_2==timestamp)
                    // {
                    //     line1>>res1;
                    //     //  cout<<"tx read: "<<res1<<" ";
                    //     tx=atof(res1.c_str());
                    //     //  cout<<"tx load: "<<tx<<endl;
                    //     line1>>res1;
                    //     //  cout<<"ty read: "<<res1<<" ";
                    //     ty=atof(res1.c_str());
                    //     //  cout<<"ty load: "<<ty<<endl;
                    //     line1>>res1;
                    //     //  cout<<"tz read: "<<res1<<" ";
                    //     tz=atof(res1.c_str());
                    //     //  cout<<"tz load: "<<tz<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qx read: "<<res1<<" ";
                    //     qx=atof(res1.c_str());
                    //     //  cout<<"qx load: "<< qx <<endl;
                    //     line1>>res1;
                    //     //  cout<<"qy read: "<<res1<<" ";
                    //     qy=atof(res1.c_str());
                    //     //  cout<<"qy load: "<<qy<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qz read: "<<res1<<" ";
                    //     qz=atof(res1.c_str());
                    //     //  cout<<"qz load: "<<qz<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qw read: "<<res1<<" ";
                    //     qw=atof(res1.c_str());

                    //     Quaterniond q1(qw,qx,qy,qz);  
                    //     Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                    //     Eigen::Matrix<double,7,1> q_kfInw;
                    //     q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                    //     PoseJPL* pose=new PoseJPL();
                    //     pose->set_value(q_kfInw);
                    //     image_poses.insert({timestamp1,pose});
                    //     flag=1;
                    //     break;
                    // }
                }
                if_pose.close();
                // if(flag!=1)
                // {
                //     if_pose.open(pose_file_name.data());
                //     assert(if_pose.is_open());
                //         while(getline(if_pose,str1))
                //     {
                //         stringstream line1(str1);
                //         line1>>res1;
                //         double timestamp=stod(res1); //timestamp;
                //         timestamp = floor(timestamp*10)/10.0;
                //         // timestamp= floor(timestamp*10000)/10000.0; //保留4位小数
                //         // string timestamp_str=to_string(timestamp).substr(0,15);
                //         //    cout<<"timestamp,image_name_final: "<<timestamp_str<<" "<<image_name_final<<endl;
                //         // if(timestamp_str==image_name_final1)
                //         double approx_ts1=floor(timestamp1*10)/10.0;
                //         double approx_ts1_3=floor(timestamp1*10)-1;
                //         approx_ts1_3=approx_ts1_3/10.0;
                //         if(approx_ts1_3==timestamp)
                //         {
                //             line1>>res1;
                //             //  cout<<"tx read: "<<res1<<" ";
                //             tx=atof(res1.c_str());
                //             //  cout<<"tx load: "<<tx<<endl;
                //             line1>>res1;
                //             //  cout<<"ty read: "<<res1<<" ";
                //             ty=atof(res1.c_str());
                //             //  cout<<"ty load: "<<ty<<endl;
                //             line1>>res1;
                //             //  cout<<"tz read: "<<res1<<" ";
                //             tz=atof(res1.c_str());
                //             //  cout<<"tz load: "<<tz<<endl;
                //             line1>>res1;
                //             //  cout<<"qx read: "<<res1<<" ";
                //             qx=atof(res1.c_str());
                //             //  cout<<"qx load: "<< qx <<endl;
                //             line1>>res1;
                //             //  cout<<"qy read: "<<res1<<" ";
                //             qy=atof(res1.c_str());
                //             //  cout<<"qy load: "<<qy<<endl;
                //             line1>>res1;
                //             //  cout<<"qz read: "<<res1<<" ";
                //             qz=atof(res1.c_str());
                //             //  cout<<"qz load: "<<qz<<endl;
                //             line1>>res1;
                //             //  cout<<"qw read: "<<res1<<" ";
                //             qw=atof(res1.c_str());

                //             Quaterniond q1(qw,qx,qy,qz);  
                //             Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                //             Eigen::Matrix<double,7,1> q_kfInw;
                //             q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                //             PoseJPL* pose=new PoseJPL();
                //             pose->set_value(q_kfInw);
                //             image_poses.insert({timestamp1,pose});
                //             flag=1;
                //             break;
                //         }
                //     }
                // }
                // if_pose.close();

                if(flag!=1)
                {
                    success_load=false;
                    break;
                }
                assert(flag==1);
            }
            if(image_poses.find(timestamp2)==image_poses.end())//load pose
            {
                if_pose.open(pose_file_name.data());
                assert(if_pose.is_open());
                string str1,res1;
                float tx,ty,tz,qx,qy,qz,qw;
                int flag=0;
                while(getline(if_pose,str1))
                {
                    stringstream line1(str1);
                    line1>>res1;
                    double timestamp=stod(res1); //timestamp;
                    
                    timestamp= floor(timestamp*10000)/10000.0; //保留4位小数
                    string timestamp_str=to_string(timestamp).substr(0,15);
                    //    cout<<"timestamp,image_name_final: "<<timestamp_str<<" "<<image_name_final2<<endl;
                    if(timestamp_str==image_name_final2)
                    {
                        line1>>res1;
                        //  cout<<"tx read: "<<res1<<" ";
                        tx=atof(res1.c_str());
                        //  cout<<"tx load: "<<tx<<endl;
                        line1>>res1;
                        //  cout<<"ty read: "<<res1<<" ";
                        ty=atof(res1.c_str());
                        //  cout<<"ty load: "<<ty<<endl;
                        line1>>res1;
                        //  cout<<"tz read: "<<res1<<" ";
                        tz=atof(res1.c_str());
                        //  cout<<"tz load: "<<tz<<endl;
                        line1>>res1;
                        //  cout<<"qx read: "<<res1<<" ";
                        qx=atof(res1.c_str());
                        //  cout<<"qx load: "<< qx <<endl;
                        line1>>res1;
                        //  cout<<"qy read: "<<res1<<" ";
                        qy=atof(res1.c_str());
                        //  cout<<"qy load: "<<qy<<endl;
                        line1>>res1;
                        //  cout<<"qz read: "<<res1<<" ";
                        qz=atof(res1.c_str());
                        //  cout<<"qz load: "<<qz<<endl;
                        line1>>res1;
                        //  cout<<"qw read: "<<res1<<" ";
                        qw=atof(res1.c_str());

                        Quaterniond q1(qw,qx,qy,qz);  
                        Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                        Eigen::Matrix<double,7,1> q_kfInw;
                        q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                        PoseJPL* pose=new PoseJPL();
                        pose->set_value(q_kfInw);
                        image_poses.insert({timestamp2,pose});
                        flag=1;
                        break;  
                    }

                    // timestamp = floor(timestamp*10)/10.0;
                    // double approx_ts2=floor(timestamp2*10)/10.0;
                    // double approx_ts2_2=floor(timestamp2*10)+1;
                    // approx_ts2_2=approx_ts2_2/10.0;
                    // //cout<<"approx_ts2: "<<to_string(approx_ts2)<<" approx_ts2_2: "<<to_string(approx_ts2_2)<<" timestamp: "<<to_string(timestamp)<<endl;
                    // if(to_string(approx_ts2)==to_string(timestamp))
                    // {
                    //     line1>>res1;
                    //     //  cout<<"tx read: "<<res1<<" ";
                    //     tx=atof(res1.c_str());
                    //     //  cout<<"tx load: "<<tx<<endl;
                    //     line1>>res1;
                    //     //  cout<<"ty read: "<<res1<<" ";
                    //     ty=atof(res1.c_str());
                    //     //  cout<<"ty load: "<<ty<<endl;
                    //     line1>>res1;
                    //     //  cout<<"tz read: "<<res1<<" ";
                    //     tz=atof(res1.c_str());
                    //     //  cout<<"tz load: "<<tz<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qx read: "<<res1<<" ";
                    //     qx=atof(res1.c_str());
                    //     //  cout<<"qx load: "<< qx <<endl;
                    //     line1>>res1;
                    //     //  cout<<"qy read: "<<res1<<" ";
                    //     qy=atof(res1.c_str());
                    //     //  cout<<"qy load: "<<qy<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qz read: "<<res1<<" ";
                    //     qz=atof(res1.c_str());
                    //     //  cout<<"qz load: "<<qz<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qw read: "<<res1<<" ";
                    //     qw=atof(res1.c_str());

                    //     Quaterniond q1(qw,qx,qy,qz);  
                    //     Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                    //     Eigen::Matrix<double,7,1> q_kfInw;
                    //     q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                    //     PoseJPL* pose=new PoseJPL();
                    //     pose->set_value(q_kfInw);
                    //     image_poses.insert({timestamp2,pose});
                    //     flag=1;
                    //     break;
                    // }
                    // else if(to_string(approx_ts2_2)==to_string(timestamp))
                    // {
                    //     line1>>res1;
                    //     //  cout<<"tx read: "<<res1<<" ";
                    //     tx=atof(res1.c_str());
                    //     //  cout<<"tx load: "<<tx<<endl;
                    //     line1>>res1;
                    //     //  cout<<"ty read: "<<res1<<" ";
                    //     ty=atof(res1.c_str());
                    //     //  cout<<"ty load: "<<ty<<endl;
                    //     line1>>res1;
                    //     //  cout<<"tz read: "<<res1<<" ";
                    //     tz=atof(res1.c_str());
                    //     //  cout<<"tz load: "<<tz<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qx read: "<<res1<<" ";
                    //     qx=atof(res1.c_str());
                    //     //  cout<<"qx load: "<< qx <<endl;
                    //     line1>>res1;
                    //     //  cout<<"qy read: "<<res1<<" ";
                    //     qy=atof(res1.c_str());
                    //     //  cout<<"qy load: "<<qy<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qz read: "<<res1<<" ";
                    //     qz=atof(res1.c_str());
                    //     //  cout<<"qz load: "<<qz<<endl;
                    //     line1>>res1;
                    //     //  cout<<"qw read: "<<res1<<" ";
                    //     qw=atof(res1.c_str());

                    //     Quaterniond q1(qw,qx,qy,qz);  
                    //     Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                    //     Eigen::Matrix<double,7,1> q_kfInw;
                    //     q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                    //     PoseJPL* pose=new PoseJPL();
                    //     pose->set_value(q_kfInw);
                    //     image_poses.insert({timestamp2,pose});
                    //     flag=1;
                    //     break;
                    // }
                }
                if_pose.close();
                // if(flag!=1)
                // {
                //     if_pose.open(pose_file_name.data());
                //     assert(if_pose.is_open());
                //         while(getline(if_pose,str1))
                //     {
                //         stringstream line1(str1);
                //         line1>>res1;
                //         double timestamp=stod(res1); //timestamp;
                //         timestamp = floor(timestamp*10)/10.0;
                //         // timestamp= floor(timestamp*10000)/10000.0; //保留4位小数
                //         // string timestamp_str=to_string(timestamp).substr(0,15);
                //         //    cout<<"timestamp,image_name_final: "<<timestamp_str<<" "<<image_name_final<<endl;
                //         // if(timestamp_str==image_name_final1)
                //         double approx_ts2=floor(timestamp2*10)/10.0;
                //         double approx_ts2_3=floor(timestamp2*10)-1;
                //         approx_ts2_3=approx_ts2_3/10.0;
                //         if(approx_ts2_3==timestamp)
                //         {
                //             line1>>res1;
                //             //  cout<<"tx read: "<<res1<<" ";
                //             tx=atof(res1.c_str());
                //             //  cout<<"tx load: "<<tx<<endl;
                //             line1>>res1;
                //             //  cout<<"ty read: "<<res1<<" ";
                //             ty=atof(res1.c_str());
                //             //  cout<<"ty load: "<<ty<<endl;
                //             line1>>res1;
                //             //  cout<<"tz read: "<<res1<<" ";
                //             tz=atof(res1.c_str());
                //             //  cout<<"tz load: "<<tz<<endl;
                //             line1>>res1;
                //             //  cout<<"qx read: "<<res1<<" ";
                //             qx=atof(res1.c_str());
                //             //  cout<<"qx load: "<< qx <<endl;
                //             line1>>res1;
                //             //  cout<<"qy read: "<<res1<<" ";
                //             qy=atof(res1.c_str());
                //             //  cout<<"qy load: "<<qy<<endl;
                //             line1>>res1;
                //             //  cout<<"qz read: "<<res1<<" ";
                //             qz=atof(res1.c_str());
                //             //  cout<<"qz load: "<<qz<<endl;
                //             line1>>res1;
                //             //  cout<<"qw read: "<<res1<<" ";
                //             qw=atof(res1.c_str());

                //             Quaterniond q1(qw,qx,qy,qz);  
                //             Quaterniond q1_JPL(q1.toRotationMatrix().transpose());
                //             Eigen::Matrix<double,7,1> q_kfInw;
                //             q_kfInw<<q1_JPL.x(),q1_JPL.y(),q1_JPL.z(),q1_JPL.w(),tx, ty, tz;
                //             PoseJPL* pose=new PoseJPL();
                //             pose->set_value(q_kfInw);
                //             image_poses.insert({timestamp2,pose});
                //             flag=1;
                //             break;
                //         }
                //     }
                // }
                // if_pose.close();
                if(flag!=1)
                {
                    success_load=false;
                    break;
                }
                assert(flag==1);
            }
            if(success_load==false)
            {
                error_load.push_back(file_num);
                //boost::filesystem::remove(filepath_3d+files[file_num]);
                line_number++;
                skip=true;
                continue;
            }
            else
            {
                image_name_map.insert({timestamp1,image_name1});
                image_name_map.insert({timestamp2,image_name2});
                if(image_matches.find(timestamp1)==image_matches.end())
                {
                    image_matches.insert({timestamp1,matched_image});
                }
                vector<Vector4d> points;
                if(images_feature.find(timestamp1)==images_feature.end())
                {
                    images_feature.insert({timestamp1,points});
                }
                if(images_feature.find(timestamp2)==images_feature.end())
                {
                    images_feature.insert({timestamp2,points});
                }
            }
            

         }
         else if(line_number%2==0)//偶数行
         {
           if(skip==true)
           {
               line_number++;
               continue;
           }
           stringstream line(str);
           matched_point.clear();
           for(int j=0;j<matches_num;j++)
           {
               
               line>>result; //id1;
               double id1=stod(result);
               line>>result;
               double uv1_x=stod(result);
               line>>result;
               double uv1_y=stod(result);
               line>>result;
               double id2=stod(result);
               line>>result;
               double uv2_x=stod(result);
               line>>result;
               double uv2_y=stod(result);
               Matrix<double,7,1> match;
               match<<id1,uv1_x,uv1_y,id2,uv2_x,uv2_y,-1;  //-1 means this match is not used yet
            //    cout<<match.transpose()<<" ";
               matched_point.push_back(match);
           }
        //    string image_dir="/media/zzq/SAMSUNG/DataSet/kaist/urban38/urban38-pankyo_img/urban38-pankyo/image/stereo_left/";
        //    Mat image1=imread(image_dir+image_name1,0);
        //    Mat image2=imread(image_dir+image_name2,0);
        // //    cout<<"after image read"<<endl;
        // //    imshow("image1",image1);
        //    waitKey(0);
        //    vector<cv::KeyPoint> kps1;
        //    vector<cv::KeyPoint> kps2;
        //    for(int j=0;j<matched_point.size();j++)
        //    {
        //        cv::KeyPoint kp1;
        //        cv::Point2d pt1;
        //        pt1.x=matched_point[j](1,0);
        //        pt1.y=matched_point[j](2,0);
        //        kp1.pt=pt1;
        //        kps1.push_back(kp1);
        //        cv::KeyPoint kp2;
        //        cv::Point2d pt2;
        //        double col=image1.cols;
        //        pt2.x=matched_point[j](4,0)+col;
        //        pt2.y=matched_point[j](5,0);
        //        kp2.pt=pt2;
        //        kps2.push_back(kp2);
        //    }
        //    cv::Mat pair_image;
        //    cv::hconcat(image1,image2,pair_image);
        //     cv::Mat pair_image_color;
        //     cvtColor(pair_image,pair_image_color,cv::COLOR_GRAY2BGR);
        //     // cout<<"after convert color"<<endl;
        //     for(int j=0;j<kps1.size();j++)
        //     {

                    
        //         cv::circle(pair_image_color,kps1[j].pt,2,cv::Scalar(255,0,0));
        //         cv::circle(pair_image_color,kps2[j].pt,2,cv::Scalar(0,255,255));
        //         cv::line(pair_image_color,kps1[j].pt,kps2[j].pt,cv::Scalar(0,255,0),1);
        //         // cv::imshow("pair",pair_image_color);
        //     }
        //     cv::imshow("pair",pair_image_color);
        //     cv::waitKey();  
        //     // cout<<"out"<<endl;          


           map<double,vector<VectorXd>> m;
           m.insert({timestamp2,matched_point});
           assert(image_matches.find(timestamp1)!=image_matches.end());
           image_matches.at(timestamp1).push_back(m);
         }
         
         
         line_number++;
        }
        if_match.close();
        

        // cout<<"image_matches size:"<<image_matches.size()<<endl;
        // for(auto image: image_matches)
        // {
        //     cout<<"for image "<<to_string(image.first)<<" ";
        //     cout<<"match image_num is "<<image.second.size()<<endl;
        // }
        // for(auto pose:image_poses)
        // {
        //     cout<<"for image "<<to_string(pose.first)<<" ";
        //     cout<<"pose is "<<pose.second->quat().transpose()<<" "<<pose.second->pos().transpose()<<endl;
        // }

        int num=0;
        for(auto image: image_matches)
        {
            // cout<<"for image "<<to_string(image.first)<<endl;
            double root_image=image.first;
            auto matches_root=image.second;
            //image_root<-->image_i image_root<-->image_i+1 image_root<-->image_i+1 .... 
            for(int i=0;i<matches_root.size();i++)
            {
              auto m=matches_root[i];
              for(auto pair: m) //for every matched image with root image
              {
                  double match_image_ts=pair.first;
                  cout<<"image "<<to_string(root_image)<<" match with image "<<to_string(match_image_ts)<<endl;
                  vector<VectorXd> match_points=pair.second;
                  for(int j=0;j<match_points.size();j++) //for each match point
                  {
                      Vector3d pt1;
                      Vector3d pt2;
                    //   if(match_points[j](6,0)!=-1)
                    //   {
                    //       //this point match has been used 
                    //       continue;
                    //   }
                      pt1<<match_points[j](0,0),match_points[j](1,0),match_points[j](2,0);
                      pt2<<match_points[j](3,0),match_points[j](4,0),match_points[j](5,0);
                      assert(images_feature.find(root_image)!=images_feature.end());
                      assert(images_feature.find(match_image_ts)!=images_feature.end());
                      bool flag=false;
                      for(int k=0;k<images_feature[root_image].size();k++)
                      {
                          Vector3d point=images_feature[root_image][k].block(0,0,3,1);

                          if(images_feature[root_image][k].block(0,0,3,1)==pt1) //the featrue pt1 has already in images_feature[root_image]
                          {
                              flag=true;
                              size_t id=images_feature[root_image][k](3,0);
                              assert(feats_db.find(id)!=feats_db.end());
                              auto feat=feats_db.find(id);
                              feat->second->uvs.insert({match_image_ts,pt2});
                              Vector4d p=Vector4d::Zero();
                              p<<pt2(0),pt2(1),pt2(2),double(id);
                              images_feature[match_image_ts].push_back(p);

                              break;
                          }

                      }
                      if(flag==false)  //the feature pt1 is not in images_feature[root_image]
                      {
                          
                          bool mark=false;
                          for(int k=0;k<images_feature[match_image_ts].size();k++)
                          {
                              if(images_feature[match_image_ts][k].block(0,0,3,1)==pt2)
                              {
                                  //although feature pt1 is not in root_image, pt2 has already in match_image_ts, 
                                  //which means this feature has matches before.
                                  //For example: for landmark1, it was observed by image1 and image3 before (when root image is image1),
                                  //             now, our root image is image2, we find that this landmark1 is new for image2, however, 
                                  //             it is already observed in image3 before, then we should not create the new feat.
                                  //             instead, we need to make feature_link with image1,image2,image3.  
                                  size_t id=images_feature[match_image_ts][k](3,0);
                                  assert(feats_db.find(id)!=feats_db.end());
                                  auto feat=feats_db.find(id);
                                  feat->second->uvs.insert({root_image,pt1});
                                  Vector4d p=Vector4d::Zero();
                                  p<<pt1(0),pt1(1),pt1(2),double(id);
                                  images_feature[root_image].push_back(p);
                                  mark=true;
                                  break;
                              }
                          }
                          if(mark==false) //assign a new feature
                          {
                            Feat* feat=new Feat();
                            feat->id=feature_num;
                            feat->uvs.insert({root_image,pt1});
                            feat->uvs.insert({match_image_ts,pt2});
                            Vector4d p=Vector4d::Zero();
                            p<<pt1(0),pt1(1),pt1(2),double(feat->id);
                            images_feature[root_image].push_back(p);
                            p=Vector4d::Zero();
                            p<<pt2(0),pt2(1),pt2(2),double(feat->id);
                            images_feature[match_image_ts].push_back(p);
                            feats_db.insert({feature_num,feat});
                            feature_num++;
                          }
                          

                      }

                  }
              }

            }


        }
        cout<<"feature database size: "<<feats_db.size()<<endl;
        // for(auto img: images_feature)
        // {
        //     double timestamp=img.first;
        //     cout<<"image "<<to_string(timestamp)<<" feature size: "<<images_feature[timestamp].size()<<endl;
        // }
        
    //     for(auto feat:feats_db)
    //    {
    //         cout<<"size: "<<feat.second->uvs.size()<<" ";
    //         // for(auto img: feat.second->uvs)
    //         // {
    //         //     cout<<to_string(img.first)<<" ";
    //         // }
    //    }
       cv::Matx33d camK;
        camK(0, 0) = fx;
        camK(0,1)=0;
        camK(0,2)=cx;
        camK(1,0)=0;
        camK(1,1)=fy;
        camK(1,2)=cy;
        camK(2,0)=0;
        camK(2,1)=0;
        camK(2,2)=1;
       cv::Vec4d camD;
        camD(0) = k1;
        camD(1) = k2;
        camD(2) = k3;
        camD(3) = k4;
       //undistort uv point to get uv_norm
       for(auto feat:feats_db)
       {
        //    cout<<"for feat "<<feat.first<<": ";
           
           for(auto uv:feat.second->uvs)
           {
              cv::Point2f pt;
              pt.x=float(uv.second(1));
              pt.y=float(uv.second(2));
              cv::Mat mat(1, 2, CV_32F);
              mat.at<float>(0, 0) = pt.x;
              mat.at<float>(0, 1) = pt.y;
            
              mat = mat.reshape(2); // Nx1, 2-channel
            // Undistort it!
            // cv::undistortPoints(mat, mat, camK, camD);
            cv::fisheye::undistortPoints(mat, mat, camK, camD);
            // Construct our return vector
            cv::Point2f pt_out;
            mat = mat.reshape(1); // Nx2, 1-channel
            pt_out.x = mat.at<float>(0, 0);
            pt_out.y = mat.at<float>(0, 1);
            Vector3d uv_norm(uv.second(0),double(pt_out.x),double(pt_out.y));
            feat.second->uvs_norm.insert({uv.first,uv_norm});
            // cout<<"("<<uv_norm(1)<<" "<<uv_norm(2)<<") ";
            
            
           }
        //    cout<<endl;

       }

       string image_dir="/media/zzq/SAMSUNG/DataSet/kaist/urban38/urban38-pankyo_img/urban38-pankyo/image/stereo_left/";
    //    for(auto feat:feats_db)
    //    {
    //        Feat* f=feat.second;
    //        vector<Point2d> pts;
    //        vector<Mat> images;
    //        for(auto uv:f->uvs)
    //        {
    //            string image_name=image_name_map[uv.first];
    //            Mat image=imread(image_dir+image_name,0);
    //            images.push_back(image.clone());
    //            Point2d pt;
    //            pt.x=uv.second(1);
    //            pt.y=uv.second(2);
    //            pts.push_back(pt);
    //        }
    //        cv::Mat hconcat_image;
    //        cv::hconcat(images,hconcat_image);
    //        cv::Mat pair_image_color;
    //        cvtColor(hconcat_image,pair_image_color,cv::COLOR_GRAY2BGR);
    //     //    imshow("image",pair_image_color);
    //     //    waitKey();
    //        for(int i=0;i<pts.size();i++)
    //        {
    //            double col=images[i].cols;
    //            pts[i].x=pts[i].x+i*col;
    //            cv::circle(pair_image_color,pts[i],8,cv::Scalar(0,0,255));
    //        }
    //        for(int i=1;i<pts.size();i++)
    //        {
    //            cv::line(pair_image_color,pts[i-1],pts[i],cv::Scalar(0,255,0),1);
    //        }
    //        imshow("image",pair_image_color);
    //        waitKey();
    //     //    break;
    //    }
        
        //Now we have feature database with linked feature.
        int success_num=0;
        for(auto feat: feats_db)
        {
            bool success=single_triangulation(feat.second,image_poses);
            if(!success)
            {
               
               continue;
            }
            success=single_gaussnewton(feat.second,image_poses);
            if(!success)
            {
                
                continue;
            }
            feat.second->success_triang=true;
            success_num++;
            
        }
        cout<<"feature_num: "<<feats_db.size()<<endl;
        cout<<"success_num: "<<success_num<<endl;

        
        //check reproject error
        int success_filter=success_num;
        for(auto feat: feats_db)
        {
                Feat* f=feat.second;
                if(f->success_triang)
                {
                //    cout<<"for feat "<<to_string(feat.first)<<": ";
                   Vector3d pt_w=f->point_3d_W;
                    for(auto uv: f->uvs)
                    {
                        double image_id=uv.first;
                        Matrix3d R_W_C=image_poses.at(image_id)->Rot();
                        Vector3d p_W_C=image_poses.at(image_id)->pos();
                        Vector3d pt_c=R_W_C.transpose()*(pt_w-p_W_C);
                        // cout<<"pt_c: "<<pt_c.transpose()<<" pt_w: "<<pt_w.transpose()<<" ";
                        pt_c(0)=pt_c(0)/pt_c(2);
                        pt_c(1)=pt_c(1)/pt_c(2);
                        pt_c(2)=pt_c(2)/pt_c(2);

                        // Calculate distorted coordinates for fisheye
                        double r = std::sqrt(pt_c(0)*pt_c(0)+pt_c(1)*pt_c(1));
                        double theta = std::atan(r);
                        double theta_d = theta+k1*std::pow(theta,3)+k2*std::pow(theta,5)+k3*std::pow(theta,7)+k4*std::pow(theta,9);

                        // Handle when r is small (meaning our xy is near the camera center)
                        double inv_r = (r > 1e-8)? 1.0/r : 1.0;
                        double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

                        // Calculate distorted coordinates for fisheye
                        double x1 = pt_c(0)*cdist;
                        double y1 = pt_c(1)*cdist;
                        pt_c(0) = fx*x1 + cx;
                        pt_c(1) = fy*y1 + cy;
                        // cout<<"uv: "<<pt_c(0)<<" "<<pt_c(1)<<endl;
                        
                        // // Calculate distorted coordinates for pinhole 
                        // double r = std::sqrt(pt_c(0)*pt_c(0)+pt_c(1)*pt_c(1));
                        // double r_2 = r*r;
                        // double r_4 = r_2*r_2;
                        // double x1 = pt_c(0)*(1+k1*r_2+k2*r_4)+2*k3*pt_c(0)*pt_c(1)+k4*(r_2+2*pt_c(0)*pt_c(0));
                        // double y1 = pt_c(1)*(1+k1*r_2+k2*r_4)+k3*(r_2+2*pt_c(1)*pt_c(1))+2*k4*pt_c(0)*pt_c(1);
                        // pt_c(0) = fx*x1 + cx;
                        // pt_c(1) = fy*y1 + cy;
                        // cout<<"uv: "<<pt_c(0)<<" "<<pt_c(1)<<endl;

                        double error_x=abs(uv.second(1)-pt_c(0));
                        double error_y=abs(uv.second(2)-pt_c(1));
                       // cout<<"("<<error_x<<","<<error_y<<")"<<" ";
                        if(error_x>3||error_y>3)
                        {
                            f->success_triang=false;
                            success_filter--;
                            break;
                        }
                        
                    //    cout<<"measure: ("<<uv.second(1)<<" "<<uv.second(2)<<")"<<" predict: ("<<pt_c(0)<<" "<<pt_c(1)<<")"<<" ";

                    }
                   //  cout<<endl;
                }
                
           
            
        }
        cout<<"****************************feature_num: "<<feats_db.size()<<endl;
        cout<<"****************************success_num: "<<success_num<<endl;
        cout<<"****************************success_filter: "<<success_filter<<endl;


    //    for(auto feat:feats_db)
    //    {
    //        Feat* f=feat.second;
    //        vector<Point2d> pts,pts_m;
    //        vector<Mat> images;
    //        if(feat.second->success_triang)
    //        {
    //             Feat* f=feat.second;
    //             Vector3d pt_w=f->point_3d_W;
    //             for(auto uv:f->uvs)
    //             {
                    
    //                 double image_id=uv.first;
    //                     Matrix3d R_W_C=image_poses.at(image_id)->Rot();
    //                     Vector3d p_W_C=image_poses.at(image_id)->pos();
    //                     //transform the 3d point in world frame to camera frame
    //                     Vector3d pt_c=R_W_C.transpose()*(pt_w-p_W_C);
    //                     //normalize the 3d point in camera frame
    //                     pt_c(0)=pt_c(0)/pt_c(2);
    //                     pt_c(1)=pt_c(1)/pt_c(2);
    //                     pt_c(2)=pt_c(2)/pt_c(2);

    //                      // Calculate distorted coordinates for fisheye
    //                     double r = std::sqrt(pt_c(0)*pt_c(0)+pt_c(1)*pt_c(1));
    //                     double theta = std::atan(r);
    //                     double theta_d = theta+k1*std::pow(theta,3)+k2*std::pow(theta,5)+k3*std::pow(theta,7)+k4*std::pow(theta,9);

    //                     // Handle when r is small (meaning our xy is near the camera center)
    //                     double inv_r = (r > 1e-8)? 1.0/r : 1.0;
    //                     double cdist = (r > 1e-8)? theta_d * inv_r : 1.0;

    //                     // Calculate distorted coordinates for fisheye
    //                     double x1 = pt_c(0)*cdist;
    //                     double y1 = pt_c(1)*cdist;
    //                     pt_c(0) = fx*x1 + cx;
    //                     pt_c(1) = fy*y1 + cy;
    //                 string image_name=image_name_map[uv.first];
    //                 Mat image=imread(image_dir+image_name,0);
    //                 images.push_back(image.clone());
    //                 Point2d pt;
    //                 pt.x=pt_c(0);
    //                 pt.y=pt_c(1);
    //                 pts.push_back(pt);

    //                 pt.x=uv.second(1);
    //                 pt.y=uv.second(2);
    //                 pts_m.push_back(pt);
    //             }
    //             cv::Mat hconcat_image;
    //             cv::hconcat(images,hconcat_image);
    //             cv::Mat pair_image_color;
    //             cvtColor(hconcat_image,pair_image_color,cv::COLOR_GRAY2BGR);
    //             //    imshow("image",pair_image_color);
    //             //    waitKey();
    //             for(int i=0;i<pts.size();i++)
    //             {
    //                 double col=images[i].cols;
    //                 pts[i].x=pts[i].x+i*col;
    //                 pts_m[i].x=pts_m[i].x+i*col;
    //                 cv::circle(pair_image_color,pts[i],5,cv::Scalar(0,0,255),2);
    //                 cv::circle(pair_image_color,pts_m[i],5,cv::Scalar(255,0,0),2);

    //             }
    //             for(int i=1;i<pts.size();i++)
    //             {
    //                 cv::line(pair_image_color,pts[i-1],pts[i],cv::Scalar(0,255,0),1);
    //                 cv::line(pair_image_color,pts_m[i-1],pts_m[i],cv::Scalar(0,255,255),1);
    //             }
    //             imshow("image",pair_image_color);
    //             waitKey();
    //        }
           
    //    }
        // Open our statistics file!
        of_3d.open(filepath_3d+files[file_num], std::ofstream::out | std::ofstream::app);
        assert(of_3d.is_open());
        for(auto image: image_poses)
        {
            double image_ts=image.first;
            of_3d<<image_name_map[image_ts]<<" ";
            int num=0;
            for(auto feat : feats_db)
            {
                Feat* f=feat.second;
                if(f->success_triang)
                {
                    if(f->uvs.find(image_ts)!=f->uvs.end())
                    {
                        num++;
                    }
                }
            }
            of_3d<<num<<endl;
            for(auto feat : feats_db)
            {
                Feat* f=feat.second;
                if(f->success_triang)
                {
                    if(f->uvs.find(image_ts)!=f->uvs.end())
                    {
                        of_3d<<to_string(int(f->uvs[image_ts](0)))<<" "<<
                        f->uvs[image_ts](1)<<" "<<f->uvs[image_ts](2)<<" "<<f->point_3d_W(0)<<" "<<f->point_3d_W(1)<<" "<<f->point_3d_W(2)<<" ";
                    }
                }
            }
            of_3d<<endl;
        }
        of_3d.close();
        if(success_filter==0)
        {
            boost::filesystem::remove(filepath_3d+files[file_num]);
        }

        // Finally delete our system
        // delete sys;
  
    } 
    cout<<"error load file nums: "<<error_load.size()<<endl;
    for(int i=0;i<error_load.size();i++)
    {
        cout<<"error load file : "<<error_load[i]<<endl;
    }
     
     
        // Done!
        return EXIT_SUCCESS; 
      
      
//     ros::init(argc, argv, "test_kf_match");
//    ros::NodeHandle nh("~");

//    //Create our VIO system
//    VioManagerOptions params = parse_ros_nodehandler(nh);
//    cout<<"finish load parameters.............."<<endl;
//    sys = new VioManager(params);

//    cout<<"finish initialization!..........."<<endl;


//    //load posegraph
//    if(params.use_prior_map)
//    {
//        sys->loadPoseGraph(params.map_save_path,params.pose_graph_filename,params.keyframe_pose_filename);

//    }
//    cout<<"finish load Posegraph and vocabulary"<<endl;

//    // Our camera topics (left and right stereo)
//     std::string topic_imu;
//     std::string topic_camera0, topic_camera1;
//     nh.param<std::string>("topic_imu", topic_imu, "/imu0");
//     nh.param<std::string>("topic_camera0", topic_camera0, "/cam0/image_raw");
//     nh.param<std::string>("topic_camera1", topic_camera1, "/cam1/image_raw");


//     // ros::Subscriber sub0=nh.subscribe(topic_camera0.c_str(),1,callback_cam);
//     // ros::Subscriber sub1=nh.subscribe(topic_camera1.c_str(),1,callback_cam);

//     // Logic for sync stereo subscriber
//     // https://answers.ros.org/question/96346/subscribe-to-two-image_raws-with-one-function/?answer=96491#post-id-96491
//     message_filters::Subscriber<sensor_msgs::Image> image_sub0(nh,topic_camera0.c_str(),1);
//     message_filters::Subscriber<sensor_msgs::Image> image_sub1(nh,topic_camera1.c_str(),1);

// //    message_filters::Subscriber<sensor_msgs::CompressedImage> image_sub0(nh,topic_camera0.c_str(),1);
// //    message_filters::Subscriber<sensor_msgs::CompressedImage> image_sub1(nh,topic_camera1.c_str(),1);
//     //message_filters::TimeSynchronizer<sensor_msgs::Image,sensor_msgs::Image> sync(image_sub0,image_sub1,5);
// //    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CompressedImage, sensor_msgs::CompressedImage> sync_pol;
//     typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
//     message_filters::Synchronizer<sync_pol> sync(sync_pol(5), image_sub0,image_sub1);

//     // Create subscribers
//     ros::Subscriber subcam;
//     if(params.state_options.num_cameras == 1) {
//         ROS_INFO("subscribing to: %s", topic_camera0.c_str());
//         subcam = nh.subscribe(topic_camera0.c_str(), 1, callback_monocular);
//     } else if(params.state_options.num_cameras == 2) {
//         ROS_INFO("subscribing to: %s", topic_camera0.c_str());
//         ROS_INFO("subscribing to: %s", topic_camera1.c_str());
//         sync.registerCallback(boost::bind(&callback_stereo, _1, _2));

//     } else {
//         ROS_ERROR("INVALID MAX CAMERAS SELECTED!!!");
//         std::exit(EXIT_FAILURE);
//     }

//     //===================================================================================
//     //===================================================================================
//     //===================================================================================

//     // Spin off to ROS
//     ROS_INFO("done...spinning to ros");
//     ros::spin();

//    // Finally delete our system
//    delete sys;



//    // Done!
//    return EXIT_SUCCESS;

}

bool single_triangulation(Feat* feat, map<double, ov_type::PoseJPL *>& image_poses) {


    // Total number of measurements
    // Also set the first measurement to be the anchor frame
    int total_meas = 0;
    total_meas=feat->uvs.size();
    
    // Our linear system matrices
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2*total_meas, 3);
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(2*total_meas, 1);

    // Location in the linear system matrices
    size_t c = 0;

    // Get the position of the anchor pose
    double anchor_id=0;
    for(auto image: feat->uvs)
    {
        anchor_id=image.first;
        break;
    }
    feat->anchor_img_id=anchor_id;
    // cout<<"feat anchor_img_id: "<<to_string(feat->anchor_img_id)<<endl;
    PoseJPL* anchor_pose=image_poses.at(anchor_id);
    Eigen::Matrix<double,3,3> R_GtoA = anchor_pose->Rot().transpose();
    Eigen::Matrix<double,3,1> p_AinG = anchor_pose->pos();
    // cout<<"anchor pose: "<<anchor_pose->quat().transpose()<<" "<<anchor_pose->pos().transpose()<<endl; 

    // Loop through each image for this feature
    for(auto image: feat->uvs_norm)
    {
        // Get the position of this image in the global
        Eigen::Matrix<double, 3, 3> R_GtoCi = image_poses.at(image.first)->Rot().transpose();
        Eigen::Matrix<double, 3, 1> p_CiinG = image_poses.at(image.first)->pos();

        // Convert current position relative to anchor
        Eigen::Matrix<double,3,3> R_AtoCi;
        R_AtoCi.noalias() = R_GtoCi*R_GtoA.transpose();
        Eigen::Matrix<double,3,1> p_CiinA;
        p_CiinA.noalias() = R_GtoA*(p_CiinG-p_AinG);
        // Get the UV coordinate normal
        Eigen::Matrix<double, 3, 1> b_i;
        
        b_i << feat->uvs_norm.at(image.first)(1), feat->uvs_norm.at(image.first)(2), 1;
        b_i = R_AtoCi.transpose() * b_i;
        b_i = b_i / b_i.norm();
        Eigen::Matrix<double,2,3> Bperp = Eigen::Matrix<double,2,3>::Zero();
        Bperp << -b_i(2, 0), 0, b_i(0, 0), 0, b_i(2, 0), -b_i(1, 0);

        // Append to our linear system
        A.block(2 * c, 0, 2, 3) = Bperp;
        b.block(2 * c, 0, 2, 1).noalias() = Bperp * p_CiinA;
        c++;
        
    }

    // Solve the linear system
    Eigen::MatrixXd p_f = A.colPivHouseholderQr().solve(b);

    // Check A and p_f
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd singularValues;
    singularValues.resize(svd.singularValues().rows(), 1);
    singularValues = svd.singularValues();
    //condition number: max_eigenvalue/min_eigenvalue. by zzq
    double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0);

    // If we have a bad condition number, or it is too close
    // Then set the flag for bad (i.e. set z-axis to nan)
    //eruoc 1000 0.25 40
    //kaist 5000 0.25 150
    if (std::abs(condA) > 1000|| p_f(2,0) < 0.25 || p_f(2,0) > 40 || std::isnan(p_f.norm())) {
        return false;
    }

    // Store it in our feature object
    feat->point_3d_A= p_f;
    feat->point_3d_W = R_GtoA.transpose()*feat->point_3d_A + p_AinG;
    
    Vector3d uv_norm=Vector3d::Zero();
    uv_norm<<p_f(0)/p_f(2),p_f(1)/p_f(2),1;
    // cout<<"predict uv_norm: "<<uv_norm(0)<<" "<<uv_norm(1)<<endl;
    // cout<<"measure uv_norm: "<<feat->uvs_norm[feat->anchor_img_id](1)<<" "<<feat->uvs_norm[feat->anchor_img_id](2)<<endl;
    Matrix3d R_W_A=image_poses[feat->anchor_img_id]->Rot();
    Vector3d p_W_A=image_poses[feat->anchor_img_id]->pos();
    Vector3d p_A=R_W_A.transpose()*(feat->point_3d_W-p_W_A);
    // cout<<"predict uv_norm: "<<p_A(0)/p_A(2)<<" "<<p_A(1)/p_A(2)<<endl;

    return true;

}


bool single_gaussnewton(Feat* feat, map<double, ov_type::PoseJPL *>& image_poses) {

    //Get into inverse depth
    double rho = 1/feat->point_3d_A(2);
    double alpha = feat->point_3d_A(0)/feat->point_3d_A(2);
    double beta = feat->point_3d_A(1)/feat->point_3d_A(2);

    // Optimization parameters
    double lam = 1e-3;
    double eps = 10000;
    int runs = 0;

    // Variables used in the optimization
    bool recompute = true;
    Eigen::Matrix<double,3,3> Hess = Eigen::Matrix<double,3,3>::Zero();
    Eigen::Matrix<double,3,1> grad = Eigen::Matrix<double,3,1>::Zero();

    // Cost at the last iteration
    double cost_old = compute_error(image_poses,feat,alpha,beta,rho);

    // Get the position of the anchor pose
    Eigen::Matrix<double,3,3> R_GtoA = image_poses.at(feat->anchor_img_id)->Rot().transpose();
    Eigen::Matrix<double,3,1> p_AinG = image_poses.at(feat->anchor_img_id)->pos();

    // Loop till we have either
    // 1. Reached our max iteration count
    // 2. System is unstable
    // 3. System has converged
    while (runs < 20 && lam <  1e10 && eps > 1e-6) {

        // Triggers a recomputation of jacobians/information/gradients
        if (recompute) {

            Hess.setZero();
            grad.setZero();

            double err = 0;
            for(auto image:feat->uvs_norm)
            {
                // Get the position of this image in the global
                Eigen::Matrix<double, 3, 3> R_GtoCi = image_poses.at(image.first)->Rot().transpose();
                Eigen::Matrix<double, 3, 1> p_CiinG = image_poses.at(image.first)->pos();

                // Convert current position relative to anchor
                Eigen::Matrix<double,3,3> R_AtoCi;
                R_AtoCi.noalias() = R_GtoCi*R_GtoA.transpose();
                Eigen::Matrix<double,3,1> p_CiinA;
                p_CiinA.noalias() = R_GtoA*(p_CiinG-p_AinG);
                Eigen::Matrix<double,3,1> p_AinCi;
                p_AinCi.noalias() = -R_AtoCi*p_CiinA;

                double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
                    double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
                    double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);
                    // Calculate jacobian
                    double d_z1_d_alpha = (R_AtoCi(0, 0) * hi3 - hi1 * R_AtoCi(2, 0)) / (pow(hi3, 2));
                    double d_z1_d_beta = (R_AtoCi(0, 1) * hi3 - hi1 * R_AtoCi(2, 1)) / (pow(hi3, 2));
                    double d_z1_d_rho = (p_AinCi(0, 0) * hi3 - hi1 * p_AinCi(2, 0)) / (pow(hi3, 2));
                    double d_z2_d_alpha = (R_AtoCi(1, 0) * hi3 - hi2 * R_AtoCi(2, 0)) / (pow(hi3, 2));
                    double d_z2_d_beta = (R_AtoCi(1, 1) * hi3 - hi2 * R_AtoCi(2, 1)) / (pow(hi3, 2));
                    double d_z2_d_rho = (p_AinCi(1, 0) * hi3 - hi2 * p_AinCi(2, 0)) / (pow(hi3, 2));
                    Eigen::Matrix<double, 2, 3> H;
                    H << d_z1_d_alpha, d_z1_d_beta, d_z1_d_rho, d_z2_d_alpha, d_z2_d_beta, d_z2_d_rho;
                    // Calculate residual
                    Eigen::Matrix<float, 2, 1> z;
                    z << hi1 / hi3, hi2 / hi3;
                    Eigen::Matrix<float,2,1> uv_norm;
                    uv_norm<<feat->uvs_norm.at(image.first)(1),feat->uvs_norm.at(image.first)(2);
                    Eigen::Matrix<float, 2, 1> res = uv_norm - z;

                    // Append to our summation variables
                    err += std::pow(res.norm(), 2);
                    grad.noalias() += H.transpose() * res.cast<double>();
                    Hess.noalias() += H.transpose() * H;
            }
            
        }

        // Solve Levenberg iteration
        Eigen::Matrix<double,3,3> Hess_l = Hess;
        for (size_t r=0; r < (size_t)Hess.rows(); r++) {
            Hess_l(r,r) *= (1.0+lam);
        }

        Eigen::Matrix<double,3,1> dx = Hess_l.colPivHouseholderQr().solve(grad);
        //Eigen::Matrix<double,3,1> dx = (Hess+lam*Eigen::MatrixXd::Identity(Hess.rows(), Hess.rows())).colPivHouseholderQr().solve(grad);

        // Check if error has gone down
        double cost = compute_error(image_poses,feat,alpha+dx(0,0),beta+dx(1,0),rho+dx(2,0));

        // Debug print
        //cout << "run = " << runs << " | cost = " << dx.norm() << " | lamda = " << lam << " | depth = " << 1/rho << endl;

        // Check if converged
        if (cost <= cost_old && (cost_old-cost)/cost_old < 1e-6) {
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            eps = 0;
            break;
        }

        // If cost is lowered, accept step，and shrink lam to make next step larger
        // Else inflate lambda to make next step smaller (try to make more stable)
        if (cost <= cost_old) {
            recompute = true;
            cost_old = cost;
            alpha += dx(0, 0);
            beta += dx(1, 0);
            rho += dx(2, 0);
            runs++;
            lam = lam/10;
            eps = dx.norm();
        } else {
            recompute = false;
            lam = lam*10;
            continue;
        }
    }

    // Revert to standard, and set to all
    feat->point_3d_A(0) = alpha/rho;
    feat->point_3d_A(1) = beta/rho;
    feat->point_3d_A(2) = 1/rho;

    // Get tangent plane to x_hat
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(feat->point_3d_A);
    Eigen::MatrixXd Q = qr.householderQ();

    // Max baseline we have between poses
    double base_line_max = 0.0;

    // Check maximum baseline
    // Loop through each camera for this feature
    //TODO: What the geometry meaning of base_line?
    for(auto const& image:feat->uvs_norm)
    {
        Eigen::Matrix<double,3,1> p_CiinG  = image_poses.at(image.first)->pos();
            // Convert current position relative to anchor
            Eigen::Matrix<double,3,1> p_CiinA = R_GtoA*(p_CiinG-p_AinG);
            // Dot product camera pose and nullspace
            double base_line = ((Q.block(0,1,3,2)).transpose() * p_CiinA).norm();
            if (base_line > base_line_max) base_line_max = base_line;
    }

    // Check if this feature is bad or not
    // 1. If the feature is too close
    // 2. If the feature is invalid
    // 3. If the baseline ratio is large
    //euroc  0.25 40 40
    //kaist 0.25 150 500
    if(feat->point_3d_A(2) < 0.25
        || feat->point_3d_A(2) > 40
        || (feat->point_3d_A.norm() / base_line_max) > 40
        || std::isnan(feat->point_3d_A.norm())) {
        return false;
    }

    // Finally get position in global frame
    feat->point_3d_W = R_GtoA.transpose()*feat->point_3d_A+ p_AinG;
    return true;

}

double compute_error(map<double, ov_type::PoseJPL *>& image_poses,
                     Feat* feat, double alpha, double beta, double rho) {

    // Total error
    double err = 0;

    // Get the position of the anchor pose
    Eigen::Matrix<double,3,3> R_GtoA = image_poses.at(feat->anchor_img_id)->Rot().transpose();
    Eigen::Matrix<double,3,1> p_AinG = image_poses.at(feat->anchor_img_id)->pos();

    // Loop through each image for this feature
    for(auto image:feat->uvs_norm)
    {
         // Get the position of this image in the global
        Eigen::Matrix<double, 3, 3> R_GtoCi = image_poses.at(image.first)->Rot().transpose();
        Eigen::Matrix<double, 3, 1> p_CiinG = image_poses.at(image.first)->pos();

        // Convert current position relative to anchor
        Eigen::Matrix<double,3,3> R_AtoCi;
        R_AtoCi.noalias() = R_GtoCi*R_GtoA.transpose();
        Eigen::Matrix<double,3,1> p_CiinA;
        p_CiinA.noalias() = R_GtoA*(p_CiinG-p_AinG);
        Eigen::Matrix<double,3,1> p_AinCi;
        p_AinCi.noalias() = -R_AtoCi*p_CiinA;

        // Middle variables of the system
            //alpha: x/z in anchor ;beta:y/z in anchor; rho: 1/z in anchor
            double hi1 = R_AtoCi(0, 0) * alpha + R_AtoCi(0, 1) * beta + R_AtoCi(0, 2) + rho * p_AinCi(0, 0);
            double hi2 = R_AtoCi(1, 0) * alpha + R_AtoCi(1, 1) * beta + R_AtoCi(1, 2) + rho * p_AinCi(1, 0);
            double hi3 = R_AtoCi(2, 0) * alpha + R_AtoCi(2, 1) * beta + R_AtoCi(2, 2) + rho * p_AinCi(2, 0);

        // Calculate residual
            Eigen::Matrix<float, 2, 1> z;
            z << hi1 / hi3, hi2 / hi3;
            Eigen::Matrix<float,2,1> uv_norm;
            uv_norm<<feat->uvs_norm.at(image.first)(1),feat->uvs_norm.at(image.first)(2);
            Eigen::Matrix<float, 2, 1> res = uv_norm - z;
            // Append to our summation variables
            err += pow(res.norm(), 2);
    }

    return err;

}

void callback_monocular(const sensor_msgs::ImageConstPtr& msg0) {


    // Get the image
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Fill our buffer if we have not
    if(img0_buffer.rows == 0) {
        time_buffer = cv_ptr->header.stamp.toSec();
        img0_buffer = cv_ptr->image.clone();
        return;
    }

    // send it to our VIO system
    sys->feed_measurement_monocular(time_buffer, img0_buffer, 0);

    // move buffer forward
    time_buffer = cv_ptr->header.stamp.toSec();
    img0_buffer = cv_ptr->image.clone();

    cout.precision(15);
    cout<<"time : "<<cv_ptr->header.stamp.toSec()<<endl;

}



void callback_stereo(const sensor_msgs::ImageConstPtr& msg0, const sensor_msgs::ImageConstPtr& msg1) {

    // Get the image
    cout<<"in callback_stereo"<<endl;
    cv_bridge::CvImageConstPtr cv_ptr0;
    try {
        cv_ptr0 = cv_bridge::toCvShare(msg0, sensor_msgs::image_encodings::MONO8);
//        cv_ptr0 = cv_bridge::toCvCopy(msg0, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Get the image
    cv_bridge::CvImageConstPtr cv_ptr1;
    try {
        cv_ptr1 = cv_bridge::toCvShare(msg1, sensor_msgs::image_encodings::MONO8);
//        cv_ptr1 = cv_bridge::toCvCopy(msg1, sensor_msgs::image_encodings::MONO8);
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }


    // Fill our buffer if we have not
    cout<<"fill buffer"<<endl;
    if(img0_buffer.rows == 0 || img1_buffer.rows == 0) {
        time_buffer = cv_ptr0->header.stamp.toSec();
        img0_buffer = cv_ptr0->image.clone();
        time_buffer = cv_ptr1->header.stamp.toSec();
        img1_buffer = cv_ptr1->image.clone();
        return;
    }

    // send it to our VIO system
    cout<<"before test_feature_match"<<endl;
    sys->test_feature_match(time_buffer, img0_buffer, img1_buffer, 0, 1);

   


    // move buffer forward
    time_buffer = cv_ptr0->header.stamp.toSec();
    img0_buffer = cv_ptr0->image.clone();
    time_buffer = cv_ptr1->header.stamp.toSec();
    img1_buffer = cv_ptr1->image.clone();
    // cout.precision(15);
    // cout<<"time c0: "<<cv_ptr0->header.stamp.toSec()<<endl;
    // cout<<"time c1: "<<cv_ptr1->header.stamp.toSec()<<endl;

}




// {

//    // Launch our ros node
//    ros::init(argc, argv, "test_kf_match");
//    ros::NodeHandle nh("~");
//
//
//    // Create our VIO system
//    VioManagerOptions params = parse_ros_nodehandler(nh);
//    cout<<"finish load parameters.............."<<endl;
//    sys = new VioManager(params);
//
//    cout<<"finish initialization!..........."<<endl;
//
//
//    //load posegraph
//    if(params.use_prior_map)
//    {
//        sys->loadVocabulary(params.map_save_path,params.voc_filename);
//        sys->loadPoseGraph(params.map_save_path,params.pose_graph_filename);
//
//    }
//    cout<<"finish load Posegraph and vocabulary"<<endl;
//
//    //===================================================================================
//    //===================================================================================
//    //===================================================================================
////    int size_kfdatabase= sys->state->kfdatabase->get_internal_data().size();
//    std::unordered_map<double, Keyframe*> database=sys->state->get_kfdataBase()->get_internal_data();
//    cout<<"after get database,with size "<<database.size()<<endl;
//    auto iter=database.begin();
//    int id=-1;
//    Keyframe* kf_loop=nullptr;
//    while(iter!=database.end())
//    {
//        kf_loop=iter->second;
//        if(kf_loop->loop_index!=-1)
//        {
//            id=kf_loop->loop_index;
//            break;
//        }
//        else
//        {
//            iter++;
//        }
//
//    }
//    cout<<"after while"<<endl;
//    if(id!=-1)
//    {
//        cout<<id<<" with "<<kf_loop->index<<endl;
//    }
//    Keyframe* kf=sys->state->get_kfdataBase()->get_internal_data().at(id);
//    Quaterniond q_2_to_1;
//    q_2_to_1.w()=kf_loop->loop_info(3);
//    q_2_to_1.x()=kf_loop->loop_info(4);
//    q_2_to_1.y()=kf_loop->loop_info(5);
//    q_2_to_1.z()=kf_loop->loop_info(6);
//    Matrix3d R_2_to_1=q_2_to_1.toRotationMatrix();
//    Vector3d t_2_in_1;
//    t_2_in_1<<kf_loop->loop_info(0),kf_loop->loop_info(1),kf_loop->loop_info(2);
//
//    string image_dir=params.map_save_path;
//    string image_1_name=image_dir+to_string(id)+"_image.png";
//    string image_2_name=image_dir+to_string(kf_loop->index)+"_image.png";
//    cout<<"before load image////"<<endl;
//    cv::Mat image_1= cv::imread(image_1_name,0); //cur_image
//    Mat image_2 = cv::imread(image_2_name,0);  //loop_image
//    cout<<"finish load image///"<<endl;
//    cv::imshow("image1",image_1);
//    cv::waitKey(0);
//    imshow("image_2",image_2);
//    cv::waitKey(0);
//    BriefExtractor extractor((params.map_save_path+params.brief_pattern_filename).c_str());
//    cout<<"finish extractor contruction"<<endl;
//    double time_stamp_2=200;
//    double time_stamp_1=1;
//    int index=1;
//    Eigen::Matrix<double,7,1> pos1,pos2;//qx qy qz qw x y z
//
//
//    Matrix3d R_image1_to_vio= kf->_Pose_KFinWorld->Rot();
//    Matrix3d R_image2_to_vio=kf_loop->_Pose_KFinWorld->Rot();
//
//    Vector3d t_image1_in_vio=kf->_Pose_KFinWorld->pos();
//    Vector3d t_image2_in_vio=kf_loop->_Pose_KFinWorld->pos();
//
////    pos1<<-0.106458, -0.803102, -0.069980, 0.582063,4.311077, -1.411764, 0.886794;//image_1 gt
////    Quaterniond q1_image1_to_vio(pos1(3),pos1(0),pos1(1),pos1(2));
////    Vector3d t_image1_in_vio(pos1(4),pos1(5),pos1(6));
////
////    pos2<<-0.229706, -0.789011, -0.151366, 0.549350, 4.424093, -1.424327, 0.855422;//image_2 gt
////
////    Quaterniond q2_image2_to_vio(pos2(3),pos2(0),pos2(1),pos2(2));
////    Vector3d t_image2_in_vio(pos2(4),pos2(5),pos2(6));
////
////    Matrix3d R_image1_to_vio=q1_image1_to_vio.toRotationMatrix();
////    Matrix3d R_image2_to_vio=q2_image2_to_vio.toRotationMatrix();
//
//    Matrix3d R_image1_to_image2= R_image2_to_vio.transpose()*R_image1_to_vio;
//    Vector3d t_image1_in_image2= R_image2_to_vio.transpose()*(t_image1_in_vio-t_image2_in_vio);
////    cout<<"R_image2_to_image_1: "<<endl<<R_image1_to_image2.transpose()<<endl<<R_2_to_1<<endl;
////    cout<<"t_image1_in_image_2: "<<endl<<t_image1_in_image2<<endl<<-R_2_to_1.transpose()*t_2_in_1<<endl;
//
//
//    AngleAxisd rotation_vector(M_PI/4,Vector3d(0,0,1));
//    MatrixXd R_vio_to_map=rotation_vector.toRotationMatrix();
//    Vector3d t_vio_in_map(3,2.5,-2);
//
//    Matrix3d R_image2_to_map= R_vio_to_map* R_image2_to_vio;
//    Vector3d t_image2_in_map= R_vio_to_map * t_image2_in_vio;
//
//    Quaterniond q_image_2_in_map(R_image2_to_map);
//    Eigen::Matrix<double,7,1> pos2_new;
//    cout<<"before pos2_new"<<endl;
//    pos2_new<<q_image_2_in_map.x(),q_image_2_in_map.y(),q_image_2_in_map.z(),q_image_2_in_map.w(),t_image2_in_map;
//    cout<<"after pose2_new"<<endl;
//
//    kf_loop->_Pose_KFinWorld->set_value(pos2_new);
//
//
//
////    size_t loop_index = -1;
////    Eigen::Matrix<double, 8, 1 > loop_info;
////    loop_info<<1,2,3,4,5,6,7,8;
////    Eigen::VectorXd intrinsics;
////    intrinsics=params.camera_intrinsics[0];
////    intrinsics.conservativeResize(9,1);
////    intrinsics(8)=0;
////    cout<<intrinsics<<endl;
////    cout<<"after intrincs"<<endl;
////    const int fast_th = 20; // corner detector response threshold
////    vector<cv::KeyPoint> keypoints;
////     cv::FAST(image_2, keypoints, fast_th, true);
////    vector<BRIEF::bitset> brief_descriptors;
////    extractor(image_2, keypoints, brief_descriptors);
////    cout<<"after extract keypoints"<<endl;
////
////
////    std::vector<cv::KeyPoint> pts0_n;
////    for(size_t i=0; i<keypoints.size(); i++) {
////        cv::Point2f point=keypoints[i].pt;
////        cv::KeyPoint kp;
////        kp.pt=sys->trackFEATS->undistort_point(point,0);
////        pts0_n.push_back(kp);
////    }
////    cout<<"after pts0_n"<<endl;
////    vector<Eigen::Vector3d> keypoints_3d_local;
////
////
////    Keyframe* loop_keyframe = new Keyframe(time_stamp_2, index, pos2_new, pos2_new, image_2, loop_index, loop_info, keypoints, pts0_n, keypoints_3d_local,brief_descriptors,intrinsics);
////    cout<<"after loop_keyframe new"<<endl;
////    sys->loadKeyframe(loop_keyframe);
////    cout<<"after loadkeyframe"<<endl;
//
//
//
//    vector<cv::KeyPoint> keypoints;
//    keypoints=kf->keypoints;
////     cv::FAST(image_1, keypoints, 20, true);
////    vector<BRIEF::bitset> brief_descriptors;
////    extractor(image_1, keypoints, brief_descriptors);
////    cout<<"after extract keypoints"<<endl;
//
//
//    std::vector<cv::KeyPoint> pts0_n;
////    for(size_t i=0; i<keypoints.size(); i++) {
////        cv::Point2f point=keypoints[i].pt;
////        cv::KeyPoint kp;
////        kp.pt=sys->trackFEATS->undistort_point(point,0);
////        pts0_n.push_back(kp);
////    }
////    cout<<"after pts0_n"<<endl;
//////    sys->trackFEATS->feed_monocular(time_stamp_1,image_1,0);
////    cout<<"after feed_monocular"<<endl;
//
//    vector<cv::Point2f> uv_0, uv_1;
//    vector<cv::Point2f> uv_norm_0, uv_norm_1;
//    vector<size_t> feature_id_0, feature_id_1;
//
//
//    pts0_n=kf->keypoints_norm;
//    int count=0;
//
//    vector<BRIEF::bitset> brief_descriptors;
//    int n=int(keypoints.size())/200;
//    cout<<"before forloop"<<endl;
//    for (int i=0;i<min((int)keypoints.size(),200);i++)
//    {
//
//
//        uv_0.push_back(keypoints[i].pt);
//        uv_norm_0.push_back(pts0_n[i].pt);
//        feature_id_0.push_back(i);
//        brief_descriptors.emplace_back(kf->brief_descriptors[i]);
//
//    }
//
//
////    sys->get_image_features(time_stamp_1, 0, uv_0, uv_norm_0, feature_id_0);
////    cout<<"after get image features"<<endl;
//    cout<<"features' num: "<<uv_0.size()<<" "<<uv_norm_0.size()<<endl;
//    Keyframe *cur_kf_0= nullptr;
//    cur_kf_0 = new Keyframe(time_stamp_1, 2222, image_1, 0, uv_0, uv_norm_0, feature_id_0,
//                            params.camera_intrinsics[0]);
//    cout<<"after cur_kf_0"<<endl;
//
//    cur_kf_0->_Pose_KFinWorld=kf->_Pose_KFinWorld;
//
//    for(int i = 0; i < (int)cur_kf_0->point_2d_uv.size(); i++)
//    {
//        cv::KeyPoint key;
//        key.pt =cur_kf_0->point_2d_uv[i];
//        cur_kf_0->window_keypoints.push_back(key);
//        cur_kf_0->window_brief_descriptors.push_back(brief_descriptors[i]);
//    }
//    cout<<"window_keypoints.size():"<<cur_kf_0->window_keypoints.size()<<" "<<cur_kf_0->window_brief_descriptors.size()<<endl;
////    extractor(cur_kf_0->image, cur_kf_0->window_keypoints, cur_kf_0->window_brief_descriptors);
////    cout<<"after extract cur_kf_0"<<endl;
//    int loopindex=-1;
//    //detect the loop from keyframedatabase and return the loope id
//    cout<<"db size: "<<sys->matchKF->db->size()<<endl;
//    loopindex = sys->matchKF->detectLoop(cur_kf_0,cur_kf_0->index);
//    cout<<"loopindex: "<<loopindex<<endl;
//    if(1) {
////        Keyframe *loop_kf =sys->state->get_kfdataBase()->get_keyframe(loopindex);
//          Keyframe *loop_kf = kf_loop;
//
//        if (cur_kf_0->findConnection(loop_kf)) {
//            cout<<"before computeRelativePose"<<endl;
//           // sys->ComputeRelativePose(cur_kf_0, loop_kf);
//            cout<<"after computeRelativePose"<<endl;
//            vector<Point2f> pt_cur, pt_kf;
//            for(int i=0; i<cur_kf_0->_matched_id_cur.size();i++)
//            {
//                size_t feature_id=cur_kf_0->_matched_id_cur[i];
//                size_t kf_matched_id=cur_kf_0->_matched_id_old[i];
//
////                Feature* feat=sys->trackFEATS->get_feature_database()->get_feature(feature_id);
////                auto it1=feat->uvs[0].begin();
////                auto it2=feat->timestamps[0].begin();
////                while(it2!=feat->timestamps[0].end())
////                {
////                    if((*it2)==cur_kf_0->time_stamp)
////                    {
////                        VectorXf pt=*it1;
////                        Point2f pt2f;
////                        pt2f.x=pt(0);
////                        pt2f.y=pt(1);
//////                        KeyPoint kp;
//////                        kp.pt=pt2f;
////                        pt_cur.push_back(pt2f);
////                        break;
////                    }
////                }
//
//                pt_kf.push_back(loop_kf->keypoints[kf_matched_id].pt);
//                pt_cur.push_back(cur_kf_0->point_2d_uv[feature_id]);
//
//
//            }
//            cout<<"after pt_kf"<<endl;
//            cv::Mat pair_image;
//            cv::hconcat(image_1,image_2,pair_image);
//            Mat pair_image_color;
//            cvtColor(pair_image,pair_image_color,COLOR_GRAY2BGR);
//            for(int i=0;i<pt_cur.size();i++)
//            {
//
//                int col=image_1.cols;
//                circle(pair_image_color,pt_cur[i],2,Scalar(255,0,0));
//                circle(pair_image_color,pt_kf[i]+Point2f(col,0),2,Scalar(0,255,255));
//                line(pair_image_color,pt_cur[i],pt_kf[i]+Point2f(col,0),Scalar(0,255,0),2);
//            }
//            imshow("pair",pair_image_color);
//            waitKey(0);
//
//
//        }
//        else{
//            cout<<"cannot find connection"<<endl;
//            return 0;
//        }
//    }
//
//
//    Vector3d relative_t;
//    Matrix3d relative_q; //cur_frame to loop_frame
//    relative_t=cur_kf_0->getLoopRelativeT();
//    relative_q=quat_2_Rot(cur_kf_0->getLoopRelativeQ());
//
//    cout<<"relative_R: from image_1 to image_2"<<endl;
//    cout<<relative_q<<endl;
//    cout<<R_image1_to_image2<<endl;
//    cout<<"relative_t: image_1 in image2"<<endl;
//    cout<<relative_t<<endl;
//    cout<<t_image1_in_image2<<endl;
//    cout<<"R_vio_to_map:"<<endl;
//    cout<<sys->state->R_vio_to_map[time_stamp_1]<<endl;
//    cout<<R_vio_to_map<<endl;
//    cout<<"t_vio_in_map:"<<endl;
//    cout<<sys->state->p_vio_in_map[time_stamp_1]<<endl;
//    cout<<t_vio_in_map<<endl;
//
//
//
//// Finally delete our system
//    delete sys;
//
//
//    // Done!
//    return EXIT_SUCCESS;

// }
