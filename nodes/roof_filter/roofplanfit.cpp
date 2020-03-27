/*
    @file roofplanfit.cpp
    @brief ROS Node for roof plane fitting

    This is a ROS node to perform roof plan fitting.
    Implementation accoriding to <Fast Segmentation of 3D Point Clouds: A Paradigm>

    In this case, it's assumed that the x,y axis points at sea-level,
    and z-axis points up. The sort of height is based on the Z-axis value.

    @author Vincent Cheung(VincentCheungm)
    @bug Sometimes the plane is not fit.
*/

#include <iostream>
// For disable PCL complile lib, to use PointXYZIR    
#define PCL_NO_PRECOMPILE

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//typedef pcl::PointXYZINormal PointType;

#define PointType pcl::PointXYZINormal

// using eigen lib
#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;

pcl::PointCloud<PointType>::Ptr g_seeds_pc(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr g_roof_pc(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr g_not_roof_pc(new pcl::PointCloud<PointType>());

/*
    @brief Compare function to sort points. Here use z axis.
    @return z-axis accent
*/
bool point_cmp(PointType a, PointType b){
    return a.z<b.z;
}

/*
    @brief roof Plane fitting ROS Node.
    @param Velodyne Pointcloud topic.
    @param Sensor Model.
    @param Sensor height for filtering error mirror points.
    @param Num of segment, iteration, LPR
    @param Threshold of seeds distance, and roof plane distance
    
    @subscirbe:/velodyne_points
    @publish:/points_no_roof, /points_roof
*/
class roofPlaneFit{
public:
    roofPlaneFit();
private:
    ros::NodeHandle node_handle_;
    ros::Subscriber points_node_sub_;
    ros::Publisher roof_points_pub_;
    ros::Publisher roofless_points_pub_;

    std::string point_topic_;

    int sensor_model_;
    double sensor_height_;
    int num_seg_;
    int num_iter_;
    int num_lpr_;
    double th_seeds_;
    double th_dist_;


    void velodyne_callback_(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg);
    void estimate_plane_(void);
    void extract_initial_seeds_(const pcl::PointCloud<PointType>& p_sorted);

    // Model parameter for roof plane fitting
    // The roof plane model is: ax+by+cz+d=0
    // Here normal:=[a,b,c], d=d
    // th_dist_d_ = threshold_dist - d 
    float d_;
    MatrixXf normal_;
    float th_dist_d_;
};    

/*
    @brief Constructor of GPF Node.
    @return void
*/
roofPlaneFit::roofPlaneFit():node_handle_("~"){
    // Init ROS related
    ROS_INFO("Inititalizing roof Plane Fitter...");
    node_handle_.param<std::string>("point_topic", point_topic_, "/points_no_ground");
    ROS_INFO("Input Point Cloud: %s", point_topic_.c_str());

    node_handle_.param("sensor_model", sensor_model_, 32);
    ROS_INFO("Sensor Model: %d", sensor_model_);

    node_handle_.param("sensor_height", sensor_height_, 2.5);
    ROS_INFO("Sensor Height: %f", sensor_height_);

    node_handle_.param("num_seg", num_seg_, 1);
    ROS_INFO("Num of Segments: %d", num_seg_);

    node_handle_.param("num_iter", num_iter_, 3);
    ROS_INFO("Num of Iteration: %d", num_iter_);

    node_handle_.param("num_lpr", num_lpr_, 20);
    ROS_INFO("Num of LPR: %d", num_lpr_);

    node_handle_.param("th_seeds", th_seeds_, 1.2);
    ROS_INFO("Seeds Threshold: %f", th_seeds_);

    node_handle_.param("th_dist", th_dist_, 1.4);   //gai
    ROS_INFO("Distance Threshold: %f", th_dist_);

    // Listen to velodyne topic
    points_node_sub_ = node_handle_.subscribe(point_topic_, 2, &roofPlaneFit::velodyne_callback_, this);
    
    // Publish Init
    std::string no_roof_topic, roof_topic;
    node_handle_.param<std::string>("no_roof_point_topic", no_roof_topic, "/points_no_roof");
    ROS_INFO("No roof Output Point Cloud: %s", no_roof_topic.c_str());
    node_handle_.param<std::string>("roof_point_topic", roof_topic, "/points_roof");
    ROS_INFO("Only roof Output Point Cloud: %s", roof_topic.c_str());
    roofless_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>(no_roof_topic, 2);
    roof_points_pub_ = node_handle_.advertise<sensor_msgs::PointCloud2>(roof_topic, 2);
 
}

/*
    @brief The function to estimate plane model. The
    model parameter `normal_` and `d_`, and `th_dist_d_`
    is set here.
    The main step is performed SVD(UAV) on covariance matrix.
    Taking the sigular vector in U matrix according to the smallest
    sigular value in A, as the `normal_`. `d_` is then calculated 
    according to mean roof points.

    @param g_roof_pc:global roof pointcloud ptr.
    
*/
void roofPlaneFit::estimate_plane_(void){
    // Create covarian matrix.
    // 1. calculate (x,y,z) mean
    float x_mean = 0, y_mean = 0, z_mean = 0;
    for(int i=0;i<g_roof_pc->points.size();i++){
        x_mean += g_roof_pc->points[i].x;
        y_mean += g_roof_pc->points[i].y;
        z_mean += g_roof_pc->points[i].z;
    }
    // incase of divide zero
    int size = g_roof_pc->points.size()!=0?g_roof_pc->points.size():1;
    x_mean /= size;
    y_mean /= size;
    z_mean /= size;
    // 2. calculate covariance
    // cov(x,x), cov(y,y), cov(z,z)
    // cov(x,y), cov(x,z), cov(y,z)
    float xx = 0, yy = 0, zz = 0;
    float xy = 0, xz = 0, yz = 0;
    for(int i=0;i<g_roof_pc->points.size();i++){
        xx += (g_roof_pc->points[i].x-x_mean)*(g_roof_pc->points[i].x-x_mean);
        xy += (g_roof_pc->points[i].x-x_mean)*(g_roof_pc->points[i].y-y_mean);
        xz += (g_roof_pc->points[i].x-x_mean)*(g_roof_pc->points[i].z-z_mean);
        yy += (g_roof_pc->points[i].y-y_mean)*(g_roof_pc->points[i].y-y_mean);
        yz += (g_roof_pc->points[i].y-y_mean)*(g_roof_pc->points[i].z-z_mean);
        zz += (g_roof_pc->points[i].z-z_mean)*(g_roof_pc->points[i].z-z_mean);
    }
    // 3. setup covarian matrix cov
    MatrixXf cov(3,3);
    cov << xx,xy,xz,
           xy, yy, yz,
           xz, yz, zz;
    cov /= size;
    // Singular Value Decomposition: SVD
    JacobiSVD<MatrixXf> svd(cov,Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));
    // mean roof seeds value
    MatrixXf seeds_mean(3,1);
    seeds_mean<<x_mean,y_mean,z_mean;
    // according to normal.T*[x,y,z] = -d
    d_ = -(normal_.transpose()*seeds_mean)(0,0);
    // set distance threhold to `th_dist - d`
    th_dist_d_ = th_dist_ - d_;
 
    // return the equation parameters
}


/*
    @brief Extract initial seeds of the given pointcloud sorted segment.
    This function filter roof seeds points accoring to heigt.
    This function will set the `g_roof_pc` to `g_seed_pc`.
    @param p_sorted: sorted pointcloud
    
    @param ::num_lpr_: num of LPR points
    @param ::th_seeds_: threshold distance of seeds
    @param ::

*/
void roofPlaneFit::extract_initial_seeds_(const pcl::PointCloud<PointType>& p_sorted){
    // LPR is the mean of low point representative
    double sum = 0;
    int cnt = 0;
    // Calculate the mean height value.
    for(int i=0;i<p_sorted.points.size() && cnt<num_lpr_;i++){
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double lpr_height = cnt!=0?sum/cnt:0;// in case divide by 0
    g_seeds_pc->clear();
    // iterate pointcloud, filter those height is more than lpr.height+th_seeds_
    for(int i=0;i<p_sorted.points.size();i++){
        if(p_sorted.points[i].z > lpr_height + th_seeds_){     //gai
            g_seeds_pc->points.push_back(p_sorted.points[i]);
        }
    }
    // return seeds points
}

/*
    @brief Velodyne pointcloud callback function. The main GPF pipeline is here.
    PointCloud SensorMsg -> Pointcloud -> z-value sorted Pointcloud
    ->error points removal -> extract roof seeds -> roof plane fit mainloop
*/
void roofPlaneFit::velodyne_callback_(const sensor_msgs::PointCloud2ConstPtr& in_cloud_msg){
    // 1.Msg to pointcloud
    pcl::PointCloud<PointType> laserCloudIn;
    pcl::fromROSMsg(*in_cloud_msg, laserCloudIn);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn,indices);
    // 2.Sort on Z-axis value.
    sort(laserCloudIn.points.begin(),laserCloudIn.end(),point_cmp);
    // 3.Error point removal
    // As there are some error mirror reflection under the roof, 
    // here regardless point under 2* sensor_height
    // Sort point according to height, here uses z-axis in default
    pcl::PointCloud<PointType>::iterator it = laserCloudIn.points.begin();
    for(int i=0;i<laserCloudIn.points.size();i++){
        if(laserCloudIn.points[i].z < -1.5*sensor_height_){ //gai
            it++;
        }else{
            break;
        }
    }
    laserCloudIn.points.erase(laserCloudIn.points.begin(),it);
    // 4. Extract init roof seeds.
    extract_initial_seeds_(laserCloudIn);
    g_roof_pc = g_seeds_pc;
    
    // 5. roof plane fitter mainloop
    for(int i=0;i<num_iter_;i++){
        estimate_plane_();
        g_roof_pc->clear();
        g_not_roof_pc->clear();

        //pointcloud to matrix
        MatrixXf points(laserCloudIn.points.size(),3);
        int j =0;
        for(auto p:laserCloudIn.points){
            points.row(j++)<<p.x,p.y,p.z;
        }
        // roof plane model
        VectorXf result = points*normal_;
        // threshold filter
        for(int r=0;r<result.rows();r++){
            if(result[r]>th_dist_d_){
                g_roof_pc->points.push_back(laserCloudIn[r]);  //gai
            }else{
                g_not_roof_pc->points.push_back(laserCloudIn[r]);
            }
        }
    }

    // publish roof points
    sensor_msgs::PointCloud2 roof_msg;
    pcl::toROSMsg(*g_roof_pc, roof_msg);
    roof_msg.header.stamp = in_cloud_msg->header.stamp;
    roof_msg.header.frame_id = in_cloud_msg->header.frame_id;
    roof_points_pub_.publish(roof_msg);

    // publish not roof points
    sensor_msgs::PointCloud2 roofless_msg;
    pcl::toROSMsg(*g_not_roof_pc, roofless_msg);
    roofless_msg.header.stamp = in_cloud_msg->header.stamp;
    roofless_msg.header.frame_id = in_cloud_msg->header.frame_id;
    roofless_points_pub_.publish(roofless_msg);
    
}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "roofPlaneFit");
    roofPlaneFit node;
    ros::spin();

    return 0;

}