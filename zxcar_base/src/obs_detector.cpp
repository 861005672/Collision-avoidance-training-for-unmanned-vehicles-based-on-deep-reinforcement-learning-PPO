#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <gazebo_msgs/SetLinkState.h>
#include <geometry_msgs/Pose.h>
#include <zxcar_base/ObstacleDetection.h>  // 自定义服务类型
#include <gazebo_msgs/ContactsState.h>
#include <sensor_msgs/LaserScan.h>

// 互斥锁,由于本服务可能被多个进程同时调用，因此需要用互斥锁保证进程间不会同时对共享数据 pointCloudData 进行操作导致的崩溃问题
std::mutex service_mutex_;  // 串行化服务调用，保护服务回调的锁
std::mutex data_mutex_; // 保护雷达数据读写 的锁，避免在服务回调使用雷达数据时，雷达数据被修改


class ObstacleDetector {

public:
    ObstacleDetector() {
        setlocale(LC_ALL, "");

        // 获取参数
        ros::NodeHandle pnh("~");
        pnh.getParam("ray_check_range", this->ray_check_range);
        

        // 初始化服务
        service_ = nh_.advertiseService("/obs_detector", 
                                        &ObstacleDetector::detectCallback, this);

        // 订阅RaySensor数据
        scan_sub_ = nh_.subscribe("/obsDetector_ray_scan", 1, 
            &ObstacleDetector::scanCallback, this);

        // 初始化set_link_state服务的客户端
        client_callSetLinkState = nh_.serviceClient<gazebo_msgs::SetLinkState>("/gazebo/set_link_state");

		
		
		ROS_INFO("[obs_detector] 障碍物探测节点已启动, 雷达射线检测范围: %.2f", this->ray_check_range);
    }

private:
//// 成员函数
    // 回调函数 处理服务请求
    bool detectCallback(zxcar_base::ObstacleDetection::Request &req, zxcar_base::ObstacleDetection::Response &res)               
    {

		// std::lock_guard<std::mutex> lock(service_mutex_); // 加锁，串行化服务处理，避免多进程调用服务时产生冲突，导致崩溃

        ROS_INFO("[obs_detector] 收到检测障碍物的服务请求, 检测点: (%.2f,%.2f,%.2f)", req.detection_point.x, req.detection_point.y, req.detection_point.z);

        // 1. 设置响应的初值为false
        res.obstacle_detected = false;
        
        // 1. 移动传感器到目标位置
        moveSensor(req.detection_point);


		bool hasObs = false;
		for(int j=0; j<10; j++)		// 循环10次，确保检测正确
		{
			ROS_INFO("第%d次检测",j);
			// 2. 等待射线扫描数据到达
			this->pointCloudData.reset();
			ROS_INFO("已清空数据");
			ros::Time start = ros::Time::now();
			while ((ros::Time::now() - start).toSec() < 20 && !this->pointCloudData) {
			ros::spinOnce();
			ros::Duration(0.01).sleep();
			}
			ROS_INFO("收到数据/等待超时");
			std::lock_guard<std::mutex> data_lock(data_mutex_);	// 保护雷达数据,防止在处理雷达数据时，对雷达数据进行更新
			ROS_INFO("数据解析完成");
			// 3. 分析射线传感器扫描数据
			if (this->pointCloudData) {
				ROS_INFO("确认收到数据");
				// 遍历所有点并进行距离检测
				for(int i=0; i<this->pointCloudData->ranges.size(); i++){
					if(this->pointCloudData->ranges.at(i) <= this->ray_check_range){
						/// 有障碍物
						ROS_INFO("[obs_detector] 发现障碍物距离小于限制: %.2f", this->ray_check_range);
						hasObs = true;
						break;  // 退出循环
					}
				}
			}
			ROS_INFO("数据处理完成");
			if(hasObs) break;
		}

		res.obstacle_detected = hasObs;


        // 7. 将探测传感器移回较远点
        geometry_msgs::Point rePos; rePos.x = 100; rePos.y = 100; rePos.z = 0.2;
        this->moveSensor(rePos); 
        return true;
    }

    // 回调函数 保存最新ray扫描数据
    void scanCallback(const sensor_msgs::LaserScan::ConstPtr &msg) {
		std::lock_guard<std::mutex> data_lock(data_mutex_);
        this->pointCloudData = msg;
    }

    // 移动传感器到指定位置
    void moveSensor(const geometry_msgs::Point &point) {
        // 调用/gazebo/set_link_state服务，设置传感器在gazebo中的位置
        gazebo_srv.request.link_state.link_name = "obsDetector_link";
        gazebo_srv.request.link_state.pose.position.x = point.x;
        gazebo_srv.request.link_state.pose.position.y = point.y;
        gazebo_srv.request.link_state.pose.position.z = point.z;
        if (client_callSetLinkState.call(gazebo_srv)) {
            ROS_INFO("[obs_detector] 移动 障碍物探测传感器 到 (%.2f, %.2f, %.2f)坐标处成功...",
                        point.x, point.y, point.z);
        } else {
            ROS_ERROR("[obs_detector] 移动 障碍物探测传感器 失败...");
        }
    }


//// 成员属性
    ros::NodeHandle nh_;
    // 接收检测障碍物请求的服务
    ros::ServiceServer service_;

    // 调用/gazebo/set_link_state服务的客户端
    ros::ServiceClient client_callSetLinkState;
    gazebo_msgs::SetLinkState gazebo_srv;

    // 接受雷达探测传感器数据的订阅者
    ros::Subscriber scan_sub_;
    sensor_msgs::LaserScan::ConstPtr pointCloudData;

    // 扫描周围障碍物时检测的距离阈值
    double ray_check_range = 0.3;


	
};


int main(int argc, char **argv) {

    ros::init(argc, argv, "obstacle_detector_server");
    ObstacleDetector detector;

    // 使用AsyncSpinner实现多线程
    ros::AsyncSpinner spinner(2); // 使用3个线程处理3个回调函数...ros将会自动管理
    spinner.start();

    ros::waitForShutdown();
    return 0;
}