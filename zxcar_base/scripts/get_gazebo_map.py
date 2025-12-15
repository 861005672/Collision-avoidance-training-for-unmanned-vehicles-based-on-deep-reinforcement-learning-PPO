#!/usr/bin/env python3

import rospy
import subprocess
from std_srvs.srv import Empty
from gazebo_msgs.srv import GetWorldProperties

    

def call_service(service_name, timeout=10.0):
    """通用服务调用函数"""
    rospy.loginfo(f"等待服务 {service_name} 可用...")
    rospy.wait_for_service(service_name, timeout=timeout)
    
    try:
        service = rospy.ServiceProxy(service_name, Empty)
        response = service()
        rospy.loginfo(f"服务 {service_name} 调用成功")
        return True
    except rospy.ServiceException as e:
        rospy.logerr(f"服务 {service_name} 调用失败: {e}")
        return False


def save_map(map_name="my_map", timeout=30.0):
    """保存ROS地图的主函数"""
    try:
        if rospy.get_param('/use_sim_time',True) == True:
            rospy.sleep(100)
        else:
            rospy.sleep(10)
        
        # 0. 调用 gazebo_ros_2Dmap服务获取gazebo的真值地图
        call_service('/gazebo_2Dmap_plugin/generate_map')

        cmd = f"rosrun map_server map_saver -f {map_name} /map:=/map2d"
        process = subprocess.Popen(cmd, shell=True)
        rospy.loginfo(f"保存地图到 {map_name} 完成")
            
    except rospy.ROSException as e:
        rospy.logerr(f"ROS异常: {e}")
        return False
    except Exception as e:
        rospy.logerr(f"未知异常: {e}")
        return False

if __name__ == '__main__':
    rospy.init_node('get_gazebo_map')
    map_name = rospy.get_param('~map_name', 'default_map')
    success = save_map(map_name)