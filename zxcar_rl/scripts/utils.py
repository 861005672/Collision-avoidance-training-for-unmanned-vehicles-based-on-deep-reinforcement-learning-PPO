#! /home/zx567/Learning/pytorch_env/bin/python3

from geometry_msgs.msg import Point, Pose, Quaternion
import random
from geometry_msgs.msg import Twist
from zxcar_base.srv import ObstacleDetectionRequest
import tf
import math
import rospy
import numpy as np

from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid

from collections import deque
from datetime import datetime
from threading import Event


"""
通过订阅/map2d话题获取随机无障碍物的位姿

参数:
	map_width, map_height: 地图范围大小
	max_dis: 与中心点的最大距离限制
	center_pos: 中心点位置(Point类型)，可选
	timeout: 等待地图数据的超时时间(秒)
"""
def generate_random_pose_by_map2d(map_width, map_height, max_dis, center_pos=None, timeout=5.0):

    # 初始化地图数据和接收事件
    map_data = None
    map_received = Event()
    
    # 地图回调函数
    def map_callback(msg):
        nonlocal map_data
        map_data = msg
        map_received.set()  # 通知地图已接收
    
    # 订阅/map2d话题
    rospy.Subscriber('/map2d', OccupancyGrid, map_callback)
    
    # 等待地图数据
    if not map_received.wait(timeout):
        rospy.logerr("超时未收到/map2d话题数据")
        return None
    
    # 地图参数
    map_info = map_data.info
    map_array = np.array(map_data.data).reshape((map_info.height, map_info.width))
    
    # 定义安全阈值
    FREE_THRESHOLD = 10  # 0表示完全空闲，-1表示未知，100表示障碍物
    
    # 定义检查半径(米)
    CHECK_RADIUS = 0.6
    
    # 转换为地图坐标的检查半径
    check_radius_cells = int(CHECK_RADIUS / map_info.resolution)
    
    # 尝试寻找可行点
    max_attempts = 1000
    for attempt in range(max_attempts):
        # 生成随机点(世界坐标)
        posx = random.uniform(-map_width/2, map_width/2)
        posy = random.uniform(-map_height/2, map_height/2)
        
        # 检查与中心点的距离
        if center_pos is not None:
            dis = getDistance(Point(posx, posy, 0.0), center_pos)
            if dis > max_dis:  # 距离太远
                continue
            if dis < 1:        # 距离太近
                continue
        # 将世界坐标转换为地图坐标
        map_x = int((posx - map_info.origin.position.x) / map_info.resolution)
        map_y = int((posy - map_info.origin.position.y) / map_info.resolution)
        
        # 检查坐标是否在地图范围内
        if 0 <= map_x < map_info.width and 0 <= map_y < map_info.height:
            # 检查周围区域是否无障碍物
            x_min = max(0, map_x - check_radius_cells)
            x_max = min(map_info.width, map_x + check_radius_cells)
            y_min = max(0, map_y - check_radius_cells)
            y_max = min(map_info.height, map_y + check_radius_cells)
            
            # 获取周围区域
            region = map_array[y_min:y_max, x_min:x_max]
            # print(f"x_min:{x_min}, x_max:{x_max}, y_min:{y_min}, y_max:{y_max}")
            # print(f"region: {region}")
            # 检查区域内是否都是空闲区域
            if np.all(region < FREE_THRESHOLD):
                # 构造随机转角
                theta = random.uniform(0, 2*math.pi)
                qz = math.sin(theta/2)
                qw = math.cos(theta/2)
                # 返回随机位置
                return Pose(
                    position=Point(
                        x=posx,
                        y=posy,
                        z=0.0
                    ),
                    orientation=Quaternion(x=0, y=0, z=qz, w=qw)
                )
    
    rospy.logwarn(f"尝试{max_attempts}次后仍未找到合适的随机点")
    return None


""" 调用zxcar_base中的障碍物探测节点服务生成地图范围内的随机位姿 """
def generate_random_pose(map_width, map_height, max_dis, srvClt, center_pos=None):

    # 调用障碍物检测服务，检测随机的点处是否有障碍物
    isok = False
    while not isok:
        # rospy.loginfo("寻找随机的可行点-----------------------------")
        posx = random.uniform(-map_width/2, map_width/2)
        posy = random.uniform(-map_height/2, map_height/2)

        
        if not center_pos == None:
            dis = getDistance(Point(posx,posy,0.0), center_pos)
            if dis > max_dis:  # 若生成的点距离center_pos的距离大于max_dis，则重新生成随机点
                continue
            if dis < 1:        # 若生成的点距离center_pos距离小于2，则重新生成
                continue

        req = ObstacleDetectionRequest()
        req.detection_point.x = posx
        req.detection_point.y = posy
        req.detection_point.z = 0.01
        response = srvClt(req)

        if response.obstacle_detected == False:
            isok = True

    # 构造随机转角
    theta = random.uniform(0, 2*math.pi)
    qz = math.sin(theta/2)
    qw = math.cos(theta/2)
    # 返回随机位置
    return Pose(
        position=Point(
            x=posx,
            y=posy,
            z=0.0
        ),
        orientation=Quaternion(x=0, y=0, z=qz, w=qw)  # 保持默认朝向
    )


""" 根据里程计数据获取机器人位置和航向角 """
def getPosPsi(odomMsg):
    # 获取当前位置
    current_position = odomMsg.pose.pose.position
    # 获取当前航向角
    orientation = odomMsg.pose.pose.orientation
    quaternion = (
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w
    )
    # 将四元数转换为欧拉角（弧度）
    euler = tf.transformations.euler_from_quaternion(quaternion)
    current_psi = euler[2]  # 航向角为绕 Z 轴的旋转（偏航角)
    return current_position, current_psi  # 返回值



def getPosPsiV(modelStatesMsg,ns):
    try:
        # 查找目标模型在列表中的索引
        index = modelStatesMsg.name.index(ns)
        pos = modelStatesMsg.pose[index].position
        orientation = modelStatesMsg.pose[index].orientation
        quaternion = (
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w
        )
        # 将四元数转换为欧拉角（弧度）
        euler = tf.transformations.euler_from_quaternion(quaternion)
        psi = euler[2]  # 航向角为绕 Z 轴的旋转（偏航角)

        twist = modelStatesMsg.twist[index]
        linear_v_global = twist.linear
        # 将全局坐标系下的线速度转换为车体坐标系下的速度
        vx_global = linear_v_global.x
        vy_global = linear_v_global.y
        # 坐标变换矩阵（旋转矩阵）
        vx_body = vx_global * np.cos(psi) + vy_global * np.sin(psi)
        va_body = twist.angular.z
        return pos.x, pos.y, psi, vx_body, va_body

    except ValueError:
        rospy.logwarn_once(f"Model '{ns}' not found in Gazebo!")


""" 计算两点之间的距离 """
def getDistance(point1, point2):
    dx = point1.x - point2.x
    dy = point1.y - point2.y
    distance = math.sqrt(dx**2 + dy**2)
    return distance



""" 计算绝对值 """
def getAbs(value):
    if value < 0:
        return -value
    else:
        return value
    
""" 根据小车位姿和目标点位置计算目标点的方位 """
def getTgtAgl(px, py, gx, gy, psi):    
    # curPoint和tgtPoint
    curPoint = Point(px,py,0.0)
    tgtPoint = Point(gx,gy,0.0)
    # 计算全局坐标系下的坐标差
    dx_world = tgtPoint.x - curPoint.x
    dy_world = tgtPoint.y - curPoint.y
    
    # 将全局坐标差转换到机器人坐标系（考虑航向角旋转）
    dx_robot = dx_world * math.cos(psi) + dy_world * math.sin(psi)
    dy_robot = -dx_world * math.sin(psi) + dy_world * math.cos(psi)
    
    # 计算方位角（atan2自动处理象限，结果范围[-π, π]）
    target_angle = math.atan2(dy_robot, dx_robot)
    
    return target_angle

""" 根据原始的激光雷达数(numpy数组)进行预处理 """
def preDoScan(scanRange, inputDim, outputDim, maxRange):
    # 计算步长和余数
    stride = inputDim // outputDim
    remainder = inputDim % outputDim
    
    # 创建分割点索引
    split_indices = np.ones(outputDim, dtype=int) * stride
    split_indices[:remainder] += 1
    split_indices = np.cumsum(split_indices)[:-1]
    
    # 分割数组并计算每段最小值
    segments = np.array_split(scanRange, split_indices)
    min_distances = np.array([seg.min() for seg in segments])
    
    # 处理无穷大值并归一化
    min_distances[np.isinf(min_distances)] = maxRange
    processed = np.round(min_distances, 2)
    
    return processed




""" 将一个值限制在lower和upper之间 """
def clamp(value,lower,upper):
    return max(lower,min(value,upper))

""" 获取运行时间 """
def getRunTime(startTime, nowTime):
    # 适配自定义格式（假设秒为0）
    if isinstance(startTime, str):
        startTime = datetime.strptime(startTime, "%Y-%m-%d_%H_%M")
        startTime = startTime.replace(second=0)  # 假设秒为0
    if isinstance(nowTime, str):
        nowTime = datetime.strptime(nowTime, "%Y-%m-%d_%H_%M")
        nowTime = nowTime.replace(second=0)      # 假设秒为0
    # 计算时间差
    time_has = nowTime - startTime
    total_seconds = time_has.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours}h:{minutes}m:{seconds}s", int(minutes), int(total_seconds)


"""--------------------------------------------------------------------------------------------------------"""

def str2bool(v):
    '''Fix the bool BUG for argparse: transfer string to bool'''
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1', 'T'): return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0', 'F'): return False
    else: print('Wrong Input Type!')

"""--------------------------------------------------------------------------------------------------------"""
