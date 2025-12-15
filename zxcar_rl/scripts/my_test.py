#! /home/zx567/Learning/pytorch_env/bin/python3


import rospy
import utils
from threading import Event
from nav_msgs.msg import OccupancyGrid
import numpy as np


map_width = 20
map_height = 20
max_dis = 1000

rospy.init_node("test_node")


posx = rospy.get_param('~posx', 0)
posy = rospy.get_param('~posy', 0)

print(f"检测点:({posx},{posy})")

# 初始化地图数据和接收事件
map_data = None
map_received = Event()

# 地图回调函数
def map_callback(msg):
    global map_data
    map_data = msg
    map_received.set()  # 通知地图已接收

# 订阅/map2d话题
rospy.Subscriber('/map2d', OccupancyGrid, map_callback)

# 等待地图数据
if not map_received.wait(30):
    rospy.logerr("超时未收到/map2d话题数据")

# 地图参数
map_info = map_data.info
map_array = np.array(map_data.data).reshape((map_info.height, map_info.width))

# 定义安全阈值
FREE_THRESHOLD = 10  # 0表示完全空闲，-1表示未知，100表示障碍物
# 定义检查半径(米)
CHECK_RADIUS = 0.4
# 转换为地图坐标的检查半径
check_radius_cells = int(CHECK_RADIUS / map_info.resolution)

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
    print(f"x_min:{x_min}, x_max:{x_max}, y_min:{y_min}, y_max:{y_max}")
    print(f"region:\n {region}")
    # 检查区域内是否都是空闲区域
    if np.all(region < FREE_THRESHOLD):
        print("区域内无障碍物")
    else:
        print("区域内有障碍物")
