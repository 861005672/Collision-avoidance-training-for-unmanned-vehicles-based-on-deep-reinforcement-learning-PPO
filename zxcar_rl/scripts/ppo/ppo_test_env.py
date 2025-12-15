#! /home/zx567/Learning/pytorch_env/bin/python3

import sys,os
import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('zxcar_rl')
sys.path.insert(0,package_path + "/scripts/ppo")
import random
import numpy as np

import rospy
from sensor_msgs.msg import Joy



from ppo_gazebo_env import GZ_RL_ENV



test_action = 7 

def joy_callback(joy_msg):
    global test_action

    if joy_msg.axes[0] == 1.0:
        test_action = 0      # 左急转
    elif joy_msg.axes[0] == -1.0:
        test_action = 2      # 右急转
    elif joy_msg.axes[1] == 1.0:
        test_action = 1      # 快速直行
    elif joy_msg.axes[1] == -1.0:
        test_action = 6      # 倒退
    elif joy_msg.buttons[2] == 1.0:
        test_action = 3      # 左缓转
    elif joy_msg.buttons[1] == 1.0:
        test_action = 4      # 右缓转
    elif joy_msg.buttons[3] == 1.0:
        test_action = 5      # 缓慢直行
    else:
        test_action = 7      # 停止
    

def main():
    env = GZ_RL_ENV()
    env.reset()

    joy_sub = rospy.Subscriber('/joy', Joy, joy_callback)

    while not rospy.is_shutdown():
        action = test_action
        s, r, dw, tr, info = env.step(action)
        print(f"步数: {info['stp_cnt']},\n 执行动作: {action},\n  状态:{np.round(s,2)},\n 奖励:{round(r,3)}\n")
        print("-----------------------------------------------------")


if __name__ == '__main__':
    main()
