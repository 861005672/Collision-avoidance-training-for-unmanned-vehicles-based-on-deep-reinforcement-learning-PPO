#! /home/zx567/Learning/pytorch_env/bin/python3

# 本文件实现 与gazebo环境进行交互:
# 1. 向gazebo中的环境发布指令
# 2. 订阅获取gazebo中机器人的雷达消息、机器人的目标点信息、机器人当前状态信息等，组合成机器人的当前状态，返回
# 3. 根据雷达消息、目标点信息等，计算机器人的立即奖励



import sys
import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('zxcar_rl')
sys.path.insert(0,package_path + "/scripts")


import rospy
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist,Point
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState,ModelStates
from zxcar_base.srv import ObstacleDetection
import utils
from sensor_msgs.msg import Joy


import numpy as np




PI = 3.1415926



class GZ_RL_ENV:

    """ 构造函数 """
    def __init__(self,ns="car1"):
        # 初始化节点
        rospy.init_node('GZ_RL_ENV', anonymous=True)  # anonymous=True 使多个本节点的实例可以被启动，避免冲突

        # 小车名称
        self.ns = ns
        # 参数配置
        self.safety_distance = 0.28     # 安全距离（米）
        self.goal_tolerance = 0.5       # 目标点容差（米）
        self.rand_posx_max = 18.0        # 生成随机点时的地图宽度范围
        self.rand_posy_max = 18.0        # 生成随机点时的地图高度范围
        self.ctr_interval = 0.1         # 控制频率， 多少秒控制一次
        self.radar_data_dim = 150       # 雷达数据的维度(原始数据)
        self.radar_state_dim = 30       # 雷达状态的维度(处理过后)
        self.range_max = 5.0            # 雷达数据的最大值
        self.max_v = 1.0                # 最大线速度
        self.max_a = 2.0                # 最大角速度
        self.max_dis = 17.0             # 距离目标点的最远距离限制
        self.max_step = 800             # 最大步数限制
        self.isUseOdom = False          # 是否使用里程计作为小车的位姿传感器，若不使用则直接获取gazebo中模型的准确状态，若使用则利用里程计消息来获取位姿
        self.AWARD = 500                # 到达目标点的奖励
        self.PUNISH = -500              # 碰撞、超出范围的奖励
        self.action_dim = 6             # 输入的动作维度


        self.discrete_move_dict = {
            0: (0.2*self.max_v, 1.0*self.max_a),   # 左急转 
            1: (1.0*self.max_v, 0.0*self.max_a),   # 快速直行 
            2: (0.2*self.max_v, -1.0*self.max_a),  # 右急转 
            3: (0.8*self.max_v, 0.6*self.max_a),   # 左缓转
            4: (0.8*self.max_v, -0.6*self.max_a),  # 右缓转 
            5: (0.2*self.max_v, 0.0*self.max_a),   # 缓慢直行 
            6: (-0.5*self.max_v, 0.0*self.max_a),  # 倒退 
            7: (0.0*self.max_v, 0.0*self.max_a)    # 停止 
        }
        # 小车的绝对状态   [px, py, gx, gy, psi, v_linear, v_angular, ld[0],...]  [自身位置x,自身位置y,目标点位置x,目标点位置y,自身航向角,自身线速度,自身角速度, 雷达状态[0],...]
        self.abs_state = np.zeros(7 + self.radar_state_dim)
        # 环境输出的小车相对状态   [action_linear, action_angular, dx, dy, alpha, psi, v_linear, v_angular, ld[0],...]  [指令线速度,指令角速度，距离目标点的距离x,距离目标点的距离y,目标点的方向, 自身航向角,自身线速度,自身角速度, 雷达状态[0],...]
        self.relative_state = np.zeros(8 + self.radar_state_dim)
        # 小车最新的雷达原始数据信息
        self.latest_scan = np.zeros(self.radar_data_dim)
        # 小车最新的雷达状态信息
        self.radar_state = np.zeros(self.radar_state_dim)
        # 小车状态归一化时，各个状态维度的最大值
        self.state_upperbound = np.hstack((np.array([self.max_v, self.max_a, self.max_dis, self.max_dis, PI, PI, self.max_v, self.max_a]), np.full(self.radar_state_dim,self.range_max)))
        # 小车当前回合已走过的步数
        self.stepNum = 0
        # 上一个时刻小车到目标点的距离
        self.d2goal_pre = 0
        # 当前时刻小车到目标点的距离
        self.d2goal_now = 0
        # 上一时刻目标点的方向
        self.a2goal_pre = 0
        # 当前时刻目标点的方向
        self.a2goal_now = 0

        # 输出的状态维度:
        self.state_dim = self.relative_state.shape[0]



        # 用于检测到达目标点、发生碰撞的一些bool值变量
        self.isCollision = False            # 发生碰撞的标志位
        self.isReach = False                # 到达目标点的标志位
        self.isOutMaxStep = False           # 是否超出最大步数限制
        self.isOutRange = False             # 是否超出最大活动范围(距离目标点过远)
        self.isReseting = False             # 是否正在进行初始化操作
        self.isGetModelState = False
        self.isGetOdom = False
        self.isGetScan = False


        # Gazebo服务客户端（用于重置位置）
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        # 雷达消息订阅者d
        self.laser_sub = rospy.Subscriber(self.ns+"/scan", LaserScan, self._laser_callback_)
        # 里程消息订阅者
        self.odom_sub = rospy.Subscriber(self.ns+"/odom", Odometry, self._odom_callback_)
        # 模型状态消息订阅者
        self.car_state_sub = rospy.Subscriber('/gazebo/model_states',ModelStates, self._model_state_callback_)
        # 控制指令发布者
        self.cmd_pub = rospy.Publisher(self.ns+"/cmd_vel", Twist,queue_size=10)
        # 障碍物检测服务调用的客户端， 用于重置目标点
        self.obsDetect_clt = rospy.ServiceProxy('/obs_detector', ObstacleDetection)


        # # 进入ros的事件循环
        # rospy.spin()

    

    """ 执行一步动作， 返回值 s_, r, dw, tr, info """
    def step(self, action):
        self.stepNum += 1                                               # 回合步数 +1
        self._ctr_robot_discrete_(action)                               # 控制小车移动
        rospy.sleep(self.ctr_interval)                                  # 延迟一段时间以执行动作

        self._update_state_(action)                                     # 更新当前小车的状态向量(包含abs_state和relative_state)、相关标志位和全局变量的更新在此函数中进行
        reward = self._calculate_reward_(action)                        # 计算立即奖励
        dw = self.isCollision or self.isReach                           # 构造dw
        tr = self.isOutMaxStep or self.isOutRange                       # 构造tr
        info = dict(abs_state=self.abs_state, stp_cnt=self.stepNum)     # 构造info

        self._auto_reset_()                                             # 自动检测是否需要重置小车，若需要，根据不同情况采用不同的重置策略

        return self.relative_state.copy(),reward,dw,tr,info



    """ 重置任务, 返回值 s, info"""
    def reset(self):
        self.isReseting = True                                       # 此标志位置为True，防止重置期间，再次触发重置操作
        self._reset_robot_position_()                                # 重置小车位置
        self._reset_goal_position_()                                 # 重置目标点位置
        self._ctr_robot_discrete_(7)                                 # 发布指令使小车停止

        self.isCollision = False                                     # 重置标志位
        self.isReach = False                                         # 重置标志位
        self.isOutRange = False                                      # 重置标志位
        self.isOutMaxStep = False                                    # 重置标志位
        self.stepNum = 0                                             # 当前步数设置为0
        self.d2goal_now = utils.getDistance(Point(self.abs_state[0],self.abs_state[1],0.0), Point(self.abs_state[2],self.abs_state[3],0.0)) # 上一时刻小车距离目标点的距离
        self.a2goal_now = utils.getTgtAgl(px=self.abs_state[0],py=self.abs_state[1],gx=self.abs_state[2],gy=self.abs_state[3],psi=self.abs_state[4])

        rospy.sleep(0.5)                                             # 延时0.5秒，等待稳定
        self.isReseting = False                                      # 恢复 isReseting 标志位，等待下一次重置

        self._wait_for_data()
        self._update_state_(7)                                       # 更新当前的小车状态
        info = dict(abs_state=self.abs_state,stp_cnt=self.stepNum)   # 构造info

        return self.relative_state.copy(),info



    """ 自动重置任务 """
    def _auto_reset_(self):
        # 若发生碰撞，则重置起点和目标点
        if self.isCollision:
            self.isReseting = True                                       # 此标志位置为True，防止重置期间，再次触发重置操作
            self._ctr_robot_discrete_(7)                                 # 发布指令使小车停止
            self._reset_robot_position_()                                # 重置小车位置
            self._reset_goal_position_()                                 # 重置目标点位置
        # 若到达目标点、超出步数范围、超出距离范围，则只重置目标点位置
        elif self.isReach or self.isOutMaxStep or self.isOutRange:
            self.isReseting = True                                       # 此标志位置为True，防止重置期间，再次触发重置操作
            self._ctr_robot_discrete_(7)                                 # 发布指令使小车停止
            self._reset_goal_position_()                                 # 重置目标点位置
        # 若发生了自动重置，则重置标志位等
        if self.isReseting:
            rospy.sleep(0.5)                                             # 延时0.5秒，等待稳定
            self.isReseting = False                                      # 恢复 isReseting 标志位，等待下一次重置
            self._ctr_robot_discrete_(7)                                 # 发布指令使小车停止
            self.isCollision = False                                     # 重置标志位
            self.isReach = False                                         # 重置标志位
            self.isOutRange = False                                      # 重置标志位
            self.isOutMaxStep = False                                    # 重置标志位
            self.stepNum = 0                                             # 当前步数设置为0
            self.d2goal_now = utils.getDistance(Point(self.abs_state[0],self.abs_state[1],0.0), Point(self.abs_state[2],self.abs_state[3],0.0)) # 上一时刻小车距离目标点的距离
            self.a2goal_now = utils.getTgtAgl(px=self.abs_state[0],py=self.abs_state[1],gx=self.abs_state[2],gy=self.abs_state[3],psi=self.abs_state[4])



    """ 获取机器人立即奖励 """
    def _calculate_reward_(self,action):
        # R_distance 朝目标点移动时得分，背离时扣分。 范围(-1,1)
        R_distance = utils.clamp(((self.d2goal_pre - self.d2goal_now)/(self.max_v*self.ctr_interval)),-1,1)
        # R_orientation 朝目标点转动时得分，否则扣分。 范围(-1,1)
        R_orientation = utils.clamp(((utils.getAbs(self.a2goal_pre) - utils.getAbs(self.a2goal_now))/(self.max_a/PI*self.ctr_interval)),-1,1) 
        # 鼓励使用直行动作
        R_forward = self.relative_state[6]
        # 不鼓励使用急转弯、倒退、停止和缓慢直行动作
        R_retreat_slowdown = (action==0) + (action==2) + (action==5) + (action==6) + (action==7)

        # 总奖励
        reward = 5*R_distance + 5*R_orientation + 1.0*R_forward - 1.0*R_retreat_slowdown
        
        # print(f"R_distance: {round(R_distance,2)}, R_orientation: {round(R_orientation,2)}, R_forward: {round(R_forward,2)}, R_retreat_slowdown: {round(R_retreat_slowdown,2)}, reward: {round(reward,2)}")


        # 特殊奖励(reach,collision,outrange,...)
        if self.isReach: 
            reward = self.AWARD
        if self.isCollision: 
            reward = self.PUNISH
        if self.isOutRange: 
            reward = self.PUNISH

        return reward
    


    """ 计算更新机器人当前的绝对状态，并返回归一化后的相对状态 
        其中绝对状态是: [px, py, gx, gy, psi, v_linear, v_angular, ld[0],...]
        相对状态是: [action_linear, action_angular, dx, dy, alpha, psi, v_linear, v_angular, ld[0],...]"""
    def _update_state_(self,action):
        # 阻塞等待回调函数获取到数据， 若没有数据收到，则返回
        # if not self._wait_for_data():
        #     return None
        
        # 更新上一时刻小车距离目标点的距离
        self.d2goal_pre = self.d2goal_now
        # 更新上一时刻目标点方向
        self.a2goal_pre = self.a2goal_now
        # 更新当前小车距离目标点的距离
        self.d2goal_now = utils.getDistance(Point(self.abs_state[0],self.abs_state[1],0.0), Point(self.abs_state[2],self.abs_state[3],0.0)) 
        # 更新当前目标点的方向
        self.a2goal_now = utils.getTgtAgl(px=self.abs_state[0],py=self.abs_state[1],gx=self.abs_state[2],gy=self.abs_state[3],psi=self.abs_state[4])

        # 原始雷达数据为: self.latest_scan, 预处理后的雷达数据为: self.radar_state， 注意这里还没有归一化，后面统一对abs_state归一化
        self.radar_state = utils.preDoScan(self.latest_scan, self.radar_data_dim, self.radar_state_dim, self.range_max)
        # 更新机器人的绝对状态(前面的px,py,psi,v_linear,v_angular在odom消息或modelstates消息回调中更新了，gx,gy在重置小车时更新了，这里只需要更新ld数据,这里的绝对状态中的ld数据的维度是radar_state_dim，不是原始雷达数据维度radar_data_dim)
        self.abs_state[-self.radar_state_dim:] = self.radar_state
        # 此时绝对状态已经更新完毕，需要将先将绝对状态转换为相对状态，相对状态是输出给神经网络的状态，包含了小车的当前动作
        # 赋值小车当前的指令动作,作为状态 -> relative_state
        self.relative_state[0:2] = self.discrete_move_dict[action]
        # 计算目标点相对位置dx, dy -> relative_state
        self.relative_state[2:4] = self.abs_state[2:4]-self.abs_state[0:2]
        # 计算目标点的方向 -> relative_state
        self.relative_state[4] = utils.getTgtAgl(px=self.abs_state[0],py=self.abs_state[1],gx=self.abs_state[2],gy=self.abs_state[3],psi=self.abs_state[4])
        # 赋值小车的姿态 -> relative_state
        self.relative_state[5:8] = self.abs_state[4:7]
        # 赋值小车的激光雷达状态 -> relative_state
        self.relative_state[-self.radar_state_dim:] = self.abs_state[-self.radar_state_dim:]

        # 赋值 isOutMaxStep 若超出步数限制，则isOutMaxStep置为true
        if self.stepNum > self.max_step and not self.isOutMaxStep:
            self.isOutMaxStep = True

        # 归一化状态向量
        self.relative_state = self.relative_state / self.state_upperbound

        return self.relative_state



    """ 重置机器人位置 """
    def _reset_robot_position_(self):
        try:
            state = ModelState()
            state.model_name = self.ns
            state.pose = utils.generate_random_pose_by_map2d(self.rand_posx_max, self.rand_posy_max, 1000)
            # state.pose = utils.generate_random_pose(self.rand_posx_max, self.rand_posy_max,1000, self.obsDetect_clt)
            self.abs_state[0] = state.pose.position.x
            self.abs_state[1] = state.pose.position.y
            self.set_state(state)
        except Exception as e:
            rospy.logerr(f"机器人{self.ns}位置重置失败: {str(e)}")



    """ 重置目标点位置 """
    def _reset_goal_position_(self):
        try:
            state = ModelState()
            state.model_name = self.ns+'_tgt_point'  # 替换为实际模型名称
            state.pose = utils.generate_random_pose_by_map2d(self.rand_posx_max, self.rand_posy_max,self.max_dis, center_pos=Point(self.abs_state[0],self.abs_state[1],0.0))
            # state.pose = utils.generate_random_pose(self.rand_posx_max, self.rand_posy_max,self.max_dis, self.obsDetect_clt, center_pos=Point(self.abs_state[0],self.abs_state[1],0.0))
            self.set_state(state)
            # rospy.loginfo("目标点位置已重置到: {}".format(state.pose.position))
            # 更新小车的绝对状态
            self.abs_state[2] = state.pose.position.x
            self.abs_state[3] = state.pose.position.y
        except Exception as e:
            rospy.logerr(f"机器人{self.ns}的目标点位置重置失败: {str(e)}")



    """ 发布cmd_vel话题, 控制小车 """
    def _ctr_robot_discrete_(self, action):
        cmd_vel = Twist()                             # 初始化控制指令
        v_l, v_a = self.discrete_move_dict[action]
        cmd_vel.linear.x = v_l
        cmd_vel.angular.z = v_a
        # 发布话题                             
        self.cmd_pub.publish(cmd_vel)                   



    """ 处理里程计数据，检测是否到达目标点 """
    def _odom_callback_(self, data):
        if self.isReseting: return
        if self.isUseOdom:
            # 获取小车当前的位置和姿态
            current_position, current_psi = utils.getPosPsi(data)
            # 计算到目标点的距离
            goal = Point()
            goal.x = self.abs_state[2]
            goal.y = self.abs_state[3]
            distance = utils.getDistance(current_position, goal)
            # 到达目标检测逻辑
            if distance < self.goal_tolerance and not self.isReach:
                rospy.loginfo("小车到达目标点!!!")
                self._ctr_robot_discrete_(7)                                 # 发布指令使小车停止
                self.isReach = True
            if distance > self.max_dis and not self.isOutRange:
                rospy.loginfo("小车超出范围限制!!!")
                self._ctr_robot_discrete_(7)                                 # 发布指令使小车停止
                self.isOutRange = True
            self.isGetOdom = True



    """ 处理模型状态数据的回调函数"""
    def _model_state_callback_(self,msg):  # [px, py, gx, gy, psi, v_linear, v_angular, ld[0],...]
        if self.isReseting: return
        if not self.isUseOdom:
            posx, posy, psi, v_l, v_a = utils.getPosPsiV(msg,self.ns)   # 解析ModelStates消息数据
            self.abs_state[0] = posx
            self.abs_state[1] = posy
            self.abs_state[4] = psi
            self.abs_state[5] = v_l
            self.abs_state[6] = v_a

            # 计算到目标点的距离
            goal = Point()
            goal.x = self.abs_state[2]
            goal.y = self.abs_state[3]
            cur_pos = Point()
            cur_pos.x = self.abs_state[0]
            cur_pos.y = self.abs_state[1]
            distance = utils.getDistance(cur_pos, goal)
            # 到达目标检测逻辑
            if distance < self.goal_tolerance and not self.isReach:
                rospy.loginfo("小车到达目标点!!!")
                self.isReach = True
                self._ctr_robot_discrete_(7)                                 # 发布指令使小车停止
            # 距离目标点过远检测逻辑
            if distance > self.max_dis and not self.isOutRange:
                rospy.loginfo("小车超出范围限制!!!")
                self.isOutRange = True
                self._ctr_robot_discrete_(7)                                 # 发布指令使小车停止
            self.isGetModelState = True



    """ 处理激光雷达数据，检测碰撞风险 """
    def _laser_callback_(self, data):
        if self.isReseting: return
        min_distance = min(data.ranges)
        # 检查是否有无效数据
        if math.isinf(min_distance) or math.isnan(min_distance):
            return
        
        # rospy.loginfo(f"数据: {data.ranges}")
        # 储存最新的激光雷达数据， copyto是切片赋值，避免产生新内存分配，效率更高,且可以进行自动类型转换，更安全
        np.copyto(self.latest_scan, data.ranges)

        # 碰撞检测逻辑
        if min_distance < self.safety_distance and not self.isCollision:
            # rospy.loginfo(f"min_distance: {min_distance}")
            rospy.loginfo("小车发生碰撞!!!")
            self.isCollision = True
            self._ctr_robot_discrete_(7)                                 # 发布指令使小车停止
        self.isGetScan = True



    """ 阻塞等待回调函数获取到数据"""
    def _wait_for_data(self,timeout=10.0):
        start_time = rospy.Time.now()
        while (not self.isGetScan) or (not self.isGetModelState and not self.isUseOdom) or (not self.isGetOdom and self.isUseOdom):
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.ERROR("等待数据超时!")
                return False
        self.isGetModelState = False
        self.isGetOdom = False
        self.isGetScan = False
        return True



    """ 返回环境的名称 """
    def name(self):
        return 'GZ_RL_ENV'
    


    """ 关闭环境 """
    def close(self):
        rospy.loginfo("关闭Gazebo RL环境")
        self.laser_sub.unregister()          # 取消雷达订阅
        self.odom_sub.unregister()           # 取消里程计订阅
        self.car_state_sub.unregister()      # 取消模型状态订阅
        self.cmd_pub.unregister()             # 取消控制指令发布
        rospy.signal_shutdown("Environment closed")