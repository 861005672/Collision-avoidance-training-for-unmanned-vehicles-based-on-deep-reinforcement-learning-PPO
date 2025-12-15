# PPO算法训练gazebo环境 使用说明

## 概述

1. “gazebo_ros_2Dmap_plugin-master”: github上大佬写的自动生成gazebo环境的灰度图像地图的一个插件功能包，参考:
https://github.com/marinaKollmitz/gazebo_ros_2Dmap_plugin
在这里我主要用于算法训练时随机初始化机器人和目标点的位置

2. "zxcar_base": 实现了一些基础的东西，包括机器人urdf模型、障碍物探测（用于在训练中自动重置机器人位置）、设置机器人模型颜色、调用gazebo_ros_2dmap_plugin插件等功能

3. "zxcar_rl": 目前仅实现了PPO算法，包括:

		模拟gym标准接口与gazebo环境进行交互的脚本(ppo_gazebo_env.py)

		算法训练脚本(ppo_train.py)，

		算法测试脚本(ppo_test.py)，

		环境交互测试脚本(ppo_test_env.py)，

		PPO算法实现脚本(ppo.py)
4. 测试没问题的环境包括:
		
		ros noetic 
		ubuntu 20.04
		pytorch 2.4.1 + cu121
		tensorboard 2.14.1


## 一、编译工作空间
1. 将“gazebo_ros_2Dmap_plugin-master”、“zxcar_rl”、”zxcar_base“三个功能包复制到一个工作空间的src目录下
2. 终端进入工作空间，运行
```
catkin_make
```
进行编译

## 二、启动gazebo仿真环境
在工作空间中，先后执行:
```
source ./devel/setup.bash
```
```
roslaunch zxcar_rl start_rl_gazebo.launch
```

## 三、启动ppo训练gazebo中的小车
新建终端并进入工作空间，先后执行:
```
source ./devel/setup.bash
```
```
rosrun zxcar_rl ppo_train.py --car_name car1
```
默认训练的回合总数为6000回合，可以通过命令行参数修改，也可以直接在代码中修改参数，具体看ppo_train.py代码的开头部分


## 四、查看训练中回合总奖励曲线
训练中的回合总奖励将通过tensorboard记录在
```
$(工作空间)/zxcar_rl/scripts/ppo/runs
```
中，在该目录下执行指令:
```
tensorboard --logdir .
```
并根据提示用浏览器进入指定的地址即可查看训练曲线

## 五、测试训练效果
训练结束后，模型将保存在
```
$(工作空间)/zxcar_rl/scripts/ppo/model
```
中，模型名称默认为model，后面接了算法训练的日期和时间，利用这两个来启动不同的模型进行测试。
下面举例说明：
1. 首先需要按照第三步中启动gazebo环境
2. 假设需要利用model文件夹内的 "20_20_obs_test2-2025-05-27_18_42"模型进行测试，则在终端中输入指令:
```
rosrun zxcar_rl ppo_test.py --car_name car1 --ModelName 20_20_obs_test2 --ModelTime 2025-05-27_18_42 --ModelIdex 6000
```
其中,--ModelIdex 6000 指定了使用算法训练到6000回合时的模型进行测试。