#! /home/zx567/Learning/pytorch_env/bin/python3

# 路径相关设置
import sys
import rospkg
rospack = rospkg.RosPack()
package_path = rospack.get_path('zxcar_rl')
sys.path.insert(0,package_path + "/scripts/ppo")
sys.path.insert(0,package_path + "/scripts")

model_path = package_path+"/scripts/ppo/model/"
runs_path = package_path+"/scripts/ppo/runs/"

# 导入 Sparrow_V3 环境包
from utils import str2bool
import utils

# 导入其他功能包
from datetime import datetime
from ppo import PPO_agent
import os, shutil
import argparse
import torch

from ppo_gazebo_env import GZ_RL_ENV

parser = argparse.ArgumentParser()

'''PPO 算法 相关配置'''
parser.add_argument('--dvc', type=str, default='cuda', help='网络计算设备')
parser.add_argument('--write', type=str2bool, default=False, help='是否用tensorboard记录训练过程')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='是否载入模型')
parser.add_argument('--ModelIdex', type=int, default=1000, help='载入哪一个模型(模型文件名称最后的表示训练回合数的整数值),必须指定')
parser.add_argument('--ModelName', type=str, default='model', help='模型文件夹名称，必须指定')
parser.add_argument('--ModelTime', type=str, default='2000-12-28_00_00', help='模型文件夹的时间后缀，必须指定')


parser.add_argument('--seed', type=int, default=0, help='随机数种子')
parser.add_argument('--Max_train_eps', type=int, default=int(6000), help='最大训练回合数')
parser.add_argument('--Max_train_time', type=int, default=int(30), help='最大训练分钟数')
parser.add_argument('--end_mode', type=str2bool, default=True, help='按照最大训练时间来结束还是按照最大回合数来结束训练,true:回合数, false:时间')
parser.add_argument('--save_interval', type=int, default=int(200), help='多少回合保存一次模型')


parser.add_argument('--gamma', type=float, default=0.99, help='agent的gamma参数')
parser.add_argument('--net_width', type=int, default=256, help='网络隐藏层宽度')
parser.add_argument('--net_depth', type=int, default=2, help='网络隐藏层深度,默认隐藏层深度为1')
parser.add_argument('--lr_critic', type=float, default=0.001, help='Critic网络学习率')
parser.add_argument('--lr_actor', type=float, default=0.001, help='Actor网络学习率')


parser.add_argument('--batch_size', type=int, default=64, help='每次经验回放更新critic网络时,采样的个数')
parser.add_argument('--train_exp_num', type=int, default=1024, help='经验回放区的大小')
parser.add_argument('--lambd', type=float, default=0.95, help='用于GAE计算')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='adv优势函数是否归一化')
parser.add_argument('--entropy_init', type=float, default=0.2, help='熵正则项系数的初始值')
parser.add_argument('--entropy_min', type=float, default=0.00001, help='熵正则项系数的最小值')
parser.add_argument('--entropy_decay', type=float, default=0.98, help='熵正则项系数衰减率')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO 裁减率')
parser.add_argument('--K_epochs', type=int, default=10, help='每次train,训练的次数')
parser.add_argument('--l2_reg', type=float, default=0, help='critic网络权重参数的正则化')



parser.add_argument('--car_name', type=str, default='car1', help='要训练的小车名称')



opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print("ppo_train_gazebo.py 运行参数: ")
print(opt)
print("===========================================================================")





def main():
    env = GZ_RL_ENV(opt.car_name)              	# 创建同Gazebo环境交互的ros虚拟环境 GZ_RL_ENV
    opt.state_dim = env.state_dim           # opt的状态维数设置,以传递给agent
    opt.action_dim = env.action_dim         # opt的动作维度设置,以传递给agent
    MaxEpisodes = opt.Max_train_eps         # 保存最大的训练回合数
    MaxMinute = opt.Max_train_time          # 保存最大的训练时间
    algo_name = 'ppo'                       # 设置算法名称,用于保存模型和保存日志
    model_name = opt.ModelName              # 设置模型名称,用于保存模型和保存日志以及载入模型
    model_time = opt.ModelTime              # 设置模型时间后缀,用于载入模型
    model_idex = opt.ModelIdex              # 设置模型idex,用于载入模型

    # Seed Everything  设置随机种子和PyTorch的相关配置，以保证实验的可复现性。
    torch.manual_seed(opt.seed)  # 设置CPU上PyTorch的随机种子
    torch.cuda.manual_seed(opt.seed)  # 设置GPU上PyTorch的随机种子
    torch.backends.cudnn.deterministic = True  # 让cudnn的卷积算法确定性，保证每次结果一致
    torch.backends.cudnn.benchmark = False  # 禁用cudnn的自动优化，进一步保证结果可复现

    print(f"Car: {opt.car_name}, Env: {env.name}, Model: {model_name}, state_dim: {opt.state_dim}, action_dim: {opt.action_dim}, \
            Seed: {opt.seed}, max_eps: {MaxEpisodes}, max_time: {MaxMinute}, end_mode: {opt.end_mode}")

    print("===========================================================================")

    # 利用tensorboard对数据进行可视化，此处对tensorboard进行初始化， tensorboard日志文件在runs文件夹下
    # 使用 tensorboard --logdir=... 查看日志文件
    time_start = str()
    time_start = str(datetime.now())[0:-10]
    time_start = time_start[0:13] + '_' + time_start[-2::]
    time_start = time_start.replace(' ', '_')
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        writepath = runs_path + 'train--{}--S{}--{}--{}'.format(algo_name,opt.seed,opt.ModelName,time_start)
        if os.path.exists(writepath): 
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    #创建agent模型
    agent = PPO_agent(time_start, model_name, **vars(opt))          
    # 若选择了加载模型，则agent通过本地模型载入
    if opt.Loadmodel:   
        agent.load(model_name, model_time, algo_name, model_idex)


    # 初始化
    total_episode = 0                                       # 当前总的回合数
    s, info = env.reset()                                   # 重置环境，获取初始的状态,s是numpy数组
    eps_r = 0                                               # 回合总奖励
    exp_num = 0                                             # 经验数量
    train_time = 0                                          # 训练次数
    time_now = datetime.now()
    time_has = utils.getRunTime(time_start,time_now)
    while (opt.end_mode and total_episode < MaxEpisodes) or (not opt.end_mode and time_has[1] < MaxMinute) :                      # 未达到最大训练回合数时
    # while (opt.end_mode and total_episode < MaxEpisodes) :                      # 未达到最大训练回合数时
        done = False                                # 是否有机器人的回合结束了
        # 回合开始前，停止一步，获得初始状态
        a_init = 7             
        s,_1,_2,_3,_4 = env.step(a_init)
        # 与环境交互
        while not done:
            # 选择动作
            a,pi_a = agent.select_action(s)            # 得出的a 和 pi_a 都是tensor数据类型,代表每个机器人的动作和pi_a
            # 在环境中采取策略给出的动作, 返回下一状态s_next,立即奖励r,done,truncated和其他信息  
            s_, r, dw, tr, info = env.step(a)
            # 是否done掉
            done = (dw | tr) 
            # print(f"s: {s}, \a: {a}, \n s': {s_}, \n r: {r}, done: {done}, dw: {dw}, tr: {tr}")
            # print("-------------------------------------------------------------------------------------------------------------------")
            # 存储单步经验到agent
            agent.put_data(s, a, r, s_, pi_a, done, dw, exp_idx=exp_num % opt.train_exp_num)         # 存储单步经验到agent
            # 将s 替换为 s_next，以便下一步执行
            s = s_
            # 累加回合奖励
            eps_r += r 
            # 累加经验数
            exp_num += 1
            # 到点就训练一次agent
            if exp_num % opt.train_exp_num == 0:
                s,_1,_2,_3,_4 = env.step(a_init)
                time_now = datetime.now()
                train_time += 1
                print(f"===============================  总经验数量:{exp_num}  开始训练")
                agent.train()
                print(f"===============================  第{train_time}次训练结束")

            
        time_now = datetime.now()
        time_has = utils.getRunTime(time_start,time_now)
        # 回合数 + 1 
        total_episode += 1                                
        # 日志记录 and 评估模型  
        if opt.write:
            writer.add_scalar('ep_r', eps_r, global_step=total_episode if opt.end_mode else time_has[2])
        print(
            ' episode:{}'.format(int(total_episode)),
            ' seed:',opt.seed,
            ' entropy_coef:', round(agent.entropy_coef,6),
            ' exp_num:', exp_num,
            ' has_time:', time_has[2], 's',
            ' eps_r:', int(eps_r)
        )
        print("---------------------------------------------------------------")
        #将回合总奖励置0
        eps_r = 0

        # 保存模型
        if total_episode % opt.save_interval == 0 and opt.write == True:
            agent.save(algo_name,int(total_episode))


    env.close()         # 关闭sparrow环境
    time_now = datetime.now
    print(f"程序运行结束，总时长: {utils.getRunTime(time_start,time_now)[0]}, 收集到的经验数: {exp_num}, 共训练{train_time}次")


# main函数, 程序入口
if __name__ == '__main__':
    main()