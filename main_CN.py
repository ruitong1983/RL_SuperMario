import pickle
import random
from collections import deque
from datetime import datetime

import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import *



def arrange(s):
    """
    重新排列数组的维度并添加额外的维度。
    该函数用于处理numpy数组数据，确保其维度满足特定要求，以便进行后续处理或作为模型输入。

    参数:
    s (numpy.ndarray): 一个三维的numpy数组。如果输入不是numpy数组，将被转换为numpy数组。

    返回:
    numpy.ndarray: 一个四维的numpy数组，原始第三维度被移动到第一位，并且在最前面添加了一个额外的维度。
    """
    # 检查输入是否为numpy数组，如果不是，则进行转换
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    # 经过预处理后，输入是三维：84*84*4，每帧的图像素：84*84，共4帧
    assert len(s.shape) == 3
    # 调整数组的维度，将原始第三维度移到最前面，即4帧移到前面，调整后的维度：4*84*84
    ret = np.transpose(s, (2, 0, 1))
    # 在最前面添加一个额外的维度，输出的维度为 1*4*84*84
    return np.expand_dims(ret, 0)

class replay_memory(object):
    def __init__(self, N):
        self.memory = deque(maxlen=N)

    def push(self, transition):
        """
        向内存中添加一个过渡。

        参数:
        transition (object): 要添加到内存中的过渡对象。

        返回:
        无
        """
        # 添加到内存列表中
        self.memory.append(transition)

    def sample(self, n):
        """
        从记忆库中随机选择n个独特的样本。

        参数:
        n (int): 要选择的样本数量。

        返回:
        list: 包含从记忆库中随机选择的n个独特样本的列表。

        说明:
        此方法使用random.sample函数来实现，该函数适用于序列对象。
        self.memory应该是一个序列，如列表或元组，包含所有可选择的样本。
        """
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)

class model(nn.Module):
    def __init__(self, n_frame, n_action, device):   # n_frame：4，n_action：12，device：cuda/cpu
        """
        4 * 84 * 84 -> 32 * 20 * 20 -> 64 * 18 * 18 = 20736 -> 512  -> q 12   每个动作的Q值
                                                                    -> v 1
        Sequential(
        (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
        (1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))  32 * 20 * 20
        (3): Linear(in_features=512, out_features=12, bias=True)
        (4): Linear(in_features=512, out_features=1, bias=True)
        )
        """
        super(model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(20736, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)

        self.device = device

        # 打印每个层的初始化信息
        print(f"初始化 模型")
        print(f"初始化 layer1: {self.layer1}")
        print(f"初始化 layer2: {self.layer2}")
        print(f"初始化 fc: {self.fc}")
        print(f"初始化 q: {self.q}")
        print(f"初始化 v: {self.v}")

    def forward(self, x):
        """
        前向传播函数，处理输入数据并生成Q值。

        该函数首先确保输入数据为torch.Tensor类型，如果不是，则将其转换为torch.Tensor。
        之后，通过一系列的线性层和激活函数对输入数据进行处理，最后计算出优势函数（advantage function）和价值函数（value function），
        并据此生成最终的Q值。

        参数:
        x: 输入数据，可以是任意类型的数据，但推荐使用torch.Tensor类型。

        返回:
        q: 最终的Q值，表示在当前状态下采取各个行动的期望回报。
        """
        # 检查输入数据类型，如果不是torch.Tensor，则将其转换为torch.Tensor
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))                      # 通过第一层线性层和ReLU激活函数处理输入数据
        x = torch.relu(self.layer2(x))                      # 通过第二层线性层和ReLU激活函数进一步处理数据
        x = x.view(-1, 20736)                               # 调整数据形状，以便其可以输入到全连接层
        x = torch.relu(self.fc(x))                          # 通过全连接层和ReLU激活函数处理数据
        # 对决网络，以特征向量作为输入，分别计算优势头和状态价值头
        d = self.q(x)                                       # 优势头输出是一个向量，维度是动作空间的大小：12，表示每个动作的Q值
        v = self.v(x)                                       # 价值状态头是一个实数：维度：1，表示当前状态的价值s
        # 根据优势函数和价值函数计算最终的Q值：
        q = v + (d - 1 / d.shape[-1] * d.max(-1, True)[0])  # 原版：Q = V + (D - D.max().mean())
        # q = v + (d - d.mean())                            # Q = V + D - D.mean()
        return q

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        # 将偏差初始化为一个较小的正值，以避免初始时的激活函数饱和问题
        m.bias.data.fill_(0.01)

def copy_weights(q, q_target):
    q_dict = q.state_dict()
    q_target.load_state_dict(q_dict)

def train(q, q_target, memory, batch_size, gamma, optimizer, device):
    """
    训练神经网络模型。

    参数:
    q: 当前神经网络模型，用于评估状态动作价值。
    q_target: 目标神经网络模型，用于生成目标值。
    memory: 经验回放内存，存储过往经验。
    batch_size: 每次训练的样本数量。
    gamma: 折扣因子，用于计算未来奖励的现值。
    optimizer: 优化器，用于更新神经网络的权重。
    device: 设备信息，决定模型在CPU还是GPU上运行。

    返回:
    loss: 训练损失，表示模型预测值与目标值的误差。
    """
    # 从内存中随机采样一批数据，数据条数由batch_size指定
    s, r, a, s_prime, done = list(map(list, zip(*memory.sample(batch_size))))
    # 将状态和下一状态转换为numpy数组并压缩，(256, 4, 84, 84)--batch:256, 帧：4，高度：84，宽度：84
    state = np.array(s).squeeze()
    state_prime = np.array(s_prime).squeeze()
    reword = torch.FloatTensor(r).unsqueeze(-1).to(device)      # 将奖励、动作、完成标志、转换为torch张量，维度：256 * 1 ，并移动到指定设备
    action = torch.FloatTensor(a).unsqueeze(-1).to(device)
    done = torch.FloatTensor(done).unsqueeze(-1).to(device)
    # 计算目标值，即奖励加上未来状态的最大动作价值，输出维度：256 * 1
    action_max = q(state_prime).max(1)[1].unsqueeze(-1)         # 计算下一个状态的最大动作价值，并转换为索引
    # 从q_target网络中gather出下一状态state_prime的最大动作索引action_max对应的价值，并乘以折扣因子gamma，得到目标值，维度：256 * 1
    with torch.no_grad():
        y = reword + gamma * q_target(state_prime).gather(1, action_max) * done
    # 计算当前状态和动作的Q值，维度：256 * 1
    q_value = torch.gather(q(state), dim=1, index=action.view(-1, 1).long())
    loss = F.smooth_l1_loss(q_value, y).mean()                  # 计算损失，即Q值与目标值之间的差值的平滑L1损失
    optimizer.zero_grad()                                       # 清零优化器的梯度
    loss.backward()                                             # 反向传播计算梯度
    optimizer.step()                                            # 更新模型参数

    return loss

def main(env, q, q_target, optimizer, device):
    t = 0
    gamma = 0.90
    batch_size = 128        # 原版 256

    N = 50000
    eps = 0.001
    memory = replay_memory(N)
    update_interval = 100   # 原版 50
    print_interval = 10

    score_lst = []
    total_score = 0.0
    last_total_score = 0.0
    loss = 0.0

    for k in range(1000000):
        state = arrange(env.reset())                                    # 初始化环境并安排起始状态，State的维度为1*4*84*84
        round_score = 0.0                                               # 每局游戏得分
        done = False                                                    # 初始化游戏是否结束标识
        # 获取当前的系统时间，用于记录游戏开始时的时间，单位到时秒分
        start_time = datetime.now()
        print("！！！开始 | 时间 ===> ", start_time.strftime("%H:%M:%S"))
        # 当游戏没有结束时，继续执行游戏步骤
        while not done:
            env.render()                                                # 查看游戏窗口
            if eps > np.random.rand():                                  # 根据epsilon-贪婪策略选择动作
                action = env.action_space.sample()                      # 探索：随机选择一个动作
            else:
                temp = q(state)
                if device == "cpu":                                     # 利用：选择Q值最高的动作，Q-learning算法
                    action = np.argmax(temp.detach().numpy())
                else:
                    action = np.argmax(temp.cpu().detach().numpy())     # 在GPU上计算并选择动作

            # 执行选定的动作，获取下一个状态、奖励、是否完成等信息
            s_prime, r, done, _ = env.step(action)
            s_prime = arrange(s_prime)                                  # 对下一个状态进行处理
            round_score += r                                            # 累加得分
            r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r      # 对奖励进行裁剪处理，以促进学习效率

            # 将经验（状态、奖励、动作、下一个状态、完成标志）存储到记忆池中
            memory.push((state, float(r), int(action), s_prime, int(1 - done)))
            state = s_prime                                             # 更新当前状态
            stage = env.unwrapped._stage                                # 获取当前环境阶段（可选，根据环境而定）
            # 当记忆池大小超过一定阈值后，开始训练
            if len(memory) > 2000:                                      # 累加损失，进行一轮训练
                loss += train(q, q_target, memory, batch_size, gamma, optimizer, device)
                t += 1                                                  # 增加时间步的计数

            # 每隔一定时间步，同步更新目标网络的权重，t 用于记录时间步，update_interval 用于控制更新间隔
            if t % update_interval == 0 and t != 0:
                copy_weights(q, q_target)                               # 复制当前网络的权重到目标网络，保存网络的权重
                print("保存并覆盖最新的模, 当前训练次数 ==> %d "%t)
                torch.save(q.state_dict(), "mario_q.pth")
                torch.save(q_target.state_dict(), "mario_q_target.pth")
        print("！！！结束 | 时间 ===> ", datetime.now().strftime("%H:%M:%S"))
        total_time = datetime.now() - start_time                        # 取时间的秒部分，作为游戏时间
        print("---本轮游戏时间 =====> %d | %d <=====游戏分数 | 关卡 : " %(int(total_time.total_seconds()), round_score), stage)
        total_score += round_score                                      # 累加全局得分

        if k % print_interval == 0:
            if total_score > last_total_score:                          # 如果当前得分大于上次得分，则打印信息
                print("训练轮数 : %d | 训练次数: %d | 损失值 : %.2f  ====> 更好 <==== %d     | 当前得分: %d    | 上次得分: %d"
                      % (k, t, loss, (total_score - last_total_score) / print_interval,
                         total_score / print_interval,last_total_score / print_interval))
                copy_weights(q, q_target)                               # 复制当前网络的权重到目标网络，保存网络的权重
                os.makedirs("model", exist_ok=True)
                # 保存模型参数到文件中，将变量total_score的值作为模型文件的前缀
                print("备份 更好 模型到指定model目录下...")
                torch.save(q.state_dict(), os.path.join("model",f"{total_score}_mario_q.pth"))
                torch.save(q_target.state_dict(), os.path.join("model",f"{total_score}_mario_q_target.pth"))
                print("备份完成")
                last_total_score = total_score
            else:                                                       # 如果当前得分小于上次得分，则打印信息，但不做模型参数更新
                print("训练轮数 : %d |  训练次数: %d | 损失值 : %.2f ====> 更差 <==== %d     | 当前得分: %d   | 上次得分: %d"
                      % (k, t, loss, (total_score - last_total_score) / print_interval,
                         total_score / print_interval, last_total_score / print_interval))
            score_lst.append(total_score / print_interval)
            total_score = 0
            loss = 0.0
            pickle.dump(score_lst, open("score.txt", "wb"))

if __name__ == "__main__":
    # 定义帧数，用于观察的连续帧数
    n_frame = 4
    gym_super_mario_bros_env = gym_super_mario_bros.make('SuperMarioBros-v0')   # 创建Super Mario Bros环境
    joypad_space_env = JoypadSpace(gym_super_mario_bros_env, COMPLEX_MOVEMENT)  # 使用复杂动作空间包装环境
    env = wrap_mario(joypad_space_env)                                          # 对环境进行一系列预处理
    #打印环境信息
    device = "cuda" if torch.cuda.is_available() else "cpu"                     # 根据CUDA是否可用，设置设备为CUDA或CPU
    print("device : ", device)
    # 初始化评估和目标模型，4帧输入，动作空间大小为env.action_space.n
    q = model(n_frame, env.action_space.n, device).to(device)
    q_target = model(n_frame, env.action_space.n, device).to(device)
    # 如果有模型文件mario_q_target.pth，加载该模型文件参数，否则模型的参数随机化
    try:
        print('Loading pretrained model...')
        q.load_state_dict(torch.load('mario_q.pth', map_location=torch.device(device)))
        q_target.load_state_dict(torch.load('mario_q_target.pth', map_location=torch.device(device)))
        print('找到 预训练模型，加载 预训练 模型...')
    except:
        print('未找到 预训练模型, 初始化 模型参数 ...')
        q.apply(init_weights)
        q_target.apply(init_weights)
    # 使用Adam优化器，学习率为0.0001
    optimizer = optim.Adam(q.parameters(), lr=0.0001, weight_decay=1e-5)
    # 调用主函数进行训练
    main(env, q, q_target, optimizer, device)