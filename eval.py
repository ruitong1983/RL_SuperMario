import sys
import time
from datetime import datetime

import gym_super_mario_bros
import torch
import torch.nn as nn
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import *


# Same as duel_dqn.mlp (you can make model.py to avoid duplication.)
class model(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(20736, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)

        self.device = device
        # self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, self.q, self.v)

        # self.seq.apply(init_weights)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv - 1 / adv.shape[-1] * adv.max(-1, True)[0])
        return q


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def arange(s):
    if not type(s) == "numpy.ndarray":
        s = np.array(s)
    assert len(s.shape) == 3
    ret = np.transpose(s, (2, 0, 1))
    return np.expand_dims(ret, 0)


if __name__ == "__main__":
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "mario_q_target.pth"
    print(f"Load ckpt from {ckpt_path}")
    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    q = model(n_frame, env.action_space.n, device).to(device)

    q.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    total_score = 0.0
    done = False
    s = arange(env.reset())
    i = 0
    # 获取当前的系统时间，用于记录游戏开始时的时间，单位到时秒分
    start_time = datetime.now()
    print("SSSStart:  ", start_time.strftime("%H:%M:%S"))
    for i in range(100):
        while not done:
            env.render()
            if env.unwrapped._stage != 1:           # 获取当前环境阶段（可选，根据环境而定）
                done = True
            if device == "cpu":
                a = np.argmax(q(s).detach().numpy())
            else:
                a = np.argmax(q(s).cpu().detach().numpy())
            s_prime, r, done, _ = env.step(a)
            s_prime = arange(s_prime)
            total_score += r
            s = s_prime
            time.sleep(0.001)

    stage = env.unwrapped._stage
    total_time = datetime.now() - start_time  # 取时间的秒部分，作为游戏时间
    print("Total Time =====>%d | %d<=====Score" % (int(total_time.total_seconds()), total_score))
