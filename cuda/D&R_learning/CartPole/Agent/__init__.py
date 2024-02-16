import torch.optim as optim
import copy
import numpy as np
import torch
import torch.nn.functional as F

class Agent_cartpole():
    def __init__(self, Q: object, R: object, device) -> None:
        self.gamma = 0.98
        self.rate = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2
        self.device =device
        self.Q_Net = Q().to(self.device)
        self.target_Net = Q().to(self.device)
        self.Re_Buff = R(self.buffer_size, self.batch_size)
        self.optimizer = optim.Adam(self.rate)
        self.optimizer.setup(self.Q_Net)
        
    def sync_qnet(self):
        self.target_Net = copy.deepcopy(self.Q_Net)
        
    def get_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, self.action_size, (1, ))
        
        else:
            state = torch.tensor(state, torch.float32).unsqueeze(0).to(device=self.device)
            qs = self.Q_Net(state)
            return qs.data.argmax()
        
    def update(self, state, action, reward, next_state, done):
        self.Re_Buff.add(state, action, reward, next_state, done)
        #経験再生内のデータ数がミニバッチサイズ以下なら以降は実行しない
        if len(self.Re_Buff) < self.batch_size:
            return
        state, action, reward, next_state, done = self.Re_Buff.get_batch().to(self.device)
        #状態を与えたときの行動価値関数
        qs = self.Q_Net(state)
        #actionと対応する行動価値関数を取り出す。
        q = qs[torch.arange(self.batch_size), action]
        
        #次の状態の価値関数を予測
        next_qs = self.target_Net(next_state)
        next_q = next_qs.max(axis = 1)
        next_q.unchain()
        target = reward + (1 - done) * self.gamma * next_q
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()