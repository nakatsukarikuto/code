from Agent import Agent_cartpole
from Q_Net import QNet_cartpole
from ReplayBuffer import ReplayBuffer_cartpole
import gym

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(episode, buffer_size, batch_size):
    #ネットワーク
    Q = QNet_cartpole()
    #経験再生
    R = ReplayBuffer_cartpole(buffer_size=buffer_size, batch_size=batch_size)
    #エージェント
    A = Agent_cartpole(Q=Q, R=R, device=device)


