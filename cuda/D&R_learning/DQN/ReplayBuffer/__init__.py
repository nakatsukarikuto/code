from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer_cartpole():
    def __init__(self, buffer_size, batch_size) -> None:
        #dequeはリストより高速なデータ構造
        #max_lenを設定することで、これを超えてスタックしたら古いデータが消去される
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)
        
    def __len__(self):
        #現在のBufferSizeを取得するメソッド
        return len(self.buffer)
    
    def get_batch(self):
        #バッファ内のデータからバッチサイズ分ミニバッチを作成
        data = random.sample(self.buffer, self.batch_size)
        #取得したデータの各要素を配列へ結合
        #stateの場合は、stateの箇所の配列をサンプリングしたバッチサイズ分取り出し、バッチサイズ×state分の配列を作成している。
        #torch.tensorでテンソル化してそれをリストにしたものをスタック
        state = torch.stack([torch.tensor(x[0]) for x in data])
        #action, rewardは整数値のスカラーだから、ベクトルを作ればいい
        action = torch.stack([x[1] for x in data])
        reward = torch.stack([x[2] for x in data])
        #doneはbool値のため、0 or 1に変換してベクトル化
        done = torch.stack([x[3] for x in data]).int()
        return state, action, reward, done