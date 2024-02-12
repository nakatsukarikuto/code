import torch.nn as nn

class QNet_cartpole(nn.Module):
    def __init__(self):
        super(QNet_cartpole, self).__init__()
        #NNとは違い、ミニバッチ分を入力とする
        #カートの位置、速度と、棒の角度、角速度が1つの状態=state
        #入力層はstate×ミニバッチ分のため、4×32となる↓
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 512), 
            nn.LeakyReLU(), 
            nn.Linear(512, 1024), 
            nn.LeakyReLU(0.001), 
            nn.Linear(1024, 512), 
            nn.LeakyReLU(0.001), 
            nn.Linear(512, 128), 
            nn.LeakyReLU(), 
            nn.Linear(128, 2)
        )
        
    def forward(self, input):
        y = self.net(input)
        return y