# 方策反復法

import sys
sys.path.append("./GridWorld")
from GridWorld import GRID_WORLD
from collections import defaultdict

env = GRID_WORLD()
V = defaultdict(lambda: 0)

state = (1, 2)
pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

# 1ステップ分の更新関数
def eval_onestep(pi, V, env, gamma = 0.9):
    for state in env.states():
        #ゴールの価値関数を0にする
        if state == env.goal_state:
            V[state] = 0
            continue
        
        action_probs = pi[state]
        #新たな価値関数の初期化
        new_V = 0

        for action, action_prob in action_probs.item():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # 価値関数の更新
            new_V += action_prob * (r + gamma * V[next_state])
        
        V[state] = new_V
    
    return V

def policy_eval(pi, V, env, gamma, threshold = 0.001):
    #thresholdは閾値
    while True:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma=gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta<t:
                delta = t
        if delta < threshold:
            break
    return V