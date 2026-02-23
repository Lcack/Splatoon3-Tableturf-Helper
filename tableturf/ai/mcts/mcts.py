import math
import random
import copy
import json
import os
import numpy as np
from typing import List, Tuple, Dict, Optional

# 引用项目现有模型
from tableturf.model.status import Status as GameStatus
from tableturf.model.step import Step
from tableturf.model.grid import Grid
from logger import logger

class MCTSNode:
    def __init__(self, state: GameStatus, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        # 获取所有合法动作
        self.untried_actions = self.state.get_all_legal_steps()

    def uct_select_child(self, exploration_weight=1.41):
        """ 使用 UCB1 算法选择子节点 """
        return max(self.children, key=lambda c: (c.value / c.visits) + 
                   exploration_weight * math.sqrt(math.log(self.visits) / c.visits))

class MCTSWithLearning:
    def __init__(self, simulation_limit=500, weights_path="portal/profiles/ai_weights.json"):
        self.limit = simulation_limit
        self.weights_path = weights_path
        self.learning_rate = 0.005
        self.gamma = 0.95  # 折扣因子
        
        # 初始权重：参考 Alpha 算法的评估倾向
        self.default_weights = {
            "my_ink_count": 1.0,      # 占地面积
            "opp_ink_count": -0.8,    # 压制对手面积
            "my_sp": 5.5,             # SP点数的重要性
            "connectivity": 0.4,      # 领地连通性/扩张潜力
            "edge_penalty": -0.2      # 靠边的惩罚（倾向于占领中心）
        }
        self.weights = self._load_weights()

    def _load_weights(self):
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, 'r') as f:
                    return json.load(f)
            except:
                return self.default_weights
        return self.default_weights

    def save_weights(self):
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        with open(self.weights_path, 'w') as f:
            json.dump(self.weights, f, indent=4)
        logger.info(f"AI 权重已保存至: {self.weights_path}")

    def _extract_features(self, state: GameStatus) -> np.ndarray:
        """ 状态特征提取：将复杂的地图转换为简单的数值向量 """
        grid = state.stage.grid
        my_ink = np.sum(grid == Grid.MyInk.value)
        opp_ink = np.sum(grid == Grid.HisInk.value)
        
        # 简单连通性（当前位置周围的空格）
        # 这里可以复用 alpha.py 中的 Evaluation.occupied_grids 逻辑
        features = [
            float(my_ink),
            float(opp_ink),
            float(state.my_sp),
            float(len(state.get_all_legal_steps())), # 扩张潜力
            float(np.sum(grid[0,:] == Grid.Empty.value) + np.sum(grid[-1,:] == Grid.Empty.value)) # 边缘空位
        ]
        return np.array(features)

    def _get_v_value(self, state: GameStatus) -> float:
        """ 线性评估函数：V(s) = W · Φ(s) """
        features = self._extract_features(state)
        weight_vector = np.array([self.weights[k] for k in ["my_ink_count", "opp_ink_count", "my_sp", "connectivity", "edge_penalty"]])
        return np.dot(weight_vector, features)

    def search(self, initial_state: GameStatus) -> Step:
        root = MCTSNode(copy.deepcopy(initial_state))

        for i in range(self.limit):
            node = self._select(root)
            # 使用带有权重引导的“启发式走子”
            reward = self._heuristic_rollout(node.state)
            self._backpropagate(node, reward)

        # 返回访问次数最多的动作
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _select(self, node: MCTSNode):
        while not node.state.is_terminal():
            if node.untried_actions:
                return self._expand(node)
            else:
                node = node.uct_select_child()
        return node

    def _expand(self, node: MCTSNode):
        action = node.untried_actions.pop()
        # 模拟执行一步（假设对手随机或按简单逻辑出牌）
        next_state = node.state.simulate_step(action) 
        child_node = MCTSNode(next_state, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def _heuristic_rollout(self, state: GameStatus) -> float:
        """ 启发式走子：利用线性权重引导模拟，比纯随机快且准 """
        curr = copy.deepcopy(state)
        while not curr.is_terminal():
            possible_steps = curr.get_all_legal_steps()
            if not possible_steps: break
            
            # Epsilon-Greedy: 80% 概率选估值最高的步子
            if random.random() < 0.8:
                # 简单预测一步后的价值
                best_move = max(possible_steps, key=lambda s: self._get_v_value(curr.preview_step(s)))
                curr.apply_step(best_move)
            else:
                curr.apply_step(random.choice(possible_steps))
        
        # 最终得分结算
        return 1.0 if curr.my_score > curr.his_score else 0.0

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def update_learning(self, game_history: List[GameStatus], final_outcome: float):
        """ 
        核心学习逻辑：TD(λ) 学习
        在 run_qt.py 的每局结算界面调用
        """
        for i in range(len(game_history) - 1):
            s_curr = game_history[i]
            s_next = game_history[i+1]
            
            features = self._extract_features(s_curr)
            v_curr = self._get_v_value(s_curr)
            v_next = self._get_v_value(s_next)
            
            # TD Error: 实际演变趋势与预测的差距
            td_error = (self.gamma * v_next) - v_curr
            
            # 更新权重
            feature_names = ["my_ink_count", "opp_ink_count", "my_sp", "connectivity", "edge_penalty"]
            for idx, name in enumerate(feature_names):
                self.weights[name] += self.learning_rate * td_error * features[idx]
        
        self.save_weights()