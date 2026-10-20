"""
MCTS AI with Linear Learning
蒙特卡洛树搜索AI，带线性学习功能
不依赖深度学习框架，使用简单的线性回归学习
"""

import math
import random
import copy
import json
import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
import pickle
from pathlib import Path

from tableturf.model import Status, Step, Grid
from tableturf.ai import AI
from tableturf.ai.alpha import util
from logger import logger


class LinearLearner:
    """Linear learner without deep learning dependencies"""
    
    def __init__(self, feature_dim: int = 10, learning_rate: float = 0.01):
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.weights = np.random.randn(feature_dim) * 0.1
        self.bias = 0.0
        
        self.training_history = []
        self.max_history_size = 1000
        
        logger.info(f"LinearLearner initialized: dim={feature_dim}, lr={learning_rate}")
    
    def extract_features(self, state: Status, action: Optional[Step] = None) -> np.ndarray:
        """从游戏状态中提取特征"""
        features = np.zeros(self.feature_dim)
        
        # 特征1-2: 墨迹数量
        features[0] = len(state.stage.my_ink) / 100.0  # 归一化
        features[1] = len(state.stage.his_ink) / 100.0
        
        # 特征3-4: SP点数
        features[2] = state.my_sp / 10.0
        features[3] = state.his_sp / 10.0
        
        # 特征5-6: 特殊点数量
        features[4] = len(state.stage.my_sp) / 10.0
        features[5] = len(state.stage.his_sp) / 10.0
        
        # 特征7: 回合数
        features[6] = state.round / 12.0
        
        # 特征8: 手牌平均面积
        if state.hands:
            avg_size = sum(card.size for card in state.hands) / len(state.hands)
            features[7] = avg_size / 16.0  # 最大面积16
        
        # 特征9: 合法行动数量
        features[8] = len(state.get_possible_steps()) / 50.0
        
        # 特征10: 行动特征（如果有行动）
        if action:
            if action.card:
                features[9] = action.card.size / 16.0
            else:
                features[9] = 0.0
        
        return features
    
    def predict(self, features: np.ndarray) -> float:
        """预测状态价值"""
        return np.dot(features, self.weights) + self.bias
    
    def update(self, features: np.ndarray, target: float):
        """使用梯度下降更新权重"""
        prediction = self.predict(features)
        error = target - prediction
        
        # 梯度下降更新
        self.weights += self.learning_rate * error * features
        self.bias += self.learning_rate * error
        
        # 记录训练数据
        self.training_history.append({
            "features": features.copy(),
            "prediction": prediction,
            "target": target,
            "error": error
        })
        
        # 限制历史大小
        if len(self.training_history) > self.max_history_size:
            self.training_history = self.training_history[-self.max_history_size:]
        
        return error
    
    def batch_update(self, training_data: List[Tuple[np.ndarray, float]]):
        """批量更新"""
        total_error = 0.0
        for features, target in training_data:
            error = self.update(features, target)
            total_error += abs(error)
        
        avg_error = total_error / len(training_data) if training_data else 0.0
        logger.debug(f"批量更新完成: 平均误差={avg_error:.4f}")
        return avg_error
    
    def save(self, filepath: str):
        """保存学习器状态"""
        data = {
            "weights": self.weights,
            "bias": self.bias,
            "feature_dim": self.feature_dim,
            "learning_rate": self.learning_rate,
            "training_history_size": len(self.training_history)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"学习器状态已保存到: {filepath}")
    
    def load(self, filepath: str):
        """加载学习器状态"""
        if not os.path.exists(filepath):
            logger.warning(f"学习器状态文件不存在: {filepath}")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.weights = data["weights"]
            self.bias = data["bias"]
            self.feature_dim = data["feature_dim"]
            self.learning_rate = data["learning_rate"]
            
            logger.info(f"学习器状态已加载: {filepath}")
            return True
        except Exception as e:
            logger.error(f"加载学习器状态失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        if not self.training_history:
            return {"training_samples": 0}
        
        recent_errors = [item["error"] for item in self.training_history[-100:]]
        return {
            "training_samples": len(self.training_history),
            "recent_avg_error": np.mean(np.abs(recent_errors)),
            "weight_norm": np.linalg.norm(self.weights),
            "bias": self.bias
        }


class MCTSAI(AI):
    """MCTS AI with linear learning capability"""
    
    def __init__(self, simulation_limit=200, time_limit_ms=1000, 
                 learning_enabled=True, model_path=None):
        """
        MCTS AI constructor
        
        Args:
            simulation_limit: Maximum simulation count
            time_limit_ms: Maximum thinking time (ms)
            learning_enabled: Enable learning
            model_path: Model save path
        """
        self.simulation_limit = simulation_limit
        self.time_limit_ms = time_limit_ms
        self.exploration_weight = 1.41
        self.learning_enabled = learning_enabled
        
        # Initialize linear learner
        self.learner = LinearLearner(feature_dim=10, learning_rate=0.01)
        
        # Model save path
        self.model_path = model_path or "mcts_model.pkl"
        
        # Training data collection
        self.training_data = []  # (features, target_value)
        self.max_training_data = 500
        
        # Battle analysis data
        self.battle_analysis = {
            "total_battles": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total_turns": 0,
            "avg_turns_per_battle": 0.0,
            "decision_times": [],
            "feature_importance": np.zeros(10)
        }
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            self.learner.load(self.model_path)
            logger.info(f"Loaded learning model: {self.model_path}")
        
        logger.info(f"MCTSAI initialized: sims={simulation_limit}, learning={learning_enabled}")
    
    def redraw(self, hands, stage=None, my_deck=None, his_deck=None):
        """重抽决策 - 使用学习增强的启发式"""
        # 基础启发式
        avg_size = sum(card.size for card in hands) / len(hands)
        base_result = avg_size < 7
        
        logger.debug(f"LearningMCTS重抽决策: 平均面积={avg_size:.1f}, 重抽={base_result}")
        return base_result
    
    def next_step(self, status: Status) -> Step:
        """使用学习增强的MCTS选择最佳行动"""
        start_time = datetime.now()
        
        # MCTS搜索
        root_node = {
            'state': copy.deepcopy(status),
            'visits': 0,
            'value': 0.0,
            'children': [],
            'untried_actions': status.get_possible_steps()
        }
        
        simulations = 0
        for i in range(self.simulation_limit):
            simulations = i + 1
            
            # 检查时间限制
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed_ms > self.time_limit_ms:
                logger.debug(f"MCTS时间限制到达: {elapsed_ms:.0f}ms, 模拟次数: {simulations}")
                break
            
            # 选择 -> 扩展 -> 模拟 -> 回溯
            node = self._select(root_node)
            node = self._expand(node)
            reward = self._simulate(node['state'])
            self._backpropagate(node, reward, root_node)
        
        # 选择最佳行动
        if not root_node['children']:
            steps = status.get_possible_steps()
            if not steps:
                logger.warning("MCTS: 没有合法行动")
                if status.hands:
                    return Step(Step.Action.Skip, status.hands[0], None, None)
                else:
                    return Step(Step.Action.Skip, None, None, None)
            
            step = random.choice(steps)
            logger.debug(f"MCTS随机决策: {step}")
            return step
        
        # 选择访问次数最多的子节点
        best_child = max(root_node['children'], key=lambda c: c['visits'])
        
        # 记录决策时间
        decision_time = (datetime.now() - start_time).total_seconds() * 1000
        self.battle_analysis["decision_times"].append(decision_time)
        if len(self.battle_analysis["decision_times"]) > 100:
            self.battle_analysis["decision_times"] = self.battle_analysis["decision_times"][-100:]
        
        logger.debug(f"MCTS决策: 模拟次数={simulations}, 访问次数={best_child['visits']}, "
                    f"时间={decision_time:.0f}ms")
        
        return best_child['action']
    
    def _select(self, node):
        """选择节点 — 递归遍历到叶节点"""
        if node['untried_actions']:
            return node
        if node['children']:
            # 递归选择UCT值最高的子节点
            return self._select(
                max(node['children'], key=lambda c:
                    (c['value'] / max(1, c['visits'])) +
                    self.exploration_weight * math.sqrt(math.log(max(1, node['visits'])) / max(1, c['visits'])))
            )
        return node
    
    def _expand(self, node):
        """扩展节点"""
        if not node['untried_actions']:
            return node
        
        action = random.choice(node['untried_actions'])
        node['untried_actions'].remove(action)
        
        try:
            next_states = util.estimate_status(node['state'], action, expand=True)
            if next_states:
                next_state = random.choice(next_states)
                child = {
                    'state': next_state,
                    'action': action,
                    'visits': 0,
                    'value': 0.0,
                    'children': [],
                    'untried_actions': next_state.get_possible_steps(),
                    'parent': node
                }
                node['children'].append(child)
                
                # 提取特征用于学习
                features = self.learner.extract_features(next_state, action)
                child['features'] = features
                
                return child
        except Exception as e:
            logger.warning(f"MCTS扩展失败: {e}")
        
        return node
    
    def _simulate(self, state: Status) -> float:
        """模拟阶段"""
        current_state = copy.deepcopy(state)
        
        # 快速走子
        while current_state.round > 0:
            possible_steps = current_state.get_possible_steps()
            if not possible_steps:
                break
            
            # 使用学习增强的启发式
            step = self._learning_heuristic_select(current_state, possible_steps)
            if step is None:
                break
            
            try:
                next_states = util.estimate_status(current_state, step, expand=True)
                if next_states:
                    current_state = random.choice(next_states)
                else:
                    break
            except Exception as e:
                logger.debug(f"模拟阶段错误: {e}")
                break
        
        # 计算奖励
        return self._calculate_reward(current_state)
    
    def _learning_heuristic_select(self, state: Status, possible_steps: List[Step]) -> Step:
        """学习增强的启发式行动选择"""
        if not possible_steps:
            return None
        
        # 80%概率使用学习增强
        if random.random() < 0.8 and self.learning_enabled:
            action_values = []
            for step in possible_steps:
                try:
                    next_states = util.estimate_status(state, step, expand=False)
                    if next_states:
                        next_state = next_states[0]
                        features = self.learner.extract_features(next_state, step)
                        value = self.learner.predict(features)
                        action_values.append((step, value))
                    else:
                        action_values.append((step, -1000))
                except:
                    action_values.append((step, -1000))
            
            if action_values and max(v for _, v in action_values) > -1000:
                return max(action_values, key=lambda x: x[1])[0]
        
        # 随机选择
        return random.choice(possible_steps)
    
    def _calculate_reward(self, state: Status) -> float:
        """计算最终奖励"""
        my_ink = len(state.stage.my_ink)
        his_ink = len(state.stage.his_ink)
        
        if my_ink > his_ink:
            return 1.0
        elif my_ink < his_ink:
            return 0.0
        else:
            return 0.5
    
    def _backpropagate(self, node, reward, root_node):
        """回溯更新 — 从当前节点向上更新到根节点（含根节点）"""
        while node is not None:
            node['visits'] += 1
            node['value'] += reward

            # 收集学习数据
            if self.learning_enabled and 'features' in node:
                self._collect_training_data(node['features'], reward)

            if 'parent' in node:
                node = node['parent']
            else:
                break
    
    def _collect_training_data(self, features: np.ndarray, target: float):
        """收集训练数据"""
        self.training_data.append((features.copy(), target))
        
        # 限制训练数据大小
        if len(self.training_data) > self.max_training_data:
            self.training_data = self.training_data[-self.max_training_data:]
        
        # 定期批量更新
        if len(self.training_data) >= 100 and len(self.training_data) % 50 == 0:
            self._batch_train()
    
    def _batch_train(self):
        """批量训练"""
        if not self.training_data or not self.learning_enabled:
            return
        
        sample_size = min(50, len(self.training_data))
        training_sample = random.sample(self.training_data, sample_size)
        
        avg_error = self.learner.batch_update(training_sample)
        logger.debug(f"批量训练完成: 样本数={sample_size}, 平均误差={avg_error:.4f}")
        
        # 更新特征重要性
        for features, _ in training_sample:
            self.battle_analysis["feature_importance"] += np.abs(features)
        
        # 定期保存模型
        if len(self.training_data) % 200 == 0:
            self.save_model()
    
    def record_battle_result(self, result: str, turns: int):
        """记录对战结果"""
        self.battle_analysis["total_battles"] += 1
        self.battle_analysis["total_turns"] += turns
        
        if result == "win":
            self.battle_analysis["wins"] += 1
        elif result == "loss":
            self.battle_analysis["losses"] += 1
        elif result == "draw":
            self.battle_analysis["draws"] += 1
        
        # 更新平均回合数
        if self.battle_analysis["total_battles"] > 0:
            self.battle_analysis["avg_turns_per_battle"] = (
                self.battle_analysis["total_turns"] / self.battle_analysis["total_battles"]
            )
        
        # 对战结束后训练
        if self.learning_enabled and self.training_data:
            self._batch_train()
            self.save_model()
        
        logger.info(f"对战记录: 结果={result}, 回合数={turns}")
    
    def get_analysis_report(self) -> Dict[str, Any]:
        """获取分析报告"""
        stats = self.learner.get_statistics()
        
        decision_times = self.battle_analysis["decision_times"]
        if decision_times:
            avg_decision_time = np.mean(decision_times)
        else:
            avg_decision_time = 0.0
        
        feature_importance = self.battle_analysis["feature_importance"]
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        report = {
            "battle_stats": {
                "total_battles": self.battle_analysis["total_battles"],
                "wins": self.battle_analysis["wins"],
                "losses": self.battle_analysis["losses"],
                "draws": self.battle_analysis["draws"],
                "win_rate": self.battle_analysis["wins"] / max(1, self.battle_analysis["total_battles"]),
                "avg_turns_per_battle": self.battle_analysis["avg_turns_per_battle"]
            },
            "learning_stats": stats,
            "performance": {
                "avg_decision_time_ms": avg_decision_time,
                "recent_decision_times": decision_times[-10:] if len(decision_times) > 10 else decision_times
            },
            "feature_importance": {
                "my_ink_ratio": feature_importance[0] if len(feature_importance) > 0 else 0.0,
                "his_ink_ratio": feature_importance[1] if len(feature_importance) > 1 else 0.0,
                "my_sp": feature_importance[2] if len(feature_importance) > 2 else 0.0,
                "his_sp": feature_importance[3] if len(feature_importance) > 3 else 0.0,
                "my_special_count": feature_importance[4] if len(feature_importance) > 4 else 0.0,
                "his_special_count": feature_importance[5] if len(feature_importance) > 5 else 0.0,
                "round_progress": feature_importance[6] if len(feature_importance) > 6 else 0.0,
                "hand_avg_size": feature_importance[7] if len(feature_importance) > 7 else 0.0,
                "legal_action_count": feature_importance[8] if len(feature_importance) > 8 else 0.0,
                "action_card_size": feature_importance[9] if len(feature_importance) > 9 else 0.0
            }
        }

        return report

    def save_model(self):
        """保存模型和学习数据"""
        try:
            # 保存线性学习器状态
            self.learner.save(self.model_path)

            # 保存对战分析数据
            analysis_path = self.model_path.replace('.pkl', '_analysis.pkl')
            analysis_data = {
                "battle_analysis": self.battle_analysis,
                "training_data_count": len(self.training_data)
            }
            with open(analysis_path, 'wb') as f:
                pickle.dump(analysis_data, f)

            logger.debug(f"模型和对战数据已保存: {self.model_path}, {analysis_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")

    def load_model(self) -> bool:
        """加载模型和学习数据

        Returns:
            bool: 加载是否成功
        """
        try:
            # 加载线性学习器状态
            if not os.path.exists(self.model_path):
                logger.info(f"模型文件不存在，将使用新模型: {self.model_path}")
                return False

            success = self.learner.load(self.model_path)
            if not success:
                return False

            # 加载对战分析数据
            analysis_path = self.model_path.replace('.pkl', '_analysis.pkl')
            if os.path.exists(analysis_path):
                with open(analysis_path, 'rb') as f:
                    analysis_data = pickle.load(f)
                self.battle_analysis = analysis_data.get("battle_analysis", self.battle_analysis)
                logger.info(f"对战分析数据已加载: {analysis_path}")

            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False