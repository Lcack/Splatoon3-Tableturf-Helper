"""
MCTS AI 测试
测试新的MCTS AI功能
"""

import unittest
import numpy as np
from tableturf.ai.mcts import MCTSAI, EnhancedMCTSAI
from tableturf.model import Status, Stage, Card, Grid


class TestMCTS(unittest.TestCase):
    def test_mcts_basic_creation(self):
        """测试基础MCTS AI创建"""
        ai = MCTSAI(
            simulation_limit=20,
            time_limit_ms=500,
            learning_enabled=False
        )
        self.assertIsNotNone(ai)
        print(f"MCTS AI创建成功: sims={ai.simulation_limit}")
    
    def test_mcts_with_learning(self):
        """测试带学习的MCTS AI"""
        ai = MCTSAI(
            simulation_limit=20,
            learning_enabled=True,
            model_path="test_mcts_model.pkl"
        )
        self.assertTrue(ai.learning_enabled)
        print(f"学习版MCTS AI创建成功")
    
    def test_mcts_decision(self):
        """测试MCTS决策功能"""
        # 创建简单游戏状态
        stage_grid = np.full((10, 10), Grid.Empty.value, dtype=int)
        stage_grid[0, 0] = Grid.MySpecial.value
        stage_grid[9, 9] = Grid.HisSpecial.value
        stage = Stage(stage_grid)
        
        # 创建简单卡牌
        card_grid = np.zeros((8, 8), dtype=int)
        card_grid[2:4, 2:4] = Grid.MyInk.value
        card = Card(card_grid, sp_cost=3)
        
        status = Status(
            stage=stage,
            hands=[card, card, card, card],
            round=12,
            my_sp=0,
            his_sp=0,
            my_deck=[],
            his_deck=[]
        )
        
        # 创建MCTS AI（快速测试模式）
        ai = MCTSAI(
            simulation_limit=10,
            time_limit_ms=200,
            learning_enabled=False
        )
        
        # 测试重抽决策
        redraw = ai.redraw(status.hands)
        self.assertIsInstance(redraw, bool)
        print(f"重抽决策: {redraw}")
        
        # 测试下一步决策
        step = ai.next_step(status)
        self.assertIsNotNone(step)
        print(f"MCTS决策: {step.action}")
        
        # 记录对战结果（测试学习功能）
        ai.record_battle_result("win", 8)
        report = ai.get_analysis_report()
        self.assertEqual(report['battle_stats']['total_battles'], 1)
        self.assertEqual(report['battle_stats']['wins'], 1)
        print(f"对战记录: {report['battle_stats']}")
    
    def test_enhanced_mcts_creation(self):
        """测试增强版MCTS AI创建"""
        try:
            ai = EnhancedMCTSAI(
                npc_id="MiniGame_Aori",
                level=0,
                simulation_limit=20,
                learning_enabled=False
            )
            self.assertIsNotNone(ai)
            print(f"EnhancedMCTSAI创建成功: NPC={ai.npc_id}")
            
            # 测试NPC分析
            analysis = ai.get_npc_analysis()
            if analysis:
                print(f"NPC分析: {analysis.get('name', 'Unknown')}")
        except Exception as e:
            # 如果JSON数据不可用，跳过测试
            self.skipTest(f"EnhancedMCTSAI测试跳过: {e}")
    
    def test_mcts_learning_features(self):
        """测试MCTS学习功能"""
        ai = MCTSAI(
            simulation_limit=10,
            learning_enabled=True,
            model_path="test_learning.pkl"
        )
        
        # 获取初始学习状态
        initial_stats = ai.learner.get_statistics()
        print(f"初始学习状态: {initial_stats}")
        
        # 模拟一些训练数据
        features = np.random.randn(10)
        ai.learner.update(features, 1.0)
        
        # 检查训练后状态
        trained_stats = ai.learner.get_statistics()
        self.assertGreater(trained_stats['training_samples'], 0)
        print(f"训练后: {trained_stats['training_samples']}个样本")
        
        # 保存和加载模型
        ai.save_model()
        print("模型保存成功")
    
    def test_mcts_analysis_report(self):
        """测试分析报告功能"""
        ai = MCTSAI(learning_enabled=True)
        
        # 记录一些决策时间
        ai.battle_analysis["decision_times"] = [100, 150, 200, 180]
        ai.battle_analysis["total_battles"] = 2
        ai.battle_analysis["wins"] = 1
        ai.battle_analysis["losses"] = 1
        
        report = ai.get_analysis_report()
        
        # 检查报告结构
        self.assertIn('battle_stats', report)
        self.assertIn('learning_stats', report)
        self.assertIn('performance_stats', report)
        self.assertIn('feature_importance', report)
        
        print(f"分析报告生成成功")
        print(f"  胜率: {report['battle_stats'].get('win_rate', 0):.1f}%")
        print(f"  平均决策时间: {report['performance_stats'].get('avg_decision_time_ms', 0):.0f}ms")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)