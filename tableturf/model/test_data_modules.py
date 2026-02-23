#!/usr/bin/env python3
"""
测试数据模块的脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tableturf.model import DataLoader, CardFactory, StageFactory

def test_data_loader():
    """测试DataLoader"""
    print("=" * 60)
    print("测试 DataLoader")
    print("=" * 60)
    
    loader = DataLoader()
    
    # 测试加载所有数据
    print("1. 加载所有数据...")
    loader.load_all()
    
    # 测试获取数据
    print("\n2. 测试获取数据...")
    
    # 获取NPC列表
    npc_ids = loader.get_all_npc_ids()
    print(f"找到 {len(npc_ids)} 个NPC")
    if npc_ids:
        sample_npc = npc_ids[0]
        print(f"示例NPC: {sample_npc}")
        
        # 获取NPC信息
        npc_info = loader.get_npc_by_id(sample_npc)
        if npc_info:
            print(f"  NPC名称: {npc_info.get('Name')} ({npc_info.get('name_zh', '')})")
            print(f"  AI等级: {npc_info.get('AILevel', [])}")
            print(f"  场地: {npc_info.get('Map', [])}")
    
    # 获取卡牌列表
    card_ids = loader.get_all_card_ids()
    print(f"\n找到 {len(card_ids)} 张卡牌")
    if card_ids:
        sample_card = card_ids[0]
        print(f"示例卡牌: {sample_card}")
        
        # 获取卡牌信息
        card_info = loader.get_card_by_id(sample_card)
        if card_info:
            print(f"  卡牌名称: {card_info.get('Name')} ({card_info.get('name_zh', '')})")
            print(f"  面积: {card_info.get('area', 0)}")
            print(f"  SP消耗: {card_info.get('SpecialCost', 0)}")
    
    # 获取场地列表
    stage_names = loader.get_all_stage_names()
    print(f"\n找到 {len(stage_names)} 个场地")
    if stage_names:
        sample_stage = stage_names[0]
        print(f"示例场地: {sample_stage}")
        
        # 获取场地信息
        stage_info = loader.get_stage_by_name(sample_stage)
        if stage_info:
            print(f"  场地中文名: {stage_info.get('name_zh', '')}")
            dimensions = stage_info.get('Dimensions', {})
            print(f"  尺寸: {dimensions.get('rows', 0)}x{dimensions.get('cols', 0)}")
    
    return loader

def test_card_factory(loader):
    """测试CardFactory"""
    print("\n" + "=" * 60)
    print("测试 CardFactory")
    print("=" * 60)
    
    factory = CardFactory(loader)
    
    # 测试创建卡牌
    card_ids = loader.get_all_card_ids()
    if card_ids:
        # 测试前3张卡牌
        for i, card_id in enumerate(card_ids[:3]):
            print(f"\n{i+1}. 测试卡牌: {card_id}")
            
            # 获取卡牌属性
            props = factory.get_card_properties(card_id)
            print(f"   名称: {props.get('name_zh', '')} ({props.get('name', '')})")
            print(f"   面积: {props.get('area', 0)}, SP消耗: {props.get('sp_cost', 0)}")
            
            # 创建Card对象
            card = factory.create_card_from_id(card_id)
            if card:
                print(f"   成功创建Card对象")
                print(f"   实际面积: {card.size}, SP消耗: {card.sp_cost}")
                
                # 测试获取图案
                pattern = card.get_pattern(0)
                print(f"   图案尺寸: {pattern.width}x{pattern.height}")
            else:
                print(f"   创建Card对象失败")
    
    # 测试NPC卡组
    npc_ids = loader.get_all_npc_ids()
    if npc_ids:
        sample_npc = npc_ids[0]
        print(f"\n测试NPC卡组: {sample_npc}")
        
        # 创建卡组
        deck = factory.create_deck_from_npc_id(sample_npc)
        print(f"  卡组大小: {len(deck)} 张卡牌")
        
        if deck:
            # 统计卡组信息
            total_area = sum(card.size for card in deck)
            avg_sp_cost = sum(card.sp_cost for card in deck) / len(deck)
            print(f"  总面积: {total_area}, 平均SP消耗: {avg_sp_cost:.1f}")

def test_stage_factory(loader):
    """测试StageFactory"""
    print("\n" + "=" * 60)
    print("测试 StageFactory")
    print("=" * 60)
    
    factory = StageFactory(loader)
    
    # 测试创建场地
    stage_names = loader.get_all_stage_names()
    if stage_names:
        # 测试前2个场地
        for i, stage_name in enumerate(stage_names[:2]):
            print(f"\n{i+1}. 测试场地: {stage_name}")
            
            # 获取场地属性
            props = factory.get_stage_properties(stage_name)
            print(f"   中文名: {props.get('name_zh', '')}")
            print(f"   尺寸: {props.get('rows', 0)}x{props.get('cols', 0)}")
            print(f"   最大格子数: {props.get('max_cells', 0)}")
            
            # 分析场地
            analysis = factory.analyze_stage(stage_name)
            if 'error' not in analysis:
                print(f"   可涂地格子: {analysis.get('playable_cells', 0)}")
                print(f"   特殊点: {analysis.get('my_special_cells', 0)}我方 + {analysis.get('his_special_cells', 0)}敌方")
                print(f"   墙壁: {analysis.get('wall_cells', 0)}")
                if 'playable_percentage' in analysis:
                    print(f"   可涂地百分比: {analysis['playable_percentage']:.1f}%")
            
            # 创建Stage对象
            stage = factory.create_stage_from_name(stage_name)
            if stage:
                print(f"   成功创建Stage对象")
                print(f"   实际尺寸: {stage.shape[0]}x{stage.shape[1]}")
                print(f"   可涂地格子数: {stage.size}")
            else:
                print(f"   创建Stage对象失败")

def test_npc_integration():
    """测试NPC数据集成"""
    print("\n" + "=" * 60)
    print("测试 NPC 数据集成")
    print("=" * 60)
    
    loader = DataLoader()
    card_factory = CardFactory(loader)
    stage_factory = StageFactory(loader)
    
    npc_ids = loader.get_all_npc_ids()
    if not npc_ids:
        print("没有找到NPC数据")
        return
    
    # 测试前2个NPC的完整数据
    for i, npc_id in enumerate(npc_ids[:2]):
        print(f"\n{i+1}. NPC: {npc_id}")
        
        # 获取NPC信息
        npc_info = loader.get_npc_by_id(npc_id)
        if npc_info:
            print(f"   名称: {npc_info.get('Name', '')} ({npc_info.get('name_zh', '')})")
            print(f"   AI类型: {npc_info.get('AIType', [])}")
            print(f"   AI等级: {npc_info.get('AILevel', [])}")
            
            # 获取卡组
            deck_cards = loader.get_npc_deck_cards(npc_id)
            print(f"   卡组大小: {len(deck_cards)} 张卡牌")
            
            # 统计卡组信息
            if deck_cards:
                total_area = sum(card.get('area', 0) for card in deck_cards)
                avg_sp_cost = sum(card.get('SpecialCost', 0) for card in deck_cards) / len(deck_cards)
                print(f"   卡组总面积: {total_area}, 平均SP消耗: {avg_sp_cost:.1f}")
            
            # 获取场地信息
            map_list = npc_info.get('Map', [])
            if map_list:
                print(f"   使用的场地: {map_list}")
                
                # 测试第一个场地
                if map_list[0]:
                    stage_info = loader.get_stage_by_name(map_list[0])
                    if stage_info:
                        print(f"   场地 '{map_list[0]}' 信息:")
                        print(f"     中文名: {stage_info.get('name_zh', '')}")
                        dimensions = stage_info.get('Dimensions', {})
                        print(f"     尺寸: {dimensions.get('rows', 0)}x{dimensions.get('cols', 0)}")

def main():
    """主测试函数"""
    print("开始测试数据模块...")
    
    try:
        # 测试DataLoader
        loader = test_data_loader()
        
        # 测试CardFactory
        test_card_factory(loader)
        
        # 测试StageFactory
        test_stage_factory(loader)
        
        # 测试NPC集成
        test_npc_integration()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())