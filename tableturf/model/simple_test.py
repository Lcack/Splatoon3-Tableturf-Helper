#!/usr/bin/env python3
"""
简化测试 - 只测试数据加载，不测试工厂类
"""

import sys
import os
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_json_files():
    """直接测试JSON文件是否能正确加载"""
    print("测试JSON文件加载...")
    print("=" * 60)
    
    data_dir = os.path.join(project_root, "tableturf", "model")
    
    # 测试卡牌信息文件
    card_file = os.path.join(data_dir, "MiniGameCardInfo_Matrix_ZH.json")
    print(f"1. 测试卡牌信息文件: {card_file}")
    try:
        with open(card_file, 'r', encoding='utf-8') as f:
            card_data = json.load(f)
        print(f"   成功加载 {len(card_data)} 条卡牌记录")
        
        # 显示第一条记录
        if card_data:
            first_card = card_data[0]
            print(f"   示例卡牌: {first_card.get('__RowId', 'N/A')}")
            print(f"     名称: {first_card.get('Name', 'N/A')} ({first_card.get('name_zh', 'N/A')})")
            print(f"     面积: {first_card.get('area', 'N/A')}")
            print(f"     SP消耗: {first_card.get('SpecialCost', 'N/A')}")
            
            # 检查矩阵
            matrix = first_card.get('square_matrix', [])
            if matrix:
                print(f"     矩阵尺寸: {len(matrix)}x{len(matrix[0]) if matrix else 0}")
    except Exception as e:
        print(f"   加载失败: {e}")
    
    # 测试NPC数据文件
    npc_file = os.path.join(data_dir, "MiniGameGameNpcData_ZH.json")
    print(f"\n2. 测试NPC数据文件: {npc_file}")
    try:
        with open(npc_file, 'r', encoding='utf-8') as f:
            npc_data = json.load(f)
        print(f"   成功加载 {len(npc_data)} 条NPC记录")
        
        # 显示第一条记录
        if npc_data:
            first_npc = npc_data[0]
            print(f"   示例NPC: {first_npc.get('__RowId', 'N/A')}")
            print(f"     名称: {first_npc.get('Name', 'N/A')} ({first_npc.get('name_zh', 'N/A')})")
            print(f"     AI等级: {first_npc.get('AILevel', [])}")
            print(f"     AI类型: {first_npc.get('AIType', [])}")
            print(f"     场地: {first_npc.get('Map', [])}")
    except Exception as e:
        print(f"   加载失败: {e}")
    
    # 测试预设卡组文件
    deck_file = os.path.join(data_dir, "MiniGamePresetDeck.json")
    print(f"\n3. 测试预设卡组文件: {deck_file}")
    try:
        with open(deck_file, 'r', encoding='utf-8') as f:
            deck_data = json.load(f)
        print(f"   成功加载 {len(deck_data)} 条卡组记录")
        
        # 显示第一条记录
        if deck_data:
            first_deck = deck_data[0]
            print(f"   示例卡组: {first_deck.get('__RowId', 'N/A')}")
            cards = first_deck.get('Card', [])
            print(f"     包含 {len(cards)} 张卡牌")
            if cards:
                print(f"     前3张卡牌引用: {cards[:3]}")
    except Exception as e:
        print(f"   加载失败: {e}")
    
    # 测试场地文件
    stage_file = os.path.join(data_dir, "MiniGameStage.json")
    print(f"\n4. 测试场地文件: {stage_file}")
    try:
        with open(stage_file, 'r', encoding='utf-8') as f:
            stage_data = json.load(f)
        print(f"   成功加载 {len(stage_data)} 条场地记录")
        
        # 显示第一条记录
        if stage_data:
            first_stage = stage_data[0]
            print(f"   示例场地: {first_stage.get('Name', 'N/A')}")
            print(f"     中文名: {first_stage.get('name_zh', 'N/A')}")
            dimensions = first_stage.get('Dimensions', {})
            print(f"     尺寸: {dimensions.get('rows', 'N/A')}x{dimensions.get('cols', 'N/A')}")
            
            # 检查矩阵
            matrix = first_stage.get('square_matrix', [])
            if matrix:
                print(f"     矩阵尺寸: {len(matrix)}x{len(matrix[0]) if matrix else 0}")
    except Exception as e:
        print(f"   加载失败: {e}")
    
    print("\n" + "=" * 60)
    print("JSON文件加载测试完成")
    print("=" * 60)

def test_data_relationships():
    """测试数据之间的关系"""
    print("\n测试数据关系...")
    print("=" * 60)
    
    data_dir = os.path.join(project_root, "tableturf", "model")
    
    try:
        # 加载所有数据
        with open(os.path.join(data_dir, "MiniGameGameNpcData_ZH.json"), 'r', encoding='utf-8') as f:
            npc_data = json.load(f)
        
        with open(os.path.join(data_dir, "MiniGamePresetDeck.json"), 'r', encoding='utf-8') as f:
            deck_data = json.load(f)
        
        with open(os.path.join(data_dir, "MiniGameCardInfo_Matrix_ZH.json"), 'r', encoding='utf-8') as f:
            card_data = json.load(f)
        
        with open(os.path.join(data_dir, "MiniGameStage.json"), 'r', encoding='utf-8') as f:
            stage_data = json.load(f)
        
        # 转换为字典以便查找
        npc_dict = {item["__RowId"]: item for item in npc_data}
        deck_dict = {item["__RowId"]: item for item in deck_data}
        card_dict = {item["__RowId"]: item for item in card_data}
        stage_dict = {item["Name"]: item for item in stage_data}
        
        print(f"数据统计:")
        print(f"  NPC数量: {len(npc_dict)}")
        print(f"  卡组数量: {len(deck_dict)}")
        print(f"  卡牌数量: {len(card_dict)}")
        print(f"  场地数量: {len(stage_dict)}")
        
        # 测试第一个NPC的数据关系
        if npc_data:
            sample_npc = npc_data[0]
            npc_id = sample_npc.get("__RowId")
            print(f"\n测试NPC '{npc_id}' 的数据关系:")
            
            # 1. NPC -> 卡组
            deck_refs = sample_npc.get("Deck", [])
            if deck_refs:
                # 提取卡组ID
                deck_id = deck_refs[0].split("/")[-1].split(".")[0] if "/" in deck_refs[0] else deck_refs[0]
                print(f"  使用的卡组: {deck_id}")
                
                # 2. 卡组 -> 卡牌
                deck_info = deck_dict.get(deck_id)
                if deck_info:
                    card_refs = deck_info.get("Card", [])
                    print(f"  卡组包含 {len(card_refs)} 张卡牌")
                    
                    # 检查前3张卡牌是否存在
                    for i, card_ref in enumerate(card_refs[:3]):
                        card_id = card_ref.split("/")[-1].split(".")[0] if "/" in card_ref else card_ref
                        card_info = card_dict.get(card_id)
                        if card_info:
                            print(f"    卡牌{i+1}: {card_id} - {card_info.get('Name', 'N/A')} ({card_info.get('name_zh', 'N/A')})")
                        else:
                            print(f"    卡牌{i+1}: {card_id} - 未找到")
            
            # 3. NPC -> 场地
            map_list = sample_npc.get("Map", [])
            if map_list:
                stage_name = map_list[0]
                stage_info = stage_dict.get(stage_name)
                if stage_info:
                    print(f"  使用的场地: {stage_name} - {stage_info.get('name_zh', 'N/A')}")
                else:
                    print(f"  使用的场地: {stage_name} - 未找到")
        
        print("\n数据关系测试完成")
        
    except Exception as e:
        print(f"数据关系测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("简化数据模块测试")
    print("=" * 60)
    
    test_json_files()
    test_data_relationships()
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    main()