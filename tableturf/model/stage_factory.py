import numpy as np
from typing import Optional, Dict, Any, List
from tableturf.model.stage import Stage
from tableturf.model.grid import Grid

class StageFactory:
    """场地工厂，根据JSON数据创建Stage对象"""
    
    def __init__(self, data_loader=None):
        from tableturf.model.data_loader import DataLoader
        self.data_loader = data_loader or DataLoader()
        
    def create_stage_from_name(self, stage_name: str) -> Optional[Stage]:
        """根据场地名称创建Stage对象"""
        stage_info = self.data_loader.get_stage_by_name(stage_name)
        if not stage_info:
            print(f"Warning: Stage not found: {stage_name}")
            return None
        
        # 获取场地矩阵
        matrix = stage_info.get("square_matrix", [])
        if not matrix:
            print(f"Warning: Stage matrix not found: {stage_name}")
            return None
        
        # 转换为numpy数组
        try:
            grid = self.data_loader.convert_stage_matrix(matrix)
        except Exception as e:
            print(f"Error converting stage matrix for {stage_name}: {e}")
            return None
        
        # 转换矩阵值：根据Tableturf场地规则
        # 在MiniGameStage.json中，常见的值有：
        # 1: 可涂地（普通地面）
        # 4: 特殊点（可能是我方起始点）
        # 16: 特殊点（可能是敌方起始点）
        # 64: 墙壁/不可涂地
        converted_grid = np.zeros_like(grid, dtype=int)
        
        # 默认所有格子为墙壁（不可涂地）
        converted_grid.fill(Grid.Wall.value)
        
        # 将1转换为Empty（可涂地）
        converted_grid[grid == 1] = Grid.Empty.value
        
        # 将4转换为MySpecial（我方特殊点/起始点）
        converted_grid[grid == 4] = Grid.MySpecial.value
        
        # 将16转换为HisSpecial（敌方特殊点/起始点）
        converted_grid[grid == 16] = Grid.HisSpecial.value
        
        # 64保持为Wall（墙壁）
        converted_grid[grid == 64] = Grid.Wall.value
        
        # 验证场地数据
        empty_count = np.sum(converted_grid == Grid.Empty.value)
        my_special_count = np.sum(converted_grid == Grid.MySpecial.value)
        his_special_count = np.sum(converted_grid == Grid.HisSpecial.value)
        wall_count = np.sum(converted_grid == Grid.Wall.value)
        
        total_cells = grid.size
        calculated_total = empty_count + my_special_count + his_special_count + wall_count
        
        if total_cells != calculated_total:
            print(f"Warning: Stage cell count mismatch for {stage_name}")
        
        # 获取场地维度信息
        dimensions = stage_info.get("Dimensions", {})
        rows = dimensions.get("rows", grid.shape[0])
        cols = dimensions.get("cols", grid.shape[1])
        
        if rows != grid.shape[0] or cols != grid.shape[1]:
            print(f"Warning: Stage dimensions mismatch for {stage_name}: "
                  f"expected {rows}x{cols}, got {grid.shape[0]}x{grid.shape[1]}")
        
        # 创建Stage对象
        try:
            stage = Stage(converted_grid)
            print(f"Created stage: {stage_name} ({stage_info.get('name_zh', '')}) - "
                  f"Size: {grid.shape[0]}x{grid.shape[1]}, "
                  f"Empty: {empty_count}, Special: {my_special_count + his_special_count}")
            return stage
        except Exception as e:
            print(f"Error creating Stage object for {stage_name}: {e}")
            return None
    
    def create_stage_from_info(self, stage_info: dict) -> Optional[Stage]:
        """直接从场地信息字典创建Stage对象"""
        if not stage_info:
            return None
        
        # 获取场地名称
        stage_name = stage_info.get("Name", "Unknown")
        
        # 获取场地矩阵
        matrix = stage_info.get("square_matrix", [])
        if not matrix:
            return None
        
        # 转换为numpy数组
        try:
            grid = self.data_loader.convert_stage_matrix(matrix)
        except Exception as e:
            print(f"Error converting stage matrix: {e}")
            return None
        
        # 转换矩阵值
        converted_grid = np.zeros_like(grid, dtype=int)
        converted_grid.fill(Grid.Wall.value)
        converted_grid[grid == 1] = Grid.Empty.value
        converted_grid[grid == 4] = Grid.MySpecial.value
        converted_grid[grid == 16] = Grid.HisSpecial.value
        converted_grid[grid == 64] = Grid.Wall.value
        
        # 创建Stage对象
        try:
            return Stage(converted_grid)
        except Exception as e:
            print(f"Error creating Stage object: {e}")
            return None
    
    def get_stage_info(self, stage_name: str) -> Optional[dict]:
        """获取场地信息（不创建Stage对象）"""
        return self.data_loader.get_stage_by_name(stage_name)
    
    def get_stage_properties(self, stage_name: str) -> dict:
        """获取场地属性摘要"""
        stage_info = self.get_stage_info(stage_name)
        if not stage_info:
            return {}
        
        dimensions = stage_info.get("Dimensions", {})
        special_points = stage_info.get("SpecialPoints", [])
        
        return {
            "name": stage_name,
            "name_zh": stage_info.get("name_zh", ""),
            "rows": dimensions.get("rows", 0),
            "cols": dimensions.get("cols", 0),
            "max_cells": stage_info.get("MaxCells", 0),
            "special_points_count": len(special_points),
            "special_points": special_points
        }
    
    def validate_stage_matrix(self, matrix: List[List[int]]) -> bool:
        """验证场地矩阵是否有效"""
        if not matrix:
            return False
        
        # 检查矩阵是否矩形
        if not all(len(row) == len(matrix[0]) for row in matrix):
            print("Invalid matrix: not rectangular")
            return False
        
        # 检查矩阵值
        valid_values = {1, 4, 16, 64}
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if val not in valid_values:
                    print(f"Invalid value at ({i},{j}): {val}")
                    return False
        
        return True
    
    def analyze_stage(self, stage_name: str) -> Dict[str, Any]:
        """分析场地特性"""
        stage_info = self.get_stage_info(stage_name)
        if not stage_info:
            return {"error": f"Stage not found: {stage_name}"}
        
        matrix = stage_info.get("square_matrix", [])
        if not matrix:
            return {"error": f"No matrix found for stage: {stage_name}"}
        
        grid = self.data_loader.convert_stage_matrix(matrix)
        
        # 统计各种类型的格子数量
        analysis = {
            "name": stage_name,
            "name_zh": stage_info.get("name_zh", ""),
            "dimensions": grid.shape,
            "total_cells": grid.size,
            "empty_cells": np.sum(grid == 1),
            "my_special_cells": np.sum(grid == 4),
            "his_special_cells": np.sum(grid == 16),
            "wall_cells": np.sum(grid == 64),
            "playable_cells": np.sum(grid == 1) + np.sum(grid == 4) + np.sum(grid == 16),
            "special_points": stage_info.get("SpecialPoints", []),
            "max_cells": stage_info.get("MaxCells", 0)
        }
        
        # 计算可涂地百分比
        if grid.size > 0:
            analysis["playable_percentage"] = (analysis["playable_cells"] / grid.size) * 100
        
        return analysis
    
    def print_stage_summary(self, stage_name: str):
        """打印场地摘要信息"""
        stage_info = self.get_stage_info(stage_name)
        if not stage_info:
            print(f"Stage not found: {stage_name}")
            return
        
        props = self.get_stage_properties(stage_name)
        print(f"Stage Name: {props['name']} ({props['name_zh']})")
        print(f"Dimensions: {props['rows']}x{props['cols']}")
        print(f"Max Cells: {props['max_cells']}")
        print(f"Special Points: {props['special_points_count']}")
        
        # 显示特殊点信息
        if props['special_points']:
            print("Special Points:")
            for sp in props['special_points']:
                player = sp.get('player', 'Unknown')
                sp_type = sp.get('type', 'Unknown')
                print(f"  Player {player} at ({sp.get('rows', '?')},{sp.get('cols', '?')}) - {sp_type}")
        
        # 显示矩阵预览
        matrix = stage_info.get("square_matrix", [])
        if matrix:
            print("Matrix preview (first 4 rows, first 8 cols):")
            for i in range(min(4, len(matrix))):
                row = matrix[i]
                row_preview = " ".join(f"{x:2d}" for x in row[:8])
                print(f"  {row_preview}")