# 测试简化配置是否有效
print("=== 测试简化的extraPaths配置 ===")

# 这些导入应该都能工作
try:
    import common
    print("✓ 根目录模块导入成功")
except ImportError as e:
    print(f"✗ 根目录模块导入失败: {e}")

try:
    from common import utils
    print("✓ 嵌套模块导入成功")
except ImportError as e:
    print(f"✗ 嵌套模块导入失败: {e}")

try:
    # 测试从其他目录导入
    import sys
    import os
    
    # 添加路径测试
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    if project_root in sys.path:
        print("✓ 项目根目录在sys.path中")
    else:
        print("✗ 项目根目录不在sys.path中")
        
except Exception as e:
    print(f"✗ 路径测试失败: {e}")

print("简化配置测试完成")