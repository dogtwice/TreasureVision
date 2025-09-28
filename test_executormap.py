import os
import json

print("=== 变量验证测试 ===")
print(f"当前工作目录 os.getcwd(): {os.getcwd()}")
print(f"当前文件路径 __file__: {__file__}")
print(f"当前文件目录: {os.path.dirname(__file__)}")

# 检查环境变量
print("\n=== 环境变量检查 ===")
for var in ['PYTHONPATH', 'workspaceFolder', 'workspaceRoot']:
    value = os.environ.get(var, '未设置')
    print(f"{var}: {value}")

# 检查VS Code相关的环境变量
print("\n=== VS Code相关环境变量 ===")
vscode_vars = [k for k in os.environ.keys() if 'vscode' in k.lower() or 'code' in k.lower()]
for var in vscode_vars:
    print(f"{var}: {os.environ[var]}")

# 查找工作区根目录
def find_workspace_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, '.vscode')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    return None

workspace_root = find_workspace_root()
print(f"\n=== 实际工作区检测 ===")
print(f"实际工作区根目录: {workspace_root}")
print(f"当前目录是否为工作区根目录: {os.getcwd() == workspace_root}")

# 导入测试
print(f"\n=== 导入测试 ===")
try:
    import sys
    sys.path.insert(0, workspace_root) if workspace_root else None
    import denoising_config
    print("✅ 从实际工作区根目录导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")

# 模拟Code Runner应该执行的命令
expected_cmd = f'cd "{workspace_root}";$env:PYTHONPATH="{workspace_root}";'
print(f"\n=== 期望的Code Runner命令 ===")
print(f"期望命令: {expected_cmd}")