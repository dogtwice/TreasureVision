import os
import sys

print("当前工作目录:", os.getcwd())
print("PYTHONPATH:", os.environ.get('PYTHONPATH', '未设置'))
print("Python路径:", sys.path[:])

# 测试导入自定义模块
try:
    from common.utils import *
    print("✅ 成功导入 common.utils")
except ImportError as e:
    print("❌ 导入 common.utils 失败:", e)

try:
    from image_denoising.denoising_config import *
    print("✅ 成功导入 image_denoising.denoising_config")
except ImportError as e:
    print("❌ 导入 image_denoising.denoising_config 失败:", e)

