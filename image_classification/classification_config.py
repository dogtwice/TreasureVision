# ------------ 数据路径与预处理配置 ------------
from denoising_config import PACKAGE_NAME


FASHION_LABELS_PATH = "../common/fashion-labels.csv"  # 原始图像存储根目录（需确保存在子目录）
IMG_PATH = "../common/dataset/"  # 原始图像存储根目录（需确保存在子目录）
IMG_HEIGHT = 512                  # 原始图像高度（注意：实际训练时会被Resize为64x64）
IMG_WIDTH = 512                   # 原始图像宽度（需检查与数据预处理的一致性）

# ------------ 随机性与数据划分配置 ------------
SEED = 42                         # 全局随机种子（确保实验可复现性）
TRAIN_RATIO = 0.75                # 训练集划分比例（75%训练，25%验证）
VAL_RATIO = 1 - TRAIN_RATIO       # 验证集比例（自动计算，无需修改）
SHUFFLE_BUFFER_SIZE = 100         # 数据混洗缓冲区大小（影响数据加载顺序随机性）

# ------------ 训练超参数配置 ------------
LEARNING_RATE = 1e-3              # 初始学习率（AdamW优化器使用）
EPOCHS = 30                       # 总训练轮次（需平衡过拟合与欠拟合）
TRAIN_BATCH_SIZE = 32             # 训练批次大小（GPU显存不足时可调小）
TEST_BATCH_SIZE = 32              # 验证/测试批次大小（建议与训练批次一致）
FULL_BATCH_SIZE = 32              # 全量数据生成嵌入时的批次大小


PACKAGE_NAME="image_classification"
CLASSIFIER_MODEL_NAME="classifier.pt"

classification_names = {
    0: '上身衣服',  # 数字 0 对应“上身衣服”
    1: '鞋',       # 数字 1 对应“鞋”
    2: '包',       # 数字 2 对应“包”
    3: '下身衣服',  # 数字 3 对应“下身衣服”
    4: '手表'      # 数字 4 对应“手表”
}
