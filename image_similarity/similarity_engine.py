# 定义模块的公开接口
__all__ = ["train_step", "val_step", "create_embedding"]

# 导入PyTorch核心库和神经网络模块
import torch

def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device):
    """
    执行一个完整的训练迭代

    参数:
    - encoder: 卷积编码器（如ConvEncoder）
    - decoder: 卷积解码器（如ConvDecoder）
    - train_loader: 训练数据加载器，提供批次化的（输入图像, 目标图像）
    - loss_fn: 损失函数（如MSE）
    - optimizer: 优化器（如Adam）
    - device: 计算设备（"cuda" 或 "cpu"）

    返回值:
    - 当前epoch的平均训练损失（标量值）
    """

    total_loss = 0  # 累计损失
    num_batches = 0  # 批次计数器

    # 遍历训练数据加载器中的所有批次
    for train_img, target_img in train_loader:
        # 将数据移动到指定设备（GPU/CPU）
        train_img = train_img.to(device)
        target_img = target_img.to(device)

        # 清空优化器中之前的梯度
        optimizer.zero_grad()

        # 前向传播：编码器生成潜在表示
        enc_output = encoder(train_img)
        # 前向传播：解码器重建图像
        dec_output = decoder(enc_output)

        # 计算重建损失（预测图像与目标图像的差异）
        loss = loss_fn(dec_output, target_img)

        # 反向传播：计算梯度
        loss.backward()

        # 优化器更新模型参数
        optimizer.step()

        total_loss += loss.item()  # 累加损失值
        num_batches += 1

    return total_loss / num_batches  # 返回平均损失


def val_step(encoder, decoder, val_loader, loss_fn, device):
    """
    执行验证步骤（不更新参数）

    参数与train_step类似，但不含优化器参数

    返回值:
    - 验证集的平均损失（标量值）
    """

    total_loss = 0
    num_batches = 0

    # 禁用梯度计算以节省内存和计算资源
    with torch.no_grad():
        for train_img, target_img in val_loader:
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            # 前向传播
            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)

            # 计算损失
            loss = loss_fn(dec_output, target_img)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches

def create_embedding(encoder, full_loader,embedding_dim,device):
    embeding=torch.empty(0)
    with torch.no_grad():
        for train_img, target_img in full_loader:
            train_img = train_img.to(device)
            enc_output = encoder(train_img).cpu()
            enc_output=enc_output.view(enc_output.size(0),-1) # flatten the tensor
            embedding=torch.cat((embeding,enc_output),0)
    
    return embedding
