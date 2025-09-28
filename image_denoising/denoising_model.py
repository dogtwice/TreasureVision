#define the public implementations of this module, only expose the ConvEncoder class and ConvDecoder class
__all__ = ["ConvDenoiser"]

import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        # 编码器
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 68->34
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 34->17
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 17->8 (向下取整)

        # 解码器 - 需要处理尺寸不匹配问题
        self.t_conv1 = nn.ConvTranspose2d(8, 16, 2, stride=2)   # 8->16
        self.t_conv2 = nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1)  # 16->31
        self.t_conv3 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1)   # 31->61
        
        # 添加最终调整层来确保输出为68x68
        self.final_conv = nn.Conv2d(3, 3, 3, padding=1)
        self.upsample = nn.Upsample(size=(68, 68), mode='bilinear', align_corners=False)

    def forward(self, x):
        # 编码
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # 解码
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        
        # 最终调整到目标尺寸 (1,3,68,68)
        x = self.final_conv(x)
        x = self.upsample(x)
        x = torch.sigmoid(x)
        return x

if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 68,68)  # Batch size of 1, 3 channels, 512x512 image
    denoiser=ConvDenoiser()
    encoded_tensor=denoiser(input_tensor)
    print(encoded_tensor.shape,input_tensor.shape)  # Should print torch.Size([1, 3, 64, 64])
    assert encoded_tensor.shape == input_tensor.shape, "Output shape does not match input shape"
    print("Denoiser model test passed!")