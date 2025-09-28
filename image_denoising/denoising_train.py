import torch
import denoising_model
import denoising_engine
import torchvision.transforms as T
import denoising_data
import denoising_config
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from common import utils



def test(denoiser, val_loader, device):
    import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图
    dataiter=iter(val_loader)
    noise_imgs,images=next(dataiter)
    print("size of noise images:",noise_imgs.shape)

    denoiser=denoiser.to(device)
    noise_imgs, images = noise_imgs.to(device), images.to(device)
    print("size of noise images after to device:",noise_imgs.shape)
    output=denoiser(noise_imgs)
    print("size of denoised images:",output.shape)

    noise_imgs=noise_imgs.cpu().numpy()

    noise_imgs=noise_imgs.transpose(0,2,3,1)
    output=output.view(denoising_config.TEST_BATCH_SIZE,3,68,68)
    output=output.cpu().detach().numpy()
    output=output.transpose(0,2,3,1)
    print("current shape of output:",output.shape)
    original_images = images.cpu().numpy().transpose((0, 2, 3, 1))
    print("original_image shape: ", original_images.shape)

    # 绘制前 10 张输入图像和重建图像
    fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(25, 4))  # 创建 2 行 10 列的子图
    # 第一行显示噪声图像，第二行显示重建图像
    for imgs, row in zip([noise_imgs, original_images, output], axes):  # 遍历噪声图像和重建图像
        for img, ax in zip(imgs, row):  # 遍历每张图像和对应的子图
            ax.imshow(np.squeeze(img))  # 显示图像，并去除多余的维度
            ax.get_xaxis().set_visible(False)  # 隐藏 x 轴
            ax.get_yaxis().set_visible(False)  # 隐藏 y 轴
    plt.show()  # 显示图像

if __name__ == "__main__":

    if torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    
    print("当前使用的设备:",device)
    print(f"设置的随机种子是:{denoising_config.SEED}")
    utils.set_seed(denoising_config.SEED)

    transforms=T.Compose([T.Resize((68,68)),T.ToTensor()])
    print("正在创建数据集")
    full_dataset=denoising_data.FolderDataset(denoising_config.IMG_PATH,transform=transforms)
    train_size=int(denoising_config.TRAIN_RATIO*len(full_dataset))
    val_size=len(full_dataset)-train_size
    train_dataset,val_dataset=torch.utils.data.random_split(full_dataset,[train_size,val_size])

    train_loader=torch.utils.data.DataLoader(train_dataset,
                                             batch_size=denoising_config.TRAIN_BATCH_SIZE,
                                             shuffle=True,drop_last=True)
    val_loader=torch.utils.data.DataLoader(val_dataset,
                                           batch_size=denoising_config.TEST_BATCH_SIZE,
                                           shuffle=False,drop_last=False)
    
    denoiser=denoising_model.ConvDenoiser()
    loss_fn=nn.MSELoss()
    denoiser=denoiser.to(device)
    optimizer=optim.Adam(denoiser.parameters(),lr=denoising_config.LEARNING_RATE)

    min_loss=9999
    print("训练")
    for epoch in tqdm(range(denoising_config.EPOCHS)):
        train_loss=denoising_engine.train_step(denoiser,train_loader,loss_fn,optimizer,device)
        val_loss=denoising_engine.val_step(denoiser,val_loader,loss_fn,device)
        print(f"epoch:{epoch+1},train_loss:{train_loss},val_loss:{val_loss}")
        if val_loss<min_loss:
            min_loss=val_loss
            torch.save(denoiser.state_dict(),f"{denoising_config.DENOISER_MODEL_NAME}")
            print(f"模型已保存,当前最小验证集损失为:{min_loss}")
        else:
            print(f"当前最小验证集损失为:{min_loss},未保存模型")

    print("训练结束")

    load_denoiser=denoising_model.ConvDenoiser()
    load_denoiser.load_state_dict(torch.load(f"{denoising_config.DENOISER_MODEL_NAME}", map_location=device))
    load_denoiser.to(device)
    print("加载训练好的模型进行测试")
    test(load_denoiser,val_loader,device)