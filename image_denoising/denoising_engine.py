#define the public implementations of this module
__all__ = ["train_step", "val_step"]

import torch

def train_step(denoiser,train_loader,loss_fn,optimizer,device):
    """执行完整的训练周期.
    parameters:
    
    returns:
        float: 平均训练损失.
    """
     #set the model to training mode(In this mode Dropout and BatchNorm can be used)
     #encoder.train()
     #decoder.train()
    totoal_loss=0
    current_trained_samples=0

    for train_img,target_img in train_loader:
        train_img, target_img = train_img.to(device), target_img.to(device)
        optimizer.zero_grad()
        outputs=denoiser(train_img)
        #print(outputs.shape,target_img.shape)
        loss=loss_fn(outputs,target_img)
        
        loss.backward()
        optimizer.step()
        totoal_loss+=loss.item()
        current_trained_samples+=1
    return totoal_loss/current_trained_samples


def val_step(denoiser,val_loader,loss_fn,device):
    """do validation step for one epoch.
    
    returns:
        float: average validation loss.
    """

    #set the model to evaluation mode (In this mode Dropout and BatchNorm are not used)
    #encoder.eval()
    #decoder.eval()

    total_loss=0
    current_val_samples=0 
    with torch.no_grad():
        for val_img,target_img in val_loader:
            val_img, target_img = val_img.to(device), target_img.to(device)
            outputs=denoiser(val_img)
            loss=loss_fn(outputs,target_img)
            total_loss+=loss.item()
            current_val_samples+=1

    return total_loss/current_val_samples



