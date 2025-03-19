"""
@authors: A Bhattacharya, et. al
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the models that were used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
import torch.nn.utils.spectral_norm as spectral_norm
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ViTsubmodules import *

def refine_inputs(X):
    """
    N = 1
    X[0] depth_image: torch.Tensor of shape (N, 1, 60, 90)
    X[1] desired_velocity期望的速度: torch.Tensor of shape (1, 1) default is 5.0
    X[2] quaternion位姿4元组: torch.Tensor of shape (N, 4) default is [1, 0, 0, 0]
    X[3] hidden state: turple(torch.Tensor)  hidden state of LSTM  | (3, 395) x 2
    """
    # fill quaternion rotation if not given
    # make it [1, 0, 0, 0] repeated with numrows = X[0].shape[0]
    if X[2] is None:
        # X[2] = torch.Tensor([1, 0, 0, 0]).float()
        X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
        X[2][:, 0] = 1

    # if input depth images are not of right shape, resize
    if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')

    return X

class ConvNet(nn.Module):
    """
    Conv + FC Network 
    Num Params: 235,269
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 3)
        self.conv2 = nn.Conv2d(4, 10, 3, 2)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(2, 1)
        self.bn1 = nn.BatchNorm2d(4)
        
        self.fc0 = nn.Linear(845, 256, bias=False)
        self.fc1 = nn.Linear(256, 64, bias=False)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        x = -self.maxpool(- self.bn1(F.relu(self.conv1(x))))
        x = self.avgpool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        metadata = torch.cat((X[1]*0.1, X[2]), dim=1).float()

        x = torch.cat((x, metadata), dim=1).float()

        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x, None #None is passed to be compatible with hidden dimensions

class LSTMNet(nn.Module):
    """
    LSTM + FC Network 
    Num Params: 2,949,937
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, stride = 3, padding=1)
        self.conv2 = nn.Conv2d(4, 10, 3,stride =  2, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.maxpool = nn.MaxPool2d(3, 1)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(10)

        self.lstm = LSTM(input_size=665, hidden_size=395,
                         num_layers=2, dropout=0.15, bias=False)
        self.fc1 = spectral_norm(nn.Linear(395, 64))
        self.fc2 = spectral_norm(nn.Linear(64, 16))
        self.fc3 = spectral_norm(nn.Linear(16, 3))

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        x = -self.maxpool(-self.bn1(F.relu(self.conv1(x))))
        x = self.avgpool(self.bn2(F.relu(self.conv2(x))))

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.cat((x,X[1]*0.1, X[2]), dim=1).float()
        if len(X)>3:
            x,h = self.lstm(x, X[3])
        else:
            x,h = self.lstm(x)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x, h

class LSTMNetVIT(nn.Module):
    """
    ViT+LSTM Network 
    Num Params: 3,563,663   
    """
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])

        self.decoder = spectral_norm(nn.Linear(4608, 512))  # 谱归一化 本质上是对权重矩阵进行归一化 计算方法不变
        self.lstm = (nn.LSTM(input_size=517, hidden_size=128,
                         num_layers=3, dropout=0.1))
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))

        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding = 1)

    def forward(self, X):
        """
        N = 1
        X[0] depth_image: torch.Tensor of shape (N, 1, 60, 90)
        X[1] desired_velocity期望的速度: torch.Tensor of shape (1, 1) default is 5.0
        X[2] quaternion位姿4元组: torch.Tensor of shape (N, 4) default is [1, 0, 0, 0]
        X[3] hidden state: turple(torch.Tensor)  hidden state of LSTM  | (3, 395) x 2
        """
        X = refine_inputs(X)

        x = X[0]
        embeds = [x]
        for block in self.encoder_blocks:   # encode
            embeds.append(block(embeds[-1]))        
        
        # for i in range(len(embeds)):
        #     print(f"embeds[{i}].shape is {embeds[i].shape}")
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]),self.up_sample(out[0])],dim=1) 
        out = self.down_sample(out)
        # print("after down_sample",out.shape)
        out = self.decoder(out.flatten(1))  # spectral_norm(nn.Linear(4608, 512))-->torch.Size([1, 512])
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float() # torch.Size([1, 517])
        # torch.cat 这个是将三个行向量按照列拼接在一起 512 + 1 + 4 = 517
        if len(X)>3:
            out,h = self.lstm(out, X[3])
        else:
            out,h = self.lstm(out)
        out = self.nn_fc2(out)
        return out, h

class ViT(nn.Module):
    """
    ViT+FC Network 
    Num Params: 3,101,199   
    """
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])        
        self.decoder = nn.Linear(4608, 512)
        self.nn_fc1 = spectral_norm(nn.Linear(517, 256))
        self.nn_fc2 = spectral_norm(nn.Linear(256, 3))
        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding = 1)

    def forward(self, X):

        X = refine_inputs(X)

        x = X[0]
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]),self.up_sample(out[0])],dim=1) 
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        out = F.leaky_relu(self.nn_fc1(out))
        out = self.nn_fc2(out)

        return out, None

class UNetConvLSTMNet(nn.Module):
    """
    UNet+LSTM Network 
    Num Params: 2,955,822 
    """

    def __init__(self):
        super().__init__()

        self.unet_e11 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.unet_e12 = nn.Conv2d(4, 4, kernel_size=3, padding=1) #(N, 4, 60, 90)
        self.unet_pool1 = nn.MaxPool2d(kernel_size=2, stride=3,) #(N, 4, 30, 45)

        self.unet_e21 = nn.Conv2d(4, 8, kernel_size=3, padding=1) #(N, 8, 26, 41)
        self.unet_e22 = nn.Conv2d(8, 8, kernel_size=3, padding=1) #(N, 8, 24, 39)
        self.unet_pool2 = nn.MaxPool2d(kernel_size=2, stride=2,) #(N, 8, 12, 19)

        #Input: (N, 8, 12, 19)
        self.unet_e31 = nn.Conv2d(8, 16, kernel_size=3, padding=1) #(N, 8, 10, 17)
        self.unet_e32 = nn.Conv2d(16, 16, kernel_size=3, padding=1) #(N, 16, 8, 15)

        self.unet_upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2,)
        self.unet_d11 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.unet_d12 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        self.unet_upconv2 = nn.ConvTranspose2d(8, 4, kernel_size=3, stride=3,)
        self.unet_d21 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.unet_d22 = nn.Conv2d(4, 4, kernel_size=3, padding=1)

        self.unet_out = nn.Conv2d(4, 1, kernel_size=1)

        self.conv_conv1 = nn.Conv2d(2, 4, 5, 3)
        self.conv_conv2 = nn.Conv2d(4, 10, 5, 2)
        self.conv_avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.conv_maxpool = nn.MaxPool2d(2, 1)
        self.conv_bn1 = nn.BatchNorm2d(4)

        self.lstm = LSTM(input_size=3065, hidden_size=200, num_layers=2, dropout=0.15, bias=False)

        self.nn_fc1 = torch.nn.utils.spectral_norm(nn.Linear(200, 64))
        self.nn_fc2 = torch.nn.utils.spectral_norm(nn.Linear(64, 32))
        self.nn_fc3 = torch.nn.utils.spectral_norm(nn.Linear(32, 3))

    def forward(self, X):

        X = refine_inputs(X)

        img, des_vel, quat = X[0], X[1], X[2]
        y_e1 = torch.relu(self.unet_e12(torch.relu(self.unet_e11(img))))
        unet_enc1 = self.unet_pool1(y_e1)
        y_e2 = torch.relu(self.unet_e22(torch.relu(self.unet_e21(unet_enc1))))
        unet_enc2 = self.unet_pool2(y_e2)
        y_e3 = torch.relu(self.unet_e32(torch.relu(self.unet_e31(unet_enc2))))

        unet_dec1 = torch.relu(self.unet_d12(torch.relu(self.unet_d11(torch.cat([self.unet_upconv1(y_e3), y_e2], dim=1)))))
        unet_dec2 = torch.relu(self.unet_d22(torch.relu(self.unet_d21(torch.cat([self.unet_upconv2(unet_dec1), y_e1], dim=1)))))

        y_unet = self.unet_out(unet_dec2)
        x_conv = torch.cat((img, y_unet), dim=1)

        y_conv = -self.conv_maxpool(-torch.relu(self.conv_bn1(self.conv_conv1(x_conv))))
        y_conv = self.conv_avgpool(torch.relu(self.conv_conv2(y_conv)))

        x_lstm = torch.cat([torch.flatten(y_conv, 1), torch.flatten(y_e3, 1), des_vel*0.1, quat], dim=1).float()

        if len(X)>3:
            y_lstm, h = self.lstm(x_lstm, X[3])
        else:
            y_lstm, h = self.lstm(x_lstm)

    
        y_fc1 = F.leaky_relu(self.nn_fc1(y_lstm))
        y_fc2 = F.leaky_relu(self.nn_fc2(y_fc1))
        y = self.nn_fc3(y_fc2)

        return y, h


class LSTMNetVIT_Modified(nn.Module):
    """
    ViT+LSTM Network 
    Num Params: 2,383,600   
    """
    def __init__(self):
        super().__init__()
        self.encoder_blocks_front = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])
        # self.encoder_blocks_back = nn.ModuleList([
        #     MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
        #     MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        # ])
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        
        self.down_sample_front = nn.ModuleList([                # channel   (H, W)
            BottleNeck(48, 24, down_scale=True),                # 48 -> 24, (180, 240) -> (90, 120)
            ConvMoudle(24, 12, 1, 2),                           # 24 -> 12, (90, 120) -> (45, 60)
            BottleNeck(12, 6, down_scale=True),                 # 12 -> 6,  (45, 60) -> (22, 30)
            ConvMoudle(6, 6, 1, 2),                             # 6 -> 6, (22, 30) -> (11, 15)
            nn.SiLU()
        ])
        
        # self.down_sample_back = nn.ModuleList([                 # channel   (H, W)
        #     BottleNeck(48, 24, down_scale=True),                # 48 -> 24, (180, 240) -> (90, 120)
        #     ConvMoudle(24, 12, 1, 2),                           # 24 -> 12, (90, 120) -> (45, 60)
        #     BottleNeck(12, 6, down_scale=True),                 # 12 -> 6,  (45, 60) -> (22, 30)
        #     ConvMoudle(6, 6, 1, 2),                             # 6 -> 6, (22, 30) -> (11, 15)
        #     nn.SiLU()
        # ])
        self.melt_c = nn.Conv2d(6, 6, 1)
        
        self.decoder = spectral_norm(nn.Linear(1080, 512))
        self.lstm = (nn.LSTM(input_size=517, hidden_size=128,
                         num_layers=3, dropout=0.1))
        self.nn_fc2 = spectral_norm(nn.Linear(128, 6))
    
    def forward(self, X):
        """模型推理主函数
        Args:
            # X (list): X[0]: 输入的深度信息图(2, C, H:720, W:960)
            X[0]: 前方深度信息图
            X[1]: 期望的速度
            X[2]: 位姿四元数
            X[3]: LSTM隐藏状态

        Returns:
            _type_: _description_
        """
        
        x = X[0]
        embeds_front = [x]
        for block in self.encoder_blocks_front:
            embeds_front.append(block(embeds_front[-1]))        
        out_front = embeds_front[1:]
        
        # embeds_back  = [x[1]]
        # for block in self.encoder_blocks_back:
        #     embeds_back.append(block(embeds_back[-1]))
        # out_back = embeds_back[1:]
        
        # out_front(out_back):
        # 0: torch.Size([1, 32, 180, 240])
        # 1: torch.Size([1, 64, 90, 120])    
        
        out_front = torch.cat([self.pxShuffle(out_front[1]), out_front[0]], dim=1)
        for m in self.down_sample_front:
            out_front = m(out_front)
        # out_back = torch.cat([self.pxShuffle(out_back[1]), out_back[0]], dim=1)
        # for m in self.down_sample_back:
        #     out_back = m(out_back)

        out = self.melt_c(out_front)

        out = self.decoder(out.flatten(1))
        
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        if len(X)>3:
            out,h = self.lstm(out, X[3])
        else:
            out,h = self.lstm(out)
        out = self.nn_fc2(out)
        return out, h

if __name__ == '__main__':
    print("MODEL NUM PARAMS ARE")
    # model = ConvNet().float()
    # print("ConvNet: ")
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model = LSTMNet().float()
    # print("LSTMNet: ")
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model = UNetConvLSTMNet().float()
    # print("UNET: ")
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model = ViT().float()
    # print("VIT: ")
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model = LSTMNetVIT().float()
    # print("VITLSTM: ")
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    depth_image = torch.randn(1, 1, 720, 960)
    desired_velocity = torch.randn(1, 1)
    quaternion = torch.randn(1, 4)
    hidden_state = None
    model = LSTMNetVIT_Modified().float()
    model.load_state_dict(torch.load('../training/model_epoch10.pth'))
    print("LSTMNetVIT_Modified: ")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("MODEL OUTPUT SHAPE IS")
    print(model([depth_image, desired_velocity, quaternion, hidden_state])[0].shape)


    """
    MODEL NUM PARAMS ARE
    VITLSTM: 
    3563663
    MODEL OUTPUT SHAPE IS
    embeds[0].shape is torch.Size([1, 1, 60, 90])
    embeds[1].shape is torch.Size([1, 32, 15, 23]) nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)-->torch.Size([1, 32, 16, 24])
    embeds[2].shape is torch.Size([1, 64, 8, 12]) nn.PixelShuffle(upscale_factor=2)-->torch.Size([1, 16, 16, 24])
    after down_sample torch.Size([1, 12, 16, 24])
    torch.Size([1, 3])
    """
    # print("MODEL OUTPUT IS")
    # print(model([depth_image, desired_velocity, quaternion, hidden_state])[0])
    # print("MODEL HIDDEN STATE SHAPE IS")
    # print(model([depth_image, desired_velocity, quaternion, hidden_state])[1][0].shape)