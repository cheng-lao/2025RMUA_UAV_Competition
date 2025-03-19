import numpy as np
import torch
import torch.nn.functional as F

def getKMatrix():
    return np.array([[831.3843994140625,  0.0,                480.0],
        [0.0,                831.3843994140625,  360.0],
        [0.0,                0.0,                1.0]])

def getBaseline():
    """
    前视双目相机：位于机体中心向前 175mm 处，基线为 300mm，图像分辨率为 960*720，FOV 为60°，频率为 20Hz
    后视双目相机：位于机体中心向后 175mm 处，基线为 300mm，图像分辨率为 960*720，FOV 为60°，频率为 20Hz
    """ 
    return torch.Tensor([0.3]).float() # baseline = 0.3meter

class Normalize():
    '''
    RGB mode
    '''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['left'] = sample['left'] / 255.0
        sample['right'] = sample['right'] / 255.0

        sample['left'] = self.__normalize(sample['left'])
        sample['right'] = self.__normalize(sample['right'])

        return sample

    def __normalize(self, img):
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        return img

class ToTensor():

    def __call__(self, sample):
        left = sample['left']
        right = sample['right']
        # disp = sample['disp']
        # H x W x C ---> C x H x W
        sample['left'] = torch.from_numpy(left.transpose([2, 0, 1])).type(torch.FloatTensor)
        sample['right'] = torch.from_numpy(right.transpose([2, 0, 1])).type(torch.FloatTensor)
        # sample['disp'] = torch.from_numpy(disp).type(torch.FloatTensor)
        # if 'disp' in sample:
        #     sample['disp'] = torch.from_numpy(sample['disp']).type(torch.FloatTensor)

        return sample

class Pad():
    def __init__(self, H, W):
        self.w = W
        self.h = H

    def __call__(self, sample):
        pad_h = self.h - sample['left'].size(1)
        pad_w = self.w - sample['left'].size(2)

        left = sample['left'].unsqueeze(0)  # [1, 3, H, W]
        left = F.pad(left, pad=(0, pad_w, 0, pad_h))
        right = sample['right'].unsqueeze(0)  # [1, 3, H, W]
        right = F.pad(right, pad=(0, pad_w, 0, pad_h))
        # disp = sample['disp'].unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]
        # disp = F.pad(disp, pad=(0, pad_w, 0, pad_h))

        sample['left'] = left.squeeze()
        sample['right'] = right.squeeze()
        # sample['disp'] = disp.squeeze()

        return sample
