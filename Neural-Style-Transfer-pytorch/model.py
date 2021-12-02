'''
VGG19
(0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace)
(2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(3): ReLU(inplace)
(4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(6): ReLU(inplace)
(7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(8): ReLU(inplace)
(9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(11): ReLU(inplace)
(12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(13): ReLU(inplace)
(14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(15): ReLU(inplace)
(16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(17): ReLU(inplace)
(18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(20): ReLU(inplace)
(21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(22): ReLU(inplace)
(23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(24): ReLU(inplace)
(25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(26): ReLU(inplace)
(27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
(28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(29): ReLU(inplace)
(30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(31): ReLU(inplace)
(32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(33): ReLU(inplace)
(34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(35): ReLU(inplace)
(36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
'''

###############################################################
# The Neural Style Transfer includes only 10 layers of VGG19. #
# Conv: ['0','2','5','7', '10']                               #
# ReLU: ['1','3','6','8']                                     #
# Maxpool: ['4','9']                                          #
###############################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import torchvision.models as models
import numpy as np
import PIL
from matplotlib import pyplot as plt
import utils
import kornia
import torchvision.transforms as transforms


gram_matrix = utils.gram_matrix
Normalization = utils.Normalization

def tensor_to_image(tensor):
    image = tensor.clone()
    image = image.squeeze(0)
    pil = transforms.ToPILImage()
    target_img_PIL = pil(image)
    target_img_PIL= np.array(target_img_PIL)
    return target_img_PIL

def to_pil(tensor):
    image = tensor.squeeze(0)
    image = transforms.Resize((223, 259))(image)
    # print(image.shape)
    image = transforms.ToPILImage()(image)
    
    return image

## 'content loss function' 
class CL(nn.Module):
    def __init__(self, target):
        super(CL, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# style loss
class SL(nn.Module):
    def __init__(self, target):
        super(SL, self).__init__()
        self.target = gram_matrix(target).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# edge loss
class EL(nn.Module):
    def __init__(self, target, input_img):
        super(EL, self).__init__()
        self.input_img = input_img
        # print(target.shape)
        # self.target = tensor_to_image(target.cpu())
        
        
        # print(target.shape)
        # self.target = tensor_to_image(target)
        # print(self.target.shape)
        # self.target = torch.Tensor(cv.Canny(image=self.target,threshold1= 85, threshold2 =255))
        self.threshold = nn.Threshold(205, 1e-6)

        filter_x = torch.tensor(np.array([[[[-0.5, 0, 0.5], [-1, 0, 1], [-0.5, 0, 0.5]]]]))
        filter_y = torch.tensor(np.array([[[[-0.5, -1, -0.5], [0, 0, 0], [0.5, 1, 0.5]]]]))

        self.sobel_filter_x = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1, bias=False);
        self.sobel_filter_x.weight = nn.Parameter(filter_x.float(), requires_grad=False)
        self.sobel_filter_x.to(input_img.device)

        self.sobel_filter_y = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), padding=1, bias=False);
        self.sobel_filter_y.weight = nn.Parameter(filter_y.float(), requires_grad=False)
        self.sobel_filter_y.to(input_img.device)

        image = target.clone().detach().squeeze(dim=0)[0].unsqueeze(0).unsqueeze(0)
        self.target_r = self.sobel_filter_x(image) + self.sobel_filter_y(image)
        image = target.clone().detach().squeeze(dim=0)[1].unsqueeze(0).unsqueeze(0)
        self.target_g = self.sobel_filter_x(image) + self.sobel_filter_y(image)
        image = target.clone().detach().squeeze(dim=0)[2].unsqueeze(0).unsqueeze(0)
        self.target_b = self.sobel_filter_x(image) + self.sobel_filter_y(image)

        negated_r = torch.ones(image.shape).to(image.device) * 255 - (self.target_r.squeeze(0) + 50)
        negated_g = torch.ones(image.shape).to(image.device) * 255 - (self.target_g.squeeze(0) + 50)
        negated_b = torch.ones(image.shape).to(image.device) * 255 - (self.target_b.squeeze(0) + 50)

        self.target_r = self.threshold(negated_r)    
        self.target_g = self.threshold(negated_g)   
        self.target_b = self.threshold(negated_b)   

    def forward(self, input):
        # img = (self.input_img.clone().detach().cpu().numpy().squeeze(0).transpose((1,2,0))*255).astype(np.uint8)
        # edge_map = cv.Canny(image=img,threshold1= 85,threshold2= 255)
        
        image = input.squeeze(dim=0)[0].unsqueeze(0).unsqueeze(0)
        edge_r = self.sobel_filter_x(image) + self.sobel_filter_y(image)
        image = input.squeeze(dim=0)[1].unsqueeze(0).unsqueeze(0)
        edge_g = self.sobel_filter_x(image) + self.sobel_filter_y(image)
        image = input.squeeze(dim=0)[2].unsqueeze(0).unsqueeze(0)
        edge_b = self.sobel_filter_x(image) + self.sobel_filter_y(image)

        negated_r = torch.ones(image.shape).to(image.device) * 255 - (edge_r.squeeze(0) + 50)
        negated_g = torch.ones(image.shape).to(image.device) * 255 - (edge_g.squeeze(0) + 50)
        negated_b = torch.ones(image.shape).to(image.device) * 255 - (edge_b.squeeze(0) + 50)
        
        edge_r = self.threshold(negated_r)    
        edge_g = self.threshold(negated_g)   
        edge_b = self.threshold(negated_b) 

        print("edge require grad: ", edge_r.requires_grad)

        self.loss = F.mse_loss(edge_r, self.target_r) + F.mse_loss(edge_g, self.target_g) + F.mse_loss(edge_b, self.target_b)

        return input


## Generating the 'Neural Style Transfer' Model
def nst_model(content_img, style_img, input_img = None, device = 'cuda:0'):
    # print(content_img.shape)
    vgg = models.vgg19(pretrained=True).features.eval()
    vgg = vgg.to(device)
    normalization = Normalization(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).cuda()
    content_img = content_img.detach()
    style_img = style_img.detach()
    content_losses = []
    style_losses = []
    edge_losses = []

    model = nn.Sequential(normalization).to(device)
    
    i = 0
    for name, layer in vgg._modules.items():
        if name in ['0','2','5','10']:
            if name =='0':
                edge_loss = EL(content_img, input_img)
                edge_losses.append(edge_loss)
                model.add_module('edgeloss_{}'.format(i),edge_loss)
            model.add_module('conv_{}'.format(i),layer)
            style_target = model(style_img)
            style_loss = SL(style_target)
            style_losses.append(style_loss)
            model.add_module('styleloss_{}'.format(i),style_loss)
            i += 1

        elif name in ['7']:
            model.add_module('conv_{}'.format(i),layer)
            content_target = model(content_img)
            content_loss = CL(content_target)
            content_losses.append(content_loss)
            model.add_module('contentloss_{}'.format(i),content_loss)
            style_target = model(style_img)
            style_loss = SL(style_target)
            style_losses.append(style_loss)
            model.add_module('styleloss_{}'.format(i),style_loss)

            i += 1

        elif name in ['1','3','6','8']:
            layer = nn.ReLU(inplace=False)
            model.add_module('relu_{}'.format(i),layer)
            i += 1

        elif name in ['4','9']:
            model.add_module('maxpool_{}'.format(i),layer)
            i += 1

        elif name == '11':
            break

            
    return model, style_losses, content_losses, edge_losses