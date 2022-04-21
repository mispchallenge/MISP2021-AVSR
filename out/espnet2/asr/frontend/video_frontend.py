#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import torch.nn as nn
import numpy as np
import math
import os
user_path = os.path.expanduser('~')
EPS = 1e-16
from .network_resnet_conv2d import ResNet2D
from torchvision.transforms import CenterCrop, RandomCrop, RandomHorizontalFlip, Compose
#skip_gray True (B, T, 96, 96,3)->(B, T, 96, 96,3)  (B, T, 96, 96)->(B, T, 96, 96) 
# False  (B, T, 96, 96,3)->(B, T, 96, 96)
# size

class VideoFrontend(nn.Module):  ##(B, T, 96, 96,3) ->(B,T,512)
    def __init__(self,random=True,channel_input="bgr",size=[80,80],downsampling=False,hidden_channel_num=64):
        super(VideoFrontend, self).__init__()
        self.graycropflip=GrayCropFlip(random=True,skip_gray=False,channel_input="bgr",size=[80,80])
        
        self.downsampling = downsampling
        if downsampling:
            self.video_frontend = nn.Sequential(
                nn.Conv3d(1, hidden_channel_num, kernel_size=(5, 7, 7), stride=(2, 2, 2), padding=(1, 3, 3), bias=False),
                nn.BatchNorm3d(hidden_channel_num), nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1)))
        else:
            self.video_frontend = nn.Sequential(
            nn.Conv3d(1, hidden_channel_num, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(hidden_channel_num), nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        backbone_setting = {  
                        "block_type":"basic2d",
                        "block_num": 2,
                        "act_type": "prelu",
                        "hidden_channels": [ 64, 128, 256, 512 ],
                        "stride": [ 1, 2, 2, 2 ],
                        "expansion": 1,
                        "downsample_type": "avgpool",
                        "in_channels": hidden_channel_num}
        self.output_dim = backbone_setting["hidden_channels"][-1]
        self.resnet = ResNet2D(**backbone_setting)
    
    def output_size(self) -> int:
        return self.output_dim

    def forward(self, x,x_len):    
        x,_=self.graycropflip(x) #”bgr“
        assert x.dim() == 4, f'shape error: input must  (B, T, 88, 88)'
        B, T, _, _ = x.size() #(B, T, 88, 88) or  
        if self.downsampling:
            T = ((T-1)//2-1)//2
            x_len = ((x_len-1)//2-1)//2
        x = x.unsqueeze(1) #(B, 1, T, 88, 88) 
        x = self.video_frontend(x)  #(B, 64, T, 88, 88) 
        x = x.transpose(1, 2).contiguous()  #(B, T, 64, 88, 88) 
        x = x.view(-1, 64, x.size(3), x.size(4)) #(B*T, 64, 88, 88) 
        x,x_len = self.resnet(x,x_len) #(B*T, 512) 
        x = x.view(B, T, -1) #(B,T,512) 
        return x,x_len

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bnfc = nn.BatchNorm1d(num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bnfc(x)
        return x

# visual feature
class GrayCropFlip(nn.Module):
    def __init__(self, channel_input='bgr', size=None, random=False, skip_gray=False, **other_params):
        super(GrayCropFlip, self).__init__()
        self.skip_gray = skip_gray
        if not self.skip_gray:
            self.channel2idx = {channel_input[i]: i for i in range(len(channel_input))}
        if size is not None:
            self.random = random
            self.train_transform = Compose([
                RandomCrop(size=size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'),
                RandomHorizontalFlip(p=0.5)])
            self.eval_transform = Compose([CenterCrop(size=size)])

    def forward(self, x, length=None):
        if not self.skip_gray:
            assert x.shape[-1] == 3, 'shape error: input must have r,g,b 3 channels, but got {}'.format(x.shape)
            x_split = x.split(1, dim=-1)
            gray_frames = 0.114 * x_split[self.channel2idx['b']] + 0.587 * x_split[
                self.channel2idx['g']] + 0.299 * x_split[self.channel2idx['r']]
            x = gray_frames.sum(dim=-1)
        if hasattr(self, 'random'):
            x = self.train_transform(x) if self.training and self.random else self.eval_transform(x)
        return x, length

# frontend = VideoFrontend(downsampling=False)
# feats = torch.rand(16,55,96,96) 
# lengths = torch.randint(50,55,(16,))
# output,output_length = frontend(feats,lengths)#[B,T,D]->[B,T,D]
# print(output.shape,output_length) 