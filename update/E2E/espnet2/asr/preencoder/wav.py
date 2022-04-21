import copy
from typing import Optional
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from typeguard import check_argument_types
from .network_resnet_conv1d import ResNet1D
from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs
from torch import nn


def variable_activate(act_type, in_channels=None, **other_params):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'prelu':
        return nn.PReLU(num_parameters=in_channels)
    else:
        raise NotImplementedError('activate type not implemented')

class WavPreEncoder(AbsPreEncoder): #[B,T]->[B,T,512]
    def __init__(
    self,
    conv1d_dim=64,
    conv1d_kernel_size=80,
    conv1d_stride=4,
    res_block_num=2,
    res_stride=[2, 2, 2, 2],
    res_expansion=1,
    res_hidden_channels=[64, 128, 256, 512],
    res_downsample_type="avgpool",
    act_type='prelu',
    ):   
        super().__init__()
        
        default_frontend_setting = {
            "in_channels":1,
            "out_channels":conv1d_dim,
            "kernel_size":conv1d_kernel_size,
            "stride":conv1d_stride,
            "padding":(conv1d_kernel_size-conv1d_stride)//2,#(kernel_size - stride) // 2
            "bias":False,
        }
        self.frontend = nn.Sequential(
            nn.Conv1d(**default_frontend_setting),
            nn.BatchNorm1d(default_frontend_setting["out_channels"]),
            variable_activate(act_type=act_type, in_channels=default_frontend_setting["out_channels"]))

        default_backbone_setting = {
            'block_type': 'basic1d', 'block_num': res_block_num, 'act_type': act_type,
            'hidden_channels': res_hidden_channels, 'stride': res_stride, 'expansion': res_expansion,
            'downsample_type': res_downsample_type} 

        self.backbone = ResNet1D(**default_backbone_setting)
        self.pool = nn.AvgPool1d(10, stride=10) # original size=3 errorr
        # self.pool = nn.AvgPool1d(3, stride=10) # original size=3 errorr

    def forward(self,x: torch.Tensor, length: torch.Tensor
    )-> Tuple[torch.Tensor, torch.Tensor]:
        # log_file = open("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/exp/wav_farfarlip_a/test","a")
        # print("-----before wav frontend ------",file=log_file)
        # print(f"x_size:{x.size()},lengths:{length}",file=log_file)
        x = x.unsqueeze(1)# [B,T] -> [B,1,T]
        x = self.frontend(x)# [B,1,T] ->  [B,64,T//4] 
        length = length // 4
        x, length = self.backbone(x, length) # [B,64,T//4] -> [B,512,T//64,] 
        x = self.pool(x).transpose(1,2) # [B,512,T//64]->[B,T//640,512]
        length = length // 10
        if x.size(1) > length.max(): # x may lager than length for 1
            length+=(x.size(1)-length.max())
        
        # print("-----after wav frontend ------",file=log_file)
        # print(f"x_size:{x.size()},lengths:{length}",file=log_file)
        return x, length


    def output_size(self) -> int:
        return 512

class featPreEncoder(AbsPreEncoder): #[B,T,D]->[B,T,512]
    def __init__(
    self,
    feat_dim=80,
    conv1d_dim=64,
    conv1d_kernel_size=1,
    conv1d_stride=1,
    res_block_num=2,
    res_stride=[1, 1, 1, 1],
    res_expansion=1,
    res_hidden_channels=[64, 128, 256, 512],
    res_downsample_type="avgpool",
    act_type='prelu',
    ):   
        super().__init__()
   
        default_frontend_setting = {
            "out_channels":conv1d_dim,
            "kernel_size":conv1d_kernel_size,
            "stride":conv1d_stride,
            "bias":False,
            "padding":0,
            "in_channels":feat_dim,
        }
        self.frontend = nn.Sequential(
            nn.Conv1d(**default_frontend_setting),
            nn.BatchNorm1d(default_frontend_setting["out_channels"]),
            variable_activate(act_type=act_type, in_channels=default_frontend_setting["out_channels"]))

        default_backbone_setting = {
            'block_type': 'basic1d', 'block_num': res_block_num, 'act_type': act_type,
            'hidden_channels': res_hidden_channels, 'stride': res_stride, 'expansion': res_expansion,
            'downsample_type': res_downsample_type}

        self.backbone = ResNet1D(**default_backbone_setting)


    def forward(self,x: torch.Tensor, length: torch.Tensor
    )-> Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1,2)
        x = self.frontend(x) 
        x, length = self.backbone(x, length)  
        x = x.transpose(1,2)
        return x, length


    def output_size(self) -> int:
        return 512


# frontend = featPreEncoder()
# feats = torch.rand(16,55,80) 
# lengths = torch.randint(50,55,(16,))
# output,output_length = frontend(feats,lengths)#[B,T,D]->[B,T,D]
# print(output.shape,output_length) 

# frontend = WavPreEncoder()
# feats = torch.rand(16,80000) 
# lengths = torch.randint(80000,80001,(16,))
# output,output_length = frontend(feats,lengths)#[B,T]->[B,T,512]
# print(output.shape,output_length) 
