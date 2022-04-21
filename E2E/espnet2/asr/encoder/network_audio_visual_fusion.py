#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import torch
import numpy as np
import torch.nn as nn
from .network_tcn_conv1d import MultibranchTemporalConv1DNet



class AudioVisualFuse(nn.Module):
    def __init__(self, fuse_type, fuse_setting):
        super(AudioVisualFuse, self).__init__()
        self.fuse_type = fuse_type
        if self.fuse_type == 'cat':
            self.out_channels = np.sum(fuse_setting['in_channels'])
        elif self.fuse_type == 'tcn':
            fuse_setting['in_channels'] = np.sum(fuse_setting['in_channels'])
            default_fuse_setting = {
                'hidden_channels': [256 *3, 256 * 3, 256 * 3], 'kernels_size': [3, 5, 7], 'dropout': 0.2,
                'act_type': 'prelu', 'dwpw': False, 'downsample_type': 'norm'}
            default_fuse_setting.update(fuse_setting)
            self.fusion = MultibranchTemporalConv1DNet(**default_fuse_setting)
            self.out_channels = default_fuse_setting['hidden_channels'][-1]
        else:
            raise NotImplementedError('unknown fuse_type')

    def forward(self, audios, videos, length=None):
        if self.fuse_type == 'cat':
            x = torch.cat(unify_time_dimension(*audios, *videos), dim=1)
        elif self.fuse_type == 'tcn':
            x = torch.cat(unify_time_dimension(*audios, *videos), dim=1)
            x, length = self.fusion(x, length)
        else:
            raise NotImplementedError('unknown fuse_type')
        return x, length

#auto check times between arrays 
def unify_time_dimension(*xes):
    lengths = [x.shape[2] for x in xes]
    # import pdb;pdb.set_trace()
    if len([*set(lengths)]) == 1:
        outs = [*xes]
    else:
        max_length = max(lengths)
        outs = []
        for x in xes:
            if max_length // x.shape[2] != 1:
                if max_length % x.shape[2] == 0:
                    x = torch.stack([x for _ in range(max_length // x.shape[2])], dim=-1).reshape(*x.shape[:-1],
                                                                                                  max_length)
                else:
                    raise ValueError('length error, {}'.format(lengths))
            else:
                pass
            outs.append(x)
    return outs


if __name__ == '__main__':
    audio_visual_fusion = AudioVisualFuse(
            fuse_type="tcn", fuse_setting={'in_channels': [512, 512]})
    # fused_z, length = audio_visual_fusion([audio_x], [visual_y], length)
    audio_x = torch.rand(16,512,100)
    visual_y = torch.rand(16,512,100)
    length = torch.randint(50,100,(16,))
    fused_z, length = audio_visual_fusion([audio_x], [visual_y], length) ##[b,512,t]->[b,512,t]  
    
    print(fused_z.shape,length)
    # y = unify_time_dimension(torch.ones(2, 2, 4), torch.ones(2, 2, 2), torch.ones(2, 2, 1))
    # for i in y:
    #     print(i.shape)


