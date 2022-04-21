# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

from typing import Optional
from typing import Tuple

import logging
import torch

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from .network_audio_visual_fusion import AudioVisualFuse
# from network_audio_visual_fusion import AudioVisualFuse

class AVConformerEncoder(AbsEncoder):
    def __init__(
        self,
        conformer_conf:dict,
        feat_dim:int,
        alayer_num1:int,
        alayer_num2:int,
        alayer_num3:int,
        vlayer_num1:int,
        vlayer_num2:int,
        vlayer_num3:int,
        avlayer_num:int
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.input_layer = Conv2dSubsampling(
                feat_dim,
                conformer_conf["output_size"],
                conformer_conf["dropout_rate"],
                None)

        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf) #incluee embedding layer
        self.alayer2 = ConformerEncoder(num_blocks=alayer_num2,**conformer_conf)
        self.alayer3 = ConformerEncoder(num_blocks=alayer_num3,**conformer_conf)
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**conformer_conf) #incluee embedding layer
        self.vlayer2 = ConformerEncoder(num_blocks=vlayer_num2,**conformer_conf)
        self.vlayer3 = ConformerEncoder(num_blocks=vlayer_num3,**conformer_conf)
        self.fusion = torch.nn.Sequential(
                        torch.nn.Linear(conformer_conf["output_size"]*2, conformer_conf["output_size"]),
                        torch.nn.LayerNorm(conformer_conf["output_size"]),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(conformer_conf["dropout_rate"]),              
        )  
        self.avlayer = ConformerEncoder(num_blocks=avlayer_num,**conformer_conf)
          
         
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """

        # aduio feat and video subsampling
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device)
        org_feats,org_masks = self.input_layer(feats,masks)
        org_feats_lengths = org_masks.squeeze(1).sum(1)
        masks = (~make_pad_mask(video_lengths)[:, None, :]).to(video.device)
       

        # log_file = open("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/exp/avsr_train_avsr_conformer_raw_zh_char_sp/test","a")
        # print("#"*40,"input layer","#"*40,file=log_file)
        # print(org_feats.shape,org_feats_lengths,file=log_file)
        # print(org_video.shape,org_video_lengths,file=log_file)
        
       
        #layer 1
        outfeats1,outfeats_lengths1,_ = self.alayer1(org_feats,org_feats_lengths)
        outvideo1,outvideo_lengths1,_ = self.vlayer1(video,org_feats_lengths)

        #fusion 1+layer 2
        # import pdb;pdb.set_trace()
        x_concat = torch.cat((outfeats1, outvideo1), dim=-1)
        amid_feat1= self.fusion(x_concat)
       
        outfeats2,outfeats_lengths2,_ = self.alayer2(amid_feat1,outfeats_lengths1)
        outvideo2,outvideo_lengths2,_ = self.vlayer2(outvideo1,outvideo_lengths1)


        # print("#"*40,"layertwo","#"*40,file=log_file)
        # print(outfeats2.shape,outfeats_lengths2,file=log_file)
        # print(outvideo2.shape,outvideo_lengths2,file=log_file)
        #skip connection + layer 3
        outfeats3,outfeats_lengths3,_ = self.alayer3(org_feats+outfeats2,outfeats_lengths2)
        outvideo3,outvideo_lengths3,_ = self.vlayer3(video+outvideo2,outvideo_lengths2)

        # print("#"*40,"layerthree","#"*40,file=log_file)
        # print(outfeats3.shape,outfeats_lengths3,file=log_file)
        # print(outvideo3.shape,outvideo_lengths3,file=log_file)
        #fusion 2 + layer 4
        x_concat = torch.cat((outfeats3, outvideo3), dim=-1)
        amid_feat2= self.fusion(x_concat)
        hidden_feat,hidden_feat_lengths,_ = self.avlayer(amid_feat2,outfeats_lengths3)

        return hidden_feat,hidden_feat_lengths,_
              
class AVConformerEncoder2(AbsEncoder):
    def __init__(
        self,
        conformer_conf:dict,
        feat_dim:int,
        alayer_num1:int,
        alayer_num2:int,
        alayer_num3:int,
        vlayer_num1:int,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.input_layer = Conv2dSubsampling(
                feat_dim,
                conformer_conf["output_size"],
                conformer_conf["dropout_rate"],
                None)
        
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf) #incluee embedding layer
        self.alayer2 = ConformerEncoder(num_blocks=alayer_num2,**conformer_conf)
        self.alayer3 = ConformerEncoder(num_blocks=alayer_num3,**conformer_conf)
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**conformer_conf) #incluee embedding layer
        self.fusion = torch.nn.Sequential(
                        torch.nn.Linear(conformer_conf["output_size"]*2, conformer_conf["output_size"]),
                        torch.nn.LayerNorm(conformer_conf["output_size"]),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(conformer_conf["dropout_rate"]),              
        )  
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """

        # aduio downsampling while video has subsampling in frontend 
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device)
        org_feats,org_masks = self.input_layer(feats,masks)
        org_feats_lengths = org_masks.squeeze(1).sum(1)
        
        #fix length
        if not org_feats_lengths.equal(video_lengths):
            org_feats_lengths = org_feats_lengths.min(video_lengths)
            video_lengths = org_feats_lengths.clone()
            feats = feats[:,:max(org_feats_lengths)]
            video = video[:,:max(video_lengths)]

        #fusion 1 + layer 1
        x_concat = torch.cat((org_feats, video), dim=-1)
        amid_feat1= self.fusion(x_concat)
        outfeats1,outfeats_lengths1,_ = self.alayer1(amid_feat1,org_feats_lengths)
        outvideo1,outvideo_lengths1,_ = self.vlayer1(video,org_feats_lengths)

        #fusion 2+layer 2
        x_concat = torch.cat((outfeats1, outvideo1), dim=-1)
        amid_feat1= self.fusion(x_concat)
        outfeats2,outfeats_lengths2,_ = self.alayer2(amid_feat1,outfeats_lengths1)

        #skip connection + layer av
        hidden_feat,hidden_feat_lengths,_ = self.alayer3(org_feats+outfeats2,outfeats_lengths2)

        return hidden_feat,hidden_feat_lengths,_
      
class AVConformerEncoder3(AbsEncoder):
    def __init__(
        self,
        conformer_conf:dict,
        feat_dim:int,
        alayer_num1:int,
        alayer_num2:int,
        vlayer_num1:int,
   
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.input_layer = Conv2dSubsampling(
                feat_dim,
                conformer_conf["output_size"],
                conformer_conf["dropout_rate"],
                None)
        
      
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf) 
        self.alayer2 = ConformerEncoder(num_blocks=alayer_num2,**conformer_conf)
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**conformer_conf)  
        self.fusion = torch.nn.Sequential(
                        torch.nn.Linear(conformer_conf["output_size"]*2, conformer_conf["output_size"]),
                        torch.nn.LayerNorm(conformer_conf["output_size"]),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(conformer_conf["dropout_rate"]),              
        )  
  
          
         
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """

        # aduio feat and video subsampling
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device)
        org_feats,org_masks = self.input_layer(feats,masks)
        org_feats_lengths = org_masks.squeeze(1).sum(1)
        masks = (~make_pad_mask(video_lengths)[:, None, :]).to(video.device)

       
        #fusion 1+layer 1
        x_concat = torch.cat((org_feats, video), dim=-1)
        amid_feat1= self.fusion(x_concat)
        outfeats1,outfeats_lengths1,_ = self.alayer1(org_feats,org_feats_lengths)
        outvideo1,outvideo_lengths1,_ = self.vlayer1(video,org_feats_lengths)

        #fusion 2+layer 2
        x_concat = torch.cat((outfeats1, outvideo1), dim=-1)
        amid_feat2= self.fusion(x_concat)
        hidden_feat,hidden_feat_lengths,_ = self.alayer2(amid_feat2,outfeats_lengths1)

        return hidden_feat,hidden_feat_lengths,_

class AVConformerEncoder4(AbsEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        conformer_conf:dict,
        alayer_num1:int,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.fusion = DimConvert(in_channels=512*2,out_channels=256) 
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf) #incluee embedding layer
        self.subsampling = Conv2dSubsampling(
                            512,
                            512,
                            conformer_conf["dropout_rate"],
                            None)
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats_lengths.device)
        # log_file = open("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/expfarlipfar/comformer_avsr_far_av/test","a")
        # print("feattype:{feats.device} {feats_lengths.device}",file=log_file)
        # print("videotype:{video.device} {video_lengths.device}",file=log_file)
        # print("masktype:{mask.device}",file=log_file)
        feats, masks = self.subsampling(feats, masks)
        feats_lengths = masks.squeeze(1).sum(1)
        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        feats = feats[:,:max(feats_lengths)]
        video = video[:,:max(feats_lengths)]
        #fusion 1+layer 1
        x_concat = torch.cat((feats, video), dim=-1) #B,T,1024
        amid_feat1= self.fusion(x_concat) #B,T,256
        hidden_feat,hidden_feat_lengths,_ = self.alayer1(amid_feat1,feats_lengths)
        return hidden_feat,hidden_feat_lengths,_

class DimConvert(torch.nn.Module): #(B,T,D)->(B,T,D)
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
    ):  
        super().__init__()
        settings = {
            "in_channels":in_channels,
            "out_channels":out_channels,
            "kernel_size":1,
            "stride":1,
            "bias":False,
            "padding":0,
        }
        self.convert = torch.nn.Sequential(
            torch.nn.Conv1d(**settings),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.PReLU(out_channels),
            torch.nn.Dropout(0.1)
        )
    def forward(self,tensor):
        return self.convert(tensor.transpose(1,2)).transpose(1,2)

"""
AVConformerEncoder5 is similar to AVConformerEncoder2 and is used for wav preencode 25ps +video 25 ps ,which have the same fps;for feat preencode 100ps + video 25ps you can use  AVConformerEncoder6
"""
class AVConformerEncoder5(AbsEncoder):
    def __init__(
        self,
        conformer_conf:dict,
        alayer_num1:int=3,
        alayer_num2:int=3,
        alayer_num3:int=3,
        vlayer_num1:int=3,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
       
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf)
        self.alayer2 = ConformerEncoder(num_blocks=alayer_num2,**conformer_conf)
        self.alayer3 = ConformerEncoder(num_blocks=alayer_num3,**conformer_conf)
        video_conformer_conf = conformer_conf.copy()
        video_conformer_conf["input_size"] = 512
        video_conformer_conf["input_layer"] = "linear"
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**video_conformer_conf) 
        self.fusion1 = DimConvert(in_channels=512*2,out_channels=256)
        self.fusion2 = DimConvert(in_channels=256*2,out_channels=256)
        self.audioturner = DimConvert(in_channels=256*2,out_channels=256)
      
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, T, 256).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, T, 256). #video and audio both 25 ps
                video_lengths (torch.Tensor): Input length (#batch)
        """
        # both have nearly same dimensional
        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        feats = feats[:,:max(feats_lengths)]
        video = video[:,:max(feats_lengths)]
        # log_file = open("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/expwavlip/test1","a")
        # print("------in encoder -----")
        # print(f"feats.shape:{feats.shape},feats_lengths:{feats_lengths}",file=log_file)
        # print(f"video.shape:{video.shape},video_lengths:{video_lengths}",file=log_file)
        #fusion 1 + layer 1
        x_concat = torch.cat((feats, video), dim=-1)
        amid_feat = self.fusion1(x_concat)
        
        # print(f"famid_feat.shape:{amid_feat.shape},outfeats_lengths:{feats_lengths}",file=log_file)
        outfeats1,outfeats_lengths1,_ = self.alayer1(amid_feat,feats_lengths)
        outvideo1,outvideo_lengths1,_ = self.vlayer1(video,video_lengths)

        #fusion 2+layer 2
        x_concat1 = torch.cat((outfeats1, outvideo1), dim=-1)
        amid_feat1 = self.fusion2(x_concat1)
        outfeats2,outfeats_lengths2,_ = self.alayer2(amid_feat1,outfeats_lengths1)

        #skip connection + layer av
        res = self.audioturner(feats)
        hidden_feat,hidden_feat_lengths,_ = self.alayer3(outfeats2+res,outfeats_lengths2)

        return hidden_feat,hidden_feat_lengths,_

"""
AVConformerEncoder6 is similar to AVConformerEncoder2, and is used for feat preencode 100ps  video 25ps ,it will downsampling feat preencode first
"""
class AVConformerEncoder6(AbsEncoder):
    def __init__(
        self,
        conformer_conf:dict,
        alayer_num1:int=3,
        alayer_num2:int=3,
        alayer_num3:int=3,
        vlayer_num1:int=3,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"] #512
        self.conformer_conf =conformer_conf
       
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf)
        self.alayer2 = ConformerEncoder(num_blocks=alayer_num2,**conformer_conf)
        self.alayer3 = ConformerEncoder(num_blocks=alayer_num3,**conformer_conf)
        video_conformer_conf = conformer_conf.copy()
        video_conformer_conf["input_size"] = 512
        video_conformer_conf["input_layer"] = "linear"
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**video_conformer_conf) 
        self.fusion1 = DimConvert(in_channels=512*2,out_channels=256)
        self.fusion2 = DimConvert(in_channels=256*2,out_channels=256)
        self.audioturner = DimConvert(in_channels=256*2,out_channels=256)
        self.subsampling = Conv2dSubsampling(
                            512,
                            512,
                            conformer_conf["dropout_rate"],
                            None)
      
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, T, 256).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, T, 256). #video and audio both 25 ps
                video_lengths (torch.Tensor): Input length (#batch)
        """
        #downsampling audio 100fps -> 25fps
        # log_file = open("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/exp_gssfar_lipfar/featfinal_avsr_far_av/test","a")
        # print(f"feats:{feats.device},feats_lengths:{feats_lengths.device}",file=log_file)
        # print(f"video.shape:{video.device},video_lengths:{video_lengths.device}",file=log_file)
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(video_lengths.device)
        feats, masks = self.subsampling(feats, masks)
        feats_lengths = masks.squeeze(1).sum(1)
        # both have nearly same dimensional
        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        feats = feats[:,:max(feats_lengths)]
        video = video[:,:max(feats_lengths)]
       
        #fusion 1 + layer 1
        x_concat = torch.cat((feats, video), dim=-1)
        amid_feat = self.fusion1(x_concat)
        
        # print(f"famid_feat.shape:{amid_feat.shape},outfeats_lengths:{feats_lengths}",file=log_file)
        outfeats1,outfeats_lengths1,_ = self.alayer1(amid_feat,feats_lengths)
        outvideo1,outvideo_lengths1,_ = self.vlayer1(video,video_lengths)

        #fusion 2+layer 2
        x_concat1 = torch.cat((outfeats1, outvideo1), dim=-1)
        amid_feat1 = self.fusion2(x_concat1)
        outfeats2,outfeats_lengths2,_ = self.alayer2(amid_feat1,outfeats_lengths1)

        #skip connection + layer av
        res = self.audioturner(feats)
        hidden_feat,hidden_feat_lengths,_ = self.alayer3(outfeats2+res,outfeats_lengths2)

        return hidden_feat,hidden_feat_lengths,_

"""
AVConformerEncoder7 is similar to AVConformerEncoder4, and is used for wav preencode 25ps +video 25 ps ,which don't need  downsampling 
"""
class AVConformerEncoder7(AbsEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        conformer_conf:dict,
        alayer_num1:int,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.fusion = DimConvert(in_channels=512*2,out_channels=256) 
        self.alayer1 = ConformerEncoder(num_blocks=alayer_num1,**conformer_conf) #incluee embedding layer
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). 
                video_lengths (torch.Tensor): Input length (#batch)
        """

        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        feats = feats[:,:max(feats_lengths)]
        video = video[:,:max(feats_lengths)]
        #fusion 1+layer 1
        x_concat = torch.cat((feats, video), dim=-1) #B,T,1024
        amid_feat1= self.fusion(x_concat) #B,T,256
        hidden_feat,hidden_feat_lengths,_ = self.alayer1(amid_feat1,feats_lengths)
        return hidden_feat,hidden_feat_lengths,_

# conformer_conf = {"output_size": 256  ,  # dimension of attention
#     "attention_heads": 4,
#     "linear_units": 2048 , # the number of units of position-wise feed forward
#     "dropout_rate": 0.1,
#     "positional_dropout_rate": 0.1,
#     "attention_dropout_rate": 0.0,
#     "input_layer": "conv2d" ,# encoder architecture type
#     "normalize_before": True,
#     "pos_enc_layer_type": "rel_pos",
#     "selfattention_layer_type": "rel_selfattn",
#     "activation_type": "swish",
#     "macaron_style": True,
#     "use_cnn_module": True,
#     "cnn_module_kernel": 15}

# encoder = AVConformerEncoder5(conformer_conf)
# feats = torch.rand(16,90,512)
# video = torch.rand(16,90,512)
# feats_l = torch.randint(40,91,(16,))
# feats_l[0] = 90
# video_l = torch.randint(90,91,(16,))
# video_l[0] = 90
# hidden_feat,hidden_feat_lengths,_ = encoder(feats,feats_l,video,video_l)
# print(hidden_feat.shape,hidden_feat_lengths.shape)


class TCNFusionEncoder(AbsEncoder): # [b,T,512]->[b,T,256*3] 
    def __init__(
        self,
        single_input_dim=512,
        fuse_type="tcn",
        hidden_channels=[256 *3, 256 * 3, 256 * 3],
        kernels_size= [3, 5, 7],
        dropout=0.2,
        act_type="prelu",
        downsample_type="norm"
    ):  
        super().__init__()
        fuse_setting = {
            'in_channels': [single_input_dim, single_input_dim],
            "hidden_channels":hidden_channels,
            "kernels_size":kernels_size,
            "dropout":dropout,
            "act_type":act_type,
            "downsample_type":downsample_type,
            }
        self.subsampling = Conv2dSubsampling(
                            single_input_dim,
                            single_input_dim,
                            dropout,
                            None)
        
        self.fusion = AudioVisualFuse(fuse_type=fuse_type, fuse_setting=fuse_setting)
        self.dimturner = DimConvert(in_channels=256*3,out_channels=256)

    def output_size(self) -> int:
        return 256

    def forward(self,feats,feats_lengths,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """
        #100 fps ->25fps downsampling and alignment
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats_lengths.device)
        # log_file = open("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/expfarlipfar/tcn_avsr_far_av/test","a")
        # print("feattype:{feats.device} {feats_lengths.device}",file=log_file)
        # print("videotype:{video.device} {video_lengths.device}",file=log_file)
        # print("masktype:{mask.device}",file=log_file)
        feats, masks = self.subsampling(feats, masks)
        feats_lengths = masks.squeeze(1).sum(1)

        feats_lengths = feats_lengths.min(video_lengths)
        video_lengths = feats_lengths.clone()
        feats = feats[:,:max(feats_lengths)]
        video = video[:,:max(feats_lengths)]
        #fusion TCN
        feats = feats.transpose(1,2)#[B,T,D]->[B,D,T]
        video = video.transpose(1,2)
        hidden_feat, hidden_feat_lengths = self.fusion([feats], [video], feats_lengths) #[B,D,T]->[B,D,T]
        hidden_feat = self.dimturner(hidden_feat.transpose(1,2))
    
        return hidden_feat,hidden_feat_lengths,None

# print("hhh")
# fusionnet = TCNFusionEncoder(**dict(  single_input_dim=512,
#         fuse_type="tcn",
#         hidden_channels=[256 *3, 256 * 3, 256 * 3],
#         kernels_size= [3, 5, 7],
#         dropout=0.2,
#         act_type="prelu",
#         downsample_type="norm"))
# feats = torch.rand(16,90,512)
# video = torch.rand(16,90,512)
# feats_l = torch.randint(40,91,(16,))
# feats_l[0] = 90
# video_l = torch.randint(90,91,(16,))
# video_l[0] = 90
# print("hhh")
# hidden_feat,hidden_feat_lengths = fusionnet(feats,feats_l,video,video_l)
# print(hidden_feat.shape,hidden_feat_lengths)


class VConformerEncoder(AbsEncoder): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        conformer_conf:dict,
        vlayer_num1:int,
    ):  

        super().__init__()
        conformer_conf["input_layer"] = None
        conformer_conf["input_size"] =  conformer_conf["output_size"]
        self.conformer_conf =conformer_conf
        self.vlayer1 = ConformerEncoder(num_blocks=vlayer_num1,**conformer_conf) 
    
    def output_size(self) -> int:
        return self.conformer_conf["output_size"]

    def forward(self,video,video_lengths
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
                feats (torch.Tensor): Input tensor (#batch, L, input_size).
                feats_lengths (torch.Tensor): Input length (#batch).
                video (torch.Tensor): Input tensor (#batch, L, input_size). #video has subsampling
                video_lengths (torch.Tensor): Input length (#batch)
        """
        hidden_feat,hidden_feat_lengths,_ = self.vlayer1(video,video_lengths)
        return hidden_feat,hidden_feat_lengths,_
