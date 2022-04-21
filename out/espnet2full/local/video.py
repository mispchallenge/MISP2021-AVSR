# -*- coding: utf-8 -*-
# file: to_video.py
# author: JinTian
# time: 16/03/2018 2:24 PM
# Copyright 2018 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions

import os
import cv2
from colorama import Fore, Back, Style
import numpy as np
import sys
import torch 

class VideoCombiner(object):
    def __init__(self, np_path,target_file,interpolation=False):
        self.path = np_path
        self.target_file = target_file
        self.interpolation = interpolation
        if interpolation:
            self.fps = 25*4
        else:
            self.fps = 25

    def ararryfile_load(self):
        if ".npz" in self.path:
            output = np.load(path)["data"].astype(np.uint8)
        if ".pt" in self.path:
            output = torch.load(self.path).numpy().astype(np.uint8)
        self.video_shape = [output.shape[1],output.shape[2]] 
        if self.interpolation:
            output = output.repeat(4,axis=0)
        self.video_shape = [output.shape[1],output.shape[2]] 
        return output #(T,W,H,3) or #(T,W,H)

    def combine(self):
            stack = self.ararryfile_load()
            size = (self.video_shape[1], self.video_shape[0])
            video_writer = cv2.VideoWriter(self.target_file, cv2.VideoWriter_fourcc(*'XVID'), self.fps, size)
            for i,ararry in enumerate(stack):       
                video_writer.write(ararry)             
            video_writer.release()
        
np_path = "/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/train_middle_video_lip_segment/pt/S000_R01_S000001_C07_I0_002872-003024.pt"
target_file = "./nointerpolation.avi"
combiner = VideoCombiner(np_path,target_file,interpolation=False)
combiner.combine()