#!/usr/bin/env python
# -- coding: UTF-8 
import os
import glob
import codecs
import argparse
import sys
from nara_wpe.utils import stft, istft
from tqdm import tqdm
import scipy.io.wavfile as wf
import numpy as np
from test_gss import *

def check(storepath):
    if os.path.exists(storepath):
        return 1
    return 0
def wfread(f):
    fs, data = wf.read(f)
    if data.dtype == np.int16:
        data = np.float32(data) / 32768
    return data, fs
def wfwrite(z, fs, store_path):
    tmpwav = np.int16(z * 32768)
    wf.write(store_path, fs, tmpwav)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('run_gss')
    parser.add_argument('wav_scp', type=str, default='./local/tmp/wpe.scp', help='list file of wav, format is scp')
    parser.add_argument('data_root', type=str, default='gss', help='input misp data root/wpe root')
    parser.add_argument('output_root', type=str, default='wpe', help='output gss data root')
    parser.add_argument('atype', type=str, default='middle', help='audio type')
    args = parser.parse_args()
    atype = args.atype
    adic = {'middle': ['middle', 'Middle'], 'far': ['far', 'Far']}
    atype = adic[atype]
    data_root, output_root = args.data_root, args.output_root
    stft_window, stft_shift = 512, 256
    gss = GSS(20, 1)
    bf = Beamformer('mvdrSouden_ban', 'mask_mul')
    wav_scp = args.wav_scp
    with codecs.open(wav_scp, 'r') as handle:
        lines_content = handle.readlines()
    wav_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
    cnt = 0
    for wav_idx in tqdm(range(len(wav_lines)), leave=True, desc='0'):
        file_list = wav_lines[wav_idx].split(' ')
        name, wav_list = file_list[0], file_list[1:]
        texts = glob.glob(wav_list[0].replace('wav', 'TextGrid').replace(atype[1]+'_0', 'Near_*').replace(atype[0]+'_audio', 'near_transcription').replace('_wpe', '').replace('feature', 'released_data'))
        print(wav_list[0].replace('wav', 'TextGrid').replace(atype[1]+'_0', 'Near_*').replace(atype[0]+'_audio', 'near_transcription').replace('_wpe', '').replace('feature', 'released_data'))
        texts.sort()
        print(texts)
        print('=================================================')
        signal_list = []
        time_activity = []
        speaker_check = texts[0].split('.')[-2][-3:]
        checkpath = wav_list[0].replace(data_root, output_root).replace('_0.', '_{}.'.format(speaker_check))
        if not check(checkpath):
            cnt += 1
            print(cnt, checkpath)
            for wav in wav_list:
                data, fs = wfread(wav)
                signal_list.append(data)
            try:
                obstft = np.stack(signal_list, axis=0)
            except:
                mlen = len(signal_list[0])
                for i in range(1, len(signal_list)):
                    mlen = min(mlen, len(signal_list[i]))
                for i in range(len(signal_list)):
                    signal_list[i] = signal_list[i][:mlen]
                obstft = np.stack(signal_list, axis=0)
            obstft = stft(obstft, stft_window, stft_shift)
            wavlen = len(data)
            speaker_list = []
            for text in texts:
                print(text)
                speaker_list.append(text.split('.')[-2][-3:])
                time_activity.append(get_time_activity(text, wavlen, fs))
            print('-----------------------------------')
            time_activity.append([True] * wavlen)
            frequency_activity = get_frequency_activity(time_activity, stft_window, stft_shift)
            masks = gss(obstft, frequency_activity)
            masks_bak = masks
            wavlist = []
            print('************')
            for i in range(masks.shape[0]-1):
                print(masks.shape)
                target_mask = masks[i]
                print(target_mask.shape)
                distortion_mask = np.sum(
                    np.delete(masks, i, axis=0),
                    axis=0,
                )
                print('******************')
                print(target_mask.shape, distortion_mask.shape)
                Xhat = bf(obstft, target_mask=target_mask, distortion_mask=distortion_mask)
                print('Xhat shape:{}'.format(Xhat.shape))
                xhat = istft(Xhat, stft_window, stft_shift)
                newpath = wav_list[0].replace(data_root, output_root).replace('_0.', '_{}.'.format(speaker_list[i]))
                print('newpath ++++++++++++++++++++++++++++ ' + newpath)
                wfwrite(xhat, fs, newpath)
                masks = masks_bak




