import io
import functools
# import soundfile as sf
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from IPython.display import display, Audio

from nara_wpe.utils import stft, istft

from pb_bss.distribution import CACGMMTrainer
from pb_bss.evaluation import InputMetrics, OutputMetrics
from dataclasses import dataclass
from beamforming_wrapper import beamform_mvdr_souden_from_masks
from pb_chime5.utils.numpy_utils import segment_axis_v2
from text_grid import *

def get_time_activity(file_path, wavlen, sr):
    time_activity = [False] * wavlen
    text = read_textgrid_from_file(file_path)
    for interval in text.tiers[1].intervals:
        if 'NOISE' not in interval.text:
            xmax = int(interval.xmax * sr)
            xmin = int(interval.xmin * sr)
            if xmax > wavlen:
                break
            for i in range(xmin, xmax):
                time_activity[i] = True
    print('num of true {}'.format(time_activity.count(True)))
    return time_activity

def get_frequency_activity(time_activity,stft_window_length,stft_shift,stft_fading=True,stft_pad=True,):
    time_activity = np.asarray(time_activity)

    if stft_fading:
        pad_width = np.array([(0, 0)] * time_activity.ndim)
        pad_width[-1, :] = stft_window_length - stft_shift  # Consider fading
        time_activity = np.pad(
            time_activity,
            pad_width,
            mode='constant'
        )

    return segment_axis_v2(
        time_activity,
        length=stft_window_length,
        shift=stft_shift,
        end='pad' if stft_pad else 'cut'
    ).any(axis=-1)



@dataclass
class Beamformer:
    type: str
    postfilter: str

    def __call__(self, Obs, target_mask, distortion_mask, debug=False):
        bf = self.type

        if bf == 'mvdrSouden_ban':
            from pb_chime5.speech_enhancement.beamforming_wrapper import (
                beamform_mvdr_souden_from_masks
            )
            X_hat = beamform_mvdr_souden_from_masks(
                Y=Obs,
                X_mask=target_mask,
                N_mask=distortion_mask,
                ban=True,
            )
        elif bf == 'ch0':
            X_hat = Obs[0]
        elif bf == 'sum':
            X_hat = np.sum(Obs, axis=0)
        else:
            raise NotImplementedError(bf)

        if self.postfilter is None:
            pass
        elif self.postfilter == 'mask_mul':
            X_hat = X_hat * target_mask
        else:
            raise NotImplementedError(self.postfilter)

        return X_hat
@dataclass
class GSS:
    iterations: int = 20
    iterations_post: int = 0

    verbose: bool = True

    # use_pinv: bool = False
    # stable: bool = True

    def __call__(self, Obs, acitivity_freq=None, debug=False):
        initialization = np.asarray(acitivity_freq, dtype=np.float64)
        initialization = np.where(initialization == 0, 1e-10, initialization)
        initialization = initialization / np.sum(initialization, keepdims=True,
                                                axis=0)
        initialization = np.repeat(initialization[None, ...], 257, axis=0)

        source_active_mask = np.asarray(acitivity_freq, dtype=np.bool)
        source_active_mask = np.repeat(source_active_mask[None, ...], 257, axis=0)
        print('*****************************')
        print(initialization.shape, source_active_mask.shape)
        print(Obs.shape)
        print(Obs.T.shape)
        print(Obs.T[0, ...].shape)
        print('****************************')
        cacGMM = CACGMMTrainer()

        if debug:
            learned = []
        all_affiliations = []
        F = Obs.shape[-1]
        T = Obs.T.shape[-2]
        print(Obs.shape)
        print(Obs.T.shape)
        for f in range(F):
            if self.verbose:
                if f % 50 == 0:
                    print(f'{f}/{F}')

            # T: Consider end of signal.
            # This should not be nessesary, but activity is for inear and not for
            # array.
            cur = cacGMM.fit(
                y=Obs.T[f, ...],
                initialization=initialization[f, ..., :T],
                iterations=self.iterations,
                # num_classes=3
                source_activity_mask=source_active_mask[f, ..., :T],
                # return_affiliation=True,
            )
            affiliation = cur.predict(
                # np.swapaxes(Obs.T[f, ...], -1, -2),
                Obs.T[f, ...],
                source_activity_mask=source_active_mask[f, ..., :T]
            )

            all_affiliations.append(affiliation)

        posterior = np.array(all_affiliations).transpose(1, 2, 0)

        return posterior

# if __name__ == '__main__':
#     signal = []
#     ob = '/yrfs2/cv1/hangchen2/data/MISP_121h_WPE_/R12/S201202/C02/R12_S201202_C02_I1_Middle_0.wav'
#     data, fs = sf.read(ob)
#     # signal.append(data[:1000])
#     signal.append(data)
#     ob = '/yrfs2/cv1/hangchen2/data/MISP_121h_WPE_/R12/S201202/C02/R12_S201202_C02_I1_Middle_1.wav'
#     data, fs = sf.read(ob)
#     # signal.append(data[:1000])
#     signal.append(data)
#     data = np.stack(signal, axis=0)
#     print(data.shape)
#     obstft = stft(data, 512, 256)
#     # obstft = np.reshape(obstft, (1, -1, 257))
#     print(obstft.shape)
#     # plot_stft(obstft[0].T)
#     gss=GSS(20, 0)
#     po = gss(obstft)
#     print(po.shape)
#     print('****************')
#     print(po)
#     masks = po
#     bf = Beamformer('mvdrSouden_ban', 'mask_mul')
#     target_speaker_index = [0, 1, 2]
#     masks_bak = masks
#     wavlist = []
#     for i in target_speaker_index:
#         print(masks.shape)
#         target_mask = masks[i]
#         print(target_mask.shape)
#         distortion_mask = np.sum(
#             np.delete(masks, i, axis=0),
#             axis=0,
#         )
#         print('******************')
#         print(target_mask.shape, distortion_mask.shape)
#         Xhat = bf(obstft, target_mask=target_mask, distortion_mask=distortion_mask)
#         print('Xhat shape:{}'.format(Xhat.shape))
#         xhat = istft(Xhat, 512, 128)
#         print('xhat shape:{}'.format(xhat.shape))
#         wavlist.append(xhat)
#         sf.write('tmp{}.wav'.format(i), xhat, fs)
#         masks = masks_bak

