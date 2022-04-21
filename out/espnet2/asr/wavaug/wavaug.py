import augment
import torch
import numpy as np


def return_pich_co():
    return np.random.randint(-300, +300)
def return_reverb_co():
    return np.random.randint(0, 100)
class WavAug():
    def __init__(self):
        effect_chain_past = augment.EffectChain()
        effect_chain_past.pitch("-q", return_pich_co).rate("-q", 16_000)
        effect_chain_past.reverb(50, 50, return_reverb_co).channels()
        effect_chain_past.time_dropout(max_seconds=40 / 1000)
        self.effect_chain_past = effect_chain_past
    
    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        src_info = {'channels':  x.shape[0],  # number of channels
                    'length': x.shape[0],   # length of the sequence
                    'precision': 32,       # precision (16, 32 bits)
                    'rate': 16000.0,       # sampling rate
                    'bits_per_sample': 32}  # size of the sample
        target_info = {'channels': 1,
                       'length': x.shape[0],
                       'precision': 32,
                       'rate': 16000.0,
                       'bits_per_sample': 32}
        y = self.effect_chain_past.apply(
            x, src_info=src_info, target_info=target_info)
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()
        y = y.numpy()
        return y #y.T ?

# #usage
# wavaguer = WavAug()
# wav = torch.rand(16000)
# newwav = wavaguer(wav)
# print(type(newwav))
# print(newwav.shape)