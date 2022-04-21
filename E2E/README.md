## End-to-End models

In signal processing, weighted prediction error（[WPE](https://github.com/fgnt/nara_wpe)）and Guided source separation([GSS](https://github.com/fgnt/pb_chime5)) are applied to six-channel speech for dereverberation and enhancement. Three end-to-end models all consist of a hybrid [CTC/Attention-based](https://arxiv.org/abs/1609.06773) acoustic model and a transformer-based language model denoted as E2E1, E2E2, and E2E3 respectively. Both E2E1 and E2E2 use FBANK as network input and [SpecAug](https://github.com/DemisEom/SpecAugment) is applied during the training stage. The acoustic model encoder of E2E1 consists of 1D/2Dconv+Resnet18 based audio/video embeddng extractors(Table 2) and a three-layer [MS-TCN](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks) sequence model, while E2E2 use a six-layer conformer-based sequence model. E2E3 uses a 3Dconv+Resnet18+Avgpooling(Table 2) embedding extractor to get embedding from the raw waveform and [WavAug](https://github.com/facebookresearch/WavAugment) is adopted during training. As for the sequence model, E2E3 has a more flexible conformer-based encoder backbone with skip-connection and multiple fusion(Fig 1).

</br>
</br>
<div align="center"> Tabel 1: The performance of E2E models on MISP2021-AVSR</div>

<div align="center"><img src="https://github.com/mispchallenge/MISP2021-AVSR/blob/main/images/results.png" width="640"/></div>
</br>
</br>
<div align="center">Tabel 2: Settings of embedding extractors</div>

<div align="center"><img src="https://github.com/mispchallenge/MISP2021-AVSR/blob/main/images/extractors.png" alt="extractors" width="640" /></div>
</br>
</br>
<div align="center"><img src="https://github.com/mispchallenge/MISP2021-AVSR/blob/main/images/e2e3.png" width="640"></div>		

<div align="center">Figure 1: The sequence model of E2E3</div>

## Quick start

1. Put scripts in espnet2 folder in this repoitorie in their respective folders in your espnet tool.
2. Get enhancement channel and lip roi by runing  stage 2 and 18 in [NN-HMM/run_misp.sh](https://github.com/mispchallenge/MISP2021-AVSR/blob/main/NN-HMM/run_misp.sh)
3. Start to run:

```
run_gss_feattcn_lipfar.sh --stage 1
run_gss_featcom_lipfar.sh --stage 1
run_gss_wavfinal_lipfar.sh --stage 1
```

## Requirments

- [espnet](https://github.com/espnet/espnet)

- [nara_wpe](https://github.com/fgnt/nara_wpe)

- [pb_chime5](https://github.com/fgnt/pb_chime5)

  
