#!/usr/bin/env bash
source ./bashrc
# export cmd=./shared/run.pl
set -eou pipefail

train_set=train_far_lip 
valid_set=dev_far_lip
test_sets=eval_far_lip

asr_config=conf/tuning/train_avsr_model4.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

use_lm=true
use_word_lm=false


./avsr_wav_lip.sh                             \
    --stage 14 \
    --stop_stage 14 \
    --avsr_exp expfarlip/comformer_avsr_far_av   \
    --expdir expfarlip \
    --lang zh                              \
    --nj 8\
    --audio_format wav                     \
    --nlsyms_txt data/nlsyms.txt           \
    --ngpu 4                               \
    --gpu_inference  true                 \
    --token_type char                      \
    --feats_type raw                       \
    --use_lm ${use_lm}                     \
    --asr_config "${asr_config}"           \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}"             \
    --lm_exp /yrfs2/cv1/hangchen2/espnet/misp2021/asr1/exp/lm_train_lm_zh_char \
    --use_word_lm ${use_word_lm}           \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --lm_train_text "data/${train_set}/text" "$@"
    
