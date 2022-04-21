#!/usr/bin/env bash
source ./bashrc
# export cmd=./shared/run.pl
set -eou pipefail

train_set=train_near 
valid_set=dev_near
test_sets=eval_near

asr_config=conf/tuning/train_avsr_conformer.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

use_lm=true
use_word_lm=false


./avsr_near.sh                                   \
    --stage 14 \
    --stop_stage 14 \
    --avsr_exp expnear/avsr_near_av_model1   \
    --expdir expnear \
    --lang zh                              \
    --speed_perturb_factors "1.0 0.8 1.33333"    \
    --nj 8\
    --audio_format wav                     \
    --nlsyms_txt data/nlsyms.txt           \
    --ngpu 3                               \
    --token_type char                      \
    --feats_type raw                       \
    --use_lm ${use_lm}                     \
    --asr_config "${asr_config}"           \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}"             \
    --use_word_lm ${use_word_lm}           \
    --train_set "${train_set}"             \
    --valid_set "${valid_set}"             \
    --test_sets "${test_sets}"             \
    --lm_train_text "data/${train_set}/text" "$@"
    
