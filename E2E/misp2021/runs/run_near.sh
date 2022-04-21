#!/usr/bin/env bash
source ./bashrc
# export cmd=./shared/run.pl
set -eou pipefail

train_set=train_near
valid_set=dev_near
test_sets=eval_near

asr_config=conf/tuning/train_asr_conformer.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

use_lm=false # use pretrained lm
use_word_lm=false


./asr.sh                                   \
    --stage 12 \
    --expdir expnear \
    --lang zh                              \
    --speed_perturb_factors "1.0 0.8 1.33333"    \
    --nj 8\
    --audio_format wav                     \
    --nlsyms_txt data/nlsyms.txt           \
    --ngpu 1                               \
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
    --lm_exp exp/lm_train_lm_zh_char \
    --lm_train_text "data/${train_set}/text" "$@"
    
