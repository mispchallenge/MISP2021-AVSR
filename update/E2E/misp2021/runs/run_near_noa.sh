#!/usr/bin/env bash
source ./bashrc
# export cmd=./shared/run.pl
set -eou pipefail

train_set=train_near
valid_set=dev_near
test_sets=eval_near

asr_config=/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/conf/tuning/tmp_unuse/train_asr_conformer.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

use_lm=false # use pretrained lm
use_word_lm=false


./asr.sh                                   \
    --stage 12 \
    --stop_stage 13 \
    --expdir expnear \
    --asr_exp expnear/nosp_asr_near_a  \
    --lang zh                              \
    --nj 8\
    --audio_format wav                     \
    --nlsyms_txt data/nlsyms.txt           \
    --ngpu 0                               \
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
    
