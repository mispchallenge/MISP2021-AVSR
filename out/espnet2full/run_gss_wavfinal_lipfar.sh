#!/usr/bin/env bash
source ./bashrc
# export cmd=./shared/run.pl
set -eou pipefail

train_set=gss_train_far_lipfar
valid_set=gss_dev_far_lipfar
test_sets=gss_sum_eval_far_lipfar

asr_config=conf/tuning/train_avsr_final.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

use_lm=true
use_word_lm=false


./avsr_gsswav_lip.sh                             \
    --stage 15 \
    --stop_stage 16 \
    --avsr_exp exp_gssfar_lipfar/new_avsr_far_av   \
    --use_wavaug_preprocessor true      \
    --feats_normalize none \
    --expdir exp_gssfar_lipfar             \
    --lang zh                              \
    --nj 8                                 \
    --audio_format wav                     \
    --nlsyms_txt data/nlsyms.txt           \
    --ngpu 1                              \
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
    
