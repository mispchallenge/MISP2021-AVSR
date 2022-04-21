#!/usr/bin/env bash
source ./bashrc
# export cmd=./shared/run.pl
set -eou pipefail

train_set=gss_train_far_lipfar
valid_set=gss_dev_far_lipfar
test_sets=gss_eval_far_lipfar

asr_config=conf/tuning/train_avsr_com.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

use_lm=true
use_word_lm=false


./avsr_gsswav_lip.sh                             \
    --stage 14 \
    --stop_stage 14 \
    --avsr_exp exp_gssfar_lipfar/conformer_avsr_far_av   \
    --expdir exp_gssfar_lipfar \
    --lang zh                              \
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
    --lm_train_text "data/${train_set}/text" "$@"
    
