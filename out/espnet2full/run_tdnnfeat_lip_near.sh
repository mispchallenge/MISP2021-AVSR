#!/usr/bin/env bash
source ./bashrc
# export cmd=./shared/run.pl
set -eou pipefail

train_set=train_near_lip 
valid_set=dev_near_lip
test_sets=eval_near_lip


asr_config=conf/tuning/train_avsr_TCN.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

use_lm=true
use_word_lm=false


./avsr.sh                             \
    --stage 14 \
    --stop_stage 14 \
    --avsr_exp expnearlip/tcn_avsr_near_av   \
    --expdir expnearlip \
    --lang zh                              \
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
    

    
