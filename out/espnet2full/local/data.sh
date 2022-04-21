#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=0
stop_stage=1
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;



# if [ ! -e "${MISP2021}" ]; then
#     log "Fill the value of 'MISP2021' of db.sh"
#     exit 1
# fi


enhancement_dir=/yrfs2/cv1/hangchen2/data/misp2021

###########################################################################
# wpe+beamformit
###########################################################################
# use nara-wpe and beamformit to enhance multichannel misp data
# if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
#   log "stage 0: Nara-wpe and Beamformit"
#   for x in dev train ; do
#     local/enhancement.sh $MISP2021/audio/$x ${enhancement_dir}/audio/$x  || exit 1;
#   done
# fi

# # download DaCiDian raw resources, convert to Kaldi lexicon format ps:only
# if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
#   log "Stage 0: prepare_dict"
#   if [[ ! -f $dict_dir/.done ]]; then
#     local/prepare_dict.sh --python_path $python_path $dict_dir || exit 1;
#     touch $dict_dir/.done
#   fi
# fi


# ###########################################################################
# # prepare data
# ###########################################################################
# if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
#   log "Stage 1: Prepare misp2021 datadir in kaldi format"
#   for pos in far mid near;do
#     for setclass in train dev eval ; do
#       if [[ ! -f data/${setclass}_${pos}/.done ]]; then
#         local/prepare_data.sh $enhancement_dir ${enhancement_dir} \
#           $setclass  data/${setclass}_${pos} || exit 1;
#            touch data/${setclass}_${pos}/.done
#       fi
#       utils/fix_data_dir.sh data/${setclass}_${pos}
#     done
#   done
# fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare misp2021 datadir in kaldi format"
  for pos in far near;do
    for setclass in eval ; do
      if [[ ! -f data/${setclass}_${pos}/.done ]]; then
        local/prepare_data.sh $enhancement_dir ${enhancement_dir} \
          $setclass  data/${setclass}_${pos} || exit 1;
           touch data/${setclass}_${pos}/.done
      fi
      utils/fix_data_dir.sh data/${setclass}_${pos}
    done
  done
fi


