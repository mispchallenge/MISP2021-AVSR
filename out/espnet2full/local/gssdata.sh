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
stop_stage=3
log "$0 $*"
. utils/parse_options.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;


data_dir=/yrfs2/cv1/hangchen2/data/misp2021 # the path to store video and transprit
enhancement_dir=/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr # the path to store gss audio 


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare misp2021 datadir in kaldi format only include far audio and far lip"
  for pos in far ;do
    for setclass in train dev eval; do
      if [[ ! -f data/gss_${setclass}_${pos}/.done ]]; then
        if [ $setclass == "eval" ];then
            data_dir=${data_dir}_eval
        fi
        local/prepare_gss_data.sh  ${data_dir} $enhancement_dir/${setclass}_${pos}_audio_gss \
          $setclass  data/gss_${setclass}_${pos} || exit 1;
          touch data/gss_${setclass}_${pos}/.done
      fi
      utils/fix_data_dir.sh data/gss_${setclass}_${pos}
    done
  done
fi

