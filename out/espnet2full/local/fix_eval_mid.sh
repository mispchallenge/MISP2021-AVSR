#!/usr/bin/env bash
# Copyright 2018 USTC (Authors: Hang Chen)
# Apache 2.0

# transform misp data to kaldi format

set -e -o pipefail
echo "$0 $@"
nj=1
. ./cmd.sh || exit 1
. ./path.sh || exit 1
. ./utils/parse_options.sh || exit 1



enhancement_root=/yrfs2/cv1/hangchen2/data/misp2021_eval
data_root=/yrfs2/cv1/hangchen2/data/misp2021_eval
data_type=eval
store_dir=data/eval_mid

# mp4
python local/prepare_far_data.py --only_mp4 true -nj $nj $enhancement_root/audio $data_root/video $data_root/transcription $data_type $store_dir
cat $store_dir/temp/mp4.scp | sort -k 1 | uniq > $store_dir/mp4.scp
echo "eval mid misp fix success"
exit 0

