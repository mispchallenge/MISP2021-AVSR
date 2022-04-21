#!/usr/bin/env bash
# Copyright 2018 USTC (Authors: Dalision)
# Apache 2.0

# transform misp data to kaldi format

set -e -o pipefail
echo "$0 $@"
nj=1
stage=0
. ./cmd.sh || exit 1
. ./path.sh || exit 1
. ./utils/parse_options.sh || exit 1


if [ $# != 4 ]; then
  echo "Usage: $0 <original-corpus-data-dir> <enhancement-audio-dir> <data-set>  <store-dir>"
  echo " $0 /path/misp /path/misp_gss train data/gss_train_far"
  exit 1;
fi

data_root=$1
enhancement_wav=$2
data_type=$3
store_dir=$4

# wav.scp segments text_sentence utt2spk
# for example: python local/prepare_gss_data.py -nj $nj --without_mp4 True /misp2021_avsr/feature/misp2021_avsr/addition_far_audio_gss/wav /misp2021_avsr/released_data/misp2021_avsr/gss_middle_video /misp2021_avsr/TextGrid eval data/gss_addition_far
# you can change --without_mp4,enhancement_wav,video_path to combine different audio and video filed as you like
if [ ${stage} -le 1 ];then
echo "prepare wav.scp segments text_sentence utt2spk"
python local/prepare_gss_data.py -nj $nj $enhancement_wav/wav $data_root/video $data_root/transcription $data_type $store_dir
fi

#fix kaldi data dir
if [ ${stage} -le 2 ];then
cat $store_dir/temp/wav.scp | sort -k 1 | uniq > $store_dir/wav.scp
if [[ -f $store_dir/temp/mp4.scp ]];then
cat $store_dir/temp/mp4.scp | sort -k 1 | uniq > $store_dir/mp4.scp
fi
cat $store_dir/temp/segments | sort -k 1 | uniq > $store_dir/segments
cat $store_dir/temp/utt2spk | sort -k 1 | uniq > $store_dir/utt2spk
cat $store_dir/temp/text_sentence | sort -k 1 | uniq > $store_dir/text
rm -r $store_dir/temp
echo "prepare done"

# generate spk2utt and nlsyms
utils/utt2spk_to_spk2utt.pl $store_dir/utt2spk | sort -k 1 | uniq > $store_dir/spk2utt
touch data/nlsyms.txt
fi

echo "local/prepare_gss_data.sh succeeded"
exit 0