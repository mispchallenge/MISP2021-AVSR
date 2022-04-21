#!/usr/bin/env bash
# Copyright 2021 USTC (Authors: Dilision)
# Apache 2.0

# extract region of interest (roi) in the video, store as npz file, item name is "data"

set -e
# configs for 'chain'
python_path=
stage=3
nj=15
gpu_nj=4
# End configuration section.

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: $0 <set_class> <lip_posisition> <wav_position>"
  echo " $0 train middle far"
  exit 1;
fi

echo "$0 $@"  # Print the command line for logging
org_path=dump/raw
setclass=$1
lip_position="middle"
[[ $2 == *"far"* ]] && lip_position="far"
wav_position=$3
# pt_dir=/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/${setclass}_${lip_position}_video_lip_segment/pt #middle far dili
pt_dir=/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/train_${lip_position}_video_lip_segment/pt #middle far dili

[[ $setclass == *"train"* ]] && pt_dir=/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/train_${lip_position}_video_lip_segment/pt 
[[ $setclass == *"dev"* ]] && pt_dir=/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/dev_${lip_position}_video_lip_segment/pt 
[[ $setclass == *"eval"* ]] && pt_dir=/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/eval_${lip_position}_video_lip_segment/pt 

[[ $lip_position == *"middle"* ]] && tag_set=dump/raw/${setclass}_${wav_position}_lip 
[[ $lip_position == *"far"* ]] && tag_set=dump/raw/${setclass}_${wav_position}_lipfar 


[[ $setclass == *"eval"* ]] && org_path=dump/raw

###########################################################################
# prepare text,wav.scp,roi.scp
###########################################################################
if [ $stage -le 1 ]; then
    mkdir -p $tag_set
    cat ${org_path}/${setclass}_${wav_position}/wav.scp > $tag_set/wav.scp
    cat ${org_path}/${setclass}_${wav_position}/text > $tag_set/text


  if [[ $setclass == *"eval"* ]];then
    python local/lip_roi.py --pt_dir $pt_dir --roiscpdir $tag_set --filename roi1.scp
    python local/lip_roi.py --pt_dir /raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/addition_${lip_position}_video_lip_segment/pt \
                                                                --roiscpdir $tag_set --filename roi2.scp
    cat $tag_set/roi1.scp $tag_set/roi2.scp > $tag_set/roi.scp
    rm $tag_set/roi1.scp && rm $tag_set/roi2.scp
  else
    python local/lip_roi.py --pt_dir $pt_dir --roiscpdir $tag_set 
  fi
fi

###########################################################################
# copy and fix roi.scp wav.scp
###########################################################################
if [ $stage -le 2 ]; then
  cat  $tag_set/roi.scp | sort -k 1 | uniq >$tag_set/roi.scp.tmp
  mv $tag_set/roi.scp.tmp  $tag_set/roi.scp
  cat  $tag_set/roi.scp  | wc -l
  cat $tag_set/roi.scp | awk '{print $1}' > $tag_set/temp_uid.tmp
  utils/filter_scp.pl  $tag_set/temp_uid.tmp $tag_set/wav.scp > $tag_set/wav.scp.tmp
  mv $tag_set/wav.scp.tmp  $tag_set/wav.scp
  cat $tag_set/wav.scp | awk '{print $1}' > $tag_set/temp_uid.tmp
  utils/filter_scp.pl  $tag_set/temp_uid.tmp $tag_set/roi.scp > $tag_set/roi.scp.tmp
  mv $tag_set/roi.scp.tmp  $tag_set/roi.scp
  utils/filter_scp.pl  $tag_set/temp_uid.tmp $tag_set/text > $tag_set/text.tmp
  mv $tag_set/text.tmp  $tag_set/text
  rm $tag_set/*.tmp
  cat  $tag_set/roi.scp  | wc -l
  echo "raw" > $tag_set/feats_type
fi

###########################################################################
# del short wav in eval datadir
###########################################################################
if [ $stage -le 3 ]; then
  if [[ $setclass == *"eval"* ]];then 
    python local/del_empty_wav.py --dirpath $tag_set
    cat ${org_path}/${setclass}_${wav_position}/utt2spk > ${tag_set}/utt2spk
    utils/fix_data_dir.sh ${tag_set}
  fi
fi
