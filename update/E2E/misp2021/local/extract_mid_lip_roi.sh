#!/usr/bin/env bash
# Copyright 2021 USTC (Authors: Dilision)
# Apache 2.0

# 1.created audio dir 2.create roi.scp for roilist just use lip roi which has been extract by hybrid system 2.fix audio and video dir make sure they have same uttids

set -e
python_path=
stage=3
nj=15
gpu_nj=4


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
[[ $setclass == *"eval"* ]] && org_path=dump/raw
setclass=$1
lip_position=$2 #far_lip or mid_lip
wav_position=$3
tag_set=dump/raw/${setclass}_${wav_position}_lipfar 
[[ $setclass == *"train"* ]] && pt_dir=#the path you store train lip crop as .pt files
[[ $setclass == *"dev"* ]] && pt_dir=#the path you store dev lip crop as .pt files
[[ $setclass == *"eval"* ]] && pt_dir=#the path you store eval lip crop as .pt files


###########################################################################
# prepare text,wav.scp,roi.scp
###########################################################################
if [ $stage -le 1 ]; then
    mkdir -p $tag_set
    cat ${org_path}/${setclass}_${wav_position}/wav.scp > $tag_set/wav.scp
    cat ${org_path}/${setclass}_${wav_position}/text > $tag_set/text
    python local/lip_roi.py --pt_dir $pt_dir --roiscpdir $tag_set 
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

 

