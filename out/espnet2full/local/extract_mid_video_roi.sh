#!/usr/bin/env bash
# Copyright 2021 USTC (Authors: Hang Chen)
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


if [ $# != 2 ]; then
  echo "Usage: $0 <data-set> <roi-json-dir>"
  echo " $0 data/train_far /path/roi"
  exit 1;
fi

echo "$0 $@"  # Print the command line for logging
save_path=dump/raw/org
setclass=$1
roi_json_dir=$2
audio_set=${setclass}_near
# [ $data_set =~ "train" ] && audio_set=${data_set}_sp
video_dir=${save_path}/${setclass}_mid_video
segment_video_dir=${video_dir}/data



###########################################################################
# prepare mp4.scp, segments, vid2spk, spk2vid, text
###########################################################################
if [ $stage -le 1 ]; then

  # #create mp4.scp in train_far_sp 
  # if  [ ! -f $data_set/mp4.scp ] ;then
  #   echo "mp4.scp doesn't exist in data/$data_set, please have a check in ./data.sh"
  #   exit 1
  # else
  #     if [ ! -f ${data_set}_sp/mp4.scp ]; then
  #         python local/train_sp_mp4.py
  #     fi
  # fi

  if [[ ! -f $video_dir/file.done ]]; then
    mkdir -p $video_dir
    # sed -e 's/.wav/.mp4/;s/audio/video/' $data_set/wav.scp > $video_dir/mp4.scp

    cat data/${setclass}_mid/mp4.scp > $video_dir/mp4.scp 
    cat data/$audio_set/segments > $video_dir/segments
    cat data/$audio_set/spk2utt > $video_dir/spk2vid
    cat data/$audio_set/utt2spk > $video_dir/vid2spk
    cat data/$audio_set/text > $video_dir/text
    sed 's/Near/Middle/' $video_dir/segments  > $video_dir/segments.tmp 
    mv  $video_dir/segments.tmp $video_dir/segments
    touch $video_dir/file.done 
    
  fi

fi


###########################################################################
# segment mp4 and crop roi, store as npz, item name is data
###########################################################################
if [ $stage -le 2 ]; then


  if [[ ! -f $video_dir/roi.done ]]; then
    mkdir -p $video_dir/log
    for n in `seq $nj`; do
      cat <<-EOF > $video_dir/log/roi.$n.sh
        python local/prepare_far_video_roi.py --ji $((n-1)) --nj $nj $video_dir $roi_json_dir $segment_video_dir
EOF
    done
    chmod a+x $video_dir/log/roi.*.sh
    $train_cmd JOB=1:$nj $video_dir/log/roi.JOB.log $video_dir/log/roi.JOB.sh || exit 1;
    rm -f $video_dir/log/roi.*.sh
    cat $video_dir/log/roi.*.scp | sort -k 1 | uniq > $video_dir/roi.scp
    rm -f $video_dir/log/roi.*.scp
    echo 'roi done'
    touch $video_dir/roi.done
  fi
fi


###########################################################################
# copy and fix roi.scp wav.scp
###########################################################################
if [ $stage -le 3 ]; then

  #add roi.sp
  if [ $setclass == "train" ];then
  echo "$tag_set"
    tag_set=train_near_sp
    cat  $video_dir/roi.scp | sort -k 1 | uniq > dump/raw/$tag_set/roi.scp
    cat dump/raw/$tag_set/roi.scp | awk '{print "sp0.8-"$1" "$2}' > dump/raw/$tag_set/roi.scp.1
    cat dump/raw/$tag_set/roi.scp | awk '{print "sp1.33333-"$1" "$2}' > dump/raw/$tag_set/roi.scp.2
    cat dump/raw/$tag_set/roi.* | sort -k 1 | uniq > dump/raw/$tag_set/roi.scp.sum
    mv  dump/raw/$tag_set/roi.scp.sum  dump/raw/$tag_set/roi.scp
    rm dump/raw/$tag_set/roi.scp.*
  else 
    tag_set=${setclass}_near
    cat  $video_dir/roi.scp | sort -k 1 | uniq > dump/raw/$tag_set/roi.scp
  fi
  cat  dump/raw/$tag_set/roi.scp  | wc -l
  #jiaoji roi.scp wav.scp
  cat dump/raw/$tag_set/roi.scp | awk '{print $1}' > dump/raw/$tag_set/temp_uid.tmp
  utils/filter_scp.pl  dump/raw/$tag_set/temp_uid.tmp dump/raw/$tag_set/wav.scp > dump/raw/$tag_set/wav.scp.tmp
  mv dump/raw/$tag_set/wav.scp.tmp  dump/raw/$tag_set/wav.scp
  cat dump/raw/$tag_set/wav.scp | awk '{print $1}' > dump/raw/$tag_set/temp_uid.tmp
  utils/filter_scp.pl  dump/raw/$tag_set/temp_uid.tmp dump/raw/$tag_set/roi.scp > dump/raw/$tag_set/roi.scp.tmp
  mv dump/raw/$tag_set/roi.scp.tmp  dump/raw/$tag_set/roi.scp
  rm dump/raw/$tag_set/*.tmp

  cat  dump/raw/$tag_set/roi.scp  | wc -l
  utils/fix_data_dir.sh dump/raw/$tag_set
fi