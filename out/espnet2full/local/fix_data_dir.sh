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
tag_set=
# End configuration section.
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


# if [ $# != 1 ]; then
#   echo "Usage: $0 tag_set"
#   echo " $0 tag_set"
#   exit 1;
# fi

# echo "$0 $@"  # Print the command line for logging


cat  $tag_set/roi.scp | sort -k 1 | uniq >$tag_set/roi.scp.tmp
mv $tag_set/roi.scp.tmp  $tag_set/roi.scp
cat  $tag_set/roi.scp  | wc -l
#jiaoji roi.scp wav.scp
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
utils/fix_data_dir.sh ${tag_set}
echo "raw" > $tag_set/feats_type
