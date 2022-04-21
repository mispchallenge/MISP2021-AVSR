#!/usr/bin/env bash

source ./bashrc
# export cmd=./shared/run.pl
set -eou pipefail


utt1=$(echo "gss_sum_eval_far_lipfar" | awk '{split($1, arr, "_"); print arr[1]"_"arr[2]}')
utt1=$(echo "gss_sum_eval_far_lipfar" | awk '{split($1, arr, "_"); print arr[3]}')
echo $utt1
# for pos in "far" "near";do
# data_dir1=data/gss_oldeval_far
# data_dir2=data/gsss_addition_far
# utils/combine_data.sh data/eval_${pos} $data_dir1 $data_dir2
# done

# local/prepare_gss_data.sh  data_dir enhancement_dir setclass  data/gss_|| exit 1;
# touch data/gss_addition_far/.done
# utils/fix_data_dir.sh data/gss_addition_far
# python dili.py 
# dir=/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw/train_far_lip 
# for x in roi.scp wav.scp text;do
#     sed '/sp0.8/d' ${dir}/$x > ${dir}/$x.tmp
#     sed '/sp1.3/d' ${dir}/$x.tmp > ${dir}/$x
#     rm ${dir}/$x.tmp
# done

# for seclass in train dev eval;do
#     for pos in far near;do
#         utils/fix_data_dir.sh data/${setclass}_${pos}
#     done
# done


# for setclass in eval ;do
#     for pos in near;do
#         utils/fix_data_dir.sh /yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw/${setclass}_${pos}_lip
#     done
# done


# < tmp.txt  awk  '{print $1}' >  tmp1.txt
# cat tmp1.txt tmp.txt > tmp2.txt
# local/score.sh "/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/expnear/asr_train_asr_conformer_raw_zh_char_sp"

# avsr_exp="/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/expnear/asr_train_asr_conformer_raw_zh_char_sp"
# scripts/utils/show_asr_result.sh "${avsr_exp}" > "${avsr_exp}"/RESULTS.md
# cat "${avsr_exp}"/RESULTS.md

