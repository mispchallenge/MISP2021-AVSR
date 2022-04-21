#!/usr/bin/env bash
#
# This recipe is for misp2021 task2, it recognise
# a given evaluation utterance given ground truth
# diarization information
#
# Copyright  2021  USTC (Author: Hang Chen, Zhaoxu Nian)
# Apache 2.0
#

# Begin configuration section.
nj=15
nnet_stage=0
oovSymbol="<UNK>"
boost_sil=1.0 # note from Dan: I expect 1.0 might be better (equivalent to not
              # having the option)... should test.
numLeavesTri1=7000
numGaussTri1=56000
numLeavesMLLT=10000
numGaussMLLT=80000
numLeavesSAT=12000
numGaussSAT=96000
# End configuration section

. ./utils/parse_options.sh

. ./cmd.sh
. ./path.sh
source ./bashrc

set -e

stage=$1

# path settings

enhancement=gss   # gss or wpe
beamformit_path=/yrfs2/cv1/hangchen2/code/se_misp/BeamformIt-master
python_path=/home/cv1/hangchen2/anaconda3/envs/py37/bin/
misp2021_corpus=released_data/misp2021_avsr/
#enhancement_dir=${misp2021_corpus}_${enhancement}
dict_dir=data/local/dict
data_roi=data/local/roi
device=0,1,2,3

##########################################################################
# wpe+beamformit or wpe+gss
##########################################################################

#if [ $stage -le -2 ]; then
#  for type in Far Middle ; do
#    local/enhancement.sh --stage 1 --python_path $python_path --beamformit_path $beamformit_path --enhancement $enhancement --type $type \
#      /yrfs2/cv1/hangchen2/data/misp2021/audio/dev/far_correct /yrfs2/cv1/hangchen2/data/misp2021/audio/dev/far_correct_wpe_beamformit  || exit 1;
#    done
#fi

# use nara-wpe and beamformit or gss to enhance multichannel misp data
# notice: if you use beamformit, make sure you install nara-wpe and beamformit and you need to compile BeamformIt with the kaldi script install_beamformit.sh
# If you use gss, make sure you have installed pb_chime5
if [ $stage -le -1 ]; then
  for type in far middle ; do
    for x in dev train addition ; do
      if [[ ! -f ${enhancement_dir}/audio/$x.done ]]; then
        local/enhancement.sh --stage 0 --python_path $python_path --beamformit_path $beamformit_path --enhancement $enhancement --type $type \
          $misp2021_corpus/${x}_${type}_audio $misp2021_corpus/${x}_${type}_audio_${enhancement}  || exit 1;
        touch $misp2021_corpus/${x}_${type}_audio_${enhancement}/$x.done
      fi
    done
  done
fi

###########################################################################
# prepare dict
###########################################################################

# download DaCiDian raw resources, convert to Kaldi lexicon format
if [ $stage -le 0 ]; then
  local/prepare_dict.sh --python_path $python_path $dict_dir || exit 1;
fi

###########################################################################
# prepare data
###########################################################################

if [ $stage -le 1 ]; then
  # awk '{print $1}' $dict_dir/lexicon.txt | sort | uniq | awk '{print $1,99}'> $dict_dir/word_seg_vocab.txt
  for x in addition eval dev train; do
    ${python_path}python local/prepare_data.py -nj 1 feature/misp2021_avsr/${x}_far_audio_${enhancement}/wav/'*.wav' \
      released_data/misp2021_avsr/${x}_near_transcription/TextGrid/'*.TextGrid' data/${x}_far_audio_${enhancement}|| exit 1;
    # spk2utt
    utils/utt2spk_to_spk2utt.pl data/${x}_far_audio_${enhancement}/utt2spk | sort -k 1 | uniq > data/${x}_far_audio_${enhancement}/spk2utt
    echo "word segmentation"
    ${python_path}python local/word_segmentation.py $dict_dir/word_seg_vocab.txt data/${x}_far_audio_${enhancement}/text_sentence > data/${x}_far_audio_${enhancement}/text
  done
fi

###########################################################################
# prepare language module
###########################################################################

# L
if [ $stage -le 2 ]; then
  utils/prepare_lang.sh --position-dependent-phones false \
    $dict_dir "$oovSymbol" data/local/lang data/lang  || exit 1;
fi

# arpa LM
if [ $stage -le 3 ]; then
  local/train_lms_srilm.sh --train-text data/train_far_audio/text --dev-text data/dev_far_audio/text --oov-symbol "$oovSymbol" data/ data/srilm
fi

# prepare lang_test
if [ $stage -le 4 ]; then
  utils/format_lm.sh data/lang data/srilm/lm.gz data/local/dict/lexicon.txt data/lang_test
fi

mkdir -p exp

###########################################################################
# feature extraction
###########################################################################
if [ $stage -le 5 ]; then
  mfccdir=mfcc
  # eval_far dev_far train_far eval_near dev_near train_near
  for x in addition_far eval_far dev_far train_far; do
    utils/fix_data_dir.sh data/${x}_audio_${enhancement}
    steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $nj data/${x}_audio_${enhancement} feature/misp2021_avsr/${x}_mfcc_pitch_kaldi_${enhancement}/log \
      feature/misp2021_avsr/${x}_mfcc_pitch_kaldi_${enhancement}/ark
    steps/compute_cmvn_stats.sh data/${x}_audio_${enhancement} feature/misp2021_avsr/${x}_mfcc_pitch_kaldi_${enhancement}/log \
      feature/misp2021_avsr/${x}_mfcc_pitch_kaldi_${enhancement}/ark
    utils/fix_data_dir.sh data/${x}_audio_${enhancement}
  done
  # for x in train ; do
  #   cp data/${x}_far_audio/utt2spk data/${x}_near_audio
  #   utils/fix_data_dir.sh data/${x}_near_audio
  # done
  # subset the training data for fast startup
  # for x in 50 100; do
  #   utils/subset_data_dir.sh data/train_far ${x}000 data/train_far_${x}k
  # done
fi

###########################################################################
# mono phone train
###########################################################################
if [ $stage -le 6 ]; then
  for x in middle far; do
    steps/train_mono.sh --boost-silence $boost_sil --nj $nj --cmd "$train_cmd" data/train_${x}_audio_${enhancement} data/lang exp/mono_${x}_audio_${enhancement} || exit 1;
    utils/mkgraph.sh data/lang_test exp/mono_${x}_audio_${enhancement} exp/mono_${x}_audio_${enhancement}/graph || exit 1;
    # steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj exp/mono/graph data/dev_far exp/mono/decode_dev_far || exit 1;
  done
fi

if [ $stage -le 7 ]; then
  steps/train_mono.sh --boost-silence $boost_sil --nj $nj --cmd "$train_cmd" data/train_near_audio data/lang exp/mono_near_audio || exit 1;
  utils/mkgraph.sh data/lang_test exp/mono_near_audio exp/mono_near_audio/graph || exit 1;
  # steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj $nj exp/mono/graph data/dev_far exp/mono/decode_dev_far || exit 1;
fi

###########################################################################
# tr1 delta+delta-delta
###########################################################################
if [ $stage -le 8 ]; then
  for x in middle far; do
    # alignment
    steps/align_si.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_${x}_audio_${enhancement} data/lang \
      exp/mono_${x}_audio_${enhancement} exp/mono_${x}_audio_${enhancement}_ali || exit 1;
    # training
    steps/train_deltas.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 data/train_${x}_audio_${enhancement} data/lang \
      exp/mono_${x}_audio_${enhancement}_ali exp/tri1_${x}_audio_${enhancement} || exit 1;
    # make graph
    utils/mkgraph.sh data/lang_test exp/tri1_${x}_audio_${enhancement} exp/tri1_${x}_audio_${enhancement}/graph || exit 1;
  done
#   # decoding
#   if [ ! -f exp/tri1/tri1.decode.done ]; then
#     steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} exp/tri1/graph data/dev_far exp/tri1/decode_dev_far || exit 1;
#     touch exp/tri1/tri1.decode.done
#   fi
fi

if [ $stage -le 9 ]; then
  # alignment
  steps/align_si.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_near_audio data/lang \
    exp/mono_near_audio exp/mono_near_audio_ali || exit 1;
  # training
  steps/train_deltas.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 data/train_near_audio data/lang \
    exp/mono_near_audio_ali exp/tri1_near_audio || exit 1;
  # make graph
  utils/mkgraph.sh data/lang_test exp/tri1_near_audio exp/tri1_near_audio/graph || exit 1;
#   # decoding
#   if [ ! -f exp/tri1/tri1.decode.done ]; then
#     steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} exp/tri1/graph data/dev_far exp/tri1/decode_dev_far || exit 1;
#     touch exp/tri1/tri1.decode.done
#   fi
fi

###########################################################################
# tri2 all lda+mllt
###########################################################################
if [ $stage -le 10 ]; then
  for x in middle far; do
    # alignment
    steps/align_si.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_${x}_audio_${enhancement} data/lang \
      exp/tri1_${x}_audio_${enhancement} exp/tri1_${x}_audio_${enhancement}_ali || exit 1;
    # training
    steps/train_lda_mllt.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesMLLT $numGaussMLLT data/train_${x}_audio_${enhancement} data/lang \
      exp/tri1_${x}_audio_${enhancement}_ali exp/tri2_${x}_audio_${enhancement} || exit 1;
    # make graph
    utils/mkgraph.sh data/lang_test exp/tri2_${x}_audio_${enhancement} exp/tri2_${x}_audio_${enhancement}/graph || exit 1;
  done
#   # decoding
#   if [ ! -f exp/tri3/tri3.decode.done ]; then
#     steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} exp/tri3/graph data/dev_far exp/tri3/decode_dev_far || exit 1;
#     touch exp/tri3/tri3.decode.done
#   fi
fi

if [ $stage -le 11 ]; then
  # alignment
  steps/align_si.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_near_audio data/lang \
    exp/tri1_near_audio exp/tri1_near_audio_ali || exit 1;
  # training
  steps/train_lda_mllt.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesMLLT $numGaussMLLT data/train_near_audio data/lang \
    exp/tri1_near_audio_ali exp/tri2_near_audio || exit 1;
  # make graph
  utils/mkgraph.sh data/lang_test exp/tri2_near_audio exp/tri2_near_audio/graph || exit 1;
#   # decoding
#   if [ ! -f exp/tri3/tri3.decode.done ]; then
#     steps/decode.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} exp/tri3/graph data/dev_far exp/tri3/decode_dev_far || exit 1;
#     touch exp/tri3/tri3.decode.done
#   fi
fi


###########################################################################
# tri3 all sat
###########################################################################
if [ $stage -le 12 ]; then
   for x in middle far; do
     # alignment
     steps/align_fmllr.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_${x}_audio_${enhancement} data/lang \
       exp/tri2_${x}_audio_${enhancement} exp/tri2_${x}_audio_${enhancement}_ali || exit 1;
     # training
     steps/train_sat.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesSAT $numGaussSAT data/train_${x}_audio_${enhancement} data/lang \
       exp/tri2_${x}_audio_${enhancement}_ali exp/tri3_${x}_audio_${enhancement} || exit 1;
     # make graph
     utils/mkgraph.sh data/lang_test exp/tri3_${x}_audio_${enhancement} exp/tri3_${x}_audio_${enhancement}/graph || exit 1;

  # alignment
     steps/align_fmllr.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_${x}_audio data/lang \
       exp/tri3_${x}_audio exp/tri3_${x}_audio_ali || exit 1;
     $train_cmd JOB=1:$nj exp/tri3_${x}_audio_ali/log/align2pdf_id.JOB.log \
       ali-to-pdf exp/tri3_${x}_audio_ali/final.mdl "ark:gunzip -c exp/tri3_${x}_audio_ali/ali.JOB.gz|" \
       "ark,scp:exp/tri3_${x}_audio_ali/pdf.JOB.ark,exp/tri3_${x}_audio_ali/pdf.JOB.scp"
     for n in $(seq $nj); do
       cat exp/tri3_${x}_audio_ali/pdf.$n.scp || exit 1;
     done > data/train_${x}_audio/pdf_from_tri3_${x}_audio_ali.scp || exit 1
     ${python_path}python tool/pdf_ark2pt.py -nj $nj data/train_${x}_audio/pdf_from_tri3_${x}_audio_ali.scp \
       0.02 0.01 feature/misp2021_avsr/train_${x}_tri3_ali/pt
   done
  # alignment 
  for x in addition eval dev train ; do
    for y in middle far ; do
      ali_dir=exp/tri3_${y}_audio_${enhancement}_ali_${x}_${y}_audio_${enhancement}
      steps/align_fmllr.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/${x}_${y}_audio_${enhancement} data/lang \
        exp/tri3_${y}_audio_${enhancement} $ali_dir || exit 1;
      # output_dir=feature/misp2021_avsr/${x}_${y}_tri3_ali/pt
      # mkdir -p ${output_dir}
      # num_pdf=$(hmm-info $ali_dir/final.mdl | awk '/pdfs/{print $4}')
      # echo $num_pdf > $output_dir/../num_pdf
      # labels_tr_pdf="ark:ali-to-pdf $ali_dir/final.mdl \"ark:gunzip -c $ali_dir/ali.*.gz |\" ark:- |"
      # analyze-counts --verbose=1 --binary=false --counts-dim=$num_pdf "$labels_tr_pdf" $output_dir/../ali_train_pdf.counts
      tool/format_ali.sh --python_path ${python_path} --nj ${nj} --cmd ${train_cmd} --frame_dur 0.02 --frame_shift 0.01 \
        exp/tri3_${y}_audio_${enhancement}_ali_${x}_${y}_audio_${enhancement} feature/misp2021_avsr/tri3_${y}_audio_${enhancement}_ali_${x}_${y}_audio_${enhancement}/pt || exit 1;
    done
  done
  #   # decoding
  #   if [ ! -f exp/tri4/tri4.decode.done ]; then
  #     steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} exp/tri4/graph data/dev_far exp/tri4/decode_dev_far || exit 1;
  #     touch exp/tri4/tri4.decode.done
  #   fi
fi

if [ $stage -le 13 ]; then
  # # alignment
   steps/align_fmllr.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_near_audio data/lang \
     exp/tri2_near_audio exp/tri2_near_audio_ali || exit 1;
   # training
   steps/train_sat.sh --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesSAT $numGaussSAT data/train_near_audio data/lang \
     exp/tri2_near_audio_ali exp/tri3_near_audio|| exit 1;
   # make graph
   utils/mkgraph.sh data/lang_test exp/tri3_near_audio exp/tri3_near_audio/graph || exit 1;
  # alignment
   steps/align_fmllr.sh --boost-silence $boost_sil --cmd "$train_cmd" --nj $nj data/train_near_audio data/lang \
     exp/tri3_near_audio exp/tri3_near_audio_ali || exit 1;
   $train_cmd JOB=1:$nj exp/tri3_near_audio_ali/log/align2pdf_id.JOB.log \
      ali-to-pdf exp/tri3_near_audio_ali/final.mdl "ark:gunzip -c exp/tri3_near_audio_ali/ali.JOB.gz|" \
      "ark,scp:exp/tri3_near_audio_ali/pdf.JOB.ark,exp/tri3_near_audio_ali/pdf.JOB.scp"
   for n in $(seq $nj); do
     cat exp/tri3_near_audio_ali/pdf.$n.scp || exit 1;
   done > data/train_near_audio/pdf_from_tri3_near_audio_ali.scp || exit 1
   ${python_path}python tool/pdf_ark2pt.py -nj $nj data/train_near_audio/pdf_from_tri3_near_audio_ali.scp \
     0.02 0.01 feature/misp2021_avsr/train_near_tri3_ali/pt
  echo "analysis alignment"
  ali_dir=exp/tri3_near_audio_ali
  output_dir=feature/misp2021_avsr/train_near_tri3_ali/pt
  num_pdf=$(hmm-info $ali_dir/final.mdl | awk '/pdfs/{print $4}')
  echo $num_pdf > $output_dir/../num_pdf
  labels_tr_pdf="ark:ali-to-pdf $ali_dir/final.mdl \"ark:gunzip -c $ali_dir/ali.*.gz |\" ark:- |"
  analyze-counts --verbose=1 --binary=false --counts-dim=$num_pdf "$labels_tr_pdf" $output_dir/../ali_train_pdf.counts
#   # decoding
#   if [ ! -f exp/tri4/tri4.decode.done ]; then
#     steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.conf --nj ${dev_nj} exp/tri4/graph data/dev_far exp/tri4/decode_dev_far || exit 1;
#     touch exp/tri4/tri4.decode.done
#   fi
fi

if [ $stage -le 14 ]; then
  for x in addition eval dev train ; do
    for y in middle far ; do
      data_dir=data/${x}_${y}_audio_${enhancement}
      store_dir=feature/misp2021_avsr/${x}_${y}_audio_${enhancement}_segment/pt
      echo "============================================================"
      echo "segment $data_dir, store in $store_dir"
      ${python_path}python tool/segment_wav_to_pt.py -nj $nj $data_dir $store_dir
      cat $store_dir/segment.log
      ${python_path}python local/index_file2json.py
    done
  done
fi

#if [ $stage -le 15 ]; then
#  for x in dev ; do
#    for y in far ; do
#      data_dir=data/${x}_${y}_video
#      # store_dir=/yrfs2/cv1/hangchen2/code/misp2021_avsr/feature/misp2021_avsr/${x}_${y}_video_segment/pt
#      store_dir=feature/misp2021_avsr/${x}_${y}_video_segment/pt
#      echo "============================================================"
#      echo "segment $data_dir, store in $store_dir"
#      ${python_path}python tool/segment_mp4_to_pt.py -nj 10 $data_dir $store_dir || exit 1
#      cat $store_dir/segment.log
#    done
#  done
#fi

if [ $stage -le 16 ]; then
  CUDA_VISIBLE_DEVICES=${divice} ${python_path}python local/feature_cmvn.py 0
fi

if [ $stage -le 17 ]; then
  num_targets=$(tree-info exp/tri3_near_audio/tree |grep num-pdfs|awk '{print $2}')
  echo $num_targets
fi

if [ $stage -le 18 ]; then
  for x in addition; do
    for y in far middle; do
      for roi_type in head lip; do
        if [[ $y = far ]]; then
          need_speaker=true
        else
          need_speaker=false
        fi
        local/prepare_roi.sh --python_path $python_path --local false --nj 2 --roi_type $roi_type --roi_size "96 96" --need_speaker $need_speaker \
          --roi_sum true data/${x}_${y}_video released_data/misp2021_avsr/${x}_${y}_detection_result feature/misp2021_avsr/${x}_${y}_video_${roi_type}_segment/pt
      done
    done
  done
fi

if [ $stage -le 19 ]; then
  for config in 0_1 1_1 1_2 1_3 1_4; do
    agi=${divice}
    gpu_num=`echo ${agi//,/} | wc -L`
    CUDA_VISIBLE_DEVICES=$agi \
    ${python_path}python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=12354 \
    local/run_gpu.py -c $config -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train dev eval \
    -ms ce acc -si 2 -ci 1 -co max -pd dev -um -1
  done
fi

#if [ $stage -le 19 ]; then
#  agi=${divice}
#  gpu_num=`echo ${agi//,/} | wc -L`
#  CUDA_VISIBLE_DEVICES=$agi \
#  ${python_path}python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=12345 \
#  local/run_gpu.py -c 0_2 -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train dev eval \
#  -ms ce acc -si 2 -ci 1 -co max -pd dev -um -1
#fi
#
#if [ $stage -le 21 ]; then
#  agi=${divice}
#  gpu_num=`echo ${agi//,/} | wc -L`
#  CUDA_VISIBLE_DEVICES=$agi \
#  ${python_path}python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=12345 \
#  local/run_gpu.py -c 1_1 -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train dev eval \
#  -ms ce acc -si 2 -ci 1 -co max -pd dev -um 20
#fi
#
#if [ $stage -le 22 ]; then
#  agi=${divice}
#  gpu_num=`echo ${agi//,/} | wc -L`
#  CUDA_VISIBLE_DEVICES=$agi \
#  ${python_path}python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=12345 \
#  local/run_gpu.py -c 1_2 -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train dev eval \
#  -ms ce acc -si 2 -ci 1 -co max -pd dev -um -1
#fi
#
#if [ $stage -le 23 ]; then
#  agi=${divice}
#  gpu_num=`echo ${agi//,/} | wc -L`
#  CUDA_VISIBLE_DEVICES=$agi \
#  ${python_path}python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=12345 \
#  local/run_gpu.py -c 1_3 -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train dev eval \
#  -ms ce acc -si 2 -ci 1 -co max -pd dev -um 23
#fi
#
#if [ $stage -le 24 ]; then
#  agi=${divice}
#  gpu_num=`echo ${agi//,/} | wc -L`
#  CUDA_VISIBLE_DEVICES=$agi \
#  ${python_path}python -m torch.distributed.launch --nproc_per_node=$gpu_num --master_port=12345 \
#  local/run_gpu.py -c 1_4 -m predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train dev eval \
#  -ms ce acc -si 2 -ci 1 -co max -pd dev -um 23
#fi

#if [ $stage -le 25 ]; then
#  for x in 1_5; do
#    for y in eval addition; do
#      local/decode_score_from_posteriors.sh --python_path $python_path --exp_root exp --predict_data ${y} --predict_item posteriori --used_model -1 \
#        --ali_count feature/misp2021_avsr/train_far_tri3_ali/ali_train_pdf.counts --nj 16 --cmd "$decode_cmd" --stage 0 \
#        ${x} exp/tri3_far_audio data/${y}_far_audio
#    done
#    # python tool/merge_json.py -o /result_cer.json result_cer.json result_cer.json
#    # python local/sorce_replume_result.py --stage 1 s s result_cer.json
#  done
#fi
#
#if [ $stage -le 26 ]; then
#  for x in 0_2; do
#    for y in addition; do
#      local/decode_score_from_posteriors.sh --python_path $python_path --exp_root exp --predict_data ${y} --predict_item posteriori --used_model -1 \
#        --ali_count feature/misp2021_avsr/train_near_tri3_ali/ali_train_pdf.counts --nj 16 --cmd "$decode_cmd" --stage 0 \
#        ${x} exp/tri3_near_audio data/${y}_near_audio
#    done
#  done
#fi
#
#if [ $stage -le 27 ]; then
#  for x in 0_3; do
#    for y in dev eval addition; do
#      local/decode_score_from_posteriors.sh --python_path $python_path --exp_root exp --predict_data ${y} --predict_item posteriori --used_model -1 \
#        --ali_count feature/misp2021_avsr/train_middle_tri3_ali/ali_train_pdf.counts --nj 16 --cmd "$decode_cmd" --stage 0 \
#        ${x} exp/tri3_middle_audio data/${y}_middle_audio
#    done
#  done
#fi

config=(1_5 0_2 0_3)
type=(far near middle)
if [ $stage -le 27 ]; then
  for((i=0;i<=2;i++)); do
    for y in dev eval addition; do
      local/decode_score_from_posteriors.sh --python_path $python_path --exp_root exp --predict_data ${y} --predict_item posteriori --used_model -1 \
        --ali_count feature/misp2021_avsr/train_${type[$i]}_tri3_ali/ali_train_pdf.counts --nj 16 --cmd "$decode_cmd" --stage 0 \
        ${config[$i]} exp/tri3_middle_audio data/${y}_${type[$i]}_audio
    done
  done
fi

if [ $stage -le 28 ]; then
  for x in 1_9 0_4; do
    for y in addition; do
      local/decode_score_from_posteriors.sh --python_path $python_path --exp_root exp --predict_data ${y} --predict_item posteriori --used_model -1 \
        --ali_count feature/misp2021_avsr/train_far_${enhancement}_tri3_ali/ali_train_pdf.counts --nj 16 --cmd "$decode_cmd" --stage 1 \
        ${x} exp/tri3_far_audio_${enhancement} data/${y}_far_audio_${enhancement}
    done
  done
fi
#
#if [ $stage -le 29 ]; then
#  for x in 0_4; do
#    for y in addition; do
#      local/decode_score_from_posteriors.sh --python_path $python_path --exp_root exp --predict_data ${y} --predict_item posteriori --used_model -1 \
#        --ali_count feature/misp2021_avsr/train_far_${enhancement}_tri3_ali/ali_train_pdf.counts --nj 16 --cmd "$decode_cmd" --stage 1 \
#        ${x} exp/tri3_far_audio_${enhancement} data/${y}_far_audio_${enhancement}
#    done
#  done
#fi

# if [ $stage -le 29 ]; then
#   ${python_path}python -m torch.distributed.launch --nproc_per_node=2 --master_port=12345 \
#     local/run_gpu.py -c 0_3 -m train predict -rs 123456 -be 0 -es 100 -sp 1 -pf 500 -ss train dev eval \
#     -ms ce acc -si 2 -ci 1 -co max -pd eval -um -1
# fi

#if [ $stage -le 30 ]; then
#  bash submit.sh --pn 2-1 --pd vsr_far --numg 4 --gputype TeslaV100-PCIE-12GB --logfile submit_log/2_1.log "run_gpu.sh 2_1 4"
#fi
#
#if [ $stage -le 31 ]; then
#  bash submit.sh --pn 2-2 --pd vsr_middle --numg 4 --gputype TeslaV100-PCIE-12GB --logfile submit_log/2_2.log "run_gpu.sh 2_2 4"
#fi
## --queue dlp3-cv1-pretrain-reserved --node dlp2-7-198
#if [ $stage -le 32 ]; then
#  bash submit.sh --pn 1-5 --pd avsr_far_middle --numg 4 --gputype TeslaV100-PCIE-12GB --logfile submit_log/1_5.log "run_gpu.sh 1_5 4"
#fi
#
#if [ $stage -le 33 ]; then
#  bash submit.sh --pn 1-6 --pd avsr_near_far --numg 4 --gputype TeslaV100-PCIE-12GB --logfile submit_log/1_6.log "run_gpu.sh 1_6 4"
#fi
#
#if [ $stage -le 34 ]; then
#  bash submit.sh --pn 0-3 --pd asr_middle --numg 4 --gputype TeslaV100-PCIE-12GB --logfile submit_log/0_3.log "run_gpu.sh 0_3 4"
#fi
#
#if [ $stage -le 35 ]; then
#  bash submit.sh --pn 1-7 --pd avsr_middle_far --numg 4 --gputype TeslaV100-PCIE-12GB --logfile submit_log/1_7.log "run_gpu.sh 1_7 4"
#fi
#
#if [ $stage -le 36 ]; then
#  bash submit.sh --pn 1-8 --pd avsr_middle_middle --numg 4 --gputype TeslaV100-PCIE-12GB --logfile submit_log/1_8.log "run_gpu.sh 1_8 4"
#fi
#
#if [ $stage -le 37 ]; then
#  bash submit.sh --pn 0-4 --pd asr_far_${enhancement} --numg 4 --gputype TeslaV100-PCIE-12GB --logfile submit_log/0_4.log "run_gpu.sh 0_4 4"
#fi
#
#if [ $stage -le 38 ]; then
#  bash submit.sh --pn 1-9 --pd avsr_far_${enhancement}_far --numg 4 --gputype TeslaV100-PCIE-12GB --logfile submit_log/1_9.log "run_gpu.sh 1_9 4"
#fi

# ###########################################################################
# # tri5 audio-only tdnnf with sp
# ###########################################################################
# if [ $stage -le 11 ]; then
#   # chain TDNN
#   local/chain/run_tdnn_1b.sh --nj ${nj} --stage $nnet_stage --train-set train_far --test-sets "dev_far" --gmm tri4 \
#     --nnet3-affix _train_far
# fi


# ###########################################################################
# # tri6 all audio-only tdnnf without sp
# ###########################################################################
# if [ $stage -le 12 ]; then
#   # chain TDNN without sp
#   local/chain/run_tdnn_1b_no_sp.sh --nj ${nj} --stage $nnet_stage --train-set train_far --test-sets "dev_far" --gmm tri4 \
#     --nnet3-affix _train_far
# fi


# ###########################################################################
# # tri7 all audio-visual tdnnf withoutsp
# ###########################################################################
# if [ $stage -le 13 ]; then
#   # extract visual ROI, store as npz (item: data); extract visual embedding; concatenate visual embedding and mfcc
#   for x in dev_far train_far ; do
#     local/extract_far_video_roi.sh --python_path $python_path --nj ${nj} data/${x} $data_roi/${x} data/${x}_hires || exit 1;
#   done
#   # chain audio-visual TDNN
#   local/chain/run_tdnn_1b_av.sh --nj ${nj} --stage $nnet_stage --train-set train_far --test-sets "dev_far" --gmm tri4 \
#     --nnet3-affix _train_far
# fi


# ###########################################################################
# # tri8 all audio-visual tdnnf withsp
# ###########################################################################
# if [ $stage -le 14 ]; then
#   # extract visual ROI, store as npz (item: data); extract visual embedding; concatenate visual embedding and mfcc
#   local/extract_far_video_roi_sp.sh --python_path $python_path --nj ${nj} data/train_far $data_roi data/train_far_sp_hires
#   # chain audio-visual TDNN
#   local/chain/run_tdnn_1b_sp_av.sh --nj ${nj} --stage $nnet_stage --train-set train_far --test-sets "dev_far" --gmm tri4 \
#     --nnet3-affix _train_far
# fi

# ###########################################################################
# # show result
# ###########################################################################
if [ $stage -le 155 ]; then
  # getting results (see RESULTS file)
  for x in exp/*/predict_*/result_*/decode; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null
fi