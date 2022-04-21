#!/usr/bin/env bash
# Copyright 2018 USTC (Authors: Zhaoxu Nian, Hang Chen)
# Apache 2.0

# use nara-wpe and GSS/Beamformit to enhance multichannel data

set -eo pipefail
# configs
stage=0
nj=10
python_path=
beamformit_path=
enhancement=
type=

. ./path.sh || exit 1
. ./utils/parse_options.sh || exit 1
if [ $# != 2 ]; then
 echo "Usage: $0 <corpus-data-dir> <enhancement-dir>"
 echo " $0 /path/misp2021 /path/gss_output"
 exit 1;
fi

data_root=$1
out_root=$2

echo "start speech enhancement"
# wpe
if [ $stage -le 0 ]; then
  echo "start wpe"
  ${python_path}python local/find_wav.py -nj $nj $data_root ${data_root}_wpe/log wpe ${type}
  for n in `seq $nj`; do
    cat <<-EOF > $out_root/log/wpe.$n.sh
    ${python_path}python local/run_wpe.py $out_root/log/wpe.$n.scp $data_root $out_root
EOF
  done
  chmod a+x ${data_root}_wpe/log/wpe.*.sh
  $train_cmd JOB=1:$nj ${data_root}_wpe/log/wpe.JOB.log ${data_root}_wpe/log/wpe.JOB.sh
  echo "finish wpe"
fi

# GSS or beamformit

if [ $stage -le 1 ]; then
  if [[ $enhancement = gss ]]; then
    echo "start gss"
    mkdir -p $out_root/log
    mkdir -p $out_root/wav
    ~/anaconda3/bin/python local/find_wav.py ${data_root}_wpe $out_root/log gss ${type} -nj $nj
    for n in `seq $nj`; do
      cat <<-EOF > $out_root/log/gss.$n.sh
      ~/anaconda3/bin/python local/run_gss.py $out_root/log/gss.$n.scp ${data_root}_wpe $out_root ${type}
      EOF
    done
    chmod a+x $out_root/log/gss.*.sh
    $train_cmd JOB=1:$nj $out_root/log/gss.JOB.log $out_root/log/gss.JOB.sh
    echo "finish gss"
  else
    echo "start beamformit"
    ${python_path}python local/find_wav.py $out_root $out_root/log beamformit ${type}
    sed -i 's|${out_root}/||g' $out_root/log/beamformit.1.scp
    ${python_path}python local/run_beamformit.py $beamformit_path/BeamformIt conf/all_conf.cfg / $out_root/log/beamformit.1.scp
    echo "end beamformit"
  fi
fi