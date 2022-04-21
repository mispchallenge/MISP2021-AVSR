#!/bin/bash
pn=gpu_project
pd=gpu_project
world=1
numg=2
numc=20
mem=20
task=gpu
gputype=TeslaV100-PCIE-12GB
stamp=`date +%Y%m%d_%H%M%S`
logfile=./submit_log/submit_$stamp.log
queue=
node=

. ./utils/parse_options.sh || exit 1;
if [ $# != 1 ]; then
  echo "Usage: $0 <task_type> <run.sh>"
  echo "e.g.: $0 gpu run.sh"
  echo "Options: "
  echo "  --pn <project_name>                       # project_name."
  echo "  --pd <project describe>                   # project describe."
  echo "  --word <word_num>                         # number of machine."
  echo "  --numg <numg>                             # number of gpus."
  echo "  --numc <numc>                             # number of cpus."
  echo "  --mem <mem>                               # memory limit."
  echo "  --task <task_type>                        # task type."
  echo "  --queue <reserved_queue>                  # reserved queue."
  echo "  --gputype TeslaM40 # how to run jobs.     # gputype"
  echo "  --logfile ./submit.log                    # logfile"
  exit 1;
fi
script=$1

submit_opt="-n $pn -d $pd -a hangchen2 -i reg.deeplearning.cn/dlaas/cv_dist:0.1 -e $script -l $logfile"
if [ $task == gpu ]; then
  submit_opt="$submit_opt --useGpu -g $numg -t PtJob -k $gputype"
  if [ $queue ]; then
    submit_opt="$submit_opt -r $queue"
  fi
  if [ $world != 1 ]; then
    submit_opt="$submit_opt --useDist -w $world"
  fi
  if [ $node ]; then
    submit_opt="$submit_opt -s $node"
  fi
else
  submit_opt="$submit_opt -t CommonJob -c $numc -m $mem"
fi

dlp submit $submit_opt

# reg.deeplearning.cn/dlaas/cv_dist:0.1
# -x dlp2-4-078
# bash submit.sh --pn 0-1 --pd asr_far --numg 4 --gputype TeslaV100-PCIE-12GB --logfile submit_log/0_1.log "run_misp.sh 18"