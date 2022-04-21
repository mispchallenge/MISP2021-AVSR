from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
import os
import soundfile
import numpy as np 
import torch 
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.encoder.avconformer_encoder import AVConformerEncoder
from espnet2.asr.espnet_model import ESPnetAVSRModel
from espnet2.fileio.sound_scp import SoundScpReader
import argparse
def video_load(uid,value):
    # print(uid,value)
    if "sp0.8" in uid:
        output = np.load(value)["data"].repeat(5,axis=0)
    elif "sp1.3" in uid:
        output = np.load(value)["data"].repeat(3,axis=0)
    else:
        output = np.load(value)["data"].repeat(4,axis=0)

    return output
if  __name__ == '__main__':
    refpath = Path("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw/gss_sum_eval_far_lipfar/text")
    refdir = refpath.parent
    reftoken = refdir / "textchar"
    # dic = read_2column_text(refpath)
    # with DatadirWriter(str(refdir)) as writer:
    #     subwriter = writer[str(reftoken)]
    #     for key,vaule in dic.items():
    #         newtokes = (" ").join([char for char in dic[key]])
    #         subwriter[key] = newtokes
    
    import subprocess
    decode = Path("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/exp_gssfar_lipfar")
    hpys = decode.glob("*/decode*/gss*/token")
    for hpy in hpys :
        scheduler_order = f"./local/cer.sh --ref {str(reftoken)} --hyp {hpy}"
        return_info = subprocess.Popen(scheduler_order, shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        for next_line in return_info.stdout:
            return_line = next_line.decode("utf-8", "ignore")
            # print(return_line)
            if "WER" in return_line:
                print(return_line[5:-11])
    # import torch
    # tensor2 = torch.randint(1,2,(1,4)).to("cuda")
    # tensor1 = torch.randint(3,5,(1,4)).to("cuda:0")
    # print(tensor1,tensor2)
    # print(tensor2.max(tensor1))
    # path = "/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/data/gss_train_far/wav.scp"
    # choice_dic = {}
    # dic = read_2column_text(path)
    # for key,value in dic.items():
    #     _,spks,conf,*_= key.split("_") 
    #     spksnum = (len(spks)-1)/3
    #     if spksnum >=4 and conf in ["C04","C10"]:
    #         choice_dic[key] = dic[key]
    
    # from pathlib import Path 
    # import shutil
    # import os 
    # dir = Path("/yrfs2/cv1/hangchen2/data/misp2021")
    # subdirs = ["audio","transcription","video"]
    # targets = ["R08_S158159160_C01_I0*"]
    # to_path = "/yrfs2/cv1/hangchen2/k2/correct"
    # files = []
    # for subdir in subdirs:
    #     dirpath = dir / subdir
    #     for target in targets:
    #         filepaths = list(dirpath.rglob(target))
    #         files += filepaths
    # filepaths = [str(file) for file in files]
    # # for filepath in filepaths:
    # #     for detele in ["train copy","beamformit"]:
    # #         if detele  in filepath: 
    # #             filepaths.remove(filepath)
    # for filepath in filepaths:   
    #     for target in targets:
    #         if target[:-1] in filepath:
    #             shutil.copy(filepath,os.path.join(to_path,target[:-1],filepath.split("/")[-1]))
    # def from_shape_file_find_zero(path):
    #     dic = read_2column_text(path)
    #     keys = []
    #     for key,value in dic.items():
    #         if value == "0":
    #             keys.append(key)
    #     return keys


    # def readkeys(path):
    #     dic = read_2column_text(path)
    #     keys = []
    #     for key,value in dic.items():
    #         keys.append(key)
    #     return keys
    
    # def savekeys(keys,path):
    #     dic = read_2column_text(path)
    #     savekeys = list(set(dic.keys()) & set(keys))
    #     savekeys.sort()
    #     path = Path(path)
    #     parent = path.parent
    #     file = path.name
    #     with DatadirWriter(str(parent)) as writer:
    #         subwriter = writer[str(file)]
    #         for key in savekeys:
    #              subwriter[key] = dic[key]

    # def del_keys(keys,path):
    #     dic = read_2column_text(path)
    #     for key in keys:
    #         if key in dic:
    #             del dic[key]
    #     path = Path(path)
    #     parent = path.parent
    #     file = path.name
    #     with DatadirWriter(str(parent)) as writer:
    #         subwriter = writer[str(file)]
    #         for key,value in dic.items():
    #              subwriter[key] = value

    # subdir = Path("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw/dev_far_lipfar")
    # shape_file = "/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/expfarlipfar/nosp_avsr_stats_raw_zh_char/valid/speech_shape"
    # save_uids = readkeys(shape_file)
    # filelist = subdir.glob("*")
    # # import pdb;pdb.set_trace()
    # for file in filelist:
    #     if not ".back" in str(file) and not "feats_type" in str(file):
    #         savekeys(save_uids,file)

    # dirpath = Path("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/expfarlipfar/nosp_avsr_stats_raw_zh_char")
    # for set in ["train","valid"]:
    #     subdir = dirpath / set
    #     checkscp = subdir / "speech_shape"
    #     shape_file = subdir / "speech_shape" 
    #     del_uids = from_shape_file_find_zero(shape_file)
    #     filelist = subdir.glob("*")
    #     # print(filelist)
    #     # import pdb;pdb.set_trace()
    #     for file in filelist:
    #         if not ".npz" in str(file):
    #             del_keys(del_uids,file)

    # noise_path="dump/raw/org/train_far_sp/data/format.1/S062_R02_S062063_C07_I0_122424-122492.wav"
    # with soundfile.SoundFile(noise_path) as f:
    #     noise = f.read(dtype=np.float64, always_2d=True)
    #     print(noise.shape)


    # rootpath = Path("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw")
    # log_file = open("./check","a")
    # dirlist = rootpath.glob("eval*")

    # for dir in dirlist:
    #     print(dir,file=log_file)
    #     checkscp = dir / "wav.scp"
    #     reader = SoundScpReader(checkscp)
    #     for k in reader.keys():
    #         rate, array = reader[k]
    #         if array.shape[0] == 0:
    #             print(k,file=log_file)
            
        

    



    # roi = "/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw/train_far_sp/roi.scp"
    # wav = "/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw/train_far_sp/wav.scp"
    # roi_dic = read_2column_text(roi)
    # wav_dic = read_2column_text(wav)
    # ids = ["S000_R01_S000001_C07_I0_005072-005604","S000_R01_S000001_C07_I0_003864-004048","S000_R01_S000001_C07_I0_004176-004288"]
    # uids = [[id,"sp1.33333-"+id,"sp0.8-"+id] for id in ids]
    # # uids = ["S000_R01_S000001_C07_I0_005072-005604","sp1.33333-S000_R01_S000001_C07_I0_005072-005604","sp0.8-S000_R01_S000001_C07_I0_005072-005604"]
    # frontend = DefaultFrontend(win_length=400,hop_length=160)
    # input_size = frontend.output_size()
    # # import pdb;pdb.set_trace()
    # for uidset in uids:
    #     for uid in uidset:
    #         roitensor = video_load(uid,roi_dic[uid])
    #         wavnp = torch.tensor(soundfile.read(wav_dic[uid])[0],dtype=torch.float32).unsqueeze(0)
    #         input_feats, feats_lens = frontend(wavnp,torch.tensor([wavnp.shape[1]]))
    #         print(input_feats.shape)
    #         # print(roitensor.shape[0],input_feats.shape[1])
    #         # print(roitensor.shape[0]/input_feats.shape[1])
    # # array = np.array([1,2,3])
    # # print(np.prod(array))
    # import yaml
    # from espnet2.asr.frontend.default import DefaultFrontend
    # from espnet2.tasks.avsr import AVSRTask
    # config_file = Path("conf/tuning/train_avsr_conformer.yaml")
    # with config_file.open("r", encoding="utf-8") as f:
    #         args = yaml.safe_load(f)
    # args = argparse.Namespace(**args)
    # print(args.encoder_conf)
    # encoder = AVConformerEncoder(conformer_conf=args.encoder_conf,feat_dim=args.frontend_conf["n_mels"],**args.avlayer_num_conf).to("cuda")
    # video = torch.rand(16,100,256).to("cuda")
    # feats = torch.rand(16,100,80).to("cuda")
    # video_lengths = torch.full((16,),100).to("cuda")
    # feats_lengths = torch.full((16,),100).to("cuda")
    # encoder_out, encoder_out_lens, _ = encoder(feats,feats_lengths,video,video_lengths)

    # # print(encoder_out.shape,encoder_out_lens.shape)
    # # video_frontend = VideoFrontend(args.videofront_conf["output_size"]) #equal to  hidden layer dim
    # model = AVSRTask.build_model(args=args)
    # parser = AVSRTask.get_parser()
    # args = parser.parse_args(cmd)
    # dir = "/yrfs2/cv1/hangchen2/data/detection_results/train/middle"
    # dir = Path(dir)
    # filelist = dir.glob("*.json") 
    # for file in filelist:
    #     name = file.stem
    #     # newname = dir / (("_").join(name.split("_")[:-1])+".json")
    #     # file.rename(newname)
    #     # # print(newname)
    #     # # exit(1)
    #     # if "C0001" in name:
    #     #     newname = dir / (name.replace("C0001","C01")+".json")
    #     #     file.rename(newname)
    #     if "C0009" in name:
    #         newname = dir / (name.replace("C0009","C09")+".json")
    #         file.rename(newname)

    # x2 = torch.load('/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/train_far_video_head_segment/pt/S000_R01_S000001_C07_I0_001444-001712.pt')
    # # x = np.load("/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw/org/eval_mid_video/data/S239_R16_S239240241_C07_I2_000220-000292.npz")
    # import pdb;pdb.set_trace()
    # def find_zero(path):
    #     dic = read_2column_text(path)
    #     keys = []
    #     # import pdb;pdb.set_trace()
    #     for key,value in dic.items():
    #         if value == "0":
    #             keys.append(key)
    #     return keys
    # def del_keys(keys,path):
    #     print(path)
    #     dic = read_2column_text(path)
    #     for key in keys:
    #         if key in dic:
    #             del dic[key]
    #     path = Path(path)
    #     parent = path.parent
    #     file = path.name
    #     # import pdb;pdb.set_trace()
    #     with DatadirWriter(str(parent)) as writer:
    #         subwriter = writer[str(file)]
    #         for key,value in dic.items():
    #              subwriter[key] = value

    # shape_file = "/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/expfarlip/nosp_avsr_stats_raw_zh_char/train/speech_shape" 
    # keys = find_zero(shape_file)
    # totalkeys = []
    # for prefix in ["","sp0.8-","sp1.33333-"]:
    #     for key in keys:
    #         totalkeys.append(prefix+key)
    # dirs = ["/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/expfarlip/nosp_avsr_stats_raw_zh_char/train"]
   
   
    # dirs = ["/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/expnearlip/nosp_avsr_stats_raw_zh_char","/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/expfarlip/nosp_avsr_stats_raw_zh_char"]

    # for dir in dirs:
    #     dir  = Path(dir)
    #     for subdir in ["train","valid"]:
    #         tagpath = dir / subdir / "video_shape"
    #         tagdir = str(dir / subdir)
    #         dic = read_2column_text(tagpath)
    #         with DatadirWriter(tagdir) as writer:
    #             subwriter = writer["original_video_shape"]
    #         for key,value in dic.items():
    #             time = str(int(int(value.split(",")[0])/4))
    #             # import pdb;pdb.set_trace()
    #             orginal_value = (",").join([str(time)]+value.split(",")[1:])
    #             subwriter[key] = orginal_value

    

            



    # from espnet2.fileio.read_text import read_2column_text
    # import numpy as np 
    # import torch 
    # def video_load(uid,value):
    #     # print(uid,value)
    #     if ".npz" in value:
    #         if "sp0.8" in uid:
    #             output = np.load(value)["data"].repeat(5,axis=0)
    #         elif "sp1.3" in uid:
    #             output = np.load(value)["data"].repeat(3,axis=0)
    #         else:
    #             output = np.load(value)["data"].repeat(4,axis=0)
    #         return output.astype(np.float32)

            
    #     if ".pt" in value:
    #         if "sp0.8" in uid:
    #             output = torch.load(value).repeat(5,1,1,1)
    #         elif "sp1.3" in uid:
    #             output = torch.load(value).repeat(3,1,1,1)
    #         else:
    #             output = torch.load(value).repeat(4,1,1,1)
    #         return output

    # value = "/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/dev_middle_video_lip_segment/pt/S186_R10_S186187188_C09_I2_048116-048164.pt"
    # uid = "S186_R10_S186187188_C09_I2_048116-048164"
   

     
  
  


    