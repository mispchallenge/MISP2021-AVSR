from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.sound_scp import SoundScpReader
import argparse

if  __name__ == '__main__':

    def from_wavscp_find_zero(path):
        dirpath = path.parent
        log_file = dirpath / "check.tmp"
        if not log_file.exists():
            reader = SoundScpReader(checkscp)
            with open(log_file,"w") as f:
                for k in reader.keys():
                    rate, array = reader[k]
                    if array.shape[0] == 0:
                        f.write(k)
        del_uids = [line.strip() for line  in log_file.open().readlines()]
        return del_uids
    
    def from_wavscp_find_short(path,min_sample):
        dirpath = path.parent
        log_file = dirpath / "check.tmp"
        if not log_file.exists():
            reader = SoundScpReader(checkscp)
            with open(log_file,"w") as f:
                for k in reader.keys():
                    rate, array = reader[k]
                    if array.shape[0] <= min_sample:
                        f.write(k+"\n")
        del_uids = [line.strip() for line  in log_file.open().readlines()]
        return del_uids

    def from_shape_file_find_short(path,min_sample):
        dic = read_2column_text(path)
        keys = []
        for key,value in dic.items():
            if int(value) <= min_sample:
                keys.append(key)
        return keys

    def del_keys(keys,path):
        dic = read_2column_text(path)
        for key in keys:
            if key in dic:
                del dic[key]
        path = Path(path)
        parent = path.parent
        file = path.name
        with DatadirWriter(str(parent)) as writer:
            subwriter = writer[str(file)]
            for key,value in dic.items():
                 subwriter[key] = value


    parser = argparse.ArgumentParser()
    parser.add_argument( "--dirpath",type=str,default="",help="dump/raw/eval_far")
    parser.add_argument( "--shape_file",type=str,default="",help="shapefilepath",)
    parser.add_argument( "--min_time",type=float,default=0.08,help="min_time for utters",)
    args = parser.parse_args()
    dirpath = Path(args.dirpath)

    checkscp = dirpath / "wav.scp"
    log_file = dirpath / "check.tmp" # to store del uttid
    min_sample = args.min_time * 16000 # time * fs
  
    # find zero uids by shapefile ordirectly by wav.scp 
    if args.shape_file:
        del_uids = from_shape_file_find_short(args.shape_file,min_sample)
    else:
        del_uids = from_wavscp_find_short(checkscp,min_sample)
    # del uttids in wav.scp
    del_keys(del_uids,checkscp)

   
