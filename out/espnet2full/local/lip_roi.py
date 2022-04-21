from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
import argparse

def gen_roiscp(pt_dir,roiscpdir,filename):
    print(pt_dir)
    print(roiscpdir)
    files = pt_dir.glob("*.pt")
    with DatadirWriter(roiscpdir) as writer:
        subwriter = writer[filename]
        for file in files:
            subwriter[file.stem] = str(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("run_wpe")
    parser.add_argument( "--pt_dir",type=str,default="/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/eval_far_video_lip_segment/pt",
    help="the path where roi.pt files stored",)
    parser.add_argument( "--roiscpdir", type=str, default="/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw/org/eval_mid_lip", help="the path where roi.scp storened")
    parser.add_argument( "--filename", type=str, default="roi.scp", help="the path where roi.scp storened")
    # pt_dir = Path("/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/eval_far_video_lip_segment/pt")
    # roiscpdir = "/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/dump/raw/org/eval_mid_lip"
    args = parser.parse_args()
    gen_roiscp(Path(args.pt_dir),args.roiscpdir,args.filename)