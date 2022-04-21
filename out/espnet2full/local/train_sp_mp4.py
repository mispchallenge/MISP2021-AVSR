from espnet2.fileio.read_text import read_2column_text
from pathlib import Path
from espnet2.fileio.datadir_writer import DatadirWriter
def main(nosp_dir,sp_dir):
    scr_mp4 = Path(nosp_dir) / "mp4.scp"
    scr_wav = Path(sp_dir) / "wav.scp"
    src_mp4_dic = read_2column_text(scr_mp4)
    scr_wav_dic = read_2column_text(scr_wav)
    with DatadirWriter(sp_dir) as writer:
        subwriter = writer["mp4.scp"]
        for k in scr_wav_dic:
            subkey = k.split("-")[-1]
            if subkey in src_mp4_dic:
                subwriter[k] = scr_wav_dic[k]

if __name__ == '__main__':
    nosp_dir = "data/train_far"
    sp_dir = "data/train_far_sp"
    main(nosp_dir,sp_dir)
