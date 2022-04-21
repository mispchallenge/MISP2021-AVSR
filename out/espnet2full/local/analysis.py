import numpy as np 
import codecs
def text2lines(textpath, lines_content=None):
    """
    read lines from text or write lines to txt
    :param textpath: filepath of text
    :param lines_content: list of lines or None, None means read
    :return: processed lines content for read while None for write
    """
    if lines_content is None:
        with codecs.open(textpath, 'r') as handle:
            lines_content = handle.readlines()
        processed_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
        return processed_lines
    else:
        processed_lines = [*map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), lines_content)]
        with codecs.open(textpath, 'w') as handle:
            handle.write(''.join(processed_lines))
        return None


def spk2genfunc(spk2gen_path):
    spklines = text2lines(spk2gen_path)
    spk2gen = {}
    for line in spklines:
        spk,_,gen,*_ = line.split(" ")
        spk2gen["S"+spk] = gen
    return spk2gen
def status_info(textpath,spk2gen_path):
    lines = text2lines(textpath)
    spks = []
    rooms = []
    duration_factor = 4
    segment2utt = {}
    utt2vad_array = {} #wav level
    for line in lines:
        segid = line.split()[0]
        spk,room,*_,time = segid.split("_")
        utt_id ='_'.join([room]+_)
        spks.append(spk)
        rooms.append(room)
        start,end = time.split("-")
        start = int(round(int(start) / duration_factor))
        end = int(round(int(end) / duration_factor))
        segment2utt[segid] = [utt_id, start, end]
        if utt_id in utt2vad_array:
            vad_array = utt2vad_array[utt_id]
            if end > vad_array.shape[0]:
                current_vad_array = np.zeros((end, ))
                current_vad_array[start: end] = 1
                current_vad_array[:vad_array.shape[0]] = current_vad_array[:vad_array.shape[0]]+ vad_array
                utt2vad_array[utt_id] = current_vad_array
            else:
                vad_array[start: end] = vad_array[start: end] + 1
                utt2vad_array[utt_id] = vad_array
        else:
            current_vad_array = np.zeros((end, ))
            current_vad_array[start: end] = 1
            utt2vad_array[utt_id] = current_vad_array
    ##rooms and spk 
    # import pdb;pdb.set_trace()

    urooms = (list(set(rooms)))
    uspks = (list(set(spks)))
    spk2gen = spk2genfunc(spk2gen_path)
    gens = [spk2gen[spk] for spk in uspks]

    info_dict = dict(
        room_num = len(urooms), 
        spk_num = len(uspks),
        female_num = gens.count("female"),
        male_num = gens.count("male"))

    duration = 0
    overlap_duration = 0
    for line in lines:
        key = line.split()[0]
        _, _, _, config_id, _, _ = key.split('_')
        vad_array = utt2vad_array[segment2utt[key][0]][segment2utt[key][1]: segment2utt[key][2]]
    
        overlap_duration += np.sum(vad_array > 1)
        duration +=vad_array.shape[0]

    info_dict["overlaprate"] = overlap_duration/duration
    info_dict["duration"] = duration/25/3600
    info_dict["overlap_duration"] = overlap_duration/25/3600
    print(info_dict)
    return info_dict

if __name__ == '__main__':
    textpath = dict(train="/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/data/gss_train_far/segments",
                dev="/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/data/gss_dev_far/segments",
                eval="/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/data/gss_sum_eval_far/segments")
    spk2gen_path = "/yrfs2/cv1/hangchen2/espnet/misp2021/asr1/local/spk2gen"
    for key,path in textpath.items():
        print(key)
        status_info(path,spk2gen_path)