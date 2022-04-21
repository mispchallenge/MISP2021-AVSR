#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import json
import codecs


def json2dic(jsonpath, dic=None):
    """
    read dic from json or write dic to json
    :param jsonpath: filepath of json
    :param dic: content dic or None, None means read
    :return: content dic for read while None for write
    """
    if dic is None:
        with codecs.open(jsonpath, 'r') as handle:
            output = json.load(handle)
        return output
    else:
        assert isinstance(dic, dict)
        with codecs.open(jsonpath, 'w') as handle:
            json.dump(dic, handle)
        return None


def index_file():
    for dataset in ['eval', 'dev', 'train', 'addition']:
        item2dir = {
            'far_wave': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_far_audio_segment'.format(dataset),
            'far_pdf': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_far_tri3_ali'.format(dataset),
            'far_gss_wave': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_far_audio_gss_segment'.format(dataset),
            'far_gss_pdf': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_far_gss_tri3_ali'.format(dataset),
            'near_wave': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_near_audio_segment'.format(dataset),
            'near_pdf': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_near_tri3_ali'.format(dataset),
            'middle_wave': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_middle_audio_segment'.format(dataset),
            'middle_pdf': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_middle_tri3_ali'.format(dataset),
            'far_lip': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_far_video_lip_segment'.format(dataset),
            'far_head': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_far_video_head_segment'.format(dataset),
            'middle_lip': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_middle_video_lip_segment'.format(dataset),
            'middle_head': '/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_middle_video_head_segment'.format(dataset)
        }
        index_dict = {'keys': [], 'duration': [], 'key2path': {}}
        # sum key
        item_list = sorted([*item2dir.keys()])
        summed_key = set(json2dic(os.path.join(item2dir[item_list[0]], 'key2shape.json')).keys())
        for item in item_list[1:]:
            summed_key = summed_key & set(json2dic(os.path.join(item2dir[item], 'key2shape.json')).keys())
        for key in [*summed_key]:
            start, end = key.split('_')[-1].split('-')
            start, end = int(start), int(end)
            duration = round((end - start) / 100., 2)
            index_dict['keys'].append(key)
            index_dict['duration'].append(duration)
            index_dict['key2path'][key] = {k: os.path.join(v, 'pt', '{}.pt'.format(key)) for k,v in item2dir.items()}
        json2dic(jsonpath='/raw7/cv1/hangchen2/misp2021_avsr/feature/misp2021_avsr/{}_with_gss.json'.format(dataset), dic=index_dict)
    return None


if __name__ == '__main__':
    index_file()
