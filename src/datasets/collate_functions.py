import torch
from xtuner.utils import  IGNORE_INDEX
from typing import Dict, Sequence
from torch.nn.utils.rnn import pad_sequence
from functools import partial
from dataclasses import dataclass


def collate_func_gen(instances: Sequence[Dict],
                     pad_index: int = 151645):
    pixel_values_src, pixel_values, input_ids, input_lengths = [], [], [], []

    for example in instances:
        # 提取图像数据
        if 'pixel_values_src' in example:
            pixel_values_src.append(example.pop('pixel_values_src'))
        if 'pixel_values' in example:
            pixel_values.append(example.pop('pixel_values'))

        input_lengths.append(len(example['input_ids']))
        input_ids.append(example.pop('input_ids'))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_index)
    attention_mask = torch.zeros_like(input_ids).bool()
    for i in range(len(input_ids)):
        attention_mask[i, :input_lengths[i]] = True

    data_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

    if pixel_values:
        data_dict['pixel_values'] = torch.stack(pixel_values)
    if pixel_values_src:
        data_dict['pixel_values_src'] = torch.stack(pixel_values_src)

    return {'data': data_dict, 'data_samples': None}


def collate_func_und(instances, pad_index=151645):
    input_ids_list, labels_list, pixel_values_list = [], [], []

    for sample in instances:
        input_ids_list.append(torch.LongTensor(sample['input_ids']))
        labels_list.append(torch.LongTensor(sample['labels']))

        if 'pixel_values' in sample:
            pixel_values_list.append(sample['pixel_values'])

    ori_length = [len(input_ids_) for input_ids_ in input_ids_list]
    # right padding
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels_list, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids_list)
        labels = torch.stack(labels_list)

    attention_mask = torch.zeros_like(input_ids).bool()
    for i, length in enumerate(ori_length):
        attention_mask[i, :length] = True        # right padding

    data_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'pixel_values': torch.stack(pixel_values_list) if len(pixel_values_list) > 0 else None
    }

    return {'data': data_dict, 'data_samples': None}


class CollateConcat(object):
    def __init__(self, collate_fns, keys):
        self.keys = keys
        self.collate_fns = {}
        for key, collate_fn in zip(keys, collate_fns):
            func = collate_fn.pop('type')
            self.collate_fns[key] = partial(func, **collate_fn)

    def __call__(self, data_samples):
        data_samples = [data_sample for data_sample in data_samples if len(data_sample) > 0]
        data_dict = {}
        key = data_samples[0]['type']
        data_dict[key] = self.collate_fns[key](data_samples)['data']

        return {'data': data_dict, 'data_samples': None}
