import os
from tqdm import tqdm
from itertools import groupby
from typing import List
from collections import Counter


def process_input_props_list(input_path:str) -> List[List]:
    lines = list(open(input_path))
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

    return info_list


def check_props_is_in_raw_list(new_props_path, raw_props_path):
    '''Check whether each proposal in generated proposal list is in raw proposals list'''
    new_props_info_list = process_input_props_list(new_props_path)
    raw_props_info_list = process_input_props_list(raw_props_path)
    props_num = []
    for video_info in tqdm(raw_props_info_list):
        vid_path = video_info[0]
        vid_num_frames = int(float(video_info[1]) * float(video_info[2]))
        WINDOW_SIZE = vid_num_frames

        num_gt = int(video_info[3])
        gt_offset = 4
        gt_boxes = [x.split() for x in video_info[gt_offset: gt_offset + num_gt]]

        props_offset = 4 + num_gt
        num_props = int(video_info[props_offset])
        props_offset += 1
        props_boxes = [x.split() for x in video_info[props_offset: props_offset + num_props]]
        props_num.append(len(props_boxes))
        continue
        '''For each sliding window'''
        # confirm whether starting with 1
        # sw_start_indices = list(range(1, vid_num_frames-WINDOW_SIZE, WINDOW_STRIDE))
        sw_start_indices = list(range(1, vid_num_frames, vid_num_frames))

        for idx, window_start in enumerate(sw_start_indices):
            window_end = window_start + WINDOW_SIZE
            gt_within_window = list(
                filter(lambda x: (float(x[1]) >= window_start) and (float(x[2]) <= window_end),
                       gt_boxes)
            )
            # 注意要筛选·越界·项
            props_within_window = list(
                filter(
                    lambda x: (float(x[-2]) >= window_start) and (float(x[-1]) <= window_end),
                    props_boxes
                )
            )
    count_dict = {i:0 for i in range(1000, 20000+1, 1000)}
    for p_num in props_num:
        num_length = len(str(p_num))
        if num_length == 3:
            count_dict[1000] += 1
        if num_length == 4:
            first_num = (p_num // 1000 + 1) * 1000
            count_dict[first_num] += 1
        if num_length == 5:
            first_num = (p_num // 1000 + 1) * 1000
            count_dict[first_num] += 1

    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    x = range(len(count_dict))
    y = [v for i, v in count_dict.items()]
    labels = [str(i) for i, v in count_dict.items()]
    ind = np.linspace(1,20, 20)
    ax.bar(ind, y, 0.8, color='green')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Proposal Number')
    ax.set_ylabel('Video Number')
    for i, y_i in enumerate(y):
        plt.text(ind[i]+0.05, y_i+0.05, f'{y_i}', ha='center')
    plt.grid(True)

    # plt.bar(x, y)
    plt.show()
    fig.savefig('props_num_2.jpg')

if __name__ == '__main__':
    new_props_path = "/home/xuhaoming/Projects/PAMI_workspace/PGCN/tools/BSN_Train_SW_proposals_unitFull_stride0.txt"
    raw_props_path = "/home/xuhaoming/Projects/PAMI_workspace/PGCN/data/bsn_train_proposal_list.txt"
    check_props_is_in_raw_list(new_props_path, raw_props_path)