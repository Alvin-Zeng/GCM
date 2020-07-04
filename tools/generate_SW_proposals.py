import os
from tqdm import tqdm
from itertools import groupby
from typing import List
from random import sample
# params for sliding window
WINDOW_SIZE=1024
WINDOW_STRIDE=512


def temporal_iou(span_A, span_B):
    """
    Calculates the intersection over union of two temporal "bounding boxes"

    span_A: (start, end)
    span_B: (start, end)
    """
    union = min(span_A[0], span_B[0]), max(span_A[1], span_B[1])
    inter = max(span_A[0], span_B[0]), min(span_A[1], span_B[1])

    if inter[0] >= inter[1]:
        return 0,0
    else:
        overlap = float(inter[1] - inter[0]) / float(span_A[1] - span_A[0])
        return float(inter[1] - inter[0]) / float(union[1] - union[0]), overlap


def process_input_props_list(input_path:str) -> List[List]:
    lines = list(open(input_path))
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

    return info_list


def generate_sw_based_proposals(info_list:List[List]):
    NUM_PROPS = 2500
    samples_num = 0
    # write_props_file = open(f'BSN_Train_SW_proposals_unit{WINDOW_SIZE}_stride{WINDOW_STRIDE}.txt', 'w')
    write_props_file = open(f'BSN_Train_SW_proposals_unitFull_stride0_Random{NUM_PROPS}.txt', 'w')
    '''For each video'''
    for video_info in tqdm(info_list):
        vid_path = video_info[0]
        vid_num_frames = int(float(video_info[1]) * float(video_info[2]))
        WINDOW_SIZE = vid_num_frames

        num_gt = int(video_info[3])
        gt_offset = 4
        gt_boxes = [x.split() for x in video_info[gt_offset: gt_offset + num_gt]]

        props_offset = 4 + num_gt
        num_props = int(video_info[props_offset])
        props_offset += 1
        props_boxes = [x.split() for x in video_info[props_offset: props_offset+num_props]]

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

            # ********************* Sample Props *******************************
            samples_num = NUM_PROPS if len(props_within_window) > NUM_PROPS else len(props_within_window)
            props_within_window = sample(props_within_window, samples_num)

            '''Update IoU and overlap value of proposals'''
            # updated_props_list = []
            # for props in props_within_window:
            #     iou_overlap_list = [temporal_iou((float(props[-2]), float(props[-1])), (float(gt[-2]), float(gt[-1])))
            #                 for gt in gt_boxes]
            #     sorted_iou_list = sorted(iou_overlap_list, key=lambda x:x[0], reverse=True)
            #     sorted_overlap_list = sorted(iou_overlap_list, key=lambda x:x[1], reverse=True)
            #
            #     best_iou = sorted_iou_list[0]
            #     # 确认一下overlap的定义，以及是否取独立的（不是对应best iou）最大的overlap
            #     best_overlap = sorted_overlap_list[0]
            #
            #
            #     label, raw_best_iou, raw_overlap, p_start, p_end = props
            #     updated_props = [label, best_iou, best_overlap, p_start, p_end]
            #
            #     updated_props_list.append(updated_props)



            '''Generate proposals within this sliding window'''
            window_samples = []
            window_samples.append(f'#{samples_num}')
            samples_num += 1
            window_samples.append(f'{vid_path}_{idx}') # vid name
            window_samples.append(f'{vid_num_frames}')
            # write sliding window
            window_samples.append(f'{window_start} {window_end}')
            # GT num
            window_samples.append(f'{len(gt_within_window)}')
            # write gt
            str_gt_within_window = list(map(lambda x: ' '.join(x), gt_within_window))
            window_samples.extend(str_gt_within_window)
            # props num within sw
            window_samples.append(f'{len(props_within_window)}')
            # write props
            str_props_within_window = list(map(lambda x: ' '.join(x), props_within_window))
            window_samples.extend(str_props_within_window)

            window_samples = list(map(lambda x:f'{x}\n', window_samples))

            write_props_file.writelines(window_samples)

    write_props_file.close()



if __name__ == '__main__':
    input_file_path = "/home/xuhaoming/Projects/PAMI_workspace/PGCN/data/bsn_train_proposal_list.txt"
    info_list = process_input_props_list(input_file_path)
    generate_sw_based_proposals(info_list)