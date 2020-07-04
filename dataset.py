import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import os
import os.path
import numpy as np
from numpy.random import randint
from ops.I3D_Pooling import I3D_Pooling
from ops.io import load_proposal_file
from ops.utils import temporal_iou
from ops.detection_metrics import segment_tiou
from ops.detection_metrics import segment_distance
from tqdm import tqdm
import time
import pickle
import torch
import math


class Instance:

    def __init__(self, start_frame, end_frame, video_frame_count, window_start=None, window_end=None,
                 fps=1, label=None, best_iou=None, overlap_self=None):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, video_frame_count)
        self.sw_start = window_start
        self.sw_end = window_end
        self._label = label
        self.fps = fps

        self.coverage = (end_frame - start_frame) / video_frame_count

        self.best_iou = best_iou
        self.overlap_self = overlap_self

        self.loc_reg = None
        self.size_reg = None

    def compute_regression_targets(self, gt_list, fg_thresh):
        if self.best_iou < fg_thresh:
            # background proposals do not need this
            return
        # find the groundtruth instance with the highest IOU
        ious = [temporal_iou((self.start_frame, self.end_frame), (gt.start_frame, gt.end_frame)) for gt in gt_list]
        best_gt_id = np.argmax(ious)
        best_gt = gt_list[best_gt_id]
        prop_center = (self.start_frame + self.end_frame) / 2
        gt_center = (best_gt.start_frame + best_gt.end_frame) / 2
        prop_size = self.end_frame - self.start_frame + 1
        gt_size = best_gt.end_frame - best_gt.start_frame + 1

        # get regression target:
        # (1). center shift propotional to the proposal duration
        # (2). logarithm of the groundtruth duration over proposal duraiton

        self.loc_reg = (gt_center - prop_center) / prop_size
        try:
            self.size_reg = math.log(gt_size / prop_size)
        except:
            print(gt_size, prop_size, self.start_frame, self.end_frame)
            raise

    @property
    def start_time(self):
        return self.start_frame / self.fps

    @property
    def end_time(self):
        return self.end_frame / self.fps

    @property
    def label(self):
        return self._label if self._label is not None else -1

    @property
    def regression_targets(self):
        return [self.loc_reg, self.size_reg] if self.loc_reg is not None else [0, 0]


class VideoRecord:
    def __init__(self, prop_record, mode='Train'):
        self._data = prop_record
        self.mode = mode

        if mode == 'Train':
            window_start, window_end = self._data[-1]
            frame_count = window_end - window_start

            self.gt = [
                Instance(int(x[1]), int(x[2]), frame_count, window_start=window_start,
                         window_end=window_end, label=int(x[0]), best_iou=1.0) for x in self._data[2]
                if int(x[2]) > int(x[1])
            ]

            self.proposals = [
                Instance(int(x[3]), int(x[4]), frame_count, window_start=window_start,
                         window_end=window_end, label=int(x[0]), best_iou=float(x[1]),
                         overlap_self=float(x[2])) for x in self._data[3] if int(x[4]) > int(x[3])
            ]

        else:
            frame_count = self._data[1]
            self.gt = [
                Instance(int(x[1]), int(x[2]), frame_count, label=int(x[0]), best_iou=1.0) for x in self._data[2]
                if int(x[2]) > int(x[1])
            ]

            self.proposals = [
                Instance(int(x[3]), int(x[4]), frame_count, label=int(x[0]),
                         best_iou=float(x[1]), overlap_self=float(x[2])) for x in self._data[3] if int(x[4]) > int(x[3])
            ]

        self.gt = list(filter(lambda x: x.start_frame < frame_count, self.gt))

        self.proposals = list(filter(lambda x: x.start_frame < frame_count, self.proposals))

    @property
    def id(self):
        return self._data[0].strip("\n").split("/")[-1]

    @property
    def num_frames(self):
        # return int(self._data[1])
        if self.mode != 'Train': # test and val
            n_frame = self._data[1]
        else:
            w_start, w_end = self._data[-1]
            n_frame = w_end - w_start

        return n_frame

    @property
    def video_frames(self):
        return self._data[1]

    def get_fg(self, fg_thresh, with_gt=True):
        fg = [p for p in self.proposals if p.best_iou > fg_thresh]
        if with_gt: # 是否把GT也作为fg props
            fg.extend(self.gt)

        for x in fg: # 计算loc regression target和size regression target for every `FG` proposals.
            x.compute_regression_targets(self.gt, fg_thresh)
        return fg

    def get_negatives(self, incomplete_iou_thresh, bg_iou_thresh,
                      bg_coverage_thresh=0.01, incomplete_overlap_thresh=0.7):

        tag = [0] * len(self.proposals)

        incomplete_props = []
        background_props = []

        for i in range(len(tag)):
            if self.proposals[i].best_iou < incomplete_iou_thresh \
                    and self.proposals[i].overlap_self > incomplete_overlap_thresh:
                tag[i] = 1 # incomplete
                incomplete_props.append(self.proposals[i])

        for i in range(len(tag)):
            if tag[i] == 0 and \
                self.proposals[i].best_iou < bg_iou_thresh and \
                            self.proposals[i].coverage > bg_coverage_thresh:
                background_props.append(self.proposals[i])

        return incomplete_props, background_props


class PGCNDataSet(data.Dataset):

    def __init__(self, dataset_configs, graph_configs, prop_file, prop_dict_path, ft_path, exclude_empty=True,
                 epoch_multiplier=1, mode='Train', gt_as_fg=True, reg_stats=None):

        self.ft_path = ft_path
        self.prop_file = prop_file
        self.prop_dict_path = prop_dict_path

        self.exclude_empty = exclude_empty
        self.epoch_multiplier = epoch_multiplier
        self.gt_as_fg = gt_as_fg
        self.mode = mode

        self.fg_ratio = dataset_configs['fg_ratio']
        self.incomplete_ratio = dataset_configs['incomplete_ratio']
        self.bg_ratio = dataset_configs['bg_ratio']
        self.prop_per_video = dataset_configs['prop_per_video']


        self.fg_iou_thresh = dataset_configs['fg_iou_thresh']
        self.bg_iou_thresh = dataset_configs['bg_iou_thresh']
        self.bg_coverage_thresh = dataset_configs['bg_coverage_thresh']
        self.incomplete_iou_thresh = dataset_configs['incomplete_iou_thresh']
        self.incomplete_overlap_thresh = dataset_configs['incomplete_overlap_thresh']

        self.starting_ratio = dataset_configs['starting_ratio']
        self.ending_ratio = dataset_configs['ending_ratio']


        self.adj_num = graph_configs['adj_num']
        self.child_num = graph_configs['child_num']
        self.child_iou_num = graph_configs['iou_num']
        self.child_dis_num = graph_configs['dis_num']

        denum = self.fg_ratio + self.bg_ratio + self.incomplete_ratio
        self.fg_per_video = int(self.prop_per_video * (self.fg_ratio / denum))
        self.bg_per_video = int(self.prop_per_video * (self.bg_ratio / denum))
        self.incomplete_per_video = self.prop_per_video - self.fg_per_video - self.bg_per_video

        parse_time = time.time()
        self._parse_prop_file(stats=reg_stats)
        print("File parsed. Time:{:.2f}".format(time.time() - parse_time))

        """pre-compute iou and distance among proposals"""
        if os.path.exists(self.prop_dict_path):
            construct_time = time.time()
            # if "val" not in self.prop_dict_path:
            dicts = pickle.load(open(self.prop_dict_path, "rb"))
            print("Dict constructed. Time:{:.2f}".format(time.time() - construct_time))
            self.act_iou_dict, self.act_dis_dict, self.prop_dict = dicts[0], dicts[1], dicts[2]
        else:
            self.prop_dict = {}
            self.act_iou_dict = {}
            self.act_dis_dict = {}
            construct_time = time.time()
            if self.mode == 'Test':
                self._prepare_test_iou_dict()
            else:
                self._prepare_iou_dict()
            print("Dict constructed. Time:{:.2f}".format(time.time() - construct_time))

            pickle.dump([self.act_iou_dict, self.act_dis_dict, self.prop_dict], open(self.prop_dict_path, "wb"))

    def _prepare_iou_dict(self):
        pbar = tqdm(total=len(self.video_list))
        for cnt, video in enumerate(self.video_list):
            pbar.update(1)
            fg = video.get_fg(self.fg_iou_thresh, self.gt_as_fg)
            incomp, bg = video.get_negatives(self.incomplete_iou_thresh, self.bg_iou_thresh,
                                             self.bg_coverage_thresh, self.incomplete_overlap_thresh)
            self.prop_dict[video.id] = [fg, incomp, bg]
            video_pool = fg + incomp + bg
            # calculate act iou matrix
            prop_array = np.array([[prop.start_frame, prop.end_frame] for prop in video_pool])
            iou_array, overlap_array = segment_tiou(prop_array, prop_array)
            self.act_iou_dict[video.id] = torch.from_numpy(iou_array).cuda().float()
            # calculate act distance matrix
            prop_array = np.array([[prop.start_frame, prop.end_frame] for prop in video_pool])
            distance_array = segment_distance(prop_array, prop_array)
            self.act_dis_dict[video.id] = torch.from_numpy(distance_array).cuda().float()
        pbar.close()

    def _prepare_test_iou_dict(self):
        pbar = tqdm(total=len(self.video_list))
        for cnt, video in enumerate(self.video_list):
            pbar.update(1)
            video_pool = video.proposals
            # calculate act iou matrix
            prop_array = np.array([[prop.start_frame, prop.end_frame] for prop in video_pool])
            iou_array, overlap_array = segment_tiou(prop_array, prop_array)
            self.act_iou_dict[video.id] = iou_array
            # calculate act distance matrix
            prop_array = np.array([[prop.start_frame, prop.end_frame] for prop in video_pool])
            distance_array = segment_distance(prop_array, prop_array)
            self.act_dis_dict[video.id] = distance_array
        pbar.close()

    def _parse_prop_file(self, stats=None):
        prop_info = load_proposal_file(self.prop_file, self.mode)

        self.video_list = [VideoRecord(p, self.mode) for p in prop_info]

        if self.exclude_empty:
            self.video_list = list(filter(lambda x: len(x.gt) > 0, self.video_list))

        self.video_dict = {v.id: v for v in self.video_list}

        # construct three pools:
        # 1. Foreground
        # 2. Background
        # 3. Incomplete

        self.fg_pool = []
        self.bg_pool = []
        self.incomp_pool = []
        # 注意：是所有视频的props放在一起的
        for v in self.video_list:
            self.fg_pool.extend([(v.id, prop) for prop in v.get_fg(self.fg_iou_thresh, self.gt_as_fg)]) # if add gt into `FG`

            incomp, bg = v.get_negatives(self.incomplete_iou_thresh, self.bg_iou_thresh,
                                         self.bg_coverage_thresh, self.incomplete_overlap_thresh)

            self.incomp_pool.extend([(v.id, prop) for prop in incomp])
            self.bg_pool.extend([(v.id, prop) for prop in bg])

        if stats is None:
            self._compute_regresssion_stats()
        else:
            self.stats = stats


    def _sample_indices(self, valid_length, num_seg):
        """
        :param record: VideoRecord
        :return: list
        """
        average_duration = (valid_length + 1) // num_seg
        if average_duration > 0:
            # normal cases
            offsets = np.multiply(list(range(num_seg)), average_duration) \
                             + randint(average_duration, size=num_seg)
        elif valid_length > num_seg:
            offsets = np.sort(randint(valid_length, size=num_seg))
        else:
            offsets = np.zeros((num_seg, ))

        return offsets


    def _sample_pgcn_indices(self, prop, frame_cnt):
        '''Expand props'''
        start_frame = prop.start_frame + 1
        end_frame = prop.end_frame

        duration = end_frame - start_frame + 1
        assert duration != 0, (prop.start_frame, prop.end_frame, prop.best_iou)

        valid_starting = max(1, start_frame - int(duration * self.starting_ratio))
        valid_ending = min(frame_cnt, end_frame + int(duration * self.ending_ratio))

        # get starting
        act_s_e = (start_frame, end_frame)
        comp_s_e = (valid_starting, valid_ending)

        offsets = np.concatenate((act_s_e, comp_s_e))

        return offsets

    def _load_prop_data(self, prop):

        # read frame count
        frame_cnt = self.video_dict[prop[0][0]].num_frames

        # sample segment indices
        prop_indices = self._sample_pgcn_indices(prop[0][1], frame_cnt)

        # get label
        if prop[1] == 0:
            label = prop[0][1].label
        elif prop[1] == 1:
            label = prop[0][1].label  # incomplete
        elif prop[1] == 2:
            label = 0  # background
        else:
            raise ValueError()

        # get regression target
        if prop[1] == 0:
            reg_targets = prop[0][1].regression_targets
            reg_targets = (reg_targets[0] - self.stats[0][0]) / self.stats[1][0], \
                          (reg_targets[1] - self.stats[0][1]) / self.stats[1][1]
        else:
            reg_targets = (0.0, 0.0)

        return prop_indices, label, reg_targets, prop[1]


    def _compute_regresssion_stats(self):

        targets = []
        for video in self.video_list:
            fg = video.get_fg(self.fg_iou_thresh, False)
            for p in fg: # [loc_reg_target, size_reg_target]
                targets.append(list(p.regression_targets))

        self.stats = np.array((np.mean(targets, axis=0), np.std(targets, axis=0)))


    def get_test_data(self, video):

        props = video.proposals
        video_id = video.id
        frame_cnt = video.num_frames

        # process proposals to subsampled sequences
        rel_prop_list = []
        proposal_tick_list = []

        for proposal in props:

            rel_prop = proposal.start_frame / frame_cnt, proposal.end_frame / frame_cnt
            rel_duration = rel_prop[1] - rel_prop[0]
            rel_starting_duration = rel_duration * self.starting_ratio
            rel_ending_duration = rel_duration * self.ending_ratio
            rel_starting = rel_prop[0] - rel_starting_duration
            rel_ending = rel_prop[1] + rel_ending_duration

            real_rel_starting = max(0.0, rel_starting)
            real_rel_ending = min(1.0, rel_ending)

            proposal_ticks =  int(rel_prop[0] * frame_cnt), int(rel_prop[1] * frame_cnt), \
                              int(real_rel_starting * frame_cnt), int(real_rel_ending * frame_cnt)

            rel_prop_list.append(rel_prop)
            proposal_tick_list.append(proposal_ticks)
        # [N, 2], [N, 4]
        return torch.from_numpy(np.array(rel_prop_list)), \
               torch.from_numpy(np.array(proposal_tick_list)), \
               video_id, video.num_frames


    def get_props(self, video):
        fg, incomp, bg = self.prop_dict[video.id][0], self.prop_dict[video.id][1], self.prop_dict[video.id][2]

        out_props = []
        video_pool_list = [fg, incomp, bg]
        # TODO: if `fg` or `incomp` or `bg` is empty
        for i, props_pool in enumerate(video_pool_list):
            for props in props_pool:
                sample = ((video.id, props), i)
                out_props.append(sample)

        return out_props


    def get_training_data(self, index):

        video = self.video_list[index]
        # props = self._video_centric_sampling(video)
        props = self.get_props(video)

        out_prop_ind = []
        out_prop_type = []
        out_prop_labels = []
        out_prop_reg_targets = []

        for idx, p in enumerate(props):
            prop_indices, prop_label, reg_targets, prop_type = self._load_prop_data(p)

            out_prop_ind.append(prop_indices)
            out_prop_labels.append(prop_label)
            out_prop_reg_targets.append(reg_targets)
            out_prop_type.append(prop_type)

        out_prop_labels = torch.from_numpy(np.array(out_prop_labels))
        out_prop_reg_targets = torch.from_numpy(np.array(out_prop_reg_targets, dtype=np.float32))
        out_prop_type = torch.from_numpy(np.array(out_prop_type))

        #load prop fts
        vid_full_name = video.id
        vid = vid_full_name.split('/')[-1]
        full_vid = vid

        if self.mode=="Train":
            vid = vid[:-2]
        act_prop_ft, comp_prop_ft = I3D_Pooling(out_prop_ind, vid, self.ft_path, video.video_frames)

        return (act_prop_ft, comp_prop_ft), out_prop_type, out_prop_labels, out_prop_reg_targets, full_vid

    def get_all_gt(self):
        gt_list = []
        for video in self.video_list:
            vid = video.id
            gt_list.extend([[vid, x.label - 1, x.start_frame / video.num_frames,
                             x.end_frame / video.num_frames] for x in video.gt])
        return gt_list

    def __getitem__(self, index):
        real_index = index % len(self.video_list)
        if self.mode == 'Test':
            return self.get_test_data(self.video_list[real_index])
        else:
            return self.get_training_data(real_index)

    def __len__(self):
        return len(self.video_list) * self.epoch_multiplier


def collate_fn(batch):
    act_ft_list , comp_ft_list, props_type_list, props_label_list, reg_target_list, vid_list = [[] for i in range(6)]

    props_num_list = []
    for sample_data in batch:
        feature = sample_data[0]
        act_ft, comp_ft = feature
        prop_type = sample_data[1]
        prop_label = sample_data[2]
        reg_target = sample_data[3]
        vid = sample_data[4]

        act_ft_list.append(act_ft)
        comp_ft_list.append(comp_ft)
        props_type_list.append(prop_type)
        props_label_list.append(prop_label)
        reg_target_list.append(reg_target)
        props_num_list.append(len(prop_label))
        vid_list.append(vid)

    batch_act_fts = pad_sequence(act_ft_list, batch_first=True)
    batch_comp_fts = pad_sequence(comp_ft_list, batch_first=True)
    batch_props_type = pad_sequence(props_type_list, batch_first=True, padding_value=-1)
    batch_props_label = pad_sequence(props_label_list, batch_first=True, padding_value=-1)
    batch_reg_target = pad_sequence(reg_target_list, batch_first=True)

    return batch_act_fts, batch_comp_fts, batch_props_type, \
           batch_props_label, batch_reg_target, props_num_list, vid_list



if __name__ == '__main__':
    from pgcn_opts import parser
    from ops.utils import get_and_save_args
    import random
    from ops.pgcn_ops import CompletenessLoss, ClassWiseRegressionLoss

    SEED = 777
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    best_loss = 100
    pin_memory = True
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"

    configs = get_and_save_args(parser)
    parser.set_defaults(**configs)
    dataset_configs = configs["dataset_configs"]
    model_configs = configs["model_configs"]
    graph_configs = configs["graph_configs"]
    args = parser.parse_args()

    train_dataset = PGCNDataSet(dataset_configs, graph_configs,
                    prop_file=dataset_configs['train_prop_file'],
                    prop_dict_path=dataset_configs['train_dict_path'],
                    ft_path=dataset_configs['train_ft_path'],
                    epoch_multiplier=dataset_configs['training_epoch_multiplier'],
                    mode='Train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=4, shuffle=True,collate_fn=collate_fn,
        num_workers=4, pin_memory=True, drop_last=True)

    """construct model"""
    from pgcn_models import PGCN

    act_iou_dict, act_dis_dict = train_dataset.act_iou_dict, train_dataset.act_dis_dict
    model = PGCN(model_configs, graph_configs, act_iou_dict, act_dis_dict)
    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    activity_criterion = torch.nn.CrossEntropyLoss().cuda()
    completeness_criterion = CompletenessLoss().cuda()
    regression_criterion = ClassWiseRegressionLoss().cuda()

    for i, (batch_act_fts, batch_comp_fts, batch_props_type,
            batch_props_label, batch_reg_target, props_num_list, vid_list) in enumerate(tqdm(train_loader)):
        batch_size = batch_act_fts.size(0)
        activity_out, activity_target, activity_prop_type, \
        completeness_out, completeness_target, \
        regression_out, regression_labels, regression_target, ohem_num = model(batch_act_fts, batch_comp_fts, batch_props_type,
                                                                     batch_props_label, batch_reg_target,
                                                                     props_num_list, vid_list)
        act_loss = activity_criterion(activity_out, activity_target)
        comp_loss = completeness_criterion(completeness_out, completeness_target, ohem_num, 7*ohem_num)
        reg_loss = regression_criterion(regression_out, regression_labels, regression_target)
    #
    # for i, batch_data in enumerate(tqdm(train_loader)):
    #     pass
    # print('Training done!')
    val_loader = torch.utils.data.DataLoader(
        PGCNDataSet(dataset_configs, graph_configs,
                    prop_file=dataset_configs['test_prop_file'],
                    prop_dict_path=dataset_configs['val_dict_path'],
                    ft_path=dataset_configs['test_ft_path'],
                    epoch_multiplier=dataset_configs['testing_epoch_multiplier'],
                    reg_stats=train_loader.dataset.stats,
                    mode='Val'),
        batch_size=4, shuffle=False, collate_fn=collate_fn,
        num_workers=4, pin_memory=True)



    print('Testing done!')