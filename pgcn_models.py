import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import pickle
from numpy.random import randint
from torch.nn.utils.rnn import pad_sequence


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = input.view(-1, input.size(-1)).contiguous()
        support = torch.mm(input, self.weight)
        support = support.view(adj.size(0), -1, support.size(-1)).contiguous()
        output = torch.bmm(adj, support)
        # output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        return x


class PGCN(torch.nn.Module):
    def __init__(self, model_configs, graph_configs, dataset_configs, test_mode=False):
        super(PGCN, self).__init__()

        self.dataset_configs = dataset_configs
        self.num_class = model_configs['num_class']
        self.adj_num = graph_configs['adj_num']
        self.child_num = graph_configs['child_num']
        self.child_iou_num = graph_configs['iou_num']
        self.child_dis_num = graph_configs['dis_num']
        self.iou_threshold = graph_configs['iou_threshold']
        self.dis_threshold = graph_configs['dis_threshold']
        self.dropout = model_configs['dropout']
        self.test_mode = test_mode
        self.act_feat_dim = model_configs['act_feat_dim']
        self.comp_feat_dim = model_configs['comp_feat_dim']

        self._get_iou_dis_dict()
        self._prepare_pgcn()
        self.Act_GCN = GCN(self.act_feat_dim, 512, self.act_feat_dim, dropout=model_configs['gcn_dropout'])
        self.Comp_GCN = GCN(self.comp_feat_dim, 512, self.comp_feat_dim, dropout=model_configs['gcn_dropout'])
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def _get_iou_dis_dict(self):
        # training
        train_dict_path = self.dataset_configs['train_dict_path']
        self.train_act_iou_dict, self.train_act_dis_dict, _ = pickle.load(open(train_dict_path, 'rb'))
        # validation
        val_dict_path = self.dataset_configs['val_dict_path']
        self.val_act_iou_dict, self.val_act_dis_dict, _ = pickle.load(open(val_dict_path, 'rb'))

    def _prepare_pgcn(self):

        self.activity_fc = nn.Linear(self.act_feat_dim * 1, self.num_class + 1)
        self.completeness_fc = nn.Linear(self.comp_feat_dim * 1, self.num_class)
        self.regressor_fc = nn.Linear(self.comp_feat_dim * 1, 2 * self.num_class)

        nn.init.normal_(self.activity_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.activity_fc.bias.data, 0)
        nn.init.normal_(self.completeness_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.completeness_fc.bias.data, 0)
        nn.init.normal_(self.regressor_fc.weight.data, 0, 0.001)
        nn.init.constant_(self.regressor_fc.bias.data, 0)

    def train(self, mode=True):

        super(PGCN, self).train(mode)

    def get_optim_policies(self):

        normal_weight = []
        normal_bias = []

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, GraphConvolution):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
        ]

    def forward(self, batch_act_fts, batch_comp_fts, batch_props_type=None,
                batch_props_label=None, batch_reg_target=None, props_num_list=None, vid_list=None, is_train=None,
                iou_dict=None, dis_dict=None):
        if not self.test_mode:
            return self.train_forward(batch_act_fts, batch_comp_fts, batch_props_type,
                                      batch_props_label, batch_reg_target, props_num_list, vid_list, is_train)
        else:
            return self.test_forward(batch_act_fts, batch_comp_fts, iou_dict, dis_dict)

    def train_forward(self, batch_act_fts, batch_comp_fts, batch_props_type,
                      batch_props_label, batch_reg_target, props_num_list, vid_list, is_train):

        activity_fts = batch_act_fts.float()
        completeness_fts = batch_comp_fts.float()
        batch_size, padded_length, _ = activity_fts.size()
        # ******** construct iou edge and dis edge **************
        iou_adj_matrix = activity_fts.new_zeros([batch_size, padded_length, padded_length])
        dis_adj_matrix = activity_fts.new_zeros([batch_size, padded_length, padded_length])
        sem_adj_matrix = activity_fts.new_zeros([batch_size, padded_length, padded_length])
        zeros_adj_matrix = activity_fts.new_zeros(iou_adj_matrix.size())
        ones_adj_matrix = activity_fts.new_ones(iou_adj_matrix.size())
        eye_tensor = torch.eye(padded_length, device='cuda')

        # ************** calculate cosine similarity for action feautre **************
        dot_product_mat = torch.bmm(activity_fts, torch.transpose(activity_fts, 1, 2))
        len_vec = torch.sqrt(torch.sum(activity_fts * activity_fts, dim=-1)).unsqueeze(-1)
        len_mat = torch.bmm(len_vec, torch.transpose(len_vec, 1, 2)) + 1.e-6
        act_cos_sim_mat = dot_product_mat / len_mat

        # ************** comp cosine similarity for completeness feautre **************
        dot_product_mat = torch.bmm(completeness_fts, torch.transpose(completeness_fts, 1, 2))
        len_vec = torch.sqrt(torch.sum(completeness_fts * completeness_fts, dim=-1)).unsqueeze(-1)
        len_mat = torch.bmm(len_vec, torch.transpose(len_vec, 1, 2)) + 1.e-6
        comp_cos_sim_mat = dot_product_mat / len_mat
        # set the self-similarity to 0
        act_sem_matrix_wo_self = act_cos_sim_mat - eye_tensor

        vid_propsNum = zip(vid_list, props_num_list)
        for i, (vid, props_num) in enumerate(vid_propsNum):
            # get pre-computed iou and dis matrix
            act_iou_matrix = self.train_act_iou_dict[vid] if is_train else self.val_act_iou_dict[vid]
            act_dis_matrix = self.train_act_dis_dict[vid] if is_train else self.val_act_dis_dict[vid]
            act_sem_matrix = act_sem_matrix_wo_self[i, :props_num, :props_num]

            iou_adj_matrix[i, :props_num, :props_num] = act_iou_matrix

            # use to set 0
            temp_zeros = act_dis_matrix.new_zeros(act_dis_matrix.size())
            temp_ones = act_dis_matrix.new_ones(act_dis_matrix.size())
            # use to replace 0 with 2
            temp_large_num = torch.empty_like(temp_zeros).fill_(2)
            # TODO： if all the props have overlap with GT
            # if len((batch_props_type == 2).nonzero().squeeze()) == 0:
            #     print('There are videos containing 0 bg.', vid)
            #     exit()

            # ************** dis edge **************
            act_dis_matrix = torch.where(act_iou_matrix > 0, temp_large_num, act_dis_matrix)
            topk_dis, topk_dis_idx = torch.topk(act_dis_matrix, k=self.child_dis_num, largest=False, dim=-1)
            ones_matrix = topk_dis.new_ones(topk_dis.size())
            selected_adj_matrix = temp_zeros.scatter(-1, topk_dis_idx, ones_matrix)
            # to ensure that the edge with iou>0 is not selected
            act_dis_mask = torch.where(act_iou_matrix > 0, temp_zeros, temp_ones)
            dis_adj_matrix[i, :props_num, :props_num] = selected_adj_matrix * act_dis_mask

            # ************** sem edge **************
            act_sem_matrix = torch.where(act_iou_matrix > 0, temp_zeros, act_sem_matrix)
            topk_sem, topk_sem_idx = torch.topk(act_sem_matrix, k=2, dim=-1)
            ones_matrix = topk_sem.new_ones(topk_sem.size())
            selected_adj_matrix = temp_zeros.scatter(-1, topk_sem_idx, ones_matrix)
            # to ensure that the edge with iou>0 is not selected
            act_sem_mask = torch.where(act_iou_matrix > 0, temp_zeros, temp_ones)
            sem_adj_matrix[i, :props_num, :props_num] = selected_adj_matrix * act_sem_mask

        # ************** iou edge **************
        iou_adj_matrix -= eye_tensor
        origin_iou_matrix = iou_adj_matrix
        iou_adj_matrix = torch.where(origin_iou_matrix > self.iou_threshold, iou_adj_matrix, zeros_adj_matrix)
        # topk_iou, topk_iou_idx = torch.topk(iou_adj_matrix, k=self.child_iou_num, dim=-1)
        # ones_matrix = topk_iou.new_ones(topk_iou.size())
        # iou_adj_matrix = zeros_adj_matrix.scatter(-1, topk_iou_idx, ones_matrix)
        # # to ensure that the edge with iou < threshold is not selected
        # act_iou_mask = torch.where(origin_iou_matrix > self.iou_threshold, ones_adj_matrix, zeros_adj_matrix)
        # iou_adj_matrix = iou_adj_matrix * act_iou_mask

        # TODO: Find a better way
        # adj_matrix_mask = iou_adj_matrix | dis_adj_matrix
        temp_mask = iou_adj_matrix + dis_adj_matrix + sem_adj_matrix
        adj_matrix_mask = torch.where(temp_mask > 0, ones_adj_matrix, zeros_adj_matrix).float()
        # adj_matrix_mask -= eye_tensor  ####
        # normalization
        child_node_num = adj_matrix_mask.sum(-1, keepdim=True)
        adj_matrix_mask /= child_node_num + 1e-6
        # self-connection
        adj_matrix_mask += eye_tensor

        # compute cosine similarity
        # act cosine similarity


        act_adj_mat = act_cos_sim_mat * adj_matrix_mask
        comp_adj_mat = comp_cos_sim_mat * adj_matrix_mask

        # normalized by the number of nodes.
        act_adj_mat = F.relu(act_adj_mat)
        comp_adj_mat = F.relu(comp_adj_mat)

        act_gcn_ft = self.Act_GCN(activity_fts, act_adj_mat)
        comp_gcn_ft = self.Comp_GCN(completeness_fts, comp_adj_mat)

        out_act_fts = act_gcn_ft + activity_fts
        # act_fts = out_act_fts[:-1: self.adj_num, :]
        act_fts = self.dropout_layer(out_act_fts)

        comp_fts = comp_gcn_ft + completeness_fts
        # comp_fts = out_comp_fts[:-1: self.adj_num, :]

        output_act_fts = self.activity_fc(act_fts)
        output_comp_fts = self.completeness_fc(comp_fts)
        output_reg_fts = self.regressor_fc(comp_fts)

        # reshape input data
        output_act_fts = output_act_fts.view(-1, output_act_fts.size(-1)).contiguous()
        output_comp_fts = output_comp_fts.view(-1, output_comp_fts.size(-1)).contiguous()
        output_reg_fts = output_reg_fts.view(-1, self.completeness_fc.out_features, 2).contiguous()
        props_type = batch_props_type.view(-1).contiguous()
        action_labels = batch_props_label.view(-1).contiguous()
        reg_target = batch_reg_target.view(-1, 2).contiguous()
        # control proportion
        fg_indices = (props_type == 0).nonzero().squeeze()
        incomp_indices = (props_type == 1).nonzero().squeeze()
        bg_indices = (props_type == 2).nonzero().squeeze()

        unit_num = max(min(min(len(fg_indices), len(bg_indices)), len(incomp_indices) // 6), 1)
        rand_fg = randint(len(fg_indices), size=unit_num)
        rand_bg = randint(len(bg_indices), size=unit_num)
        rand_incomp = randint(len(incomp_indices), size=6 * unit_num)  # 1:6

        act_fg_indices = fg_indices[rand_fg]
        act_bg_indices = bg_indices[rand_bg]
        incomp_indices = incomp_indices[rand_incomp]
        action_indices = torch.cat([act_fg_indices, act_bg_indices], dim=0)

        regression_indices = act_fg_indices
        completeness_indices = torch.cat([act_fg_indices, incomp_indices], dim=0)

        # if output_act_fts[action_indices].size(0) == 0:
        #     a = 0
        return output_act_fts[action_indices], action_labels[action_indices], props_type[action_indices], \
               output_comp_fts[completeness_indices], action_labels[completeness_indices], \
               output_reg_fts[regression_indices], action_labels[regression_indices], reg_target[regression_indices], \
               unit_num

    def test_forward(self, input_act_fts, input_comp_fts, vid_iou_dict, vid_dis_dict):
        '''During testing, the input of network are proposals of a single video'''
        activity_fts = input_act_fts.float()
        completeness_fts = input_comp_fts.float()
        props_num = input_act_fts.size(0)
        # ******** construct iou edge and dis edge **************
        # iou_adj_matrix = activity_fts.new_zeros([props_num, props_num])
        # dis_adj_matrix = activity_fts.new_zeros([props_num, props_num])
        temp_zeros_iou = vid_iou_dict.new_zeros(vid_iou_dict.size())
        temp_zeros_dis = vid_dis_dict.new_zeros(vid_dis_dict.size())
        temp_zeros_sem = vid_dis_dict.new_zeros(vid_dis_dict.size())
        temp_zeros = vid_iou_dict.new_zeros(vid_iou_dict.size())
        temp_ones = vid_dis_dict.new_ones(vid_dis_dict.size())
        eye_tensor = torch.eye(props_num, device='cuda')

        # compute cosine similarity
        # act cosine similarity
        dot_product_mat = torch.mm(activity_fts, torch.transpose(activity_fts, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(activity_fts * activity_fts, dim=-1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec) + 1.e-6
        act_cos_sim_mat = dot_product_mat / len_mat

        # comp cosine similarity
        dot_product_mat = torch.mm(completeness_fts, torch.transpose(completeness_fts, 0, 1))
        len_vec = torch.unsqueeze(torch.sqrt(torch.sum(completeness_fts * completeness_fts, dim=-1)), dim=0)
        len_mat = torch.mm(torch.transpose(len_vec, 0, 1), len_vec) + 1.e-6
        comp_cos_sim_mat = dot_product_mat / len_mat
        # set the self-similarity to 0
        act_sem_matrix_wo_self = act_cos_sim_mat - eye_tensor

        # use to replace 0
        temp_large_num = torch.empty_like(temp_zeros_dis).fill_(2.)
        # TODO： if all the props have overlap with GT

        # ************** dis edge **************
        act_dis_matrix = torch.where(vid_iou_dict > 0, temp_large_num, vid_dis_dict)
        topk_dis, topk_dis_idx = torch.topk(act_dis_matrix, k=self.child_dis_num, largest=False, dim=-1)
        ones_matrix = topk_dis.new_ones(topk_dis.size())
        dis_adj_matrix = temp_zeros_dis.scatter(-1, topk_dis_idx, ones_matrix)
        # to ensure that the edge with iou>0 is not selected
        dis_adj_mask = torch.where(vid_iou_dict > 0, temp_zeros, temp_ones)
        dis_adj_matrix = dis_adj_matrix * dis_adj_mask

        # ************** iou edge **************
        vid_iou_dict -= eye_tensor.double()
        iou_adj_matrix = torch.where(vid_iou_dict > self.iou_threshold, vid_iou_dict, temp_zeros)
        # topk_iou, topk_iou_idx = torch.topk(iou_adj_matrix, k=self.child_iou_num, dim=-1)
        # ones_matrix = topk_iou.new_ones(topk_iou.size())
        # # to ensure that the edge with iou < threshold is not selected
        # iou_adj_matrix = temp_zeros_iou.scatter(-1, topk_iou_idx, ones_matrix)
        # iou_adj_mask = torch.where(vid_iou_dict > self.iou_threshold, temp_ones, temp_zeros)
        # iou_adj_matrix = iou_adj_matrix * iou_adj_mask

        # ************** sem edge **************
        act_sem_matrix = act_sem_matrix_wo_self.double()
        act_sem_matrix = torch.where(vid_iou_dict > 0, temp_zeros, act_sem_matrix)
        topk_sem, topk_sem_idx = torch.topk(act_sem_matrix, k=2, dim=-1)
        ones_matrix = topk_sem.new_ones(topk_sem.size())
        sem_adj_matrix = temp_zeros_sem.scatter(-1, topk_sem_idx, ones_matrix)
        # to ensure that the edge with iou>0 is not selected
        sem_adj_mask = torch.where(vid_iou_dict > 0, temp_zeros, temp_ones)
        sem_adj_matrix = sem_adj_matrix * sem_adj_mask

        # Note: the number of adj neighbor may be less than `self.child_iou_num`
        # # self-connection
        # tmp_none_eye = torch.ones((props_num, props_num), device='cuda') - eye_tensor
        # iou_adj_matrix = temp_zeros_iou.scatter(-1, topk_iou_idx, ones_matrix).float() * tmp_none_eye + eye_tensor

        # TODO: Find a better way
        # adj_matrix_mask = iou_adj_matrix | dis_adj_matrix
        temp_mask = iou_adj_matrix.float() + dis_adj_matrix.float() + sem_adj_matrix.float()
        temp_ones = temp_mask.new_ones(temp_mask.size())
        temp_zeros = temp_ones.new_zeros(temp_ones.size())
        adj_matrix_mask = torch.where(temp_mask > 0, temp_ones, temp_zeros)
        # adj_matrix_mask -= eye_tensor
        # normalization
        child_node_num = adj_matrix_mask.sum(-1, keepdim=True)
        adj_matrix_mask /= child_node_num + 1e-6
        # self-connection
        adj_matrix_mask += eye_tensor

        act_adj_mat = act_cos_sim_mat * adj_matrix_mask
        comp_adj_mat = comp_cos_sim_mat * adj_matrix_mask

        # normalized by the number of nodes. if neccessary?
        act_adj_mat = F.relu(act_adj_mat)
        comp_adj_mat = F.relu(comp_adj_mat)

        act_gcn_ft = self.Act_GCN(activity_fts[None, ...], act_adj_mat[None, ...])
        comp_gcn_ft = self.Comp_GCN(completeness_fts[None, ...], comp_adj_mat[None, ...])

        out_act_fts = act_gcn_ft + activity_fts[None, ...]
        act_fts = self.dropout_layer(out_act_fts)
        comp_fts = comp_gcn_ft + completeness_fts[None, ...]

        output_act_fts = self.activity_fc(act_fts)
        output_comp_fts = self.completeness_fc(comp_fts)
        output_reg_fts = self.regressor_fc(comp_fts)

        return output_act_fts, output_comp_fts, output_reg_fts





