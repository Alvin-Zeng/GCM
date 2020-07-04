import os
import pickle
from math import *

# dir = "/home/datasets/ActivityNet/I3D_split/RGB_Val"
# vid_list = os.listdir(dir)
# vid_num = len(vid_list)
# split_num = 100
# ave_num = floor(vid_num/split_num)
#
#
# for i in range(split_num):
#     start_num = i * ave_num
#     if i == split_num - 1:
#         end_num = max(vid_num, int((i + 1) * ave_num))
#     else:
#         end_num = min(vid_num, int((i + 1) * ave_num))
#     print(start_num, end_num)
#     for j in range(start_num, end_num):
#         src_path = os.path.join(dir, vid_list[j])
#         dst_path = os.path.join(dir, "split{:02d}".format(i))
#         if not os.path.isdir(dst_path):
#             os.makedirs(dst_path)
#         cmd = " ".join(("mv", src_path, dst_path))
#         os.system(cmd)
#         print(cmd)
#
# path_dict = {}
# dir = "/home/datasets/ActivityNet/I3D_split/RGB_Val"
# split_list = os.listdir(dir)
# for split_dir in split_list:
#     split_vid_list = os.listdir(os.path.join(dir, split_dir))
#     for vid in split_vid_list:
#         path_dict[vid] = os.path.join(dir, split_dir)
#
# pickle.dump(path_dict, open("Anet1.3_val_path_dict", 'wb'), pickle.HIGHEST_PROTOCOL)

import torch
dir = "/home/datasets/THUMOS14/I3D/Flow_Test"
vid_list = os.listdir(dir)
vid_num = len(vid_list)

for i in range(len(vid_list)):
    print("{}/{}".format(i, len(vid_list)))
    vid_name = vid_list[i]
    print(vid_name)
    ft_list = os.listdir(os.path.join(dir, vid_list[i]))
    ft_num_list = [int(ft_name.split("_")[-2]) for ft_name in ft_list]
    sorted_list = sorted(ft_num_list)

    ft_all = []
    for ind in sorted_list:
        s_frame = ind
        e_frame = ind + 64
        ft_path = os.path.join(dir, vid_list[i], "_".join([vid_name, str(s_frame), str(e_frame)]))
        ft_all.append(torch.load(ft_path))
    ft_all = torch.stack(ft_all)
    save_path = "/home/datasets/THUMOS14/I3D_video_level/Flow_Test_All/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    save_name = save_path + vid_name

    torch.save(ft_all, save_name)

    del ft_all

# import torch
# import pickle
#
# dir = "/home/datasets/THUMOS14/I3D_video_level/RGB_Test_All/"
# vid_list = os.listdir(dir)
# vid_num = len(vid_list)
# ft_dict = {}
# for cnt, vid in enumerate(vid_list):
#     print(cnt)
#     ft_dict[vid] = torch.load(os.path.join(dir, vid))
# pickle.dump(ft_dict, open("/home/datasets/THUMOS14/I3D_video_level/THUMOS14_test_ft_dict", 'wb'), pickle.HIGHEST_PROTOCOL)


print("done")