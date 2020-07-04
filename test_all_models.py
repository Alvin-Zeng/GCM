import os
from glob import glob
from tqdm import tqdm

checkpoints_root = "/home/datasets/pami_thumos/bsn_2d_flow/"
model_path_list = glob(os.path.join(checkpoints_root, '*.pth.tar'))
# # test command
for ckp_path  in tqdm(model_path_list):
    ckp_name = '_'.join(os.path.basename(ckp_path).split('.')[0].split('_')[-3:])
    results_path = os.path.join(checkpoints_root, f'{ckp_name}_results')
    # _model_best.pth.tar" result -j 1
    cmd = f'python pgcn_test_new.py thumos14 {ckp_path} {results_path} -j 1'
    os.system(cmd)
#
results_path_list = glob(os.path.join(checkpoints_root, '*_results'))
# print('All Test done!')
# eval command
for ckp_results in tqdm(results_path_list):
    log_path = os.path.join(checkpoints_root, "results_log.txt")
    # thumos14 result --nms_threshold 0.35
    cmd = f'python eval_detection_results.py thumos14 {ckp_results} --nms_threshold 0.35 2>&1 | tee -a {log_path}'
    os.system(cmd)
print('All Eval done!')

