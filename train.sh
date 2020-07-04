#!/bin/bash
save_path=$1

python pgcn_train.py thumos14 -b 8 --lr 0.0001 --snapshot_pref ${save_path} 
