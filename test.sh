#! /bin/bash/

python pgcn_test_new.py thumos14 $1 result -j2 | tee -a test_results.txt
python eval_detection_results.py thumos14 result  --nms_threshold 0.35 | tee -a test_results.txt
