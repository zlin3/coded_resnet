#!/bin/bash
cd /home/rcf-proj/zl3/zhifeng/coded_resnet/models
source ./source_zlin.sh
bash start_train.sh $1
source deactivate
