#!/bin/bash
# 该脚本用于依次执行多个 fine-tuning 训练任务

# bash "scripts/train/finetune_1*1.sh" &&
bash "scripts/train/finetune_2*2_noalign.sh" &&
bash "scripts/train/finetune_2*2.sh" &&
bash "scripts/train/finetune_2*2_fastervideo.sh"