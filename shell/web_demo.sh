#!/bin/bash

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# export PROJECT_PATH=/home/jovyan/zhubin/LLaMA-Factory/
# cd ${PROJECT_PATH}

# CUDA_VISIBLE_DEVICES=4 python src/web_demo.py \
# 	--model_name_or_path /home/jovyan/zhubin/saved_checkpoint/simpo_v1_ep3_lr5e6_bs2/checkpoint-1285/ \
# 	--template honor \
# 	--finetuning_type full

CUDA_VISIBLE_DEVICES=0
llamafactory-cli webchat \
 	--model_name_or_path  /home/jovyan/zhubin/saved_checkpoint/simpo_v1_ep3_lr5e6_bs2/checkpoint-1285/ \
	--template honor \
	--finetuning_type full
