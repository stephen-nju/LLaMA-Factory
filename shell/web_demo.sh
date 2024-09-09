#!/bin/bash

unset http_proxy
unset https_proxy

export PROJECT_PATH=/home/jovyan/zhubin/LLaMA-Factory/
cd ${PROJECT_PATH}

CUDA_VISIBLE_DEVICES=0 python src/web_demo.py \
	--model_name_or_path /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_v0.1_si_callsum_v3_lr2e5_1epoch_bs4/ \
	--template honor \
	--finetuning_type full
