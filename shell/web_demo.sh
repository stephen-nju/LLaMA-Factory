
#!/bin/bash

unset http_proxy
unset https_proxy

export PROJECT_PATH=/home/jovyan/zhubin/LLaMA-Factory/
cd ${PROJECT_PATH}

CUDA_VISIBLE_DEVICES=0 python src/web_demo.py \
    --model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patch/ \
    --adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/honor_2.5b_conv_abstract_v1_lora64_lr2e5_2epoch_bs4/ \
    --template honor \
    --finetuning_type lora
