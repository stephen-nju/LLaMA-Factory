# 先激活环境
export PROJECT_PATH=/home/jovyan/zhubin/code/LLaMA-Factory/
export PYTHONPATH=/home/jovyan/zhubin/DATA/models/honor_2_5b_patched_tokenizer:$PYTHONPATH
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=${PROJECT_PATH}/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=${PROJECT_PATH}/config/deepspeed/zero_stage2_config.json

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

###运行训练时需要先进行配置
export DATASET="baichuan_30w,union_conversations_v2,cnewsum_5w,vcsum_headlines,\
firefly_summary_part,COIG_PC_core_summary_part,\
csl_9w,lcsts_5w,dialogsum,csds_dialogue,alimeeting,docsum"

export WANDB_PROJECT="MagicLM_Nano"
# fd=full_baichuan_30w
export WANDB_NAME="less_fb30_magiclm_nano_conv_sum_v2_full_lr2e5_3epoch_bs4"
export OUTPUT_DIR=/home/jovyan/zhubin/saved_checkpoint/${WANDB_NAME}
export HOSTFILE=/home/jovyan/zhubin/code/LLaMA-Factory/config/hostfile
mkdir -p ${OUTPUT_DIR}

# FIRST_NODE=$(awk 'NR==1 {print $1}' ${HOSTFILE})
# MASTER_ADDR=$(hostname -I |awk '{print $1}')
# echo "Using IP address of ${MASTER_ADDR} for node ${FIRST_NODE}"
# --include="node12:0,1,2,3,4,5,6,7"

wandb offline
deepspeed --hostfile=${HOSTFILE} --include="node3" --master_port=${MASTER_PORT} --no_local_rank \
	src/train.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--stage sft \
	--template honor \
	--do_train \
	--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patched_tokenizer/ \
	--resize_vocab true \
	--use_fast_tokenizer false \
	--report_to wandb \
	--overwrite_output_dir \
	--overwrite_cache \
	--dataset ${DATASET} \
	--cutoff_len 4096 \
	--output_dir ${OUTPUT_DIR} \
	--num_train_epochs 3 \
	--overwrite_cache \
	--finetuning_type full \
	--warmup_ratio 0.03 \
	--logging_steps 5 \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--preprocessing_num_workers 16 \
	--save_steps 5000 \
	--save_total_limit 2 \
	--learning_rate 2e-5 \
	--bf16 true \
	2>&1 | tee ${OUTPUT_DIR}/train.log
