# 先激活环境
export PROJECT_PATH=/home/jovyan/zhubin/LLaMA-Factory/
export PYTHONPATH=/home/jovyan/zhubin/DATA/models/honor2_5b_patch:$PYTHONPATH
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=${PROJECT_PATH}/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=${PROJECT_PATH}/config/deepspeed/zero_stage2_config.json

MASTER_PORT=$(shuf -n 1 -i 10000-65535)



###运行训练时需要先进行配置
export DATASET="cls_headlines,vcsum_headlines,firefly_summary_part,COIG_PC_core_summary_part,conversation_abstract_part0,conversation_abstract_part1"

# export DATASET="cls_headlines"
export WANDB_PROJECT="HONOR_CHAT_2.5B"
export WANDB_NAME="honor_2.5b_conv_abstract_v1_lora64_lr5e5_3epoch_bs4"
export OUTPUT_DIR=/home/jovyan/zhubin/saved_checkpoint/honor_2.5b_conv_abstract_v1_lora64_lr5e5_3epoch_bs4
mkdir -p ${OUTPUT_DIR}

cat <<'EOT' > ${OUTPUT_DIR}/hostfile
node12 slots=8
EOT

# FIRST_NODE=$(awk 'NR==1 {print $1}' ${HOSTFILE})
# MASTER_ADDR=$(hostname -I |awk '{print $1}')
# echo "Using IP address of ${MASTER_ADDR} for node ${FIRST_NODE}"
# --include="node12:0,1,2,3,4,5,6,7"

wandb offline
deepspeed --hostfile=${OUTPUT_DIR}/hostfile --include="node12:0,1,2,3,4,5,6,7" --master_port=${MASTER_PORT}  --no_local_rank \
	src/train_bash.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--stage sft \
	--template honor \
	--do_train \
	--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patch/ \
	--resize_vocab true \
	--use_fast_tokenizer false \
	--report_to wandb \
	--overwrite_output_dir \
	--overwrite_cache \
	--dataset ${DATASET} \
	--cutoff_len 2048 \
	--output_dir  ${OUTPUT_DIR} \
	--num_train_epochs 3 \
	--overwrite_cache \
	--finetuning_type lora \
	--lora_rank 64 \
	--lora_target all \
	--warmup_ratio 0.1 \
	--logging_steps 5 \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 8 \
	--save_steps 500 \
	--save_total_limit 2 \
	--learning_rate 5e-5 \
	--bf16 true

