export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0  
export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=160
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7
export NCCL_ALGO=Ring
export NCCL_DEBUGE=INFO

export PROJECT_PATH=/home/jovyan/zhubin/LLaMA-Factory/
export PYTHONPATH=/home/jovyan/zhubin/DATA/models/honor2_5b_patch:$PYTHONPATH
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=${PROJECT_PATH}/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=${PROJECT_PATH}/config/deepspeed/zero_stage2_config.json

MASTER_PORT=$(shuf -n 1 -i 10000-65535)


###运行训练时需要先进行配置
export DATASET="alpace_gpt4_zh_retain,cnewsum,vcsum_headlines,\
firefly_summary_part,COIG_PC_core_summary_part,\
conversation_abstract_v1_part0,\
csl_9w,lcsts_5w,dialogsum,csds_dialogue,alimeeting,docsum"

export WANDB_PROJECT="HONOR_CHAT_2.5B"
export WANDB_NAME="honor_2.5b_doc_sum_v0_full_lr1e6_3epoch_bs4"
export OUTPUT_DIR=/home/jovyan/zhubin/saved_checkpoint/${WANDB_NAME}
mkdir -p ${OUTPUT_DIR}

cat <<'EOT' > ${OUTPUT_DIR}/hostfile
node14 slots=8
EOT

# FIRST_NODE=$(awk 'NR==1 {print $1}' ${HOSTFILE})
# MASTER_ADDR=$(hostname -I |awk '{print $1}')
# echo "Using IP address of ${MASTER_ADDR} for node ${FIRST_NODE}"
# --include="node12:0,1,2,3,4,5,6,7"

wandb offline
deepspeed --hostfile=${OUTPUT_DIR}/hostfile --include="node14:0,1,2,3,4,5,6,7" --master_port=${MASTER_PORT}  --no_local_rank \
	src/train.py \
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
	--finetuning_type full \
	--warmup_ratio 0.03 \
	--logging_steps 5 \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--preprocessing_num_workers 16 \
	--save_steps 1000 \
	--save_total_limit 2 \
	--learning_rate 2e-5 \
	--bf16 true \
2>&1  | tee ${OUTPUT_DIR}/train.log

