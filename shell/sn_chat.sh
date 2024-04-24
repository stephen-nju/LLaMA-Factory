# 先激活环境
export PROJECT_PATH=/home/zb/LLaMA-Factory/
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage2_config.json

# 运行baichuan1 7b的lora模型
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

export DATASET="who_are_you,alpaca_zh_retained,\
sn_generate_part0,sn_generate_part1,short_title_part0,short_title_part1,long_title_part0,\
long_title_part1,long_title_part2,sn_title,\
sn_xhs_multigds,sn_xhs_singlegds,sn_seo_phb,sn_seo_cp,sn_seo_other,sn_seo_zc,\
livestream,product_extract,\
param_qa,sn_chat_ir,sn_chat_rc,\
yiwen_skill_sence,yiwen_skill_gds_recom"

export WANDB_PROJECT="SN_CHAT"
export WANDB_NAME="sn_lora_5e5_2epoch_bs4"

deepspeed --include=localhost:0,1,2,3,5,6 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--stage sft \
	--template qwen \
	--do_train \
	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen1.5-14B/ \
	--report_to wandb \
	--overwrite_output_dir \
	--overwrite_cache \
	--dataset ${DATASET} \
	--cutoff_len 2048 \
	--output_dir /home/zb/saved_checkpoint/base_qwen1.5_sn_v14_lora_lr5e5_2epoch \
	--num_train_epochs 2 \
	--overwrite_cache \
	--finetuning_type lora \
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
