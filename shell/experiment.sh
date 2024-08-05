export PROJECT_PATH=/home/zb/code/LLaMA-Factory/
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/
export BAICHUAN2_13B_CHAT=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Chat/
export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage2_config.json
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

# export WANDB_PROJECT="GeneralAbilityTest"
# export WANDB_NAME="qwen1.5_sn_vd_30_lora_1e4_2epoch_bs4"
# deepspeed --include=localhost:0,1,2,3 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage sft \
# 	--template qwen \
# 	--do_train \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen1.5-14B/ \
# 	--report_to wandb \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset vertical_domain_30,alpaca_zh_retained \
# 	--cutoff_len 2048 \
# 	--output_dir /home/zb/saved_checkpoint/ability_test/qwen1.5_sn_vd_30_lora_1e4_2epoch_bs4 \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 4 \
# 	--per_device_eval_batch_size 4 \
# 	--gradient_accumulation_steps 4 \
# 	--save_steps 500 \
# 	--save_total_limit 2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true

# export WANDB_PROJECT="GeneralAbilityTest"
# export WANDB_NAME="qwen1.5_sn_vd_100_lora_1e4_2epoch_bs4"
# deepspeed --include=localhost:0,1,2,3 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage sft \
# 	--template qwen \
# 	--do_train \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen1.5-14B/ \
# 	--report_to wandb \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset vertical_domain_100,alpaca_zh_retained \
# 	--cutoff_len 2048 \
# 	--output_dir /home/zb/saved_checkpoint/ability_test/qwen1.5_sn_vd_100_lora_1e4_2epoch_bs4 \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 4 \
# 	--per_device_eval_batch_size 4 \
# 	--gradient_accumulation_steps 4 \
# 	--save_steps 500 \
# 	--save_total_limit 2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true

export WANDB_PROJECT="GeneralAbilityTest"
export WANDB_NAME="qwen1.5_sn_vd_300_lora_1e4_2epoch_bs4"
deepspeed --include=localhost:4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--stage sft \
	--template qwen \
	--do_train \
	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen1.5-14B/ \
	--report_to wandb \
	--overwrite_output_dir \
	--overwrite_cache \
	--dataset vertical_domain_300,alpaca_zh_retained \
	--cutoff_len 2048 \
	--output_dir /home/zb/saved_checkpoint/ability_test/qwen1.5_sn_vd_300_lora_1e4_2epoch_bs4 \
	--num_train_epochs 2 \
	--overwrite_cache \
	--finetuning_type lora \
	--lora_target all \
	--warmup_ratio 0.1 \
	--logging_steps 10 \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size 4 \
	--per_device_eval_batch_size 4 \
	--gradient_accumulation_steps 4 \
	--save_steps 500 \
	--save_total_limit 2 \
	--learning_rate 1e-4 \
	--bf16 true
