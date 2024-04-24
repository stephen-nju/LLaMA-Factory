# 先激活环境
export PROJECT_PATH=/home/zb/code/LLaMA-Factory/
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export BAICHUAN2_13B=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Base/
export BAICHUAN2_13B_CHAT=/data/SHARE/MODELS/BAICHUAN/Baichuan2-13B-Chat/
export DS_CONFIG_STAGE_3=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=/home/zb/NlpTaskSpace-llm/config/deepspeed/zero_stage2_config.json

# 运行baichuan1 7b的lora模型
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

# wait
# baichuan13b 2个epoch用于对比测试学习率
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_2epoch \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true
# wait
# baichuan一个epoch 调整lora学习率加十倍
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_2epoch_lr_e4 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-4 \
# 	--bf16 true \
# 	--tf32 true

# wait
# baichuan 2epoch 调整lora学习率减少10倍
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_2epoch_lr2e6 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-6 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:6,7 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_single_task \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
# 	--warmup_ratio 0.0 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:6,7 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_single_task_512 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 512 \
# 	--max_target_length 512 \
# 	--warmup_ratio 0.0 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_1epoch_warmup0 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.0 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# 下采样测试

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/down_sample_1000/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_downsample_1000 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.0 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/down_sample_3000/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_downsample_3000 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.0 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait
# # baichuan13b 过拟合数据集
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/overfit/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_overfit_1epoch \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_teps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 truje \
# 	--tf32 true

#更新prompt形式V1版本
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/enhance_prompt_v1/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/enhance_prompt_v1/dev.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_enhance_prompt_v1 \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# 更新prompt形式V2版本
# deepspeed --include=localhost:2,3 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/merge_general_instruction/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/merge_general_instruction/train.json \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_merge_instruction \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 100 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn/dev.json \
# 	--num_train_epochs 100 \
# 	--overwrite_cache \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_overfit_1epoch \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 128 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 10000 \
# 	--save_total_limit=5 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# 淘宝数据集的任务
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_tb/single_task_v1/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_tb/single_task_v1/dev.json \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_tb_single_task \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 500 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/dev.json \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_single_task_v2_new_data \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/dev.json \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_split_task_v2_new_data \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:0,1,2,3,6,7 --master_port=29503 --hostfile="" task/baichuan/pt_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_pt/query_pretrain/train.json \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_query_pt \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--block_size 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 100 \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--lr_scheduler_type cosine \
# 	--save_steps 1000 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# deepspeed --include=localhost:0,1,4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" task/baichuan/sft_lora_trainer.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--train_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/merge_single_and_split/train.json \
# 	--validation_file /home/zb/train_data/baichuan_sft/single_task_sn_v2/split_task_v2/dev.json \
# 	--output_dir /home/zb/saved_checkpoint/baichuan_13b_sn_v2_merge_single_and_split \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--use_lora true \
# 	--lora_target W_pack \
# 	--max_source_length 1024 \
# 	--max_target_length 1024 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# --save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# # 少量垂直领域数据上 学习率大点，epoch少点
# deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B_CHAT} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate_v4_plus \
# 	--cutoff 3200 \
# 	--output_dir /home/zb/saved_checkpoint/chat_sn_v4_plus_1epoch_lr1e4 \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit 2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B_CHAT} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate_v4_plus_with_alpaca \
# 	--cutoff 3200 \
# 	--output_dir /home/zb/saved_checkpoint/chat_sn_v4_plus_alpaca_5epoch_lr2e5 \
# 	--num_train_epochs 5 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit 2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait
# deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B_CHAT} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate_v4_plus_with_alpaca \
# 	--cutoff 3200 \
# 	--output_dir /home/zb/saved_checkpoint/chat_sn_v4_plus_alpaca_1epoch_lr1e4 \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit 2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true \
# 	--tf32 true

# wait
# deepspeed --include=localhost:0,1,2,3,4,5 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B_CHAT} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate_v4_plus_with_alpaca \
# 	--cutoff 3200 \
# 	--output_dir /home/zb/saved_checkpoint/chat_sn_v4_plus_alpaca_3epoch_lr1e4 \
# 	--num_train_epochs 3 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target W_pack,o_proj \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 500 \
# 	--save_total_limit 2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true \
# 	--tf32 true

# baichuan13b-base版本的人设数据注入实验
# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate_v4 \
# 	--cutoff 2048 \
# 	--output_dir /home/zb/saved_checkpoint/base_sn_generate_v4_alpaca_3epoch \
# 	--num_train_epochs 3 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target W_pack \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 16 \
# 	--per_device_eval_batch_size 16 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 5e-5 \
# 	--bf16 true \
# 	--tf32 true

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B_CHAT} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate_v4 \
# 	--cutoff 2048 \
# 	--output_dir /home/zb/saved_checkpoint/chat_sn_generate_v4_3epoch \
# 	--num_train_epochs 3 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target W_pack \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 5e-5 \
# 	--bf16 true \
# 	--tf32 true

# # 训练base模型，使用alpaca数据，测试对齐效果
# deepspeed --include=localhost:0,1,2,3 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate_v4_plus_with_alpaca \
# 	--cutoff 3200 \
# 	--output_dir /home/zb/saved_checkpoint/base_sn_v4_plus_alpaca_lr1e4_2epoch \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 500 \
# 	--save_total_limit 2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:2,3,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate \
# 	--cutoff 2048 \
# 	--output_dir /home/zb/saved_checkpoint/bbase_sn_v4_with_alpaca_lora_all_lr1e4_3epoch \
# 	--num_train_epochs 3 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:2,3,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate \
# 	--cutoff 2048 \
# 	--output_dir /home/zb/saved_checkpoint/base_sn_v4_with_alpaca_lora_all_lr2e5_3epoch \
# 	--num_train_epochs 3 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:2,3,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate \
# 	--cutoff 2048 \
# 	--output_dir /home/zb/saved_checkpoint/base_sn_v4_with_alpaca_lora_all_lr2e5_6epoch \
# 	--num_train_epochs 6 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 2e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:4,5 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate \
# 	--cutoff 2048 \
# 	--output_dir /home/zb/saved_checkpoint/base_sn_v4_with_alpaca_lora_all_lr1e4_10epoch \
# 	--num_train_epochs 10 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--gradient_accumulation_steps 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true \
# 	--tf32 true

# # 增大学习率，测试chat的结果
# deepspeed --include=localhost:0,1 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B_CHAT} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate \
# 	--cutoff 2048 \
# 	--output_dir /home/zb/saved_checkpoint/chat_sn_generate_v4_alpaca_4epoch \
# 	--num_train_epochs 4 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target W_pack \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true \
# 	--tf32 true

# 5e-5 8个epoch chat模型 lora all 微调
# deepspeed --include=localhost:0,1 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B_CHAT} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate \
# 	--cutoff 2048 \
# 	--output_dir /home/zb/saved_checkpoint/chat_sn_generate_v4_alpaca_8epoch_lora_all \
# 	--num_train_epochs 8 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 5e-5 \
# 	--bf16 true \
# 	--tf32 true

# deepspeed --include=localhost:0,1,2,3 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset sn_generate \
# 	--cutoff 2048 \
# 	--output_dir /home/zb/saved_checkpoint/base_sn_v4_with_alpaca_lora_all_lr5e5_2epoch_grad8 \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_rank 16 \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--gradient_accumulation_steps 8 \
# 	--save_steps 500 \
# 	--save_total_limit=2 \
# 	--learning_rate 5e-5 \
# 	--bf16 true \
# 	--tf32 true

# deepspeed --include=localhost:0,1,4,5 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset who_are_you,sn_generate_v6,livestream,param_qa,alpaca_zh_retained \
# 	--cutoff 3200 \
# 	--output_dir /home/zb/saved_checkpoint/base_baichuan_sn_v6_lora_lr1e4_2epoch \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 16 \
# 	--per_device_eval_batch_size 16 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:0,1,4,5 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template baichuan2 \
# 	--do_train \
# 	--model_name_or_path ${BAICHUAN2_13B} \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset who_are_you,sn_generate_v6,livestream,param_qa,alpaca_zh_retained \
# 	--cutoff 3200 \
# 	--output_dir /home/zb/saved_checkpoint/base_baichuan_sn_v6_lora_lr5e5_2epoch \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 16 \
# 	--per_device_eval_batch_size 16 \
# 	--save_steps 200 \
# 	--save_total_limit=2 \
# 	--learning_rate 5e-5 \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:0,1,2,3,4,5 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template qwen \
# 	--do_train \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen-14B/Qwen-14B/ \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset who_are_you,sn_generate_v6,livestream,param_qa,alpaca_zh_retained \
# 	--cutoff 3200 \
# 	--output_dir /home/zb/saved_checkpoint/base_qwen_sn_v6_lora_lr1e4_2epoch \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 200 \
# 	--save_total_limit 2 \
# 	--learning_rate 1e-4 \
# 	--additional_target wte \
# 	--bf16 true \
# 	--tf32 true

# wait

# deepspeed --include=localhost:4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_3} \
# 	--stage sft \
# 	--template qwen \
# 	--do_train \
# 	--neft_alpha 5 \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen-14B/ \
# 	--report_to tensorboard \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset who_are_you,sn_generate_v7,livestream,param_qa,alpaca_zh_retained,long_title,long_title_part2,short_title,sn_title,sn_xhs \
# 	--cutoff 2048 \
# 	--output_dir /home/zb/saved_checkpoint/base_qwen_sn_v7_lora_lr1e4_neft_2epoch \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 4 \
# 	--per_device_eval_batch_size 4 \
# 	--save_steps 500 \
# 	--save_total_limit 2 \
# 	--learning_rate 1e-4 \
# 	--additional_target wte,lm_head \
# 	--bf16 true \
# 	--tf32 true

# deepspeed --include=localhost:0,1,2,3,4 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage sft \
# 	--template qwen \
# 	--do_train \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen1.5-14B/ \
# 	--report_to wandb \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset who_are_you,livestream,param_qa,alpaca_zh_retained,sn_generate_part0,sn_generate_part1,short_title_part0,short_title_part1,long_title_part0,long_title_part1,long_title_part2,sn_title,sn_xhs,sn_seo_phb,sn_seo_cp,sn_seo_other,sn_seo_zc,sn_chat_ir,sn_chat_rc,sn_qwen_tj \
# 	--cutoff_len 2048 \
# 	--output_dir /home/zb/saved_checkpoint/base_qwen1.5_sn_v13_lora_lr1e4_2epoch \
# 	--num_train_epochs 2 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 20 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 4 \
# 	--per_device_eval_batch_size 4 \
# 	--gradient_accumulation_steps 8 \
# 	--save_steps 500 \
# 	--save_total_limit 2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true


# export WANDB_PROJECT="GeneralAbilityTest"
# export WANDB_NAME="sn_lora_1e4_2epoch_bs4"
# deepspeed --include=localhost:4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage sft \
# 	--template qwen \
# 	--do_train \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen1.5-14B/ \
# 	--report_to wandb \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset who_are_you,livestream,sn_generate_part0,sn_generate_part1,short_title_part0,short_title_part1,long_title_part0,long_title_part1,long_title_part2,sn_title,sn_xhs,sn_seo_phb,sn_seo_cp,sn_seo_other,sn_seo_zc,sn_chat_ir,sn_chat_rc,sn_qwen_tj \
# 	--cutoff_len 2048 \
# 	--output_dir /home/zb/saved_checkpoint/ability_test/base_qwen1.5_sn_lora_lr1e4_1epoch \
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
# wait

# 使用dpo进行训练sft模型

# export WANDB_PROJECT="GeneralAbilityTest"
# export WANDB_NAME="sn_lora_1e4_2epoch_bs4_dpo"
# deepspeed --include=localhost:1,2,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage dpo \
# 	--template qwen \
# 	--do_train \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen1.5-14B/ \
# 	--adapter_name_or_path /home/zb/saved_checkpoint/ability_test/base_qwen1.5_sn_lora_lr1e4_1epoch/ \
# 	--report_to wandb \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset comparison_cvalues,comparison_hh_rlhf,comparison_zhihu_rlhf,comparison_gpt4 \
# 	--cutoff_len 2048 \
# 	--output_dir /home/zb/saved_checkpoint/ability_test/base_qwen1.5_sn_lora_lr1e4_2epoch_dpo_1epoch \
# 	--num_train_epochs 1 \
# 	--overwrite_cache \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
#       --dpo_beta 0.01 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 4 \
# 	--per_device_eval_batch_size 4 \
# 	--gradient_accumulation_steps 4 \
# 	--save_steps 500 \
# 	--save_total_limit 2 \
# 	--learning_rate 1e-4 \
# 	--bf16 true

# export WANDB_PROJECT="rlhf"
# 先训练sft 模型在训练reward
# deepspeed --include=localhost:0,1,2,3,4,5,6 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage sft \
# 	--template qwen \
# 	--do_train \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen-14B/ \
# 	--resize_vocab true \
# 	--report_to wandb \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--dataset who_are_you,livestream,param_qa,alpaca_zh_retained,sn_generate_part0,sn_generate_part1,short_title_part0,short_title_part1,long_title_part0,long_title_part1,long_title_part2,sn_title,sn_xhs,sn_seo_phb,sn_seo_cp,sn_seo_other,sn_seo_zc,sn_chat_ir,sn_chat_rc \
# 	--cutoff 2048 \
# 	--output_dir /home/zb/saved_checkpoint/sft_base_qwen_sn_v1_lora_lr5e5_2epoch \
# 	--num_train_epochs 2 \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--save_steps 1000 \
# 	--save_total_limit 2 \
# 	--learning_rate 5e-5 \
# 	--additional_target wte,lm_head \
# 	--bf16 true \
# 	--tf32 true

# # wait
# export WANDB_NAME="reward_model"
# deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage rm \
# 	--do_train \
# 	--template qwen \
# 	--dataset comparison_zhihu_rlhf \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen-14B/ \
# 	--resize_vocab true \
# 	--adapter_name_or_path /home/zb/saved_checkpoint/sft_base_qwen_sn_v1_lora_lr5e5_2epoch/,/home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr5e6_2epoch/checkpoint-4000 \
# 	--report_to wandb \
# 	--resume_from_checkpoint /home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr5e6_2epoch/checkpoint-4000/ \
# 	--output_dir /home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr5e6_2epoch \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--cutoff 2048 \
# 	--num_train_epochs 1 \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 4 \
# 	--per_device_eval_batch_size 4 \
# 	--gradient_accumulation_steps 8 \
# 	--save_steps 1000 \
# 	--save_total_limit 2 \
# 	--learning_rate 5e-6 \
# 	--additional_target wte,lm_head \
# 	--bf16 true \
# 	--tf32 true

# 模型恢复训练
# export WANDB_NAME="reward_model"
# deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage rm \
# 	--do_train \
# 	--template qwen \
# 	--dataset comparison_cvalues,comparison_hh_rlhf,comparison_zhihu_rlhf,comparison_gpt4 \
# 	--model_name_or_path /home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr5e6_2epoch/checkpoint-4000/merge/ \
# 	--report_to wandb \
# 	--output_dir /home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr5e6_2epoch_resume \
# 	--overwrite_output_dir \
# 	--overwrite_cache \
# 	--cutoff 2048 \
# 	--num_train_epochs 1 \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 4 \
# 	--per_device_eval_batch_size 4 \
# 	--gradient_accumulation_steps 8 \
# 	--save_steps 1000 \
# 	--save_total_limit 2 \
# 	--learning_rate 5e-6 \
# 	--additional_target wte,lm_head \
# 	--bf16 true \
# 	--tf32 true
# ppo 模型训练

# export WANDB_NAME="ppo_v2_lora_lr5e6_1epoch "
# deepspeed --include=localhost:0,1,2,3,4,5,6 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage ppo \
# 	--do_train \
# 	--template qwen \
# 	--resize_vocab true \
# 	--dataset rlhf_ppo_train_v2 \
# 	--overwrite_cache \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen-14B/ \
# 	--adapter_name_or_path /home/zb/saved_checkpoint/sft_base_qwen_sn_v1_lora_lr5e5_2epoch/ \
# 	--create_new_adapter \
# 	--output_dir /home/zb/saved_checkpoint/ppo_qwen_v2_lora_lr5e6_1epoch \
# 	--overwrite_output_dir \
# 	--reward_model /home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr5e6_2epoch_resume/ \
# 	--reward_model_type lora \
# 	--top_k 0 \
# 	--top_p 0.9 \
# 	--cutoff_len 2048 \
# 	--max_new_token 1024 \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--num_train_epochs 1 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 4 \
# 	--ppo_epochs 4 \
# 	--save_steps 200 \
# 	--save_total_limit 4 \
# 	--learning_rate 5e-6 \
# 	--additional_target wte,lm_head \
# 	--report_to wandb \
# 	--ppo_logger wandb \
# 	--bf16 true

# export WANDB_NAME="ppo_v2_lora_lr3e5_1epoch_norm_rm_resume"
# deepspeed --include=localhost:0,1,2,3,4,5 --master_port=${MASTER_PORT} --hostfile="" src/train_bash.py \
# 	--deepspeed ${DS_CONFIG_STAGE_2} \
# 	--stage ppo \
# 	--do_train \
# 	--template qwen \
# 	--resize_vocab true \
# 	--dataset rlhf_ppo_train_v2 \
# 	--overwrite_cache \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen-14B/ \
# 	--adapter_name_or_path /home/zb/saved_checkpoint/sft_base_qwen_sn_v1_lora_lr5e5_2epoch/ \
# 	--create_new_adapter \
# 	--output_dir /home/zb/saved_checkpoint/ppo_qwen_v2_lora_lr3e5_1epoch_rm_resume_norm \
# 	--overwrite_output_dir \
# 	--reward_model /home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr5e6_2epoch_resume \
# 	--reward_model_type lora \
# 	--top_k 0 \
# 	--top_p 0.9 \
# 	--cutoff_len 2048 \
# 	--max_new_token 512 \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--num_train_epochs 1 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 4 \
# 	--ppo_epochs 4 \
# 	--ppo_score_norm \
# 	--save_steps 200 \
# 	--save_total_limit 4 \
# 	--learning_rate 3e-5 \
# 	--additional_target wte,lm_head \
# 	--report_to wandb \
# 	--ppo_logger wandb \
# 	--bf16 true

# export WANDB_NAME="ppo_lora_lr1e6_1epoch "
# accelerate launch --config_file examples/lora_multi_gpu/single_config.yaml src/train_bash.py \
# 	--stage ppo \
# 	--do_train \
# 	--template qwen \
# 	--resize_vocab true \
# 	--dataset rlhf_ppo_train \
# 	--overwrite_cache \
# 	--model_name_or_path /home/zb/saved_checkpoint/sft_base_qwen_sn_v1_lora_lr5e5_2epoch/merge/ \
# 	--create_new_adapter \
# 	--output_dir /home/zb/saved_checkpoint/ppo_qwen_sn_v12_lora_lr1e6_1epoch \
# 	--overwrite_output_dir \
# 	--reward_model /home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr5e6_2epoch_resume/ \
# 	--reward_model_type lora \
# 	--top_k 0 \
# 	--top_p 0.9 \
# 	--max_new_token 512 \
# 	--finetuning_type lora \
# 	--lora_target all \
# 	--num_train_epochs 1 \
# 	--warmup_ratio 0.1 \
# 	--logging_steps 10 \
# 	--lr_scheduler_type cosine \
# 	--per_device_train_batch_size 4 \
# 	--gradient_accumulation_steps 4 \
# 	--ppo_epochs 2 \
# 	--save_steps 500 \
# 	--save_total_limit 4 \
# 	--learning_rate 1e-6 \
# 	--additional_target lm_head \
# 	--report_to wandb \
# 	--ppo_logger wandb \
# 	--bf16 true \
# 	--tf32 true
