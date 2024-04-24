PROJECT_PATH=/home/zb/code/LLaMA-Factory/
cd ${PROJECT_PATH}

CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
	--stage rm \
	--model_name_or_path /home/zb/saved_checkpoint/reward_qwen_sn_v12_lora_lr5e6_2epoch_resume/merge/ \
	--resize_vocab true \
	--do_predict \
	--dataset comparison_zhihu_rlhf_test \
	--template qwen \
	--finetuning_type lora \
	--output_dir /home/zb/code/LLaMA-Factory/output/reward/ \
	--per_device_eval_batch_size 4
