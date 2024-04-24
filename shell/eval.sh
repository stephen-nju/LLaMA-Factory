export PROJECT_PATH=/home/zb/code/LLaMA-Factory/
cd ${PROJECT_PATH}

CUDA_VISIBLE_DEVICES=6 python src/train_bash.py \
	--stage sft \
	--model_name_or_path /home/zb/saved_checkpoint/ability_test/base_qwen1.5_sn_lora_lr1e4_2epoch_dpo_1epoch/merge \
	--resize_vocab true \
	--do_predict \
	--dataset mix_test \
	--template qwen \
	--finetuning_type lora \
	--output_dir /home/zb/code/LLaMA-Factory/output/dpo \
	--cutoff_len 2048 \
	--max_new_tokens 1024 \
	--per_device_eval_batch_size 4 \
	--predict_with_generate
