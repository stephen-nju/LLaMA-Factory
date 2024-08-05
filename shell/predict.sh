PROJECT_PATH=/home/jovyan/zhubin/LLaMA-Factory/
cd ${PROJECT_PATH}

CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
	--stage sft \
	--model_name_or_path /home/jovyan/zhubin/saved_checkpoint/base_qwen1_5B_honor_lora_lr5e5_epoch2/ \
	--resize_vocab true \
	--do_predict \
	--dataset cnewsum_test \
	--template honor \
	--finetuning_type lora \
	--output_dir /home/jovyan/zhubin/saved_output/abstract/preds/ \
	--per_device_eval_batch_size 4 \
