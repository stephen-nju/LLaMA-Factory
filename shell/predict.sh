PROJECT_PATH=/home/jovyan/zhubin/code/LLaMA-Factory/
cd ${PROJECT_PATH}

CUDA_VISIBLE_DEVICES=7 python src/train.py \
	--stage sft \
	--eval_dataset union_conversations_dev_v2 \
	--overwrite_cache \
	--model_name_or_path /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_conv_sum_v2_full_lr2e5_3epoch_bs4/ \
	--do_predict \
	--predict_with_generate \
	--template honor \
	--output_dir /home/jovyan/zhubin/code/LLaMA-Factory/saved_output/ \
	--per_device_eval_batch_size 4
