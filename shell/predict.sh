PROJECT_PATH=/home/jovyan/zhubin/code/LLaMA-Factory/
cd ${PROJECT_PATH}

CUDA_VISIBLE_DEVICES=7 python src/train.py \
	--stage sft \
	--eval_dataset union_conversations_dev_v2 \
	--overwrite_cache \
	--model_name_or_path /home/jovyan/zhubin/saved_checkpoint/magiclm_nano_callsum_v3_ep3_lr2e5_bs4/checkpoint-1904/ \
	--do_predict \
	--predict_with_generate \
	--template honor \
	--do_sample false \
	--output_dir /home/jovyan/zhubin/code/LLaMA-Factory/saved_output/ \
	--per_device_eval_batch_size 4
