export PROJECT_PATH=/home/jovyan/zhubin/LLaMA-Factory/
cd ${PROJECT_PATH}
export PYTHONPATH=/home/jovyan/zhubin/DATA/models/honor2_5b/:$PYTHONPATH


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/train_bash.py \
	--stage sft \
	--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b/ \
	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/honor_2.5b_summary_lora_5e5_2epoch_bs4 \
	--resize_vocab true \
	--do_predict \
	--dataset firefly_summary_part_test \
	--template honor \
	--finetuning_type lora \
	--output_dir /home/jovyan/zhubin/saved_output/abstract/preds/ \
	--cutoff_len 2048 \
	--max_new_tokens 512 \
	--per_device_eval_batch_size 1 \
	--predict_with_generate \
