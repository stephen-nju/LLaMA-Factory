export PROJECT_PATH=/home/jovyan/zhubin/code/LLaMA-Factory/
cd ${PROJECT_PATH}
# CUDA_VISIBLE_DEVICES=7 python src/export_model.py \
#     --model_name_or_path /data/SHARE/MODELS/XVERSE-13B/ \
#     --template xverse \
#     --finetuning_type lora \
#     --checkpoint_dir /home/zb/saved_checkpoint/base_xverse_sn_v6_lora_lr1e4_2epoch/ \
#     --export_dir /home/zb/saved_checkpoint/base_xverse_sn_v6_lora_lr1e4_2epoch/merge/

# wait
CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
	--model_name_or_path /home/jovyan/zhubin/DATA/models/honor2_5b_patch/ \
	--template honor \
	--finetuning_type lora \
	--adapter_name_or_path /home/jovyan/zhubin/saved_checkpoint/honor_2.5b_conv_abstract_v2_lora64_lr6e5_3epoch_bs4/ \
	--export_dir /home/jovyan/zhubin/saved_checkpoint/honor_2.5b_conv_abstract_v2_lora64_lr6e5_3epoch_bs4/merge/

# CUDA_VISIBLE_DEVICES=7 python src/export_model.py \
# 	--model_name_or_path /data/SHARE/MODELS/Qwen/Qwen-14B/ \
# 	--resize_vocab true \
# 	--template qwen \
# 	--finetuning_type lora \
# 	--adapter_name_or_path /home/zb/saved_checkpoint/sft_base_qwen_sn_v1_lora_lr5e5_2epoch/ \
# 	--export_dir /home/zb/saved_checkpoint/sft_base_qwen_sn_v1_lora_lr5e5_2epoch/merge/
