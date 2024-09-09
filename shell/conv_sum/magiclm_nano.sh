Usage() {
	cat <<EOF
Usage: train magiclm nano
-m --model_name_or_path     base model name or path
-n  --name 				    runing experiment name 
-h  --help                  display help
-e  --epoch                 num train epochs
-l  --lr					learning rate
-b  --bs					train batch size
-d  --dataset               train dataset
EOF
}

export PROJECT_PATH=/home/jovyan/zhubin/code/LLaMA-Factory/
export PYTHONPATH=/home/jovyan/zhubin/DATA/models/honor_2_5b_patched_tokenizer:$PYTHONPATH
cd ${PROJECT_PATH}
export PYTHONPATH=${PROJECT_PATH}
export DS_CONFIG_STAGE_3=${PROJECT_PATH}/config/deepspeed/zero_stage3_config.json
export DS_CONFIG_STAGE_2=${PROJECT_PATH}/config/deepspeed/zero_stage2_config.json
export WANDB_PROJECT="MagicLM_Nano"

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

###运行训练时需要先进行配置
# wo_fb30 =without full baichuan_30
# wo_si without summary instrcution 移除了摘要类的指令数据
export dataset="union_conversations_v2,cnewsum_5w,vcsum_headlines,\
firefly_summary_part,COIG_PC_core_summary_part,\
csl_9w,lcsts_5w,dialogsum,csds_dialogue,alimeeting,docsum"
export name="magiclm_nano"
export lr=2e-5
export epochs=3
export template=honor
export finetuning_type=full
export batch_size=4
export include="node1"
export gradient_accumulation_steps=1
export model_name_or_path=/home/jovyan/zhubin/DATA/models/honor2_5b_patched_tokenizer/
export save_steps=5000
export save_total_limit=2
export do_train=false
export do_eval=false
export logging_steps=5
export cutoff_len=4096
export warmup_ratio=0.03
export save_strategy=step


options=$(getopt -l "help,do_train,do_eval,model_name_or_path:,name:,epochs:,lr:,bs:,template:,\
finetuning_type:,dataset:,include:,gradient_accumulation_steps:,save_total_limit:,\
save_steps:,logging_steps:,cutoff_len:,warmup_ratio:,save_strategy:" -o "e:l:d:b:n:m:g:" -a -- "$@")

eval set -- "$options"
# echo "$options"

while true; do
	case "$1" in
	-h | --help)
		Usage
		exit 0
		;;
    --do_train)
        do_train=true
        ;;
    --do_eval)
        do_eval=true
        ;;
	-m | --model_name_or_path)
		shift
		model_name_or_path="$1"
		;;
	-e | --epochs)
		shift
		epochs="$1"
		;;
	-l | --lr)
		shift
		lr="$1"
		;;
	-b | --bs)
		shift
		batch_size="$1"
		;;
	-t | --template)
		shift
		template="$1"
		;;
	--finetuning_type)
		shift
		finetuning_type="$1"
		;;
	-g | --gradient_accumulation_steps)
		shift
		gradient_accumulation_steps="$1"
		;;
	-d | --dataset)
		shift
		dataset="$1"
		;;
	-n | --name)
		shift
		name="$1"
		;;
	--include)
		shift
		include="$1"
		;;
	--save_total_limit)
		shift
		save_total_limit="$1"
		;;
	--save_strategy)
		shift
		save_strategy="$1"
		;;
	--save_steps)
		shift
		save_steps="$1"
		;;
    --logging_steps)
        shift
        logging_steps="$1"
        ;;
    --cutoff_len)
        shift
        cutoff_len="$1"
        ;;
    --warmup_ratio)
        shift
        warmup_ratio="$1"
        ;;
	--)
		shift
		break
		;;
	esac
	shift
done

export OUTPUT_DIR=/home/jovyan/zhubin/saved_checkpoint/$name
export WANDB_DIR=$OUTPUT_DIR/logs
export HOSTFILE=/home/jovyan/zhubin/code/LLaMA-Factory/config/hostfile
mkdir -p ${OUTPUT_DIR}
mkdir -p ${WANDB_DIR}

echo "wandb dir=$WANDB_DIR"
wandb offline
deepspeed --hostfile=${HOSTFILE} --include=${include} --master_port=${MASTER_PORT} --no_local_rank \
	src/train.py \
	--deepspeed ${DS_CONFIG_STAGE_2} \
	--stage sft \
	--template ${template} \
	--do_train ${do_train} \
	--do_eval ${do_eval} \
	--model_name_or_path $model_name_or_path \
	--resize_vocab true \
	--use_fast_tokenizer false \
	--report_to wandb \
	--overwrite_output_dir \
	--overwrite_cache \
	--dataset ${dataset} \
	--cutoff_len 4096 \
	--output_dir ${OUTPUT_DIR} \
	--num_train_epochs ${epochs} \
	--overwrite_cache \
	--finetuning_type full \
	--warmup_ratio 0.03 \
	--logging_steps ${logging_steps} \
	--lr_scheduler_type cosine \
	--per_device_train_batch_size ${batch_size} \
	--per_device_eval_batch_size ${batch_size} \
	--gradient_accumulation_steps ${gradient_accumulation_steps} \
	--preprocessing_num_workers 16 \
	--save_strategy ${save_strategy} \
	--save_steps ${save_steps} \
	--save_total_limit ${save_total_limit} \
	--learning_rate ${lr} \
	--bf16 true \
	2>&1 | tee ${OUTPUT_DIR}/train.log
