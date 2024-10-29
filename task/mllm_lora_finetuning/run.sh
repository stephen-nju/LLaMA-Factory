./mllmtrain.sh --name 'Qwen2-VL-2B-Instruct_rym_ner_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-2B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name rym_ner --dataset rym_ner_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-2B-Instruct_common_rym_ner_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-2B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name common_rym_ner --dataset common_rym_ner_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-2B-Instruct_yjzl_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-2B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name yjzl --dataset yjzl_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-2B-Instruct_thzy_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-2B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name thzy --dataset thzy_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-2B-Instruct_rccq_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-2B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name rccq --dataset rccq_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-2B-Instruct_gjccq_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-2B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name gjccq --dataset gjccq_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-2B-Instruct_tzjx_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-2B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name tzjx --dataset tzjx_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-7B-Instruct_rym_ner_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-7B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name rym_ner --dataset rym_ner_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-7B-Instruct_common_rym_ner_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-7B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name common_rym_ner --dataset common_rym_ner_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-7B-Instruct_yjzl_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-7B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name yjzl --dataset yjzl_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-7B-Instruct_thzy_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-7B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name thzy --dataset thzy_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-7B-Instruct_rccq_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-7B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name rccq --dataset rccq_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-7B-Instruct_gjccq_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-7B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name gjccq --dataset gjccq_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
./mllmtrain.sh --name 'Qwen2-VL-7B-Instruct_tzjx_train_lora=_finetuning_typelora_freeze_vision_towerFalse_lora_rank32_lora_targetqkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2_freeze_visionFalse_maxseq2048_lr0.0001' --batch_size 4 --cutoff_len 2048 --do_train True --epochs 3 --gradient_accumulation_steps 1 --include node11 --lr 0.0001 --model_name_or_path /home/jovyan/zhubin/DATA/models/Qwen2-VL-7B-Instruct --template qwen2_vl --preprocessing_num_workers 16 --stage sft --task_name tzjx --dataset tzjx_train --finetuning_type lora --freeze_vision_tower False --lora_rank 32 --lora_target 'qkv,o_proj,v_proj,fc2,down_proj,k_proj,up_proj,q_proj,proj,gate_proj,fc1,mlp.0,mlp.2' 
wait
