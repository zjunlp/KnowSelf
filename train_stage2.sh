MODEL_NAME=llama3-8b-alfworld-rpo
MODEL_TYPE=llama3
NUM_GPUS=3
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=3
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

export LOCAL_RANK=0
echo "Training model ${MODEL_NAME} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

TORCH_USE_CUDA_DSA=1 CUDA_VISIBLE_DEVICES=0,1,2 CUDA_LAUNCH_BLOCKING=1 accelerate launch \
    --mixed_precision fp16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file train/stage3_no_offloading_accelerate.conf \
    train/train_stage2.py \
    --model_name_or_path train/output/knowself_llama3-8b-alfworld \
    --model_type ${MODEL_TYPE} \
    --tokenizer_name train/output/knowself_llama3-8b-alfworld \
    --use_slow_tokenizer \
    --train_file train/knowself_train_data/rpo/alfworld_llama_rpo.json \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --beta 0.5 \
    --learning_rate 5e-7 \
    --lr_scheduler_type constant_with_warmup \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --num_train_epochs 1 \
    --output_dir output/knowself_${MODEL_NAME}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 5 \
    --use_special_tokens \