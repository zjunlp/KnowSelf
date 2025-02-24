MODEL_NAME=llama3-8b-alfworld
MODEL_TYPE=llama3
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=8
TRAIN_TYPE=knowself
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

export LOCAL_RANK=0
echo "Training model ${MODEL_NAME} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --mixed_precision fp16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file train/stage3_no_offloading_accelerate.conf \
    train/train_stage1.py \
    --model_name_or_path /path/to/llama-3.1-8b-instruct \
    --model_type ${MODEL_TYPE} \
    --tokenizer_name /path/to/llama-3.1-8b-instruct \
    --use_slow_tokenizer \
    --train_file train/knowself_train_data/sft/alfworld_llama_sft.json \
    --max_seq_length 3072 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir train/output/${TRAIN_TYPE}_${MODEL_NAME}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens