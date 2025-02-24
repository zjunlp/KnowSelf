task=alfworld
model_name=llama3-8b-alfworld-rpo
model_type=llama3
train_type=knowself # knowself sft vanilla
split=test
exp_name=${split}_${train_type}-${model_name}-${task}
output_path=outputs/${exp_name}
model_name_or_path=/path/to/knowself_model

VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1,2,3 python -m eval_agent.knowself_eval_vllm \
    --gpu_num 4 \
    --exp_config ${task} \
    --output_path ${output_path} \
    --select_agent_config deepseek \
    --select_agent_name deepseek-chat \
    --model_name_or_path ${model_name_or_path} \
    --select_knowledge_inst eval_agent/prompt/instructions/select_knowledge_${task}.txt\
    --knowledge_base_path knowledge_system_construction/automanual_${task}/autobuild_logs/rule_manager.json\
    --model_type ${model_type} \
    --split ${split} \
    --debug \
    --override