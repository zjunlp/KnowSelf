task=alfworld
model_name=llama3
exp_name=sample-${model_name}-${task}-test
save_pair_file_name=pair_data_llama3_alfworld.json
save_no_find_data_file_name=no_find_data_llama3_alfworld.json

python -m knowledge_system_construction.sample_pair_${task} \
    --agent_config llama_factory \
    --exp_config ${task} \
    --model_name ${model_name} \
    --two_shot \
    --sft_file_path knowledge_system_construction/data/${task}_sft.json \
    --save_path knowledge_system_construction/output \
    --save_pair_file_name ${save_pair_file_name} \
    --save_no_find_data \
    --save_no_find_data_file_name ${save_no_find_data_file_name} \
    --debug \

save_reflect_file_name=${model_name}_reflect.json
python -m training_data_construction.reflect_${task} \
    --agent_config llama_factory \
    --exp_config ${task} \
    --model_name ${model_name} \
    --pair_path knowledge_system_construction/output/pair_data/alfworld/${save_pair_file_name} \
    --save_file_name ${save_reflect_file_name} \
    --debug \

save_knowledge_file_name=${model_name}_knowledge.json
python -m training_data_construction.select_rule_${task} \
    --agent_config deepseek \
    --exp_config ${task} \
    --model_name deepseek-chat \
    --inst_path training_data_construction/prompt/rule_select_inst_${task}.txt \
    --pair_path training_data_construction/reflect/alfworld/wrong/${save_reflect_file_name} \
    --rule_path knowledge_system_construction/automanual_${task}/autobuildcase_logs/rule_manager.json \
    --save_file_name ${save_knowledge_file_name} \
    --debug \

knowledgeable_data=training_data_construction/rule/${task}/${save_knowledge_file_name}
reflect_data=training_data_construction/reflect/${task}/right/${save_reflect_file_name}
no_find_data=knowledge_system_construction/output/no_find_data/${task}/${save_no_find_data_file_name}
sft_data=knowledge_system_construction/data/${task}_sft.json
save_file_name=${task}_${model_name}_sft_train_data.json

python -m training_data_construction.format_train_data_${task} \
    --knowledgeable_data ${knowledgeable_data} \
    --reflect_data ${reflect_data} \
    --no_find_data ${no_find_data} \
    --sft_data ${sft_data} \
    --save_file_name ${save_file_name} \
    --debug 