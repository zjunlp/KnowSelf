# export OPENAI_API_KEY=<your_openai_api_key>
# export OPENAI_BASE_URL=<your_openai_base_url>
# export OPENAI_API_BASE=<your_openai_base_url>

task=alfworld
model_name=gpt-4o-2024-08-06
exp_name=sample-${model_name}-${task}-test
save_pair_file_name=pair_data_${model_name}_${task}$.json

python -m knowledge_system_construction.sample_pair_alfworld \
    --agent_config openai \
    --exp_config ${task} \
    --model_name ${model_name} \
    --sft_file_path knowledge_system_construction/data/alfworld_sft.json \
    --save_path knowledge_system_construction/output \
    --save_pair_file_name ${save_pair_file_name} \
    --debug \
    --type_limit 6

cd knowledge_system_construction/automanual_alfworld

python main_build.py \
    --agent_type "autobuild" \
    --run_name "autobuild_logs" \
	--model_name "gpt-4o-2024-08-06" \
    --pair_data_path ../output/pair_data/alfworld/${save_pair_file_name}