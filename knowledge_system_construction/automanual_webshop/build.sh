# export OPENAI_API_KEY=<your_openai_api_key>
# export OPENAI_BASE_URL=<your_openai_base_url>
# export OPENAI_API_BASE=<your_openai_base_url>

# DEBUG="-m debugpy --listen 0.0.0.0:5679 --wait-for-client" 
python main_build.py --agent_type "autobuild" --run_name autobuild_logs \
			--model_name "gpt-4o-2024-08-06" \
			--num_env_per_task 10 \
            --pair_data_path path/to/pair_data/webshop/pair_data_gpt-4o-2024-08-06_webshop.json