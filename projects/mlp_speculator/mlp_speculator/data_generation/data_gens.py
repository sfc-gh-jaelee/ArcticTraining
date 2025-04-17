import os

model_name = tokenizer_name = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
local_save_folder_name = "llama31_8b_gen_mlpspec_nodetok_hf"
s3_bucket_save_path = "s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/datas"
script_save_path = 'scripts'
os.makedirs(script_save_path, exist_ok=True)

vllm_tensor_parallel = 1
## Ultrachat generation
total_num_of_scripts = 4
for i in range(total_num_of_scripts):
    output_dir = f"{local_save_folder_name}/ultrachat"
    json_save_path = f"{output_dir}/{i}_{total_num_of_scripts}.jsonl"
    script = f"""
python vllm_data_generation.py --model={model_name} --tensor_parallel={vllm_tensor_parallel} --tokenizer={tokenizer_name} --cur_split={i} --output_dataset_path={json_save_path} --total_split={total_num_of_scripts}
aws s3 sync {output_dir} {s3_bucket_save_path}/{output_dir}
    """
    with open(f"{script_save_path}/{i:02}.sh", 'w') as f:
        f.write(script)

## Magicoder generation
total_num_of_scripts = 2
for i in range(total_num_of_scripts):
    output_dir = f"{local_save_folder_name}/magicoder"
    json_save_path = f"{output_dir}/{i}_{total_num_of_scripts}.jsonl"
    script = f"""
python vllm_data_generation.py --hf_dataset magicoder --model={model_name} --tensor_parallel={vllm_tensor_parallel} --tokenizer={tokenizer_name} --cur_split={i} --output_dataset_path={json_save_path} --total_split={total_num_of_scripts}
aws s3 sync {output_dir} {s3_bucket_save_path}/{output_dir}
    """
    with open(f"{script_save_path}/magic_{i:02}.sh", 'w') as f:
        f.write(script)
