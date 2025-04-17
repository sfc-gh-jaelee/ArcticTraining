## Data Generation

1. First, run `data_gens.py` to generate script files to run
- Here, please modify `model_name`, and `s3_bucket_save_path`, `local_save_folder_name`, `vllm_tensor_parallel`.
  - `total_num_of_scripts` specifies the number of splits you want to generate (in parallel). 
  - For example, if total_num_of_scripts=4 for ultrachat, you will have 4 different scripts, which you can run in different GPUs/servers in parallel.
- If you want to add other datasets, you need to add them in `load_hf_dataset` function in `vllm_data_generation.py`. You need to align with the format the ultrachat is using, `{"messages": [{"role": "...", "content": "..."}]}`. Preprocess the dataset properly.

2. Then run the generated scripts. You can generate them in parallel nodes.

3. Finally, run `concat_generated_datasets.py` to combine the parallel-generated datasets.
- Before running, please align `s3_bucket_save_path`, `local_save_folder_name` with those used in `data_gens.py`.
- The result will saved in `disk_save_location`.