import glob
import json
import os

from datasets import Dataset
from tqdm.auto import tqdm

# the same variable from data_gens.py
local_save_folder_name = "llama31_8b_gen_mlpspec_nodetok_hf"
s3_bucket_save_path = "s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/datas"
disk_save_location = local_save_folder_name + '_disk'


total_data = {
    "input_ids": [],
    "labels": [],
}
os.system(f"aws s3 sync {s3_bucket_save_path}/{local_save_folder_name} {local_save_folder_name}")
all_jsonl_files = list(sorted(glob.glob(os.path.join(local_save_folder_name, "*/*.jsonl"))))
for f in tqdm(all_jsonl_files):
    for line in open(f):
        data = json.loads(line)
        outputs = data.pop("output")
        assert len(outputs) == 256
        total_data["input_ids"].append(outputs)
        total_data["labels"].append(outputs)

print("Doing from_dict...")
dataset = Dataset.from_dict(total_data)
print("Done")
dataset.save_to_disk(disk_save_location)
