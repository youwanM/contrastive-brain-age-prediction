import pandas as pd
import os
import shutil

# Set your path
data_root = "/home/ymahe/NAS/share/projects/Stroke/data/openBHB"
participants_path = os.path.join(data_root, "participants.tsv")

# Load the main index
df = pd.read_csv(participants_path, sep="\t")

# Define the splits to create
splits = {
    "train": "train.tsv",
    "internal_test": "internal_test.tsv",
    "external_test": "external_test.tsv"
}

for split_name, file_name in splits.items():
    split_df = df[df['split'] == split_name]
    output_path = os.path.join(data_root, file_name)
    split_df.to_csv(output_path, sep="\t", index=False)

    if split_name in ["internal_test", "external_test"]:
        src_dir = os.path.join(data_root, "val")
        dst_dir = os.path.join(data_root, split_name)
        if os.path.exists(src_dir):
            if os.path.islink(dst_dir) or os.path.exists(dst_dir):
                os.remove(dst_dir) if os.path.islink(dst_dir) else shutil.rmtree(dst_dir)
            os.symlink(src_dir, dst_dir)
            print(f"Created symlink from val/ to {split_name}/")

    print(f"Created {file_name} with {len(split_df)} records.")

print("Done generating split files.")