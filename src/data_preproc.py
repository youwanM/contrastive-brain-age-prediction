import pandas as pd
import os

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
    print(f"Created {file_name} with {len(split_df)} records.")

print("Done generating split files.")