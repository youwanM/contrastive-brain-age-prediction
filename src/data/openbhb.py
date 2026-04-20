import numpy as np
import os
import nibabel
import torch
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from collections import OrderedDict
from nilearn.masking import unmask

def bin_age(age_real: torch.Tensor):
    bins = [i for i in range(4, 92, 2)]
    age_binned = age_real.clone()
    for value in bins[::-1]:
        age_binned[age_real <= value] = value
    return age_binned.long()

def read_data(path, dataset):
    print(f"Read {dataset.upper()} index")
    df = pd.read_csv(os.path.join(path, dataset + ".tsv"), sep="\t")
    if "site" in df.columns:
        df.loc[df["split"] == "external_test", "site"] = np.nan
    return df

class OpenBHB(torch.utils.data.Dataset):
    def __init__(self, root, train=True, internal=True, transform=None, 
                 label="cont", fast=False, load_feats=None):
        self.root = root
        self.train = train
        self.internal = internal
        self.label = label
        self.fast = fast
        self.T = transform

        if train:
            self.dataset_name = "train"
        else:
            self.dataset_name = "internal_test" if internal else "external_test"
        
        self.df = read_data(root, self.dataset_name)
        
        self.bias_feats = None
        if load_feats:
            print("Loading biased features", load_feats)
            self.bias_feats = torch.load(load_feats, map_location="cpu")
        
        print(f"Dataset initialized with {len(self.df)} records")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        sub_id = str(row['participant_id'])
        # Handle cases where sub- prefix might already be in the TSV
        if not sub_id.startswith("sub-"):
            sub_id = f"sub-{sub_id}"
            
        age = row['age']
        site = row.get('site', np.nan)

        file_name = f"{sub_id}_preproc-cat12vbm_desc-gm_T1w.npy"
        file_path = os.path.join(self.root, self.dataset_name, "derivatives", sub_id, "ses-1", file_name)
        
        if not self.fast:
            try:
                x = np.load(file_path)
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}")
                # Fallback to zeros for the specific modality (VBM size)
                x = np.zeros(519945) 
        else:
            x = np.zeros(519945)

        if self.T is not None:
            x = self.T(x)
        
        if self.label == "bin":
            age = bin_age(torch.tensor(age))
        
        if self.bias_feats is not None:
            return x, age, self.bias_feats[index]
        else:
            return x, age, site

class FeatureExtractor(BaseEstimator, TransformerMixin):
    MODALITIES = OrderedDict([
        ("vbm", {"shape": (1, 121, 145, 121), "size": 519945}),
        ("quasiraw", {"shape": (1, 182, 218, 182), "size": 1827095}),
        ("xhemi", {"shape": (8, 163842), "size": 1310736}),
        ("vbm_roi", {"shape": (1, 284), "size": 284}),
        ("desikan_roi", {"shape": (7, 68), "size": 476}),
        ("destrieux_roi", {"shape": (7, 148), "size": 1036})
    ])
    MASKS = {
        "vbm": {"path": None, "thr": 0.05},
        "quasiraw": {"path": None, "thr": 0}
    }

    def __init__(self, dtype, mock=False):
        if dtype not in self.MODALITIES:
            raise ValueError("Invalid input data type.")
        self.dtype = dtype
        data_types = list(self.MODALITIES.keys())
        index = data_types.index(dtype)
        cumsum = np.cumsum([item["size"] for item in self.MODALITIES.values()])
        self.start = cumsum[index - 1] if index > 0 else 0
        self.stop = cumsum[index]
        
        self.masks = {"vbm": "./data/masks/cat12vbm_space-MNI152_desc-gm_TPM.nii.gz",
                      "quasiraw": "./data/masks/quasiraw_space-MNI152_desc-brain_T1w.nii.gz"}

        self.mock = mock
        if not mock:
            for key in self.masks:
                if not os.path.isfile(self.masks[key]):
                    raise ValueError(f"Mask file not found: {self.masks[key]}")
                arr = nibabel.load(self.masks[key]).get_fdata()
                thr = self.MASKS[key]["thr"]
                arr = (arr > thr).astype(np.int16)
                self.masks[key] = nibabel.Nifti1Image(arr, np.eye(4))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # NEW: Check if X is already a volume (3D, 4D, or 5D)
        if len(X.shape) > 1:
            target_shape = self.MODALITIES[self.dtype]["shape"]
            # Reshape ensuring we match the expected (C, D, H, W)
            return X.reshape(target_shape)
        
        # Original logic for flattened competition vectors
        select_X = X[self.start:self.stop]
        if self.dtype in ("vbm", "quasiraw"):
            im = unmask(select_X, self.masks[self.dtype])
            select_X = im.get_fdata().transpose(2, 0, 1)
        
        return select_X.reshape(self.MODALITIES[self.dtype]["shape"])