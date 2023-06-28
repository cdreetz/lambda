# Script 1: load_data.py

from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset


def any_keyword_in_string(string, keywords):
    for keyword in keywords:
        if keyword in string:
            return True
    return False


def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    for sample in tqdm(iter(dataset)):
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    return Dataset.from_dict(filtered_dict)


if __name__ == "__main__":
    filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
    raw_datasets = DatasetDict(
        {
            "train": ds_train,
            "valid": ds_valid
        }
    )
    print(raw_datasets)
