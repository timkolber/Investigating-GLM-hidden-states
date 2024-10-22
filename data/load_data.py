import ast

import pandas as pd
from datasets import Dataset


def load_data(split: str, radius: int, mask_item: str):
    df = pd.read_csv(f"data/radius={radius}/{mask_item}_masked/{split}.csv")
    df["graph"] = df["graph"].apply(lambda x: ast.literal_eval(x))
    ds = Dataset.from_pandas(df)
    return ds


test_data = load_data("test", 1, "subject")
