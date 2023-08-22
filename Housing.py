import os.path
import numpy as np
import pandas as pd
import hashlib


def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data("datasets")

"""
print(housing.head())

print(housing.info())

print(housing["ocean_proximity"].value_counts())

print(housing.describe())

housing.hist(bins=50, figsize=(20, 15))
"""


def split_train_test(dataset, test_ratio):
    shuffled_indices = np.random.permutation(len(dataset))
    test_set_size = int(len(dataset) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return dataset.iloc[train_indices], dataset.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)
"""
print(f"{len(train_set)} train + {len(test_set)} test")
"""


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256*test_ratio


def split_train_test_by_id(dataset, test_ratio, id_column, hash=hashlib.md5):
    ids = dataset[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return dataset.loc[~in_test_set], dataset.loc[in_test_set]
