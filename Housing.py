import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import hashlib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data("datasets")

# housing.style.set_caption('Housing prices')
# housing.style.format({
#     "Height": "{:20, .0f} cm",
#     "Weight": "{:20, .0f} kgs",
#     "Saving": "${:20, .0f}",
# })

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


# train_set, test_set = split_train_test(housing, 0.2)
"""
print(f"{len(train_set)} train + {len(test_set)} test")
"""


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(dataset, test_ratio, id_column, hash=hashlib.md5):
    ids = dataset[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return dataset.loc[~in_test_set], dataset.loc[in_test_set]


housing_with_id = housing.reset_index()
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print(housing["income_cat"].value_counts() / len(housing))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude")

housing.plot(kind="scatter", x="longitude", y="latitude",
             alpha=0.4, s=housing["population"] / 100, label="population",
             figsize=(10, 7), c="median_house_value",
             cmap=plt.get_cmap("jet"), colorbar=True, )
plt.legend()

corr_matrix = housing.corr(numeric_only=True)

print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing = housing.copy()[(housing["median_house_value"] < 500_000) &
                         (housing["median_house_value"] != 450_000) &
                         (housing["median_house_value"] != 350_000) &
                         (housing["median_house_value"] != 280_000)]

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))
