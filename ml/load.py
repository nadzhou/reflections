import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from zlib import crc32
import numpy as np
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.impute import SimpleImputer

def load_data(file_path): 
    data = pd.read_csv(file_path)

    return data

def stratify(housing): 
    # CATEGORIZE THE DATA ACCORDINGS TO USER DEFINITED GROUPS

    # Categorize the dataset into categories
    housing['income_cat'] = pd.cut(housing['median_income'],
                                bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, 
                        test_size=0.2, random_state=42)
    # Create a stratified dataset based 
    # on the new categories added
    for train_index, test_index in split.split(housing, housing["income_cat"]): 
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
        
        return strat_train_set, strat_test_set


def split_train_test_by_id(data, test_rate, id_column): 
    ids = data[id_column]
    in_test_et = ids.apply(lambda id_: test_set_check(id_, test_rate))
    
    return data.loc[-in_test_et], data.loc[in_test_et]


def main(): 
    # uploaded my data
    housing = load_data("/home/nadzhou/ml/datasets/housing/housing.csv")

    # Divided my data to a train and test set
    housing['id'] = housing['longitude'] * 1000 + housing["latitude"]
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    # Stratify the data    
    strat_train, strat_test = stratify(housing)

    # Now drop the category column
    for set_ in (strat_train, strat_test): 
        set_.drop("income_cat", axis=1, inplace=True)
        
    # Set aside the test data
    housing = strat_train.copy()

    # The missing in data need to be filled or deleted.
    # I'm fitting it. 
    imputer = SimpleImputer(strategy='median')
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)

    x = imputer.transform(housing_num)
    housing_str = pd.DataFrame(x, columns=housing_num.columns, 
                            index=housing_num.index)


    print(housing_str)

if __name__ == '__main__': 
    main()



# FAMILY OF PLOTTERS
# palette = sns.cubehelix_palette(light=.8, n_colors=7)
# sns.relplot(x='latitude', y='longitude', alpha=0.1, data=strat_train, 
#             legend="brief", hue='median_income')   

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#     s=housing["population"]/100, label="population", figsize=(10,7),
#     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
# )
# plt.legend()
# plt.show() 


    # attributes = ["median_income", "longitude", "latitude", "population"]

    # #scatter_matrix(housing[attributes], figsize=(12, 8))
    # #housing.plot(kind="scatter", x="median_income", y="median_house_value",
    # #                       alpha=0.1)

    # # housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    # # housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    # # housing["population_per_household"]=housing["population"]/housing["households"]

    # # housing = strat_train.drop("median_house_value", axis=0)
    # # housing_labels = strat_test["median_house_value"].copy()

    # # median = housing["total_bedrooms"].median() # option 3
    # # housing["total_bedrooms"].fillna(median, inplace=True)