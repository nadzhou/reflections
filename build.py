import numpy as np
import pandas as pd

# LOAD.PY FILE
from load import load_data
from load import stratify
from load import split_train_test_by_id

# TRAIN.PY FILE
from train import CombinedAttributesAdder
from train import piped_transform

# SKLEARN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin



rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


def main(): 
    housing = load_data("/home/nadzhou/ml/datasets/housing/housing.csv")
    train_set, test_set = train_test_split(housing, 
                            test_size=0.2, random_state=42)

    # Stratify based on income categories                       
    strat_train, strat_test = stratify(housing)

    # Now drop the category column
    for set_ in (strat_train, strat_test):
        set_.drop("income_cat", axis=1, inplace=True)

    # Set aside the test data
    housing = strat_train.copy()
    housing_labels = strat_train["median_house_value"].copy()
    housing_prepared = piped_transform(housing)
        
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), 
        ("attribs_adder", CombinedAttributesAdder()), 
        ("std_scalar", StandardScaler()),

    ])
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    imputer = SimpleImputer(strategy='median')
    imputer.fit(housing_num)


# BUILD MODEL 
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]

    some_data_prepared = full_pipeline.fit_transform(some_data)
    print(lin_reg.predict(some_data_prepared))







if __name__ == '__main__': 
    main()
