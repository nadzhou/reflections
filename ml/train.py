import numpy as np
import pandas as pd

from load import load_data
from load import stratify
from load import split_train_test_by_id

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


# Some fucked up transformation shit happening here
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
            bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

def nad_grid_searcher(reg_algo, housing_prepared, housing_labels, full_pipeline): 
    
    param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    grid_search = GridSearchCV(reg_algo, param_grid, cv=5,
                    scoring='neg_mean_squared_error',
                    return_train_score=True)

    grid_search.fit(housing_prepared, housing_labels)
    # cv_scores = grid_search.cv_results_

    # for mean_score, params in zip(cv_scores["mean_test_scores"], cv_scores["params"]): 
    #     print(np.sqrt(mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_

    #print(feature_importances)

    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    cat_encoder = full_pipeline.named_transformers_["cat"]
    at_one_hot_attribs = list(cat_encoder.categories_[0])
    ttributes = num_attribs + extra_attribs + cat_one_hot_attrib

def nad_linear_regressor(housing_labels, 
                    housing_prepared, some_data_prepared, some_labels): 
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    
    housing_predictions =  lin_reg.predict(some_data_prepared)

    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, 
                        scoring="neg_mean_squared_error", cv=10)

    print("\LINEAR REGRESSION SCORES")
    lin_rmse_score = np.sqrt(-lin_scores)
    display_scores(lin_rmse_score)

def display_scores(scores): 
    print(f"\nScores: {scores}")
    print(f"Scores mean: {scores.mean()}")
    print(f"Standard deviation: {scores.std()}")

def nad_random_forest_regressor(housing_labels, 
                    housing_prepared, some_data_prepared, some_labels, full_pipeline): 

    rf_reg = RandomForestRegressor()
    rf_reg.fit(housing_prepared, housing_labels)
    housing_rf_predictions = rf_reg.predict(some_data_prepared)

    rf_scores = cross_val_score(rf_reg, housing_prepared, housing_labels, 
                        scoring="neg_mean_squared_error", cv=10)

    rf_mean_score = np.sqrt(-rf_scores)
    print("\RANDOM FORST REGRESSION SCORES")
    display_scores(rf_mean_score)

    nad_grid_searcher(rf_reg, housing_prepared, housing_labels, full_pipeline)


def nad_tree_regressor(housing_labels, 
            housing_prepared, some_data_prepared, some_labels): 
    # Now to use a different model - more complex

    dt_reg = DecisionTreeRegressor()
    
    dt_reg.fit(housing_prepared, housing_labels)
    housing_dt_predictions = dt_reg.predict(some_data_prepared)
    print(f"\nDecision tree predictions: {housing_dt_predictions}")
    dt_mse = mean_squared_error(some_labels, housing_dt_predictions)
    print(f"Decision tree mean error: {dt_mse}")

    dt_scores = cross_val_score(dt_reg, housing_prepared, housing_labels, 
                        scoring="neg_mean_squared_error", cv=10)

    print("\nDECISION TREE SCORES")
    tree_rmse_score = np.sqrt(-dt_scores)
    display_scores(tree_rmse_score)





def main():
# LOAD UP THE DATA
    # uploaded my data
    housing = load_data("/home/nadzhou/ml/datasets/housing/housing.csv")

# GETTING NEW HEADINGS IN
    # Divided my data to a train and test set
    housing['id'] = housing['longitude'] * 1000 + housing["latitude"]
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# CATEGORIZE THE DATA ACCORDINGS TO USER DEFINITED GROUPS

    # Categorize the dataset into categories
    housing['income_cat'] = pd.cut(housing['median_income'],
                                bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                labels=[1, 2, 3, 4, 5])

# BUILD A TRAIN AND TEST SET
    # Stratify the data
    strat_train, strat_test = stratify(housing)

    # Now drop the category column
    for set_ in (strat_train, strat_test):
        set_.drop("income_cat", axis=1, inplace=True)

    # Set aside the test data
    housing = strat_train.copy()
    housing = strat_train.drop("median_house_value", axis=1)
    housing_labels = strat_train["median_house_value"].copy()

# FILLING UP EMPTY COLUMNS
    # The missing in data need to be filled or deleted.
    # I'm fitting it.
    imputer = SimpleImputer(strategy='median')
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)

    x = imputer.transform(housing_num)
    housing_str = pd.DataFrame(x, columns=housing_num.columns,
                            index=housing_num.index)

    housing_cat = housing[["ocean_proximity"]]

    # Either encode with ordinal or One Hot
    ordinal_encoder = OrdinalEncoder()

# NECODING

    #print(housing_cat.head())
    housing_cat_enoded = ordinal_encoder.fit_transform(housing_cat)
    #print(ordinal_encoder.categories_)

    one_hot_encoder = OneHotEncoder()

    housing_ohe_encoded = one_hot_encoder.fit_transform(housing_cat)
    #print(one_hot_encoder.categories_)

# STANDARDIZATION - NORMALIZE or STANDARDIZE

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), 
        ("attribs_adder", CombinedAttributesAdder()), 
        ("std_scalar", StandardScaler()),

    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)

# BUILD A MODEL (LINEAR REGRESSION HERE)
    

    # Linear regression here
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]

    some_data_prepared = full_pipeline.transform(some_data)

    #nad_linear_regressor(housing_labels, 
                #housing_prepared, some_data_prepared, some_labels)
    #nad_tree_regressor(housing_labels, 
                #housing_prepared, some_data_prepared, some_labels)

    nad_random_forest_regressor(housing_labels, housing_prepared, 
                    some_data_prepared, some_labels, full_pipeline)


if __name__ == '__main__':
    main()
