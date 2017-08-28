import tarfile
import requests
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


URL = 'https://raw.githubusercontent.com/ageron/'\
    'handson-ml/master/datasets/housing/housing.tgz'
FILE_DIR = './datasets/housing/'
FILE_PATH = FILE_DIR + 'housing.tgz'
CSV_PATH = FILE_DIR + 'housing.csv'


def fetch_data():
    r = requests.get(URL)
    with open(FILE_PATH, 'wb+') as f:
        f.write(r.content)
    tar = tarfile.open(FILE_PATH)
    tar.extractall(path=FILE_DIR)
    tar.close()


def load_housing_data():
    return pd.read_csv(CSV_PATH)


fetch_data()
data = load_housing_data()

print('Data info:')
print(data.info())

print('\nCategory feature value breakdown:')
print(data['ocean_proximity'].value_counts())

print('\nData description:')
print(data.describe())

data.hist(bins=50, figsize=(12, 8))
plt.show()

# random sampling
# train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# but...
# median_income feature does not have many samples outside of
# categories 2 to 5, so random sampling may put a few number of
# samples outside of categories 2 to 5 in our train or test set
data['median_income'].hist(bins=1000)
plt.title('Why do stratified train/test splitting? '
          'Not many samples outside of 2-5')
plt.show()

# STRATIFIED SAMPLING FIXES THIS...
# therefore we will reduce number of categories by div by 1.5,
# round up to have discrete categories, and
# put all categories above 5 into category

# create feature
data['income_cat'] = np.ceil(data['median_income'] / 1.5)
data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace=True)
# print(data['income_cat'].value_counts())
data['income_cat'].hist()
plt.title('Newly created income_cat feature\'s distribution')
plt.show()

# stratified split based on created feature
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(data, data['income_cat']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

# random split
splitter = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(data, data['income_cat']):
    rdm_train_set = data.loc[train_index]
    rdm_test_set = data.loc[test_index]


# compare proportions of income categories in test set
# note bigger differences between overall and random set
# overall and stratified set
print("\nProportions of income categories:")
print("Overall data set:")
print(data['income_cat'].value_counts() / len(data))
print("Stratified test set:")
print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))
print("Random test set:")
print(rdm_test_set['income_cat'].value_counts() / len(rdm_test_set))

strat_train_set.drop(['income_cat'], axis=1, inplace=True)
strat_test_set.drop(['income_cat'], axis=1, inplace=True)

train_set = strat_train_set.copy()
train_set.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
plt.title('District density')
plt.show()

train_set.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
               s=train_set['population'] / 100, label='population',
               c=train_set['median_house_value'], colormap='jet',
               colorbar=True)

plt.title('Population and median house value by district')
plt.legend()
plt.show()

correlation_matrix = train_set.corr()
print('\nCorrelation of features and median_house_value '
      '(1 to -1 correlation coefficient):')
print(correlation_matrix['median_house_value'].sort_values(ascending=False))


attributes = ['median_house_value', 'median_income',
              'total_rooms', 'housing_median_age']
scatter_matrix(train_set[attributes], figsize=(12, 8))
plt.title('Pandas scatter matrix')
plt.show()


train_set.plot(kind='scatter', x='median_income',
               y='median_house_value', alpha=0.1)
plt.title('Median house value versus median income')
plt.show()

# create more meaningful features
train_set['rooms_per_household'] = train_set['total_rooms'] / train_set['households']
train_set['bedrooms_per_room'] = train_set['total_bedrooms'] / train_set['total_rooms']
train_set['population_per_household'] = train_set['population'] / train_set['households']
correlation_matrix = train_set.corr()
print('\nCorrelation of features (plus new ones) and median_house_value '
      '(1 to -1 correlation coefficient):')
print(correlation_matrix['median_house_value'].sort_values(ascending=False))

# reset data and split labels from features
# (we'll put them back later via custom transformers)
X_train = strat_train_set.drop('median_house_value', axis=1)
Y_train = strat_train_set['median_house_value'].copy()


# clean data

# remember total_bedrooms has empty values as per DF.info()
print('\nNumber of missing total_bedroom values:',
      data['total_bedrooms'].isnull().sum())

# insert median into missing values
imputer = Imputer(strategy='median')
X_train_num = X_train.drop('ocean_proximity', axis=1)
imputer.fit(X_train_num)
print('\nImputer statistics_:', imputer.statistics_)
foo = imputer.transform(X_train_num)

# one hot encoding of category feature
encoder = LabelBinarizer()
X_train_cat = X_train['ocean_proximity']
X_train_cat_1hot = encoder.fit_transform(X_train_cat)
print('\nOcean proximity feature one-hot:', X_train_cat_1hot)


# custom transformer
rooms_idx, bedrooms_idx, population_idx, household_idx, = 3, 4, 5, 6


class CustomAttributesCreator(BaseEstimator, TransformerMixin):
    def __init__(self):
        # could pass in which parameters to add here and other init
        pass

    def fit(self, X, y=None):
        return self  # we aren't doing any fitting in this class

    def transform(serlf, X, y=None):
        rooms_per_household = X[:, rooms_idx] / X[:, household_idx]
        population_per_household = X[:, population_idx] / X[:, household_idx]
        bedrooms_per_room = X[:, bedrooms_idx] / X[:, household_idx]
        return np.c_[X,
                     rooms_per_household,
                     population_per_household,
                     bedrooms_per_room]


X_train_extra = CustomAttributesCreator().transform(X_train.values)
print('\nCreated features:', X_train_extra)


# chain number feature transformations together via pipeline
number_pipeline = Pipeline([
    ('imputer', Imputer(strategy='median')),
    ('attributes_adder', CustomAttributesCreator()),
    ('std_scaler', StandardScaler()),
])

X_train_housing = number_pipeline.fit_transform(X_train_num)


# union number feature transformations and category feature transforms
# into one pipeline

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


number_attributes = list(X_train_num)
category_attributes = ['ocean_proximity']

number_pipeline = Pipeline([
    ('selector', DataFrameSelector(number_attributes)),
    ('imputer', Imputer(strategy='median')),
    ('attributes_adder', CustomAttributesCreator()),
    ('std_caler', StandardScaler())
])

category_pipeline = Pipeline([
    ('selector', DataFrameSelector(category_attributes)),
    ('label_binarizer', LabelBinarizer())
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', number_pipeline),
    ('cat_pipeline', category_pipeline),
])

X_train_prepared = full_pipeline.fit_transform(X_train)
print('\nFully prepared train set shape:', X_train_prepared.shape)


def print_scores(scores):
    print('Cross validation scores:', scores)
    print('Mean:', np.mean(scores))
    print('Std:', np.std(scores))


# train the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train_prepared, Y_train)
print('\nLinear regression score on train set:',
      lin_reg.score(X_train_prepared, Y_train))
# use cross-validation to split training set into smaller
# training sets and one validation set
# (validation set not the same as the test set)
# e.g. cv=10 means 10 training runs - 9 folds trained,
# one random fold is picked as evaluation set
scores = cross_val_score(lin_reg, X_train_prepared, Y_train,
                         scoring='r2', cv=10)
print_scores(scores)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train_prepared, Y_train)
print('\nDecision tree score on train set:',
      tree_reg.score(X_train_prepared, Y_train))
scores = cross_val_score(tree_reg, X_train_prepared, Y_train,
                         scoring='r2', cv=10)
print_scores(scores)

forest_reg = RandomForestRegressor()
forest_reg.fit(X_train_prepared, Y_train)
print('\nRandom forest score on train set:',
      forest_reg.score(X_train_prepared, Y_train))
scores = cross_val_score(forest_reg, X_train_prepared, Y_train,
                         scoring='r2', cv=10)
print_scores(scores)

# save model to disk
joblib.dump(forest_reg, 'forest.pkl')
my_loaded_forest = joblib.load('forest.pkl')
print('\nUnpickled model has same score as pre-pickled model:',
      forest_reg.oob_score == my_loaded_forest.oob_score)

# use grid search to find best hyper-parameters
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_prepared, Y_train)
print('\nGrid Search score on train set:', grid_search.best_score_)
print('Best params:', grid_search.best_params_)
print('Best estimator:', grid_search.best_estimator_)
# print('All results:', grid_search.cv_results_)

# which features are most important?
feature_importances = grid_search.best_estimator_.feature_importances_
# output of Pipeline() is a numpy array
# (doesn't have column names like DataFrame does)
attributes = list(X_train_num) + \
    ['rooms_per_household',
     'population_per_household',
     'bedrooms_per_room'] + list(encoder.classes_)
print('\nFeature importances:', sorted(zip(feature_importances, attributes),
      reverse=True))

# see how model works against unseen (test) set
best_model = grid_search.best_estimator_
X_test = strat_test_set.drop('median_house_value', axis=1)
Y_test = strat_test_set['median_house_value'].copy()

# do not use fit_tranform() here as we want to use the
# scaler values from the train set, not the test set
# i.e. don't call fit on test data
X_test_prepared = full_pipeline.transform(X_test)
best_score = best_model.score(X_test_prepared, Y_test)
print('The best score we can obtain against test data is:', best_score)
