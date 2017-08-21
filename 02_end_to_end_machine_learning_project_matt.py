import tarfile
import requests
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import Imputer


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
# (why don't we keep the newly created features?)
X_train = strat_train_set.drop('median_house_value', axis=1)
Y_train = strat_train_set['median_house_value'].copy()

# clean data

# remember total_bedrooms has empty values as per DF.info()
print('\nNumber of missing total_bedroom values:',
      data['total_bedrooms'].isnull().sum())


