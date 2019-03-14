# importing the necessary modules to visualise
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
warnings.filterwarnings("ignore")
plt.show()
"""
this will load in the files to train the program, to test the program
and to have a look at some basic mathematical information about the SalePrice
of housing
"""
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.columns
train['SalePrice'].describe()

# drop the id column from the data as they do not have any correlation with the data whatsoever
trainID = train["Id"]
testID = test["Id"]
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
"""
# graph to look for SalePrice distribution, skewness and its peak value
sns.distplot(train['SalePrice'])
print(f"Skewness of graph: {train['SalePrice'].skew()}")
print(f"Kurtosis (peak) of graph: {train['SalePrice'].kurt()}")
# scatter plot of grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# scatter plot of totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# box plot of overallqual/saleprice
var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
# box plot of YearBuilt/SalePrice
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90)

# correlation matrix of all attributes
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.9, square=True)
plt.show()

# saleprice correlation matrix
n = 20 #number of variables for heatmap
columns = corrmat.nlargest(n, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[columns].values.T)
sns.set(font_scale=0.85)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=columns.values, xticklabels=columns.values)
plt.show()

# scatterplot
sns.set()
columns = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[columns], size = 2.5)
plt.show()

# finding missing data within the data set
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# dealing with the missing data to allow for better prediction of SalePrice
train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
train.isnull().sum().max() #just checking that there's no missing data missing..

# standardizing the data so that we can predict the data much more accurately
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis])
@@ -95,11 +86,6 @@
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# deleting points that are outliers in the data
train.sort_values(by = 'GrLivArea', ascending = False)[:2]
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)

# bivariate analysis of saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
# histogram and normal probability plot of SalePrice
sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
# applying log transformation
train['SalePrice'] = np.log(train['SalePrice'])
# transformed histogram and normal probability plot
sns.distplot(train['SalePrice'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
# histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)
plt.show()
# data transformation
train['GrLivArea'] = np.log(train['GrLivArea'])
# transformed histogram and normal probability plot
sns.distplot(train['GrLivArea'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)
# histogram and normal probability plot
sns.distplot(train['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)
"""
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
# create column for new variable (one is enough because it's a binary categorical feature)
# if area>0 it gets 1, for area==0 it gets 0
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0 
train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1
"""
#transform data
# transform data
train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

#histogram and normal probability plot
# histogram and normal probability plot
sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

#scatter plot
# scatter plots
plt.scatter(train['GrLivArea'], train['SalePrice'])

#scatter plot
plt.scatter(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], train[train['TotalBsmtSF']>0]['SalePrice']);
"""
#convert categorical variable into dummy
train = pd.get_dummies(train)

# concatenating both training and test data into one dataframe and removing SalePrice
allData = pd.concat((train, test)).reset_index(drop = True)
allData.drop(["SalePrice"], axis = 1,inplace = True)

# imputing values in the dataset to deal with missing values or categorical features that are hard to predict
allData["HasBsmt"] = allData["HasBsmt"].fillna("None")
allData["FireplaceQu"] = allData["FireplaceQu"].fillna("None")
allData["PoolQC"] = allData["PoolQC"].fillna("None")
allData["Fence"] = allData["Fence"].fillna("None")
allData["MiscFeature"] = allData["MiscFeature"].fillna("None")
allData["Alley"] = allData["Alley"].fillna("None")
#allData["LotFrontage"] = allData.groupby("Neighborhood")["LotFrontage"].transform(
    #lambda x: x.fillna(x.median()))
for i in ("GarageType", "GarageFinish", "GarageQual", "GarageCond"):
    allData[i] = allData[i].fillna('None')
for i in ("GarageYrBlt", "GarageArea", "GarageCars"):
    allData[i] = allData[i].fillna(0)
for i in ("BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"):
    allData[i] = allData[i].fillna(0)
for i in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
    allData[i] = allData[i].fillna('None')
allData['MSZoning'] = allData['MSZoning'].fillna(allData['MSZoning'].mode()[0])
allData["MasVnrType"] = allData["MasVnrType"].fillna("None")
allData["MasVnrArea"] = allData["MasVnrArea"].fillna(0)
allData["MSSubClass"] = allData["MSSubClass"].fillna("None")
allData["Functional"] = allData["Functional"].fillna("Typ")

# For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
allData = allData.drop(["Utilities"], axis=1)

# There is one missing value from these categories so we can will them with the most frequent value
allData["SaleType"] = allData["SaleType"].fillna(allData["SaleType"].mode()[0])
allData["KitchenQual"] = allData["KitchenQual"].fillna(allData["KitchenQual"].mode()[0])
allData["Electrical"] = allData["Electrical"].fillna(allData["Electrical"].mode()[0])
allData["Exterior1st"] = allData["Exterior1st"].fillna(allData["Exterior1st"].mode()[0])
allData["Exterior2nd"] = allData["Exterior2nd"].fillna(allData["Exterior2nd"].mode()[0])

# checking for any remaining data that is missing
allDataNull = (allData.isnull().sum() / len(allData)) * 100
allDataNull = allDataNull.drop(allDataNull[allDataNull == 0]. index).sort_values(ascending = False)[:30]
missingData = pd.DataFrame({"Missing Ratio" :allDataNull})
print(missingData.head(20))

#MSSubClass=The building class
allData["MSSubClass"] = allData["MSSubClass"].apply(str)

#Changing OverallCond into a categorical variable
allData["OverallCond"] = allData["OverallCond"].astype(str)

#Year and month sold are transformed into categorical features.
allData["YrSold"] = allData["YrSold"].astype(str)
allData["MoSold"] = allData["MoSold"].astype(str)

# label encoding categorical variables that may contain information in their order set
cols = ("FireplaceQu", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", 
        "ExterQual", "ExterCond","HeatingQC", "PoolQC", "KitchenQual", "BsmtFinType1", 
        "BsmtFinType2", "Functional", "Fence", "BsmtExposure", "GarageFinish", "LandSlope",
        "LotShape", "PavedDrive", "Street", "Alley", "CentralAir", "MSSubClass", "OverallCond", 
        "YrSold", "MoSold")

# process columns, apply LabelEncoder to categorical features
for i in cols:
    label = LabelEncoder() 
    label.fit(list(allData[i].values)) 
    allData[i] = label.transform(list(allData[i].values))

# the target variable that we want to predict
targetVar = train["SalePrice"]

# getting dummy categorical features
allData = pd.get_dummies(allData)

# normalising the data
features = list(set(list(allData.columns)) - set(targetVar))
allData[features] = allData[features] / allData[features].max() # <---- current error

# creating a validation set
from sklearn.model_selection import train_test_split
x = train[features].values
y = train[targetVar].values
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.3, random_state = 42)

# building a model using Random Forest
import random
import keras

# defining the model for the deep learning model
model = Sequential()
model.add(Dense(100, input_dim = 79, activation = "relu"))
model.add(Dense(1))
model.summary()

# compile the model
model.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["mean_squared_error"])

# evaluation
pred = model.predict(x_valid)
score = np.sqrt(mean_squared_error(y_valid, pred))
print(score)

#prediction
x_test = test[features].values
y_test = model.predict(x_test)
