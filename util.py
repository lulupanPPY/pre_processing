#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import pandas as pd


#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser

data_raw = pd.read_csv('D://Kaggle//Titanic//train.csv')
data_val  = pd.read_csv('D://Kaggle//Titanic//test.csv')
data1 = data_raw.copy()
data_cleaner = [data1, data_val]
print (data_raw.info())


# 检查null value
print('Train columns with null values:\n', data1.isnull().sum())
print("-"*10)

# 打印特征统计值 计数，种类，排名，均值，标准差，最小，25%分位点，50%分位点，75%分位点，最大
print (data_raw.describe(include = 'all'))

# select distinct
print ("Cabin mode \n",data_raw['Cabin'].mode())

#NA填充
for dataset in data_cleaner:
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)


# 删除特征
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)