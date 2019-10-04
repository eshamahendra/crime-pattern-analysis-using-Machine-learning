import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('C:/Users/Lenovo/Desktop/01_District_wise_crimes_committed_IPC_2001_2012.csv')
df.shape
df.describe()
y=df.iloc[:,0].values

X=df.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]].values
df.isnull().values.any()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
y=le.fit_transform(y)
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
from sklearn import metrics
a=metrics.accuracy_score(y_test, y_pred)
