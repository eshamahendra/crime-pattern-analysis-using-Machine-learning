
import pandas as pd




import numpy as np
import matplotlib as plt




df=pd.read_csv("C:/Users/Lenovo/Desktop/01_District_wise_crimes_committed_IPC_2001_2012.csv")




X=df.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]].values
y=df.iloc[:,[0]].values




df.isnull().values.any()




from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)





#from sklearn.preprocessing import OneHotEncoder
#oh=OneHotEncoder(categorical_features=[0])
#y=oh.fit_transform(y).toarray()








from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)




from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=500, random_state=300)




dtree.fit(X_train,y_train)




y_pred=dtree.predict(X_test)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test, y_pred)



dtree.score(X_test,y_test)



dtree.score(X_train,y_train)



y_pred
