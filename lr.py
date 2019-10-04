#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES AND DATASET

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:/Users/Lenovo/Desktop/43_Arrests_under_crime_against_women.csv")


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


y=df.iloc[:,[15]].values
X=df.iloc[:,[1]].values


# # DATA PREPROCESSING

# In[6]:


df.isnull().values.any()


# In[7]:









from sklearn.model_selection import train_test_split


# In[13]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


# In[14]:


X_train.shape


# In[15]:


X_test.shape


# ## IMPLEMENTING LINEAR REGRESSION MODEL

# In[16]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[17]:


reg=linear_model.LinearRegression()


# In[18]:


reg.fit(X_train, y_train)


# In[19]:


y_pred=reg.predict(X_test)


# In[20]:


acc=reg.score(X_test,y_test)
acc


# In[21]:


acc1=reg.score(X_test,y_pred)
acc1


# In[22]:


plt.scatter(X_test,y_pred, color='red')
plt.title('Regression plot')
plt.xlabel('Year')
plt.ylabel('Crime against women')


# In[ ]:





