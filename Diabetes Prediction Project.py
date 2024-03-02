#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv('diabetes.csv')


# In[3]:


df.head()


# In[5]:


df.shape #row,col


# In[7]:


df.describe()


# In[8]:


df['Outcome'].value_counts()


# ## 0- non diabetic
# ## 1- diabetic

# In[10]:


x=df.drop(columns='Outcome',axis=1)
y=df['Outcome']


# In[11]:


x


# In[12]:


y


# In[17]:


scaler=StandardScaler()


# In[18]:


scaler.fit(x)


# In[19]:


std_scaler=scaler.transform(x)


# In[21]:


std_scaler


# In[24]:


x=std_scaler


# In[33]:


x


# In[26]:


y


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[29]:


x_train.shape #80% data for training


# In[30]:


x_test.shape #20% data for testing


# # Train the model

# In[34]:


clf=svm.SVC(kernel='linear')


# In[35]:


clf.fit(x_train,y_train)


# In[36]:


x_train_prediction=clf.predict(x_train)
accuracy_score(x_train_prediction,y_train)


# # Accuracy of test data

# In[38]:


x_test_prediction=clf.predict(x_test)
accuracy_score(x_test_prediction,y_test)


# In[39]:


input_sample=(5,166,72,19,175,22.7,0.6,51)


# In[42]:


input_np_array=np.asarray(input_sample)


# In[45]:


input_np_array_reshaped=input_np_array.reshape(1,-1)


# In[46]:


std_data=scaler.transform(input_np_array_reshaped)


# In[47]:


std_data


# In[50]:


prediction=clf.predict(std_data)


# In[52]:


if (prediction[0]==0):
    print("Person is not diabetic")
else:
    print("Person is diabetic")


# In[ ]:




