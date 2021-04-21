#!/usr/bin/env python
# coding: utf-8

# # AUTHOR :- KISHAN ZARU

# # GRIP@The Sparks Foundation

# # Task 6 - Prediction using Decision Tree Algorithm
# In this task we create the Decision Tree classifier and visualize it graphically.
# The Main purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# In[22]:



#importing all the essential libraries
import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree


# In[2]:


df = pd.read_csv("Iris.csv")
df.head()


# In[3]:


df.drop('Id',1,inplace=True) ###We don't need this column


# In[4]:


#Checking the information of the dataframe
df.info()


# In[5]:


#checking the shape
df.shape


# In[6]:



#Chceking the null values
df.isnull().sum()


# # Let's check the correlation of all the features

# In[7]:


sns.pairplot(data=df,hue ='Species')
plt.show()


# In[8]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# # From the above two plots it is very much clear that PetalLength and PetalWidth are highly correlated and species 'Iris-Setosa' always form a different cluster .
# Now we will separate the target(y) and features(x) from the dataset.

# In[9]:


y = df['Species']
X = df.drop('Species',1)


# In[10]:


# Since , the target variable is object ,so we need to use labelencoder . 
le = LabelEncoder() #instantiate
y = le.fit_transform(y)
y


# So,we get 'Iris-setosa' as 0,'Iris-Versicolor' as 1 and 'Iris-Virginica' as 2

# In[11]:


#Train-Test split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=3)


# In[12]:


X_train.shape


# In[13]:



X_test.shape


# In[14]:


y_train.shape


# In[15]:


y_test.shape


# # Modelling tree and testing it

# In[16]:



dtree=DecisionTreeClassifier() #instantiate
dtree.fit(X_train,y_train)


# In[18]:


# predicting the values of test data
y_pred = dtree.predict(X_test)


# In[19]:


print(classification_report(y_test,y_pred))


# In[20]:


cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,annot=True,cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[21]:


plt.figure(figsize=(15,8))
plot_tree(decision_tree = dtree,feature_names = X_train.columns,filled=True,class_names =["setosa", "versicolor", "verginica"])

