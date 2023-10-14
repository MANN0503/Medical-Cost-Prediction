#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Medical_cost_prediction.csv')


# In[3]:


df.head()


# In[23]:


df.describe()


# In[24]:


df.size


# In[25]:


df.shape


# In[26]:


df['label'].unique()


# In[4]:


features = df[['age', 'sex','bmi','children', 'smoker']]
target = df['label']
labels = df['label']


# In[5]:


# Initializing empty lists to append all model's name and corresponding name
acc = []
model = []


# In[6]:


# Splitting into train and test data

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)


# In[7]:


#############################################################
##########            DECISION TREE         #################
#############################################################


from sklearn.tree import DecisionTreeClassifier
DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)
DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)

print(classification_report(Ytest,predicted_values))


# In[8]:


from sklearn.model_selection import cross_val_score


# In[9]:


# Cross validation score (Decision Tree)
score = cross_val_score(DecisionTree, features, target,cv=5)
score


# In[10]:


#############################################################
##########               SVM                  ###############
#############################################################

from sklearn.svm import SVC
SVM = SVC(gamma='auto')
SVM.fit(Xtrain,Ytrain)
predicted_values = SVM.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[11]:


# Cross validation score (SVM)
score = cross_val_score(SVM,features,target,cv=5)
score


# In[12]:


#############################################################
##########       LOGISTIC REGRESSION        #################
#############################################################


from sklearn.linear_model import LogisticRegression
LogReg = LogisticRegression(random_state=2)
LogReg.fit(Xtrain,Ytrain)
predicted_values = LogReg.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[13]:


# Cross validation score (Logistic Regression)
score = cross_val_score(LogReg,features,target,cv=5)
score


# In[27]:


#############################################################
##########            RANDOM FOREST         #################
#############################################################



from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))


# In[17]:


# Cross validation score (Random Forest)
score = cross_val_score(RF,features,target,cv=5)
score


# In[18]:


import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()


# In[19]:


#############################################################
##########          ACCURACY COMPARISON       ###############
#############################################################



plt.figure(figsize=[10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')


# In[20]:


accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print (k, '-->', v)


# In[22]:


data = np.array([[24,male, 23.655, 0, no]])
prediction = RF.predict(data)
print(prediction)


# In[21]:


data = np.array([[24,'female', 23.655, 0, 'no']])
prediction = RF.predict(data)
print(prediction)
