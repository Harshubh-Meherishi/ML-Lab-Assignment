#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import sklearn
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve,precision_score,recall_score
import matplotlib.pyplot as plt


# ### Naming features

# In[16]:


columns = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over', 'word_freq_remove',
           'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people',
           'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you',
           'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl',
           'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data',
           'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
           'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu', 'word_freq_table',
           'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
           'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']
df = pd.read_csv('spambase.data', names=columns+['is_spam'], index_col=False)


# In[17]:


df.head()


# ### Cleaning data and dropping unrequired columns

# In[18]:


df=df.drop(columns=['capital_run_length_average','capital_run_length_longest','capital_run_length_total'])


# ### Splitting into train and test

# In[19]:


train,test=train_test_split(df,train_size=0.8,random_state=42)#,shuffle=True


# In[20]:


x_train=train.iloc[:,:-1]
y_train=train.iloc[:,-1]

x_test=test.iloc[:,:-1]
y_test=test.iloc[:,-1]


# In[21]:


x_train.shape


# In[22]:


y_train.shape


# In[23]:


x_test.shape


# In[24]:


y_test.shape


# In[25]:


x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)


# #### Importing svm library

# In[26]:


from sklearn import svm


# ## Radial Kernel
# Taking parameters gamma=0.01 C=10 Accuracy=93.91%

# #### Calculating training and test accuracy

# In[27]:


model=svm.SVC(kernel='rbf',gamma=0.01,C=10)
#g=0.001 C=100 A=0.9349240780911063
#g=0.001 C=1000 A=0.9414316702819957

model.fit(x_train,y_train)
print('Training accuracy: {}'.format(model.score(x_train,y_train)))

predicted=model.predict(x_test)
print('Testing accuracy: ',accuracy_score(y_test,predicted))


# #### Evaluating Algorithm 

# In[28]:


print(confusion_matrix(y_test,predicted))
print('Precision: ',precision_score(y_test,predicted))
print('Recall: ',recall_score(y_test,predicted))


# In[29]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=0)
# Print ROC curve
plt.plot(tpr,fpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("True Positive Rate Vs False Positive Rate")
plt.show()


# gamma=0.001 C=10000 Accuracy=94.49%

# #### Calculating training and test accuracy 

# In[30]:


model=svm.SVC(kernel='rbf',gamma=0.001,C=10000)

model.fit(x_train,y_train)
print('Training accuracy: {}'.format(model.score(x_train,y_train)))

predicted=model.predict(x_test)
print('Testing accuracy: ',accuracy_score(y_test,predicted))


# #### Evaluating algorithm

# In[31]:


print(confusion_matrix(y_test,predicted))
print('Precision: ',precision_score(y_test,predicted))
print('Recall: ',recall_score(y_test,predicted))


# In[32]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=0)
# Print ROC curve
plt.plot(tpr,fpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("True Positive Rate Vs False Positive Rate")
plt.show()


# ## Linear Kernel
# Taking parameters gamma=1 C=10 accuracy=91.74%

# #### Calculating training and test accuracy

# In[33]:


model=svm.SVC(C=10,gamma=1, kernel='linear')

model.fit(x_train,y_train)
print('Training accuracy: {}'.format(model.score(x_train,y_train)))

predicted=model.predict(x_test)
print('Testing accuracy: ',accuracy_score(y_test,predicted))


# #### Evaluating algorithm

# In[34]:


print(confusion_matrix(y_test,predicted))
print('Precision: ',precision_score(y_test,predicted))
print('Recall: ',recall_score(y_test,predicted))


# In[35]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=0)
# Print ROC curve
plt.plot(tpr,fpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("True Positive Rate Vs False Positive Rate")
plt.show()


# ### Using LinearSVC
# Taking parameters C=1000 Accuracy= 91.53%

# #### Calculating training and test accuracy

# In[36]:


model=svm.LinearSVC(dual=False,C=1000)

model.fit(x_train,y_train)
print('Training accuracy: {}'.format(model.score(x_train,y_train)))

predicted=model.predict(x_test)
print('Testing accuracy: ',accuracy_score(y_test,predicted))


# #### Evaluating algorithm

# In[37]:


print(confusion_matrix(y_test,predicted))
print('Precision: ',precision_score(y_test,predicted))
print('Recall: ',recall_score(y_test,predicted))


# In[38]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=0)
# Print ROC curve
plt.plot(tpr,fpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("True Positive Rate Vs False Positive Rate")
plt.show()


# ## Quadratic Kernel

# #### Claculating training and test accuracy

# In[39]:


model=svm.SVC(C=100,kernel='poly',degree=4,gamma=0.1)
#gamma=0.1 C=1000 A=0.9154013015184381
#C=100,kernel='poly',degree=4,gamma=0.1 A=0.8915401301518439

model.fit(x_train,y_train)
print('Training accuracy: {}'.format(model.score(x_train,y_train)))

predicted=model.predict(x_test)
print('Testing accuracy: ',accuracy_score(y_test,predicted))


# #### Evaluating algorithm

# In[40]:


print(confusion_matrix(y_test,predicted))
print('Precision: ',precision_score(y_test,predicted))
print('Recall: ',recall_score(y_test,predicted))


# In[41]:


from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted, pos_label=0)
# Print ROC curve
plt.plot(tpr,fpr)
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.title("True Positive Rate Vs False Positive Rate")
plt.show()


# ## Finding best parameters
# We have used below algorithm to claculate the best parameters for linear,rbf and radial kernel.

# In[42]:


from sklearn.svm import SVC
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['linear']}

clf = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
clf.fit(x_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_estimator_)


# In[54]:


param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf']}

clf = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
clf.fit(x_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_estimator_)


# In[14]:


from sklearn.svm import SVC
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['poly']}

clf = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
clf.fit(x_train, y_train)
print("Best parameters set found on development set:")
print(clf.best_estimator_)


# In[ ]:




