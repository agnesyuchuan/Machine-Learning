#!/usr/bin/env python
# coding: utf-8

# ### Only take screen shots of streamlit code instead of machine_learning model

# In[5]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import metrics


# In[7]:


df=pd.read_csv('/Users/apple/Desktop/Machine_Learning/week_7/IMDB_movie_reviews_train.csv')
# /Users/apple/Desktop/Machine_Learning/week_7


# In[8]:


df.shape


# In[9]:


df.head(5)


# In[12]:


df.isna().sum()


# In[13]:


df.sentiment.value_counts()


# In[19]:


X=df.loc[:,['review']]
y=df.sentiment


# In[30]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y)


# In[31]:


y_train.value_counts()


# In[32]:


X_train_docs=[doc for doc in X_train.review]


# In[33]:


vect=CountVectorizer(ngram_range=(1,3),stop_words='english',max_features=1000).fit(X_train_docs)


# In[34]:


X_train_features=vect.transform(X_train_docs)


# In[35]:


print('X_train_features:\n{}'.format(repr(X_train_features)))


# In[36]:


feature_names=vect.get_feature_names()


# In[37]:


print("Number of features:{}".format(len(feature_names)))
print("First 100 features:\n{}".format(feature_names[:100]))
print("Every 100th feature:\n{}".format(feature_names[::100]))


# In[38]:


lin_svc=LinearSVC(max_iter=120000)


# In[40]:


scores=cross_val_score(lin_svc, X_train_features, y_train, cv=5)
print("Mean cross-validation accuracy:{:.2f}".format(np.mean(scores)))


# In[41]:


lin_svc.fit(X_train_features, y_train)


# In[43]:


X_test_docs=[doc for doc in X_test.review]
X_test_features=vect.transform(X_test_docs)


# In[44]:


y_test_pred=lin_svc.predict(X_test_features)


# In[45]:


metrics.accuracy_score(y_test, y_test_pred)


# In[46]:


import pickle


# In[47]:


pickle.dump(lin_svc,open('linear_svc_model','wb'))


# In[48]:


lin_svc=pickle.load(open('linear_svc_model','rb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




