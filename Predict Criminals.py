
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np


# In[42]:


train = pd.read_csv('criminal_train.csv')
#train.head()


# In[43]:


test = pd.read_csv('criminal_test.csv')
#test.head()


# In[ ]:


person_id = test['PERID']
del train['PERID']
del test['PERID']


# In[44]:


y = train['Criminal'].values
del train['Criminal']


# In[45]:


X_train = train.values
X_test = test.values


# In[46]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[47]:


X_train.shape


# In[48]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[49]:


#import matplotlib.pyplot as plt


# In[52]:


##plt.scatter(X_train,y)


# In[53]:

import time

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
classifier = SVC(kernel = 'rbf')
parameters = [
            # {'C': [1,10,100,1000], 'kernel': ['linear']},
            {'C': [1,10,100,1000], 'kernel': ['rbf'], 'gamma': [0.5,0.1,0.01,0.0001]}
            # {'C': [1,10,1000], 'kernel': ['poly'], 'degree':[2,3], 'gamma': [0.5,0.1,0.01] }
            # {'C': [1,10,1000], 'kernel': ['sigmoid'],'coef0':[0.5,0.1,0.01,0.0001],'gamma': [0.5,0.1,0.01,0.0001]}
            ]


grid_search = GridSearchCV(estimator= classifier, 
                          param_grid= parameters,
                          scoring= 'accuracy',
                          n_jobs= -1)

st_time = time.time()

grid_search.fit(X_train,y)

en_time = time.time()

print('Time taken :' , (en_time - st_time))
print('Best Parameters : ', grid_search.best_params_)
print('Best Score : ', grid_search.best_score_)
clf = grid_search.best_estimator_


# In[54]:


y_pred = clf.predict(X_test)


# In[55]:


def get_csv(y_pred):
    import csv
    ans = np.append(pd.DataFrame(person_id) ,pd.DataFrame(y_pred), axis = 1)
    filename = 'Criminals.csv'
    ans = list(ans)
    header = ['PERID','Criminal']
    with open(filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        csv_writer.writerows(ans)


# In[ ]:


get_csv(y_pred)

