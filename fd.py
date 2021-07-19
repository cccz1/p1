#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


df = pd.read_csv('qt_dataset.csv')

# give columns friendlier names
data = pd.read_csv('qt_dataset.csv')
data.columns = ['Patient ID', 'Oxygen_Level', 'Pulse', 'Temperature', 'Outcome']
data['Outcome'].replace(['Negative','Positive'],[0, 1],inplace=True)

data_neg = data[data['Outcome']==0]
data_pos = data[data['Outcome']==1]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(data_neg['Pulse'], data_neg['Oxygen_Level'], c='b', label='Negative')
ax1.scatter(data_pos['Pulse'], data_pos['Oxygen_Level'], c='r', label='Positive')
plt.legend(loc='upper left')
plt.xlabel('Pulse')
plt.ylabel('Oxygen_Level')
plt.show()
# Graph clearly shows how patients with a lower oxygen level are much more likely to test positive for COVID-19. 


# In[3]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(data[['Oxygen_Level']+['Pulse']+['Temperature']],data.Outcome, test_size=0.2)
lrg = sm.Logit(y_train, X_train).fit()
print (lrg.summary())


# In[4]:


pr = lrg.predict(X_test)
print(pr)


# In[5]:


from sklearn.metrics import confusion_matrix as cm
y_pred = np.where(pr> 0.5,1,0)
com = cm(y_test, y_pred)
print(com)


# In[6]:


ac = (com[1,1]+com[0,0])/2000
print('Accuracy:',ac)

ss = com[1,1]/(com[1,1]+com[1,0])
print('Sensitivity : ', ss )

sf = com[1,1]/(com[0,1]+com[0,0])
print('Specificity : ', sf)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




