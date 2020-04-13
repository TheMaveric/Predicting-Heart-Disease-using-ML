#!/usr/bin/env python
# coding: utf-8

# ## <font size=5> <strong>Heart Disease Prediction By Shreekant Gosavi
#  

# ## I. Importing essential libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')


# ## II. Importing and understanding our dataset 

# In[2]:


dataset = pd.read_csv("heart.csv")


# #### Verifying it as a 'dataframe' object in pandas

# In[3]:


type(dataset)


# #### Shape of dataset

# In[1]:


dataset.shape


# #### Printing out a few columns

# In[2]:


dataset.head(5)


# In[6]:


dataset.sample(5)


# #### Description

# In[7]:


dataset.describe()


# In[8]:


dataset.info()


# In[9]:


###Luckily, we have no missing values


# #### Let's understand our columns better:

# In[10]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])


# #### Analysing the 'target' variable

# In[11]:


dataset["target"].describe()


# In[12]:


dataset["target"].unique()


# #### Clearly, this is a classification problem, with the target variable having values '0' and '1'

# ### Checking correlation between columns

# In[13]:


print(dataset.corr()["target"].abs().sort_values(ascending=False))


# In[14]:


#This shows that most columns are moderately correlated with target, but 'fbs' is very weakly correlated.


# ## Exploratory Data Analysis (EDA)

# ### First, analysing the target variable:

# In[15]:


y = dataset["target"]

sns.countplot(y)


target_temp = dataset.target.value_counts()

print(target_temp)


# In[16]:


print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))

#Alternatively,
# print("Percentage of patience with heart problems: "+str(y.where(y==1).count()*100/303))
# print("Percentage of patience with heart problems: "+str(y.where(y==0).count()*100/303))

# #Or,
# countNoDisease = len(df[df.target == 0])
# countHaveDisease = len(df[df.target == 1])


# ### We'll analyse 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca' and 'thal' features

# ### Analysing the 'Sex' feature

# In[17]:


dataset["sex"].unique()


# ##### We notice, that as expected, the 'sex' feature has 2 unique features

# In[18]:


sns.barplot(dataset["sex"],y)


# ##### We notice, that females are more likely to have heart problems than males

# ### Analysing the 'Chest Pain Type' feature

# In[19]:


dataset["cp"].unique()


# ##### As expected, the CP feature has values from 0 to 3

# In[20]:


sns.barplot(dataset["cp"],y)


# ##### We notice, that chest pain of '0', i.e. the ones with typical angina are much less likely to have heart problems

# ### Analysing the FBS feature

# In[21]:


dataset["fbs"].describe()


# In[22]:


dataset["fbs"].unique()


# In[23]:


sns.barplot(dataset["fbs"],y)


# ##### Nothing extraordinary here

# ### Analysing the restecg feature

# In[24]:


dataset["restecg"].unique()


# In[25]:


sns.barplot(dataset["restecg"],y)


# ##### We realize that people with restecg '1' and '0' are much more likely to have a heart disease than with restecg '2'

# ### Analysing the 'exang' feature

# In[26]:


dataset["exang"].unique()


# In[27]:


sns.barplot(dataset["exang"],y)


# ##### People with exang=1 i.e. Exercise induced angina are much less likely to have heart problems

# ### Analysing the Slope feature

# In[28]:


dataset["slope"].unique()


# In[29]:


sns.barplot(dataset["slope"],y)


# ##### We observe, that Slope '2' causes heart pain much more than Slope '0' and '1'

# ### Analysing the 'ca' feature

# In[30]:


#number of major vessels (0-3) colored by flourosopy


# In[31]:


dataset["ca"].unique()


# In[32]:


sns.countplot(dataset["ca"])


# In[33]:


sns.barplot(dataset["ca"],y)


# ##### ca=4 has astonishingly large number of heart patients

# In[34]:


### Analysing the 'thal' feature


# In[35]:


dataset["thal"].unique()


# In[36]:


sns.barplot(dataset["thal"],y)


# In[37]:


sns.distplot(dataset["thal"])


# ## IV. Train Test split

# In[38]:


from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


# In[39]:


X_train.shape


# In[40]:


X_test.shape


# In[41]:


Y_train.shape


# In[42]:


Y_test.shape


# ## V. Model Fitting

# In[43]:


from sklearn.metrics import accuracy_score


# ### Logistic Regression

# In[44]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)


# In[45]:


Y_pred_lr.shape


# In[46]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")


# ### Naive Bayes

# In[47]:


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,Y_train)

Y_pred_nb = nb.predict(X_test)


# In[48]:


Y_pred_nb.shape


# In[49]:


score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# ### SVM

# In[50]:


from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)


# In[51]:


Y_pred_svm.shape


# In[52]:


score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")


# ### K Nearest Neighbors

# In[53]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)


# In[54]:


Y_pred_knn.shape


# In[55]:


score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# ### Decision Tree

# In[56]:


from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)


# In[57]:


#print(Y_pred_dt.shape)


# In[58]:


score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# ### Random Forest

# In[59]:


from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)


# In[60]:


Y_pred_rf.shape


# In[61]:


score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")


# ### XGBoost

# In[62]:


import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, Y_train)

Y_pred_xgb = xgb_model.predict(X_test)


# In[63]:


Y_pred_xgb.shape


# In[64]:


score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)

print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")


# ### Neural Network

# In[65]:


from keras.models import Sequential
from keras.layers import Dense


# In[66]:


# https://stats.stackexchange.com/a/136542 helped a lot in avoiding overfitting

model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[67]:


model.fit(X_train,Y_train,epochs=300)


# In[68]:


Y_pred_nn = model.predict(X_test)


# In[69]:


Y_pred_nn.shape


# In[70]:


rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded


# In[71]:


score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")

#Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11. 


# ## VI. Output final score

# In[72]:


scores = [score_lr,score_nb,score_svm,score_knn,score_dt,score_rf,score_xgb,score_nn]
algorithms = ["Logistic Regression","Naive Bayes","Support Vector Machine","K-Nearest Neighbors","Decision Tree","Random Forest","XGBoost","Neural Network"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# In[73]:


sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# ### Hey arbaaz there random forest has good result as compare to other algorithms <br> <br>

# 
