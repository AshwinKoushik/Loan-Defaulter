# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 21:11:00 2020

@author: ASUS
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pickle
%matplotlib inline

mydata = pd.read_csv("E:\\Data Science\\Project\\Bank Default\\bank_final.csv")
type(mydata) # checking for dataframe
mydata.head(10)
mydata.tail(10)
mydata.shape # Gives the rows and columns of the data
mydata.dtypes # Checking data types
mydata.columns #It finds columns of data
mydata.describe() #Summary of the data
mydata.isnull().sum() # finding NA values

data = mydata.sample(frac=0.1,replace=True, random_state=0) # Random sampling
data.head()
data.tail()
data.shape # Gives the rows & columns of the data
c = data.corr()
sns.heatmap(c,annot=True)
sns.pairplot(data)
data.columns
data.dtypes

sns.countplot(data["Name"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title='defaulters',labels=['defaulters','Nondefaulters'])

sns.countplot(data["ApprovalDate"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"]) # Not having much impact on the model

sns.countplot(data["City"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"]) # Can be dropped

sns.countplot(data["State"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"])

sns.countplot(data["Bank"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"]) # Can be dropped

sns.countplot(data["BankState"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"])

sns.countplot(data["RevLineCr"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"])

sns.countplot(data["LowDoc"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"])

sns.countplot(data["ChgOffDate"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"])

sns.countplot(data["DisbursementDate"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"]) # Can be dropped

sns.countplot(data["DisbursementGross"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"]) # Can be dropped

sns.countplot(data["BalanceGross"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"])

sns.countplot(data["ChgOffPrinGr"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"])

sns.countplot(data["GrAppv"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"])

sns.countplot(data["SBA_Appv"],data=data)
plt.subplots_adjust(wspace=0.5)
plt.legend(title="defaulters",labels=["defaulters","Nondefaulters"])

# Finding crosstab
pd.crosstab(data.ApprovalDate,data.MIS_Status,normalize='index')

bank =mydata.drop(["Name","ApprovalDate","City","Bank","BankState","CCSC","Zip","DisbursementDate","ChgOffPrinGr","ChgOffDate","GrAppv","SBA_Appv"],axis = 1)
bank.shape
bank.dtypes
bank.isnull().sum()
bank['State'].mode()
bank['MIS_Status'].mode()
bank['RevLineCr'].mode()

value_mode = {'State':"CA",'MIS_Status':"P I F",'RevLineCr':"N"} # Creating Dict by giving mode
banking = bank.fillna(value = value_mode) # Filling NA values
banking.shape
banking.dtypes
banking.isnull().sum()

# Converting MIS status to numericals
target_name = banking['MIS_Status']
target = []

for i in target_name:
    if i == "CHGOFF":
        target.append(0)
    else:
        target.append(1)
        

# Converting RevLineCr to numericals

target_Rev = banking['RevLineCr']
revlinecr = []

for j in target_Rev:
    j = str(j).strip()
    if j == "N":
        revlinecr.append(0)
    else:
        revlinecr.append(1)

# Converting LowDoc to numericals
        
targetLow = banking['LowDoc']
lowdoc = []    

for k in targetLow:
    if k == "N":
        lowdoc.append(0)
    else:
        lowdoc.append(1)

# Assigning the above to the dataframe

banking = banking.drop(["MIS_Status","RevLineCr","LowDoc"],axis=1)
banking.columns
banking.dtypes

 banking['RevLineCr']=revlinecr 
 banking['LowDoc']=lowdoc 
 banking['target']=target
 banking.head(10)      

# Converting strings to float
banking['BalanceGross']=banking['BalanceGross'].str.replace('\W','').astype(float)
banking['DisbursementGross']=banking['DisbursementGross'].str.replace('\W','').astype(float)

categorical = ['State']
from sklearn import preprocessing
for i in categorical:
    num = preprocessing.LabelEncoder()
    banking[i]= num.fit_transform(banking[i])
    
# cat = ['BankState']
# for j in cat:
#     num = preprocessing.LabelEncoder()          ## Deleted BankState Column
#     banking[j] = num.fit_transform(banking[j])

c = banking.corr()
sns.heatmap(c,annot=True)

# Model Building
## KNN Model

from sklearn.model_selection import train_test_split
train,test = train_test_split(banking,test_size=0.3,random_state=100)

from sklearn.neighbors import KNeighborsClassifier as KNC # Importing Knn algorithm from sklearn.neighbors

model_knn = KNC(n_neighbors = 5) # k=5
model_knn.fit(train.iloc[:,0:13],train.iloc[:,13])

train_acc = np.mean(model_knn.predict(train.iloc[:,0:13])==train.iloc[:,13]) # Train Accuracy
train_acc # 84.32%

test_acc = np.mean(model_knn.predict(test.iloc[:,0:13])==test.iloc[:,13]) # Test Accuracy
test_acc # 77.13

# running KNN algorithm for 3 to 100 nearest neighbours(odd numbers) and 
# storing the accuracy values 

acc =[] # Creating Empty list 
for i in range(3,100,2):
    model = KNC(n_neighbors = i)
    model.fit(train.iloc[:,0:13],train.iloc[:,13])
    train_acc = np.mean(model.predict(train.iloc[:,0:13])==train.iloc[:,13])
    test_acc = np.mean(model.predict(test.iloc[:,0:13])==test.iloc[:,13])
    acc.append([train_acc,test_acc])
    
import matplotlib.pyplot as plt # library to do visualizations 

plt.plot(np.arange(3,100,2),[i[0] for i in acc],"bo-") # Train Accuracy
plt.plot(np.arange(3,100,2),[i[1] for i in acc],"ro-") # Test Accuracy

## K= 5 is giving the best accuracy

## Decision tree
from sklearn.tree import DecisionTreeClassifier
colnames = list(banking.columns)
predictors = colnames[0:13]
target = colnames[13]

model = DecisionTreeClassifier(criterion = "entropy")
model.fit(train[predictors],train[target])

train_acc = np.mean(model.predict(train[predictors])==train[target])
train_acc # 99.99%

test_acc = np.mean(model.predict(test[predictors])==test[target])
test_acc # 90.64%

## Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs = 5,oob_score = True,n_estimators = 100,criterion = "entropy")
rf.fit(banking[predictors],banking[target])
rf.estimators_
rf.classes_ # class labels (output)
rf.n_classes_  # Number of levels in class labels 
rf.n_features_ # Number of input features in model
rf.n_outputs_ # Number of outputs when fit performed
rf.oob_score_ # 93.27%, 93.91%

train['target_predict'] = rf.predict(train[predictors])

from sklearn.metrics import confusion_matrix 
confusion_matrix(train['target'],train['target_predict'])
2697+7802
10499/10500 # 99.99% Train Accuracy
27347+77624
104971/104999 # 99.97  150000 data

test['target_predict'] = rf.predict(test[predictors])
confusion_matrix(test['target'],test['target_predict'])
1203+3297
4500/4500 # 100% Test Accuracy
11624+33359
44983/45000 # 99.96%  150000 data

pickle.dump(rf,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

# I have considered Random forest model as it gives the best accuracy.