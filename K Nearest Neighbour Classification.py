# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 09:52:50 2020

@author: Shrikant Agrawal
"""

"""K Nearest Neighbors with Python
You've been given a classified data set from a company!
They've hidden the feature column names but have given you the data and the target classes.

We'll try to use KNN to create a model that directly predicts a class for a new data point based off of the features.

"""

# Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Get the Data
#Set index_col=0 to use the first column as the index.

df = pd.read_csv("Classified Data",index_col=0)

# Standardize the Variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

# Draw pair plot to decide wheather to use K-NN or Logistic Regression
import seaborn as sns
sns.pairplot(df,hue='TARGET CLASS')

#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)

#Using KNN
""" we are trying to come up with a model to predict whether someone will 
TARGET CLASS or not. We'll start with k=1.
As the K values increases error rate will go down initially and then it will start increasing
as K value increases more.
Here we have considered K value as 1 and predict the accuracy socre"""

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

# Predictions and Evaluations of the Model
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
 
#Pecision, recall and f1 scores are accuracy which we have already discussed in earlier program

# Choosing a K Value using error_rate. we can do it ussing accuracy scores also
#use the elbow method to pick a good K Value

error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

""" We can do above same steps using accuracy rate also
accuracy_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,df_feat,df['TARGET CLASS'],cv=10)
    accuracy_rate.append(score.mean()) """

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# Here we can see that that after arouns K>23 the error rate just tends to
# hover around 0.06-0.05 Let's retrain the model with that and check the classification report!

# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

# NOW WITH K=23 Acuuracy rate increased
knn = KNeighborsClassifier(n_neighbors=23)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))








