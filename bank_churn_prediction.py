# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import seaborn as sb

data = pd.read_csv("Churn_Modelling.csv")

#print(data.head())

#print(data.shape)

#print(data.describe())

#print(data.info())

update_data = data.drop(['RowNumber','CustomerId','Surname'],axis=1)

#print(update_data.head())

update_data = update_data.drop(['Geography','Gender'], axis=1)

Geography = pd.get_dummies(data.Geography)
Gender = pd.get_dummies(data.Gender)

# print(Geography)
# print(Gender)

update_data = pd.concat([update_data,Geography,Gender], axis=1)

# print(update_data.head())

churned = update_data[update_data['Exited']==1]['Exited'].count()
not_churned = update_data[update_data['Exited']==0]['Exited'].count()

# print("Number of people that churned: ",churned)
# print("Number of people that not chhurned: ",not_churned)

labels=[0,1]
# plt.bar(labels[1], churned, width=0.1,color="red")
# plt.bar(labels[0], not_churned, width=0.1, color="blue")

nr_male = data[data['Gender']=='Male']['Gender'].count()
nr_female = data[data['Gender']=='Female']['Gender'].count()
# print(nr_male,nr_female)
# fig,aix = plt.subplots(figsize=(20,16))
# aix = sb.countplot(hue='Exited', y='Geography',data=data)

# age_customer = np.array(update_data['Age'])
# figure,axis = plt.subplots(figsize = (20,16))
# axis = sb.distplot(age_customer,kde=False,bins=200)

# new_plot = sb.FacetGrid(update_data,hue= 'Exited', height = 10)
# (new_plot.map(plt.hist, 'Age',edgecolor='w').add_legend())

X = update_data.drop(['Exited'],axis=1)
y = update_data['Exited']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

classifier = RandomForestClassifier(n_estimators = 100, random_state = 42)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# print(classification_report(y_test,predictions))
# print(accuracy_score(y_test, predictions))

logistic_regression = LogisticRegression(random_state = 42)
logistic_regression.fit(X_train, y_train)

y_predict = logistic_regression.predict(X_test)

print(classification_report(y_test, y_predict))
print(accuracy_score(y_test,y_predict))
