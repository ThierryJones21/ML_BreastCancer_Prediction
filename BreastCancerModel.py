#importing the libraries
#import 'wdbcBreastCancer.csv'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#importing our cancer dataset
breastcancerdata = pd.read_csv("C:/Users/jones/OneDrive - Queen's University/3rd Year/ELEC 390/wdbcBreastCancer.csv")
#Summarizing our data
summary = breastcancerdata.describe()
summary = summary.transpose()
summary

breastcancerdata.drop(breastcancerdata.columns[[-1, 0]], axis=1, inplace=True)
breastcancerdata.info()

#Correlation plot
breastcancerdata_corr = breastcancerdata.corr()
breastcancerdata_corr
f,ax = plt. subplots(figsize = (10,10))
#HeatMap
sns.heatmap(breastcancerdata_corr)

featureMeans = list(breastcancerdata.columns[1:10])
correlationData = breastcancerdata[featureMeans].corr()
sns.pairplot(breastcancerdata[featureMeans].corr(), diag_kind='kde', size=2)

#Assigning x and y data to train our models
x = breastcancerdata.drop("Diagnosis", axis = 1)
y = breastcancerdata.Diagnosis

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.7)

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
print(vif.round(1))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(x)
#Logistical regression model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)
log_pred = log_reg.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("confusion_matrix :")
print(confusion_matrix(y_test, log_pred))
print("Top Left is accurate predictions, bottom left is false positives")
print("classification_report :")
print(classification_report(y_test, log_pred))
print("acc_score :")
print(accuracy_score(y_test, log_pred))

#Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
dtree_pred = dtree.predict(x_test)
print("confusion_matrix :")
print(confusion_matrix(y_test, dtree_pred))
print("classification_report :")
print(classification_report(y_test, dtree_pred))
print("acc_score :")
print(accuracy_score(y_test, dtree_pred))

#Random Forrest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
print("confusion_matrix :")
print(confusion_matrix(y_test, rf_pred))
print("classification_report :")
print(classification_report(y_test, rf_pred))
print("acc_score :")
print(accuracy_score(y_test, rf_pred))