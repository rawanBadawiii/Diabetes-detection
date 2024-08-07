import numpy as np
import pandas as pd
import seaborn as sns

import warnings

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn import metrics

from sklearn.preprocessing import StandardScaler

#import SVC classifier
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
#import metrics to compute accuracy
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

# Save Model Using joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib


from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


#from sklearn import cross_validation


#from sklearn.model_selection import cross_validate
#from sklearn.model_selection import cross_validation
#import pylab as pl


df = pdread_csv("C:\\Users\hp\Downloads\diabetes_binary_health_indicators_BRFSS2015.csv")

#################DATA CLEANING############################
print("mean:")
physical_mean=df["PhysHlth"].mean()
BMI_mean=df["BMI"].mean()
Gen_mean=df["GenHlth"].mean(
Men_mean=df["MentHlth"].mean(
df["PhysHlth"]=df.PhysHlth.mask(df.PhysHlth==0,physical_mean)
df["BMI"]=df.BMI.mask(df.BMI==0,BMI_mean)
df["GenHlth"]=df.GenHlth.mask(df.GenHlth==0,Gen_mean)
df["MentHlth"]=df.MentHlth.mask(df.MentHlth==0,Men_mean)


df['Diabetes_binary'] = df['Diabetes_binary'].astype('int')
df['HighBP'] = df['HighBP'].astype('int')
df['HighChol'] = df['HighChol'].astype('int')
df['CholCheck'] = df['CholCheck'].astype('int')
df['BMI'] = df['BMI'].astype('int')
df['Smoker'] = df['Smoker'].astype('int')
df['Stroke'] = df['Stroke'].astype('int')
df['HeartDiseaseorAttack'] = df['HeartDiseaseorAttack'].astype('int'
df['PhysActivity'] = df['PhysActivity'].astype('int')
df['Fruits'] = df['Fruits'].astype('int')
df['Veggies'] = df['Veggies'].astype('int')
df['HvyAlcoholConsump'] = df['HvyAlcoholConsump'].astype('int')
df['AnyHealthcare'] = df['AnyHealthcare'].astype('int')
df['NoDocbcCost'] = df['NoDocbcCost'].astype('int')
df['GenHlth'] = df['GenHlth'].astype('int')
df['MentHlth'] = df['MentHlth'].astype('int')
df['PhysHlth'] = df['PhysHlth'].astype('int')
df['DiffWalk'] = df['DiffWalk'].astype('int')
df['Sex'] = df['Sex'].astype('int')
df['Age'] = df['Age'].astype('int')
df['Education'] = df['Education'].astype('int')
df['Income'] = df['Income'].astype('int')

print(df[["PhysHlth","BMI","GenHlth","MentHlth"]])
print("--------------------")




from sklearn.utils import resample



seed = 42
np.random.seed(seed)


print(df["Diabetes_binary"].value_counts())

###########################pie plot of Diabetes_binary befor balancing#####################
df.groupby('Diabetes_binary').size().plot(kind='pie', y = "Diabetes_binary", label = "Type", autopct='%1.1f%%')

#########################balance dataset###############################
zero_counter = df[df["Diabetes_binary"] == 0]
one_counter = df[df["Diabetes_binary"] == 1]
print(zero_counter.shape)
print(one_counter.shape)
zero_downsample = resample(zero_counter, replace=True, n_samples=len(one_counter), random_state=42)
print(zero_downsample.shape)
data_downsampled = pd.concat([zero_downsample, one_counter])
print(data_downsampled["Diabetes_binary"].value_counts())
df = data_downsampled
print(df)
###########################pie plot of Diabetes_binary after balancing#####################
data_downsampled.groupby('Diabetes_binary').size().plot(kind='pie', y = "Diabetes_binary", label = "Type", autopct='%1.1f%%')




###SPLITTING DATA########
##splitting the data helps in the prediction method also in getting linear and logistic regression
##also to see if the dataset is overfitted or underfitted
##thinking this dataset is overfitted as it has acomplex structure
##droping diabetes binary column and taking all the other columns as an input(##2Darray)(matrix)(rewrew fataia hena m7d4 y2lha)
X =df.drop('Diabetes_binary', axis = 1)
##X=undersampled_data.values
##y is the value existing in the diabete binary column which will be considered as output(1D array)
y=df['Diabetes_binary']
##y=undersampled_data.Class.values
##train_tes_split is imported from sklearn library
##importing numpy for the split also
##first parameter(not optional)its like a two dimensional with the inputs x arrays that holds the data that you want to split
#second parameter(not optional)like a one dimensional array with the outputs y that holds the data already existing
#test_size:should be provided either train_size or test_size ,it defines the number of the test set(percentage)
#random_state:randomaization during splitting(it draws random numbers from various probability distributions),default(none)
##x_train: The training part of the first sequence (x)
##x_test: The test part of the first sequence (x)
##y_train: The training part of the second sequence (y)
##y_test: The test part of the second sequence (y)
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# Normalize Features
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)







########################################################
###########FEATURE SELECTION USING FILTER METHOD AND PEARSON CORRELATION######
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#plt.show()
cor_target = abs(cor["Diabetes_binary"])
relevant_features = cor_target[cor_target>0.4]
print(relevant_features)
#print(df[["GenHlth","HighBP"]].corr())
#print(df[["GenHlth","DiffWalk"]].corr())
print("############################")
###########################################################




#####################LOGISTIC REGRESSION############
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)




#heatmap matrix
plt.figure(figsize = (8,6))
sns.heatmap(confusion_matrix, annot = True, fmt = ".0f", cmap = 'viridis')
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

print("f1_score",  metrics.f1_score(y_test, y_pred))

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
print("############################################")




# generating one row 
rows = df.sample(frac =.25)

# checking if sample is 0.25 times data or not

if (0.25*(len(df))== len(rows)):
    print( "Cool")
    print(len(df), len(rows))

# display
print(rows)


X =rows.drop('Diabetes_binary', axis = 1)
y=rows['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25 , random_state=0)




model = svm.SVC(kernel='linear')
clf = model.fit(X, y)

from sklearn import svm
from sklearn.metrics import accuracy_score

# Creating the SVM model.
clf = svm.SVC(kernel='rbf')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
# Get support vector indices
support_vector_indices = clf.support_
print(support_vector_indices)
# Get number of support vectors per class
support_vectors_per_class = clf.n_support_
print(support_vectors_per_class)
# Get support vectors themselves
support_vectors = clf.support_vectors_


print("Accuracy:", accuracy_score(y_test, y_pred))




from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


#heatmap matrix
plt.figure(figsize = (8,6))
sns.heatmap(confusion_matrix, annot = True, fmt = ".0f", cmap = 'viridis')
plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

print("f1_score",  metrics.f1_score(y_test, y_pred))

############saving the model############


# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
np.random.seed(2)
# we create 40 linearly separable points
X = np.r_[np.random.randn(20, 2) -  [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20
# fit the model
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, Y)
# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.plot(xx, yy, "k-")
plt.plot(xx, yy_down, "k-")
plt.plot(xx, yy_up, "k-")
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
 facecolors="none", zorder=10, edgecolors="k")
plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
 edgecolors="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, y_test)
print(result)