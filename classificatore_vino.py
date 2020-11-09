# -*- coding: utf-8 -*-
"""
@author: Giovanni Figliuolo, Perla Catucci
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier


wine_dataset = pd.read_csv("Dataset/winequality-red.csv")
group_names = ['bad', 'good']

#Shows the plot
def plot_results(classifier,X_test,y_test, name):
    
    pred = classifier.predict(X_test)
    print(classification_report(y_test, pred)) 
    confusion_matrix = confusion_matrix(y_test, pred, group_names)
    ax = plt.subplot()
    sns.heatmap(confusion_matrix, annot=True, ax = ax, fmt='g', cmap="Reds"); 

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title(name); 
    ax.xaxis.set_ticklabels(group_names);
    ax.yaxis.set_ticklabels(group_names);
    plt.show()

# Boxplot used for visualization of correlation 
def plot(x_value, y_value):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(x= x_value, y= y_value, data=wine_dataset,  palette='RdBu_r')
    

# Splittig data in two bins 
# Marked as 'bad' ==> 1, 2, 3 ,4 , 5, 
# Marked as '' ==> 6, 7, 8, 9, 10
def split_data(): 
    bins = (2, 5.5, 8)
    wine_dataset['quality'] = pd.cut(wine_dataset['quality'], bins = bins, labels = group_names)
    return wine_dataset

#Importing dataset
wine_dataset = pd.read_csv("Dataset/winequality-red.csv")
print(wine_dataset['quality'].value_counts())

# Dataset visualization  
print('Rows in the dataset: ', wine_dataset.shape[0])
print('Columns in the dataset: ', wine_dataset.shape[1])
print(wine_dataset.isnull().sum())
wine_dataset.describe()
wine_dataset['quality'].unique()
sns.countplot(wine_dataset['quality'], palette='RdBu_r')

#Data correlation
plot('quality', 'fixed acidity')
plot('quality', 'volatile acidity')
plot('quality', 'citric acid')
plot('quality', 'residual sugar')
plot('quality', 'chlorides')
plot('quality', 'free sulfur dioxide')
plot('quality', 'sulphates')
plot('quality', 'alcohol')
plot('quality', 'pH')
plot('quality', 'density')
plot('quality', 'total sulfur dioxide')
correlations = wine_dataset.corr()['quality'].drop('quality')
print(correlations)
plt.figure(figsize=(12,8))
corr = wine_dataset.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr,mask=mask, annot=True, linewidths=1, cmap='RdBu_r')
plt.show()

# Pre Processing
split_data()
sns.countplot(x='quality', data=wine_dataset, palette='RdBu_r')
plt.show()

#Subset x with every feature except 'quality' and subset y with feature quality
X = wine_dataset.iloc[:,:11]
y = wine_dataset['quality']

# Normalize
sc = StandardScaler()
X = sc.fit_transform(X)

# Principal Component Analysis
pca = PCA()
X_pca = pca.fit_transform(X)

#Calculate Variance Ratios
variance = pca.explained_variance_ratio_ 
var = np.cumsum(np.round(variance, decimals=3)*100)
print(var)

plt.figure(figsize=(7,6))
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.plot(var, 'ro-')
plt.grid()

#As per the graph,there are 8 principal components for 90% of variation in the data. 
#pick the first 8 components for prediction.
pca_new = PCA(n_components=8)
X_new = pca_new.fit_transform(X)
print(wine_dataset['quality'].value_counts())

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=.25, random_state=0)
print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ", y_train.shape)
print("Shape of y_test", y_test.shape)


# Classification

# Support Vector Machines
# Setting parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf', 'sigmoid']}

grid_svm = GridSearchCV(SVC(), param_grid=param_grid, cv=5, refit = True, verbose = False) 
grid_svm.fit(X_train, y_train) 
print("best_params", grid_svm.best_params_) 
print("best_estimator", grid_svm.best_estimator_) 
plot_results(grid_svm, X_test, y_test,'Support Vector Machines')


#Random Forest
rf=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [100, 250, 500],
    'max_features': ['auto', 'log2'],
    'criterion' :['gini', 'entropy']
}

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5, refit = True, verbose = False)
grid_rf.fit(X_train, y_train)

print("best_params", grid_rf.best_params_) 
print("best_estimator", grid_rf.best_estimator_) 
plot_results(grid_rf, X_test, y_test,'Random Forest')

#K-Nearest Neighbors
knn = KNeighborsClassifier()
param_grid = {'n_neighbors':[1,4,5,6,7,8],
              'leaf_size':[1,3,5,10],
}

grid_knn = GridSearchCV(knn, param_grid=param_grid)
grid_knn.fit(X_train ,y_train)

print("best_params", grid_knn.best_params_) 
print("best_estimator", grid_knn.best_estimator_) 
plot_results(grid_knn, X_test, y_test,'K-nearest neighbors')

# AdaBoost
Ada = AdaBoostClassifier(random_state=1)
Ada.fit(X_train, y_train)
plot_results(Ada, X_test, y_test,'AdaBoost')


#Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
plot_results(gaussian, X_test, y_test,'Gaussian Naive Bayes')