# In[0]

import pandas as pd
from pandas import DataFrame
import numpy as np
from scipy.stats import pearsonr

from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

import warnings;warnings.filterwarnings('ignore')


# In[1]

df_SPYhistoryOrg = pd.read_csv('C:\\data\\SPYhIstory_epidemic_org.csv', encoding = 'Big5')
df_SPYhistoryAvg_ = pd.read_csv('C:\\data\\SPYhIstory_epidemic_avg_.csv', encoding = 'Big5')
df_SPYhistoryAvg = pd.read_csv('C:\\data\\SPYhIstory_epidemic_avg.csv', encoding = 'Big5')
df_Xorg = df_SPYhistoryOrg[['美國增加人數','美國確診增加比例','美國死亡人數','美國死亡增加比例','歐洲增加人數','歐洲確診增加比例','世界增加人數','世界確診增加比例']]
df_Xavg = df_SPYhistoryAvg_[['美國增加人數','美國確診增加比例','美國死亡人數','美國死亡增加比例','歐洲增加人數','歐洲確診增加比例','世界增加人數','世界確診增加比例']]
df_X = df_SPYhistoryAvg[['美國增加人數','美國確診增加比例','美國死亡人數','美國死亡增加比例','歐洲增加人數','歐洲確診增加比例','世界增加人數','世界確診增加比例']]
df_y = df_SPYhistoryOrg[['漲跌']]


loo = LeaveOneOut()
loo.get_n_splits(df_X)

for train_index, test_index in loo.split(df_X):
    X_train, X_test = df_X.iloc[train_index], df_X.iloc[test_index]
    y_train, y_test =df_y.iloc[train_index],df_y.iloc[test_index]

loocv = model_selection.LeaveOneOut()

# In[1.5]

print('\n ****************************************** \n')
print('KNearestNeighbors \n ---')
knn = KNeighborsClassifier(n_neighbors = 5)
score = model_selection.cross_val_score(knn, df_Xorg, df_y, cv=loocv)
print("Accuracy(org): %.3f%%" % (score.mean()*100.0))

score = model_selection.cross_val_score(knn, df_Xavg, df_y, cv=loocv)
print("Accuracy(without normalized): %.3f%%" % (score.mean()*100.0))

score = model_selection.cross_val_score(knn, df_X, df_y, cv=loocv)
print("Accuracy(avg): %.3f%%" % (score.mean()*100.0))
print('\n ****************************************** \n')


# In[2]

features = ['美國增加人數','美國確診增加比例','美國死亡人數','美國死亡增加比例','歐洲增加人數','歐洲確診增加比例','世界增加人數','世界確診增加比例']
print('\n ****************************************** \n')
for feature in features :
    feature_data = np.array(df_X[feature][:], dtype = float)
    y = np.array(df_y.iloc[:, 0], dtype = float)
    corr, _ = pearsonr(feature_data, y)
    print('Pearsons correlation: %.5f' % corr, feature)
print('\n ****************************************** \n')

    
# In[3]

from sklearn.linear_model  import LogisticRegression

print('\n ****************************************** \n')
print('LogisticRegression') 
LR = LogisticRegression = LogisticRegression(solver='lbfgs')
score = model_selection.cross_val_score(LR , df_Xorg, df_y, cv=loocv) 
print("Accuracy: %.3f%%" % (score.mean()*100.0))
print('\n ****************************************** \n')


# In[4]

print('\n ****************************************** \n')
print('SupportVectorMachine')
svm = SVC(kernel='linear', C=1, probability=True)
score = model_selection.cross_val_score(svm, df_X, df_y, cv=loocv)
print("Accuracy: %.3f%%" % (score.mean()*100.0))

svm = SVC(kernel='poly', C=1, probability=True)
score = model_selection.cross_val_score(svm, df_X, df_y, cv=loocv)
print("Accuracy: %.3f%%" % (score.mean()*100.0))

svm = SVC(kernel='rbf', gamma = 5, probability=True)
score = model_selection.cross_val_score(svm, df_X, df_y, cv=loocv)
print("Accuracy: %.3f%%" % (score.mean()*100.0))
print('\n ****************************************** \n')


# In[5]

print('\n ****************************************** \n')
print('KNearestNeighbors')
knn = KNeighborsClassifier(n_neighbors = 5)
score = model_selection.cross_val_score(knn, df_X, df_y, cv=loocv)
print("Accuracy: %.3f%%" % (score.mean()*100.0))
print('\n ****************************************** \n')


# In[6]

print('\n ****************************************** \n')
print('RandomForest')
RF = RandomForestClassifier(n_estimators = 220, min_samples_leaf = 21, oob_score = True, criterion = 'gini')
score = model_selection.cross_val_score(RF, df_X, df_y, cv=loocv)      
print("Accuracy: %.3f%%" % (score.mean()*100.0))
print('\n ****************************************** \n')


# In[7]

X = df_X[['美國確診增加比例','美國死亡增加比例','歐洲確診增加比例','世界確診增加比例']]

print('\n ****************************************** \n')
print('RandomForest')
RF = RandomForestClassifier(n_estimators = 220, min_samples_leaf = 21, oob_score = True, criterion = 'gini')
score = model_selection.cross_val_score(RF, X, df_y, cv=loocv)      
print("Accuracy: %.3f%%" % (score.mean()*100.0))
print('\n ****************************************** \n')


# In[8]

print('\n ****************************************** \n')
print('KNearestNeighbors')
knn = KNeighborsClassifier(n_neighbors = 5)
score = model_selection.cross_val_score(knn, X, df_y, cv=loocv)
print("Accuracy: %.3f%%" % (score.mean()*100.0))
print('\n ****************************************** \n')



# In[9]
