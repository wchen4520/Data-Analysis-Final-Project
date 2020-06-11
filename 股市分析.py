# In[0]

import pandas as pd
from pandas import DataFrame
import numpy as np
import scipy.stats as ss
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.neighbors import KNeighborsClassifier


import warnings;warnings.filterwarnings('ignore')


# In[1]

df = pd.read_csv('C:\\data\\SPYrow_new.csv').drop(["Date", "MACD"], axis=1)

df_X = df.drop(["Rise"], axis=1)
df_X = df_X.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
df_y = df[['Rise']]

def buildTrain(df_X, df_y, pastDay, futureDay):
    X_train, Y_train = [], []
    for i in range(df_X.shape[0]-futureDay-pastDay):
        X_train.append(np.array(df_X.iloc[i+pastDay]))
        Y_train.append(np.array(df_y.iloc[i+pastDay+futureDay]["Rise"]))
    return np.array(X_train), np.array(Y_train)

features = df_X.columns.ravel().tolist()




# In[2]

from sklearn.linear_model  import LogisticRegression

X, y = buildTrain(df_X, df_y, 1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle = False)

LR = LogisticRegression(solver='lbfgs')
LR.fit(X_train,y_train)

print('-----------------')
print('LogisticRegression')
print('-----------------')

print('---\n score(training data) = ', LR.score(X_train, y_train))
print('score(testing data) = ', LR.score(X_test, y_test),'\n --')


# In[3]

corr = list()
print('\n ****************************************** \n')
for feature in features :
    feature_data = np.array(df_X[feature][:], dtype = float)
    y = np.array(df_y.iloc[:, 0], dtype = float)
    corr_, _ = pearsonr(feature_data, y)
    print('Pearsons correlation: %.5f' % corr_, feature)
    corr.append(-abs(corr_))
print('\n ****************************************** \n')

tempFeatures = list()
for i in range(0, len(corr)) :
    if abs(corr[i]) > 0.3 :
        tempFeatures.append(features[i])
temp_df_X = df_X[tempFeatures]
temp_X, temp_y = buildTrain(temp_df_X, df_y, 1, 1)
temp_X_train, temp_X_test, temp_y_train, temp_y_test = train_test_split(temp_X, temp_y, test_size=0.25, shuffle = False)        

from sklearn.linear_model  import LogisticRegression

LR = LogisticRegression(solver='lbfgs')

print('*********************')
print('LogisticRegression')
print('*********************')

LR.fit(temp_X_train, temp_y_train)
print('---\n score(training data) = ', LR.score(temp_X_train, temp_y_train))
print('score(testing data) = ', LR.score(temp_X_test, temp_y_test),'\n ---')


# In[4]

from sklearn.linear_model  import LogisticRegression

print('*********************')
print('Lasso')
print('*********************')

lasso_reg = LogisticRegression(solver='liblinear', penalty = 'l1')
lasso_reg.fit(X_train, y_train)

count = 0
coef = lasso_reg.coef_.ravel().tolist()
for i in range(0, len(coef)) :
    if coef[i] != 0 :
        count += 1

print(lasso_reg.intercept_, lasso_reg.coef_,'\n ---')
print('score(training data) = ',lasso_reg.score(X_train, y_train))
print('score(testing data) = ',lasso_reg.score(X_test, y_test),'\n ---')
print("Optimal number of features : %d" % count)
print('\n ---')

print('*********************')
print('Ridge')
print('*********************')

ridge_reg = LogisticRegression(solver='liblinear', penalty = 'l2')
ridge_reg.fit(X_train, y_train)

print(ridge_reg.intercept_, ridge_reg.coef_,'\n ---')
print('score(training data) = ', ridge_reg.score(X_train, y_train))
print('score(testing data) = ', ridge_reg.score(X_test, y_test),'\n ---')
print('\n ---')


# In[5]

from sklearn.linear_model  import LogisticRegression

LR = LogisticRegression(solver='lbfgs')

print('*********************')
print('LogisticRegression')
print('*********************')

LR.fit(X_train, y_train)
print('---\n score(training data) = ', LR.score(X_train, y_train))
print('score(testing data) = ', LR.score(X_test, y_test),'\n ---')
print('\n \n ***********************************************************')
    
rfecv = RFECV(estimator=LR, step=1, cv=StratifiedKFold(3), scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)
print("Ranking of features : %s" % rfecv.ranking_)
print("Support is %s" % rfecv.support_)
print("Grid Scores %s" % rfecv.grid_scores_)
print('---\n score(training data) = ',rfecv.score(X_train, y_train))
print('score(testing data) = ',rfecv.score(X_test, y_test),'\n ---')
print('\n \n ***********************************************************')
for i in range(0,len(features)) :
    support = rfecv.support_
    if support[i] == True :
        print(features[i])
    


# In[6]

rankCorr = ss.rankdata(corr)

coefLasso = lasso_reg.coef_.ravel().tolist()
rankLasso = ss.rankdata(coefLasso)

coefRidge = ridge_reg.coef_.ravel().tolist()
rankRidge = ss.rankdata(coefRidge)

coefRfecv = rfecv.grid_scores_.ravel().tolist()
rankRfecv = ss.rankdata(coefRfecv)

coefRanking = pd.concat([DataFrame(rankCorr), DataFrame(rankLasso), DataFrame(rankRidge), DataFrame(rankRfecv)], axis = 1)

coefRanking_ = ss.rankdata(rankCorr*LR.score(X_test, y_test) + (coefLasso + rankRidge)*(lasso_reg.score(X_test, y_test)+ridge_reg.score(X_test, y_test))/4 + rankRfecv*rfecv.score(X_test, y_test))
coefRanking = pd.concat([coefRanking, DataFrame(coefRanking_)], axis = 1)
coefRanking.columns = ['Corr', 'Lasso', 'Ridge', 'Rfevc', 'Avg']
coefRanking.index = [features]

print(coefRanking)

# In[6.5]


from sklearn.linear_model  import LogisticRegression

LR = LogisticRegression(solver='lbfgs')

print('\n \n *********************')
print('LogisticRegression')
print('*********************')

'''
score = 0
bestFeatureAmount = 0
tempFeatures = list()
for i in range(1, len(features)+1) :
    tempFeatures = list()
    for j in range(0, len(coefRanking['Avg'])) :
        if coefRanking['Avg'][j] <= i :
            tempFeatures.append(features[j])
    temp_df_X = df_X[tempFeatures]
    temp_X, temp_y = buildTrain(temp_df_X, df_y, 1, 1)
    temp_X_train, temp_X_test, temp_y_train, temp_y_test = train_test_split(temp_X, temp_y, test_size=0.25, shuffle = False)  
    LR.fit(temp_X_train, temp_y_train)
    score_ =  LR.score(temp_X_test, temp_y_test)
    if score_ > score :
        score = score_
        bestFeatureAmount = i        
'''

bestFeatureAmount = 14
tempFeatures = list()
for j in range(0, len(coefRanking['Avg'])) :
    if coefRanking['Avg'][j] <=  bestFeatureAmount:
        tempFeatures.append(features[j])
temp_df_X = df_X[tempFeatures]
temp_X, temp_y = buildTrain(temp_df_X, df_y, 1, 1)
temp_X_train, temp_X_test, temp_y_train, temp_y_test = train_test_split(temp_X, temp_y, test_size=0.25, shuffle = False)  
LR = LogisticRegression(solver='lbfgs')
LR.fit(temp_X_train, temp_y_train)
print('score(training data) = ',LR.score(temp_X_train, temp_y_train))
print('score(testing data) = ',LR.score(temp_X_test, temp_y_test),'\n ---')
print('\n ---')


# In[7]
svm = SVC(kernel='linear', C=1, probability=True)

svm.fit(X_train,y_train)
pred = svm.predict(X_test)
fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
roc_auc = metrics.auc(fpr, tpr)

print('*********************')
print('Support Vector Machine')
print('*********************')
print('---\n score(training data) = ',svm.score(X_train, y_train))
print('score(testing data) = ',svm.score(X_test, y_test),'\n ---')
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[8]
    
K_ideal = 35

knn = KNeighborsClassifier(n_neighbors = K_ideal)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
roc_auc = metrics.auc(fpr, tpr)

print('*********************')
print('KNeighborsClassifier')
print('*********************')
print('---\n score(training data) = ',knn.score(X_train, y_train))
print('score(testing data) = ',knn.score(X_test, y_test),'\n ---')
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# In[9]


RandomForest = RandomForestClassifier(n_estimators = 280, min_samples_leaf = 70, max_depth = 18, oob_score = True, criterion = 'gini')

print('*********************')
print('RandomForest')
print('*********************')
RF = RandomForestClassifier(n_estimators = 220, min_samples_leaf = 21, max_depth = 3, oob_score = True, criterion = 'gini')
RF.fit(X_train,y_train) 
pred = RF.predict(X_test)
fpr, tpr, threshold = metrics.roc_curve(y_test, pred)
roc_auc = metrics.auc(fpr, tpr)

print('---\n score(training data) = ',knn.score(X_train, y_train))
print('score(testing data) = ',knn.score(X_test, y_test),'\n ---')
plt.figure(figsize=(6,6))
plt.title('Validation ROC')
plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()