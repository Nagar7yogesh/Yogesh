# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:53:54 2020

@author: nagar
"""
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
train =pd.read_csv('train.csv')
test =pd.read_csv('test.csv')
ytrain=train.iloc[:,-1].values
ytrain = ytrain.reshape(-1,1)

#Selecting columns with maximum correlation
corr_mat=train.corr()
col = corr_mat.nlargest(20, 'SalePrice')['SalePrice'].index
col=col[1:]
xtrain=train[col]
xtest=test[col]
#print(xtest.info())
xtrain=np.array(xtrain)
xtest=np.array(xtest)

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer1 = imputer1.fit(xtrain[:,1:8 ])
xtrain[:,1:8] = imputer1.transform(xtrain[:,1:8])
xtest[:,1:8] = imputer1.transform(xtest[:,1:8])
imputer2 = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
imputer2 = imputer2.fit(xtrain[:,8:11 ])
xtrain[:,8:11] = imputer2.transform(xtrain[:,8:11])
xtest[:,8:11] = imputer2.transform(xtest[:,8:11])
imputer3 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer3 = imputer1.fit(xtrain[:,11:])
xtrain[:,11:] = imputer1.transform(xtrain[:,11:])
xtest[:,11:] = imputer1.transform(xtest[:,11:])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
xtrain = sc_X.fit_transform(xtrain)
xtest = sc_X.transform(xtest)
#sc_y = StandardScaler()
#ytrain = sc_y.fit_transform(ytrain)

#Evaluation
from sklearn.model_selection import cross_val_score
def rmse_score(model):
    scores =(cross_val_score(model,xtrain,ytrain,scoring="neg_mean_squared_error", cv=7))
    rmse_scores = np.sqrt(-scores)
    return(rmse_scores.mean())

#linear regression model training
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(xtrain,ytrain)

#Decision tree Regression
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(xtrain,ytrain)

# Fitting SVR to the dataset
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(xtrain,ytrain)

#Random Forest regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(xtrain,ytrain)
print(rmse_score(forest_reg ))

#lightgbm
import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,learning_rate=0.05, n_estimators=720,max_bin = 55, bagging_fraction = 0.8,bagging_freq = 5, feature_fraction = 0.2319,feature_fraction_seed=9, bagging_seed=9,min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(xtrain,ytrain)
print(rmse_score(model_lgb))

"""#Lasso Regression
from sklearn.linear_model import ElasticNet, Lasso
model_lasso =Lasso(alpha =0.0005, random_state=1)
model_lasso.fit(xtrain,ytrain)

#ElasticNet Regression
model_ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
model_ENet.fit(xtrain,ytrain)

#KernelRidge Regression
from sklearn.kernel_ridge import KernelRidge
model_KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
model_KRR.fit(xtrain,ytrain)

#Gradient Booster
from sklearn.ensemble import GradientBoostingRegressor
model_GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5)
model_GBoost.fit(xtrain,ytrain)

#Xgboosting
import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, min_child_weight=1.7817, n_estimators=2200,reg_alpha=0.4640, reg_lambda=0.8571,subsample=0.5213, silent=1,random_state =7, nthread = -1)
model_xgb.fit(xtrain,ytrain)"""



# Predicting the Test set results
predictions = model_lgb.predict(xtest)


# output
submission = pd.DataFrame()
submission['Id'] = test.Id
submission['SalePrice'] = predictions
submission.to_csv('house_price_predictions.csv',index=False)
