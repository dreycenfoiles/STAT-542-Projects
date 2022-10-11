###########################################
# Step 0: Load necessary libraries

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import timeit
import scipy
import glmnet_python
from glmnet import glmnet
from glmnetPredict import glmnetPredict
from glmnetCoef import glmnetCoef

def numeric_convert(frame):
    # We may want to normalize data as well
    for col in frame:
        try:
            frame[col] = pd.to_numeric(frame[col])
        except:
            frame[col] = pd.factorize(frame[col])[0]
    return frame

###########################################
# Step 1: Preprocess training data
#         and fit two models

bad_cols = ['PID', 'Garage_Yr_Blt', 'Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude']

train = pd.read_csv('train.csv', delim_whitespace=' ', header=None)
train_frame = clean_data(train, bad_cols)

xtrain = numeric_convert(train_frame.iloc[:,:-1])
for c in xtrain:
    xtrain[c] = xtrain[c] / np.max(np.abs(x[c]))

xtrain = xtrain.to_numpy()
ytrain = np.log(frame.iloc[:,-1]).to_numpy()

# LR model
lr_model = glmnet(x=xtrain, y=ytrain, family = 'gaussian')

# RF model
rf_model = RandomForestRegressor(criterion='squared_error')
rf_model.fit(xtrain, ytrain)


###########################################
# Step 2: Preprocess test data
#         and output predictions into two files
#

test = pd.read_csv('test.csv', delim_whitespace=' ', header=None)
test_frame = clean_data(test, bad_cols)

xtest = numeric_convert(test_frame)
for c in xtest:
    xtest[c] = xtest[c] / np.max(np.abs(xtest[c]))

xtest = xtest.to_numpy()


lr_yhat = glmnetPredict(lr_model, xtest, s = scipy.float64([0.00018263318833249426])).flatten()
lr_yhat = np.exp(lr_yhat)
pd.DataFrame(data={'PID': PID, 'Sale_Price': pd.Series(lr_yhat, name='Sale_Price')}).to_csv('mysubmission1.txt', index=False)

rf_yhat = np.exp(rf_model.predict(xtest))
pd.DataFrame(data={'PID': PID, 'Sale_Price': pd.Series(rf_yhat, name='Sale_Price')}).to_csv('mysubmission2.txt', index=False)
