###########################################
# Step 0: Load necessary libraries

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats.mstats import winsorize
import scipy

###########################################
# Step 1: Preprocess training data
#         and fit two models


def numeric_convert(frame, model):
    # We may want to normalize data as well
    if model == 'LR':
        frame = frame.drop(columns=bad_cols)
    elif model == 'RF':
        frame = frame.drop(columns=['PID', 'Garage_Yr_Blt'])

    numeric_data = pd.DataFrame()

    category_cols = []
    for col in frame:
        try:
            numeric_data[col] = pd.to_numeric(frame[col])
            if col in limit_cols:
                numeric_data[col] = winsorize(frame[col], (0, .05))

        except:
            category_cols.append(col)

    category_cols = np.array(category_cols)
    categorical_data = pd.get_dummies(frame[category_cols])

    final_data = pd.concat([numeric_data, categorical_data], axis=1)

    return final_data


bad_cols = ['PID', 'Garage_Yr_Blt', 'Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude', 'Latitude', 'Alley', 'Land_Contour', 'Land_Slope',
            'Exter_Cond', 'Bsmt_Half_Bath', 'Three_season_porch', 'MS_Zoning', "Misc_Val", "Kitchen_AbvGr", "Pool_Area", "Garage_Cond"]
limit_cols = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF",
              'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Screen_Porch", "Misc_Val"]

train = pd.read_csv('train.csv')
PID_train = train["PID"]
xtrain_frame_LR = numeric_convert(train.iloc[:, :-1], "LR")
xtrain_frame_RF = numeric_convert(train.iloc[:, :-1], "RF")

xtrain_LR = xtrain_frame_LR.to_numpy()
xtrain_RF = xtrain_frame_RF.to_numpy()
ytrain = np.log(train.iloc[:, -1]).to_numpy()

# LR model
lr_model = Ridge(alpha=2, max_iter=4000, positive=False)
lr_model.fit(xtrain_LR, ytrain)

# RF model
rf_model = GradientBoostingRegressor(
    n_estimators=500, random_state=1, max_depth=4)
rf_model.fit(xtrain_RF, ytrain)


###########################################
# Step 2: Preprocess test data
#         and output predictions into two files
#

test = pd.read_csv('test.csv')
PID_test = test["PID"]

xtest_frame_LR = numeric_convert(test, 'LR')
xtest_LR = xtest_frame_LR.reindex(
    columns=xtrain_frame_LR.columns, fill_value=0).values

xtest_frame_RF = numeric_convert(test, 'RF')
xtest_RF = xtest_frame_RF.reindex(
    columns=xtrain_frame_RF.columns, fill_value=0).values


lr_yhat = np.exp(lr_model.predict(xtest_LR))
pd.DataFrame(data={'PID': PID_test, 'Sale_Price': pd.Series(
    lr_yhat, name='Sale_Price')}).to_csv('mysubmission1.txt', index=False, sep=",")

rf_yhat = np.exp(rf_model.predict(xtest_RF))
pd.DataFrame(data={'PID': PID_test, 'Sale_Price': pd.Series(
    rf_yhat, name='Sale_Price')}).to_csv('mysubmission2.txt', index=False, sep=",")
