###########################################
# Step 0: Load necessary libraries

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats.mstats import winsorize
import timeit
import scipy

###########################################
# Step 1: Preprocess training data
#         and fit two models

def numeric_convert(frame):
    # We may want to normalize data as well
    frame = frame.drop(columns=bad_cols)

    numeric_data = pd.DataFrame()

    category_cols = []
    for col in frame:
        try:
            numeric_data[col] = pd.to_numeric(frame[col])
            if col in limit_cols:
                numeric_data[col] = winsorize(frame[col], (0, .1))

        except:
            category_cols.append(col)

    category_cols = np.array(category_cols)
    categorical_data = pd.get_dummies(frame[category_cols])

    final_data = pd.concat([numeric_data, categorical_data], axis=1)

    return final_data


bad_cols = ['PID', 'Garage_Yr_Blt', 'Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude',
            'Latitude', 'Alley', 'Land_Contour', 'Land_Slope', 'Exter_Cond', 'Electrical', 'Bsmt_Half_Bath', 'Three_season_porch', 'Central_Air', 'MS_Zoning']
limit_cols = ["Lot_Frontage", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF",
              'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Screen_Porch", "Misc_Val"]

train = pd.read_csv('train.csv')

PID = train["PID"]

xtrain_frame = numeric_convert(train.iloc[:,:-1])

xtrain = xtrain_frame.to_numpy()
ytrain = np.log(train.iloc[:,-1]).to_numpy()

# LR model
lr_model = Lasso(alpha=.01)
lr_model.fit(xtrain, ytrain)

# RF model
rf_model = RandomForestRegressor()
rf_model.fit(xtrain, ytrain)


###########################################
# Step 2: Preprocess test data
#         and output predictions into two files
#

test = pd.read_csv('test.csv')
xtest_frame = numeric_convert(test)
xtest_frame = xtest_frame.reindex(columns=xtrain_frame.columns, fill_value=0)
xtest = xtest_frame.to_numpy()


lr_yhat = np.exp(lr_model.predict(xtest))
pd.DataFrame(data={'PID': PID, 'Sale_Price': pd.Series(lr_yhat, name='Sale_Price')}).to_csv('mysubmission1.txt', index=False)

rf_yhat = np.exp(rf_model.predict(xtest))
pd.DataFrame(data={'PID': PID, 'Sale_Price': pd.Series(rf_yhat, name='Sale_Price')}).to_csv('mysubmission2.txt', index=False)
