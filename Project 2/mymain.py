import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression



def mypredict(train, test, next_fold, t):

    if not isinstance(next_fold, type(None)):
        next_fold = next_fold
        train = pd.concat([train,next_fold])

    tmp_train = pd.DataFrame()
    tmp_test = pd.DataFrame() 

    tmp_train['Week'] = pd.to_datetime(train['Date']).dt.isocalendar().week
    tmp_train[['Store','Dept','IsHoliday']] = train[['Store', 'Dept', 'IsHoliday']]

    tmp_train = pd.get_dummies(tmp_train, columns=['Week','Store','Dept','IsHoliday'])

    start_date = pd.to_datetime("2011-03") + pd.DateOffset(months=2*t)
    end_date = pd.to_datetime("2011-04") + pd.DateOffset(months=2*t)

    date_filter = (test['Date'] >= start_date) & (test['Date'] < end_date)

    tmp_test = test.copy().loc[date_filter].drop(['Date'],axis=1)
    tmp_test['Week'] = pd.to_datetime(test['Date']).dt.isocalendar().week
    tmp_test = pd.get_dummies(tmp_test, columns=['Week','Store','Dept','IsHoliday'])
    
    tmp_test = tmp_test.reindex(columns=tmp_train.columns, fill_value=0)

    xtrain = tmp_train.values
    ytrain = train['Weekly_Sales'].values

    model = LinearRegression()

    model.fit(xtrain, ytrain)

    ypred = model.predict(tmp_test.values)
    test.loc[date_filter, "Weekly_Pred"] = ypred

    return train, test


train = pd.read_csv('train_ini.csv', parse_dates=['Date'], index_col=[0])
test = pd.read_csv('test.csv', parse_dates=['Date'], index_col=[0])

# save weighed mean absolute error WMAE
n_folds = 10
next_fold = None
wae = []

# time-series prediction
for t in range(1, n_folds+1):
    print(f'Fold{t}...')

    # *** THIS IS YOUR PREDICTION FUNCTION ***
    train, test_pred = mypredict(train, test, next_fold, t-1)

    # Load fold file
    # You should add this to your training data in the next call to mypredict()
    fold_file = 'fold_{t}.csv'.format(t=t)
    next_fold = pd.read_csv(fold_file, parse_dates=['Date'], index_col=[0])

    # extract predictions matching up to the current fold
    scoring_df = next_fold.merge(
        test_pred, on=['Date', 'Store', 'Dept'], how='left', suffixes=("", "_dummy"))

    print(scoring_df)

    # extract weights and convert to numpy arrays for wae calculation
    weights = scoring_df['IsHoliday'].apply(
        lambda is_holiday: 5 if is_holiday else 1).to_numpy()
    actuals = scoring_df['Weekly_Sales'].to_numpy()
    preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()

    wae.append(
        (np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())

print(wae)
print(sum(wae)/len(wae))
