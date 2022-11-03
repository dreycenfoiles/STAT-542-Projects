import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def mypredict(train, test, next_fold, t):

    x_cols = ['Year', 'Week', 'Store', 'Dept', 'IsHoliday']

    start_date = pd.to_datetime("2011-03") + pd.DateOffset(months=2*t)
    end_date = pd.to_datetime("2011-04") + pd.DateOffset(months=2*t)

    date_filter = (test['Date'] >= start_date) & (test['Date'] < end_date)

    current_test = test[x_cols].copy().loc[date_filter]

    #tmp = pd.DataFrame()

    if not isinstance(next_fold, type(None)):
        next_fold = next_fold
        train = pd.concat([train,next_fold])

    #dates = train['Date']
    #tmp = train.copy()
    #tmp['Date'] = (dates - dates.min()).dt.days

    #current_test_dates = current_test['Date']
    #current_test['Date'] = (current_test_dates - dates.min()).dt.days

    xtrain = train[x_cols].values
    ytrain = train['Weekly_Sales'].values

    model = LinearRegression()

    model.fit(xtrain, ytrain)

    ypred = model.predict(current_test.values)
    test.loc[date_filter, "Weekly_Pred"] = ypred

    return train, test


train = pd.read_csv('train_ini.csv', parse_dates=['Date'])
test = pd.read_csv('test.csv', parse_dates=['Date'])

n_years = 3
yrs = pd.to_datetime(train['Date']).dt.to_period('Y').unique().year
for date in train['Date'].unique():
    wk = np.zeros(52)
    yr = np.zeros(n_years)
    wk_idx = (train.loc[train['Date'] == date]['Week']).unique()[0]
    yr_idx = (train.loc[train['Date'] == date]['Year']).unique()[0]
    wk[wk_idx] = 1
    yr[yr_idx] = 1
    idx = train.loc[train['Date'] == date].index
    n = len(idx)
    s_wk = pd.Series(data=n * [wk], index=idx)
    s_yr = pd.Series(data=n * [yr], index=idx)
    # it takes too long to write all the arrays to a file, so
    # we'll just do that in memeory in the next step
    train.loc[train['Date']==date,'Week'] = s_wk
    train.loc[train['Date']==date,'Year'] = s_yr


yrs = pd.to_datetime(test['Date']).dt.to_period('Y').unique().year
for date in test['Date'].unique():
    wk = np.zeros(52)
    yr = np.zeros(n_years)
    wk_idx = (test.loc[test['Date'] == date]['Week']).unique()[0]
    yr_idx = (test.loc[test['Date'] == date]['Year']).unique()[0]
    wk[wk_idx] = 1
    yr[yr_idx] = 1
    idx = test.loc[test['Date'] == date].index
    n = len(idx)
    s_wk = pd.Series(data=n * [wk], index=idx)
    s_yr = pd.Series(data=n * [yr], index=idx)
    # it takes too long to write all the arrays to a file, so
    # we'll just do that in memeory in the next step
    test.loc[test['Date']==date,'Week'] = s_wk
    test.loc[test['Date']==date,'Year'] = s_yr

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

    #yrs = pd.to_datetime(next_fold['Date']).dt.to_period('Y').unique().year
    #for date in next_fold['Date'].unique():
    #    wk = np.zeros(52)
    #    yr = np.zeros(n_years)
    #    wk_idx = (next_fold.loc[next_fold['Date'] == date]['Week']).unique()[0]
    #    yr_idx = (next_fold.loc[next_fold['Date'] == date]['Year']).unique()[0]
    #    wk[wk_idx] = 1
    #    yr[yr_idx] = 1
    #    idx = next_fold.loc[next_fold['Date'] == date].index
    #    n = len(idx)
    #    s_wk = pd.Series(data=n * [wk], index=idx)
    #    s_yr = pd.Series(data=n * [yr], index=idx)
    #    # it takes too long to write all the arrays to a file, so
    #    # we'll just do that in memeory in the next step
    #    next_fold.loc[next_fold['Date']==date,'Week'] = s_wk
    #    next_fold.loc[next_fold['Date']==date,'Year'] = s_yr

    # extract predictions matching up to the current fold
    scoring_df = next_fold.merge(
        test_pred, on=['Date', 'Store', 'Dept'], how='left', suffixes=("", "_dummy"))

    scoring_df.drop_duplicates()

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
