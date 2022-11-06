import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


def mypredict(train, test, next_fold, t):

    if not isinstance(next_fold, type(None)):
        next_fold = next_fold
        train = pd.concat([train, next_fold])

    def get_week(date): 
        
        if date.isocalendar().year == 2010:
            return date.isocalendar().week - 1
        else: 
            return date.isocalendar().week

    start_date = pd.to_datetime("2011-03") + pd.DateOffset(months=2*(t-1))
    end_date = pd.to_datetime("2011-05") + pd.DateOffset(months=2*(t-1))

    date_filter1 = (test['Date'] >= start_date) & (test['Date'] < end_date)

    test_current = test.copy().loc[date_filter1].drop(columns=['IsHoliday'])
    start_last_year = np.min(test_current['Date']) - pd.Timedelta(days=375)
    end_last_year = np.max(test_current['Date']) - pd.Timedelta(days=350)

    date_filter2 = (train['Date'] >= start_last_year) & (
        train['Date'] < end_last_year)

    tmp_train = train.copy().loc[date_filter2]

    tmp_train['Week'] = pd.to_datetime(tmp_train['Date']).apply(get_week)
    tmp_train = tmp_train.rename(columns={'Weekly_Sales':'Weekly_Pred'})
    tmp_train = tmp_train.drop(columns=['Date','IsHoliday'])

    test_current['Week'] = pd.to_datetime(test_current['Date']).dt.isocalendar().week

    test_pred = test_current.merge(
        tmp_train, on=['Week', 'Store', 'Dept'], how='left')

    return train, test_pred


# mypredict = function(){

#     start_date < - ymd("2011-03-01") % m+% months(2 * (t - 1))
#     end_date < - ymd("2011-05-01") % m+% months(2 * (t - 1))
#     test_current < - test % > %
#     filter(Date >= start_date & Date < end_date) % > %
#     select(-IsHoliday)

#     start_last_year = min(test_current$Date) - 375
#     end_last_year = max(test_current$Date) - 350
#     tmp_train < - train % > %
#     filter(Date > start_last_year & Date < end_last_year) % > %
#     mutate(Wk=ifelse(year(Date) == 2010, week(Date)-1, week(Date))) % > %
#     rename(Weekly_Pred=Weekly_Sales) % > %
#     select(-Date, -IsHoliday)

#     test_current < - test_current % > %
#     mutate(Wk=week(Date))

#     test_pred < - test_current % > %
#     left_join(tmp_train, by=c('Dept', 'Store', 'Wk')) % > %
#     select(-Wk)
#     return(test_pred)
# }


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
    train, test_pred = mypredict(train, test, next_fold, t)

    # Load fold file
    # You should add this to your training data in the next call to mypredict()
    fold_file = 'fold_{t}.csv'.format(t=t)
    next_fold = pd.read_csv(fold_file, parse_dates=['Date'], index_col=[0])

    # extract predictions matching up to the current fold
    scoring_df = next_fold.merge(
        test_pred, on=['Date', 'Store', 'Dept'], how='left')

    # extract weights and convert to numpy arrays for wae calculation
    weights = scoring_df['IsHoliday'].apply(
        lambda is_holiday: 5 if is_holiday else 1).to_numpy()
    actuals = scoring_df['Weekly_Sales'].to_numpy()
    preds = scoring_df['Weekly_Pred'].fillna(0).to_numpy()

    wae.append(
        (np.sum(weights * np.abs(actuals - preds)) / np.sum(weights)).item())

print(wae)
print(sum(wae)/len(wae))
