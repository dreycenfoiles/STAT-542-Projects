import numpy as np
import pandas as pd


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

    tmp_train['Week'] = pd.to_datetime(tmp_train['Date']).dt.isocalendar().week
    tmp_train = tmp_train.rename(columns={'Weekly_Sales':'Weekly_Pred'})
    tmp_train = tmp_train.drop(columns=['Date'])

    test_current['Week'] = pd.to_datetime(test_current['Date']).dt.isocalendar().week

    test_pred = test_current.merge(
        tmp_train, on=['Week', 'Store', 'Dept'], how='left')

    return train, test_pred