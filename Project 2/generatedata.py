import pandas as pd
from dateutil.relativedelta import relativedelta
import datetime
import numpy as np

# read raw data and extract date column
train_raw = pd.read_csv('https://liangfgithub.github.io/Data/train.csv.zip')

# Preproccess data
# Add classifiers for weeks, years
yrs = pd.to_datetime(train_raw['Date']).dt.to_period('Y').unique().year
n_years = len(yrs)
for date in train_raw['Date'].unique():
    wk = np.zeros(52)
    yr = np.zeros(n_years)
    y, m, d = date.split('-')
    wk_idx = datetime.date(int(y), int(m), int(d)).isocalendar()[1] - 1
    yr_idx = np.where(yrs == int(y))[0][0]
    # it takes too long to write all the arrays to a file, so
    # we'll just do that in memeory in the next step
    train_raw.loc[train_raw['Date']==date,'Week'] = wk_idx
    train_raw.loc[train_raw['Date']==date,'Year'] = yr_idx
train_raw.Week = train_raw.Week.astype(int)
train_raw.Year = train_raw.Year.astype(int)
# training data from 2010-02 to 2011-02
start_date = pd.to_datetime('2010-02-01')
end_date = start_date + relativedelta(months=13)

# split dataset into training / testing
train_ids = (pd.to_datetime(train_raw['Date']) >= start_date) & (pd.to_datetime(train_raw['Date']) < end_date)
train = train_raw.loc[train_ids, ]
test = train_raw.loc[~train_ids, ]

# create the initial training data
print('exportint train_ini.csv')
train.to_csv('train_ini.csv', index=False)

# create 10 time-series
num_folds = 10

# month 1 --> 2011-03, and month 20 --> 2012-10.
# Fold 1 : month 1 & month 2, Fold 2 : month 3 & month 4 ...
print('Making folds')
for i in range(num_folds):
  # filter fold for dates
  start_date = pd.to_datetime('2011-03-01') + relativedelta(months = 2 * i)
  end_date = pd.to_datetime('2011-05-01') + relativedelta(months = 2 * i)
  test_ids = (pd.to_datetime(test['Date']) >= start_date) & (pd.to_datetime(test['Date']) < end_date)
  test_fold = test.loc[test_ids, ]

  # write fold to a file
  test_fold.to_csv('fold_{}.csv'.format(i + 1), index=False)

# create test.csv
# removes weekly sales
test = test.drop(columns=['Weekly_Sales'])
test.to_csv('test.csv')
