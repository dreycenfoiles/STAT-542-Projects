import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 

df = pd.read_csv("https://liangfgithub.github.io/Data/Ames_data.csv")
testID = pd.read_csv(
    'https://liangfgithub.github.io/Data/project1_testIDs.dat',delim_whitespace=' ',header=None)


def numeric_convert(frame):
    # We may want to normalize data as well 
    for col in frame:
        try:
            frame[col] = pd.to_numeric(frame[col])
        except:
            frame[col] = pd.factorize(frame[col])[0] + 1
    
    return frame

def get_split(frame, index):

    frame = frame.drop('Garage_Yr_Blt', axis=1)

    num_rows = np.arange(len(frame))

    test_index = testID.iloc[:,index]
    train_index = np.array([i for i in num_rows if i not in test_index])

    xtest = numeric_convert(frame.iloc[test_index,1:-1].copy())
    xtrain = numeric_convert(frame.iloc[train_index,1:-1].copy())

    # convert to log to get better model
    ytest = np.log(frame.iloc[test_index,-1].copy())
    ytrain = np.log(frame.iloc[train_index,-1].copy())

    return xtrain,xtest,ytrain,ytest


def clean_data(df, cols):
    for col in cols:
        df = df.drop(col, axis=1)
    return df


bad_cols = ['PID', 'Garage_Yr_Blt']
# Set up data for use with scikit-learn
frame = clean_data(df, bad_cols)
cvsplits = []
num_rows = np.arange(len(frame))
for index in range(0,10):
    test_index = testID.iloc[:,index]
    train_index = np.array([i for i in num_rows if i not in test_index])
    cvsplits.append((train_index, test_index.values))


scaler = MinMaxScaler()

x = numeric_convert(frame.iloc[:,:-1].copy())
# x = scaler.fit_transform(x)
# for c in x:
#     x[c] = x[c] / np.max(np.abs(x[c]))
    # x[c] = scaler.fit_transform(x[c])

# convert to log to get better model
y = np.log(frame.iloc[:,-1].copy())


split_number = 0

xtrain = x.values[cvsplits[split_number][0]]
xtest = x.values[cvsplits[split_number][1]]

ytrain = y.values[cvsplits[split_number][0]]
ytest = y.values[cvsplits[split_number][1]]


