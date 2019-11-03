"""Script to analyze results.csv."""

import pandas as pd
import numpy as np

df = pd.read_csv('results.csv')

df['VAL_ERROR'] = (df['VAL_ERROR'] * 100).round(2)
df['TRAIN_ERROR'] = (df['TRAIN_ERROR'] * 100).round(2)

print('\nAnalyzing results.csv...\n\n')
print('Best experiment :\n')
print(df.iloc[df['VAL_ERROR'].idxmin()])


print('\n\nMinimum of all experiments :\n')
print(pd.pivot_table(df, values=['VAL_ERROR', 'TRAIN_ERROR'],
                     columns='MODEL', aggfunc=np.min))

print('\n\nAverage of all experiments :\n')
df2 = pd.pivot_table(df, values=['VAL_ERROR', 'TRAIN_ERROR'],
                     columns='MODEL', aggfunc=np.mean)
print(df2.round(2))
