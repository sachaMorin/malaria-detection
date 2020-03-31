"""Script to analyze random search results."""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPORT_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../reports'
)

all_files = list()

for f in os.listdir(REPORT_PATH):
    if f.endswith('.csv'):
        all_files.append(os.path.join(REPORT_PATH, f))


df = pd.concat((pd.read_csv(f) for f in all_files))

top_20 = df.sort_values(by='mean_test_score', ascending=False).iloc[0:20]

# pivot = pd.pivot_table(top_20,
#                        values='mean_test_score',
#                        index='param_lr',
#                        columns='param_batch_size',
#                        aggfunc=np.mean)
#
# print(pivot)
#
# # Analyser les résultats pour 0.001, Adam et batch_size = 64
# mask = (df['param_lr'] == 0.001) & \
#        (df['param_batch_size'] == 64) & \
#        (df['param_optimizer'] == "<class 'torch.optim.adam.Adam'>")
#
# df2 = df[mask]
# df2.plot(x='param_max_epochs', y='mean_test_score',
#          style='o', title='0.001 - 64')
# plt.show()
#
# # Analyser les résultats pour 0.001, Adam et batch_size = 128
# mask = (df['param_lr'] == 0.001) & \
#        (df['param_batch_size'] == 128) & \
#        (df['param_optimizer'] == "<class 'torch.optim.adam.Adam'>")
#
# df2 = df[mask]
# df2.plot(x='param_max_epochs', y='mean_test_score',
#          style='o', title='0.001 - 128')
# plt.show()
#
# # Analyser les résultats pour 0.01, Adam et batch_size = 32
# mask = (df['param_lr'] == 0.01) & \
#        (df['param_batch_size'] == 32) & \
#        (df['param_optimizer'] == "<class 'torch.optim.adam.Adam'>")
#
# df3 = df[mask]
# df3.plot(x='param_max_epochs', y='mean_test_score',
#          style='o', title='0.01 - 32')
# plt.show()
