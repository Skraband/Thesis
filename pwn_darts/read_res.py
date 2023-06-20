import pandas as pd

name = "res/experiments/06_13_2023_04_39_05__PWNEM-ReadM4_Yearly.pkl"
unpickled_df = pd.read_pickle(name)
print(unpickled_df[3]['SMAPE'])

y=0
for x in unpickled_df[3]['SMAPE']:
    y += unpickled_df[3]['SMAPE'][x]

#y = y/4
y = y/23000
print(y)