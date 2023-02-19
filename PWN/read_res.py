import pandas as pd


name = "res/experiments/02_13_2023_15_19_34__PWNEM-ReadPowerPKL.pkl"
unpickled_df = pd.read_pickle(name)
print(unpickled_df[3]['MSE'])
print(unpickled_df[3]['MAE'])