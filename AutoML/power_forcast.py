import numpy as np
import warnings
import pickle
warnings.simplefilter(action='ignore', category=FutureWarning)

from data_source import BasicSelect, Mackey, ReadPowerPKL, ReadM4
from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask
import pandas as pd
from preprocessing import prepare_power



data = ReadPowerPKL().data
# Dataset optimized by APT-TS can be a list of np.ndarray/ pd.DataFrame where each series represents an element in the
# list, or a single pd.DataFrame that records the series
# index information: to which series the timestep belongs? This id can be stored as the DataFrame's index or a separate
# column
window_size = 96
context_timespan = int(15*96)
prediction_timespan = int(1.5*96)
X_train, y_train, X_test, y_test, start_times = prepare_power.preparingPowerForAPT(data,window_size, context_timespan, prediction_timespan)

known_future_features = X_train[0].keys()

freq = "15T"
api = TimeSeriesForecastingTask()
# Search for an ensemble of machine learning algorithms
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    optimize_metric='mean_MAPE_forecasting',
    n_prediction_steps=144,
    memory_limit=16 * 1024,  # Currently, forecasting models use much more memories
    freq=freq,
    start_times=start_times,
    func_eval_time_limit_secs=50,
    total_walltime_limit=60,
    min_num_test_instances=1000,  # proxy validation sets. This only works for the tasks with more than 1000 series
    known_future_features=known_future_features,
)

print('done')

