from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask

# data and metric imports
from sktime.datasets import load_longley
targets, features = load_longley()

# define the forecasting horizon
forecasting_horizon = 3

# Dataset optimized by APT-TS can be a list of np.ndarray/ pd.DataFrame where each series represents an element in the
# list, or a single pd.DataFrame that records the series
# index information: to which series the timestep belongs? This id can be stored as the DataFrame's index or a separate
# column
# Within each series, we take the last forecasting_horizon as test targets. The items before that as training targets
# Normally the value to be forecasted should follow the training sets
y_train = [targets[: -forecasting_horizon]]
y_test = [targets[-forecasting_horizon:]]

# same for features. For uni-variant models, X_train, X_test can be omitted and set as None
X_train = [features[: -forecasting_horizon]]
# Here x_test indicates the 'known future features': they are the features known previously, features that are unknown
# could be replaced with NAN or zeros (which will not be used by our networks). If no feature is known beforehand,
# we could also omit X_test
known_future_features = list(features.columns)
X_test = [features[-forecasting_horizon:]]

start_times = [targets.index.to_timestamp()[0]]
freq = '1Y'

# initialise Auto-PyTorch api
api = TimeSeriesForecastingTask()

# Search for an ensemble of machine learning algorithms
api.search(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    optimize_metric='mean_MAPE_forecasting',
    n_prediction_steps=forecasting_horizon,
    memory_limit=16 * 1024,  # Currently, forecasting models use much more memories
    freq=freq,
    start_times=start_times,
    func_eval_time_limit_secs=50,
    total_walltime_limit=60,
    min_num_test_instances=1000,  # proxy validation sets. This only works for the tasks with more than 1000 series
    known_future_features=known_future_features,
)

# our dataset could directly generate sequences for new datasets
test_sets = api.dataset.generate_test_seqs()

# Calculate test accuracy
y_pred = api.predict(test_sets)
#score = api.score(y_pred, y_test)
print("Done")
