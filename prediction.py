import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import logging

from preprocess import *

### Configure logger ###
logger = logging.getLogger()
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()

log_format = "%(asctime)s - %(message)s"
formatter = logging.Formatter(
    log_format,
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
#########################

# --- START --- #
raw_df = melt_raw_file('mini_stats.csv')
metrics_df = add_features(raw_df)

features = ['person', 'weekday', 'difficulty', 'avg_last_3', 'streak']
target = 'solved_in_seconds'

# drop rows with a rolling avg of NULL
model_df = metrics_df.dropna(subset=['avg_last_3']).copy()

X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# pipeline
categorical = ['person', 'weekday']
numerical = ['difficulty', 'avg_last_3', 'streak']

preprocessor = ColumnTransformer(
    [('cat', OneHotEncoder(handle_unknown='ignore'), categorical)],
    remainder='passthrough'
    )

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

print(f'\nModel Performance:')
print(f'MAE: {mae:.2f} sec')
print(f'RMSE: {rmse:.2f} sec')


print("\nSample Predictions:")
sample_preds = pd.DataFrame({
    'date': model_df.loc[y_test.index, 'date'],
    'person': X_test['person'].values,
    'actual': [sec_to_min(sec) for sec in y_test],
    'predicted': [sec_to_min(sec) for sec in y_pred]
})
print(sample_preds.head(10))


def predict_tomorrow(data, model_pipeline: Pipeline) -> pd.DataFrame:
    """_
    Args:
        data (_type_): data df
        model_pipline (Pipeline): trained pipeline

    Returns:
        pd.DataFrame: predictions df
    """
    predictions = []

    persons = data['person'].unique()

    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    weekday = tomorrow.strftime('%A')  # Get the weekday name for tomorrow

    for person in persons:
        person_data = data[data['person'] == person].sort_values(by='date', ascending=False)

        # Only use completed solves (i.e., non 'N/A' or missing values)
        completed_data = person_data[person_data['time'].notna() & (person_data['time'].str.lower() != 'n/a')]

        # Calculate rolling average of the last 3 completed solves (if available)
        avg_last_3 = completed_data.tail(3)['solved_in_seconds'].mean() if len(completed_data) >= 3 else completed_data['solved_in_seconds'].mean()

        difficulty = {
            'Monday': 1,
            'Tuesday': 2,
            'Wednesday': 3,
            'Thursday': 4,
            'Friday': 5,
            'Saturday': 7,
            'Sunday': 6
        }.get(weekday, 3)  # Default difficulty is 'Wednesday' if no match

        input_data = pd.DataFrame({
            'person': [person],
            'weekday': [weekday],
            'difficulty': [difficulty],
            'avg_last_3': [avg_last_3],
            'streak': [completed_data['streak'].iloc[-1] if len(completed_data) > 0 else 0]
        })

        predicted_seconds = model_pipeline.predict(input_data)[0]
        predicted_time = sec_to_min(predicted_seconds)

        predictions.append({
            'person': person,
            'predicted_time': predicted_time,
            'predicted_seconds': predicted_seconds,
            'weekday': weekday,
            'predicted_for_date': tomorrow
        })

    predictions_df = pd.DataFrame(predictions)

    return predictions_df


if __name__ == "__main__":
    prediction_df = predict_tomorrow(metrics_df, pipeline)
    print(prediction_df[['person', 'predicted_time']])