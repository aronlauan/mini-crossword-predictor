import pandas as pd
import datetime as dt
import numpy as np
from datetime import timedelta


def melt_raw_file(filepath: str) -> pd.DataFrame:
    """
    Args:
        filepath (str): path to exported stats .csv

    Returns:
        pd.DataFrame: converted dataframe
    """
    df = pd.read_csv(filepath)
    if df.columns[0] != 'date':
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    
    df_long = pd.melt(df, id_vars='date', var_name='person', value_name='time')
    df_long['date'] = pd.to_datetime(df_long['date'])
    df_long['time'] = df_long['time'].str.strip()

    df_long = df_long[df_long['time'].notna()]
    df_long = df_long[df_long['time'].str.lower() != 'n/a']

    return df_long


def min_to_sec(raw_time: str) -> float:
    """
    Args:
        raw_time: string of min:sec time

    Returns:
        int: min:sec converted to seconds
    """
    try:
        minutes, seconds = map(int, raw_time.split(':'))
        return minutes * 60 + seconds
    except:
        return np.nan
    

def sec_to_min(seconds: int) -> str:
    """
    Args:
        seconds (int): time in seconds

    Returns:
        str: time in min:sec as str
    """
    minutes, seconds = divmod(int(seconds), 60)
    return f'{minutes}:{seconds:02d}'


def add_features(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        metrics_df (pd.DataFrame): converted dataframe

    Returns:
        pd.DataFrame: dataframe with new stat features
    """
    metrics_df['weekday'] = metrics_df['date'].dt.day_name()
    
    difficulty = {
        'Monday': 1,
        'Tuesday': 2,
        'Wednesday': 3,
        'Thursday': 4,
        'Friday': 5,
        'Saturday': 7,
        'Sunday': 6
    }
    metrics_df['difficulty'] = metrics_df['weekday'].map(difficulty)
    metrics_df['solved_in_seconds'] = metrics_df['time'].apply(min_to_sec)

    # Sort for rolling calculations
    metrics_df =  metrics_df.sort_values(by=['person', 'date'])
    
    metrics_df['avg_last_5'] = (
        metrics_df
        .groupby('person')['solved_in_seconds']
        .shift(1)  # exclude today's time
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Streak calculation
    metrics_df['streak'] = 0

    for user, group in metrics_df.groupby('person'):
        streaks = []
        last_date = None
        current_streak = 0
        for idx, row in group.iterrows():
            if last_date is not None and (row['date'] - last_date).days == 1:
                current_streak += 1
            else:
                current_streak = 1
            streaks.append(current_streak)
            last_date = row['date']
        metrics_df.loc[group.index, 'streak'] = streaks

    return metrics_df


if __name__ == "__main__":
    df = melt_raw_file('mini_stats.csv')
    metrics_df = add_features(df)
    print(metrics_df)