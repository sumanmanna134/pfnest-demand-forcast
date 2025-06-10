from typing import Tuple
import pandas as pd
import numpy as np
import logging

Logger = logging.getLogger(__name__)

CONFIG = {
    'min_data_days': 29,
    'rf_n_estimators': 200,
    'rf_random_state': 42,
    'seasonal_period': 7,
    'required_columns': {'Check-in', 'Check-out', 'Price', 'Status', 'Unit type', 'Duration (nights)', 'Cancellation date', 'People'},
    'default_dorm_types': ['MaleDorm', 'FemaleDorm', 'MixedDorm'],
    'default_rooms_per_dorm': 4
}

def prepare_ml_features(
    df: pd.DataFrame,
    metric: str,
    forecast_horizon: int = None,
    forecast_start: pd.Timestamp = None,
    forecast_end: pd.Timestamp = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    Logger.info(f"Preparing features for metric: {metric} with forecast horizon: {forecast_horizon}")
    all_metrics = ['occupancy_rate', 'cancellation_rate', 'no_show_rate', 'Revenue']
    features_list = []
    # --- Add lead_time feature ---
    # If booking_date exists, use it; else fallback to check_in
    if 'booking_date' in df.columns:
        df['booking_date'] = pd.to_datetime(df['booking_date'], errors='coerce')
    else:
        df['booking_date'] = df.index.get_level_values('Date')
    if 'check_in' in df.columns:
        df['check_in'] = pd.to_datetime(df['check_in'], errors='coerce')
    else:
        df['check_in'] = df.index.get_level_values('Date')
    df['lead_time'] = (df['check_in'] - df['booking_date']).dt.days.clip(lower=0)
    # ----------------------------

    for dorm in df.index.get_level_values('Dorm').unique():
        df_dorm = df.xs(dorm, level='Dorm').copy()
        # Calendar/time features
        df_dorm['day_of_week'] = df_dorm.index.dayofweek
        df_dorm['month'] = df_dorm.index.month
        df_dorm['is_weekend'] = df_dorm['day_of_week'].isin([5, 6]).astype(int)
        df_dorm['day_of_month'] = df_dorm.index.day
        df_dorm['quarter'] = df_dorm.index.quarter
        df_dorm['week_of_year'] = df_dorm.index.isocalendar().week
        df_dorm['is_month_start'] = df_dorm.index.is_month_start.astype(int)
        df_dorm['is_month_end'] = df_dorm.index.is_month_end.astype(int)
        df_dorm['days_since_start'] = (df_dorm.index - df_dorm.index.min()).days
        # Lag/rolling features for all metrics
        for m in all_metrics:
            for lag in [1, 7, 14]:
                df_dorm[f'{m}_lag_{lag}'] = df_dorm[m].shift(lag)
            df_dorm[f'{m}_rolling_mean_7'] = df_dorm[m].rolling(window=7).mean()
            df_dorm[f'{m}_rolling_mean_30'] = df_dorm[m].rolling(window=30).mean()
            df_dorm[f'{m}_rolling_std_7'] = df_dorm[m].rolling(window=7).std()
            df_dorm[f'{m}_rolling_min_7'] = df_dorm[m].rolling(window=7).min()
            df_dorm[f'{m}_rolling_max_7'] = df_dorm[m].rolling(window=7).max()
        # Interaction features
        df_dorm['occ_x_cancel'] = df_dorm['occupancy_rate'] * df_dorm['cancellation_rate']
        df_dorm['occ_x_no_show'] = df_dorm['occupancy_rate'] * df_dorm['no_show_rate']
        # Lead time feature (already added above, but ensure it's present)
        if 'lead_time' not in df_dorm.columns:
            df_dorm['lead_time'] = 0
        df_dorm['Dorm'] = dorm
        features_list.append(df_dorm)
    features_df = pd.concat(features_list).dropna()

    # Only use data up to the forecast_start for training and feature generation
    if forecast_start is not None:
        features_df = features_df[features_df.index.get_level_values('Date') < pd.to_datetime(forecast_start)]

    # Prepare forecast features
    if forecast_start is not None and forecast_end is not None:
        forecast_dates = pd.date_range(forecast_start, forecast_end, freq='D')
    elif forecast_horizon is not None:
        last_date = features_df.index.get_level_values('Date').max()
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
    else:
        last_date = features_df.index.get_level_values('Date').max()
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7, freq='D')

    forecast_features = []
    for dorm in df.index.get_level_values('Dorm').unique():
        last_data = df.xs(dorm, level='Dorm')
        last_data = last_data[last_data.index < pd.to_datetime(forecast_start)]
        if last_data.empty:
            continue
        for date in forecast_dates:
            last_rows = last_data.tail(30)
            last_row = last_rows.iloc[-1].copy()
            # Calendar/time features for forecast date
            last_row['day_of_week'] = date.dayofweek
            last_row['month'] = date.month
            last_row['is_weekend'] = int(date.dayofweek in [5, 6])
            last_row['day_of_month'] = date.day
            last_row['quarter'] = date.quarter
            last_row['week_of_year'] = date.isocalendar()[1]
            last_row['is_month_start'] = int(date.is_month_start)
            last_row['is_month_end'] = int(date.is_month_end)
            last_row['days_since_start'] = (date - last_data.index.min()).days
            for m in all_metrics:
                last_row[f'{m}_lag_1'] = last_rows[m].iloc[-1]
                last_row[f'{m}_lag_7'] = last_rows[m].iloc[-7] if len(last_rows) >= 7 else last_rows[m].iloc[-1]
                last_row[f'{m}_lag_14'] = last_rows[m].iloc[-14] if len(last_rows) >= 14 else last_rows[m].iloc[-1]
                last_row[f'{m}_rolling_mean_7'] = last_rows[m].rolling(window=7).mean().iloc[-1]
                last_row[f'{m}_rolling_mean_30'] = last_rows[m].rolling(window=30).mean().iloc[-1]
                last_row[f'{m}_rolling_std_7'] = last_rows[m].rolling(window=7).std().iloc[-1]
                last_row[f'{m}_rolling_min_7'] = last_rows[m].rolling(window=7).min().iloc[-1]
                last_row[f'{m}_rolling_max_7'] = last_rows[m].rolling(window=7).max().iloc[-1]
            last_row['occ_x_cancel'] = last_row['occupancy_rate'] * last_row['cancellation_rate']
            last_row['occ_x_no_show'] = last_row['occupancy_rate'] * last_row['no_show_rate']
            # Lead time for forecast: assume average of last 30 days
            last_row['lead_time'] = last_rows['lead_time'].mean() if 'lead_time' in last_rows else 0
            last_row['Date'] = date
            last_row['Dorm'] = dorm
            forecast_features.append(last_row)
    forecast_features_df = pd.DataFrame(forecast_features)
    return features_df, forecast_features_df