import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import logging


from utils import CONFIG, prepare_ml_features


Logger = logging.getLogger(__name__)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_model_path(metric):
    return os.path.join(MODEL_DIR, f"{metric}_model.pkl")

def get_model_date_path(metric):
    return os.path.join(MODEL_DIR, f"{metric}_last_trained.txt")

def should_retrain_model(metric):
    model_path = get_model_path(metric)
    date_path = get_model_date_path(metric)
    if not os.path.exists(model_path) or not os.path.exists(date_path):
        return True
    with open(date_path, "r") as f:
        last_trained = datetime.strptime(f.read().strip(), "%Y-%m-%d")
    return (datetime.now() - last_trained).days >= 15

def save_model(model, metric):
    joblib.dump(model, get_model_path(metric))
    with open(get_model_date_path(metric), "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))

def load_model(metric):
    return joblib.load(get_model_path(metric))

def train_and_save_model(df, metric, forecast_horizon, forecast_start=None, forecast_end=None):
    features_df, _ = prepare_ml_features(df, metric, forecast_horizon, forecast_start, forecast_end)
    features_cols = [col for col in features_df.columns if (
        ('lag' in col or 'rolling' in col or col in ['day_of_week', 'month', 'is_weekend', 'lead_time'])
    )]
    models = {}
    for dorm in df.index.get_level_values('Dorm').unique():
        train_data = features_df[features_df['Dorm'] == dorm]
        x_train = train_data[features_cols]
        y_train = train_data[metric]
        if len(x_train) < 3:
            Logger.warning(f"Skipping dorm {dorm}: not enough samples for CV (n={len(x_train)})")
            continue
        if metric in ['no_show_rate', 'cancellation_rate']:
            # Classification: predict probability
            y_train_bin = (y_train > 0).astype(int)
            xgb = XGBClassifier(
                random_state=CONFIG['rf_random_state'],
                n_jobs=-1,
                eval_metric='logloss'
            )
            param_dist = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [3, 5, 7, 10, 12],
                'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3],
                'min_child_weight': [1, 3, 5, 7]
            }
            search = RandomizedSearchCV(
                xgb,
                param_distributions=param_dist,
                n_iter=20,
                scoring='roc_auc',
                cv=3,
                verbose=0,
                n_jobs=-1
            )
            search.fit(x_train, y_train_bin)
            models[dorm] = (search.best_estimator_, features_cols, 'classifier')
        else:
            xgb = XGBRegressor(
                random_state=CONFIG['rf_random_state'],
                n_jobs=-1,
                objective='reg:squarederror'
            )
            param_dist = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [3, 5, 7, 10, 12],
                'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3],
                'min_child_weight': [1, 3, 5, 7]
            }
            search = RandomizedSearchCV(
                xgb,
                param_distributions=param_dist,
                n_iter=20,
                scoring='neg_mean_absolute_error',
                cv=3,
                verbose=0,
                n_jobs=-1
            )
            search.fit(x_train, y_train)
            models[dorm] = (search.best_estimator_, features_cols, 'regressor')
    save_model(models, metric)
    Logger.info(f"Model for {metric} trained and saved.")
    return models

def forecast_with_saved_model(df, metric, forecast_horizon, forecast_start=None, forecast_end=None):
    models = load_model(metric)
    _, forecast_features_df = prepare_ml_features(df, metric, forecast_horizon, forecast_start, forecast_end)
    forecast_data = []
    for dorm in df.index.get_level_values('Dorm').unique():
        if dorm not in models:
            continue
        model, features_cols, model_type = models[dorm]
        forecast_data_dorm = forecast_features_df[forecast_features_df['Dorm'] == dorm]
        X_forecast = forecast_data_dorm[features_cols]
        if metric in ['no_show_rate', 'cancellation_rate'] and model_type == 'classifier':
            predictions = model.predict_proba(X_forecast)[:, 1]  # Probability of event
        else:
            predictions = model.predict(X_forecast)
        dorm_forecast_df = pd.DataFrame({
            'Dorm': dorm,
            'Date': forecast_data_dorm['Date'].values,
            metric: predictions
        })
        if metric == 'occupancy_rate':
            dorm_forecast_df['occupancy_rate'] = dorm_forecast_df['occupancy_rate'].clip(upper=1.0)
        forecast_data.append(dorm_forecast_df)
    if forecast_data:
        return pd.concat(forecast_data).reset_index(drop=True)
    else:
        return pd.DataFrame()
def train_or_load_and_forecast(df, metric, forecast_horizon, forecast_start=None, forecast_end=None):
    if should_retrain_model(metric):
        train_and_save_model(df, metric, forecast_horizon, forecast_start, forecast_end)
    return forecast_with_saved_model(df, metric, forecast_horizon, forecast_start, forecast_end)