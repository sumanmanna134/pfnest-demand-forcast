import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from typing import Optional, Tuple, Dict
import io
import logging
import os
from pathlib import Path
import yaml
from streamlit.runtime.uploaded_file_manager import UploadedFile
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from train import train_and_save_model,should_retrain_model, train_or_load_and_forecast
from utils import CONFIG

# Configuration dictionary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

Logger = logging.getLogger(__name__)



st.set_page_config(
    page_title="PathfindersNest Booking Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

def normalize_dorm_columns(df: pd.DataFrame) -> pd.DataFrame:
    unit_map = {
        '4-Bed Male Dormitory Room': 'MaleDorm',
        'Bed in Male Dormitory Room': 'MaleDorm',
        '4-Bed Female Dormitory Room': 'FemaleDorm',
        '4-Bed Mixed Dormitory Room': 'MixedDorm',
        'Deluxe Capsule Bed in Mixed Dorm': 'MixedDorm',
        'Deluxe Capsule Bed in Female Dorm': 'FemaleDorm',
    }
    def map_unit_type(value: str) -> str:
        parts = [part.strip() for part in value.split(',')]
        mapped_parts = [unit_map.get(part, part) for part in parts]
        return ', '.join(mapped_parts)
    df['dorm'] = df['dorm'].astype(str).apply(map_unit_type)
    return df

@st.cache_data
def load_data(file: UploadedFile) -> Optional[pd.DataFrame]:
    try:
        file_ext = file.name.split('.')[-1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(file)
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(file, engine='openpyxl' if file_ext == 'xlsx' else None)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            Logger.error(f"Unsupported file format: {file_ext}")
            return None
        if not CONFIG['required_columns'].issubset(set(df.columns)):
            missing_cols = CONFIG['required_columns'] - set(df.columns)
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            Logger.error(f"Missing required columns: {missing_cols}")
            return None
        df = df.copy()
        df['Check-in'] = pd.to_datetime(df['Check-in'], errors='coerce')
        df['Check-out'] = pd.to_datetime(df['Check-out'], errors='coerce')
        df['Cancellation date'] = pd.to_datetime(df['Cancellation date'], errors='coerce')
        df['People'] = pd.to_numeric(df['People'], errors='coerce').fillna(0).astype(int)
        df['Price'] = pd.to_numeric(df['Price'].astype(str).str.extract(r'([\d\.]+)')[0], errors='coerce')
        df['Duration (nights)'] = pd.to_numeric(df['Duration (nights)'], errors='coerce').fillna(0)
        df['is_cancelled'] = df['Status'].str.lower().str.contains('cancelled_by_guest', na=False)
        df['is_no_show'] = df['Status'].str.lower().str.contains('no_show', na=False)
        df.rename(columns={
            'Book Number': 'booking_number',
            'Check-in': 'check_in',
            'Check-out': 'check_out',
            'Price': 'price',
            'Status': 'status',
            'Unit type': 'dorm',
            'Duration (nights)': 'nights',
            'Cancellation date': 'cancellation_date',
            'People': 'no_of_dorms'
        }, inplace=True)
        df = normalize_dorm_columns(df)
        Logger.info(f"Data loaded successfully with {len(df)} records.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        Logger.error(f"Error loading data: {e}")
        return None

# def forecast_metrics(df: pd.DataFrame, forecast_horizon: int, metric: str, forecast_start: pd.Timestamp = None, forecast_end: pd.Timestamp = None) -> pd.DataFrame:
#     try:
#         if len(df) < CONFIG['min_data_days']:
#             st.error(f"Insufficient data for forecasting. Minimum {CONFIG['min_data_days']} days required.")
#             Logger.error(f"Insufficient data for forecasting. Minimum {CONFIG['min_data_days']} days required.")
#             return None
#         features_df, forecast_features_df = prepare_ml_features(df, metric, forecast_horizon, forecast_start=forecast_start, forecast_end=forecast_end)
#         features_cols = [col for col in features_df.columns if (
#             ('lag' in col or 'rolling' in col or col in ['day_of_week', 'month', 'is_weekend'])
#         )]
#         forecast_data = []
#         for dorm in df.index.get_level_values('Dorm').unique():
#             train_data = features_df[features_df['Dorm'] == dorm]
#             x_train = train_data[features_cols]
#             y_train = train_data[metric]

#             # Hyperparameter grid for XGBoost
#             param_dist = {
#                 'n_estimators': [100, 200, 300],
#                 'max_depth': [3, 5, 7, 10],
#                 'learning_rate': [0.01, 0.05, 0.1, 0.2],
#                 'subsample': [0.7, 0.8, 1.0],
#                 'colsample_bytree': [0.7, 0.8, 1.0]
#             }
#             xgb = XGBRegressor(
#                 random_state=CONFIG['rf_random_state'],
#                 n_jobs=-1,
#                 objective='reg:squarederror'
#             )
#             search = RandomizedSearchCV(
#                 xgb,
#                 param_distributions=param_dist,
#                 n_iter=10,
#                 scoring='neg_mean_absolute_error',
#                 cv=3,
#                 verbose=0,
#                 n_jobs=-1
#             )
#             search.fit(x_train, y_train)
#             best_model = search.best_estimator_

#             forecast_data_dorm = forecast_features_df[forecast_features_df['Dorm'] == dorm]
#             X_forecast = forecast_data_dorm[features_cols]
#             predictions = best_model.predict(X_forecast)
#             dorm_forecast_df = pd.DataFrame({
#                 'Dorm': dorm,
#                 'Date': forecast_data_dorm['Date'].values,
#                 metric: predictions
#             })

#             # Cap occupancy_rate at 1.0 if metric is occupancy_rate
#             if metric == 'occupancy_rate':
#                 dorm_forecast_df['occupancy_rate'] = dorm_forecast_df['occupancy_rate'].clip(upper=1.0)

#             forecast_data.append(dorm_forecast_df)
#         forecast_df = pd.concat(forecast_data).reset_index(drop=True)
#         Logger.info(f"Forecasting completed for {metric} with {forecast_horizon} days horizon.")
#         return forecast_df
#     except Exception as e:
#         st.error(f"Error during forecasting: {e}")
#         Logger.error(f"Error during forecasting: {e}")
#         return None

def calculate_daily_metrics(df: pd.DataFrame, dorm_rooms: Dict[str, int]) -> pd.DataFrame:
    if df.empty:
        st.warning("No data available for calculations.")
        Logger.warning("No data available for calculations.")
        return pd.DataFrame()
    df['check_in'] = pd.to_datetime(df['check_in'], errors='coerce')
    df['check_out'] = pd.to_datetime(df['check_out'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['nights'] = pd.to_numeric(df['nights'], errors='coerce').fillna(1)
    df['no_of_dorms'] = pd.to_numeric(df['no_of_dorms'], errors='coerce').fillna(1)
    df = df.dropna(subset=['check_in', 'check_out'])
    if df.empty:
        st.warning("No valid data available after filtering invalid dates.")
        Logger.warning("No valid data available after filtering invalid dates.")
        return pd.DataFrame()
    min_date = df['check_in'].min()
    max_date = df['check_out'].max()
    date_range = pd.date_range(min_date, max_date, freq='D')
    daily_data = []
    def dorm_match(dorm_col_val, dorm):
        return dorm in [d.strip() for d in str(dorm_col_val).split(',')]
    with st.spinner("Calculating daily metrics..."):
        for date in date_range:
            for dorm, total_rooms in dorm_rooms.items():
                mask = (
                    df['check_in'] <= date
                ) & (
                    df['check_out'] > date
                ) & (
                    df['dorm'].apply(lambda x: dorm_match(x, dorm))
                )
                dorm_bookings = df[mask]
                active = dorm_bookings[
                    (~dorm_bookings['is_cancelled']) &
                    (~dorm_bookings['is_no_show'])
                ]
                cancelled = dorm_bookings[dorm_bookings['is_cancelled']]
                no_show = df[
                    (df['check_in'] == date) &
                    (df['dorm'].apply(lambda x: dorm_match(x, dorm))) &
                    (df['is_no_show'])
                ]
                booked_dorms = active['no_of_dorms'].sum()
                cancelled_dorms = cancelled['no_of_dorms'].sum()
                no_show_dorms = no_show['no_of_dorms'].sum()
                occupancy_rate = min(booked_dorms / total_rooms, 1.0) if total_rooms > 0 else 0
                cancellation_rate = cancelled_dorms / (booked_dorms + cancelled_dorms) if (booked_dorms + cancelled_dorms) > 0 else 0
                no_show_rate = no_show_dorms / (booked_dorms + no_show_dorms) if (booked_dorms + no_show_dorms) > 0 else 0
                revenue = active['price'].sum()
                total_nights = active['nights'].sum()
                daily_revenue = (revenue / total_nights) if total_nights > 0 else 0
                daily_data.append({
                    'Date': date,
                    'Dorm': dorm,
                    'occupancy_rate': occupancy_rate,
                    'cancellation_rate': cancellation_rate,
                    'no_show_rate': no_show_rate,
                    'Revenue': revenue,
                    'Daily Revenue': daily_revenue,
                    'booked_dorms': booked_dorms
                })
    daily_df = pd.DataFrame(daily_data).set_index(['Date', 'Dorm'])
    Logger.info("Daily metrics calculated successfully.")
    return daily_df


# def safe_forecast_metrics(df: pd.DataFrame, forecast_horizon: int, metric: str, forecast_start: pd.Timestamp = None, forecast_end: pd.Timestamp = None) -> pd.DataFrame:
#     """
#     Like forecast_metrics, but skips dorms with less than 3 samples to avoid CV errors.
#     """
#     try:
#         if len(df) < CONFIG['min_data_days']:
#             st.error(f"Insufficient data for forecasting. Minimum {CONFIG['min_data_days']} days required.")
#             Logger.error(f"Insufficient data for forecasting. Minimum {CONFIG['min_data_days']} days required.")
#             return None
#         features_df, forecast_features_df = prepare_ml_features(df, metric, forecast_horizon, forecast_start=forecast_start, forecast_end=forecast_end)
#         features_cols = [col for col in features_df.columns if (
#             ('lag' in col or 'rolling' in col or col in ['day_of_week', 'month', 'is_weekend'])
#         )]
#         forecast_data = []
#         for dorm in df.index.get_level_values('Dorm').unique():
#             train_data = features_df[features_df['Dorm'] == dorm]
#             x_train = train_data[features_cols]
#             y_train = train_data[metric]

#             # Skip if not enough samples for CV
#             if len(x_train) < 3:
#                 Logger.warning(f"Skipping dorm {dorm} for this window: not enough samples for CV (n={len(x_train)})")
#                 continue

#             param_dist = {
#                 'n_estimators': [100, 200, 300],
#                 'max_depth': [3, 5, 7, 10],
#                 'learning_rate': [0.01, 0.05, 0.1, 0.2],
#                 'subsample': [0.7, 0.8, 1.0],
#                 'colsample_bytree': [0.7, 0.8, 1.0]
#             }
#             xgb = XGBRegressor(
#                 random_state=CONFIG['rf_random_state'],
#                 n_jobs=-1,
#                 objective='reg:squarederror'
#             )
#             search = RandomizedSearchCV(
#                 xgb,
#                 param_distributions=param_dist,
#                 n_iter=10,
#                 scoring='neg_mean_absolute_error',
#                 cv=3,
#                 verbose=0,
#                 n_jobs=-1
#             )
#             search.fit(x_train, y_train)
#             best_model = search.best_estimator_

#             forecast_data_dorm = forecast_features_df[forecast_features_df['Dorm'] == dorm]
#             X_forecast = forecast_data_dorm[features_cols]
#             predictions = best_model.predict(X_forecast)
#             dorm_forecast_df = pd.DataFrame({
#                 'Dorm': dorm,
#                 'Date': forecast_data_dorm['Date'].values,
#                 metric: predictions
#             })

#             if metric == 'occupancy_rate':
#                 dorm_forecast_df['occupancy_rate'] = dorm_forecast_df['occupancy_rate'].clip(upper=1.0)

#             forecast_data.append(dorm_forecast_df)
#         if forecast_data:
#             forecast_df = pd.concat(forecast_data).reset_index(drop=True)
#         else:
#             forecast_df = pd.DataFrame()
#         Logger.info(f"Safe forecasting completed for {metric} with {forecast_horizon} days horizon.")
#         return forecast_df
#     except Exception as e:
#         st.error(f"Error during safe forecasting: {e}")
#         Logger.error(f"Error during safe forecasting: {e}")
#         return None

# def safe_rolling_forecast_metrics(df, metric, window_size=30):
#     """
#     Like rolling_forecast_metrics, but uses safe_forecast_metrics to avoid CV errors.
#     """
#     all_forecasts = []
#     min_date = df.index.get_level_values('Date').min()
#     max_date = df.index.get_level_values('Date').max()
#     current_start = min_date

#     while current_start < max_date:
#         current_end = min(current_start + pd.Timedelta(days=window_size-1), max_date)
#         train_df = df[df.index.get_level_values('Date') <= current_end]
#         forecast_start = current_end + pd.Timedelta(days=1)
#         forecast_end = min(forecast_start + pd.Timedelta(days=window_size-1), max_date)
#         if forecast_start > max_date:
#             break
#         forecast_horizon = (forecast_end - forecast_start).days + 1
#         forecast_df = safe_forecast_metrics(
#             train_df, forecast_horizon, metric, forecast_start, forecast_end
#         )
#         if forecast_df is not None and not forecast_df.empty:
#             all_forecasts.append(forecast_df)
#         current_start += pd.Timedelta(days=window_size)
#     if all_forecasts:
#         return pd.concat(all_forecasts, ignore_index=True)
#     else:
#         return pd.DataFrame()

def create_visualization_forecast(df:pd.DataFrame, forecast_occ_df: pd.DataFrame, forecast_rev_df: pd.DataFrame, forecast_cancel_df: pd.DataFrame, forecast_no_show_df: pd.DataFrame):

    """
    Create visualizations for occupancy, revenue, cancellation, and no-show forecasts.
    
    Args:
        df: Original DataFrame with daily metrics
        forecast_occ_df: Forecasted occupancy DataFrame
        forecast_rev_df: Forecasted revenue DataFrame
        forecast_cancel_df: Forecasted cancellation rate DataFrame
        forecast_no_show_df: Forecasted no_show_rate DataFrame
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add original data traces
    fig.add_trace(go.Scatter(
        x=df.index.get_level_values('Date'),
        y=df['occupancy_rate'],
        mode='lines+markers',
        name='Actual Occupancy Rate',
        line=dict(color='blue')
    ))

    # Add forecast traces
    fig.add_trace(go.Scatter(
        x=forecast_occ_df['Date'],
        y=forecast_occ_df['occupancy_rate'],
        mode='lines+markers',
        name='Forecasted Occupancy Rate',
        line=dict(color='lightblue', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=forecast_rev_df['Date'],
        y=forecast_rev_df['Revenue'],
        mode='lines+markers',
        name='Forecasted Revenue',
        line=dict(color='green', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=forecast_cancel_df['Date'],
        y=forecast_cancel_df['cancellation_rate'],
        mode='lines+markers',
        name='Forecasted Cancellation Rate',
        line=dict(color='red', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=forecast_no_show_df['Date'],
        y=forecast_no_show_df['no_show_rate'],
        mode='lines+markers',
        name='Forecasted no_show_rate',
        line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(
        title="Booking Metrics Forecast",
        xaxis_title="Date",
        yaxis_title="Rate / Revenue",
        legend_title="Metrics",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
def show_dashboard(df: pd.DataFrame):
    """
    Display a professional analytics dashboard using Streamlit and Plotly.
    """
    import streamlit as st
    import plotly.express as px
    import pandas as pd

    st.title("Booking Analytics Dashboard", anchor=False)

    if df.empty:
        st.warning("No data available to display.", icon="⚠️")
        return

    # Convert date if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Metrics
    total_booked_dorms = df['no_of_dorms'].sum()
    total_cancellation = df[df['is_cancelled']]['no_of_dorms'].sum()
    total_no_show = df[df['is_no_show']]['no_of_dorms'].sum()
    total_confirmed = df[(~df['is_cancelled']) & (~df['is_no_show'])]['no_of_dorms'].sum()
    total_nights = df[(~df['is_cancelled']) & (~df['is_no_show'])]['nights'].sum()
    total_revenue = df[(~df['is_cancelled']) & (~df['is_no_show'])]['price'].sum()

    # Commission calculation (assume 15% commission, adjust as needed)
    commission_rate = 0.15
    commission_amount = total_revenue * commission_rate
    net_revenue = total_revenue - commission_amount

    # Net revenue per bed (per dorm night)
    net_revenue_per_bed = net_revenue / total_nights if total_nights else 0

    ok_booking_percentage = (total_confirmed / total_booked_dorms) * 100 if total_booked_dorms else 0
    cancellation_percentage = (total_cancellation / total_booked_dorms) * 100 if total_booked_dorms else 0
    no_show_percentage = (total_no_show / total_booked_dorms) * 100 if total_booked_dorms else 0
    # Enhanced Metric Cards CSS
    st.markdown("""
        <style>
        .metric-card {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            height: 130px;
            aspect-ratio: 3/4;
            width: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            box-sizing: border-box;
            border: 1px solid #e0e0e0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s ease-in-out;
            border: 0px solid #e0e0e0;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }
        .metric-header {
            font-size: 12px;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 19px;
            font-weight: 700;
            color: #1f2937;
        }
        .metric-container {
            margin-bottom: 24px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Single row with equal space for all metric cards
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    metrics = [
        ("Booked", int(total_booked_dorms)),
        ("Confirmed Bookings", int(total_confirmed)),
        ("No-Shows", int(total_no_show)),
        ("Cancellations", int(total_cancellation)),
        ("Booking (INR)", f"₹{total_revenue:,.2f}"),
        ("Commission Amount", f"₹{commission_amount:,.2f}"),
        ("Collected", f"₹{net_revenue:,.2f}"),
        ("Collected amount/Bed", f"₹{net_revenue_per_bed:,.2f}"),
        ("Total Nights", int(total_nights)),
        ("Confirmed Booking %", f"{ok_booking_percentage:.2f}%"),
    ]
    # Display metrics in two rows of 5 columns each, with space between rows
    for i in range(0, len(metrics), 5):
        cols = st.columns(5, gap="small")
        for col, (header, value) in zip(cols, metrics[i:i+5]):
            with col:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-header'>{header}</div><div class='metric-value'>{value}</div></div>",
                    unsafe_allow_html=True
                )
        if i + 5 < len(metrics):
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)  # Add vertical space between rows
                
    st.markdown("</div>", unsafe_allow_html=True)

    # Booking Breakdown Bar Chart
    bar_df = pd.DataFrame({
        "Category": ["Confirmed", "No-Show", "Cancelled"],
        "Dorms": [total_confirmed, total_no_show, total_cancellation]
    })
    bar_chart = px.bar(
        bar_df,
        x="Category",
        y="Dorms",
        color="Category",
        title="Dorm Booking Status Breakdown",
        text="Dorms",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#d62728"],
        template="plotly_white"
    )
    bar_chart.update_layout(
        title_x=0.5,
        font=dict(size=18, family="Arial, sans-serif"),
        xaxis_title="Booking Status",
        yaxis_title="Number of Dorms",
        xaxis=dict(tickmode='linear'),
        yaxis=dict(tickformat=',d'),
        legend_title_text="Booking Status",
    )
    st.plotly_chart(bar_chart, use_container_width=True)

    # Pie Chart of Booking Percentages
    pie_df = pd.DataFrame({
        "Status": ["Confirmed", "No-Show", "Cancelled"],
        "Percentage": [ok_booking_percentage, no_show_percentage, cancellation_percentage]
    })
    pie_chart = px.pie(
        pie_df,
        names="Status",
        values="Percentage",
        title="Booking Outcomes Distribution",
        color_discrete_sequence=["#1f77b4", "#ff7f0e", "#d62728"],
        template="plotly_white"
    )
    pie_chart.update_layout(
        title_x=0.5,
        font=dict(family="Arial", size=12),
        margin=dict(t=80, b=40)
    )
    st.plotly_chart(pie_chart, use_container_width=True)

    # Revenue Trend Line Chart
    if 'date' in df.columns:
        revenue_by_date = df[(~df['is_cancelled']) & (~df['is_no_show'])].groupby('date')['price'].sum().reset_index()
        revenue_chart = px.line(
            revenue_by_date,
            x='date',
            y='price',
            title="Daily Revenue Trend",
            markers=True,
            color_discrete_sequence=["#1f77b4"],
            template="plotly_white"
        )
        revenue_chart.update_layout(
            title_x=0.5,
            font=dict(family="Arial", size=12),
            margin=dict(t=80, b=40)
        )
        st.plotly_chart(revenue_chart, use_container_width=True)

    # Expandable Section for Details
    with st.expander("View Detailed Data", expanded=False):
        st.dataframe(df, use_container_width=True)
        st.markdown(f"**Confirmed Booking %:** {ok_booking_percentage:.2f}%")
        st.markdown(f"**Cancellation %:** {cancellation_percentage:.2f}%")
        st.markdown(f"**No-Show %:** {no_show_percentage:.2f}%")  

    # plot dorm types
    st.subheader("Dorm Types Overview") 
    if 'dorm' in df.columns:
        # Normalize dorm names using the same method as in preprocessing
        dorm_df = normalize_dorm_columns(df)
        # Split multivalued dorms and explode
        dorm_df['dorm'] = dorm_df['dorm'].astype(str)
        dorm_df = dorm_df.assign(dorm_type=dorm_df['dorm'].str.split(',')).explode('dorm_type')
        dorm_df['dorm_type'] = dorm_df['dorm_type'].str.strip()
        dorm_counts = dorm_df['dorm_type'].value_counts().reset_index()
        dorm_counts.columns = ['Dorm Type', 'Count']
        dorm_chart = px.bar(
            dorm_counts,
            x='Dorm Type',
            y='Count',
            title="Dorm Types Overview",
            color='Dorm Type',
            text='Count',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            template="plotly_white"
        )
        dorm_chart.update_layout(
            title_x=0.5,
            font=dict(size=14, family="Arial, sans-serif"),
            xaxis_title="Dorm Type",
            yaxis_title="Number of Dorms",
            xaxis=dict(tickmode='linear'),
            yaxis=dict(tickformat=',d'),
        )
        st.plotly_chart(dorm_chart, use_container_width=True)

def main():
    st.title("PathfindersNest Booking Analysis")
    uploaded_file = st.sidebar.file_uploader(
        type=['csv', 'xls', 'xlsx'],
        accept_multiple_files=False,
        key="file_uploader",
        label="Guest Check-in Data",
        label_visibility="visible",
        on_change=None,
        disabled=False,
        help="Upload a CSV or Excel file containing guest check-in data")
    if uploaded_file:
        df = load_data(uploaded_file)
        dorm_types = CONFIG['default_dorm_types']
        dorm_rooms = {}
        st.sidebar.info("Adjust the number of rooms per dorm as needed for your analysis.")
        for dorm in dorm_types:
            dorm_rooms[dorm] = st.sidebar.number_input(
                f"Total Rooms in {dorm}",
                min_value=1,
                value=CONFIG['default_rooms_per_dorm'],
                step=1,
                help=f"Number of rooms in {dorm}"
            )
        st.sidebar.subheader("Forecast Settings", help="Set the month for which you want to forecast bookings. Ensure the data covers at least 30 days for accurate forecasting.")
        forecast_month = st.sidebar.date_input(
            "Forecast Month",
            value=datetime(2025, 7, 1),
            min_value=datetime(2025, 1, 1),
            help="Select the month to forecast"
        )
        st.sidebar.info("⚠️ **Note:** Forecasting is based on historical data trends and may not be accurate.", icon="ℹ️")
        if df is not None:
            st.session_state['data'] = df
            st.success("Data loaded successfully!")
            Logger.info("Data loaded successfully and stored in session state.")
            show_dashboard(df)
            price_trend_visualization(df)
            reservation_trends_over_time(df)
            calculate_daily_metrics_df = calculate_daily_metrics(df, dorm_rooms=dorm_rooms)
            forecast_start = forecast_month
            forecast_end = (pd.to_datetime(forecast_start) + pd.offsets.MonthEnd(0)).date()
            forecast_horizon = (forecast_end - forecast_start).days + 1
            Logger.info(f"Forecasting for period: {forecast_start} to {forecast_end} ({forecast_horizon} days)")
            with st.spinner("Training ML models and generating forecasts..."):
                occ_forecast = train_or_load_and_forecast(calculate_daily_metrics_df, 'occupancy_rate', forecast_horizon, forecast_start, forecast_end)
                rev_forecast = train_or_load_and_forecast(calculate_daily_metrics_df, 'Revenue', forecast_horizon, forecast_start, forecast_end)
                cancel_forecast = train_or_load_and_forecast(calculate_daily_metrics_df, 'cancellation_rate', forecast_horizon, forecast_start, forecast_end)
                no_show_forecast = train_or_load_and_forecast(calculate_daily_metrics_df, 'no_show_rate', forecast_horizon, forecast_start, forecast_end)
            if all([df is not None and not df.empty for df in [occ_forecast, rev_forecast, cancel_forecast, no_show_forecast]]):
                monthly_forecast = occ_forecast.merge(
                    rev_forecast, on=['Dorm', 'Date'], how='outer'
                ).merge(
                    cancel_forecast, on=['Dorm', 'Date'], how='outer'
                ).merge(
                    no_show_forecast, on=['Dorm', 'Date'], how='outer'
                )
                monthly_agg = monthly_forecast.groupby('Dorm').agg(
                    occupancy_rate=('occupancy_rate', lambda x: np.minimum(x, 1.0).mean()),
                    Revenue=('Revenue', 'sum'),
                    cancellation_rate=('cancellation_rate', 'mean'),
                    no_show_rate=('no_show_rate', 'mean')
                ).reset_index()
                st.header("Monthly Forecasted Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_cancellation_rate = monthly_agg['cancellation_rate'].mean()
                    st.metric("Average Cancellation Rate", f"{avg_cancellation_rate:.2%}")
                with col2:
                    avg_no_show_rate = monthly_agg['no_show_rate'].mean()
                    st.metric("Average No-Show Rate", f"{avg_no_show_rate:.2%}")
                with col3:
                    avg_occ = monthly_agg['occupancy_rate'].mean()
                    st.metric("Average Occupancy Rate", f"{avg_occ:.2%}")
                with col4:
                    most_booked_dorm = monthly_agg.loc[monthly_agg['occupancy_rate'].idxmax(), 'Dorm']
                    st.metric("Most Booked Dorm", most_booked_dorm)
                
                st.subheader("Dorm Metrics Forecast")
                dorm_summary = pd.DataFrame({
                    'Dorm': monthly_agg['Dorm'],
                    'Average Occupancy Rate': monthly_agg['occupancy_rate'],
                    'Total Revenue': monthly_agg['Revenue'],
                    'Average Cancellation Rate': monthly_agg['cancellation_rate'],
                    'Average No-Show Rate': monthly_agg['no_show_rate']
                })
                st.dataframe(dorm_summary.style.format({
                    'Average Occupancy Rate': '{:.2%}',
                    'Total Revenue': '₹{:.2f}',
                    'Average Cancellation Rate': '{:.2%}',
                    'Average No-Show Rate': '{:.2%}'
                }), use_container_width=True)
                st.subheader("Daily Forecasted Metrics")
                daily_forecast = occ_forecast.merge(rev_forecast, on=['Dorm', 'Date'], how='outer').merge(cancel_forecast, on=['Dorm', 'Date'], how='outer').merge(no_show_forecast, on=['Dorm', 'Date'], how='outer')
                st.dataframe(daily_forecast.style.format({
                    'occupancy_rate': '{:.2%}',
                    'Revenue': '₹{:.2f}',
                    'cancellation_rate': '{:.2%}',
                    'no_show_rate': '{:.2%}'
                }), use_container_width=True)
                create_visualization_forecast(calculate_daily_metrics_df, occ_forecast, rev_forecast, cancel_forecast, no_show_forecast)
                total_predicted_revenue = get_total_revenue_prediction(daily_forecast)
                st.metric("Total Predicted Revenue", f"₹{total_predicted_revenue:,.2f}")
                csv_buffer = io.StringIO()
                daily_forecast.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Daily Forecasted Metrics",
                    data=csv_buffer.getvalue(),
                    file_name="daily_forecasted_metrics.csv",
                    mime="text/csv",
                    help="Download the daily forecasted metrics as a CSV file"
                )
            pd.DataFrame(calculate_daily_metrics_df).to_csv("daily_metrics.csv")
            # st.subheader("Daily Metrics")
            # st.dataframe(calculate_daily_metrics_df)
def get_total_revenue_prediction(daily_forecast: pd.DataFrame) -> float:
    """
    Returns the total predicted revenue for the forecast period.
    """
    if 'Revenue' not in daily_forecast.columns:
        st.warning("Revenue column not found in forecast data.")
        return 0.0
    return daily_forecast['Revenue'].sum()
def reservation_trends_over_time(df):
    st.subheader("Booking Trends: Cancellation, No-Show, and Confirmed")
    if 'check_in' in df.columns and 'is_cancelled' in df.columns and 'is_no_show' in df.columns:
        df['check_in'] = pd.to_datetime(df['check_in'])
                # Prepare monthly trends
        monthly = df.copy()
        monthly['month'] = monthly['check_in'].dt.to_period('M').dt.to_timestamp()
        trends = monthly.groupby('month').agg(
                    cancellations=('is_cancelled', 'sum'),
                    no_shows=('is_no_show', 'sum'),
                    confirmed=('no_of_dorms', lambda x: x[(~monthly.loc[x.index, 'is_cancelled']) & (~monthly.loc[x.index, 'is_no_show'])].sum())
                ).reset_index()

                # Melt for plotting
        trends_melted = trends.melt(id_vars='month', value_vars=['cancellations', 'no_shows', 'confirmed'],
                                            var_name='Status', value_name='Count')

                # Define color mapping
        color_map = {
                    'confirmed': 'skyblue',
                    'cancellations': 'red',
                    'no_shows': 'orange'
                }

        fig = px.line(
                    trends_melted,
                    x='month',
                    y='Count',
                    color='Status',
                    markers=True,
                    title='Booking Trends Over Time',
                    labels={'month': 'Month', 'Count': 'Number of Bookings'},
                    template='plotly_white',
                    color_discrete_map=color_map,
                    line_shape='linear',
                    hover_data={'month': '|%B %Y'}
                )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Check-in, Cancellation or No-Show columns are missing for trends visualization.")

def price_trend_visualization(df):
    st.subheader("Price Per Dorm Trend Visualization")
    if 'check_in' in df.columns and 'price' in df.columns and 'no_of_dorms' in df.columns and 'nights' in df.columns:
        df['check_in'] = pd.to_datetime(df['check_in'])
        # Calculate price per dorm per night
        df['price_per_dorm'] = df['price'] / (df['no_of_dorms'] * df['nights'])
        price_trend = df.groupby(df['check_in'].dt.to_period('M'))['price_per_dorm'].mean().reset_index()
        price_trend['check_in'] = price_trend['check_in'].dt.to_timestamp()
        fig = px.line(
            price_trend,
            x='check_in',
            y='price_per_dorm',
            title='Average Price Per Dorm Per Night Trend Over Time',
            markers=True,
            labels={'check_in': 'Month', 'price_per_dorm': 'Avg Price Per Dorm/Night'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Check-in, Price, No. of Dorms, or Nights columns are missing for visualization.")
            

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        Logger.error(f"An error occurred: {e}")
        st.stop()

