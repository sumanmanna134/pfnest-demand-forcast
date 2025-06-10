# pfnest Demand Forecasting

A robust, production-ready machine learning application for forecasting occupancy, revenue, cancellations, and no-shows in dormitory/hospitality environments. Built with **Streamlit**, **XGBoost**, and advanced feature engineering, this tool empowers property managers with actionable, data-driven insights.

---

## Features

- **Flexible Data Ingestion:** Upload guest check-in data in CSV or Excel format.
- **Automated Feature Engineering:** Generates lags, rolling statistics, lead time, and calendar-based features for each dorm and date.
- **Multi-Target Forecasting:**
  - **Occupancy Rate** (regression)
  - **Revenue** (regression)
  - **Cancellation Probability** (classification)
  - **No-Show Probability** (classification)
- **Scheduled Model Retraining:** Models are automatically retrained every 30 days or on demand, with persistent storage.
- **Hyperparameter Optimization:** Uses `RandomizedSearchCV` for both regression and classification models.
- **Interactive Visualizations:** Explore trends, price evolution, and booking status with Plotly charts.
- **Downloadable Results:** Export forecast outputs for further analysis.

---

## Technical Overview

### Data Pipeline

- **Input:** Tabular data with columns for check-in/out, price, dorm type, booking/cancellation status, and optionally booking date.
- **Indexing:** Data is indexed by `Dorm` and `Date` for time series processing.
- **Feature Engineering:**
  - **Temporal:** Day of week, month, is_weekend, quarter, week_of_year, month start/end, days since start.
  - **Lagged Features:** 1, 7, 14-day lags for all targets.
  - **Rolling Features:** 7/30-day rolling mean, std, min, max for all targets.
  - **Lead Time:** Days between booking and check-in.
  - **Interaction Terms:** e.g., occupancy × cancellation rate.

### Modeling

- **Regression:** XGBoost Regressor for occupancy and revenue.
- **Classification:** XGBoost Classifier for cancellation and no-show (predicts probability per day/dorm).
- **Hyperparameter Tuning:** Extensive search grid for n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, min_child_weight.
- **Model Persistence:** Models and training metadata are saved to disk and only retrained if 30 days have elapsed or on manual trigger.

### Forecasting

- **Rolling/Expanding Window:** Forecasts are generated for each dorm and date using the latest available data.
- **Probability Outputs:** For cancellation and no-show, outputs are probabilities (0–1) representing the likelihood of the event on the check-in day.
- **Postprocessing:** Occupancy rates are clipped to [0, 1].

---

## Installation

```bash
git clone https://github.com/yourusername/reservation-forecasting.git
cd reservation-forecasting
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app.py
```

- Upload your data file when prompted.
- View forecasts and visualizations.
- Download results as needed.

---

## Data Requirements

Your input file should include at least:

- `check_in` (date)
- `check_out` (date)
- `price`
- `nights`
- `no_of_dorms`
- `dorm` (dorm name/type)
- `is_cancelled` (0/1)
- `is_no_show` (0/1)
- _(Optional)_ `booking_date`

---

## Project Structure

```
reservation-forecasting/
│
├── app.py                # Streamlit app UI and workflow
├── train.py              # Model training, tuning, and forecasting logic
├── utils.py              # Feature engineering and configuration
├── models/               # Saved models and metadata
├── requirements.txt
└── README.md
```

---

## Customization

- **Retrain Interval:** Adjust the days in `should_retrain_model()` in `train.py`.
- **Feature Engineering:** Add or modify features in `prepare_ml_features()` in `utils.py`.
- **Model/Hyperparameters:** Tune or extend search grids in `train.py`.
- **Add New Targets:** Extend `all_metrics` and update feature engineering/modeling logic.

---

## License

MIT License

---

## Authors

- [Suman Manna](https://github.com/sumanmanna134)

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [XGBoost](https://xgboost.ai/)
- [scikit-learn](https://scikit-learn.org/)
