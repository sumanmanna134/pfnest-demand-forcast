# pfnest demand Forecasting

A machine learning-powered application for forecasting dormitory/hospitality occupancy, revenue, cancellations, and no-shows. Built with Streamlit, XGBoost, and robust feature engineering, this tool helps hostel/hotel managers make data-driven decisions.

---

## Features

- **Upload guest check-in data** (CSV/Excel)
- **Automatic feature engineering** (lags, rolling stats, lead time, calendar features)
- **Forecasts for:**
  - Occupancy rate
  - Revenue
  - Cancellation probability
  - No-show probability
- **Model retraining every 30 days** (or on demand)
- **Interactive visualizations** (trends, price, booking status)
- **Downloadable forecast results**

---

## How It Works

1. **Upload Data:**  
   Upload your guest check-in data file via the Streamlit interface.

2. **Feature Engineering:**  
   The app automatically generates features such as lead time, rolling averages, and calendar variables.

3. **Model Training:**

   - Uses XGBoost for regression (occupancy, revenue) and classification (cancellation, no-show).
   - Hyperparameter tuning with RandomizedSearchCV.
   - Models are saved and only retrained every 30 days (or when forced).

4. **Forecasting:**
   - Predicts future occupancy, revenue, cancellation, and no-show probabilities for each dorm and date.
   - Results are shown in tables and interactive charts.

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
- Explore forecasts and visualizations.
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
├── app.py                # Streamlit app
├── train.py              # Model training and forecasting logic
├── utils.py              # Feature engineering and config
├── models/               # Saved models and metadata
├── requirements.txt
└── README.md
```

---

## Customization

- **Retrain interval:** Change the days in `should_retrain_model()` in `train.py`.
- **Feature engineering:** Add or modify features in `prepare_ml_features()` in `utils.py`.
- **Model/hyperparameters:** Tune in `train.py`.

---

## License

MIT License

---

## Authors

- [Your Name](https://github.com/yourusername)

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [XGBoost](https://xgboost.ai/)
- [scikit-learn](https://scikit-learn.org/)
