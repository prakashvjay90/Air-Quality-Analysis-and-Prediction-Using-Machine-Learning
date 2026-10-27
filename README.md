# PADV Activity 2 — Air Quality Analysis and NO₂ Prediction

Interactive Streamlit dashboard based on the Jupyter Notebook project using the UCI Air Quality dataset.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload `AirQualityUCI.xlsx` through the app. The dataset is not bundled in this repository.

## Features

- Data cleaning and missing-value handling (-200 as missing)
- Daily trends, distribution and correlation visualizations
- Random Forest Regression (150 trees, min_samples_leaf=2)
- Chronological 80/20 split and MAE, RMSE, R² evaluation
- Feature importance and interactive NO₂ estimation

**Important:** NO₂ estimation from same-time sensor measurements is not future forecasting.
