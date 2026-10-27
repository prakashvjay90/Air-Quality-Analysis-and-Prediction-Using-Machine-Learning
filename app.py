import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title='PADV Activity 2 | Air Quality', page_icon='🌫️', layout='wide')
st.title('🌫️ Air Quality Analysis & NO₂ Prediction')
st.caption('PADV Activity 2 · UCI Air Quality dataset · Random Forest Regression')

@st.cache_data
def read_data(file_bytes, filename):
    import io
    f = io.BytesIO(file_bytes)
    if filename.lower().endswith('.csv'):
        return pd.read_csv(f, sep=None, engine='python')
    return pd.read_excel(f)

@st.cache_data
def clean_data(df):
    data = df.copy()
    data.columns = data.columns.astype(str).str.strip()
    data = data.loc[:, ~data.columns.str.startswith('Unnamed')]
    if 'Date' not in data.columns or 'Time' not in data.columns or 'NO2(GT)' not in data.columns:
        raise ValueError('Required columns: Date, Time and NO2(GT). Please upload the UCI Air Quality dataset.')
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)
    times = data['Time'].astype(str).str.strip().str.replace('.', ':', regex=False)
    data['DateTime'] = pd.to_datetime(data['Date'].dt.strftime('%Y-%m-%d') + ' ' + times, errors='coerce')
    for col in data.columns:
        if col not in ['Date', 'Time', 'DateTime']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    numeric_cols = data.select_dtypes(include=np.number).columns
    data[numeric_cols] = data[numeric_cols].replace(-200, np.nan)
    for name, val in {'Year':data.DateTime.dt.year, 'Month':data.DateTime.dt.month, 'Day':data.DateTime.dt.day, 'Hour':data.DateTime.dt.hour, 'DayOfWeek':data.DateTime.dt.dayofweek}.items():
        data[name] = val
    return data.dropna(subset=['NO2(GT)', 'DateTime']).sort_values('DateTime').reset_index(drop=True)

@st.cache_resource(show_spinner='Training Random Forest model...')
def train_model(data):
    target = 'NO2(GT)'
    X = data.drop(columns=[target, 'Date', 'Time', 'DateTime'], errors='ignore').select_dtypes(include=np.number)
    y = data[target]
    n = int(len(data)*.8)
    if n < 2 or len(data)-n < 2:
        raise ValueError('Not enough valid rows to train and test the model.')
    model = Pipeline([('imputer', SimpleImputer(strategy='median')), ('regressor', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, min_samples_leaf=2))])
    model.fit(X.iloc[:n], y.iloc[:n])
    pred = model.predict(X.iloc[n:])
    metrics = {'MAE':mean_absolute_error(y.iloc[n:],pred), 'RMSE':np.sqrt(mean_squared_error(y.iloc[n:],pred)), 'R²':r2_score(y.iloc[n:],pred)}
    comparison = pd.DataFrame({'DateTime':data['DateTime'].iloc[n:].values, 'Actual NO₂':y.iloc[n:].values, 'Predicted NO₂':pred})
    imp = pd.DataFrame({'Feature':X.columns, 'Importance':model.named_steps['regressor'].feature_importances_}).sort_values('Importance', ascending=False)
    return model, X, metrics, comparison, imp, n

uploaded = st.file_uploader('Upload AirQualityUCI.xlsx or equivalent CSV', type=['xlsx','xls','csv'])
if uploaded is None:
    st.info('Upload the AirQualityUCI.xlsx dataset to display analysis, evaluation and prediction. The dashboard will work on Streamlit Community Cloud.')
    st.stop()
try:
    raw = read_data(uploaded.getvalue(), uploaded.name)
    data = clean_data(raw)
    model, X, metrics, comparison, importance, split = train_model(data)
except Exception as exc:
    st.error(f'Unable to process this dataset: {exc}')
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric('Original rows', f'{len(raw):,}')
c2.metric('Valid NO₂ readings', f'{len(data):,}')
c3.metric('Training records', f'{split:,}')
c4.metric('Testing records', f'{len(data)-split:,}')

analysis, performance, predict_tab, dataset_tab = st.tabs(['📈 Trends & EDA','🎯 Model Evaluation','🧪 Prediction Simulator','📋 Dataset'])
with analysis:
    min_date, max_date = data.DateTime.min().date(), data.DateTime.max().date()
    dates = st.date_input('Filter observation dates', (min_date, max_date), min_value=min_date, max_value=max_date)
    if not isinstance(dates, (tuple,list)) or len(dates) != 2:
        st.warning('Select a start and end date.'); st.stop()
    filtered = data[(data.DateTime.dt.date >= dates[0]) & (data.DateTime.dt.date <= dates[1])]
    pollutant_options = [c for c in ['NO2(GT)','CO(GT)','NMHC(GT)','C6H6(GT)','NOx(GT)','T','RH','AH'] if c in data.columns]
    selected = st.selectbox('Measurement', pollutant_options)
    daily = filtered.set_index('DateTime')[selected].resample('D').mean().dropna().reset_index()
    st.plotly_chart(px.line(daily, x='DateTime', y=selected, title=f'Daily average — {selected}'), use_container_width=True)
    left,right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(filtered, x=selected, nbins=40, title=f'Distribution of {selected}'), use_container_width=True)
    with right:
        numeric = filtered.select_dtypes(include=np.number)
        candidates = [c for c in ['NO2(GT)','CO(GT)','C6H6(GT)','NOx(GT)','T','RH'] if c in numeric.columns]
        if len(candidates) > 1:
            st.plotly_chart(px.imshow(numeric[candidates].corr(), text_auto='.2f', title='Correlation heatmap', aspect='auto'), use_container_width=True)
with performance:
    a,b,c = st.columns(3)
    a.metric('MAE', f"{metrics['MAE']:.4f}")
    b.metric('RMSE', f"{metrics['RMSE']:.4f}")
    c.metric('R² score', f"{metrics['R²']:.4f}")
    st.caption('Chronological split: first 80% for training, last 20% for testing. Results are computed from the uploaded file.')
    st.plotly_chart(px.line(comparison, x='DateTime', y=['Actual NO₂','Predicted NO₂'], title='Actual vs predicted NO₂ on test data'), use_container_width=True)
    st.plotly_chart(px.bar(importance.head(15).sort_values('Importance'), x='Importance', y='Feature', orientation='h', title='Top feature importances'), use_container_width=True)
    st.download_button('Download prediction results', comparison.to_csv(index=False), 'no2_prediction_results.csv', 'text/csv')
with predict_tab:
    st.write('Adjust sensor readings and time features to estimate NO₂. This is an estimate based on contemporaneous measurements, not a future forecast.')
    sample_idx = st.slider('Choose a reference observation', 0, len(X)-1, 0)
    values = X.iloc[sample_idx].copy()
    options = st.multiselect('Features to adjust', list(X.columns), default=[c for c in ['CO(GT)','NOx(GT)','T','RH'] if c in X.columns])
    for col in options:
        default = float(values[col]) if pd.notna(values[col]) else float(X[col].median()) if pd.notna(X[col].median()) else 0.0
        values[col] = st.number_input(col, value=default, format='%.3f')
    if st.button('Predict NO₂', type='primary'):
        st.metric('Estimated NO₂(GT)', f'{model.predict(pd.DataFrame([values], columns=X.columns))[0]:.2f}')
with dataset_tab:
    st.write('Data sample after cleaning')
    st.dataframe(data.head(100), use_container_width=True)
    st.write('Missing values by column')
    st.dataframe(data.isna().sum().rename('Missing').to_frame(), use_container_width=True)
    st.download_button('Download cleaned data', data.to_csv(index=False), 'air_quality_cleaned.csv', 'text/csv')
