# Appliances Energy Consumption Forecasting Using Time Series Machine Learning

## Project Overview

This project applies various time series machine learning methods to predict household appliance energy consumption, including traditional machine learning models, specialized time series models, and deep learning approaches.

## Dataset

### Data Source
- **Dataset URL**: [Kaggle - Appliances Energy Prediction Data Set](https://www.kaggle.com/datasets/sohommajumder21/appliances-energy-prediction-data-set)
- **Original Author**: Luis Candanedo, University of Mons (UMONS)

### Dataset Characteristics
- **Data Type**: Multivariate, Time-Series, Regression
- **Time Period**: January - May 2016 (4.5 months)
- **Data Size**: 19,735 records with 29 features
- **Target Variable**: Appliances (energy consumption in Wh)
- **Sampling Rate**: 10-minute intervals

### Data Collection Context
This experimental data was collected to create regression models of appliances energy use in a low energy building. Hourly weather data from Chievres Airport, Belgium was downloaded from Reliable Prognosis (rp5.ru) and interpolated to 10-minute intervals.

### Citations
**Primary Reference:**
Luis M. Candanedo, Veronique Feldheim, Dominique Deramaix, "Data driven prediction models of energy use of appliances in a low-energy house", Energy and Buildings, Volume 140, 1 April 2017, Pages 81-97, ISSN 0378-7788.

**Dataset Repository:**
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

## Project Structure

```
├── data/                           # Data files
│   ├── energydata_complete_raw.csv     # Raw data
│   └── energydata_complete_cleaned.csv # Cleaned data
├── models/                         # Trained models (empty)
├── notebooks/                      # Jupyter notebooks
│   ├── 1.0-EDA-Descriptive_Stats.ipynb           # Exploratory Data Analysis
│   ├── 2.0-Feature_Engineering.ipynb              # Feature Engineering
│   ├── 3.0-Traditional_ML.ipynb                   # Traditional Machine Learning
│   ├── 3.1-Traditional_ML_Feature_Abaltion.ipynb  # Feature Ablation Study
│   ├── 4.0-TimeSeries_ML.ipynb                    # Time Series Machine Learning
│   ├── 4.1-TimeSeries_DL_LSTM.ipynb               # LSTM Deep Learning
│   ├── 4.1-TimeSeries_DL_RNN.ipynb                  # RNN Deep Learning
│   ├── 4.1-TimeSeries_DL_Transformer.ipynb        # Transformer Deep Learning
│   ├── 5.0-Visualisation.ipynb                     # Results Visualisation
│   └── 6.0-Shapley_Explanation.ipynb              # SHAP Explainable AI
├── results/                        # Result files
│   ├── eda_plots/                     # EDA visualization plots
│   ├── prediction_plots/              # Model prediction plots
│   ├── ablation_study/                  # Feature ablation study results
│   │   ├── ML_metrics.csv              # ML model metrics comparison
│   │   └── ML_metrics_ablation.csv     # ML model metrics with feature ablation
│   ├── ML_predictions.csv              # Traditional ML model predictions
│   ├── SARIMA_predictions.csv          # SARIMA model predictions
│   ├── prophet_predictions.csv         # Prophet model predictions
│   ├── lstm_predictions.csv            # LSTM model predictions
│   ├── rnn_predictions.csv             # RNN model predictions
│   ├── transformer_predictions.csv     # Transformer model predictions
│   ├── timeseries_model_results.csv   # Model performance comparison
│   └── shap_explanation/              # SHAP explainability results
│       ├── shap_beeswarm.png          # Overall feature importance (beeswarm plot)
│       ├── shap_waterfall_0.png       # Individual prediction explanation (sample 0)
│       └── shap_waterfall_100.png     # Individual prediction explanation (sample 100)
├── requirements.txt                # Dependencies
└── README.md                      # Project documentation
```

## Experiments

### 1. Exploratory Data Analysis (EDA)
- Data distribution analysis
- Time series visualization
- Correlation analysis
- Seasonality and trend analysis

### 2. Feature Engineering
- Time feature extraction (hour, weekday, weekend)
- Cyclical feature encoding (sin/cos transformation)
- Lag feature creation
- Rolling statistics features

### 3. Traditional Machine Learning Models
- Linear Regression (As a basic model, not show in final result)
- Random Forest
- XGBoost
- Support Vector Regression
- Multi-layer Perceptron

### 3.1 Feature Ablation Study
- Systematic feature importance analysis
- Performance comparison with feature subsets
- Model sensitivity to different feature groups

### 4. Time Series Machine Learning Models
- **Prophet**: Facebook's time series forecasting tool
- **SARIMA**: Seasonal Autoregressive Integrated Moving Average model

### 5. Deep Learning Models
- **RNN**: Recurrent Neural Network (`4.1-TimeSeries_DL_RNN.ipynb`)
- **LSTM**: Long Short-Term Memory neural network (`4.1-TimeSeries_DL_LSTM.ipynb`)
- **Transformer**: Attention-based time series forecasting model (`4.1-TimeSeries_DL_Transformer.ipynb`)

### 6. Visualisation
- **Model Performance Comparison**: Comprehensive comparison of all model performances
- **Prediction Results Visualization**: Time series plots showing actual vs predicted values
- **Feature Ablation Study Results**: Impact analysis of different feature groups on model performance

### 7. SHAP Explainable AI
- **Model Prediction Interpretability Analysis**: Understanding how models make predictions
- **Feature Importance Ranking**: Identifying most influential features for energy consumption prediction
- **Individual Prediction Explanation**: Waterfall plots showing feature contributions for specific predictions
- **Overall Feature Impact Analysis**: Beeswarm plots visualizing feature importance distribution across all predictions


## Model Performance Evaluation

All models are evaluated using the following metrics:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

## Setup and Usage

### 1. Requirements
```bash
Python 3.9.6
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Experiments
Execute the notebooks in the following order:
1. `1.0-EDA-Descriptive_Stats.ipynb` - Data exploration
2. `2.0-Feature_Engineering.ipynb` - Feature engineering
3. `3.0-Traditional_ML.ipynb` - Traditional machine learning
4. `3.1-Traditional_ML_Feature_Abaltion.ipynb` - Feature ablation study
5. `4.0-TimeSeries_ML.ipynb` - Time series machine learning (Prophet & SARIMA)
6. `4.1-TimeSeries_DL_*.ipynb` - Deep learning models (RNN, LSTM, Transformer)
7. `5.0-Visualisation.ipynb` - Prediction results and metrics visualisation
8. `6.0-Shapley_Explanation.ipynb` - SHAP model interpretability and feature importance analysis
