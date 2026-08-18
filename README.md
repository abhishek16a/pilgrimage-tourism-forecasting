Forecasting Religious Tourism: Machine Learning Analysis of Pilgrimage Patterns to Sacred Sites in Nepal
MSc Data Science & Computational Intelligence Thesis

Student: Abhishek Adhikari (9635164)
Module: STW7048CEM
Institution: Softwarica College of IT & E-Commerce (Coventry University)
Supervisor: Mr. Manoj Shrestha

Overview

This research develops a machine learning based forecasting system that analyses historical visitor data from four major sacred sites in Nepal and generates predictions of future pilgrimage patterns. The system compares time series models (ARIMA, SARIMA) with feature based machine learning models (Random Forest, XGBoost) to determine which approach best captures the complex dynamics of religious tourism.

Sacred Sites Studied
Pashupatinath — Holiest Hindu temple in Nepal, Kathmandu
Lumbini — Birthplace of the Buddha, UNESCO World Heritage Site
Muktinath — Sacred to both Hindus and Buddhists, Annapurna region
Janakpur Dham — Birthplace of Sita, Terai region
Key Findings
Feature based ML models outperformed time series models at all four sites
Random Forest achieved best accuracy at 3 sites (MAPE: 13.16% to 19.02%)
XGBoost achieved best accuracy at Janakpur Dham (MAPE: 15.45%)
Adding external features reduced forecasting error by an average of 53%
Each site is driven by fundamentally different factors (festivals, monsoon, visitor momentum, seasonal timing)
Project Structure
thesis_project/
├── data/
│   ├── pilgrimage_data_raw.csv         # Raw messy dataset (13 quality issues)
│   └── pilgrimage_data_clean.csv       # Cleaned dataset (756 rows)
├── notebooks/
│   ├── 01_data_cleaning.ipynb          # Load, clean, validate data
│   ├── 02_eda.ipynb                    # Exploratory data analysis (11 charts)
│   ├── 03_time_series.ipynb            # ARIMA and SARIMA training
│   ├── 04_ml_models.ipynb              # Random Forest and XGBoost training
│   ├── 05_evaluation.ipynb             # Model comparison and final results
│   ├── run_all.ipynb                   # Execute all notebooks in sequence
│   └── run_dashboard.ipynb             # Launch Streamlit dashboard
├── app.py                              # Streamlit dashboard (4 pages)
└── requirements.txt                    # Python dependencies
Models Used
Model	Type	Description
ARIMA(2,1,2)	Time Series	Baseline trend forecasting
SARIMA(1,1,1)(1,1,1,12)	Time Series	Seasonal pattern forecasting
Random Forest	Machine Learning	200 trees, max depth 10, 11 input features
XGBoost	Machine Learning	200 rounds, learning rate 0.1, 11 input features
Features Used (ML Models)
Feature	Description
month	Month of the year (1 to 12)
year	Year (captures growth trend)
temperature_c	Monthly average temperature
rainfall_mm	Monthly total rainfall
festival	Major festival flag (0 or 1)
lag_1	Previous month visitor count
lag_12	Same month last year visitor count
rolling_3	3 month rolling average
season_*	One hot encoded season (4 variables)
Results
Site	ARIMA	SARIMA	Random Forest	XGBoost	Best Model
Pashupatinath	38.88%	24.29%	13.16%	14.91%	Random Forest
Lumbini	32.22%	41.97%	13.72%	14.09%	Random Forest
Muktinath	54.74%	41.70%	19.02%	24.91%	Random Forest
Janakpur Dham	33.73%	37.79%	16.52%	15.45%	XGBoost

MAPE (%) — lower is better

Dashboard

The interactive Streamlit dashboard has four pages:

Overview — Dataset summary, visitor trends, key statistics
Forecasts — Select site, model, and horizon to generate predictions
Compare — Side by side model accuracy comparison with heatmap
Insights — Seasonal patterns, feature importance, festival impact
How to Run
Prerequisites
Python 3.10 or higher
pip package manager
Installation
bash
# Clone the repository
git clone https://github.com/abhishek16a/pilgrimage-tourism-forecasting.git
cd pilgrimage-tourism-forecasting

# Install dependencies
pip install -r requirements.txt
Run Notebooks

Run notebooks in order (01 through 05):

bash
cd notebooks
jupyter notebook

Open each notebook and run all cells sequentially. Alternatively, open run_all.ipynb and execute each cell to run all notebooks automatically.

Launch Dashboard
bash
cd thesis_project
streamlit run app.py

The dashboard will open at http://localhost:8501

Technologies
Python 3.11 — Programming language
pandas, NumPy — Data processing
matplotlib, seaborn — Visualisation
statsmodels — ARIMA and SARIMA
scikit-learn — Random Forest and evaluation metrics
XGBoost — Gradient boosting
Streamlit — Interactive dashboard
Jupyter Notebook — Development environment
Ethical Framework

This research is guided by four design principles for responsible ML at heritage sites:

Transparency — Feature importance charts make model reasoning visible
Aggregate data only — No individual pilgrim tracking or profiling
Stewardship framing — Outputs designed for heritage planning, not commercial exploitation
Cultural sensitivity — No recommendations that override religious calendars or spiritual practices
License

This project is submitted as academic coursework for MSc Data Science & Computational Intelligence at Softwarica College (Coventry University). All code is open source for academic and non commercial use.
