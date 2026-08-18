"""
Streamlit Dashboard — Forecasting Religious Tourism in Nepal
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG — Must be the first Streamlit command
# ============================================================
st.set_page_config(
    page_title="Nepal Pilgrimage Forecasting",
    page_icon="🕉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD DATA AND MODELS
# ============================================================
@st.cache_data
def load_data():
    """Load the cleaned dataset"""
    df = pd.read_csv('data/pilgrimage_data_clean.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Create features needed for ML models
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    df = df.sort_values(['site', 'date']).reset_index(drop=True)
    df['lag_1'] = df.groupby('site')['total_visitors'].shift(1)
    df['lag_12'] = df.groupby('site')['total_visitors'].shift(12)
    df['rolling_3'] = df.groupby('site')['total_visitors'].transform(
        lambda x: x.shift(1).rolling(window=3).mean()
    )
    return df

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    sites_lower = ['janakpur_dham', 'lumbini', 'muktinath', 'pashupatinath']
    sites_proper = ['Janakpur Dham', 'Lumbini', 'Muktinath', 'Pashupatinath']
    
    for sl, sp in zip(sites_lower, sites_proper):
        models[sp] = {}
        for model_type in ['arima', 'sarima', 'rf', 'xgb']:
            path = f'models/{model_type}_{sl}.pkl'
            if os.path.exists(path):
                models[sp][model_type] = joblib.load(path)
    return models

@st.cache_data
def load_results():
    """Load saved accuracy results"""
    results = {}
    if os.path.exists('models/time_series_results.pkl'):
        ts = joblib.load('models/time_series_results.pkl')
        results['arima'] = ts['arima']
        results['sarima'] = ts['sarima']
    if os.path.exists('models/ml_results.pkl'):
        ml = joblib.load('models/ml_results.pkl')
        results['rf'] = ml['random_forest']
        results['xgb'] = ml['xgboost']
    return results

@st.cache_data
def load_feature_columns():
    """Load feature column names"""
    if os.path.exists('models/feature_columns.pkl'):
        return joblib.load('models/feature_columns.pkl')
    return None

# Load everything
try:
    df = load_data()
    models = load_models()
    results = load_results()
    feature_columns = load_feature_columns()
    sites = sorted(df['site'].unique())
    data_loaded = True
    colors = {'Pashupatinath': '#0F6E56', 'Lumbini': '#534AB7', 'Muktinath': '#993C1D', 'Janakpur Dham': '#185FA5'}
except Exception as e:
    data_loaded = False
    st.error(f"Error loading data: {e}")
    st.info("Make sure you have run all 5 notebooks first.")
    st.stop()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🕉️ Nepal Pilgrimage")
st.sidebar.markdown("**Tourism Forecasting Dashboard**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "📈 Forecasts", "⚖️ Compare", "💡 Insights"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**About**")
st.sidebar.markdown(
    "MSc Data Science Thesis\n\n"
    "Softwarica College\n"
    "(Coventry University)\n\n"
    "Module: STW7048CEM"
)

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "📊 Overview":
    st.title("📊 Overview")
    st.markdown("Summary of the cleaned pilgrimage visitor dataset across four sacred sites in Nepal.")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Sacred Sites", f"{df['site'].nunique()}")
    col3.metric("Date Range", f"{df['date'].dt.year.min()} – {df['date'].dt.year.max()}")
    col4.metric("Months/Site", f"{len(df) // df['site'].nunique()}")
    
    st.markdown("---")
    
    # All sites trend chart
    st.subheader("Monthly Visitor Trends — All Sites")
    
    fig, ax = plt.subplots(figsize=(14, 5))
    for site in sites:
        site_data = df[df['site'] == site].sort_values('date')
        ax.plot(site_data['date'], site_data['total_visitors'], 
                label=site, color=colors.get(site, 'gray'), linewidth=1, alpha=0.8)
    
    ax.axvspan(pd.Timestamp('2020-03'), pd.Timestamp('2021-06'), 
               alpha=0.08, color='red')
    ax.set_ylabel('Total Visitors')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left')
    ax.set_title('Monthly Visitors (2010-2025)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Summary statistics table
    st.subheader("Summary Statistics Per Site")
    
    summary = df.groupby('site')['total_visitors'].agg(
        ['count', 'mean', 'min', 'max', 'sum']
    ).round(0).astype(int)
    summary.columns = ['Months', 'Avg Monthly', 'Min Month', 'Max Month', 'Total Visitors']
    st.dataframe(summary.style.format("{:,}"), use_container_width=True)
    
    st.markdown("---")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df[['date', 'site', 'total_visitors', 'domestic_visitors', 
                      'indian_visitors', 'international_visitors', 
                      'temperature_c', 'rainfall_mm', 'festival', 'season']].head(20),
                 use_container_width=True)

# ============================================================
# PAGE 2: FORECASTS
# ============================================================
elif page == "📈 Forecasts":
    st.title("📈 Forecasts")
    st.markdown("Select a site and model to generate visitor forecasts.")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_site = st.selectbox("Select Site", sites)
    with col2:
        selected_model = st.selectbox("Select Model", 
                                       ["ARIMA", "SARIMA", "Random Forest", "XGBoost"])
    with col3:
        forecast_months = st.selectbox("Forecast Horizon", [6, 12, 24], index=1)
    
    st.markdown("---")
    
    # Get site data
    site_data = df[df['site'] == selected_site].sort_values('date')
    
    # Model mapping
    model_key = {'ARIMA': 'arima', 'SARIMA': 'sarima', 
                 'Random Forest': 'rf', 'XGBoost': 'xgb'}
    key = model_key[selected_model]
    
    # Generate forecast
    if selected_site in models and key in models[selected_site]:
        model = models[selected_site][key]
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        # Plot historical data
        ax.plot(site_data['date'], site_data['total_visitors'], 
                color=colors.get(selected_site, 'gray'), linewidth=1.2, 
                label='Historical', alpha=0.8)
        
        # Generate forecast based on model type
        if key in ['arima', 'sarima']:
            # Time series forecast
            forecast = model.forecast(steps=forecast_months)
            future_dates = pd.date_range(
                start=site_data['date'].max() + pd.DateOffset(months=1),
                periods=forecast_months, freq='MS'
            )
            
            ax.plot(future_dates, forecast.values, 
                    color=colors.get(selected_site, 'gray'),
                    linewidth=2, linestyle='--', label=f'{selected_model} Forecast')
            
            # Forecast metrics
            forecast_values = forecast.values
            
        else:
            # ML forecast — need to create future features
            last_date = site_data['date'].max()
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_months, freq='MS'
            )
            
            # Build future features
            forecast_values = []
            recent_visitors = site_data['total_visitors'].values[-12:].tolist()
            
            for fd in future_dates:
                month = fd.month
                year = fd.year
                
                # Get average weather for this month from historical data
                month_data = site_data[site_data['date'].dt.month == month]
                avg_temp = month_data['temperature_c'].mean()
                avg_rain = month_data['rainfall_mm'].mean()
                
                # Festival flag (use historical pattern)
                festival = int(month_data['festival'].mode().iloc[0]) if len(month_data) > 0 else 0
                
                # Season
                if month in [3,4,5]: season = 'pre-monsoon'
                elif month in [6,7,8]: season = 'monsoon'
                elif month in [9,10,11]: season = 'post-monsoon'
                else: season = 'winter'
                
                # Lag features
                lag_1 = recent_visitors[-1] if len(recent_visitors) > 0 else 0
                lag_12 = recent_visitors[-12] if len(recent_visitors) >= 12 else lag_1
                rolling_3 = np.mean(recent_visitors[-3:]) if len(recent_visitors) >= 3 else lag_1
                
                # Build feature row
                feature_row = {
                    'month': month, 'year': year,
                    'temperature_c': avg_temp, 'rainfall_mm': avg_rain,
                    'festival': festival,
                    'lag_1': lag_1, 'lag_12': lag_12, 'rolling_3': rolling_3,
                }
                
                # Add season dummies
                for s in ['monsoon', 'post-monsoon', 'pre-monsoon', 'winter']:
                    feature_row[f'season_{s}'] = 1 if season == s else 0
                
                # Predict
                feature_df = pd.DataFrame([feature_row])
                if feature_columns:
                    feature_df = feature_df[feature_columns]
                
                pred = model.predict(feature_df)[0]
                forecast_values.append(max(0, pred))
                recent_visitors.append(pred)
            
            ax.plot(future_dates, forecast_values, 
                    color=colors.get(selected_site, 'gray'),
                    linewidth=2, linestyle='--', label=f'{selected_model} Forecast')
        
        # Divider line
        ax.axvline(x=site_data['date'].max(), color='gray', 
                   linestyle=':', alpha=0.5, linewidth=1)
        ax.text(site_data['date'].max(), ax.get_ylim()[1] * 0.95, 
                '  Forecast →', fontsize=9, color='gray', va='top')
        
        ax.set_title(f'{selected_site} — {selected_model} Forecast ({forecast_months} months)', 
                     fontsize=13, fontweight='bold')
        ax.set_ylabel('Visitors')
        ax.set_xlabel('Date')
        ax.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Forecast summary metrics
        st.markdown("---")
        st.subheader("Forecast Summary")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Next Month Forecast", f"{int(forecast_values[0]):,}")
        col2.metric("Peak in Forecast", f"{int(max(forecast_values)):,}")
        col3.metric("Lowest in Forecast", f"{int(min(forecast_values)):,}")
        
        # Forecast table
        forecast_df = pd.DataFrame({
            'Month': future_dates.strftime('%Y-%m'),
            'Predicted Visitors': [int(v) for v in forecast_values]
        })
        st.dataframe(forecast_df, use_container_width=True)
        
    else:
        st.warning(f"Model {selected_model} not found for {selected_site}. Run Notebooks 03/04 first.")

# ============================================================
# PAGE 3: COMPARE
# ============================================================
elif page == "⚖️ Compare":
    st.title("⚖️ Model Comparison")
    st.markdown("Compare accuracy of all four models across sacred sites.")
    
    # Site selector
    selected_site = st.selectbox("Select Site for Detailed Comparison", ["All Sites"] + sites)
    
    st.markdown("---")
    
    if results:
        # Build comparison data
        model_names = {'arima': 'ARIMA', 'sarima': 'SARIMA', 'rf': 'Random Forest', 'xgb': 'XGBoost'}
        
        if selected_site == "All Sites":
            # Full comparison table
            st.subheader("MAPE Comparison — All Models × All Sites")
            
            comp_data = []
            for site in sites:
                row = {'Site': site}
                for key, name in model_names.items():
                    if key in results and site in results[key]:
                        row[name] = results[key][site]['MAPE']
                comp_data.append(row)
            
            comp_df = pd.DataFrame(comp_data)
            
            # Find best model per site
            model_cols = [n for n in model_names.values() if n in comp_df.columns]
            comp_df['Best Model'] = comp_df[model_cols].idxmin(axis=1)
            comp_df['Best MAPE'] = comp_df[model_cols].min(axis=1)
            
            st.dataframe(comp_df.style.format(
                {col: "{:.1f}%" for col in model_cols}
            ).format({'Best MAPE': '{:.1f}%'}), use_container_width=True)
            
            # Grouped bar chart
            st.subheader("Visual Comparison")
            
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(sites))
            width = 0.2
            bar_colors = ['#B4B2A9', '#5DCAA5', '#0F6E56', '#534AB7']
            
            for j, (col, color) in enumerate(zip(model_cols, bar_colors)):
                values = comp_df[col].values
                bars = ax.bar(x + (j - 1.5) * width, values, width, 
                             label=col, color=color, edgecolor='white')
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
            
            ax.set_ylabel('MAPE (%)')
            ax.set_title('Forecast Accuracy — All Models', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(sites)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Heatmap
            st.subheader("Accuracy Heatmap")
            heatmap_df = comp_df.set_index('Site')[model_cols]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='YlGn_r',
                       linewidths=1, linecolor='white', ax=ax,
                       cbar_kws={'label': 'MAPE (%) — lower is better'})
            ax.set_title('MAPE Heatmap', fontsize=13, fontweight='bold')
            ax.set_ylabel('')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Overall ranking
            st.subheader("Overall Model Ranking")
            avg_mape = {col: comp_df[col].mean() for col in model_cols}
            ranking = pd.DataFrame({
                'Model': avg_mape.keys(),
                'Average MAPE (%)': [round(v, 2) for v in avg_mape.values()]
            }).sort_values('Average MAPE (%)').reset_index(drop=True)
            ranking.index = ranking.index + 1
            ranking.index.name = 'Rank'
            st.dataframe(ranking, use_container_width=True)
            
        else:
            # Single site detailed comparison
            st.subheader(f"Detailed Comparison — {selected_site}")
            
            # Metrics table
            metrics_data = []
            for key, name in model_names.items():
                if key in results and selected_site in results[key]:
                    r = results[key][selected_site]
                    metrics_data.append({
                        'Model': name,
                        'MAE': f"{r['MAE']:,.0f}",
                        'RMSE': f"{r['RMSE']:,.0f}",
                        'MAPE (%)': f"{r['MAPE']:.1f}%"
                    })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Best model highlight
            if metrics_data:
                best = min(metrics_data, key=lambda x: float(x['MAPE (%)'].replace('%', '')))
                st.success(f"🏆 Best model for {selected_site}: **{best['Model']}** (MAPE: {best['MAPE (%)']})")

# ============================================================
# PAGE 4: INSIGHTS
# ============================================================
elif page == "💡 Insights":
    st.title("💡 Insights")
    st.markdown("Seasonal patterns, feature importance, and peak/off-peak analysis.")
    
    selected_site = st.selectbox("Select Site", sites)
    
    st.markdown("---")
    
    site_data = df[df['site'] == selected_site]
    site_no_covid = site_data[~site_data['year'].isin([2020, 2021])]
    
    # Seasonal heatmap
    st.subheader("Monthly Visitor Pattern")
    
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg = site_no_covid.groupby('month')['total_visitors'].mean()
    
    fig, ax = plt.subplots(figsize=(12, 3))
    hm_data = pd.DataFrame(monthly_avg).T
    hm_data.columns = month_labels
    hm_data.index = [selected_site]
    
    sns.heatmap(hm_data, annot=True, fmt=',.0f', cmap='YlGnBu',
               linewidths=1, linecolor='white', ax=ax,
               cbar_kws={'label': 'Average Visitors'})
    ax.set_title(f'Average Monthly Visitors — {selected_site} (excluding COVID)', 
                fontsize=12, fontweight='bold')
    ax.set_ylabel('')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Peak vs off-peak
    st.subheader("Peak vs Off-Peak")
    
    peak_month = monthly_avg.idxmax()
    offpeak_month = monthly_avg.idxmin()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Peak Month", month_labels[int(peak_month) - 1], 
                f"{int(monthly_avg.max()):,} visitors")
    col2.metric("Off-Peak Month", month_labels[int(offpeak_month) - 1],
                f"{int(monthly_avg.min()):,} visitors")
    col3.metric("Peak/Off-Peak Ratio", 
                f"{monthly_avg.max() / monthly_avg.min():.1f}x")
    
    st.markdown("---")
    
    # Festival impact
    st.subheader("Festival Impact")
    
    festival_avg = site_no_covid.groupby('festival')['total_visitors'].mean()
    if len(festival_avg) == 2:
        non_fest = festival_avg[0]
        fest = festival_avg[1]
        increase = ((fest - non_fest) / non_fest) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Non-Festival Average", f"{int(non_fest):,}")
        col2.metric("Festival Average", f"{int(fest):,}")
        col3.metric("Festival Boost", f"+{increase:.1f}%")
    
    st.markdown("---")
    
    # Feature importance (XGBoost)
    st.subheader("Feature Importance (XGBoost)")
    st.markdown("Which factors drive visitor numbers most at this site?")
    
    if selected_site in models and 'xgb' in models[selected_site] and feature_columns:
        xgb_model = models[selected_site]['xgb']
        importance = xgb_model.feature_importances_
        
        feat_imp = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(feat_imp['Feature'], feat_imp['Importance'], 
               color='#534AB7', alpha=0.7)
        ax.set_title(f'Feature Importance — {selected_site}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Top 3 features
        top3 = feat_imp.nlargest(3, 'Importance')
        st.markdown("**Top 3 most important features:**")
        for _, row in top3.iterrows():
            st.markdown(f"- **{row['Feature']}**: {row['Importance']:.4f}")
    else:
        st.info("XGBoost model not found. Run Notebook 04 first.")
    
    st.markdown("---")
    
    # Weather correlation
    st.subheader("Weather Correlation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temp_corr = site_no_covid['total_visitors'].corr(site_no_covid['temperature_c'])
        st.metric("Temperature Correlation", f"{temp_corr:.3f}")
        if abs(temp_corr) > 0.5:
            st.caption("Strong relationship")
        elif abs(temp_corr) > 0.3:
            st.caption("Moderate relationship")
        else:
            st.caption("Weak relationship")
    
    with col2:
        rain_corr = site_no_covid['total_visitors'].corr(site_no_covid['rainfall_mm'])
        st.metric("Rainfall Correlation", f"{rain_corr:.3f}")
        if abs(rain_corr) > 0.5:
            st.caption("Strong relationship")
        elif abs(rain_corr) > 0.3:
            st.caption("Moderate relationship")
        else:
            st.caption("Weak relationship")

# ============================================================
# FOOTER
# ============================================================
st.sidebar.markdown("---")
st.sidebar.caption("Forecasting Religious Tourism\nML Analysis of Pilgrimage Patterns\nto Sacred Sites in Nepal")
