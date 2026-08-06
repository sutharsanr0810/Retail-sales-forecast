
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Retail Demand Forecasting",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        padding: 40px 45px;
        border-radius: 16px;
        margin-bottom: 30px;
        color: white;
    }
    .hero-title {
        font-size: 30px;
        font-weight: 800;
        margin: 0 0 6px 0;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 15px;
        color: #94a3b8;
        margin: 0;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 14px;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        height: 100%;
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 4px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        margin: 30px 0 4px 0;
    }
    .section-sub {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 18px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f5f9;
        padding: 5px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 9px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13.5px;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #0f172a !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
    }

    .stFileUploader {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero">
        <div class="hero-badge">MACHINE LEARNING · RANDOM FOREST</div>
        <div class="hero-title">Retail Demand Forecasting & Inventory Planning</div>
        <div class="hero-sub">Category-wise sales prediction using historical transaction patterns, lag features, and seasonal trend analysis</div>
    </div>
""", unsafe_allow_html=True)

def metric_card(label, value, sub=""):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload retail transaction data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['order_year'] = df['Order Date'].dt.year
    df['order_month'] = df['Order Date'].dt.month

    monthly = (
        df.groupby(['order_year', 'order_month', 'Category of Goods'])
        .agg({'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum', 'Discount': 'mean'})
        .reset_index()
        .sort_values(['Category of Goods', 'order_year', 'order_month'])
    )

    monthly['sales_lag_1'] = monthly.groupby('Category of Goods')['Sales'].shift(1)
    monthly['sales_lag_2'] = monthly.groupby('Category of Goods')['Sales'].shift(2)
    monthly['sales_lag_3'] = monthly.groupby('Category of Goods')['Sales'].shift(3)
    monthly['rolling_mean_3'] = monthly.groupby('Category of Goods')['Sales'].transform(lambda x: x.rolling(3).mean())
    monthly['sales_growth'] = monthly.groupby('Category of Goods')['Sales'].pct_change()
    monthly['demand_spike'] = (monthly['sales_growth'] > 0.20).astype(int)
    monthly.dropna(inplace=True)

    monthly_encoded = pd.get_dummies(monthly, columns=['Category of Goods'], drop_first=True)

    features = [c for c in monthly_encoded.columns
                if c not in ['Sales', 'Profit', 'Quantity', 'Discount', 'demand_spike', 'sales_growth']]

    latest_year = monthly_encoded['order_year'].max()
    train = monthly_encoded[monthly_encoded['order_year'] < latest_year]
    test = monthly_encoded[monthly_encoded['order_year'] == latest_year]

    X_train, y_train = train[features], train['Sales']
    X_test, y_test = test[features], test['Sales']

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    pred_sales = rf.predict(X_test)

    mae = np.mean(np.abs(y_test - pred_sales))
    rmse = np.sqrt(np.mean((y_test - pred_sales) ** 2))
    mape = np.mean(np.abs((y_test - pred_sales) / y_test)) * 100

    # Top strip of key metrics, always visible
    k1, k2, k3, k4 = st.columns(4)
    with k1: metric_card("Dataset Size", f"{df.shape[0]:,}", "transactions")
    with k2: metric_card("Categories", df['Category of Goods'].nunique(), "product categories")
    with k3: metric_card("Forecast Accuracy", f"{mape:.2f}%", "MAPE (lower is better)")
    with k4: metric_card("Model", "Random Forest", "200 estimators")

    st.write("")
    tab1, tab2, tab3, tab4 = st.tabs(["Forecast Performance", "Inventory Planning", "Demand Spikes", "Business Insights"])

    with tab1:
        st.markdown('<div class="section-title">Model Accuracy</div><div class="section-sub">Evaluated on out-of-time test data (most recent year held out)</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Mean Absolute Error", f"{mae:,.0f}")
        with c2: metric_card("RMSE", f"{rmse:,.0f}")
        with c3: metric_card("MAPE", f"{mape:.2f}%")

        st.markdown('<div class="section-title">Actual vs Predicted Sales</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('white')
        ax.plot(y_test.values, label="Actual", marker="o", color="#0f172a", linewidth=2.2, markersize=5)
        ax.plot(pred_sales, label="Predicted", marker="o", color="#3b82f6", linewidth=2.2, markersize=5, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e2e8f0')
        ax.spines['bottom'].set_color('#e2e8f0')
        ax.grid(axis='y', color='#f1f5f9', linewidth=1)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=10)
        st.pyplot(fig)

        st.markdown('<div class="section-title">Feature Importance</div><div class="section-sub">Which signals the model relies on most</div>', unsafe_allow_html=True)
        importance_df = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
        importance_df = importance_df.sort_values('Importance', ascending=False).head(8)
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        fig2.patch.set_facecolor('white')
        colors = ['#0f172a' if i == 0 else '#3b82f6' for i in range(len(importance_df))]
        ax2.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
        ax2.invert_yaxis()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.grid(axis='x', color='#f1f5f9', linewidth=1)
        ax2.set_axisbelow(True)
        st.pyplot(fig2)

    with tab2:
        st.markdown('<div class="section-title">Recommended Inventory</div><div class="section-sub">Forecasted sales plus 15% safety stock buffer</div>', unsafe_allow_html=True)
        results = test[['order_year', 'order_month']].copy()
        results['Predicted Sales'] = pred_sales
        results['Safety Stock'] = results['Predicted Sales'] * 0.15
        results['Recommended Inventory'] = results['Predicted Sales'] + results['Safety Stock']
        st.dataframe(
            results.style.format({
                'Predicted Sales': '{:,.0f}',
                'Safety Stock': '{:,.0f}',
                'Recommended Inventory': '{:,.0f}'
            }).background_gradient(subset=['Recommended Inventory'], cmap='Blues'),
            use_container_width=True
        )

    with tab3:
        st.markdown('<div class="section-title">Rule-Based Demand Spike Detection</div><div class="section-sub">Flagged when month-over-month sales growth exceeds 20%</div>', unsafe_allow_html=True)
        spikes = monthly[monthly['demand_spike'] == 1]

        c1, c2 = st.columns(2)
        with c1: metric_card("Total Spike Events", len(spikes), f"out of {len(monthly)} observations")
        with c2: metric_card("Spike Rate", f"{len(spikes)/len(monthly)*100:.1f}%", "of all months")

        st.write("")
        fig3, ax3 = plt.subplots(figsize=(6, 3))
        fig3.patch.set_facecolor('white')
        counts = monthly['demand_spike'].value_counts().sort_index()
        ax3.bar(['No Spike', 'Spike'], counts.values, color=["#e2e8f0", "#3b82f6"], width=0.5)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.grid(axis='y', color='#f1f5f9', linewidth=1)
        ax3.set_axisbelow(True)
        st.pyplot(fig3)

        st.dataframe(
            spikes[['order_year', 'order_month', 'Category of Goods', 'Sales', 'sales_growth']],
            use_container_width=True
        )

    with tab4:
        st.markdown('<div class="section-title">Key Business Insights</div>', unsafe_allow_html=True)
        monthly_avg = df.groupby('order_month')['Sales'].mean()
        best_month = monthly_avg.idxmax()
        top_category = df.groupby('Category of Goods')['Sales'].sum().idxmax()
        top_region = df.groupby('Region')['Sales'].sum().idxmax()

        c1, c2 = st.columns(2)
        with c1: metric_card("Best Performing Month", best_month, "highest average sales")
        with c2: metric_card("Weakest Month", monthly_avg.idxmin(), "lowest average sales")
        c3, c4 = st.columns(2)
        with c3: metric_card("Top Category", top_category, "by total sales")
        with c4: metric_card("Top Region", top_region, "by total sales")

else:
    st.markdown("""
        <div style="text-align:center; padding: 60px 20px; color: #94a3b8;">
            <div style="font-size:15px;">Upload your retail transaction CSV above to generate the forecast dashboard</div>
        </div>
    """, unsafe_allow_html=True)