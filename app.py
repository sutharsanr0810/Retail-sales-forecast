import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Retail Intelligence | Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL UI STYLING
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {
    background: #0f172a;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

.sidebar-brand {
    padding: 8px 0 25px 0;
}

.sidebar-brand-title {
    font-size: 20px;
    font-weight: 800;
    color: white;
}

.sidebar-brand-sub {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 4px;
}

/* ---------------- HERO ---------------- */

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #172554 55%, #1e3a8a 100%);
    padding: 38px 42px;
    border-radius: 20px;
    margin-bottom: 25px;
    color: white;
    position: relative;
    overflow: hidden;
}

.hero:after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    right: -90px;
    top: -120px;
    border-radius: 50%;
    background: rgba(59,130,246,0.15);
}

.hero-badge {
    display: inline-block;
    background: rgba(59,130,246,0.18);
    color: #93c5fd;
    padding: 6px 13px;
    border-radius: 30px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    border: 1px solid rgba(147,197,253,0.25);
    margin-bottom: 14px;
}

.hero-title {
    font-size: 32px;
    font-weight: 800;
    margin: 0;
    color: white;
    letter-spacing: -0.8px;
}

.hero-sub {
    font-size: 14px;
    color: #cbd5e1;
    margin-top: 9px;
    max-width: 800px;
    line-height: 1.6;
}

/* ---------------- SECTION ---------------- */

.section-title {
    font-size: 19px;
    font-weight: 750;
    color: #0f172a;
    margin-top: 28px;
    margin-bottom: 4px;
}

.section-sub {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 17px;
}

/* ---------------- KPI CARDS ---------------- */

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
    padding: 20px 21px;
    min-height: 118px;
    box-shadow: 0 3px 10px rgba(15,23,42,0.04);
}

.metric-label {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.metric-value {
    font-size: 25px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 8px;
}

.metric-sub {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 5px;
}

/* ---------------- STATUS CARDS ---------------- */

.status-card {
    border-radius: 14px;
    padding: 18px 20px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.status-title {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
}

.status-value {
    font-size: 20px;
    font-weight: 800;
    color: #0f172a;
    margin-top: 6px;
}

/* ---------------- TABS ---------------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background: #f1f5f9;
    padding: 5px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 17px;
    border-radius: 9px;
    font-weight: 600;
    font-size: 13px;
    color: #64748b;
}

.stTabs [aria-selected="true"] {
    background: white !important;
    color: #0f172a !important;
    box-shadow: 0 2px 5px rgba(15,23,42,0.08);
}

/* ---------------- TABLES ---------------- */

div[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    overflow: hidden;
}

/* ---------------- FILE UPLOADER ---------------- */

[data-testid="stFileUploader"] {
    background: #f8fafc;
    border: 1px dashed #cbd5e1;
    border-radius: 14px;
    padding: 8px;
}

/* ---------------- BUTTON ---------------- */

.stDownloadButton button {
    border-radius: 9px;
    font-weight: 600;
}

/* ---------------- DIVIDER ---------------- */

.soft-divider {
    height: 1px;
    background: #e2e8f0;
    margin: 25px 0;
}

/* ---------------- FOOTER ---------------- */

.app-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 11px;
    padding-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(label, value, sub=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_header(title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-sub">{subtitle}</div>
        """,
        unsafe_allow_html=True
    )


def format_number(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">📦 Retail Intelligence</div>
            <div class="sidebar-brand-sub">
                Demand Forecasting & Inventory Planning
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### Project")

    st.markdown(
        """
        **Model**  
        Random Forest

        **Estimators**  
        200 Trees

        **Learning Type**  
        Supervised ML

        **Forecast Target**  
        Future Sales
        """
    )

    st.markdown("---")

    st.markdown("### Dashboard")

    st.markdown(
        """
        Use the sections on the main dashboard to explore:

        • Forecast performance  
        • Inventory planning  
        • Demand spikes  
        • Business insights  
        • Next-month forecast
        """
    )

    st.markdown("---")

    st.caption("Retail ML Project")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            MACHINE LEARNING • RANDOM FOREST • RETAIL ANALYTICS
        </div>

        <div class="hero-title">
            Retail Demand Forecasting & Inventory Planning
        </div>

        <div class="hero-sub">
            Analyze historical retail transactions, forecast future demand,
            identify demand patterns, and support inventory planning through
            machine learning.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Retail Transaction Data",
    type="csv",
    help="Upload the retail CSV dataset used by the forecasting system."
)


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:65px 20px;
            background:#f8fafc;
            border:1px solid #e2e8f0;
            border-radius:18px;
            margin-top:20px;
        ">

            <div style="font-size:45px;">📊</div>

            <div style="
                font-size:21px;
                font-weight:750;
                color:#0f172a;
                margin-top:12px;
            ">
                Ready to Analyze Your Retail Data
            </div>

            <div style="
                color:#64748b;
                font-size:13px;
                margin-top:8px;
            ">
                Upload your retail transaction CSV above to generate
                forecasts, insights, and inventory recommendations.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(f"Unable to read the uploaded CSV: {e}")
    st.stop()


# ============================================================
# REQUIRED COLUMN VALIDATION
# ============================================================

required_columns = [
    "Order Date",
    "Category of Goods",
    "Sales",
    "Profit",
    "Quantity",
    "Discount",
    "Region"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "The uploaded dataset is missing required columns:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

try:

    df["Order Date"] = pd.to_datetime(df["Order Date"])

except Exception:

    st.error("The 'Order Date' column could not be converted to a date.")
    st.stop()


df["order_year"] = df["Order Date"].dt.year
df["order_month"] = df["Order Date"].dt.month


# ============================================================
# MONTHLY AGGREGATION
# ============================================================

monthly = (
    df.groupby(
        [
            "order_year",
            "order_month",
            "Category of Goods"
        ]
    )
    .agg(
        {
            "Sales": "sum",
            "Profit": "sum",
            "Quantity": "sum",
            "Discount": "mean"
        }
    )
    .reset_index()
    .sort_values(
        [
            "Category of Goods",
            "order_year",
            "order_month"
        ]
    )
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

monthly["sales_lag_1"] = (
    monthly
    .groupby("Category of Goods")["Sales"]
    .shift(1)
)

monthly["sales_lag_2"] = (
    monthly
    .groupby("Category of Goods")["Sales"]
    .shift(2)
)

monthly["sales_lag_3"] = (
    monthly
    .groupby("Category of Goods")["Sales"]
    .shift(3)
)

monthly["rolling_mean_3"] = (
    monthly
    .groupby("Category of Goods")["Sales"]
    .transform(
        lambda x: x.rolling(3).mean()
    )
)

monthly["sales_growth"] = (
    monthly
    .groupby("Category of Goods")["Sales"]
    .pct_change()
)

monthly["demand_spike"] = (
    monthly["sales_growth"] > 0.20
).astype(int)

monthly.dropna(inplace=True)


# ============================================================
# ENCODING
# ============================================================

monthly_encoded = pd.get_dummies(
    monthly,
    columns=["Category of Goods"],
    drop_first=True
)


features = [
    c
    for c in monthly_encoded.columns
    if c not in [
        "Sales",
        "Profit",
        "Quantity",
        "Discount",
        "demand_spike",
        "sales_growth"
    ]
]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

latest_year = monthly_encoded["order_year"].max()

train = monthly_encoded[
    monthly_encoded["order_year"] < latest_year
]

test = monthly_encoded[
    monthly_encoded["order_year"] == latest_year
]


X_train = train[features]
y_train = train["Sales"]

X_test = test[features]
y_test = test["Sales"]


# ============================================================
# RANDOM FOREST
# ============================================================

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

pred_sales = rf.predict(X_test)


# ============================================================
# MODEL EVALUATION
# ============================================================

mae = np.mean(
    np.abs(y_test - pred_sales)
)

rmse = np.sqrt(
    np.mean(
        (y_test - pred_sales) ** 2
    )
)

# Protect MAPE against zero actual values
non_zero_mask = y_test != 0

if non_zero_mask.sum() > 0:

    mape = np.mean(
        np.abs(
            (
                y_test[non_zero_mask]
                - pred_sales[non_zero_mask]
            )
            / y_test[non_zero_mask]
        )
    ) * 100

else:

    mape = np.nan


# ============================================================
# FUTURE FORECAST
# ============================================================

category_cols = [
    c
    for c in monthly_encoded.columns
    if c.startswith("Category of Goods_")
]


latest_rows = (
    monthly_encoded
    .sort_values(
        [
            "order_year",
            "order_month"
        ]
    )
    .groupby(
        category_cols,
        dropna=False
    )
    .tail(1)
    .copy()
)


future_rows = latest_rows.copy()


future_rows["order_year"] = latest_rows.apply(
    lambda r:
        r["order_year"] + 1
        if r["order_month"] == 12
        else r["order_year"],
    axis=1
)

future_rows["order_month"] = (
    latest_rows["order_month"] % 12
) + 1


future_rows["sales_lag_3"] = (
    latest_rows["sales_lag_2"]
)

future_rows["sales_lag_2"] = (
    latest_rows["sales_lag_1"]
)

future_rows["sales_lag_1"] = (
    latest_rows["Sales"]
)

future_rows["rolling_mean_3"] = (
    latest_rows[
        [
            "Sales",
            "sales_lag_1",
            "sales_lag_2"
        ]
    ].mean(axis=1)
)


future_pred = rf.predict(
    future_rows[features]
)

future_rows[
    "Predicted Next Month Sales"
] = future_pred


def label_category(row):

    for c in category_cols:

        if row[c]:

            return c.replace(
                "Category of Goods_",
                ""
            )

    return "Dairy Products"


future_rows["Category"] = (
    future_rows
    .apply(label_category, axis=1)
)


# ============================================================
# EXECUTIVE KPI STRIP
# ============================================================

st.markdown(
    '<div class="section-title">Executive Overview</div>',
    unsafe_allow_html=True
)

k1, k2, k3, k4 = st.columns(4)

with k1:

    metric_card(
        "Transactions",
        f"{df.shape[0]:,}",
        "records analyzed"
    )

with k2:

    metric_card(
        "Categories",
        f"{df['Category of Goods'].nunique():,}",
        "product categories"
    )

with k3:

    metric_card(
        "MAPE",
        f"{mape:.2f}%" if not np.isnan(mape) else "N/A",
        "lower is better"
    )

with k4:

    metric_card(
        "Model",
        "Random Forest",
        "200 estimators"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

st.markdown(
    '<div class="soft-divider"></div>',
    unsafe_allow_html=True
)

summary_cols = st.columns(4)

with summary_cols[0]:

    metric_card(
        "Total Sales",
        format_number(df["Sales"].sum()),
        "historical sales"
    )

with summary_cols[1]:

    metric_card(
        "Total Quantity",
        format_number(df["Quantity"].sum()),
        "units sold"
    )

with summary_cols[2]:

    metric_card(
        "Total Profit",
        format_number(df["Profit"].sum()),
        "historical profit"
    )

with summary_cols[3]:

    metric_card(
        "Latest Year",
        str(int(latest_year)),
        "held-out test year"
    )


# ============================================================
# NAVIGATION TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Forecast Performance",
        "📦 Inventory Planning",
        "🚨 Demand Spikes",
        "💡 Business Insights",
        "🔮 Next Month Forecast"
    ]
)


# ============================================================
# TAB 1 — FORECAST PERFORMANCE
# ============================================================

with tab1:

    section_header(
        "Model Performance",
        "Random Forest evaluated on out-of-time test data"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        metric_card(
            "MAE",
            f"{mae:,.0f}",
            "Mean Absolute Error"
        )

    with c2:

        metric_card(
            "RMSE",
            f"{rmse:,.0f}",
            "Root Mean Squared Error"
        )

    with c3:

        metric_card(
            "MAPE",
            f"{mape:.2f}%" if not np.isnan(mape) else "N/A",
            "Mean Absolute Percentage Error"
        )


    section_header(
        "Actual vs Predicted Sales",
        "Comparison of model predictions against held-out observations"
    )


    fig, ax = plt.subplots(
        figsize=(12, 4.5)
    )

    ax.plot(
        y_test.values,
        label="Actual",
        linewidth=2.3,
        marker="o",
        markersize=4
    )

    ax.plot(
        pred_sales,
        label="Predicted",
        linewidth=2.3,
        linestyle="--",
        marker="o",
        markersize=4
    )

    ax.set_xlabel(
        "Test Observation"
    )

    ax.set_ylabel(
        "Sales"
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"
        )
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # Feature importance

    section_header(
        "Feature Importance",
        "Signals the Random Forest relies on most"
    )

    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": rf.feature_importances_
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(8)
    )


    fig2, ax2 = plt.subplots(
        figsize=(10, 4.5)
    )

    ax2.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax2.invert_yaxis()

    ax2.set_xlabel(
        "Importance"
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2.grid(
        axis="x",
        alpha=0.2
    )

    st.pyplot(
        fig2,
        use_container_width=True
    )

    plt.close(fig2)


# ============================================================
# TAB 2 — INVENTORY PLANNING
# ============================================================

with tab2:

    section_header(
        "Inventory Planning",
        "Recommended inventory based on predicted sales plus a 15% safety-stock buffer"
    )


    results = test[
        [
            "order_year",
            "order_month"
        ]
    ].copy()

    results["Predicted Sales"] = (
        pred_sales
    )

    results["Safety Stock"] = (
        results["Predicted Sales"] * 0.15
    )

    results["Recommended Inventory"] = (
        results["Predicted Sales"]
        + results["Safety Stock"]
    )


    c1, c2, c3 = st.columns(3)

    with c1:

        metric_card(
            "Forecasted Demand",
            format_number(
                results["Predicted Sales"].sum()
            ),
            "test-period prediction"
        )

    with c2:

        metric_card(
            "Safety Stock",
            format_number(
                results["Safety Stock"].sum()
            ),
            "15% buffer"
        )

    with c3:

        metric_card(
            "Recommended Inventory",
            format_number(
                results["Recommended Inventory"].sum()
            ),
            "forecast + safety stock"
        )


    st.write("")


    display_results = results.copy()

    display_results["Period"] = (
        display_results["order_year"].astype(str)
        + "-"
        + display_results["order_month"]
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )

    display_results = display_results[
        [
            "Period",
            "Predicted Sales",
            "Safety Stock",
            "Recommended Inventory"
        ]
    ]


    st.dataframe(
        display_results.style.format(
            {
                "Predicted Sales": "{:,.0f}",
                "Safety Stock": "{:,.0f}",
                "Recommended Inventory": "{:,.0f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    csv_inventory = display_results.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇ Download Inventory Plan",
        data=csv_inventory,
        file_name="inventory_recommendations.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 3 — DEMAND SPIKES
# ============================================================

with tab3:

    section_header(
        "Demand Spike Detection",
        "A spike is flagged when month-over-month sales growth exceeds 20%"
    )


    spikes = monthly[
        monthly["demand_spike"] == 1
    ]


    spike_rate = (
        len(spikes) / len(monthly) * 100
        if len(monthly) > 0
        else 0
    )


    c1, c2 = st.columns(2)

    with c1:

        metric_card(
            "Spike Events",
            f"{len(spikes):,}",
            f"out of {len(monthly):,} observations"
        )

    with c2:

        metric_card(
            "Spike Rate",
            f"{spike_rate:.1f}%",
            "of monthly observations"
        )


    section_header(
        "Demand Spike Distribution",
        "Monthly observations classified by the rule-based spike detector"
    )


    counts = (
        monthly["demand_spike"]
        .value_counts()
        .sort_index()
    )


    labels = []

    values = []

    if 0 in counts.index:

        labels.append("No Spike")
        values.append(counts[0])

    if 1 in counts.index:

        labels.append("Spike")
        values.append(counts[1])


    fig3, ax3 = plt.subplots(
        figsize=(8, 4)
    )

    ax3.bar(
        labels,
        values,
        width=0.5
    )

    ax3.set_ylabel(
        "Number of Observations"
    )

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    ax3.grid(
        axis="y",
        alpha=0.2
    )

    st.pyplot(
        fig3,
        use_container_width=True
    )

    plt.close(fig3)


    section_header(
        "Detected Spike Events",
        "Periods where month-over-month sales growth exceeded 20%"
    )


    spike_display = spikes[
        [
            "order_year",
            "order_month",
            "Category of Goods",
            "Sales",
            "sales_growth"
        ]
    ].copy()


    spike_display["sales_growth"] = (
        spike_display["sales_growth"] * 100
    )


    st.dataframe(
        spike_display.style.format(
            {
                "Sales": "{:,.0f}",
                "sales_growth": "{:.1f}%"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 4 — BUSINESS INSIGHTS
# ============================================================

with tab4:

    section_header(
        "Business Insights",
        "High-level patterns extracted from the historical retail data"
    )


    monthly_avg = (
        df.groupby("order_month")["Sales"]
        .mean()
    )


    best_month = (
        monthly_avg.idxmax()
    )

    weakest_month = (
        monthly_avg.idxmin()
    )

    top_category = (
        df.groupby(
            "Category of Goods"
        )["Sales"]
        .sum()
        .idxmax()
    )

    top_region = (
        df.groupby("Region")["Sales"]
        .sum()
        .idxmax()
    )


    c1, c2, c3, c4 = st.columns(4)

    with c1:

        metric_card(
            "Best Month",
            str(best_month),
            "highest average sales"
        )

    with c2:

        metric_card(
            "Weakest Month",
            str(weakest_month),
            "lowest average sales"
        )

    with c3:

        metric_card(
            "Top Category",
            top_category,
            "highest total sales"
        )

    with c4:

        metric_card(
            "Top Region",
            top_region,
            "highest total sales"
        )


    section_header(
        "Sales by Category",
        "Total historical sales contribution by category"
    )


    category_sales = (
        df.groupby(
            "Category of Goods"
        )["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
    )


    fig4, ax4 = plt.subplots(
        figsize=(10, 5)
    )


    ax4.bar(
        category_sales.index,
        category_sales.values
    )


    ax4.set_ylabel(
        "Total Sales"
    )

    ax4.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"
        )
    )

    ax4.tick_params(
        axis="x",
        rotation=30
    )

    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    ax4.grid(
        axis="y",
        alpha=0.2
    )

    st.pyplot(
        fig4,
        use_container_width=True
    )

    plt.close(fig4)


# ============================================================
# TAB 5 — NEXT MONTH FORECAST
# ============================================================

with tab5:

    section_header(
        "Next Month Sales Forecast",
        "Predicted category-level sales using the most recent known trend"
    )


    forecast_display = future_rows[
        [
            "Category",
            "order_year",
            "order_month",
            "Predicted Next Month Sales"
        ]
    ].copy()


    forecast_display[
        "Recommended Inventory"
    ] = (
        forecast_display[
            "Predicted Next Month Sales"
        ] * 1.15
    )


    forecast_display = (
        forecast_display
        .rename(
            columns={
                "order_year": "Year",
                "order_month": "Month"
            }
        )
    )


    total_forecast = (
        forecast_display[
            "Predicted Next Month Sales"
        ].sum()
    )


    total_inventory = (
        forecast_display[
            "Recommended Inventory"
        ].sum()
    )


    c1, c2 = st.columns(2)

    with c1:

        metric_card(
            "Next Month Forecast",
            format_number(
                total_forecast
            ),
            "predicted category sales"
        )

    with c2:

        metric_card(
            "Recommended Inventory",
            format_number(
                total_inventory
            ),
            "forecast + 15% buffer"
        )


    st.write("")


    st.dataframe(
        forecast_display.style.format(
            {
                "Predicted Next Month Sales": "{:,.0f}",
                "Recommended Inventory": "{:,.0f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    csv_forecast = forecast_display.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇ Download Next Month Forecast",
        data=csv_forecast,
        file_name="next_month_forecast.csv",
        mime="text/csv"
    )


    section_header(
        "Category Forecast",
        "Expected sales for the upcoming month"
    )


    chart_data = (
        forecast_display
        .sort_values(
            "Predicted Next Month Sales",
            ascending=False
        )
    )


    fig5, ax5 = plt.subplots(
        figsize=(11, 5)
    )


    ax5.bar(
        chart_data["Category"],
        chart_data[
            "Predicted Next Month Sales"
        ]
    )


    ax5.set_ylabel(
        "Predicted Sales"
    )

    ax5.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"
        )
    )

    ax5.tick_params(
        axis="x",
        rotation=30
    )

    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    ax5.grid(
        axis="y",
        alpha=0.2
    )

    st.pyplot(
        fig5,
        use_container_width=True
    )

    plt.close(fig5)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        Retail Demand Forecasting & Inventory Planning
        • Random Forest • 200 Estimators
    </div>
    """,
    unsafe_allow_html=True
)
