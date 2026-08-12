import textwrap

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    textwrap.dedent(
        """
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
            max-width: 1450px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        /* ====================================================
           SIDEBAR
        ==================================================== */

        section[data-testid="stSidebar"] {
            background: #0f172a;
        }

        section[data-testid="stSidebar"] * {
            color: #e2e8f0;
        }

        .sidebar-brand {
            padding: 10px 0 20px 0;
        }

        .sidebar-title {
            font-size: 21px;
            font-weight: 800;
            color: #ffffff;
        }

        .sidebar-subtitle {
            font-size: 12px;
            color: #94a3b8;
            margin-top: 5px;
        }

        .sidebar-info {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 14px;
            margin-top: 10px;
        }

        .sidebar-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.7px;
            color: #94a3b8;
            font-weight: 700;
        }

        .sidebar-value {
            font-size: 14px;
            color: white;
            font-weight: 600;
            margin-top: 3px;
            margin-bottom: 12px;
        }

        /* ====================================================
           HERO
        ==================================================== */

        .hero {
            background: linear-gradient(
                135deg,
                #0f172a 0%,
                #172554 55%,
                #1d4ed8 100%
            );

            border-radius: 22px;
            padding: 42px;
            margin-bottom: 25px;
            color: white;
            position: relative;
            overflow: hidden;
        }

        .hero-circle {
            position: absolute;
            width: 330px;
            height: 330px;
            right: -110px;
            top: -160px;
            border-radius: 50%;
            background: rgba(255,255,255,0.06);
        }

        .hero-circle-two {
            position: absolute;
            width: 200px;
            height: 200px;
            right: 80px;
            bottom: -150px;
            border-radius: 50%;
            background: rgba(255,255,255,0.04);
        }

        .hero-content {
            position: relative;
            z-index: 2;
        }

        .hero-badge {
            display: inline-block;
            padding: 7px 13px;
            border-radius: 30px;
            background: rgba(147,197,253,0.12);
            border: 1px solid rgba(147,197,253,0.25);
            color: #bfdbfe;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.8px;
            margin-bottom: 15px;
        }

        .hero-title {
            font-size: 34px;
            line-height: 1.15;
            font-weight: 800;
            letter-spacing: -1px;
            color: white;
        }

        .hero-description {
            margin-top: 12px;
            max-width: 850px;
            font-size: 14px;
            line-height: 1.65;
            color: #cbd5e1;
        }

        /* ====================================================
           SECTION HEADERS
        ==================================================== */

        .section-title {
            font-size: 20px;
            font-weight: 800;
            color: #0f172a;
            margin-top: 28px;
            margin-bottom: 4px;
        }

        .section-description {
            color: #64748b;
            font-size: 13px;
            margin-bottom: 17px;
        }

        /* ====================================================
           METRIC CARDS
        ==================================================== */

        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 21px;
            min-height: 125px;
            box-shadow: 0 4px 14px rgba(15,23,42,0.045);
        }

        .metric-label {
            color: #64748b;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 700;
        }

        .metric-value {
            color: #0f172a;
            font-size: 25px;
            font-weight: 800;
            margin-top: 8px;
            word-break: break-word;
        }

        .metric-description {
            color: #94a3b8;
            font-size: 11px;
            margin-top: 5px;
        }

        /* ====================================================
           INFO CARDS
        ==================================================== */

        .info-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 15px;
            padding: 20px;
            margin-top: 10px;
        }

        .info-title {
            font-size: 12px;
            font-weight: 800;
            color: #334155;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .info-text {
            color: #64748b;
            font-size: 13px;
            line-height: 1.6;
            margin-top: 8px;
        }

        /* ====================================================
           UPLOAD AREA
        ==================================================== */

        [data-testid="stFileUploader"] {
            background: #f8fafc;
            border: 1px dashed #cbd5e1;
            border-radius: 15px;
            padding: 8px;
        }

        /* ====================================================
           TABS
        ==================================================== */

        .stTabs [data-baseweb="tab-list"] {
            gap: 5px;
            background: #f1f5f9;
            padding: 5px;
            border-radius: 13px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 10px 15px;
            border-radius: 9px;
            font-size: 12px;
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            background: white !important;
            color: #0f172a !important;
            box-shadow: 0 2px 6px rgba(15,23,42,0.08);
        }

        /* ====================================================
           DIVIDER
        ==================================================== */

        .divider {
            height: 1px;
            background: #e2e8f0;
            margin: 27px 0;
        }

        /* ====================================================
           EMPTY STATE
        ==================================================== */

        .empty-state {
            text-align: center;
            padding: 65px 25px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            margin-top: 20px;
        }

        .empty-icon {
            font-size: 48px;
        }

        .empty-title {
            color: #0f172a;
            font-size: 21px;
            font-weight: 800;
            margin-top: 12px;
        }

        .empty-description {
            color: #64748b;
            font-size: 13px;
            margin-top: 8px;
        }

        /* ====================================================
           FOOTER
        ==================================================== */

        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 11px;
            padding-top: 35px;
        }

        </style>
        """
    ),
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(label, value, description=""):
    html = textwrap.dedent(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-description">{description}</div>
        </div>
        """
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def section_header(title, description=""):
    html = textwrap.dedent(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-description">{description}</div>
        """
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def format_number(value):

    if pd.isna(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"

    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"

    return f"{value:,.0f}"


def safe_mape(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        textwrap.dedent(
            """
            <div class="sidebar-brand">

                <div class="sidebar-title">
                    📦 Retail Intelligence
                </div>

                <div class="sidebar-subtitle">
                    Demand Forecasting & Inventory Planning
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Project")

    st.markdown(
        textwrap.dedent(
            """
            <div class="sidebar-info">

                <div class="sidebar-label">Model</div>
                <div class="sidebar-value">Random Forest</div>

                <div class="sidebar-label">Estimators</div>
                <div class="sidebar-value">200 Trees</div>

                <div class="sidebar-label">Learning Type</div>
                <div class="sidebar-value">Supervised ML</div>

                <div class="sidebar-label">Forecast Target</div>
                <div class="sidebar-value">Future Sales</div>

            </div>
            """
        ),
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Dashboard")

    st.caption(
        "Explore forecast performance, inventory planning, "
        "demand spikes, business insights and next-month forecasts."
    )

    st.divider()

    st.caption(
        "Retail Demand Forecasting Project"
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <div class="hero">

            <div class="hero-circle"></div>
            <div class="hero-circle-two"></div>

            <div class="hero-content">

                <div class="hero-badge">
                    MACHINE LEARNING • RANDOM FOREST • RETAIL ANALYTICS
                </div>

                <div class="hero-title">
                    Retail Demand Forecasting
                    <br>
                    & Inventory Planning
                </div>

                <div class="hero-description">
                    Analyze historical retail transactions, forecast future
                    demand, identify demand patterns, and support inventory
                    planning through machine learning.
                </div>

            </div>

        </div>
        """
    ),
    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    "### Upload Retail Dataset"
)

uploaded_file = st.file_uploader(
    "Upload your retail transaction CSV",
    type=["csv"],
    label_visibility="collapsed"
)


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.markdown(
        textwrap.dedent(
            """
            <div class="empty-state">

                <div class="empty-icon">
                    📊
                </div>

                <div class="empty-title">
                    Ready to Analyze Your Retail Data
                </div>

                <div class="empty-description">
                    Upload your retail transaction CSV above to generate
                    forecasts, model performance metrics, demand insights,
                    and inventory recommendations.
                </div>

            </div>
            """
        ),
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"Unable to read the uploaded CSV: {e}"
    )

    st.stop()


# ============================================================
# COLUMN VALIDATION
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
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "The uploaded dataset is missing required columns."
    )

    st.write(
        missing_columns
    )

    st.stop()


# ============================================================
# DATE PREPARATION
# ============================================================

try:

    df["Order Date"] = pd.to_datetime(
        df["Order Date"]
    )

except Exception:

    st.error(
        "The 'Order Date' column could not be converted to a valid date."
    )

    st.stop()


df["order_year"] = (
    df["Order Date"].dt.year
)

df["order_month"] = (
    df["Order Date"].dt.month
)


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

# Existing project demand-spike rule:
# month-over-month growth > 20%

monthly["demand_spike"] = (
    monthly["sales_growth"] > 0.20
).astype(int)


# Remove rows without enough historical lag information

monthly.dropna(
    inplace=True
)


# ============================================================
# ONE-HOT ENCODING
# ============================================================

monthly_encoded = pd.get_dummies(
    monthly,
    columns=["Category of Goods"],
    drop_first=True
)


# ============================================================
# FEATURES
# ============================================================

excluded_columns = [
    "Sales",
    "Profit",
    "Quantity",
    "Discount",
    "demand_spike",
    "sales_growth"
]

features = [
    column
    for column in monthly_encoded.columns
    if column not in excluded_columns
]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

latest_year = int(
    monthly_encoded["order_year"].max()
)

train = monthly_encoded[
    monthly_encoded["order_year"] < latest_year
]

test = monthly_encoded[
    monthly_encoded["order_year"] == latest_year
]


if train.empty or test.empty:

    st.error(
        "The dataset does not contain enough yearly data "
        "to create a train/test split."
    )

    st.stop()


X_train = train[features]

y_train = train["Sales"]

X_test = test[features]

y_test = test["Sales"]


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

rf = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

pred_sales = rf.predict(
    X_test
)


# ============================================================
# MODEL METRICS
# ============================================================

mae = np.mean(
    np.abs(
        y_test.values - pred_sales
    )
)

rmse = np.sqrt(
    np.mean(
        (y_test.values - pred_sales) ** 2
    )
)

mape = safe_mape(
    y_test.values,
    pred_sales
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

section_header(
    "Executive Overview",
    "A quick snapshot of the retail dataset and forecasting model"
)


k1, k2, k3, k4 = st.columns(4)


with k1:

    metric_card(
        "Transactions",
        f"{len(df):,}",
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
        "Forecast MAPE",
        f"{mape:.2f}%" if not np.isnan(mape) else "N/A",
        "lower is better"
    )


with k4:

    metric_card(
        "Model",
        "Random Forest",
        "200 estimators"
    )


st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)


# ============================================================
# BUSINESS KPI ROW
# ============================================================

b1, b2, b3, b4 = st.columns(4)


with b1:

    metric_card(
        "Total Sales",
        format_number(
            df["Sales"].sum()
        ),
        "historical sales"
    )


with b2:

    metric_card(
        "Units Sold",
        format_number(
            df["Quantity"].sum()
        ),
        "total quantity"
    )


with b3:

    metric_card(
        "Total Profit",
        format_number(
            df["Profit"].sum()
        ),
        "historical profit"
    )


with b4:

    metric_card(
        "Test Year",
        str(latest_year),
        "most recent year held out"
    )


st.write("")


# ============================================================
# DASHBOARD TABS
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
# TAB 1
# FORECAST PERFORMANCE
# ============================================================

with tab1:

    section_header(
        "Model Accuracy",
        "Random Forest evaluated on out-of-time test data"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        metric_card(
            "Mean Absolute Error",
            f"{mae:,.0f}",
            "MAE • lower is better"
        )


    with c2:

        metric_card(
            "Root Mean Squared Error",
            f"{rmse:,.0f}",
            "RMSE • lower is better"
        )


    with c3:

        metric_card(
            "Mean Absolute Percentage Error",
            f"{mape:.2f}%" if not np.isnan(mape) else "N/A",
            "MAPE • lower is better"
        )


    section_header(
        "Actual vs Predicted Sales",
        "Comparison between observed sales and Random Forest predictions"
    )


    fig, ax = plt.subplots(
        figsize=(12, 5)
    )


    ax.plot(
        y_test.values,
        label="Actual Sales",
        linewidth=2.2,
        marker="o",
        markersize=3
    )


    ax.plot(
        pred_sales,
        label="Predicted Sales",
        linewidth=2.2,
        linestyle="--",
        marker="o",
        markersize=3
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


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    section_header(
        "Feature Importance",
        "Relative importance of model input features"
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
        .head(10)
    )


    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
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
# TAB 2
# INVENTORY PLANNING
# ============================================================

with tab2:

    section_header(
        "Inventory Planning",
        "Predicted demand plus the project's 15% safety-stock buffer"
    )


    inventory = test[
        [
            "order_year",
            "order_month"
        ]
    ].copy()


    inventory["Predicted Demand"] = (
        pred_sales
    )


    inventory["Safety Stock"] = (
        inventory["Predicted Demand"] * 0.15
    )


    inventory["Recommended Inventory"] = (
        inventory["Predicted Demand"]
        + inventory["Safety Stock"]
    )


    i1, i2, i3 = st.columns(3)


    with i1:

        metric_card(
            "Predicted Demand",
            format_number(
                inventory["Predicted Demand"].sum()
            ),
            "test-period forecast"
        )


    with i2:

        metric_card(
            "Safety Stock",
            format_number(
                inventory["Safety Stock"].sum()
            ),
            "15% buffer"
        )


    with i3:

        metric_card(
            "Recommended Inventory",
            format_number(
                inventory["Recommended Inventory"].sum()
            ),
            "forecast + safety stock"
        )


    st.write("")


    inventory_display = inventory.copy()


    inventory_display["Period"] = (
        inventory_display["order_year"]
        .astype(str)
        + "-"
        + inventory_display["order_month"]
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )


    inventory_display = inventory_display[
        [
            "Period",
            "Predicted Demand",
            "Safety Stock",
            "Recommended Inventory"
        ]
    ]


    st.dataframe(
        inventory_display.style.format(
            {
                "Predicted Demand": "{:,.0f}",
                "Safety Stock": "{:,.0f}",
                "Recommended Inventory": "{:,.0f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    inventory_csv = (
        inventory_display
        .to_csv(index=False)
        .encode("utf-8")
    )


    st.download_button(
        "⬇ Download Inventory Plan",
        inventory_csv,
        "inventory_plan.csv",
        "text/csv"
    )


# ============================================================
# TAB 3
# DEMAND SPIKES
# ============================================================

with tab3:

    section_header(
        "Demand Spike Detection",
        "A spike is detected when month-over-month sales growth exceeds 20%"
    )


    spikes = monthly[
        monthly["demand_spike"] == 1
    ]


    spike_count = len(spikes)


    spike_rate = (
        spike_count / len(monthly) * 100
        if len(monthly) > 0
        else 0
    )


    s1, s2 = st.columns(2)


    with s1:

        metric_card(
            "Demand Spike Events",
            f"{spike_count:,}",
            "detected observations"
        )


    with s2:

        metric_card(
            "Spike Rate",
            f"{spike_rate:.1f}%",
            "of monthly observations"
        )


    section_header(
        "Demand Spike Distribution",
        "Distribution of observations with and without detected spikes"
    )


    spike_counts = (
        monthly["demand_spike"]
        .value_counts()
        .sort_index()
    )


    labels = []

    values = []


    if 0 in spike_counts.index:

        labels.append(
            "No Spike"
        )

        values.append(
            spike_counts.loc[0]
        )


    if 1 in spike_counts.index:

        labels.append(
            "Spike"
        )

        values.append(
            spike_counts.loc[1]
        )


    fig3, ax3 = plt.subplots(
        figsize=(8, 4.5)
    )


    ax3.bar(
        labels,
        values,
        width=0.5
    )


    ax3.set_ylabel(
        "Observations"
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
        "Monthly observations where sales growth exceeded the 20% threshold"
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
# TAB 4
# BUSINESS INSIGHTS
# ============================================================

with tab4:

    section_header(
        "Business Insights",
        "Key patterns extracted from the historical retail data"
    )


    monthly_average = (
        df.groupby("order_month")["Sales"]
        .mean()
    )


    best_month = int(
        monthly_average.idxmax()
    )


    weakest_month = int(
        monthly_average.idxmin()
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


    q1, q2, q3, q4 = st.columns(4)


    with q1:

        metric_card(
            "Best Month",
            str(best_month),
            "highest average sales"
        )


    with q2:

        metric_card(
            "Weakest Month",
            str(weakest_month),
            "lowest average sales"
        )


    with q3:

        metric_card(
            "Top Category",
            top_category,
            "highest total sales"
        )


    with q4:

        metric_card(
            "Top Region",
            top_region,
            "highest total sales"
        )


    section_header(
        "Sales by Category",
        "Historical sales contribution by product category"
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
        figsize=(11, 5)
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
# TAB 5
# NEXT MONTH FORECAST
# ============================================================

with tab5:

    section_header(
        "Next Month Forecast",
        "Category-level forecast generated using the trained Random Forest model"
    )


    # --------------------------------------------------------
    # Get latest row for each category
    # --------------------------------------------------------

    category_column = "Category of Goods"

    latest_category_rows = (
        monthly
        .sort_values(
            [
                category_column,
                "order_year",
                "order_month"
            ]
        )
        .groupby(
            category_column
        )
        .tail(1)
        .copy()
    )


    # --------------------------------------------------------
    # Prepare future features
    # --------------------------------------------------------

    future_base = latest_category_rows.copy()


    future_base["order_year"] = np.where(
        future_base["order_month"] == 12,
        future_base["order_year"] + 1,
        future_base["order_year"]
    )


    future_base["order_month"] = (
        future_base["order_month"] % 12
    ) + 1


    future_base["sales_lag_3"] = (
        future_base["sales_lag_2"]
    )


    future_base["sales_lag_2"] = (
        future_base["sales_lag_1"]
    )


    future_base["sales_lag_1"] = (
        future_base["Sales"]
    )


    future_base["rolling_mean_3"] = (
        future_base[
            [
                "Sales",
                "sales_lag_1",
                "sales_lag_2"
            ]
        ].mean(axis=1)
    )


    # --------------------------------------------------------
    # Encode future category data
    # --------------------------------------------------------

    future_encoded = pd.get_dummies(
        future_base,
        columns=[
            "Category of Goods"
        ],
        drop_first=True
    )


    # Add any missing training columns

    for column in features:

        if column not in future_encoded.columns:

            future_encoded[column] = 0


    # Keep only model features

    future_X = future_encoded[
        features
    ]


    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    future_predictions = rf.predict(
        future_X
    )


    future_base[
        "Predicted Next Month Sales"
    ] = future_predictions


    # --------------------------------------------------------
    # Inventory recommendation
    # --------------------------------------------------------

    future_base[
        "Recommended Inventory"
    ] = (
        future_base[
            "Predicted Next Month Sales"
        ] * 1.15
    )


    # --------------------------------------------------------
    # Forecast KPIs
    # --------------------------------------------------------

    total_forecast = (
        future_base[
            "Predicted Next Month Sales"
        ].sum()
    )


    total_recommended_inventory = (
        future_base[
            "Recommended Inventory"
        ].sum()
    )


    f1, f2 = st.columns(2)


    with f1:

        metric_card(
            "Next Month Forecast",
            format_number(
                total_forecast
            ),
            "predicted sales"
        )


    with f2:

        metric_card(
            "Recommended Inventory",
            format_number(
                total_recommended_inventory
            ),
            "forecast + 15% buffer"
        )


    section_header(
        "Category Forecast",
        "Predicted sales and recommended inventory for the upcoming month"
    )


    forecast_display = future_base[
        [
            "Category of Goods",
            "order_year",
            "order_month",
            "Predicted Next Month Sales",
            "Recommended Inventory"
        ]
    ].copy()


    forecast_display = (
        forecast_display
        .rename(
            columns={
                "Category of Goods": "Category",
                "order_year": "Year",
                "order_month": "Month"
            }
        )
    )


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


    forecast_csv = (
        forecast_display
        .to_csv(index=False)
        .encode("utf-8")
    )


    st.download_button(
        "⬇ Download Next Month Forecast",
        forecast_csv,
        "next_month_forecast.csv",
        "text/csv"
    )


    # --------------------------------------------------------
    # Forecast chart
    # --------------------------------------------------------

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
    textwrap.dedent(
        """
        <div class="footer">
            Retail Demand Forecasting & Inventory Planning
            &nbsp;•&nbsp;
            Random Forest
            &nbsp;•&nbsp;
            200 Estimators
        </div>
        """
    ),
    unsafe_allow_html=True
)
