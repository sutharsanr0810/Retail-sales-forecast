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
    page_title="Retail Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .stApp {
        background-color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background-color: #0f172a;
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .hero {
        padding: 35px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #0f172a,
            #172554,
            #1d4ed8
        );
        margin-bottom: 25px;
    }

    .hero-badge {
        color: #bfdbfe;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    .hero-title {
        color: white;
        font-size: 34px;
        font-weight: 800;
        line-height: 1.2;
    }

    .hero-text {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.6;
        max-width: 850px;
        margin-top: 12px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 25px;
    }

    .section-text {
        color: #64748b;
        font-size: 13px;
        margin-bottom: 15px;
    }

    .metric-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 18px;
        min-height: 115px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
    }

    .metric-label {
        color: #64748b;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .metric-value {
        color: #0f172a;
        font-size: 25px;
        font-weight: 800;
        margin-top: 7px;
    }

    .metric-sub {
        color: #94a3b8;
        font-size: 11px;
        margin-top: 5px;
    }

    .info-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def metric_card(label, value, description=""):

    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{description}</div>
        </div>
        """,
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


def calculate_mape(actual, predicted):

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

    st.title("📦 Retail Intelligence")

    st.caption(
        "Demand Forecasting & Inventory Planning"
    )

    st.divider()

    st.subheader("Project")

    st.write("**Model**")
    st.write("Random Forest")

    st.write("**Estimators**")
    st.write("200 Trees")

    st.write("**Learning Type**")
    st.write("Supervised ML")

    st.write("**Forecast Target**")
    st.write("Future Sales")

    st.divider()

    st.subheader("Dashboard")

    st.caption(
        "Use the dashboard sections to explore "
        "forecast performance, inventory planning, "
        "demand spikes, business insights and "
        "next-month forecasts."
    )

    st.divider()

    st.caption(
        "Retail Demand Forecasting Project"
    )


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
            Retail Demand Forecasting
            <br>
            & Inventory Planning
        </div>

        <div class="hero-text">
            Analyze historical retail transactions, forecast future demand,
            identify demand patterns, and support inventory planning
            through machine learning.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD DATASET
# ============================================================

st.subheader("Upload Retail Dataset")

uploaded_file = st.file_uploader(
    "Upload your retail transaction CSV",
    type=["csv"]
)


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    st.info(
        "📊 Upload your retail transaction CSV above "
        "to generate forecasts and business insights."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"Unable to read the CSV file: {e}"
    )

    st.stop()


# ============================================================
# REQUIRED COLUMNS
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
        "The uploaded dataset is missing required columns:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# DATE PROCESSING
# ============================================================

try:

    df["Order Date"] = pd.to_datetime(
        df["Order Date"]
    )

except Exception:

    st.error(
        "Order Date could not be converted to a valid date."
    )

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


# ============================================================
# DEMAND SPIKE
# EXISTING PROJECT THRESHOLD = 20%
# ============================================================

monthly["demand_spike"] = (
    monthly["sales_growth"] > 0.20
).astype(int)


# Remove rows where lag/rolling features don't exist

monthly = monthly.dropna().reset_index(drop=True)


# ============================================================
# ENCODING
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
        "Not enough yearly data to create the "
        "train/test split."
    )

    st.stop()


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


# ============================================================
# PREDICTION
# ============================================================

pred_sales = rf.predict(
    X_test
)


# ============================================================
# MODEL EVALUATION
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

mape = calculate_mape(
    y_test.values,
    pred_sales
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">Executive Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-text">Quick snapshot of the dataset and forecasting model</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)

with c1:

    metric_card(
        "Transactions",
        f"{len(df):,}",
        "records analyzed"
    )

with c2:

    metric_card(
        "Categories",
        f"{df['Category of Goods'].nunique():,}",
        "product categories"
    )

with c3:

    metric_card(
        "Forecast Accuracy",
        f"{mape:.2f}%",
        "MAPE • lower is better"
    )

with c4:

    metric_card(
        "Model",
        "Random Forest",
        "200 estimators"
    )


# ============================================================
# BUSINESS KPIs
# ============================================================

st.write("")

c1, c2, c3, c4 = st.columns(4)

with c1:

    metric_card(
        "Total Sales",
        format_number(
            df["Sales"].sum()
        ),
        "historical sales"
    )

with c2:

    metric_card(
        "Units Sold",
        format_number(
            df["Quantity"].sum()
        ),
        "total quantity"
    )

with c3:

    metric_card(
        "Total Profit",
        format_number(
            df["Profit"].sum()
        ),
        "historical profit"
    )

with c4:

    metric_card(
        "Test Year",
        str(latest_year),
        "most recent year held out"
    )


# ============================================================
# TABS
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
# TAB 1 - FORECAST PERFORMANCE
# ============================================================

with tab1:

    st.header("Model Accuracy")

    st.caption(
        "Evaluated on the most recent year held out from training."
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
            f"{mape:.2f}%",
            "MAPE • lower is better"
        )


    st.subheader("Actual vs Predicted Sales")

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        y_test.values,
        label="Actual Sales",
        linewidth=2
    )

    ax.plot(
        pred_sales,
        label="Predicted Sales",
        linewidth=2,
        linestyle="--"
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

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": rf.feature_importances_
        }
    ).sort_values(
        "Importance",
        ascending=False
    ).head(10)


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Importance"
    )

    ax.grid(
        axis="x",
        alpha=0.2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# TAB 2 - INVENTORY PLANNING
# ============================================================

with tab2:

    st.header("Inventory Planning")

    st.caption(
        "Forecast-based inventory planning using the project's "
        "15% safety-stock buffer."
    )


    inventory = test[
        [
            "order_year",
            "order_month"
        ]
    ].copy()


    inventory["Predicted Demand"] = pred_sales

    inventory["Safety Stock"] = (
        inventory["Predicted Demand"] * 0.15
    )

    inventory["Recommended Inventory"] = (
        inventory["Predicted Demand"]
        + inventory["Safety Stock"]
    )


    c1, c2, c3 = st.columns(3)

    with c1:

        metric_card(
            "Predicted Demand",
            format_number(
                inventory["Predicted Demand"].sum()
            ),
            "test-period forecast"
        )

    with c2:

        metric_card(
            "Safety Stock",
            format_number(
                inventory["Safety Stock"].sum()
            ),
            "15% buffer"
        )

    with c3:

        metric_card(
            "Recommended Inventory",
            format_number(
                inventory["Recommended Inventory"].sum()
            ),
            "forecast + safety stock"
        )


    st.subheader("Inventory Plan")


    inventory_display = inventory.copy()

    inventory_display["Period"] = (
        inventory_display["order_year"].astype(str)
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
        inventory_display,
        use_container_width=True,
        hide_index=True
    )


    st.download_button(
        "⬇ Download Inventory Plan",
        inventory_display.to_csv(
            index=False
        ).encode("utf-8"),
        "inventory_plan.csv",
        "text/csv"
    )


# ============================================================
# TAB 3 - DEMAND SPIKES
# ============================================================

with tab3:

    st.header("Demand Spike Detection")

    st.caption(
        "A demand spike is detected when month-over-month "
        "sales growth exceeds 20%."
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


    c1, c2 = st.columns(2)

    with c1:

        metric_card(
            "Demand Spike Events",
            f"{spike_count:,}",
            "detected observations"
        )

    with c2:

        metric_card(
            "Spike Rate",
            f"{spike_rate:.1f}%",
            "monthly observations"
        )


    st.subheader("Demand Spike Distribution")


    spike_counts = (
        monthly["demand_spike"]
        .value_counts()
        .sort_index()
    )


    labels = []
    values = []


    if 0 in spike_counts.index:

        labels.append("No Spike")
        values.append(spike_counts.loc[0])


    if 1 in spike_counts.index:

        labels.append("Spike")
        values.append(spike_counts.loc[1])


    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        labels,
        values
    )

    ax.set_ylabel(
        "Observations"
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    st.subheader("Detected Spike Events")


    spike_display = spikes[
        [
            "order_year",
            "order_month",
            "Category of Goods",
            "Sales",
            "sales_growth"
        ]
    ].copy()


    spike_display["sales_growth"] *= 100


    st.dataframe(
        spike_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 4 - BUSINESS INSIGHTS
# ============================================================

with tab4:

    st.header("Business Insights")

    st.caption(
        "Key patterns identified from the historical retail data."
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


    category_sales = (
        df.groupby(
            "Category of Goods"
        )["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
    )


    region_sales = (
        df.groupby(
            "Region"
        )["Sales"]
        .sum()
        .sort_values(
            ascending=False
        )
    )


    top_category = category_sales.index[0]

    top_region = region_sales.index[0]


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


    st.subheader("Sales by Category")


    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        category_sales.index,
        category_sales.values
    )

    ax.set_ylabel(
        "Total Sales"
    )

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"
        )
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    st.subheader("Regional Sales")


    regional_df = region_sales.reset_index()

    regional_df.columns = [
        "Region",
        "Sales"
    ]


    st.dataframe(
        regional_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 5 - NEXT MONTH FORECAST
# ============================================================

with tab5:

    st.header("Next Month Forecast")

    st.caption(
        "Category-level forecast generated using the trained "
        "Random Forest model."
    )


    # Latest observation for every category

    latest_category_rows = (
        monthly
        .sort_values(
            [
                "Category of Goods",
                "order_year",
                "order_month"
            ]
        )
        .groupby(
            "Category of Goods"
        )
        .tail(1)
        .copy()
    )


    future_base = latest_category_rows.copy()


    # --------------------------------------------------------
    # Move to next month
    # --------------------------------------------------------

    future_base["order_month"] = (
        future_base["order_month"] % 12
    ) + 1


    future_base["order_year"] = np.where(
        latest_category_rows["order_month"] == 12,
        latest_category_rows["order_year"] + 1,
        latest_category_rows["order_year"]
    )


    # --------------------------------------------------------
    # Update lag features
    # --------------------------------------------------------

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
    # Encode future data
    # --------------------------------------------------------

    future_encoded = pd.get_dummies(
        future_base,
        columns=["Category of Goods"],
        drop_first=True
    )


    for column in features:

        if column not in future_encoded.columns:

            future_encoded[column] = 0


    future_X = future_encoded[
        features
    ]


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    future_predictions = rf.predict(
        future_X
    )


    future_base[
        "Predicted Next Month Sales"
    ] = future_predictions


    # --------------------------------------------------------
    # 15% inventory buffer
    # --------------------------------------------------------

    future_base[
        "Recommended Inventory"
    ] = (
        future_base[
            "Predicted Next Month Sales"
        ] * 1.15
    )


    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    total_forecast = future_base[
        "Predicted Next Month Sales"
    ].sum()


    total_inventory = future_base[
        "Recommended Inventory"
    ].sum()


    c1, c2 = st.columns(2)

    with c1:

        metric_card(
            "Next Month Forecast",
            format_number(
                total_forecast
            ),
            "predicted sales"
        )

    with c2:

        metric_card(
            "Recommended Inventory",
            format_number(
                total_inventory
            ),
            "forecast + 15% buffer"
        )


    # --------------------------------------------------------
    # Forecast table
    # --------------------------------------------------------

    st.subheader("Category Forecast")


    forecast_display = future_base[
        [
            "Category of Goods",
            "order_year",
            "order_month",
            "Predicted Next Month Sales",
            "Recommended Inventory"
        ]
    ].copy()


    forecast_display.columns = [
        "Category",
        "Year",
        "Month",
        "Predicted Sales",
        "Recommended Inventory"
    ]


    st.dataframe(
        forecast_display,
        use_container_width=True,
        hide_index=True
    )


    st.download_button(
        "⬇ Download Next Month Forecast",
        forecast_display.to_csv(
            index=False
        ).encode("utf-8"),
        "next_month_forecast.csv",
        "text/csv"
    )


    # --------------------------------------------------------
    # Forecast chart
    # --------------------------------------------------------

    chart_data = forecast_display.sort_values(
        "Predicted Sales",
        ascending=False
    )


    fig, ax = plt.subplots(
        figsize=(11, 5)
    )


    ax.bar(
        chart_data["Category"],
        chart_data["Predicted Sales"]
    )


    ax.set_ylabel(
        "Predicted Sales"
    )


    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"
        )
    )


    ax.tick_params(
        axis="x",
        rotation=30
    )


    ax.grid(
        axis="y",
        alpha=0.2
    )


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    st.pyplot(
        fig,
        use_container_width=True
    )


    plt.close(fig)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Retail Demand Forecasting & Inventory Planning • "
    "Random Forest • 200 Estimators"
)
