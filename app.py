import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📦 Retail Demand Forecasting & Inventory Planning")

st.caption(
    "Machine Learning • Random Forest • Retail Analytics"
)

st.write(
    "Analyze historical retail transactions, forecast future demand, "
    "identify demand patterns, and support inventory planning "
    "through machine learning."
)

st.divider()

with st.sidebar:
    st.title("📦 Retail Intelligence")
    st.caption("Demand Forecasting & Inventory Planning")

    st.divider()

    st.subheader("Project")
    st.write("**Model:** Random Forest")
    st.write("**Estimators:** 200 Trees")
    st.write("**Learning Type:** Supervised ML")
    st.write("**Forecast Target:** Future Sales")

    st.divider()

    st.subheader("Dashboard")
    st.write(
        "Use the sections on the dashboard to explore "
        "forecast performance, inventory planning, "
        "demand spikes, business insights and "
        "next-month forecasts."
    )

    st.divider()

    st.caption("Retail Demand Forecasting Project")

st.header("Upload Retail Dataset")

uploaded_file = st.file_uploader(
    "Upload your retail transaction CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Upload the retail CSV dataset to start the analysis."
    )
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read the CSV file: {e}")
    st.stop()

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
        "The uploaded dataset is missing the following required columns:"
    )

    for column in missing_columns:
        st.write(f"- {column}")

    st.stop()

try:
    df["Order Date"] = pd.to_datetime(df["Order Date"])
except Exception:
    st.error(
        "The 'Order Date' column could not be converted to a valid date."
    )
    st.stop()

df["order_year"] = df["Order Date"].dt.year
df["order_month"] = df["Order Date"].dt.month

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

monthly = (
    monthly
    .dropna()
    .reset_index(drop=True)
)

monthly_encoded = pd.get_dummies(
    monthly,
    columns=["Category of Goods"],
    drop_first=True
)

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
        "There is not enough yearly data to create the train/test split."
    )
    st.stop()

X_train = train[features]
y_train = train["Sales"]

X_test = test[features]
y_test = test["Sales"]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

actual = y_test.values

mae = np.mean(
    np.abs(
        actual - predictions
    )
)

rmse = np.sqrt(
    np.mean(
        (actual - predictions) ** 2
    )
)

non_zero_mask = actual != 0

if non_zero_mask.sum() > 0:
    mape = np.mean(
        np.abs(
            (
                actual[non_zero_mask]
                - predictions[non_zero_mask]
            )
            /
            actual[non_zero_mask]
        )
    ) * 100
else:
    mape = np.nan

st.header("Executive Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Transactions",
        f"{len(df):,}"
    )

with c2:
    st.metric(
        "Categories",
        f"{df['Category of Goods'].nunique():,}"
    )

with c3:
    st.metric(
        "Forecast MAPE",
        f"{mape:.2f}%"
    )

with c4:
    st.metric(
        "Model",
        "Random Forest"
    )

st.subheader("Business Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Total Sales",
        f"{df['Sales'].sum():,.0f}"
    )

with c2:
    st.metric(
        "Units Sold",
        f"{df['Quantity'].sum():,.0f}"
    )

with c3:
    st.metric(
        "Total Profit",
        f"{df['Profit'].sum():,.0f}"
    )

with c4:
    st.metric(
        "Test Year",
        str(latest_year)
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Forecast Performance",
        "📦 Inventory Planning",
        "🚨 Demand Spikes",
        "💡 Business Insights",
        "🔮 Next Month Forecast"
    ]
)

with tab1:
    st.header("Model Accuracy")

    st.caption(
        "The most recent year is held out as test data."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "MAE",
            f"{mae:,.0f}"
        )

    with c2:
        st.metric(
            "RMSE",
            f"{rmse:,.0f}"
        )

    with c3:
        st.metric(
            "MAPE",
            f"{mape:.2f}%"
        )

    st.subheader("Actual vs Predicted Sales")

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        actual,
        label="Actual Sales",
        linewidth=2
    )

    ax.plot(
        predictions,
        label="Predicted Sales",
        linewidth=2,
        linestyle="--"
    )

    ax.set_xlabel("Test Observation")
    ax.set_ylabel("Sales")
    ax.set_title("Actual vs Predicted Sales")
    ax.legend()
    ax.grid(
        axis="y",
        alpha=0.25
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.feature_importances_
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

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top Model Features")
    ax.grid(
        axis="x",
        alpha=0.25
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

with tab2:
    st.header("Inventory Planning")

    st.caption(
        "Inventory recommendation based on predicted demand "
        "with the project's 15% safety-stock buffer."
    )

    inventory = test[
        [
            "order_year",
            "order_month"
        ]
    ].copy()

    inventory["Predicted Demand"] = predictions

    inventory["Safety Stock"] = (
        inventory["Predicted Demand"] * 0.15
    )

    inventory["Recommended Inventory"] = (
        inventory["Predicted Demand"]
        +
        inventory["Safety Stock"]
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Predicted Demand",
            f"{inventory['Predicted Demand'].sum():,.0f}"
        )

    with c2:
        st.metric(
            "Safety Stock",
            f"{inventory['Safety Stock'].sum():,.0f}"
        )

    with c3:
        st.metric(
            "Recommended Inventory",
            f"{inventory['Recommended Inventory'].sum():,.0f}"
        )

    st.subheader("Inventory Plan")

    inventory_display = inventory.copy()

    inventory_display["Period"] = (
        inventory_display["order_year"]
        .astype(str)
        +
        "-"
        +
        inventory_display["order_month"]
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
        label="⬇ Download Inventory Plan",
        data=inventory_display.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="inventory_plan.csv",
        mime="text/csv"
    )

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

    if len(monthly) > 0:
        spike_rate = (
            spike_count
            /
            len(monthly)
            *
            100
        )
    else:
        spike_rate = 0

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Demand Spike Events",
            f"{spike_count:,}"
        )

    with c2:
        st.metric(
            "Spike Rate",
            f"{spike_rate:.1f}%"
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
        values.append(
            spike_counts.loc[0]
        )

    if 1 in spike_counts.index:
        labels.append("Spike")
        values.append(
            spike_counts.loc[1]
        )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        labels,
        values
    )

    ax.set_ylabel("Observations")
    ax.set_title("Demand Spike Distribution")
    ax.grid(
        axis="y",
        alpha=0.25
    )

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

    spike_display["sales_growth"] = (
        spike_display["sales_growth"]
        * 100
    )

    spike_display = spike_display.rename(
        columns={
            "order_year": "Year",
            "order_month": "Month",
            "Category of Goods": "Category",
            "Sales": "Sales",
            "sales_growth": "Growth %"
        }
    )

    st.dataframe(
        spike_display,
        use_container_width=True,
        hide_index=True
    )

with tab4:
    st.header("Business Insights")

    st.caption(
        "Insights generated from the historical retail data."
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

    top_category = (
        category_sales.index[0]
    )

    top_region = (
        region_sales.index[0]
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Best Month",
            str(best_month)
        )

    with c2:
        st.metric(
            "Weakest Month",
            str(weakest_month)
        )

    with c3:
        st.metric(
            "Top Category",
            top_category
        )

    with c4:
        st.metric(
            "Top Region",
            top_region
        )

    st.subheader("Sales by Category")

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        category_sales.index,
        category_sales.values
    )

    ax.set_xlabel("Category")
    ax.set_ylabel("Total Sales")
    ax.set_title("Total Sales by Category")

    ax.tick_params(
        axis="x",
        rotation=30
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.subheader("Regional Sales")

    regional_df = (
        region_sales
        .reset_index()
    )

    regional_df.columns = [
        "Region",
        "Sales"
    ]

    st.dataframe(
        regional_df,
        use_container_width=True,
        hide_index=True
    )

with tab5:
    st.header("Next Month Forecast")

    st.caption(
        "Category-level forecast generated using the trained "
        "Random Forest model."
    )

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

    future_base = (
        latest_category_rows
        .copy()
    )

    previous_month = (
        latest_category_rows["order_month"]
        .values
    )

    previous_year = (
        latest_category_rows["order_year"]
        .values
    )

    future_base["order_month"] = (
        previous_month % 12
    ) + 1

    future_base["order_year"] = np.where(
        previous_month == 12,
        previous_year + 1,
        previous_year
    )

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

    future_predictions = model.predict(
        future_X
    )

    future_base[
        "Predicted Next Month Sales"
    ] = future_predictions

    future_base[
        "Recommended Inventory"
    ] = (
        future_base[
            "Predicted Next Month Sales"
        ]
        * 1.15
    )

    total_forecast = (
        future_base[
            "Predicted Next Month Sales"
        ].sum()
    )

    total_inventory = (
        future_base[
            "Recommended Inventory"
        ].sum()
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Next Month Forecast",
            f"{total_forecast:,.0f}"
        )

    with c2:
        st.metric(
            "Recommended Inventory",
            f"{total_inventory:,.0f}"
        )

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
        label="⬇ Download Next Month Forecast",
        data=forecast_display.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="next_month_forecast.csv",
        mime="text/csv"
    )

    st.subheader(
        "Predicted Sales by Category"
    )

    chart_data = (
        forecast_display
        .sort_values(
            "Predicted Sales",
            ascending=False
        )
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        chart_data["Category"],
        chart_data["Predicted Sales"]
    )

    ax.set_xlabel("Category")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Next Month Sales Forecast")

    ax.tick_params(
        axis="x",
        rotation=30
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

st.divider()

st.caption(
    "Retail Demand Forecasting & Inventory Planning | "
    "Random Forest | 200 Estimators"
)
