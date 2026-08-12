import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor

# ---- Global chart styling ----
NAVY = "#0f172a"
BLUE = "#3b82f6"
LIGHT_BLUE = "#93c5fd"
GRAY = "#94a3b8"
GRID_COLOR = "#e5e7eb"
PALETTE = ["#3b82f6", "#0f172a", "#60a5fa", "#1e3a5f", "#93c5fd", "#64748b"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": "#374151",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.titlepad": 14,
    "axes.labelsize": 10.5,
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "legend.frameon": False,
})


def style_axes(ax, hide_left=False):
    """Apply a consistent, clean look to any chart axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)


st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f8fc;
        color: #1f2937;
    }

    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    [data-testid="stSidebar"] * {
        color: #1f2937 !important;
    }

    h1, h2, h3, h4 {
        color: #111827 !important;
    }

    p, span, label {
        color: #374151;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280 !important;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #ffffff;
        padding: 8px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }

    .stTabs [data-baseweb="tab"] {
        color: #374151 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #111827 !important;
        font-weight: 600;
    }

    [data-testid="stDataFrame"] {
        background-color: #ffffff;
        border-radius: 10px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        background-color: #ffffff;
        color: #111827;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #9ca3af;
        background-color: #f3f4f6;
    }

    hr {
        border-color: #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True
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
        "Explore forecast performance, inventory planning, "
        "demand spikes, business insights and the 12-month "
        "sales forecast."
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📈 Forecast Performance",
        "📦 Inventory Planning",
        "🚨 Demand Spikes",
        "💡 Business Insights",
        "🔮 12-Month Forecast",
        "✍️ Manual Prediction"
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
        linewidth=2.4,
        color=NAVY,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.6
    )

    ax.plot(
        predictions,
        label="Predicted Sales",
        linewidth=2.4,
        linestyle="--",
        color=BLUE,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.6
    )

    ax.fill_between(
        range(len(actual)),
        actual,
        predictions,
        color=BLUE,
        alpha=0.06
    )

    ax.set_xlabel("Test Observation")
    ax.set_ylabel("Sales")
    ax.set_title("Actual vs Predicted Sales")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1)
    style_axes(ax)

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

    n_bars = len(importance_df)
    bar_colors = [NAVY if i == 0 else BLUE for i in range(n_bars)]

    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color=bar_colors,
        height=0.65,
        zorder=3
    )

    for bar, value in zip(bars, importance_df["Importance"]):
        ax.text(
            value + importance_df["Importance"].max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            color="#374151"
        )

    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top Model Features")
    ax.set_xlim(0, importance_df["Importance"].max() * 1.15)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=1, zorder=0)
    style_axes(ax, hide_left=True)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

with tab2:
    st.header("Inventory Planning")

    st.caption(
        "Inventory recommendation based on predicted demand "
        "with a 15% safety-stock buffer."
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
        figsize=(7, 4)
    )

    bar_colors = ["#e2e8f0", BLUE]
    bars = ax.bar(
        labels,
        values,
        color=bar_colors[:len(labels)],
        width=0.45,
        zorder=3
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.02,
            f"{value:,}",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#374151"
        )

    ax.set_ylabel("Observations")
    ax.set_title("Demand Spike Distribution")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1, zorder=0)
    style_axes(ax)

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

    n_cat = len(category_sales)
    cat_colors = [NAVY if i == 0 else BLUE for i in range(n_cat)]

    bars = ax.bar(
        category_sales.index,
        category_sales.values,
        color=cat_colors,
        width=0.6,
        zorder=3
    )

    for bar, value in zip(bars, category_sales.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + category_sales.values.max() * 0.015,
            f"{value:,.0f}",
            ha="center",
            fontsize=8.5,
            color="#374151"
        )

    ax.set_xlabel("Category")
    ax.set_ylabel("Total Sales")
    ax.set_title("Total Sales by Category")
    ax.set_ylim(0, category_sales.values.max() * 1.15)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )

    ax.tick_params(
        axis="x",
        rotation=30
    )

    ax.grid(axis="y", color=GRID_COLOR, linewidth=1, zorder=0)
    style_axes(ax)

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
    st.header("12-Month Sales Forecast")

    st.caption(
        "Recursive category-level forecast for the next 12 months "
        "using the trained Random Forest model."
    )

    categories = (
        monthly["Category of Goods"]
        .dropna()
        .unique()
        .tolist()
    )

    category_histories = {}

    for category in categories:
        category_data = (
            monthly[
                monthly["Category of Goods"] == category
            ]
            .sort_values(
                [
                    "order_year",
                    "order_month"
                ]
            )
            .copy()
        )

        category_histories[category] = (
            category_data["Sales"]
            .tolist()
        )

    last_date = pd.Timestamp(
        year=latest_year,
        month=int(
            monthly[
                monthly["order_year"] == latest_year
            ]["order_month"].max()
        ),
        day=1
    )

    forecast_rows = []

    for forecast_step in range(1, 13):

        forecast_date = (
            last_date
            +
            pd.DateOffset(
                months=forecast_step
            )
        )

        for category in categories:

            history = category_histories[category]

            if len(history) < 3:
                continue

            lag_1 = history[-1]
            lag_2 = history[-2]
            lag_3 = history[-3]

            rolling_mean_3 = np.mean(
                history[-3:]
            )

            previous_growth = 0

            if len(history) >= 2 and history[-2] != 0:
                previous_growth = (
                    history[-1] - history[-2]
                ) / history[-2]

            row = {
                "order_year": forecast_date.year,
                "order_month": forecast_date.month,
                "sales_lag_1": lag_1,
                "sales_lag_2": lag_2,
                "sales_lag_3": lag_3,
                "rolling_mean_3": rolling_mean_3
            }

            category_columns = [
                column
                for column in features
                if column.startswith(
                    "Category of Goods_"
                )
            ]

            for column in category_columns:
                row[column] = 0

            category_feature = (
                "Category of Goods_"
                + str(category)
            )

            if category_feature in row:
                row[category_feature] = 1

            future_row = pd.DataFrame(
                [row]
            )

            for feature in features:
                if feature not in future_row.columns:
                    future_row[feature] = 0

            future_row = future_row[
                features
            ]

            prediction = model.predict(
                future_row
            )[0]

            prediction = max(
                0,
                prediction
            )

            safety_stock = (
                prediction * 0.15
            )

            recommended_inventory = (
                prediction
                +
                safety_stock
            )

            forecast_rows.append(
                {
                    "Month": forecast_date.strftime(
                        "%b %Y"
                    ),
                    "Year": forecast_date.year,
                    "Month Number": forecast_date.month,
                    "Category": category,
                    "Predicted Sales": prediction,
                    "Safety Stock": safety_stock,
                    "Recommended Inventory": recommended_inventory
                }
            )

            history.append(
                prediction
            )

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    if forecast_df.empty:
        st.warning(
            "Unable to generate the 12-month forecast. "
            "Make sure each category has enough historical data."
        )
        st.stop()

    monthly_forecast = (
        forecast_df
        .groupby(
            [
                "Year",
                "Month Number",
                "Month"
            ],
            as_index=False
        )
        .agg(
            {
                "Predicted Sales": "sum",
                "Safety Stock": "sum",
                "Recommended Inventory": "sum"
            }
        )
        .sort_values(
            [
                "Year",
                "Month Number"
            ]
        )
    )

    total_12_month_sales = (
        monthly_forecast[
            "Predicted Sales"
        ].sum()
    )

    total_12_month_inventory = (
        monthly_forecast[
            "Recommended Inventory"
        ].sum()
    )

    average_monthly_sales = (
        monthly_forecast[
            "Predicted Sales"
        ].mean()
    )

    peak_month_index = (
        monthly_forecast[
            "Predicted Sales"
        ].idxmax()
    )

    peak_month = (
        monthly_forecast
        .loc[
            peak_month_index,
            "Month"
        ]
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "12-Month Forecast",
            f"{total_12_month_sales:,.0f}"
        )

    with c2:
        st.metric(
            "Average Monthly Sales",
            f"{average_monthly_sales:,.0f}"
        )

    with c3:
        st.metric(
            "12-Month Inventory",
            f"{total_12_month_inventory:,.0f}"
        )

    with c4:
        st.metric(
            "Peak Forecast Month",
            peak_month
        )

    st.subheader("Monthly Forecast by Category")

    st.caption(
        "Each row is one category's predicted sales for one month — "
        "this is the level you'd actually use for stocking decisions."
    )

    category_display = forecast_df.sort_values(
        ["Year", "Month Number", "Category"]
    )[
        [
            "Month",
            "Category",
            "Predicted Sales",
            "Safety Stock",
            "Recommended Inventory"
        ]
    ]

    category_filter = st.multiselect(
        "Filter by category",
        options=sorted(forecast_df["Category"].unique().tolist()),
        default=[]
    )

    if category_filter:
        filtered_display = category_display[
            category_display["Category"].isin(category_filter)
        ]
    else:
        filtered_display = category_display

    st.dataframe(
        filtered_display,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="⬇ Download 12-Month Forecast (By Category)",
        data=category_display.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="12_month_forecast_by_category.csv",
        mime="text/csv"
    )

    st.subheader("Monthly Forecast (All Categories Combined)")

    st.caption(
        "Summed across all categories — useful for a total "
        "company-wide view, not for individual product stocking."
    )

    monthly_display = monthly_forecast[
        [
            "Month",
            "Predicted Sales",
            "Safety Stock",
            "Recommended Inventory"
        ]
    ].copy()

    st.dataframe(
        monthly_display,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="⬇ Download 12-Month Forecast (Combined Total)",
        data=monthly_display.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="12_month_forecast_combined.csv",
        mime="text/csv"
    )

    st.subheader("12-Month Sales Trend")

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        monthly_forecast["Month"],
        monthly_forecast["Predicted Sales"],
        marker="o",
        linewidth=2.4,
        color=BLUE,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.8,
        zorder=3
    )

    ax.fill_between(
        range(len(monthly_forecast)),
        monthly_forecast["Predicted Sales"],
        monthly_forecast["Predicted Sales"].min() * 0.97,
        color=BLUE,
        alpha=0.08,
        zorder=1
    )

    ax.set_xlabel("Forecast Month")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Next 12 Months Sales Forecast (All Categories Combined)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )

    ax.tick_params(
        axis="x",
        rotation=35
    )

    ax.grid(axis="y", color=GRID_COLOR, linewidth=1, zorder=0)
    style_axes(ax)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.subheader("Category-Level 12-Month Sales Trend")

    category_pivot = (
        forecast_df
        .pivot_table(
            index="Month",
            columns="Category",
            values="Predicted Sales",
            aggfunc="sum"
        )
    )

    category_pivot = category_pivot.reindex(
        forecast_df.sort_values(["Year", "Month Number"])["Month"].unique()
    )

    fig, ax = plt.subplots(
        figsize=(12, 5.5)
    )

    for i, category in enumerate(category_pivot.columns):
        ax.plot(
            category_pivot.index,
            category_pivot[category],
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=PALETTE[i % len(PALETTE)],
            label=category
        )

    ax.set_xlabel("Forecast Month")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Next 12 Months Sales Forecast by Category")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=9,
        title="Category",
        title_fontsize=9.5
    )
    ax.tick_params(
        axis="x",
        rotation=35
    )
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1, zorder=0)
    style_axes(ax)

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.subheader("Category-Level 12-Month Forecast (Pivot Table)")

    st.dataframe(
        category_pivot.reset_index(),
        use_container_width=True,
        hide_index=True
    )

with tab6:
    st.header("Manual Prediction")

    st.caption(
        "Manually enter recent sales figures for a category to get "
        "an on-demand prediction from the trained model, without "
        "needing the full dataset context."
    )

    category_columns_manual = [
        column
        for column in features
        if column.startswith("Category of Goods_")
    ]

    manual_category_options = ["Dairy Products"] + [
        column.replace("Category of Goods_", "")
        for column in category_columns_manual
    ]

    with st.form("manual_prediction_form"):

        st.subheader("Enter Recent Sales History")

        col1, col2 = st.columns(2)

        with col1:
            manual_category = st.selectbox(
                "Product Category",
                options=sorted(manual_category_options)
            )

            manual_month = st.selectbox(
                "Target Month (the month you are predicting for)",
                options=list(range(1, 13)),
                format_func=lambda m: pd.Timestamp(
                    year=2000, month=m, day=1
                ).strftime("%B")
            )

        with col2:
            manual_year = st.number_input(
                "Target Year",
                min_value=2000,
                max_value=2100,
                value=int(latest_year) + 1,
                step=1
            )

        st.markdown("**Last 3 Known Months of Sales** (most recent first)")

        m1, m2, m3 = st.columns(3)

        with m1:
            manual_lag_1 = st.number_input(
                "Last Month's Sales",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                format="%.2f"
            )

        with m2:
            manual_lag_2 = st.number_input(
                "2 Months Ago Sales",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                format="%.2f"
            )

        with m3:
            manual_lag_3 = st.number_input(
                "3 Months Ago Sales",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                format="%.2f"
            )

        submitted = st.form_submit_button("Predict Sales")

    if submitted:

        rolling_mean_manual = np.mean(
            [manual_lag_1, manual_lag_2, manual_lag_3]
        )

        manual_row = {
            "order_year": int(manual_year),
            "order_month": int(manual_month),
            "sales_lag_1": manual_lag_1,
            "sales_lag_2": manual_lag_2,
            "sales_lag_3": manual_lag_3,
            "rolling_mean_3": rolling_mean_manual
        }

        for column in category_columns_manual:
            manual_row[column] = 0

        manual_category_feature = (
            "Category of Goods_" + str(manual_category)
        )

        if manual_category_feature in manual_row:
            manual_row[manual_category_feature] = 1

        manual_input_df = pd.DataFrame([manual_row])

        for feature in features:
            if feature not in manual_input_df.columns:
                manual_input_df[feature] = 0

        manual_input_df = manual_input_df[features]

        manual_prediction = model.predict(manual_input_df)[0]
        manual_prediction = max(0, manual_prediction)

        manual_safety_stock = manual_prediction * 0.15
        manual_recommended_inventory = (
            manual_prediction + manual_safety_stock
        )

        st.success("Prediction generated successfully")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Predicted Sales",
                f"{manual_prediction:,.0f}"
            )

        with r2:
            st.metric(
                "Safety Stock (15%)",
                f"{manual_safety_stock:,.0f}"
            )

        with r3:
            st.metric(
                "Recommended Inventory",
                f"{manual_recommended_inventory:,.0f}"
            )

        st.caption(
            f"Prediction for **{manual_category}** — "
            f"{pd.Timestamp(year=2000, month=int(manual_month), day=1).strftime('%B')} "
            f"{int(manual_year)}, based on a 3-month rolling average of "
            f"{rolling_mean_manual:,.0f}."
        )

st.divider()

st.caption(
    "Retail Demand Forecasting & Inventory Planning | "
    "Random Forest | 200 Estimators"
)
