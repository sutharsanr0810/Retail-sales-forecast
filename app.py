import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Retail Inventory Intelligence",
    page_icon="📦",
    layout="wide"
)


# =========================================================
# BLACK UI
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #000000 !important;
    color: #ffffff !important;
}

.stApp {
    background: #000000;
    color: #ffffff;
}

section[data-testid="stSidebar"] {
    background: #050505 !important;
    border-right: 1px solid #333333;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

p, label, span, div {
    color: #ffffff;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
}

.subtitle {
    color: #aaaaaa !important;
    font-size: 16px;
}

.card {
    background: #080808;
    border: 1px solid #333333;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 10px;
}

.card-title {
    color: #bbbbbb;
    font-size: 14px;
}

.card-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: 800;
}

.section-title {
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
    margin-top: 20px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #080808 !important;
    color: white !important;
    border-color: #444444 !important;
}

input {
    background-color: #080808 !important;
    color: white !important;
}

textarea {
    background-color: #080808 !important;
    color: white !important;
}

button {
    border-radius: 7px !important;
}

.stButton > button {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 700;
}

.stDownloadButton > button {
    background: #ffffff !important;
    color: #000000 !important;
}

[data-testid="stDataFrame"] {
    background-color: #050505 !important;
}

hr {
    border-color: #333333;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "sales_df" not in st.session_state:
    st.session_state.sales_df = None

if "inventory_df" not in st.session_state:
    st.session_state.inventory_df = pd.DataFrame()

if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()

if "model" not in st.session_state:
    st.session_state.model = None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_column(df, names):
    """Find a column using several possible names."""
    lower = {str(c).lower().strip(): c for c in df.columns}

    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]

    for c in df.columns:
        cl = str(c).lower().replace("_", " ").strip()

        for name in names:
            if name.lower() in cl:
                return c

    return None


def prepare_sales_data(df):

    df = df.copy()

    date_col = find_column(
        df,
        ["date", "order date", "sales date", "transaction date"]
    )

    product_col = find_column(
        df,
        ["product id", "product_id", "product"]
    )

    product_name_col = find_column(
        df,
        ["product name", "product_name", "name"]
    )

    category_col = find_column(
        df,
        ["category", "product category"]
    )

    sales_col = find_column(
        df,
        ["sales", "units sold", "quantity", "qty", "demand"]
    )

    if date_col is None:
        raise ValueError(
            "CSV must contain a Date column."
        )

    if product_col is None:
        raise ValueError(
            "CSV must contain a Product ID column."
        )

    if sales_col is None:
        raise ValueError(
            "CSV must contain Sales / Units Sold / Quantity column."
        )

    df["__date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df["__product"] = df[product_col].astype(str)

    df["__sales"] = pd.to_numeric(
        df[sales_col],
        errors="coerce"
    ).fillna(0)

    if product_name_col:
        df["__product_name"] = (
            df[product_name_col].astype(str)
        )
    else:
        df["__product_name"] = df["__product"]

    if category_col:
        df["__category"] = (
            df[category_col].astype(str)
        )
    else:
        df["__category"] = "Other"

    df = df.dropna(subset=["__date"])

    return df


def train_forecast_model(df, trees=200):

    daily = (
        df.groupby("__date")["__sales"]
        .sum()
        .reset_index()
    )

    daily = daily.sort_values("__date")

    if len(daily) < 15:
        return None, pd.DataFrame(), None

    daily["lag1"] = daily["__sales"].shift(1)
    daily["lag7"] = daily["__sales"].shift(7)
    daily["rolling7"] = (
        daily["__sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    daily["day"] = daily["__date"].dt.day
    daily["month"] = daily["__date"].dt.month
    daily["weekday"] = daily["__date"].dt.weekday

    daily = daily.dropna()

    if len(daily) < 10:
        return None, pd.DataFrame(), None

    features = [
        "lag1",
        "lag7",
        "rolling7",
        "day",
        "month",
        "weekday"
    ]

    X = daily[features]
    y = daily["__sales"]

    split = int(len(daily) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=trees,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, pred)
    )

    r2 = r2_score(y_test, pred)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    test_df = pd.DataFrame({
        "Date": daily["__date"].iloc[split:],
        "Actual": y_test.values,
        "Predicted": pred
    })

    return model, test_df, metrics


def calculate_product_forecast(df):

    result = []

    for product in df["__product"].unique():

        p = df[df["__product"] == product].copy()

        total_sales = p["__sales"].sum()

        days = max(
            (p["__date"].max() - p["__date"].min()).days + 1,
            1
        )

        daily_demand = total_sales / days

        forecast_30 = daily_demand * 30

        row = p.iloc[-1]

        result.append({
            "Product ID": product,
            "Product Name": row["__product_name"],
            "Category": row["__category"],
            "Forecast Demand (30 Days)": round(
                forecast_30
            )
        })

    return pd.DataFrame(result)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="font-size:24px;font-weight:800;">
    📦 Retail Inventory<br>Intelligence
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "Demand Forecasting • Inventory Management • Reorder Planning"
    )

    st.divider()

    st.markdown("### DATA & SETTINGS")

    uploaded_file = st.file_uploader(
        "Upload Sales CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            raw = pd.read_csv(uploaded_file)

            sales = prepare_sales_data(raw)

            st.session_state.sales_df = sales

            st.success(
                f"CSV loaded: {len(raw):,} records"
            )

        except Exception as e:

            st.error(str(e))

    st.divider()

    st.markdown("### MODEL SETTINGS")

    trees = st.selectbox(
        "Number of Trees",
        [50, 100, 200, 300, 500],
        index=2
    )

    safety_stock = st.number_input(
        "Safety Stock (%)",
        min_value=0,
        max_value=100,
        value=15
    )

    if st.button(
        "▶ Train / Refresh Model",
        width="stretch"
    ):

        if st.session_state.sales_df is None:

            st.warning(
                "Upload the CSV first."
            )

        else:

            try:

                model, test_df, metrics = train_forecast_model(
                    st.session_state.sales_df,
                    trees
                )

                st.session_state.model = model
                st.session_state.test_df = test_df
                st.session_state.metrics = metrics

                st.session_state.forecast_df = (
                    calculate_product_forecast(
                        st.session_state.sales_df
                    )
                )

                st.success(
                    "Model trained successfully."
                )

            except Exception as e:

                st.error(
                    f"Model error: {e}"
                )

    st.divider()

    st.markdown("### DATA SUMMARY")

    if st.session_state.sales_df is not None:

        d = st.session_state.sales_df

        st.write(
            f"**Records:** {len(d):,}"
        )

        st.write(
            f"**Products:** {d['__product'].nunique()}"
        )

        st.write(
            f"**Categories:** {d['__category'].nunique()}"
        )

        st.write(
            f"**Date Range:** "
            f"{d['__date'].min().date()} → "
            f"{d['__date'].max().date()}"
        )

    else:

        st.caption(
            "Upload a sales CSV to begin."
        )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📦 Retail Inventory Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Demand Forecasting • Manual Inventory Management • Reorder Planning'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# =========================================================
# NAVIGATION
# =========================================================

tabs = st.tabs([
    "Dashboard",
    "Manual Inventory",
    "Inventory Planning",
    "Forecast Model",
    "Alerts",
    "Analytics",
    "12-Month Forecast"
])


# =========================================================
# DASHBOARD
# =========================================================

with tabs[0]:

    st.markdown(
        '<div class="section-title">Dashboard</div>',
        unsafe_allow_html=True
    )

    if st.session_state.sales_df is None:

        st.info(
            "Upload the sales CSV from the sidebar to display the dashboard."
        )

    else:

        df = st.session_state.sales_df

        products = df["__product"].nunique()

        stock = 0

        if not st.session_state.inventory_df.empty:
            stock = st.session_state.inventory_df[
                "Current Stock"
            ].sum()

        reorder_count = 0

        if (
            not st.session_state.inventory_df.empty
            and not st.session_state.forecast_df.empty
        ):

            temp = st.session_state.inventory_df.merge(
                st.session_state.forecast_df[
                    [
                        "Product ID",
                        "Forecast Demand (30 Days)"
                    ]
                ],
                on="Product ID",
                how="left"
            )

            temp["Forecast Demand (30 Days)"] = (
                temp["Forecast Demand (30 Days)"]
                .fillna(0)
            )

            temp["Safety Stock"] = (
                temp["Forecast Demand (30 Days)"]
                * safety_stock / 100
            )

            temp["Reorder Level"] = (
                temp["Forecast Demand (30 Days)"] / 30 * 7
                + temp["Safety Stock"]
            )

            reorder_count = (
                temp["Current Stock"]
                < temp["Reorder Level"]
            ).sum()

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">Total Products</div>
                <div class="card-value">{products}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">Manual Current Stock</div>
                <div class="card-value">{stock:,.0f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">Reorder Products</div>
                <div class="card-value">{reorder_count}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:

            mae = 0

            if "metrics" in st.session_state:
                mae = st.session_state.metrics["MAE"]

            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">MAE</div>
                <div class="card-value">{mae:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c5:

            rmse = 0

            if "metrics" in st.session_state:
                rmse = st.session_state.metrics["RMSE"]

            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">RMSE</div>
                <div class="card-value">{rmse:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        if "test_df" in st.session_state:

            chart = st.session_state.test_df

            fig, ax = plt.subplots()

            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")

            ax.plot(
                chart["Date"],
                chart["Actual"],
                label="Actual Demand"
            )

            ax.plot(
                chart["Date"],
                chart["Predicted"],
                label="Predicted Demand"
            )

            ax.tick_params(colors="white")

            for spine in ax.spines.values():
                spine.set_color("#444444")

            ax.set_title(
                "Actual vs Predicted Demand",
                color="white"
            )

            ax.set_ylabel(
                "Units Sold",
                color="white"
            )

            ax.legend()

            st.pyplot(
                fig,
                width="stretch"
            )

        else:

            st.info(
                "Click Train / Refresh Model to generate demand predictions."
            )


# =========================================================
# MANUAL INVENTORY
# =========================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">✏️ Manual Inventory</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This section is completely separate from the CSV. "
        "Enter your CURRENT inventory here. The CSV is used only for demand forecasting."
    )

    if st.session_state.sales_df is None:

        st.warning(
            "Upload the sales CSV first so products can be loaded into the inventory list."
        )

    else:

        sales = st.session_state.sales_df

        products_df = (
            sales[
                [
                    "__product",
                    "__product_name",
                    "__category"
                ]
            ]
            .drop_duplicates("__product")
            .rename(
                columns={
                    "__product": "Product ID",
                    "__product_name": "Product Name",
                    "__category": "Category"
                }
            )
        )

        # Create initial inventory table
        if st.session_state.inventory_df.empty:

            inv = products_df.copy()

            inv["Current Stock"] = 0
            inv["Unit Price"] = 0.0
            inv["Reorder Level"] = 0
            inv["Lead Time (Days)"] = 7

            st.session_state.inventory_df = inv

        st.markdown("### Enter / Update Current Inventory")

        edited = st.data_editor(
            st.session_state.inventory_df,
            width="stretch",
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "Product ID": st.column_config.TextColumn(
                    disabled=True
                ),
                "Product Name": st.column_config.TextColumn(
                    disabled=True
                ),
                "Category": st.column_config.TextColumn(
                    disabled=True
                ),
                "Current Stock": st.column_config.NumberColumn(
                    min_value=0,
                    step=1
                ),
                "Unit Price": st.column_config.NumberColumn(
                    min_value=0,
                    step=1
                ),
                "Reorder Level": st.column_config.NumberColumn(
                    min_value=0,
                    step=1
                ),
                "Lead Time (Days)": st.column_config.NumberColumn(
                    min_value=0,
                    step=1
                )
            },
            key="inventory_editor"
        )

        if st.button(
            "💾 Save Inventory",
            width="stretch"
        ):

            st.session_state.inventory_df = edited.copy()

            st.success(
                "Manual inventory saved successfully."
            )

            st.rerun()

        st.download_button(
            "⬇ Download Manual Inventory",
            data=st.session_state.inventory_df.to_csv(
                index=False
            ),
            file_name="manual_inventory.csv",
            mime="text/csv",
            width="stretch"
        )


# =========================================================
# INVENTORY PLANNING
# =========================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">📦 Inventory Planning</div>',
        unsafe_allow_html=True
    )

    if st.session_state.inventory_df.empty:

        st.info(
            "Enter your inventory in the Manual Inventory tab first."
        )

    elif st.session_state.forecast_df.empty:

        st.warning(
            "Train the model first to generate demand forecasts."
        )

    else:

        inv = st.session_state.inventory_df.copy()

        forecast = st.session_state.forecast_df.copy()

        plan = inv.merge(
            forecast,
            on="Product ID",
            how="left"
        )

        plan[
            "Forecast Demand (30 Days)"
        ] = plan[
            "Forecast Demand (30 Days)"
        ].fillna(0)

        plan["Daily Demand"] = (
            plan["Forecast Demand (30 Days)"] / 30
        )

        plan["Safety Stock"] = (
            plan["Forecast Demand (30 Days)"]
            * safety_stock / 100
        )

        plan["Calculated Reorder Level"] = (
            plan["Daily Demand"]
            * plan["Lead Time (Days)"]
            + plan["Safety Stock"]
        )

        plan["Target Stock"] = (
            plan["Forecast Demand (30 Days)"]
            + plan["Safety Stock"]
        )

        plan["Order Qty"] = (
            plan["Target Stock"]
            - plan["Current Stock"]
        ).clip(lower=0)

        def get_status(row):

            stock = row["Current Stock"]
            forecast_demand = row["Forecast Demand (30 Days)"]
            target = row["Target Stock"]
            reorder = row["Calculated Reorder Level"]

            if stock <= 0:
                return "STOCKOUT"

            if stock < reorder:
                return "REORDER"

            if stock > target * 1.5:
                return "OVERSTOCK"

            return "HEALTHY"

        plan["Status"] = plan.apply(
            get_status,
            axis=1
        )

        plan["Inventory Value"] = (
            plan["Current Stock"]
            * plan["Unit Price"]
        )

        display_cols = [
            "Product ID",
            "Product Name",
            "Category",
            "Current Stock",
            "Forecast Demand (30 Days)",
            "Safety Stock",
            "Calculated Reorder Level",
            "Target Stock",
            "Order Qty",
            "Status",
            "Inventory Value"
        ]

        st.dataframe(
            plan[display_cols],
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "⬇ Download Inventory Plan",
            data=plan[display_cols].to_csv(
                index=False
            ),
            file_name="inventory_plan.csv",
            mime="text/csv",
            width="stretch"
        )


# =========================================================
# FORECAST MODEL
# =========================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">🔮 Forecast Model</div>',
        unsafe_allow_html=True
    )

    if "metrics" not in st.session_state:

        st.info(
            "Train the model from the sidebar."
        )

    else:

        metrics = st.session_state.metrics

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "MAE",
                f"{metrics['MAE']:.2f}"
            )

        with b:
            st.metric(
                "RMSE",
                f"{metrics['RMSE']:.2f}"
            )

        with c:
            st.metric(
                "R²",
                f"{metrics['R2']:.3f}"
            )

        st.dataframe(
            st.session_state.test_df,
            width="stretch",
            hide_index=True
        )


# =========================================================
# ALERTS
# =========================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">🚨 Inventory Alerts</div>',
        unsafe_allow_html=True
    )

    if st.session_state.inventory_df.empty:

        st.info(
            "No manual inventory entered."
        )

    elif st.session_state.forecast_df.empty:

        st.info(
            "Train the forecasting model first."
        )

    else:

        inv = st.session_state.inventory_df.copy()

        fc = st.session_state.forecast_df.copy()

        alerts = inv.merge(
            fc,
            on="Product ID",
            how="left"
        )

        alerts[
            "Forecast Demand (30 Days)"
        ] = alerts[
            "Forecast Demand (30 Days)"
        ].fillna(0)

        alerts["Safety Stock"] = (
            alerts["Forecast Demand (30 Days)"]
            * safety_stock / 100
        )

        alerts["Reorder Level"] = (
            alerts["Forecast Demand (30 Days)"] / 30
            * alerts["Lead Time (Days)"]
            + alerts["Safety Stock"]
        )

        alerts["Status"] = np.where(
            alerts["Current Stock"] <= 0,
            "STOCKOUT",
            np.where(
                alerts["Current Stock"]
                < alerts["Reorder Level"],
                "REORDER",
                np.where(
                    alerts["Current Stock"]
                    > alerts["Forecast Demand (30 Days)"]
                    * 1.5,
                    "OVERSTOCK",
                    "HEALTHY"
                )
            )
        )

        alert_df = alerts[
            alerts["Status"] != "HEALTHY"
        ]

        if alert_df.empty:

            st.success(
                "No inventory alerts."
            )

        else:

            st.dataframe(
                alert_df[
                    [
                        "Product ID",
                        "Product Name",
                        "Current Stock",
                        "Forecast Demand (30 Days)",
                        "Reorder Level",
                        "Status"
                    ]
                ],
                width="stretch",
                hide_index=True
            )


# =========================================================
# ANALYTICS
# =========================================================

with tabs[5]:

    st.markdown(
        '<div class="section-title">📊 Analytics</div>',
        unsafe_allow_html=True
    )

    if st.session_state.sales_df is None:

        st.info(
            "Upload the CSV first."
        )

    else:

        df = st.session_state.sales_df

        category_sales = (
            df.groupby("__category")["__sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.markdown("### Demand by Category")

        fig, ax = plt.subplots()

        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        category_sales.plot(
            kind="bar",
            ax=ax
        )

        ax.tick_params(
            axis="x",
            colors="white",
            rotation=45
        )

        ax.tick_params(
            axis="y",
            colors="white"
        )

        ax.set_ylabel(
            "Units Sold",
            color="white"
        )

        ax.set_xlabel(
            "",
            color="white"
        )

        ax.set_title(
            "Demand by Category",
            color="white"
        )

        st.pyplot(
            fig,
            width="stretch"
        )


# =========================================================
# 12 MONTH FORECAST
# =========================================================

with tabs[6]:

    st.markdown(
        '<div class="section-title">📅 12-Month Forecast</div>',
        unsafe_allow_html=True
    )

    if st.session_state.forecast_df.empty:

        st.info(
            "Train the model first."
        )

    else:

        base = st.session_state.forecast_df.copy()

        monthly = []

        for month in range(1, 13):

            temp = base.copy()

            temp["Month"] = month

            temp["Forecast"] = (
                temp["Forecast Demand (30 Days)"]
                * (1 + 0.01 * (month - 1))
            )

            monthly.append(temp)

        twelve = pd.concat(
            monthly,
            ignore_index=True
        )

        st.dataframe(
            twelve[
                [
                    "Month",
                    "Product ID",
                    "Product Name",
                    "Category",
                    "Forecast"
                ]
            ],
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "⬇ Download 12-Month Forecast",
            data=twelve.to_csv(
                index=False
            ),
            file_name="12_month_forecast.csv",
            mime="text/csv",
            width="stretch"
        )
