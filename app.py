import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


# =========================================================
# PAGE CONFIG
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
    font-family: Arial, sans-serif;
}

.stApp {
    background-color: #000000;
    color: #ffffff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #000000;
    border-right: 1px solid #333333;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Main headings */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Normal text */
p, label, span {
    color: #ffffff;
}

/* Cards */
.card {
    background: #050505;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 18px;
    min-height: 120px;
}

.card-title {
    color: #ffffff;
    font-size: 15px;
    font-weight: bold;
}

.card-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: bold;
    margin-top: 10px;
}

.card-sub {
    color: #aaaaaa;
    font-size: 13px;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #333333;
}

/* =====================================================
   ONLY FILE UPLOAD BUTTON
   ===================================================== */

[data-testid="stFileUploader"] button {
    background: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #444444 !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #ffffff !important;
}

[data-testid="stFileUploader"] button svg {
    stroke: #ffffff !important;
    fill: none !important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: #080808 !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}

/* Select boxes */
div[data-baseweb="select"] > div {
    background-color: #050505 !important;
    border-color: #444444 !important;
    color: white !important;
}

/* Number input */
input {
    background-color: #050505 !important;
    color: white !important;
}

/* Buttons except uploader */
.stButton > button {
    background-color: #ffffff;
    color: #000000;
    border: none;
    border-radius: 6px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #dddddd;
    color: #000000;
}

/* Status colors */
.status-red {
    color: #ff3333;
}

.status-yellow {
    color: #ffd000;
}

.status-blue {
    color: #2196ff;
}

.status-green {
    color: #55cc55;
}

/* Divider */
hr {
    border-color: #333333;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "inventory" not in st.session_state:
    st.session_state.inventory = pd.DataFrame()

if "sales" not in st.session_state:
    st.session_state.sales = pd.DataFrame()

if "model" not in st.session_state:
    st.session_state.model = None

if "trained" not in st.session_state:
    st.session_state.trained = False


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <h2>📦 Retail Inventory<br>Intelligence</h2>
    <p style="font-size:13px;">
    Demand Forecasting • Inventory<br>
    Management • Reorder Planning
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### DATA & SETTINGS")

    uploaded_file = st.file_uploader(
        "Upload Sales/Inventory CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            st.session_state.sales = df.copy()

            st.success("CSV loaded successfully")

        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.markdown("### MODEL SETTINGS")

    model_name = st.selectbox(
        "Model",
        ["Random Forest"]
    )

    trees = st.selectbox(
        "Number of Trees",
        [100, 200, 300, 500],
        index=1
    )

    safety_stock = st.number_input(
        "Safety Stock (%)",
        min_value=0,
        max_value=100,
        value=15
    )

    train_button = st.button(
        "▶ Train / Refresh Model",
        width="stretch"
    )

    st.markdown("### DATA SUMMARY")

    if not st.session_state.sales.empty:

        df = st.session_state.sales

        st.write("Total Records:", len(df))

        if "Date" in df.columns:
            try:
                dates = pd.to_datetime(df["Date"])
                st.write(
                    "Date Range:",
                    f"{dates.min().date()} - {dates.max().date()}"
                )
            except:
                pass

        if "Store ID" in df.columns:
            st.write("Stores:", df["Store ID"].nunique())

        if "Product ID" in df.columns:
            st.write("Products:", df["Product ID"].nunique())

        if "Category" in df.columns:
            st.write("Categories:", df["Category"].nunique())


# =========================================================
# MAIN TITLE
# =========================================================

st.markdown("""
<h1>📦 Retail Inventory Intelligence</h1>

<p style="color:#bbbbbb;">
Demand Forecasting • Manual Inventory Management • Reorder Planning
</p>
""", unsafe_allow_html=True)


# =========================================================
# TABS
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
# DATA PREPARATION
# =========================================================

df = st.session_state.sales.copy()

if not df.empty:

    # Detect common sales column names
    possible_sales = [
        "Sales",
        "Units Sold",
        "Quantity",
        "Units_Sold",
        "sales",
        "quantity"
    ]

    sales_col = None

    for col in possible_sales:
        if col in df.columns:
            sales_col = col
            break

    if sales_col is None:
        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_cols:
            sales_col = numeric_cols[-1]

    if sales_col:

        df["Demand"] = pd.to_numeric(
            df[sales_col],
            errors="coerce"
        ).fillna(0)

        # Product
        if "Product ID" not in df.columns:
            df["Product ID"] = "P001"

        # Product name
        if "Product Name" not in df.columns:
            df["Product Name"] = df["Product ID"]

        # Category
        if "Category" not in df.columns:
            df["Category"] = "General"

        # Store
        if "Store ID" not in df.columns:
            df["Store ID"] = "S001"


# =========================================================
# DASHBOARD
# =========================================================

with tabs[0]:

    if df.empty:

        st.info("Upload the retail CSV to start.")

    else:

        products = df["Product ID"].nunique()

        total_demand = int(df["Demand"].sum())

        categories = df["Category"].nunique()

        current_stock = 0

        if not st.session_state.inventory.empty:
            current_stock = int(
                st.session_state.inventory["Current Stock"].sum()
            )

        avg_demand = df["Demand"].mean()

        reorder_level = int(
            avg_demand * 30 * (1 + safety_stock / 100)
        )

        # -------------------------------------------------
        # KPI CARDS
        # -------------------------------------------------

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📦 Total Products</div>
                <div class="card-value">{products}</div>
                <div class="card-sub">Active Products</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">📦 Current Stock</div>
                <div class="card-value">{current_stock:,}</div>
                <div class="card-sub">Units</div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🛒 Reorder Level</div>
                <div class="card-value">{reorder_level:,}</div>
                <div class="card-sub">Based on settings</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:

            mae = 0

            if st.session_state.trained:
                mae = st.session_state.get(
                    "mae",
                    0
                )

            st.markdown(f"""
            <div class="card">
                <div class="card-title">〽 MAE</div>
                <div class="card-value">{mae:.2f}</div>
                <div class="card-sub">Lower is better</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:

            mape = 0

            if st.session_state.trained:
                mape = st.session_state.get(
                    "mape",
                    0
                )

            st.markdown(f"""
            <div class="card">
                <div class="card-title">🎯 MAPE</div>
                <div class="card-value">{mape:.2f}%</div>
                <div class="card-sub">Lower is better</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------
        # STATUS CARDS
        # -------------------------------------------------

        stock_products = 0
        reorder_products = 0
        overstock_products = 0
        healthy_products = 0

        if not st.session_state.inventory.empty:

            inv = st.session_state.inventory

            for _, row in inv.iterrows():

                stock = row["Current Stock"]
                reorder = row["Reorder Level"]
                target = row["Target Stock"]

                if stock <= 0:
                    stock_products += 1

                elif stock < reorder:
                    reorder_products += 1

                elif stock > target:
                    overstock_products += 1

                else:
                    healthy_products += 1

        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.markdown(f"""
            <div class="card">
                <div class="status-red">🔴 Stockout</div>
                <div class="card-value">{stock_products}</div>
                <div class="card-sub">Products</div>
            </div>
            """, unsafe_allow_html=True)

        with s2:
            st.markdown(f"""
            <div class="card">
                <div class="status-yellow">🟡 Reorder</div>
                <div class="card-value">{reorder_products}</div>
                <div class="card-sub">Products</div>
            </div>
            """, unsafe_allow_html=True)

        with s3:
            st.markdown(f"""
            <div class="card">
                <div class="status-blue">🔵 Overstock</div>
                <div class="card-value">{overstock_products}</div>
                <div class="card-sub">Products</div>
            </div>
            """, unsafe_allow_html=True)

        with s4:
            st.markdown(f"""
            <div class="card">
                <div class="status-green">🟢 Healthy</div>
                <div class="card-value">{healthy_products}</div>
                <div class="card-sub">Products</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------
        # CHARTS
        # -------------------------------------------------

        col1, col2, col3 = st.columns([1.5, 1, 1])

        with col1:

            st.subheader("Actual vs Predicted Demand")

            daily = (
                df.groupby("Product ID")["Demand"]
                .sum()
                .reset_index()
            )

            if len(daily) > 1:

                actual = daily["Demand"].values

                predicted = pd.Series(actual).rolling(
                    3,
                    min_periods=1
                ).mean().values

                fig, ax = plt.subplots()

                fig.patch.set_facecolor("#000000")
                ax.set_facecolor("#000000")

                ax.plot(
                    actual,
                    label="Actual Demand"
                )

                ax.plot(
                    predicted,
                    label="Predicted Demand"
                )

                ax.set_xlabel("Products")
                ax.set_ylabel("Units Sold")

                ax.tick_params(colors="white")

                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")

                for spine in ax.spines.values():
                    spine.set_color("#333333")

                ax.legend()

                st.pyplot(
                    fig,
                    width="stretch"
                )

        with col2:

            st.subheader("Demand by Category")

            category_data = (
                df.groupby("Category")["Demand"]
                .sum()
                .sort_values(ascending=False)
            )

            fig, ax = plt.subplots()

            fig.patch.set_facecolor("#000000")
            ax.set_facecolor("#000000")

            ax.pie(
                category_data.values,
                labels=category_data.index,
                autopct="%1.1f%%"
            )

            st.pyplot(
                fig,
                width="stretch"
            )

        with col3:

            st.subheader("Demand by Category")

            st.bar_chart(
                category_data,
                horizontal=True,
                width="stretch"
            )

        # -------------------------------------------------
        # INVENTORY TABLE
        # -------------------------------------------------

        st.subheader("Inventory Status Overview")

        if not st.session_state.inventory.empty:

            inv = st.session_state.inventory.copy()

            display_cols = [
                "Store ID",
                "Product ID",
                "Product Name",
                "Category",
                "Current Stock",
                "Forecast Demand",
                "Safety Stock",
                "Reorder Level",
                "Target Stock",
                "Order Qty",
                "Status"
            ]

            available = [
                c for c in display_cols
                if c in inv.columns
            ]

            st.dataframe(
                inv[available],
                width="stretch",
                hide_index=True
            )

        else:

            st.info(
                "Go to Manual Inventory and enter your current stock."
            )


# =========================================================
# MANUAL INVENTORY
# =========================================================

with tabs[1]:

    st.header("✏️ Manual Inventory")

    st.write(
        "Enter the current inventory manually. "
        "The uploaded CSV is used mainly for historical demand."
    )

    if df.empty:

        st.warning(
            "Upload the retail CSV first."
        )

    else:

        product_list = (
            df[[
                "Product ID",
                "Product Name",
                "Category"
            ]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        manual_rows = []

        for _, row in product_list.iterrows():

            pid = row["Product ID"]

            existing_stock = 0

            if not st.session_state.inventory.empty:

                old = st.session_state.inventory[
                    st.session_state.inventory["Product ID"] == pid
                ]

                if not old.empty:
                    existing_stock = int(
                        old.iloc[0]["Current Stock"]
                    )

            manual_rows.append({
                "Product ID": pid,
                "Product Name": row["Product Name"],
                "Category": row["Category"],
                "Current Stock": existing_stock
            })

        manual_df = pd.DataFrame(manual_rows)

        edited = st.data_editor(
            manual_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Current Stock": st.column_config.NumberColumn(
                    "Current Stock",
                    min_value=0,
                    step=1
                )
            }
        )

        if st.button(
            "💾 Save Inventory",
            width="stretch"
        ):

            result = edited.copy()

            result["Forecast Demand"] = (
                df.groupby("Product ID")["Demand"]
                .mean()
                .reindex(result["Product ID"])
                .fillna(0)
                .values * 30
            )

            result["Safety Stock"] = (
                result["Forecast Demand"]
                * safety_stock
                / 100
            ).round()

            result["Reorder Level"] = (
                result["Forecast Demand"]
                + result["Safety Stock"]
            ).round()

            result["Target Stock"] = (
                result["Reorder Level"]
                + result["Safety Stock"]
            ).round()

            result["Order Qty"] = (
                result["Target Stock"]
                - result["Current Stock"]
            ).clip(lower=0)

            def get_status(row):

                if row["Current Stock"] <= 0:
                    return "STOCKOUT"

                elif row["Current Stock"] < row["Reorder Level"]:
                    return "REORDER"

                elif row["Current Stock"] > row["Target Stock"]:
                    return "OVERSTOCK"

                return "HEALTHY"

            result["Status"] = result.apply(
                get_status,
                axis=1
            )

            result["Store ID"] = "S001"

            st.session_state.inventory = result

            st.success(
                "Manual inventory saved successfully!"
            )


# =========================================================
# INVENTORY PLANNING
# =========================================================

with tabs[2]:

    st.header("📦 Inventory Planning")

    if st.session_state.inventory.empty:

        st.info(
            "Enter your inventory in Manual Inventory first."
        )

    else:

        inv = st.session_state.inventory.copy()

        st.dataframe(
            inv,
            width="stretch",
            hide_index=True
        )

        reorder_items = inv[
            inv["Status"].isin(
                ["REORDER", "STOCKOUT"]
            )
        ]

        st.subheader("Recommended Orders")

        if reorder_items.empty:

            st.success(
                "No products currently require reordering."
            )

        else:

            st.dataframe(
                reorder_items[
                    [
                        "Product ID",
                        "Product Name",
                        "Current Stock",
                        "Reorder Level",
                        "Target Stock",
                        "Order Qty",
                        "Status"
                    ]
                ],
                width="stretch",
                hide_index=True
            )


# =========================================================
# FORECAST MODEL
# =========================================================

with tabs[3]:

    st.header("🤖 Forecast Model")

    if df.empty:

        st.info("Upload a CSV first.")

    else:

        st.write("Model:", model_name)
        st.write("Number of Trees:", trees)
        st.write("Safety Stock:", f"{safety_stock}%")

        if train_button:

            data = df.copy()

            values = (
                data.groupby("Product ID")["Demand"]
                .sum()
                .values
            )

            if len(values) >= 5:

                X = np.arange(
                    len(values)
                ).reshape(-1, 1)

                y = values

                split = int(
                    len(X) * 0.8
                )

                X_train = X[:split]
                X_test = X[split:]

                y_train = y[:split]
                y_test = y[split:]

                model = RandomForestRegressor(
                    n_estimators=trees,
                    random_state=42
                )

                model.fit(
                    X_train,
                    y_train
                )

                predictions = model.predict(
                    X_test
                )

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                mape = mean_absolute_percentage_error(
                    y_test,
                    predictions
                ) * 100

                st.session_state.model = model
                st.session_state.trained = True
                st.session_state.mae = mae
                st.session_state.mape = mape

                st.success(
                    "Model trained successfully!"
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "MAE",
                        f"{mae:.2f}"
                    )

                with c2:
                    st.metric(
                        "MAPE",
                        f"{mape:.2f}%"
                    )

            else:

                st.warning(
                    "Not enough data to train the model."
                )


# =========================================================
# ALERTS
# =========================================================

with tabs[4]:

    st.header("🚨 Inventory Alerts")

    if st.session_state.inventory.empty:

        st.info(
            "No inventory data available."
        )

    else:

        inv = st.session_state.inventory

        stockout = inv[
            inv["Status"] == "STOCKOUT"
        ]

        reorder = inv[
            inv["Status"] == "REORDER"
        ]

        overstock = inv[
            inv["Status"] == "OVERSTOCK"
        ]

        if not stockout.empty:
            st.error(
                f"🔴 {len(stockout)} product(s) have stockout."
            )

        if not reorder.empty:
            st.warning(
                f"🟡 {len(reorder)} product(s) need reordering."
            )

        if not overstock.empty:
            st.info(
                f"🔵 {len(overstock)} product(s) are overstocked."
            )

        if (
            stockout.empty
            and reorder.empty
            and overstock.empty
        ):
            st.success(
                "🟢 All inventory levels are healthy."
            )


# =========================================================
# ANALYTICS
# =========================================================

with tabs[5]:

    st.header("📊 Analytics")

    if df.empty:

        st.info("Upload a CSV first.")

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Category Demand")

            category = (
                df.groupby("Category")["Demand"]
                .sum()
                .sort_values(ascending=False)
            )

            st.dataframe(
                category.reset_index(),
                width="stretch",
                hide_index=True
            )

        with col2:

            st.subheader("Top Products")

            products_data = (
                df.groupby(
                    ["Product ID", "Product Name"]
                )["Demand"]
                .sum()
                .sort_values(
                    ascending=False
                )
                .head(10)
                .reset_index()
            )

            st.dataframe(
                products_data,
                width="stretch",
                hide_index=True
            )


# =========================================================
# 12 MONTH FORECAST
# =========================================================

with tabs[6]:

    st.header("🔮 12-Month Forecast")

    if df.empty:

        st.info("Upload a CSV first.")

    else:

        monthly_demand = (
            df.groupby("Category")["Demand"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        avg = monthly_demand.sum()

        forecast = []

        for month in range(1, 13):

            forecast.append({
                "Month": f"Month {month}",
                "Forecast Demand": round(
                    avg / 12,
                    2
                )
            })

        forecast_df = pd.DataFrame(
            forecast
        )

        st.dataframe(
            forecast_df,
            width="stretch",
            hide_index=True
        )

        st.line_chart(
            forecast_df.set_index("Month"),
            width="stretch"
        )
