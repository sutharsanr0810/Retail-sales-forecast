import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


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

.stApp {
    background: #000000;
    color: #ffffff;
}

section[data-testid="stSidebar"] {
    background: #000000;
    border-right: 1px solid #333333;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

p, label {
    color: #ffffff;
}

/* Cards */

.card {
    background: #050505;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 18px;
    min-height: 115px;
}

.card-title {
    color: #ffffff;
    font-size: 14px;
    font-weight: bold;
}

.card-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: bold;
    margin-top: 8px;
}

.card-sub {
    color: #aaaaaa;
    font-size: 12px;
}

/* =====================================================
   UPLOAD BUTTON
   ===================================================== */

[data-testid="stFileUploader"] {
    background: #080808 !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploader"] button {
    background: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #555555 !important;
}

[data-testid="stFileUploader"] button:hover {
    background: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #ffffff !important;
}

[data-testid="stFileUploader"] button svg {
    stroke: #ffffff !important;
}

/* =====================================================
   GENERAL BUTTONS
   ===================================================== */

.stButton > button {
    background: #ffffff;
    color: #000000;
    border: none;
    border-radius: 6px;
    font-weight: bold;
}

.stButton > button:hover {
    background: #dddddd;
    color: #000000;
}

/* =====================================================
   INPUTS
   ===================================================== */

div[data-baseweb="select"] > div {
    background: #050505 !important;
    color: #ffffff !important;
    border-color: #444444 !important;
}

input {
    background: #050505 !important;
    color: #ffffff !important;
}

/* =====================================================
   STATUS
   ===================================================== */

.red {
    color: #ff3333;
}

.yellow {
    color: #ffd000;
}

.blue {
    color: #2196ff;
}

.green {
    color: #55dd55;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "sales_data" not in st.session_state:
    st.session_state.sales_data = pd.DataFrame()

if "inventory" not in st.session_state:
    st.session_state.inventory = pd.DataFrame()

if "model" not in st.session_state:
    st.session_state.model = None

if "trained" not in st.session_state:
    st.session_state.trained = False

if "mae" not in st.session_state:
    st.session_state.mae = 0

if "mape" not in st.session_state:
    st.session_state.mape = 0


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def find_column(df, names):

    for name in names:
        if name in df.columns:
            return name

    lower = {
        str(c).lower().replace("_", " ").strip(): c
        for c in df.columns
    }

    for name in names:
        key = name.lower().replace("_", " ").strip()

        if key in lower:
            return lower[key]

    return None


def prepare_data(raw):

    df = raw.copy()

    date_col = find_column(
        df,
        ["Date", "date", "Order Date", "Transaction Date"]
    )

    product_id_col = find_column(
        df,
        ["Product ID", "Product_ID", "ProductID"]
    )

    product_name_col = find_column(
        df,
        ["Product Name", "Product_Name", "Product"]
    )

    category_col = find_column(
        df,
        ["Category", "category"]
    )

    store_col = find_column(
        df,
        ["Store ID", "Store_ID", "StoreID", "Store"]
    )

    sales_col = find_column(
        df,
        [
            "Units Sold",
            "Units_Sold",
            "Sales",
            "Quantity",
            "Quantity Sold",
            "Demand"
        ]
    )

    # Date
    if date_col:
        df["Date"] = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )
    else:
        df["Date"] = pd.date_range(
            start="2024-01-01",
            periods=len(df),
            freq="D"
        )

    # Product ID
    if product_id_col:
        df["Product ID"] = df[product_id_col].astype(str)
    else:
        df["Product ID"] = "P001"

    # Product Name
    if product_name_col:
        df["Product Name"] = df[product_name_col].astype(str)
    else:
        df["Product Name"] = df["Product ID"]

    # Category
    if category_col:
        df["Category"] = df[category_col].astype(str)
    else:
        df["Category"] = "General"

    # Store
    if store_col:
        df["Store ID"] = df[store_col].astype(str)
    else:
        df["Store ID"] = "S001"

    # Sales
    if sales_col:
        df["Demand"] = pd.to_numeric(
            df[sales_col],
            errors="coerce"
        ).fillna(0)
    else:

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_cols:
            df["Demand"] = pd.to_numeric(
                df[numeric_cols[-1]],
                errors="coerce"
            ).fillna(0)
        else:
            df["Demand"] = 0

    return df


def calculate_inventory(df, stock_data, safety_percent):

    result = stock_data.copy()

    avg_daily = (
        df.groupby("Product ID")["Demand"]
        .mean()
    )

    result["Avg Daily Demand"] = (
        result["Product ID"]
        .map(avg_daily)
        .fillna(0)
    )

    # 30-day forecast
    result["Forecast Demand"] = (
        result["Avg Daily Demand"] * 30
    ).round()

    # Safety stock
    result["Safety Stock"] = (
        result["Forecast Demand"]
        * safety_percent
        / 100
    ).round()

    # Reorder level
    result["Reorder Level"] = (
        result["Forecast Demand"] / 30
        * result["Lead Time"]
        + result["Safety Stock"]
    ).round()

    # Target stock
    result["Target Stock"] = (
        result["Forecast Demand"]
        + result["Safety Stock"]
    ).round()

    # Recommended order
    result["Order Qty"] = (
        result["Target Stock"]
        - result["Current Stock"]
    ).clip(lower=0).round()

    def status(row):

        stock = row["Current Stock"]
        reorder = row["Reorder Level"]
        target = row["Target Stock"]

        if stock <= 0:
            return "STOCKOUT"

        if stock < reorder:
            return "REORDER"

        if stock > target:
            return "OVERSTOCK"

        return "HEALTHY"

    result["Status"] = result.apply(
        status,
        axis=1
    )

    return result


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

    uploaded = st.file_uploader(
        "Upload Sales/Inventory CSV",
        type=["csv"]
    )

    if uploaded is not None:

        try:

            raw = pd.read_csv(uploaded)

            st.session_state.sales_data = prepare_data(raw)

            st.success("CSV loaded successfully")

        except Exception as e:

            st.error(
                f"CSV error: {e}"
            )

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
        value=15,
        step=1
    )

    train_button = st.button(
        "▶ Train / Refresh Model",
        width="stretch"
    )

    st.markdown("### DATA SUMMARY")

    df = st.session_state.sales_data

    if not df.empty:

        st.write(
            f"Total Records: {len(df)}"
        )

        if not df["Date"].isna().all():

            st.write(
                "Date Range:",
                f"{df['Date'].min().date()} - "
                f"{df['Date'].max().date()}"
            )

        st.write(
            f"Stores: {df['Store ID'].nunique()}"
        )

        st.write(
            f"Products: {df['Product ID'].nunique()}"
        )

        st.write(
            f"Categories: {df['Category'].nunique()}"
        )


# =========================================================
# MAIN HEADER
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

tab_dashboard, tab_manual, tab_planning, tab_model, \
tab_alerts, tab_analytics, tab_forecast = st.tabs([
    "Dashboard",
    "Manual Inventory",
    "Inventory Planning",
    "Forecast Model",
    "Alerts",
    "Analytics",
    "12-Month Forecast"
])


df = st.session_state.sales_data


# =========================================================
# DASHBOARD
# =========================================================

with tab_dashboard:

    st.subheader("📊 Dashboard")

    if df.empty:

        st.info(
            "Upload the retail CSV to start."
        )

    else:

        products = df["Product ID"].nunique()

        current_stock = 0

        if not st.session_state.inventory.empty:
            current_stock = int(
                st.session_state.inventory[
                    "Current Stock"
                ].sum()
            )

        reorder_level = 0

        if not st.session_state.inventory.empty:
            reorder_level = int(
                st.session_state.inventory[
                    "Reorder Level"
                ].sum()
            )

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
                <div class="card-sub">Based on inventory</div>
            </div>
            """, unsafe_allow_html=True)

        with c4:

            st.markdown(f"""
            <div class="card">
                <div class="card-title">〽 MAE</div>
                <div class="card-value">{st.session_state.mae:.2f}</div>
                <div class="card-sub">Lower is better</div>
            </div>
            """, unsafe_allow_html=True)

        with c5:

            st.markdown(f"""
            <div class="card">
                <div class="card-title">🎯 MAPE</div>
                <div class="card-value">{st.session_state.mape:.2f}%</div>
                <div class="card-sub">Lower is better</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # STATUS
        stockout = 0
        reorder = 0
        overstock = 0
        healthy = 0

        if not st.session_state.inventory.empty:

            status = st.session_state.inventory["Status"]

            stockout = (status == "STOCKOUT").sum()
            reorder = (status == "REORDER").sum()
            overstock = (status == "OVERSTOCK").sum()
            healthy = (status == "HEALTHY").sum()

        s1, s2, s3, s4 = st.columns(4)

        with s1:

            st.markdown(f"""
            <div class="card">
                <div class="red">🔴 Stockout</div>
                <div class="card-value">{stockout}</div>
                <div class="card-sub">Products</div>
            </div>
            """, unsafe_allow_html=True)

        with s2:

            st.markdown(f"""
            <div class="card">
                <div class="yellow">🟡 Reorder</div>
                <div class="card-value">{reorder}</div>
                <div class="card-sub">Products</div>
            </div>
            """, unsafe_allow_html=True)

        with s3:

            st.markdown(f"""
            <div class="card">
                <div class="blue">🔵 Overstock</div>
                <div class="card-value">{overstock}</div>
                <div class="card-sub">Products</div>
            </div>
            """, unsafe_allow_html=True)

        with s4:

            st.markdown(f"""
            <div class="card">
                <div class="green">🟢 Healthy</div>
                <div class="card-value">{healthy}</div>
                <div class="card-sub">Products</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # CHARTS

        col1, col2, col3 = st.columns([1.5, 1, 1])

        with col1:

            st.subheader(
                "Actual vs Predicted Demand"
            )

            daily = (
                df.groupby("Date")["Demand"]
                .sum()
                .sort_index()
            )

            if len(daily) > 2:

                predicted = (
                    daily.rolling(
                        7,
                        min_periods=1
                    ).mean()
                )

                fig, ax = plt.subplots()

                fig.patch.set_facecolor("#000000")
                ax.set_facecolor("#000000")

                ax.plot(
                    daily.values,
                    label="Actual Demand"
                )

                ax.plot(
                    predicted.values,
                    label="Predicted Demand"
                )

                ax.tick_params(
                    colors="white"
                )

                ax.legend()

                for spine in ax.spines.values():
                    spine.set_color("#333333")

                st.pyplot(
                    fig,
                    width="stretch"
                )

        with col2:

            st.subheader(
                "Demand by Category"
            )

            category = (
                df.groupby("Category")["Demand"]
                .sum()
                .sort_values(ascending=False)
            )

            fig, ax = plt.subplots()

            fig.patch.set_facecolor("#000000")
            ax.set_facecolor("#000000")

            ax.pie(
                category.values,
                labels=category.index,
                autopct="%1.1f%%"
            )

            st.pyplot(
                fig,
                width="stretch"
            )

        with col3:

            st.subheader(
                "Demand by Category"
            )

            st.bar_chart(
                category,
                horizontal=True,
                width="stretch"
            )

        # TABLE

        st.subheader(
            "Inventory Status Overview"
        )

        if not st.session_state.inventory.empty:

            st.dataframe(
                st.session_state.inventory,
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

with tab_manual:

    st.header("✏️ Manual Inventory")

    st.write(
        "Enter your actual current stock manually. "
        "Historical sales data is used to calculate demand."
    )

    if df.empty:

        st.warning(
            "Upload the sales CSV first."
        )

    else:

        products = (
            df[
                [
                    "Product ID",
                    "Product Name",
                    "Category"
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        rows = []

        for _, p in products.iterrows():

            product_id = p["Product ID"]

            old_stock = 0
            old_lead = 7

            if not st.session_state.inventory.empty:

                old = st.session_state.inventory[
                    st.session_state.inventory[
                        "Product ID"
                    ] == product_id
                ]

                if not old.empty:

                    old_stock = int(
                        old.iloc[0]["Current Stock"]
                    )

                    old_lead = int(
                        old.iloc[0]["Lead Time"]
                    )

            rows.append({
                "Product ID": product_id,
                "Product Name": p["Product Name"],
                "Category": p["Category"],
                "Current Stock": old_stock,
                "Lead Time": old_lead
            })

        manual_df = pd.DataFrame(rows)

        edited = st.data_editor(
            manual_df,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            column_config={

                "Product ID":
                    st.column_config.TextColumn(
                        "Product ID",
                        disabled=True
                    ),

                "Product Name":
                    st.column_config.TextColumn(
                        "Product Name",
                        disabled=True
                    ),

                "Category":
                    st.column_config.TextColumn(
                        "Category",
                        disabled=True
                    ),

                "Current Stock":
                    st.column_config.NumberColumn(
                        "Current Stock",
                        min_value=0,
                        step=1
                    ),

                "Lead Time":
                    st.column_config.NumberColumn(
                        "Lead Time (Days)",
                        min_value=1,
                        max_value=365,
                        step=1
                    )
            }
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "💾 SAVE INVENTORY",
            width="stretch"
        ):

            inventory = calculate_inventory(
                df,
                edited,
                safety_stock
            )

            inventory["Store ID"] = "S001"

            st.session_state.inventory = inventory

            st.success(
                "Inventory saved successfully!"
            )

            st.rerun()

        # Explanation

        st.markdown("---")

        st.subheader(
            "How Manual Inventory Works"
        )

        st.write(
            "• Current Stock → You enter this manually"
        )

        st.write(
            "• Lead Time → You enter the supplier delivery time"
        )

        st.write(
            "• Forecast Demand → Calculated from historical sales"
        )

        st.write(
            "• Safety Stock → Calculated using the selected percentage"
        )

        st.write(
            "• Reorder Level → Forecast demand during lead time + safety stock"
        )

        st.write(
            "• Target Stock → Forecast demand + safety stock"
        )

        st.write(
            "• Order Quantity → Target stock − current stock"
        )


# =========================================================
# INVENTORY PLANNING
# =========================================================

with tab_planning:

    st.header("📦 Inventory Planning")

    if st.session_state.inventory.empty:

        st.warning(
            "No inventory has been entered yet."
        )

        st.write(
            "Go to **Manual Inventory**, enter your current stock "
            "and lead time, then click **SAVE INVENTORY**."
        )

    else:

        inv = st.session_state.inventory.copy()

        st.subheader(
            "Inventory Recommendations"
        )

        # KPIs

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric(
                "Total Stock",
                f"{int(inv['Current Stock'].sum()):,}"
            )

        with p2:
            st.metric(
                "Recommended Orders",
                f"{int(inv['Order Qty'].sum()):,}"
            )

        with p3:
            st.metric(
                "Reorder Products",
                int(
                    (inv["Status"] == "REORDER").sum()
                    +
                    (inv["Status"] == "STOCKOUT").sum()
                )
            )

        with p4:
            st.metric(
                "Healthy Products",
                int(
                    (inv["Status"] == "HEALTHY").sum()
                )
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.dataframe(
            inv[
                [
                    "Product ID",
                    "Product Name",
                    "Category",
                    "Current Stock",
                    "Lead Time",
                    "Forecast Demand",
                    "Safety Stock",
                    "Reorder Level",
                    "Target Stock",
                    "Order Qty",
                    "Status"
                ]
            ],
            width="stretch",
            hide_index=True
        )

        st.subheader(
            "Recommended Purchase Orders"
        )

        orders = inv[
            inv["Order Qty"] > 0
        ].copy()

        if orders.empty:

            st.success(
                "No purchase orders are currently required."
            )

        else:

            st.dataframe(
                orders[
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

with tab_model:

    st.header("🤖 Forecast Model")

    if df.empty:

        st.info(
            "Upload a CSV first."
        )

    else:

        st.write(
            f"Model: **{model_name}**"
        )

        st.write(
            f"Trees: **{trees}**"
        )

        if train_button:

            daily = (
                df.groupby("Date")["Demand"]
                .sum()
                .sort_index()
            )

            if len(daily) >= 10:

                X = np.arange(
                    len(daily)
                ).reshape(-1, 1)

                y = daily.values

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

                pred = model.predict(
                    X_test
                )

                mae = mean_absolute_error(
                    y_test,
                    pred
                )

                mape = (
                    mean_absolute_percentage_error(
                        y_test,
                        pred
                    ) * 100
                )

                st.session_state.model = model
                st.session_state.trained = True
                st.session_state.mae = mae
                st.session_state.mape = mape

                st.success(
                    "Model trained successfully!"
                )

                a, b = st.columns(2)

                with a:
                    st.metric(
                        "MAE",
                        f"{mae:.2f}"
                    )

                with b:
                    st.metric(
                        "MAPE",
                        f"{mape:.2f}%"
                    )

            else:

                st.warning(
                    "Not enough historical data."
                )


# =========================================================
# ALERTS
# =========================================================

with tab_alerts:

    st.header("🚨 Inventory Alerts")

    if st.session_state.inventory.empty:

        st.info(
            "Save your manual inventory first."
        )

    else:

        inv = st.session_state.inventory

        stockouts = inv[
            inv["Status"] == "STOCKOUT"
        ]

        reorders = inv[
            inv["Status"] == "REORDER"
        ]

        overstock = inv[
            inv["Status"] == "OVERSTOCK"
        ]

        healthy = inv[
            inv["Status"] == "HEALTHY"
        ]

        if not stockouts.empty:

            st.error(
                f"🔴 {len(stockouts)} product(s) are out of stock."
            )

            st.dataframe(
                stockouts[
                    [
                        "Product ID",
                        "Product Name",
                        "Current Stock",
                        "Order Qty"
                    ]
                ],
                width="stretch",
                hide_index=True
            )

        if not reorders.empty:

            st.warning(
                f"🟡 {len(reorders)} product(s) need reordering."
            )

        if not overstock.empty:

            st.info(
                f"🔵 {len(overstock)} product(s) are overstocked."
            )

        if (
            stockouts.empty
            and reorders.empty
            and overstock.empty
        ):

            st.success(
                f"🟢 All {len(healthy)} products are healthy."
            )


# =========================================================
# ANALYTICS
# =========================================================

with tab_analytics:

    st.header("📊 Analytics")

    if df.empty:

        st.info(
            "Upload a CSV first."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Category Demand"
            )

            category = (
                df.groupby("Category")["Demand"]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.dataframe(
                category.reset_index(),
                width="stretch",
                hide_index=True
            )

        with col2:

            st.subheader(
                "Top Products"
            )

            top_products = (
                df.groupby(
                    [
                        "Product ID",
                        "Product Name"
                    ]
                )["Demand"]
                .sum()
                .sort_values(
                    ascending=False
                )
                .head(10)
                .reset_index()
            )

            st.dataframe(
                top_products,
                width="stretch",
                hide_index=True
            )


# =========================================================
# 12 MONTH FORECAST
# =========================================================

with tab_forecast:

    st.header("🔮 12-Month Forecast")

    if df.empty:

        st.info(
            "Upload a CSV first."
        )

    else:

        monthly = (
            df.set_index("Date")["Demand"]
            .resample("ME")
            .sum()
        )

        if len(monthly) > 0:

            avg_monthly = monthly.mean()

            forecast_values = []

            for i in range(12):

                forecast_values.append(
                    avg_monthly
                )

            forecast_df = pd.DataFrame({
                "Month": [
                    f"Month {i + 1}"
                    for i in range(12)
                ],
                "Forecast Demand": np.round(
                    forecast_values,
                    2
                )
            })

            st.dataframe(
                forecast_df,
                width="stretch",
                hide_index=True
            )

            st.line_chart(
                forecast_df.set_index("Month"),
                width="stretch"
            )
