import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Retail Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# BLACK UI
# =========================================================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #000000 !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] {
    background: #000000 !important;
    border-right: 1px solid #333333;
}

[data-testid="stHeader"] {
    background: #000000 !important;
}

h1, h2, h3, h4, p, label, span, div {
    color: #ffffff;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: #080808;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 20px;
    min-height: 125px;
}

.card-title {
    font-size: 14px;
    font-weight: 600;
}

.card-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: 10px;
}

.green {
    color: #4ade80 !important;
}

.red {
    color: #ef4444 !important;
}

.yellow {
    color: #facc15 !important;
}

.blue {
    color: #3b82f6 !important;
}

.nav-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #ffffff !important;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.stButton > button {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    background: #dddddd !important;
}

div[data-testid="stDataEditor"] {
    border: 1px solid #333333;
}

[data-testid="stMetric"] {
    background: #080808;
    border: 1px solid #333333;
    padding: 15px;
    border-radius: 8px;
}

hr {
    border-color: #333333 !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "inventory" not in st.session_state:
    st.session_state.inventory = pd.DataFrame(
        columns=[
            "Product ID",
            "Product Name",
            "Category",
            "Current Stock",
            "Reorder Level",
            "Safety Stock",
            "Unit Cost",
            "Lead Time"
        ]
    )

if "sales" not in st.session_state:
    st.session_state.sales = None

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def find_col(df, names):
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


def make_forecast(sales_df, inventory_df):
    if sales_df is None or sales_df.empty:
        return inventory_df.copy()

    date_col = find_col(
        sales_df,
        ["date", "order date", "sales date", "transaction date"]
    )

    product_col = find_col(
        sales_df,
        ["product id", "product", "product_id", "sku", "item"]
    )

    qty_col = find_col(
        sales_df,
        ["quantity", "qty", "units sold", "sales", "demand"]
    )

    if not date_col or not product_col or not qty_col:
        return inventory_df.copy()

    df = sales_df.copy()

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df[qty_col] = pd.to_numeric(
        df[qty_col],
        errors="coerce"
    ).fillna(0)

    df = df.dropna(subset=[date_col])

    results = []

    for _, item in inventory_df.iterrows():

        pid = item["Product ID"]

        temp = df[
            df[product_col].astype(str) == str(pid)
        ].copy()

        if temp.empty:
            forecast = 0
        else:
            daily = (
                temp.groupby(temp[date_col].dt.date)[qty_col]
                .sum()
                .reset_index()
            )

            if len(daily) >= 5:

                X = np.arange(len(daily)).reshape(-1, 1)
                y = daily[qty_col].values

                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )

                model.fit(X, y)

                future = np.arange(
                    len(daily),
                    len(daily) + 30
                ).reshape(-1, 1)

                forecast = max(
                    0,
                    model.predict(future).sum()
                )

            else:
                forecast = daily[qty_col].mean() * 30

        results.append(forecast)

    output = inventory_df.copy()

    output["Forecast Demand (30 Days)"] = np.round(
        results
    ).astype(int)

    output["Target Stock"] = (
        output["Forecast Demand (30 Days)"] +
        output["Safety Stock"]
    )

    output["Order Qty"] = (
        output["Target Stock"] -
        output["Current Stock"]
    ).clip(lower=0)

    output["Inventory Value"] = (
        output["Current Stock"] *
        output["Unit Cost"]
    )

    def status(row):

        if row["Current Stock"] <= 0:
            return "STOCKOUT"

        if row["Current Stock"] <= row["Reorder Level"]:
            return "REORDER"

        if row["Current Stock"] >
           row["Target Stock"] * 1.5:
            return "OVERSTOCK"

        return "HEALTHY"

    output["Status"] = output.apply(status, axis=1)

    return output


def create_inventory_from_sales(df):

    product_col = find_col(
        df,
        ["product id", "product", "product_id", "sku", "item"]
    )

    category_col = find_col(
        df,
        ["category", "product category", "type"]
    )

    if not product_col:
        return pd.DataFrame()

    products = df[product_col].dropna().astype(str).unique()

    rows = []

    for i, product in enumerate(products):

        category = "General"

        if category_col:
            vals = df[
                df[product_col].astype(str) == product
            ][category_col].dropna()

            if len(vals):
                category = str(vals.iloc[0])

        rows.append({
            "Product ID": product,
            "Product Name": product,
            "Category": category,
            "Current Stock": 0,
            "Reorder Level": 0,
            "Safety Stock": 0,
            "Unit Cost": 0.0,
            "Lead Time": 7
        })

    return pd.DataFrame(rows)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown(
        """
        <h2>📦 Retail Inventory<br>Intelligence</h2>
        <p>Demand Forecasting • Inventory Management • Reorder Planning</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="nav-title">DATA & SETTINGS</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Upload Sales CSV",
        type=["csv"]
    )

    if uploaded:

        try:
            sales = pd.read_csv(uploaded)

            st.session_state.sales = sales

            st.success("CSV Loaded")

            if st.session_state.inventory.empty:

                inv = create_inventory_from_sales(sales)

                if not inv.empty:
                    st.session_state.inventory = inv

        except Exception as e:
            st.error(f"CSV Error: {e}")

    st.markdown("---")

    st.markdown(
        '<div class="nav-title">MODEL SETTINGS</div>',
        unsafe_allow_html=True
    )

    model_name = st.selectbox(
        "Model",
        ["Random Forest"]
    )

    trees = st.selectbox(
        "Number of Trees",
        [50, 100, 200, 300],
        index=2
    )

    safety_percent = st.number_input(
        "Safety Stock (%)",
        min_value=0,
        max_value=100,
        value=15
    )

    if st.button("▶ Train / Refresh Model"):
        st.session_state.inventory["Safety Stock"] = (
            st.session_state.inventory["Forecast Demand (30 Days)"]
            * safety_percent / 100
            if "Forecast Demand (30 Days)"
            in st.session_state.inventory.columns
            else 0
        )

        st.success("Model refreshed")

    st.markdown("---")

    st.markdown(
        '<div class="nav-title">DATA SUMMARY</div>',
        unsafe_allow_html=True
    )

    if st.session_state.sales is not None:

        df = st.session_state.sales

        st.write(
            f"Total Records: **{len(df):,}**"
        )

        product_col = find_col(
            df,
            ["product id", "product", "product_id", "sku"]
        )

        category_col = find_col(
            df,
            ["category", "product category"]
        )

        if product_col:
            st.write(
                f"Products: **{df[product_col].nunique()}**"
            )

        if category_col:
            st.write(
                f"Categories: **{df[category_col].nunique()}**"
            )

    else:
        st.write("No sales data loaded")


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <h1>📦 Retail Inventory Intelligence</h1>
    <p style="font-size:17px;">
    Demand Forecasting • Manual Inventory Management • Reorder Planning
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================================================
# NAVIGATION
# =========================================================
pages = st.tabs([
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
with pages[0]:

    st.subheader("📊 Dashboard")

    inv = st.session_state.inventory.copy()

    if inv.empty:

        st.info(
            "Upload a sales CSV or enter products manually in Manual Inventory."
        )

    else:

        if st.session_state.sales is not None:

            inv = make_forecast(
                st.session_state.sales,
                inv
            )

            st.session_state.inventory = inv

        total_products = len(inv)

        current_stock = pd.to_numeric(
            inv["Current Stock"],
            errors="coerce"
        ).fillna(0).sum()

        reorder_level = pd.to_numeric(
            inv["Reorder Level"],
            errors="coerce"
        ).fillna(0).sum()

        stockouts = (
            (inv["Current Stock"] <= 0)
        ).sum()

        reorder = (
            (inv["Current Stock"] > 0) &
            (inv["Current Stock"] <= inv["Reorder Level"])
        ).sum()

        healthy = total_products - stockouts - reorder

        if "Target Stock" in inv.columns:
            overstock = (
                inv["Current Stock"] >
                inv["Target Stock"] * 1.5
            ).sum()
        else:
            overstock = 0

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">📦 Total Products</div>
                <div class="card-value">{total_products}</div>
                <small>Active Products</small>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">📦 Current Stock</div>
                <div class="card-value">{current_stock:,.0f}</div>
                <span class="green">Inventory Units</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">🛒 Reorder Level</div>
                <div class="card-value">{reorder_level:,.0f}</div>
                <small>Based on manual settings</small>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c4:
            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">🔴 Stockout</div>
                <div class="card-value red">{stockouts}</div>
                <small>Products</small>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c5:
            st.markdown(
                f"""
                <div class="card">
                <div class="card-title">🟡 Reorder</div>
                <div class="card-value yellow">{reorder}</div>
                <small>Products</small>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "🔵 Overstock",
                overstock
            )

        with b:
            st.metric(
                "🟢 Healthy",
                healthy
            )

        with c:
            if "Inventory Value" in inv:
                value = inv["Inventory Value"].sum()
            else:
                value = 0

            st.metric(
                "💰 Inventory Value",
                f"₹{value:,.0f}"
            )

        st.markdown("---")

        # Demand chart
        if (
            st.session_state.sales is not None
        ):

            sales = st.session_state.sales.copy()

            date_col = find_col(
                sales,
                ["date", "order date", "sales date"]
            )

            qty_col = find_col(
                sales,
                ["quantity", "qty", "units sold", "sales", "demand"]
            )

            if date_col and qty_col:

                sales[date_col] = pd.to_datetime(
                    sales[date_col],
                    errors="coerce"
                )

                sales[qty_col] = pd.to_numeric(
                    sales[qty_col],
                    errors="coerce"
                )

                daily = (
                    sales.dropna(subset=[date_col])
                    .groupby(
                        sales[date_col].dt.date
                    )[qty_col]
                    .sum()
                    .tail(60)
                )

                st.subheader(
                    "Actual Demand"
                )

                st.line_chart(
                    daily,
                    height=350
                )

        st.subheader("Inventory Status Overview")

        display_cols = [
            "Product ID",
            "Product Name",
            "Category",
            "Current Stock",
            "Reorder Level",
            "Safety Stock"
        ]

        if "Forecast Demand (30 Days)" in inv.columns:
            display_cols.append(
                "Forecast Demand (30 Days)"
            )

        if "Order Qty" in inv.columns:
            display_cols.append("Order Qty")

        if "Status" in inv.columns:
            display_cols.append("Status")

        st.dataframe(
            inv[
                [c for c in display_cols if c in inv.columns]
            ],
            width="stretch",
            hide_index=True
        )


# =========================================================
# MANUAL INVENTORY
# =========================================================
with pages[1]:

    st.subheader("✏️ Manual Inventory Entry")

    st.write(
        "Enter your **real current inventory manually**. "
        "These values are used for reorder and inventory planning."
    )

    if st.session_state.inventory.empty:

        st.warning(
            "No products available. Add products below."
        )

    # -----------------------------------------------------
    # ADD NEW PRODUCT
    # -----------------------------------------------------
    st.markdown("### ➕ Add Product")

    col1, col2, col3 = st.columns(3)

    with col1:
        new_id = st.text_input(
            "Product ID",
            placeholder="P001"
        )

    with col2:
        new_name = st.text_input(
            "Product Name",
            placeholder="Cola 500ml"
        )

    with col3:
        new_category = st.text_input(
            "Category",
            placeholder="Beverages"
        )

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        new_stock = st.number_input(
            "Current Stock",
            min_value=0,
            value=0
        )

    with col5:
        new_reorder = st.number_input(
            "Reorder Level",
            min_value=0,
            value=0
        )

    with col6:
        new_cost = st.number_input(
            "Unit Cost ₹",
            min_value=0.0,
            value=0.0
        )

    with col7:
        new_lead = st.number_input(
            "Lead Time Days",
            min_value=1,
            value=7
        )

    if st.button("➕ Add Product"):

        if new_id and new_name:

            if (
                not st.session_state.inventory.empty
                and new_id in
                st.session_state.inventory["Product ID"].astype(str).values
            ):
                st.error("Product ID already exists.")

            else:

                new_row = pd.DataFrame([{
                    "Product ID": new_id,
                    "Product Name": new_name,
                    "Category": new_category,
                    "Current Stock": new_stock,
                    "Reorder Level": new_reorder,
                    "Safety Stock": 0,
                    "Unit Cost": new_cost,
                    "Lead Time": new_lead
                }])

                st.session_state.inventory = pd.concat(
                    [
                        st.session_state.inventory,
                        new_row
                    ],
                    ignore_index=True
                )

                st.success(
                    f"{new_name} added to inventory."
                )

                st.rerun()

        else:
            st.error(
                "Product ID and Product Name are required."
            )

    st.markdown("---")

    # -----------------------------------------------------
    # EDIT INVENTORY
    # -----------------------------------------------------
    st.markdown("### 📝 Edit Current Inventory")

    if not st.session_state.inventory.empty:

        edit_df = st.session_state.inventory.copy()

        editable_columns = [
            "Product ID",
            "Product Name",
            "Category",
            "Current Stock",
            "Reorder Level",
            "Safety Stock",
            "Unit Cost",
            "Lead Time"
        ]

        edit_df = edit_df[
            [
                c for c in editable_columns
                if c in edit_df.columns
            ]
        ]

        edited = st.data_editor(
            edit_df,
            width="stretch",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Current Stock": st.column_config.NumberColumn(
                    "Current Stock",
                    min_value=0,
                    step=1
                ),
                "Reorder Level": st.column_config.NumberColumn(
                    "Reorder Level",
                    min_value=0,
                    step=1
                ),
                "Safety Stock": st.column_config.NumberColumn(
                    "Safety Stock",
                    min_value=0,
                    step=1
                ),
                "Unit Cost": st.column_config.NumberColumn(
                    "Unit Cost ₹",
                    min_value=0,
                    step=1
                ),
                "Lead Time": st.column_config.NumberColumn(
                    "Lead Time",
                    min_value=1,
                    step=1
                )
            }
        )

        if st.button("💾 Save Inventory"):

            st.session_state.inventory = edited.copy()

            st.success(
                "Manual inventory saved successfully."
            )

            st.rerun()

    else:

        st.info(
            "Add your first product using the form above."
        )


# =========================================================
# INVENTORY PLANNING
# =========================================================
with pages[2]:

    st.subheader("📦 Inventory Planning")

    inv = st.session_state.inventory.copy()

    if inv.empty:

        st.info(
            "Enter inventory manually first."
        )

    else:

        if st.session_state.sales is not None:

            inv = make_forecast(
                st.session_state.sales,
                inv
            )

        else:

            inv["Forecast Demand (30 Days)"] = 0

            inv["Target Stock"] = (
                inv["Reorder Level"] +
                inv["Safety Stock"]
            )

            inv["Order Qty"] = (
                inv["Target Stock"] -
                inv["Current Stock"]
            ).clip(lower=0)

            inv["Inventory Value"] = (
                inv["Current Stock"] *
                inv["Unit Cost"]
            )

        st.session_state.inventory = inv

        st.dataframe(
            inv,
            width="stretch",
            hide_index=True
        )

        st.markdown("### 🛒 Recommended Orders")

        order_df = inv[
            inv["Order Qty"] > 0
        ]

        if order_df.empty:

            st.success(
                "No products currently require ordering."
            )

        else:

            st.dataframe(
                order_df[
                    [
                        "Product ID",
                        "Product Name",
                        "Current Stock",
                        "Forecast Demand (30 Days)",
                        "Reorder Level",
                        "Order Qty",
                        "Lead Time"
                    ]
                ],
                width="stretch",
                hide_index=True
            )


# =========================================================
# FORECAST MODEL
# =========================================================
with pages[3]:

    st.subheader("🤖 Forecast Model")

    if st.session_state.sales is None:

        st.info(
            "Upload a sales CSV to train the forecasting model."
        )

    else:

        st.write(
            f"Model: **{model_name}**"
        )

        st.write(
            f"Number of Trees: **{trees}**"
        )

        st.write(
            "The model forecasts product demand for the next 30 days."
        )

        st.success(
            "Random Forest forecasting is ready."
        )


# =========================================================
# ALERTS
# =========================================================
with pages[4]:

    st.subheader("🚨 Inventory Alerts")

    inv = st.session_state.inventory.copy()

    if inv.empty:

        st.info(
            "Enter inventory first."
        )

    else:

        if "Status" not in inv.columns:
            inv = make_forecast(
                st.session_state.sales,
                inv
            ) if st.session_state.sales is not None else inv

        if "Status" in inv.columns:

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
                    f"🔴 {len(stockout)} product(s) have zero stock."
                )

                st.dataframe(
                    stockout,
                    width="stretch",
                    hide_index=True
                )

            if not reorder.empty:

                st.warning(
                    f"🟡 {len(reorder)} product(s) need reordering."
                )

                st.dataframe(
                    reorder,
                    width="stretch",
                    hide_index=True
                )

            if not overstock.empty:

                st.info(
                    f"🔵 {len(overstock)} product(s) are overstocked."
                )

            if (
                stockout.empty
                and reorder.empty
            ):
                st.success(
                    "🟢 No critical inventory alerts."
                )


# =========================================================
# ANALYTICS
# =========================================================
with pages[5]:

    st.subheader("📊 Analytics")

    inv = st.session_state.inventory.copy()

    if inv.empty:

        st.info(
            "Enter inventory first."
        )

    else:

        if "Category" in inv.columns:

            category_value = (
                inv.groupby("Category")["Current Stock"]
                .sum()
                .sort_values(ascending=False)
            )

            st.markdown(
                "### Current Stock by Category"
            )

            st.bar_chart(
                category_value
            )

        if "Inventory Value" in inv.columns:

            value = (
                inv.groupby("Category")["Inventory Value"]
                .sum()
                .sort_values(ascending=False)
            )

            st.markdown(
                "### Inventory Value by Category"
            )

            st.bar_chart(
                value
            )


# =========================================================
# 12 MONTH FORECAST
# =========================================================
with pages[6]:

    st.subheader("🔮 12-Month Forecast")

    if st.session_state.sales is None:

        st.info(
            "Upload historical sales data to generate forecasts."
        )

    else:

        sales = st.session_state.sales.copy()

        date_col = find_col(
            sales,
            ["date", "order date", "sales date"]
        )

        qty_col = find_col(
            sales,
            ["quantity", "qty", "units sold", "sales", "demand"]
        )

        if date_col and qty_col:

            sales[date_col] = pd.to_datetime(
                sales[date_col],
                errors="coerce"
            )

            sales[qty_col] = pd.to_numeric(
                sales[qty_col],
                errors="coerce"
            )

            monthly = (
                sales.dropna(subset=[date_col])
                .groupby(
                    sales[date_col].dt.to_period("M")
                )[qty_col]
                .sum()
            )

            if len(monthly) >= 3:

                X = np.arange(
                    len(monthly)
                ).reshape(-1, 1)

                y = monthly.values

                model = RandomForestRegressor(
                    n_estimators=trees,
                    random_state=42
                )

                model.fit(X, y)

                future_x = np.arange(
                    len(monthly),
                    len(monthly) + 12
                ).reshape(-1, 1)

                forecast = model.predict(
                    future_x
                )

                future_dates = pd.date_range(
                    monthly.index[-1].to_timestamp()
                    + pd.DateOffset(months=1),
                    periods=12,
                    freq="MS"
                )

                forecast_df = pd.DataFrame({
                    "Month": future_dates,
                    "Forecast Demand": np.round(
                        forecast
                    ).astype(int)
                })

                st.dataframe(
                    forecast_df,
                    width="stretch",
                    hide_index=True
                )

                st.line_chart(
                    forecast_df.set_index("Month")
                )

            else:

                st.warning(
                    "At least 3 months of sales history are required."
                )

        else:

            st.error(
                "Could not detect Date and Quantity columns."
            )
