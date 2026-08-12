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
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="stHeader"] {
    background:#000000 !important;
    color:#ffffff !important;
}

[data-testid="stSidebar"] {
    background:#000000 !important;
    border-right:1px solid #292929;
}

[data-testid="stSidebar"] * {
    color:#ffffff !important;
}

h1,h2,h3,h4,p,label,span {
    color:#ffffff !important;
}

.block-container {
    padding-top:2rem;
    max-width:1500px;
}

.card {
    background:#080808;
    border:1px solid #303030;
    border-radius:8px;
    padding:18px;
    min-height:110px;
}

.card-title {
    color:#dddddd;
    font-size:14px;
}

.card-value {
    color:#ffffff;
    font-size:30px;
    font-weight:bold;
    margin-top:8px;
}

.card-sub {
    color:#62d84e;
    font-size:13px;
    margin-top:5px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background:#080808 !important;
    border-color:#333333 !important;
}

input {
    background:#080808 !important;
    color:white !important;
}

[data-testid="stDataEditor"] {
    border:1px solid #333333;
}

[data-testid="stFileUploader"] {
    background:#080808;
    border:1px solid #333333;
    border-radius:8px;
}

hr {
    border-color:#222222 !important;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================
if "sales" not in st.session_state:
    st.session_state.sales = None

if "inventory" not in st.session_state:
    st.session_state.inventory = pd.DataFrame()

if "saved" not in st.session_state:
    st.session_state.saved = False


# =========================================================
# FUNCTIONS
# =========================================================
def find_column(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def prepare_sales(df):
    df = df.copy()

    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]

    date_col = find_column(
        df,
        ["date", "order_date", "sales_date", "transaction_date"]
    )

    product_col = find_column(
        df,
        ["product_id", "product", "sku", "item_id", "item"]
    )

    demand_col = find_column(
        df,
        ["units_sold", "quantity", "qty", "sales", "units", "demand"]
    )

    category_col = find_column(
        df,
        ["category", "product_category", "department"]
    )

    price_col = find_column(
        df,
        ["unit_price", "price", "selling_price"]
    )

    if date_col is None or product_col is None or demand_col is None:
        return None

    result = pd.DataFrame()

    result["date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    result["product_id"] = df[product_col].astype(str)

    result["demand"] = pd.to_numeric(
        df[demand_col],
        errors="coerce"
    ).fillna(0)

    if category_col:
        result["category"] = df[category_col].astype(str)
    else:
        result["category"] = "Other"

    if price_col:
        result["price"] = pd.to_numeric(
            df[price_col],
            errors="coerce"
        ).fillna(0)
    else:
        result["price"] = 0

    result = result.dropna(subset=["date"])

    return result


def create_inventory(sales):
    products = sales["product_id"].unique()

    rows = []

    for p in products:
        temp = sales[sales["product_id"] == p]

        rows.append({
            "Product ID": p,
            "Product Name": p,
            "Category": temp["category"].iloc[-1],
            "Current Stock": 0,
            "Reorder Level": 0,
            "Safety Stock %": 15,
            "Unit Cost": float(temp["price"].mean()),
            "Lead Time Days": 7
        })

    return pd.DataFrame(rows)


def forecast_product(sales, product, trees):

    temp = sales[
        sales["product_id"] == product
    ]

    daily = (
        temp.groupby("date")["demand"]
        .sum()
        .sort_index()
    )

    if len(daily) < 14:
        avg = daily.mean() if len(daily) else 0
        return avg * 30, None, None, None

    df = daily.reset_index()

    df["lag1"] = df["demand"].shift(1)
    df["lag7"] = df["demand"].shift(7)
    df["rolling7"] = (
        df["demand"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df = df.dropna()

    if len(df) < 10:
        return daily.mean() * 30, None, None, None

    X = df[["lag1", "lag7", "rolling7"]]
    y = df["demand"]

    split = int(len(df) * 0.8)

    if split >= len(df):
        split = len(df) - 1

    model = RandomForestRegressor(
        n_estimators=trees,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X.iloc[:split],
        y.iloc[:split]
    )

    predicted = model.predict(
        X.iloc[split:]
    )

    actual = y.iloc[split:]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    history = df["demand"].tolist()

    future = []

    for _ in range(30):

        lag1 = history[-1]

        lag7 = (
            history[-7]
            if len(history) >= 7
            else np.mean(history)
        )

        rolling7 = np.mean(history[-7:])

        x = pd.DataFrame(
            [[lag1, lag7, rolling7]],
            columns=[
                "lag1",
                "lag7",
                "rolling7"
            ]
        )

        value = model.predict(x)[0]

        value = max(0, value)

        future.append(value)

        history.append(value)

    return (
        sum(future),
        actual,
        predicted,
        (mae, rmse)
    )


def inventory_calculation(
    sales,
    inventory,
    trees
):

    output = []

    for _, row in inventory.iterrows():

        product = str(
            row["Product ID"]
        ).strip()

        if product == "":
            continue

        stock = float(
            row.get("Current Stock", 0)
        )

        manual_reorder = float(
            row.get("Reorder Level", 0)
        )

        safety_percent = float(
            row.get("Safety Stock %", 15)
        )

        cost = float(
            row.get("Unit Cost", 0)
        )

        lead_time = float(
            row.get("Lead Time Days", 7)
        )

        forecast, actual, predicted, metrics = (
            forecast_product(
                sales,
                product,
                trees
            )
        )

        daily_demand = forecast / 30

        safety_stock = (
            daily_demand
            * lead_time
            * safety_percent
            / 100
        )

        calculated_reorder = (
            daily_demand
            * lead_time
            + safety_stock
        )

        if manual_reorder > 0:
            reorder_level = manual_reorder
        else:
            reorder_level = calculated_reorder

        target_stock = (
            daily_demand * 30
            + safety_stock
        )

        order_quantity = max(
            0,
            target_stock - stock
        )

        # -------------------------------------------------
        # THIS IS THE FIXED CONDITION
        # -------------------------------------------------
        if stock <= 0:
            status = "STOCKOUT"

        elif stock <= reorder_level:
            status = "REORDER"

        elif stock > target_stock * 1.25:
            status = "OVERSTOCK"

        else:
            status = "HEALTHY"

        mae = 0

        if metrics:
            mae = metrics[0]

        output.append({

            "Product ID": product,

            "Product Name":
                row.get(
                    "Product Name",
                    product
                ),

            "Category":
                row.get(
                    "Category",
                    "Other"
                ),

            "Current Stock":
                round(stock, 2),

            "Forecast Demand (30 Days)":
                round(forecast, 2),

            "Safety Stock":
                round(safety_stock, 2),

            "Reorder Level":
                round(reorder_level, 2),

            "Target Stock":
                round(target_stock, 2),

            "Order Qty (Recommended)":
                round(order_quantity, 2),

            "Status":
                status,

            "Unit Cost":
                round(cost, 2),

            "Inventory Value":
                round(stock * cost, 2),

            "Lead Time Days":
                lead_time,

            "MAE":
                round(mae, 2)
        })

    return pd.DataFrame(output)


def metric_card(title, value, sub):

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            <div class="card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown(
        "## 📦 Retail Inventory Intelligence"
    )

    st.caption(
        "Demand Forecasting • Inventory Management • Reorder Planning"
    )

    st.divider()

    st.markdown("### DATA & SETTINGS")

    uploaded = st.file_uploader(
        "Upload Sales / Historical CSV",
        type=["csv"]
    )

    if uploaded:

        try:

            raw = pd.read_csv(uploaded)

            sales = prepare_sales(raw)

            if sales is None:

                st.error(
                    "CSV must contain Date, Product and Demand columns."
                )

            else:

                st.session_state.sales = sales

                if st.session_state.inventory.empty:

                    st.session_state.inventory = (
                        create_inventory(sales)
                    )

                st.success(
                    "CSV loaded successfully."
                )

        except Exception as e:

            st.error(
                f"CSV Error: {e}"
            )

    st.markdown("### MODEL SETTINGS")

    trees = st.selectbox(
        "Number of Trees",
        [50, 100, 200, 300, 500],
        index=2
    )

    safety = st.number_input(
        "Default Safety Stock (%)",
        min_value=0,
        max_value=100,
        value=15
    )

    st.divider()

    if st.session_state.sales is not None:

        s = st.session_state.sales

        st.markdown("### DATA SUMMARY")

        st.write(
            f"**Total Records:** {len(s):,}"
        )

        st.write(
            f"**Products:** {s['product_id'].nunique()}"
        )

        st.write(
            f"**Categories:** {s['category'].nunique()}"
        )

        st.write(
            f"**Date Range:** "
            f"{s['date'].min().date()} → "
            f"{s['date'].max().date()}"
        )


# =========================================================
# HEADER
# =========================================================
st.markdown(
    "# 📦 Retail Inventory Intelligence"
)

st.caption(
    "Demand Forecasting • Manual Inventory Management • Reorder Planning"
)

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
# MANUAL INVENTORY
# =========================================================
with tabs[1]:

    st.markdown(
        "## ✏️ Manual Inventory"
    )

    st.write(
        "Enter your current inventory manually below."
    )

    if st.session_state.inventory.empty:

        st.session_state.inventory = pd.DataFrame([{
            "Product ID": "P001",
            "Product Name": "Product 1",
            "Category": "Other",
            "Current Stock": 0,
            "Reorder Level": 0,
            "Safety Stock %": safety,
            "Unit Cost": 0,
            "Lead Time Days": 7
        }])

    edited = st.data_editor(
        st.session_state.inventory,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={

            "Product ID":
                st.column_config.TextColumn(
                    "Product ID"
                ),

            "Product Name":
                st.column_config.TextColumn(
                    "Product Name"
                ),

            "Category":
                st.column_config.TextColumn(
                    "Category"
                ),

            "Current Stock":
                st.column_config.NumberColumn(
                    "Current Stock",
                    min_value=0
                ),

            "Reorder Level":
                st.column_config.NumberColumn(
                    "Reorder Level",
                    min_value=0
                ),

            "Safety Stock %":
                st.column_config.NumberColumn(
                    "Safety Stock %",
                    min_value=0,
                    max_value=100
                ),

            "Unit Cost":
                st.column_config.NumberColumn(
                    "Unit Cost ₹",
                    min_value=0
                ),

            "Lead Time Days":
                st.column_config.NumberColumn(
                    "Lead Time Days",
                    min_value=0
                )
        }
    )

    st.markdown("### Save your inventory")

    if st.button(
        "💾 SAVE MANUAL INVENTORY",
        width="stretch"
    ):

        st.session_state.inventory = (
            edited.copy()
        )

        st.session_state.saved = True

        st.success(
            "Inventory saved successfully!"
        )

    st.info(
        "The values entered here become the current stock "
        "used by the inventory planning system."
    )


# =========================================================
# CALCULATE RESULTS
# =========================================================
results = pd.DataFrame()

if (
    st.session_state.sales is not None
    and not st.session_state.inventory.empty
):

    try:

        results = inventory_calculation(
            st.session_state.sales,
            st.session_state.inventory,
            trees
        )

    except Exception as e:

        st.error(
            f"Calculation Error: {e}"
        )


# =========================================================
# DASHBOARD
# =========================================================
with tabs[0]:

    st.markdown("## Dashboard")

    if st.session_state.sales is None:

        st.info(
            "Upload a sales CSV to start."
        )

    elif results.empty:

        st.info(
            "Go to Manual Inventory and enter your stock."
        )

    else:

        total_products = len(results)

        total_stock = results[
            "Current Stock"
        ].sum()

        total_reorder = results[
            "Reorder Level"
        ].sum()

        total_value = results[
            "Inventory Value"
        ].sum()

        avg_mae = results[
            "MAE"
        ].mean()

        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            metric_card(
                "Total Products",
                f"{total_products:,}",
                "Active Products"
            )

        with c2:
            metric_card(
                "Current Stock",
                f"{total_stock:,.0f}",
                "Units"
            )

        with c3:
            metric_card(
                "Reorder Level",
                f"{total_reorder:,.0f}",
                "Units"
            )

        with c4:
            metric_card(
                "MAE",
                f"{avg_mae:.2f}",
                "Lower is better"
            )

        with c5:
            metric_card(
                "Inventory Value",
                f"₹{total_value:,.0f}",
                "Current stock value"
            )

        st.markdown("### Inventory Status")

        counts = results[
            "Status"
        ].value_counts()

        a, b, c, d = st.columns(4)

        with a:
            metric_card(
                "🔴 Stockout",
                counts.get("STOCKOUT", 0),
                "Products"
            )

        with b:
            metric_card(
                "🟡 Reorder",
                counts.get("REORDER", 0),
                "Products"
            )

        with c:
            metric_card(
                "🔵 Overstock",
                counts.get("OVERSTOCK", 0),
                "Products"
            )

        with d:
            metric_card(
                "🟢 Healthy",
                counts.get("HEALTHY", 0),
                "Products"
            )

        st.markdown(
            "### Inventory Status Overview"
        )

        cols = [
            "Product ID",
            "Product Name",
            "Category",
            "Current Stock",
            "Forecast Demand (30 Days)",
            "Safety Stock",
            "Reorder Level",
            "Target Stock",
            "Order Qty (Recommended)",
            "Status",
            "Inventory Value"
        ]

        st.dataframe(
            results[cols],
            width="stretch",
            hide_index=True
        )


# =========================================================
# INVENTORY PLANNING
# =========================================================
with tabs[2]:

    st.markdown(
        "## 📦 Inventory Planning"
    )

    if results.empty:

        st.info(
            "Enter inventory manually first."
        )

    else:

        st.dataframe(
            results.sort_values(
                "Order Qty (Recommended)",
                ascending=False
            ),
            width="stretch",
            hide_index=True
        )

        purchase = results[
            results["Order Qty (Recommended)"] > 0
        ].copy()

        purchase[
            "Estimated Purchase Cost"
        ] = (
            purchase[
                "Order Qty (Recommended)"
            ]
            * purchase["Unit Cost"]
        )

        st.markdown(
            "### Recommended Purchases"
        )

        st.dataframe(
            purchase,
            width="stretch",
            hide_index=True
        )


# =========================================================
# FORECAST MODEL
# =========================================================
with tabs[3]:

    st.markdown(
        "## 🤖 Forecast Model"
    )

    if st.session_state.sales is None:

        st.info(
            "Upload sales data first."
        )

    else:

        metrics = []

        for product in (
            st.session_state.sales[
                "product_id"
            ].unique()
        ):

            forecast, actual, predicted, metric = (
                forecast_product(
                    st.session_state.sales,
                    product,
                    trees
                )
            )

            if metric:

                mae, rmse = metric

                metrics.append({
                    "Product ID": product,
                    "MAE": round(mae, 2),
                    "RMSE": round(rmse, 2),
                    "30-Day Forecast":
                        round(forecast, 2)
                })

        if metrics:

            st.dataframe(
                pd.DataFrame(metrics),
                width="stretch",
                hide_index=True
            )


# =========================================================
# ALERTS
# =========================================================
with tabs[4]:

    st.markdown(
        "## 🚨 Alerts"
    )

    if results.empty:

        st.info(
            "No alerts available."
        )

    else:

        stockout = results[
            results["Status"] == "STOCKOUT"
        ]

        reorder = results[
            results["Status"] == "REORDER"
        ]

        overstock = results[
            results["Status"] == "OVERSTOCK"
        ]

        if not stockout.empty:

            st.error(
                f"🔴 {len(stockout)} product(s) "
                "are out of stock."
            )

            st.dataframe(
                stockout,
                width="stretch",
                hide_index=True
            )

        if not reorder.empty:

            st.warning(
                f"🟡 {len(reorder)} product(s) "
                "need reordering."
            )

            st.dataframe(
                reorder,
                width="stretch",
                hide_index=True
            )

        if not overstock.empty:

            st.info(
                f"🔵 {len(overstock)} product(s) "
                "are overstocked."
            )

            st.dataframe(
                overstock,
                width="stretch",
                hide_index=True
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

    st.markdown(
        "## 📊 Analytics"
    )

    if st.session_state.sales is None:

        st.info(
            "Upload sales data first."
        )

    else:

        category_demand = (
            st.session_state.sales
            .groupby("category")["demand"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.bar(
            category_demand.index,
            category_demand.values
        )

        ax.set_ylabel(
            "Units Sold"
        )

        ax.tick_params(
            axis="x",
            rotation=30
        )

        ax.grid(
            axis="y",
            alpha=0.2
        )

        st.pyplot(fig)

        plt.close(fig)

        if not results.empty:

            inventory_category = (
                results
                .groupby("Category")[
                    "Inventory Value"
                ]
                .sum()
                .sort_values(
                    ascending=False
                )
            )

            st.markdown(
                "### Inventory Value by Category"
            )

            fig, ax = plt.subplots(
                figsize=(10, 4)
            )

            ax.barh(
                inventory_category.index,
                inventory_category.values
            )

            ax.set_xlabel(
                "Inventory Value ₹"
            )

            ax.grid(
                axis="x",
                alpha=0.2
            )

            st.pyplot(fig)

            plt.close(fig)


# =========================================================
# 12 MONTH FORECAST
# =========================================================
with tabs[6]:

    st.markdown(
        "## 🔮 12-Month Forecast"
    )

    if st.session_state.sales is None:

        st.info(
            "Upload sales data first."
        )

    else:

        products = (
            st.session_state.sales[
                "product_id"
            ]
            .unique()
            .tolist()
        )

        selected = st.selectbox(
            "Select Product",
            products
        )

        monthly = (
            st.session_state.sales[
                st.session_state.sales[
                    "product_id"
                ] == selected
            ]
            .set_index("date")["demand"]
            .resample("ME")
            .sum()
        )

        if len(monthly) >= 6:

            x = np.arange(
                len(monthly)
            )

            coefficient = np.polyfit(
                x,
                monthly.values,
                1
            )

            future_x = np.arange(
                len(monthly),
                len(monthly) + 12
            )

            future = np.polyval(
                coefficient,
                future_x
            )

            future = np.maximum(
                future,
                0
            )

            future_dates = pd.date_range(
                monthly.index[-1]
                + pd.offsets.MonthEnd(1),
                periods=12,
                freq="ME"
            )

            forecast_df = pd.DataFrame({
                "Month": future_dates,
                "Forecast Demand":
                    np.round(future, 2)
            })

            st.dataframe(
                forecast_df,
                width="stretch",
                hide_index=True
            )

            fig, ax = plt.subplots(
                figsize=(12, 4)
            )

            ax.plot(
                monthly.index,
                monthly.values,
                label="Historical"
            )

            ax.plot(
                future_dates,
                future,
                linestyle="--",
                label="Forecast"
            )

            ax.set_ylabel(
                "Units Sold"
            )

            ax.grid(
                alpha=0.2
            )

            ax.legend()

            st.pyplot(fig)

            plt.close(fig)

        else:

            st.warning(
                "At least 6 months of historical "
                "data are recommended."
            )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "Retail Inventory Intelligence | "
    "Manual inventory + demand forecasting + reorder planning"
)
