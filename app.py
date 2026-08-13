import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Retail Inventory Intelligence",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #000000;
    color: #ffffff;
}
[data-testid="stSidebar"] {
    background: #050505;
    border-right: 1px solid #333333;
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}
p, label, li {
    color: #d8d8d8;
}
span {
    color: inherit;
}
[data-testid="stMetric"] {
    background: #111111;
    border: 1px solid #333333;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricLabel"] {
    color: #bdbdbd !important;
}
[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-weight: 700;
}
.stTabs [data-baseweb="tab-list"] {
    background: #080808;
    border-bottom: 1px solid #333333;
}
.stTabs [data-baseweb="tab"] {
    color: #aaaaaa !important;
    background: #080808;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    font-weight: 700;
}
[data-testid="stDataFrame"] {
    background: #111111;
    border: 1px solid #333333;
}
[data-testid="stDataFrame"] * {
    color: #ffffff !important;
}
[data-testid="stFileUploader"] {
    background: #111111;
    border: 1px solid #333333;
    border-radius: 10px;
}
[data-testid="stFileUploader"] * {
    color: #ffffff !important;
}
button {
    background: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #555555 !important;
}
button:hover {
    background: #222222 !important;
    border-color: #ffffff !important;
}
input, textarea {
    background: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #444444 !important;
}
div[data-baseweb="select"] > div {
    background: #111111 !important;
    color: #ffffff !important;
    border-color: #444444 !important;
}
div[data-baseweb="select"] * {
    color: #ffffff !important;
}
[data-testid="stAlert"] {
    background: #111111;
    border: 1px solid #444444;
}
[data-testid="stAlert"] * {
    color: #ffffff !important;
}
hr {
    border-color: #333333;
}
small {
    color: #999999 !important;
}
</style>
""", unsafe_allow_html=True)

REQUIRED = [
    "Date","Store ID","Product ID","Category","Region",
    "Inventory Level","Units Sold","Units Ordered","Demand Forecast",
    "Price","Discount","Weather Condition","Holiday/Promotion",
    "Competitor Pricing","Seasonality"
]

st.title("📦 Retail Inventory Intelligence")
st.caption("Demand Forecasting • Manual Inventory Management • Reorder Planning")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader(
        "Upload sales/inventory CSV",
        type=["csv"]
    )
    st.divider()
    st.write("**Model:** Random Forest")
    st.write("**Trees:** 200")
    st.write("**Safety Stock:** 15%")
    st.divider()
    st.info(
        "Inventory values can now be entered manually inside the "
        "Inventory Manager. The CSV is used mainly for historical demand."
    )

if uploaded is None:
    st.warning("Upload the retail CSV to start.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read the CSV: {e}")
    st.stop()

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error("The CSV is missing these columns:")
    st.write(missing)
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_cols = [
    "Inventory Level","Units Sold","Units Ordered","Demand Forecast",
    "Price","Discount","Holiday/Promotion","Competitor Pricing"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=["Date","Store ID","Product ID","Units Sold"]
).copy()

df = df.sort_values(["Store ID","Product ID","Date"])

if "inventory_master" not in st.session_state:
    master = (
        df.sort_values("Date")
        .groupby(["Store ID","Product ID"], as_index=False)
        .tail(1)
        .copy()
    )

    inventory_master = master[
        ["Store ID","Product ID","Category","Region"]
    ].copy()

    inventory_master["Current Stock"] = master["Inventory Level"].fillna(0).astype(float)
    inventory_master["Unit Cost"] = master["Price"].fillna(0).astype(float)
    inventory_master["Lead Time Days"] = 7
    inventory_master["Supplier"] = "Supplier A"
    inventory_master["Reorder Level"] = np.maximum(
        master["Demand Forecast"].fillna(0).astype(float), 0
    )

    st.session_state.inventory_master = inventory_master

inventory_master = st.session_state.inventory_master.copy()

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

df["Lag_1"] = (
    df.groupby(["Store ID","Product ID"])["Units Sold"]
    .shift(1)
)

df["Lag_7"] = (
    df.groupby(["Store ID","Product ID"])["Units Sold"]
    .shift(7)
)

df["Rolling_7"] = (
    df.groupby(["Store ID","Product ID"])["Units Sold"]
    .transform(
        lambda x: x.shift(1).rolling(7, min_periods=3).mean()
    )
)

df["Rolling_30"] = (
    df.groupby(["Store ID","Product ID"])["Units Sold"]
    .transform(
        lambda x: x.shift(1).rolling(30, min_periods=7).mean()
    )
)

model_df = df.dropna(
    subset=["Lag_1","Lag_7","Rolling_7","Rolling_30"]
).copy()

categorical = [
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality"
]

model_df = pd.get_dummies(
    model_df,
    columns=categorical,
    drop_first=True
)

base_features = [
    "Inventory Level",
    "Units Ordered",
    "Demand Forecast",
    "Price",
    "Discount",
    "Holiday/Promotion",
    "Competitor Pricing",
    "Year",
    "Month",
    "DayOfWeek",
    "Lag_1",
    "Lag_7",
    "Rolling_7",
    "Rolling_30"
]

dummy_features = [
    c for c in model_df.columns
    if any(c.startswith(x + "_") for x in categorical)
]

features = base_features + dummy_features

model_df = model_df.dropna(
    subset=features + ["Units Sold"]
)

if len(model_df) < 20:
    st.error("Not enough valid historical records to train the model.")
    st.stop()

dates = sorted(
    model_df["Date"].dt.normalize().unique()
)

split_index = max(1, int(len(dates) * 0.8))

if split_index >= len(dates):
    split_index = len(dates) - 1

split_date = dates[split_index]

train = model_df[
    model_df["Date"].dt.normalize() < split_date
]

test = model_df[
    model_df["Date"].dt.normalize() >= split_date
]

if train.empty or test.empty:
    st.error("Could not create the time-based train/test split.")
    st.stop()

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

model.fit(
    train[features],
    train["Units Sold"]
)

pred = np.maximum(
    model.predict(test[features]),
    0
)

actual = test["Units Sold"].to_numpy()

mae = mean_absolute_error(
    actual,
    pred
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        pred
    )
)

mask = actual != 0

mape = (
    np.mean(
        np.abs(
            (actual[mask] - pred[mask])
            / actual[mask]
        )
    ) * 100
    if mask.any()
    else np.nan
)

latest_date = df["Date"].max()

tabs = st.tabs([
    "🏠 Dashboard",
    "✏️ Manual Inventory",
    "📦 Inventory Planning",
    "🤖 Forecast Model",
    "🚨 Alerts",
    "📊 Analytics",
    "🔮 12-Month Forecast"
])

with tabs[0]:
    st.header("Executive Dashboard")

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric(
        "Products",
        f"{inventory_master['Product ID'].nunique():,}"
    )

    c2.metric(
        "Current Stock",
        f"{inventory_master['Current Stock'].sum():,.0f}"
    )

    c3.metric(
        "Reorder Level",
        f"{inventory_master['Reorder Level'].sum():,.0f}"
    )

    c4.metric(
        "MAE",
        f"{mae:,.2f}"
    )

    c5.metric(
        "MAPE",
        f"{mape:.2f}%" if np.isfinite(mape) else "N/A"
    )

    st.info(
        f"Historical data loaded through {latest_date:%d %b %Y}. "
        "Inventory values shown here come from your manual inventory table."
    )

    status_df = inventory_master.copy()

    status_df["Safety Stock"] = (
        status_df["Reorder Level"] * 0.15
    )

    status_df["Target Stock"] = (
        status_df["Reorder Level"]
        + status_df["Safety Stock"]
    )

    status_df["Order Qty"] = np.maximum(
        np.ceil(
            status_df["Target Stock"]
            - status_df["Current Stock"]
        ),
        0
    )

    status_df["Status"] = np.select(
        [
            status_df["Current Stock"] <= 0,
            status_df["Current Stock"] < status_df["Reorder Level"],
            status_df["Current Stock"] >
            status_df["Target Stock"] * 1.5
        ],
        [
            "Stockout",
            "Reorder",
            "Overstock"
        ],
        default="Healthy"
    )

    a,b,c,d = st.columns(4)

    a.metric(
        "🔴 Stockout",
        f"{(status_df['Status']=='Stockout').sum():,}"
    )

    b.metric(
        "🟠 Reorder",
        f"{(status_df['Status']=='Reorder').sum():,}"
    )

    c.metric(
        "🔵 Overstock",
        f"{(status_df['Status']=='Overstock').sum():,}"
    )

    d.metric(
        "🟢 Healthy",
        f"{(status_df['Status']=='Healthy').sum():,}"
    )

with tabs[1]:
    st.header("✏️ Manual Inventory Manager")

    st.write(
        "This is the manual inventory feature we discussed. "
        "You can directly enter or edit the operational inventory information."
    )

    st.subheader("Inventory Master")

    edited = st.data_editor(
        inventory_master,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Store ID": st.column_config.TextColumn(
                "Store ID",
                disabled=True
            ),
            "Product ID": st.column_config.TextColumn(
                "Product ID",
                disabled=True
            ),
            "Category": st.column_config.TextColumn(
                "Category",
                disabled=True
            ),
            "Region": st.column_config.TextColumn(
                "Region",
                disabled=True
            ),
            "Current Stock": st.column_config.NumberColumn(
                "Current Stock",
                min_value=0,
                step=1
            ),
            "Unit Cost": st.column_config.NumberColumn(
                "Unit Cost",
                min_value=0,
                step=0.01
            ),
            "Lead Time Days": st.column_config.NumberColumn(
                "Lead Time Days",
                min_value=0,
                step=1
            ),
            "Supplier": st.column_config.TextColumn(
                "Supplier"
            ),
            "Reorder Level": st.column_config.NumberColumn(
                "Reorder Level",
                min_value=0,
                step=1
            )
        }
    )

    if st.button(
        "💾 Save Inventory Changes",
        type="primary"
    ):
        edited["Current Stock"] = pd.to_numeric(
            edited["Current Stock"],
            errors="coerce"
        ).fillna(0)

        edited["Unit Cost"] = pd.to_numeric(
            edited["Unit Cost"],
            errors="coerce"
        ).fillna(0)

        edited["Lead Time Days"] = pd.to_numeric(
            edited["Lead Time Days"],
            errors="coerce"
        ).fillna(0)

        edited["Reorder Level"] = pd.to_numeric(
            edited["Reorder Level"],
            errors="coerce"
        ).fillna(0)

        edited["Supplier"] = (
            edited["Supplier"]
            .fillna("Supplier A")
            .astype(str)
        )

        st.session_state.inventory_master = edited.copy()
        st.success("Inventory information saved for this session.")
        st.rerun()

    st.divider()

    st.subheader("➕ Add Manual Product")

    c1,c2,c3 = st.columns(3)

    with c1:
        new_store = st.text_input(
            "Store ID",
            value=""
        )

        new_product = st.text_input(
            "Product ID",
            value=""
        )

    with c2:
        new_category = st.text_input(
            "Category",
            value="General"
        )

        new_region = st.text_input(
            "Region",
            value="General"
        )

    with c3:
        new_stock = st.number_input(
            "Current Stock",
            min_value=0.0,
            value=0.0
        )

        new_cost = st.number_input(
            "Unit Cost",
            min_value=0.0,
            value=0.0
        )

    c4,c5,c6 = st.columns(3)

    with c4:
        new_lead = st.number_input(
            "Lead Time Days",
            min_value=0,
            value=7
        )

    with c5:
        new_supplier = st.text_input(
            "Supplier",
            value="Supplier A"
        )

    with c6:
        new_reorder = st.number_input(
            "Reorder Level",
            min_value=0.0,
            value=0.0
        )

    if st.button(
        "➕ Add Product"
    ):
        if not new_store or not new_product:
            st.error("Enter both Store ID and Product ID.")
        else:
            new_row = pd.DataFrame([{
                "Store ID": new_store,
                "Product ID": new_product,
                "Category": new_category,
                "Region": new_region,
                "Current Stock": new_stock,
                "Unit Cost": new_cost,
                "Lead Time Days": new_lead,
                "Supplier": new_supplier,
                "Reorder Level": new_reorder
            }])

            st.session_state.inventory_master = pd.concat(
                [
                    st.session_state.inventory_master,
                    new_row
                ],
                ignore_index=True
            )

            st.success("Product added.")
            st.rerun()

    st.divider()

    st.subheader("💾 Save Inventory as CSV")

    st.download_button(
        "⬇ Download Manual Inventory Master",
        st.session_state.inventory_master.to_csv(
            index=False
        ).encode("utf-8"),
        "inventory_master.csv",
        "text/csv"
    )

with tabs[2]:
    st.header("📦 Inventory Planning")

    plan = inventory_master.copy()

    plan["Safety Stock"] = (
        plan["Reorder Level"] * 0.15
    )

    plan["Target Stock"] = (
        plan["Reorder Level"]
        + plan["Safety Stock"]
    )

    plan["Order Quantity"] = np.maximum(
        np.ceil(
            plan["Target Stock"]
            - plan["Current Stock"]
        ),
        0
    )

    plan["Inventory Value"] = (
        plan["Current Stock"]
        * plan["Unit Cost"]
    )

    plan["Status"] = np.select(
        [
            plan["Current Stock"] <= 0,
            plan["Current Stock"] < plan["Reorder Level"],
            plan["Current Stock"] > plan["Target Stock"] * 1.5
        ],
        [
            "Stockout",
            "Reorder",
            "Overstock"
        ],
        default="Healthy"
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Inventory Value",
        f"{plan['Inventory Value'].sum():,.2f}"
    )

    c2.metric(
        "Safety Stock",
        f"{plan['Safety Stock'].sum():,.0f}"
    )

    c3.metric(
        "Order Quantity",
        f"{plan['Order Quantity'].sum():,.0f}"
    )

    c4.metric(
        "Lead Time Avg.",
        f"{plan['Lead Time Days'].mean():.1f} days"
    )

    plan_display_cols = [
        "Store ID", "Product ID", "Category", "Region",
        "Current Stock", "Unit Cost", "Lead Time Days",
        "Supplier", "Reorder Level", "Safety Stock",
        "Target Stock", "Order Quantity", "Inventory Value", "Status"
    ]
    plan_display_cols = [
        col for col in plan_display_cols
        if col in plan.columns
    ]

    st.dataframe(
        plan[plan_display_cols],
        width="stretch",
        hide_index=True
    )

    st.download_button(
        "⬇ Download Inventory Plan",
        plan.to_csv(index=False).encode("utf-8"),
        "inventory_plan.csv",
        "text/csv"
    )

with tabs[3]:
    st.header("🤖 Forecast Model")

    c1,c2,c3 = st.columns(3)

    c1.metric("MAE", f"{mae:,.2f}")
    c2.metric("RMSE", f"{rmse:,.2f}")
    c3.metric(
        "MAPE",
        f"{mape:.2f}%" if np.isfinite(mape) else "N/A"
    )

    st.write(
        "The model predicts future Units Sold using historical demand, "
        "lag values, rolling demand, pricing, promotion and seasonality."
    )

    n = min(500, len(test))

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        test["Date"].iloc[:n],
        actual[:n],
        label="Actual"
    )

    ax.plot(
        test["Date"].iloc[:n],
        pred[:n],
        label="Predicted",
        linestyle="--"
    )

    ax.set_title("Actual vs Predicted Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    ).head(15)

    fig, ax = plt.subplots(figsize=(10,5))

    ax.barh(
        importance["Feature"],
        importance["Importance"]
    )

    ax.invert_yaxis()
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

with tabs[4]:
    st.header("🚨 Inventory Alerts")

    plan = inventory_master.copy()

    plan["Safety Stock"] = (
        plan["Reorder Level"] * 0.15
    )

    plan["Target Stock"] = (
        plan["Reorder Level"]
        + plan["Safety Stock"]
    )

    plan["Order Quantity"] = np.maximum(
        np.ceil(
            plan["Target Stock"]
            - plan["Current Stock"]
        ),
        0
    )

    plan["Status"] = np.select(
        [
            plan["Current Stock"] <= 0,
            plan["Current Stock"] < plan["Reorder Level"],
            plan["Current Stock"] > plan["Target Stock"] * 1.5
        ],
        [
            "Stockout",
            "Reorder",
            "Overstock"
        ],
        default="Healthy"
    )

    a,b,c = st.columns(3)

    a.metric(
        "🔴 Stockout",
        int((plan["Status"] == "Stockout").sum())
    )

    b.metric(
        "🟠 Reorder",
        int((plan["Status"] == "Reorder").sum())
    )

    c.metric(
        "🔵 Overstock",
        int((plan["Status"] == "Overstock").sum())
    )

    alerts = plan[
        plan["Status"] != "Healthy"
    ]

    if alerts.empty:
        st.success("No inventory alerts.")
    else:
        alert_display_cols = [
            "Store ID", "Product ID", "Category", "Region",
            "Current Stock", "Reorder Level", "Safety Stock",
            "Target Stock", "Order Quantity", "Status"
        ]
        alert_display_cols = [
            col for col in alert_display_cols
            if col in alerts.columns
        ]

        st.dataframe(
            alerts[alert_display_cols],
            width="stretch",
            hide_index=True
        )

with tabs[5]:
    st.header("📊 Retail Analytics")

    c1,c2 = st.columns(2)

    with c1:
        category = (
            df.groupby("Category")["Units Sold"]
            .sum()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(8,4))

        ax.bar(
            category.index,
            category.values
        )

        ax.set_title("Units Sold by Category")
        ax.tick_params(
            axis="x",
            rotation=30
        )
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(
            fig,
            width="stretch"
        )

        plt.close(fig)

    with c2:
        region = (
            df.groupby("Region")["Units Sold"]
            .sum()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(8,4))

        ax.bar(
            region.index,
            region.values
        )

        ax.set_title("Units Sold by Region")
        ax.tick_params(
            axis="x",
            rotation=30
        )
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(
            fig,
            width="stretch"
        )

        plt.close(fig)

    monthly = (
        df.groupby(
            df["Date"].dt.to_period("M")
        )["Units Sold"]
        .sum()
    )

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(
        monthly.index.astype(str),
        monthly.values
    )

    ax.set_title("Historical Monthly Demand")
    ax.tick_params(
        axis="x",
        rotation=45
    )
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(
        fig,
        width="stretch"
    )

    plt.close(fig)

with tabs[6]:
    st.header("🔮 12-Month Forecast")

    st.caption(
        "Recursive product/store demand projection using the trained Random Forest model."
    )

    # Build a stable 30-observation history for every Store + Product pair.
    # This is intentionally done from the original cleaned dataframe so that
    # the forecast does not depend on the training/test split.
    group_histories = {}
    skipped_groups = []

    grouped = df.sort_values("Date").groupby(
        ["Store ID", "Product ID"],
        sort=False
    )

    for (store, product), group in grouped:
        group = group.sort_values("Date").tail(30).copy()
        history = pd.to_numeric(
            group["Units Sold"],
            errors="coerce"
        ).dropna().tolist()

        if len(history) < 7:
            skipped_groups.append((store, product))
            continue

        group_histories[(store, product)] = group

    future_rows = []
    forecast_errors = []

    # Forecast one step at a time and feed each prediction back into history.
    for (store, product), group in group_histories.items():
        group = group.sort_values("Date")
        history = pd.to_numeric(
            group["Units Sold"],
            errors="coerce"
        ).dropna().tolist()
        last = group.iloc[-1].copy()

        for step in range(1, 13):
            next_date = latest_date + pd.DateOffset(months=step)

            try:
                row = {
                    "Inventory Level": float(
                        pd.to_numeric(last["Inventory Level"], errors="coerce") or 0
                    ),
                    "Units Ordered": float(
                        pd.to_numeric(last["Units Ordered"], errors="coerce") or 0
                    ),
                    "Demand Forecast": float(
                        pd.to_numeric(last["Demand Forecast"], errors="coerce") or 0
                    ),
                    "Price": float(
                        pd.to_numeric(last["Price"], errors="coerce") or 0
                    ),
                    "Discount": float(
                        pd.to_numeric(last["Discount"], errors="coerce") or 0
                    ),
                    "Holiday/Promotion": float(
                        pd.to_numeric(last["Holiday/Promotion"], errors="coerce") or 0
                    ),
                    "Competitor Pricing": float(
                        pd.to_numeric(last["Competitor Pricing"], errors="coerce") or 0
                    ),
                    "Year": next_date.year,
                    "Month": next_date.month,
                    "DayOfWeek": next_date.dayofweek,
                    "Lag_1": float(history[-1]),
                    "Lag_7": float(history[-7]),
                    "Rolling_7": float(np.mean(history[-7:])),
                    "Rolling_30": float(np.mean(history[-30:]))
                }

                # Initialize every encoded feature to zero.
                for feature in dummy_features:
                    row[feature] = 0

                # Restore the categorical values from the latest known record.
                for prefix, column in [
                    ("Category", "Category"),
                    ("Region", "Region"),
                    ("Weather Condition", "Weather Condition"),
                    ("Seasonality", "Seasonality")
                ]:
                    value = str(last[column])
                    key = f"{prefix}_{value}"
                    if key in row:
                        row[key] = 1

                # Guarantee the exact feature order expected by the model.
                future_X = pd.DataFrame([row]).reindex(
                    columns=features,
                    fill_value=0
                )

                forecast = float(model.predict(future_X)[0])
                forecast = max(forecast, 0.0)

                safety = forecast * 0.15

                future_rows.append({
                    "Date": next_date,
                    "Month": next_date.strftime("%b %Y"),
                    "Store ID": store,
                    "Product ID": product,
                    "Category": last["Category"],
                    "Predicted Demand": forecast,
                    "Safety Stock": safety,
                    "Recommended Inventory": forecast + safety
                })

                # Feed the prediction into the next recursive step.
                history.append(forecast)

            except Exception as e:
                forecast_errors.append(
                    f"{store} / {product} / step {step}: {str(e)}"
                )
                break

    future = pd.DataFrame(future_rows)

    if future.empty:
        st.error(
            "The 12-month forecast could not be generated. "
            "Check that each Store/Product pair has at least 7 valid Units Sold records."
        )

        if skipped_groups:
            st.info(
                f"Skipped {len(skipped_groups)} Store/Product pairs because they "
                "did not have enough historical demand records."
            )

        if forecast_errors:
            st.warning(
                "Forecast errors were encountered. The first error was: "
                + forecast_errors[0]
            )

    else:
        # Aggregate all Store/Product predictions into one monthly business forecast.
        monthly_future = (
            future.groupby(
                ["Date", "Month"],
                as_index=False
            )
            .agg({
                "Predicted Demand": "sum",
                "Safety Stock": "sum",
                "Recommended Inventory": "sum"
            })
            .sort_values("Date")
        )

        # Guarantee that all 12 forecast months are represented.
        expected_dates = [
            latest_date + pd.DateOffset(months=i)
            for i in range(1, 13)
        ]

        expected = pd.DataFrame({
            "Date": expected_dates,
            "Month": [d.strftime("%b %Y") for d in expected_dates]
        })

        monthly_future = expected.merge(
            monthly_future,
            on=["Date", "Month"],
            how="left"
        )

        for col in [
            "Predicted Demand",
            "Safety Stock",
            "Recommended Inventory"
        ]:
            monthly_future[col] = (
                pd.to_numeric(monthly_future[col], errors="coerce")
                .fillna(0)
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "12-Month Demand",
            f"{monthly_future['Predicted Demand'].sum():,.0f}"
        )

        c2.metric(
            "12-Month Safety Stock",
            f"{monthly_future['Safety Stock'].sum():,.0f}"
        )

        c3.metric(
            "Recommended Inventory",
            f"{monthly_future['Recommended Inventory'].sum():,.0f}"
        )

        st.success(
            f"Generated {len(monthly_future)} forecast months for "
            f"{len(group_histories)} Store/Product combinations."
        )

        # Use only columns that are guaranteed to exist to prevent KeyError.
        forecast_display_cols = [
            "Month",
            "Predicted Demand",
            "Safety Stock",
            "Recommended Inventory"
        ]
        forecast_display_cols = [
            col for col in forecast_display_cols
            if col in monthly_future.columns
        ]

        st.dataframe(
            monthly_future[forecast_display_cols],
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "⬇ Download 12-Month Forecast",
            monthly_future.to_csv(index=False).encode("utf-8"),
            "12_month_forecast.csv",
            "text/csv"
        )

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(
            monthly_future["Month"],
            monthly_future["Predicted Demand"],
            marker="o"
        )

        ax.set_title("12-Month Forecasted Demand")
        ax.set_xlabel("Month")
        ax.set_ylabel("Predicted Units")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(fig, width="stretch")
        plt.close(fig)

        # Optional diagnostics, kept out of the main UI unless something went wrong.
        if skipped_groups:
            st.info(
                f"{len(skipped_groups)} Store/Product combinations were skipped "
                "because they had fewer than 7 valid historical demand records."
            )

        if forecast_errors:
            st.warning(
                f"{len(forecast_errors)} forecast steps could not be generated. "
                "The remaining valid forecasts are still shown above."
            )

st.divider()

st.caption(
    "Retail Inventory Intelligence • Random Forest • 200 Estimators • Manual Inventory Master"
)
