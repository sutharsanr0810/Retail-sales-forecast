import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Retail Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp { background:#f7f8fc; color:#1f2937; }
[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e5e7eb; }
[data-testid="stSidebar"] * { color:#1f2937 !important; }
h1,h2,h3,h4 { color:#111827 !important; }
[data-testid="stMetric"] {
    background:#ffffff; border:1px solid #e5e7eb; border-radius:12px;
    padding:16px; box-shadow:0 2px 8px rgba(0,0,0,.04);
}
[data-testid="stMetricLabel"] { color:#6b7280 !important; }
[data-testid="stMetricValue"] { color:#111827 !important; }
.stTabs [data-baseweb="tab-list"] {
    gap:8px; background:#ffffff; padding:8px; border-radius:10px;
    border:1px solid #e5e7eb;
}
.stTabs [data-baseweb="tab"] { color:#374151 !important; }
.stTabs [aria-selected="true"] { color:#111827 !important; font-weight:600; }
hr { border-color:#e5e7eb; }
</style>
""", unsafe_allow_html=True)

st.title("📦 Retail Inventory Intelligence")
st.caption("Demand Forecasting • Inventory Optimization • Stock Risk Analysis")

REQUIRED = [
    "Date","Store ID","Product ID","Category","Region",
    "Inventory Level","Units Sold","Units Ordered","Demand Forecast",
    "Price","Discount","Weather Condition","Holiday/Promotion",
    "Competitor Pricing","Seasonality"
]

with st.sidebar:
    st.header("📦 Retail Intelligence")
    uploaded = st.file_uploader("Upload inventory CSV", type=["csv"])
    st.divider()
    st.write("**Model:** Random Forest")
    st.write("**Trees:** 200")
    st.write("**Forecast target:** Units Sold")
    st.write("**Safety buffer:** 15%")
    st.divider()
    st.caption("Inventory decision-support dashboard")

if uploaded is None:
    default_path = "retail_store_inventory.csv"
    try:
        df = pd.read_csv(default_path)
        st.info("Using retail_store_inventory.csv from the project folder.")
    except Exception:
        st.warning("Upload the retail inventory CSV to begin.")
        st.stop()
else:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error("Missing required columns:")
    st.write(missing)
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
for c in ["Inventory Level","Units Sold","Units Ordered","Demand Forecast",
          "Price","Discount","Holiday/Promotion","Competitor Pricing"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Date","Units Sold","Inventory Level"]).copy()
df = df.sort_values(["Store ID","Product ID","Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Lag_1"] = df.groupby(["Store ID","Product ID"])["Units Sold"].shift(1)
df["Lag_7"] = df.groupby(["Store ID","Product ID"])["Units Sold"].shift(7)
df["Rolling_7"] = (
    df.groupby(["Store ID","Product ID"])["Units Sold"]
      .transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
)
df["Rolling_30"] = (
    df.groupby(["Store ID","Product ID"])["Units Sold"]
      .transform(lambda x: x.shift(1).rolling(30, min_periods=7).mean())
)

model_data = df.dropna(subset=["Lag_1","Lag_7","Rolling_7","Rolling_30"]).copy()

categorical = ["Category","Region","Weather Condition","Seasonality"]
model_data = pd.get_dummies(model_data, columns=categorical, drop_first=True)

base_features = [
    "Inventory Level","Units Ordered","Demand Forecast","Price","Discount",
    "Holiday/Promotion","Competitor Pricing","Year","Month","DayOfWeek",
    "Lag_1","Lag_7","Rolling_7","Rolling_30"
]
dummy_features = [
    c for c in model_data.columns
    if any(c.startswith(x + "_") for x in categorical)
]
features = base_features + dummy_features

model_data = model_data.dropna(subset=features + ["Units Sold"])

unique_dates = sorted(model_data["Date"].dt.normalize().unique())
if len(unique_dates) < 2:
    st.error("Not enough dates for a train/test split.")
    st.stop()

split_date = unique_dates[max(0, int(len(unique_dates) * 0.8) - 1)]
train = model_data[model_data["Date"].dt.normalize() <= split_date]
test = model_data[model_data["Date"].dt.normalize() > split_date]

if train.empty or test.empty:
    st.error("Unable to create a time-based train/test split.")
    st.stop()

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)
model.fit(train[features], train["Units Sold"])
test_pred = np.maximum(model.predict(test[features]), 0)

mae = mean_absolute_error(test["Units Sold"], test_pred)
rmse = np.sqrt(mean_squared_error(test["Units Sold"], test_pred))
actual = test["Units Sold"].to_numpy()
mask = actual != 0
mape = np.mean(np.abs((actual[mask] - test_pred[mask]) / actual[mask])) * 100 if mask.any() else np.nan

df["Stock Coverage Days"] = np.where(
    df["Units Sold"] > 0,
    df["Inventory Level"] / df["Units Sold"],
    np.inf
)
df["Safety Stock"] = np.maximum(df["Demand Forecast"], 0) * 0.15
df["Target Inventory"] = np.maximum(df["Demand Forecast"], 0) + df["Safety Stock"]
df["Order Recommendation"] = np.maximum(
    np.ceil(df["Target Inventory"] - df["Inventory Level"]), 0
)

df["Inventory Status"] = np.select(
    [
        df["Inventory Level"] <= 0,
        df["Inventory Level"] < df["Demand Forecast"],
        df["Inventory Level"] > df["Target Inventory"] * 1.5
    ],
    ["Stockout", "Reorder", "Overstock"],
    default="Healthy"
)

latest_date = df["Date"].max()
latest = (
    df[df["Date"] == latest_date]
    .sort_values(["Store ID","Product ID"])
    .copy()
)

st.header("Executive Dashboard")
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Inventory Units", f"{latest['Inventory Level'].sum():,.0f}")
c2.metric("Units Sold", f"{latest['Units Sold'].sum():,.0f}")
c3.metric("Units Ordered", f"{latest['Units Ordered'].sum():,.0f}")
c4.metric("Reorder Items", f"{(latest['Inventory Status']=='Reorder').sum():,}")
c5.metric("Stockout Items", f"{(latest['Inventory Status']=='Stockout').sum():,}")

st.caption(f"Latest inventory snapshot: {latest_date:%d %b %Y}")

tabs = st.tabs([
    "📦 Inventory",
    "🤖 Forecast Performance",
    "🚨 Alerts",
    "📊 Analytics",
    "🔮 12-Month Forecast"
])

with tabs[0]:
    st.subheader("Current Inventory Position")
    display = latest[[
        "Store ID","Product ID","Category","Region",
        "Inventory Level","Units Sold","Demand Forecast",
        "Safety Stock","Target Inventory","Order Recommendation",
        "Inventory Status"
    ]].copy()
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Download Inventory Plan",
        display.to_csv(index=False).encode("utf-8"),
        "inventory_plan.csv",
        "text/csv"
    )

    st.subheader("Inventory by Category")
    cat = latest.groupby("Category", as_index=False).agg(
        Inventory=("Inventory Level","sum"),
        Demand=("Demand Forecast","sum"),
        Order_Recommendation=("Order Recommendation","sum")
    )
    st.dataframe(cat, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Random Forest Model Performance")
    a,b,c = st.columns(3)
    a.metric("MAE", f"{mae:,.2f}")
    b.metric("RMSE", f"{rmse:,.2f}")
    c.metric("MAPE", f"{mape:.2f}%" if np.isfinite(mape) else "N/A")

    fig, ax = plt.subplots(figsize=(12,5))
    n = min(300, len(test))
    ax.plot(test["Date"].iloc[:n], test["Units Sold"].iloc[:n].to_numpy(), label="Actual")
    ax.plot(test["Date"].iloc[:n], test_pred[:n], label="Predicted", linestyle="--")
    ax.set_title("Actual vs Predicted Demand")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units Sold")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(importance["Feature"], importance["Importance"])
    ax.invert_yaxis()
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tabs[2]:
    st.subheader("Inventory Alerts")
    alerts = latest[latest["Inventory Status"] != "Healthy"].copy()

    x,y,z = st.columns(3)
    x.metric("🔴 Stockout", f"{(latest['Inventory Status']=='Stockout').sum():,}")
    y.metric("🟠 Reorder", f"{(latest['Inventory Status']=='Reorder').sum():,}")
    z.metric("🔵 Overstock", f"{(latest['Inventory Status']=='Overstock').sum():,}")

    if alerts.empty:
        st.success("No inventory alerts in the latest snapshot.")
    else:
        st.dataframe(
            alerts[[
                "Store ID","Product ID","Category","Inventory Level",
                "Demand Forecast","Target Inventory",
                "Order Recommendation","Inventory Status"
            ]],
            use_container_width=True,
            hide_index=True
        )

with tabs[3]:
    st.subheader("Retail Analytics")
    c1,c2 = st.columns(2)

    with c1:
        category_sales = df.groupby("Category")["Units Sold"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(category_sales.index, category_sales.values)
        ax.set_title("Units Sold by Category")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        region_sales = df.groupby("Region")["Units Sold"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(region_sales.index, region_sales.values)
        ax.set_title("Units Sold by Region")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Monthly Demand")
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Units Sold"].sum()
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(monthly.index.astype(str), monthly.values)
    ax.set_title("Historical Monthly Demand")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with tabs[4]:
    st.subheader("12-Month Demand Projection")
    st.caption("Recursive category/store/product projections are generated from the latest observed demand history.")

    horizon = 12
    future_rows = []

    # Forecast at product-store level using a compact recursive feature set.
    # Exogenous variables are held at the latest observed values for each series.
    latest_series = (
        df.sort_values("Date")
          .groupby(["Store ID","Product ID"], as_index=False)
          .tail(30)
    )

    series_groups = latest_series.groupby(["Store ID","Product ID"])
    for (store, product), g in series_groups:
        g = g.sort_values("Date")
        history = g["Units Sold"].tolist()
        if len(history) < 7:
            continue

        static = g.iloc[-1]
        for step in range(1, horizon + 1):
            next_date = latest_date + pd.DateOffset(months=step)
            lag1 = history[-1]
            lag7 = history[-7]
            roll7 = np.mean(history[-7:])
            roll30 = np.mean(history[-30:])

            row = {
                "Inventory Level": max(float(static["Inventory Level"]), 0),
                "Units Ordered": max(float(static["Units Ordered"]), 0),
                "Demand Forecast": max(float(static["Demand Forecast"]), 0),
                "Price": float(static["Price"]),
                "Discount": float(static["Discount"]),
                "Holiday/Promotion": float(static["Holiday/Promotion"]),
                "Competitor Pricing": float(static["Competitor Pricing"]),
                "Year": next_date.year,
                "Month": next_date.month,
                "DayOfWeek": next_date.dayofweek,
                "Lag_1": lag1,
                "Lag_7": lag7,
                "Rolling_7": roll7,
                "Rolling_30": roll30
            }

            for f in dummy_features:
                row[f] = 0

            # Match the known categorical values from the latest row.
            for prefix, column in [
                ("Category", "Category"),
                ("Region", "Region"),
                ("Weather Condition", "Weather Condition"),
                ("Seasonality", "Seasonality")
            ]:
                value = str(static[column])
                key = f"{prefix}_{value}"
                if key in row:
                    row[key] = 1

            X_future = pd.DataFrame([row]).reindex(columns=features, fill_value=0)
            prediction = max(float(model.predict(X_future)[0]), 0)
            safety = prediction * 0.15

            future_rows.append({
                "Month": next_date.strftime("%b %Y"),
                "Date": next_date,
                "Store ID": store,
                "Product ID": product,
                "Category": static["Category"],
                "Predicted Demand": prediction,
                "Safety Stock": safety,
                "Recommended Inventory": prediction + safety
            })
            history.append(prediction)

    future = pd.DataFrame(future_rows)
    if future.empty:
        st.warning("Not enough product history to generate the 12-month projection.")
    else:
        monthly_future = (
            future.groupby(["Date","Month"], as_index=False)
                  .agg(
                      Predicted_Demand=("Predicted Demand","sum"),
                      Safety_Stock=("Safety Stock","sum"),
                      Recommended_Inventory=("Recommended Inventory","sum")
                  )
                  .sort_values("Date")
        )

        q1,q2,q3 = st.columns(3)
        q1.metric("12-Month Demand", f"{monthly_future['Predicted_Demand'].sum():,.0f}")
        q2.metric("12-Month Safety Stock", f"{monthly_future['Safety_Stock'].sum():,.0f}")
        q3.metric("Target Inventory", f"{monthly_future['Recommended_Inventory'].sum():,.0f}")

        st.dataframe(
            monthly_future[[
                "Month","Predicted_Demand","Safety_Stock","Recommended_Inventory"
            ]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download 12-Month Forecast",
            monthly_future.to_csv(index=False).encode("utf-8"),
            "12_month_inventory_forecast.csv",
            "text/csv"
        )

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(monthly_future["Month"], monthly_future["Predicted_Demand"], marker="o")
        ax.set_title("12-Month Forecasted Demand")
        ax.set_xlabel("Month")
        ax.set_ylabel("Predicted Units")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

st.divider()
st.caption("Retail Inventory Intelligence • Random Forest • 200 Estimators")
app (1).py
Displaying app (1).py.
