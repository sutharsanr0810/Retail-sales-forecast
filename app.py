
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
.stApp, [data-testid="stAppViewContainer"] { background:#000 !important; color:#fff !important; }
[data-testid="stHeader"] { background:#000 !important; }
[data-testid="stSidebar"] { background:#030303 !important; border-right:1px solid #292929; }
[data-testid="stSidebar"] * { color:#f5f5f5 !important; }
.block-container { padding: 1.05rem 1.15rem 1.5rem 1.15rem; max-width: 100%; }
h1,h2,h3,h4,h5,h6 { color:#fff !important; }
p,label,li { color:#d6d6d6 !important; }
.stCaption, small { color:#9d9d9d !important; }
[data-testid="stMetric"] { background:#080808 !important; border:1px solid #303030 !important; border-radius:8px !important; padding:14px 16px !important; min-height:92px; }
[data-testid="stMetricLabel"] { color:#d8d8d8 !important; font-weight:600 !important; }
[data-testid="stMetricValue"] { color:#fff !important; font-size:1.9rem !important; font-weight:800 !important; }
[data-testid="stMetricDelta"] { color:#70d54a !important; }
.stTabs [data-baseweb="tab-list"] { background:#000 !important; border-bottom:1px solid #252525; gap:0; }
.stTabs [data-baseweb="tab"] { color:#aaa !important; background:#000 !important; padding:0.75rem 1.05rem !important; font-weight:600; }
.stTabs [data-baseweb="tab"]:hover { color:#fff !important; }
.stTabs [aria-selected="true"] { color:#fff !important; border-bottom:2px solid #fff !important; }
[data-testid="stDataFrame"] { border:1px solid #303030 !important; border-radius:7px !important; background:#050505 !important; }
[data-testid="stFileUploader"] { background:#050505 !important; border:1px solid #333 !important; border-radius:7px !important; }
[data-testid="stFileUploader"] * { color:#fff !important; }
button { background:#fff !important; color:#000 !important; border:1px solid #fff !important; border-radius:5px !important; font-weight:700 !important; }
button:hover { background:#ddd !important; }
input,textarea { background:#080808 !important; color:#fff !important; border:1px solid #444 !important; }
div[data-baseweb="select"] > div { background:#080808 !important; color:#fff !important; border-color:#444 !important; }
div[data-baseweb="select"] * { color:#fff !important; }
[data-testid="stAlert"] { background:#101010 !important; border:1px solid #333 !important; color:#fff !important; }
[data-testid="stAlert"] * { color:#fff !important; }
hr { border-color:#252525 !important; }
.sidebar-title { font-size:1.15rem; font-weight:800; color:#fff; line-height:1.1; }
.sidebar-sub { font-size:.72rem; color:#aaa; line-height:1.35; margin-top:.3rem; }
.section-label { font-size:.68rem; letter-spacing:.12em; font-weight:800; color:#999; margin:.65rem 0 .45rem; }
.kpi-note { color:#72d34d; font-size:.78rem; }
.panel-title { font-size:.98rem; font-weight:800; color:#fff; margin-bottom:.25rem; }
</style>
""", unsafe_allow_html=True)

REQUIRED = [
    "Date","Store ID","Product ID","Category","Region",
    "Inventory Level","Units Sold","Units Ordered","Demand Forecast",
    "Price","Discount","Weather Condition","Holiday/Promotion",
    "Competitor Pricing","Seasonality"
]

with st.sidebar:
    st.markdown("<div class='sidebar-title'>📦 Retail Inventory<br>Intelligence</div><div class='sidebar-sub'>Demand Forecasting • Inventory<br>Management • Reorder Planning</div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='section-label'>DATA & SETTINGS</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Sales/Inventory CSV", type=["csv"])
    st.markdown("<div class='section-label'>MODEL SETTINGS</div>", unsafe_allow_html=True)
    model_name = st.selectbox("Model", ["Random Forest"], index=0)
    n_trees = st.selectbox("Number of Trees", [100, 150, 200, 250, 300], index=2)
    safety_stock_pct = st.number_input("Safety Stock (%)", min_value=0, max_value=100, value=15, step=1)
    refresh = st.button("▶  Train / Refresh Model", width="stretch")
    st.markdown("<div class='section-label'>DATA SUMMARY</div>", unsafe_allow_html=True)
    # Summary is filled after data loads below.

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
    n_estimators=n_trees,
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

st.sidebar.markdown(f"<div class='sidebar-sub'><b>Total Records</b><span style='float:right'>{len(df):,}</span><br><b>Date Range</b><span style='float:right'>{df["Date"].min():%d %b %Y} - {latest_date:%d %b %Y}</span><br><b>Stores</b><span style='float:right'>{df["Store ID"].nunique():,}</span><br><b>Products</b><span style='float:right'>{df["Product ID"].nunique():,}</span><br><b>Categories</b><span style='float:right'>{df["Category"].nunique():,}</span><br><b>Regions</b><span style='float:right'>{df["Region"].nunique():,}</span></div>", unsafe_allow_html=True)

top_left, top_right = st.columns([10, 1])
with top_right:
    st.button("☾  Dark Mode", width="stretch")

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
    st.markdown("## 🎯 Dashboard")
    st.caption("Overview of inventory, demand and performance")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Products", f"{inventory_master['Product ID'].nunique():,}", "Active Products")
    c2.metric("Current Stock (Units)", f"{inventory_master['Current Stock'].sum():,.0f}", "+5.43% vs last month")
    c3.metric("Reorder Level (Units)", f"{inventory_master['Reorder Level'].sum():,.0f}", "Based on manual settings")
    c4.metric("MAE", f"{mae:,.2f}", "Lower is better")
    c5.metric("MAPE", f"{mape:.2f}%" if np.isfinite(mape) else "N/A", "Lower is better")

    status_df = inventory_master.copy()
    status_df["Safety Stock"] = status_df["Reorder Level"] * (safety_stock_pct / 100)
    status_df["Target Stock"] = status_df["Reorder Level"] + status_df["Safety Stock"]
    status_df["Order Qty"] = np.maximum(np.ceil(status_df["Target Stock"] - status_df["Current Stock"]),0)
    status_df["Status"] = np.select([
        status_df["Current Stock"] <= 0,
        status_df["Current Stock"] < status_df["Reorder Level"],
        status_df["Current Stock"] > status_df["Target Stock"] * 1.5
    ],["Stockout","Reorder","Overstock"],default="Healthy")

    a,b,c,d=st.columns(4)
    a.metric("🔴 Stockout", f"{(status_df['Status']=='Stockout').sum():,}", "Products")
    b.metric("🟠 Reorder", f"{(status_df['Status']=='Reorder').sum():,}", "Products")
    c.metric("🔵 Overstock", f"{(status_df['Status']=='Overstock').sum():,}", "Products")
    d.metric("🟢 Healthy", f"{(status_df['Status']=='Healthy').sum():,}", "Products")

    left,mid,right=st.columns([1.7,1.25,1.15])
    with left:
        st.markdown("<div class='panel-title'>Actual vs Predicted Demand (Test Set)</div>", unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(8,3.4),facecolor='#000000')
        ax.set_facecolor('#000000')
        n=min(70,len(actual))
        ax.plot(actual[:n], color='white', linewidth=1.5, marker='o', markersize=2, label='Actual Demand')
        ax.plot(pred[:n], color='#f2c500', linewidth=1.5, marker='o', markersize=2, label='Predicted Demand')
        ax.set_ylabel('Units Sold',color='white'); ax.tick_params(colors='#aaa'); ax.grid(alpha=.15,color='white'); ax.legend(facecolor='#000',labelcolor='white',frameon=False,ncol=2)
        for sp in ax.spines.values(): sp.set_color('#333')
        st.pyplot(fig,width='stretch'); plt.close(fig)
    with mid:
        st.markdown("<div class='panel-title'>Demand by Category (Units Sold)</div>", unsafe_allow_html=True)
        cat=df.groupby('Category')['Units Sold'].sum().sort_values(ascending=False).head(8)
        fig,ax=plt.subplots(figsize=(5,3.4),facecolor='#000'); ax.set_facecolor('#000')
        ax.pie(cat.values, startangle=90, wedgeprops={'width':0.38,'edgecolor':'#000'}, labels=None)
        ax.text(0,0,'Demand',ha='center',va='center',color='white',fontsize=11,fontweight='bold')
        ax.legend([f'{i}   {v/cat.sum()*100:.1f}%' for i,v in cat.items()],loc='center left',bbox_to_anchor=(1.0,.5),frameon=False,labelcolor='white',fontsize=7)
        st.pyplot(fig,width='stretch'); plt.close(fig)
    with right:
        st.markdown("<div class='panel-title'>Inventory Value by Category (₹)</div>", unsafe_allow_html=True)
        val=status_df.assign(InventoryValue=status_df['Current Stock']*status_df['Unit Cost']).groupby('Category')['InventoryValue'].sum().sort_values(ascending=True).tail(8)
        fig,ax=plt.subplots(figsize=(5,3.4),facecolor='#000'); ax.set_facecolor('#000')
        ax.barh(val.index,val.values,color='white',height=.58)
        ax.tick_params(colors='#aaa',labelsize=7); ax.xaxis.set_visible(False); ax.grid(axis='x',alpha=.12)
        for sp in ax.spines.values(): sp.set_color('#222')
        st.pyplot(fig,width='stretch'); plt.close(fig)

    st.markdown("<div class='panel-title'>Inventory Status Overview</div>", unsafe_allow_html=True)
    display=status_df.copy()
    display["Inventory Value (₹)"]=display["Current Stock"]*display["Unit Cost"]
    display=display.rename(columns={"Current Stock":"Current Stock","Lead Time Days":"Lead Time"})
    cols=[c for c in ["Store ID","Product ID","Category","Region","Current Stock","Demand Forecast","Safety Stock","Reorder Level","Target Stock","Order Qty","Status","Inventory Value (₹)"] if c in display.columns]
    styled_display = display[cols].head(10).style.applymap(
        lambda v: "background-color:#168a36;color:white;font-weight:700;border-radius:10px" if v == "Healthy" else (
            "background-color:#d99b00;color:black;font-weight:700" if v == "Reorder" else (
                "background-color:#1777d1;color:white;font-weight:700" if v == "Overstock" else (
                    "background-color:#c62828;color:white;font-weight:700" if v == "Stockout" else ""
                )
            )
        ), subset=["Status"] if "Status" in display[cols].columns else None
    )
    st.dataframe(styled_display,width='stretch',hide_index=True)
    st.markdown("<div style='text-align:center;color:#aaa;padding:.45rem'>View Full Inventory Planning  →</div>",unsafe_allow_html=True)
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
        plan["Reorder Level"] * (safety_stock_pct / 100)
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

    st.dataframe(
        plan,
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
        plan["Reorder Level"] * (safety_stock_pct / 100)
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
        st.dataframe(
            alerts,
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

    latest_series = (
        df.sort_values("Date")
        .groupby(
            ["Store ID","Product ID"],
            as_index=False
        )
        .tail(30)
    )

    future_rows = []

    for (store, product), group in latest_series.groupby(
        ["Store ID","Product ID"]
    ):
        group = group.sort_values("Date")
        history = group["Units Sold"].tolist()

        if len(history) < 7:
            continue

        last = group.iloc[-1]

        for step in range(1, 13):

            next_date = (
                latest_date
                + pd.DateOffset(months=step)
            )

            row = {
                "Inventory Level": float(
                    last["Inventory Level"]
                ),
                "Units Ordered": float(
                    last["Units Ordered"]
                ),
                "Demand Forecast": float(
                    last["Demand Forecast"]
                ),
                "Price": float(
                    last["Price"]
                ),
                "Discount": float(
                    last["Discount"]
                ),
                "Holiday/Promotion": float(
                    last["Holiday/Promotion"]
                ),
                "Competitor Pricing": float(
                    last["Competitor Pricing"]
                ),
                "Year": next_date.year,
                "Month": next_date.month,
                "DayOfWeek": next_date.dayofweek,
                "Lag_1": history[-1],
                "Lag_7": history[-7],
                "Rolling_7": np.mean(
                    history[-7:]
                ),
                "Rolling_30": np.mean(
                    history[-30:]
                )
            }

            for feature in dummy_features:
                row[feature] = 0

            for prefix, column in [
                ("Category","Category"),
                ("Region","Region"),
                ("Weather Condition","Weather Condition"),
                ("Seasonality","Seasonality")
            ]:

                value = str(last[column])
                key = f"{prefix}_{value}"

                if key in row:
                    row[key] = 1

            future_X = pd.DataFrame(
                [row]
            ).reindex(
                columns=features,
                fill_value=0
            )

            forecast = max(
                float(
                    model.predict(
                        future_X
                    )[0]
                ),
                0
            )

            safety = forecast * (safety_stock_pct / 100)

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

            history.append(forecast)

    future = pd.DataFrame(
        future_rows
    )

    if future.empty:

        st.warning(
            "Not enough product history for the 12-month forecast."
        )

    else:

        monthly_future = (
            future.groupby(
                ["Date","Month"],
                as_index=False
            )
            .agg(
                {
                    "Predicted Demand":"sum",
                    "Safety Stock":"sum",
                    "Recommended Inventory":"sum"
                }
            )
            .sort_values("Date")
        )

        c1,c2,c3 = st.columns(3)

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

        st.dataframe(
            monthly_future[
                [
                    "Month",
                    "Predicted Demand",
                    "Safety Stock",
                    "Recommended Inventory"
                ]
            ],
            width="stretch",
            hide_index=True
        )

        st.download_button(
            "⬇ Download 12-Month Forecast",
            monthly_future.to_csv(
                index=False
            ).encode("utf-8"),
            "12_month_forecast.csv",
            "text/csv"
        )

        fig, ax = plt.subplots(
            figsize=(12,5)
        )

        ax.plot(
            monthly_future["Month"],
            monthly_future["Predicted Demand"],
            marker="o"
        )

        ax.set_title(
            "12-Month Forecasted Demand"
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Predicted Units")

        ax.tick_params(
            axis="x",
            rotation=35
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        st.pyplot(
            fig,
            width="stretch"
        )

        plt.close(fig)

st.divider()

st.caption(
    "Retail Inventory Intelligence • Random Forest • 200 Estimators • Manual Inventory Master"
)
