import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sales Analysis Dashboard", layout="wide")

# ---------------- TITLE ----------------
st.title("📊 Sales Analysis Dashboard")
st.markdown("End-to-end sales analysis using Python & Streamlit")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df

df = load_data()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filters")

region_filter = st.sidebar.multiselect(
    "Select Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

category_filter = st.sidebar.multiselect(
    "Select Category",
    df["Category"].unique(),
    default=df["Category"].unique()
)

filtered_df = df[
    (df["Region"].isin(region_filter)) &
    (df["Category"].isin(category_filter))
]

# ---------------- KPIs ----------------
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = len(filtered_df)
profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("💰 Total Sales", f"₹{total_sales:,.0f}")
c2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
c3.metric("🧾 Orders", total_orders)
c4.metric("📊 Profit Margin", f"{profit_margin:.2f}%")

st.divider()

# ---------------- MONTHLY SALES TREND ----------------
monthly_sales = (
    filtered_df
    .groupby(["Year", "Month"])["Sales"]
    .sum()
    .reset_index()
)

fig1, ax1 = plt.subplots()
ax1.plot(monthly_sales.index, monthly_sales["Sales"])
ax1.set_title("📅 Monthly Sales Trend")
ax1.set_xlabel("Month Index")
ax1.set_ylabel("Sales")
st.pyplot(fig1)

# ---------------- SALES BY REGION ----------------
region_sales = filtered_df.groupby("Region")["Sales"].sum()

fig2, ax2 = plt.subplots()
region_sales.plot(kind="bar", ax=ax2)
ax2.set_title("🌍 Sales by Region")
ax2.set_xlabel("Region")
ax2.set_ylabel("Sales")
st.pyplot(fig2)

# ---------------- PROFIT BY CATEGORY ----------------
category_profit = filtered_df.groupby("Category")["Profit"].sum()

fig3, ax3 = plt.subplots()
category_profit.plot(kind="pie", autopct="%1.1f%%", ax=ax3)
ax3.set_ylabel("")
ax3.set_title("🧩 Profit by Category")
st.pyplot(fig3)

# ---------------- TOP PRODUCTS ----------------
top_products = (
    filtered_df
    .groupby("Product")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig4, ax4 = plt.subplots()
top_products.plot(kind="barh", ax=ax4)
ax4.set_title("🏆 Top 10 Products by Sales")
ax4.set_xlabel("Sales")
st.pyplot(fig4)

st.divider()

# ---------------- SALES FORECASTING ----------------
st.subheader("🔮 Sales Forecast (Next 6 Months)")

forecast_df = (
    filtered_df
    .groupby(["Year", "Month"])["Sales"]
    .sum()
    .reset_index()
)

forecast_df["TimeIndex"] = np.arange(len(forecast_df))

X = forecast_df[["TimeIndex"]]
y = forecast_df["Sales"]

model = LinearRegression()
model.fit(X, y)

future_steps = 6
last_index = forecast_df["TimeIndex"].max()

future_index = np.arange(last_index + 1, last_index + future_steps + 1)
future_sales = model.predict(future_index.reshape(-1, 1))

fig5, ax5 = plt.subplots()
ax5.plot(forecast_df["TimeIndex"], forecast_df["Sales"], label="Actual")
ax5.plot(future_index, future_sales, label="Forecast")
ax5.set_title("📈 Sales Forecast")
ax5.legend()
st.pyplot(fig5)

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df.head(50))

# ---------------- INSIGHTS ----------------
st.subheader("📌 Business Insights")
st.write("""
✔ Sales show seasonal trends  
✔ Few products contribute most revenue  
✔ Certain regions outperform consistently  
✔ Forecast helps business planning
""")

st.success("✅ Dashboard loaded successfully!")
