import streamlit as st
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sales Dashboard", layout="wide")

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
region_filter = st.sidebar.multiselect("Select Region", df["Region"].unique(), default=df["Region"].unique())
category_filter = st.sidebar.multiselect("Select Category", df["Category"].unique(), default=df["Category"].unique())

filtered_df = df[(df["Region"].isin(region_filter)) & (df["Category"].isin(category_filter))]

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
monthly_sales = filtered_df.groupby(["Year","Month"])["Sales"].sum().reset_index()
st.subheader("📅 Monthly Sales Trend")
st.line_chart(monthly_sales["Sales"])

# ---------------- SALES BY REGION ----------------
st.subheader("🌍 Sales by Region")
region_sales = filtered_df.groupby("Region")["Sales"].sum()
st.bar_chart(region_sales)

# ---------------- PROFIT BY CATEGORY ----------------
st.subheader("🧩 Profit by Category")
category_profit = filtered_df.groupby("Category")["Profit"].sum()
st.bar_chart(category_profit)

# ---------------- TOP PRODUCTS ----------------
st.subheader("🏆 Top 10 Products by Sales")
top_products = filtered_df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(10)
st.bar_chart(top_products)

st.divider()

# ---------------- SIMPLE SALES FORECAST ----------------
st.subheader("🔮 Sales Forecast (Next 6 Months)")

# Using manual linear regression with NumPy
sales_series = monthly_sales["Sales"].values
n = len(sales_series)
x = np.arange(n)
y = sales_series

# Linear regression formula: y = mx + b
m = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / (n*np.sum(x**2) - (np.sum(x)**2))
b = (np.sum(y) - m*np.sum(x)) / n

future_steps = 6
future_x = np.arange(n, n+future_steps)
future_y = m*future_x + b

forecast_series = np.concatenate([y, future_y])
st.line_chart(forecast_series)

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df.head(50))

# ---------------- INSIGHTS ----------------
st.subheader("📌 Business Insights")
st.write("""
✔ Sales show seasonal patterns  
✔ Few products contribute majority of revenue  
✔ Certain regions outperform consistently  
✔ Forecasting helps business planning
""")

st.success("✅ Dashboard loaded successfully!")
