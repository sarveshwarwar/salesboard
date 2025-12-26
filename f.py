import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sales Analysis Dashboard",
    layout="wide"
)

# ---------------- TITLE ----------------
st.title("📊 Sales Analysis Dashboard")
st.markdown("End-to-end sales performance analysis using Python & Streamlit")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("sales_data.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Month'] = df['Order Date'].dt.month
    df['Year'] = df['Order Date'].dt.year
    return df

df = load_data()

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("🔍 Filters")

region_filter = st.sidebar.multiselect(
    "Select Region",
    df['Region'].unique(),
    default=df['Region'].unique()
)

category_filter = st.sidebar.multiselect(
    "Select Category",
    df['Category'].unique(),
    default=df['Category'].unique()
)

filtered_df = df[
    (df['Region'].isin(region_filter)) &
    (df['Category'].isin(category_filter))
]

# ---------------- KPI CALCULATIONS ----------------
total_sales = filtered_df['Sales'].sum()
total_profit = filtered_df['Profit'].sum()
total_orders = len(filtered_df)
profit_margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0

# ---------------- KPI DISPLAY ----------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("💰 Total Sales", f"₹{total_sales:,.0f}")
c2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
c3.metric("🧾 Orders", total_orders)
c4.metric("📊 Profit Margin", f"{profit_margin:.2f}%")

st.divider()

# ---------------- MONTHLY SALES TREND ----------------
monthly_sales = (
    filtered_df
    .groupby(['Year', 'Month'])['Sales']
    .sum()
    .reset_index()
)

monthly_sales['Period'] = monthly_sales['Year'].astype(str) + "-" + monthly_sales['Month'].astype(str)

fig1 = px.line(
    monthly_sales,
    x="Period",
    y="Sales",
    markers=True,
    title="📅 Monthly Sales Trend"
)

st.plotly_chart(fig1, use_container_width=True)

# ---------------- SALES BY REGION ----------------
region_sales = filtered_df.groupby("Region")['Sales'].sum().reset_index()

fig2 = px.bar(
    region_sales,
    x="Region",
    y="Sales",
    title="🌍 Sales by Region"
)

st.plotly_chart(fig2, use_container_width=True)

# ---------------- PROFIT BY CATEGORY ----------------
category_profit = filtered_df.groupby("Category")['Profit'].sum().reset_index()

fig3 = px.pie(
    category_profit,
    names="Category",
    values="Profit",
    title="🧩 Profit by Category"
)

st.plotly_chart(fig3, use_container_width=True)

# ---------------- TOP PRODUCTS ----------------
top_products = (
    filtered_df
    .groupby("Product")['Sales']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig4 = px.bar(
    top_products,
    x="Sales",
    y="Product",
    orientation="h",
    title="🏆 Top 10 Products by Sales"
)

st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ---------------- SALES FORECASTING ----------------
st.subheader("🔮 Sales Forecast (Next 6 Months)")

forecast_df = (
    filtered_df
    .groupby(['Year', 'Month'])['Sales']
    .sum()
    .reset_index()
)

forecast_df['TimeIndex'] = np.arange(len(forecast_df))

X = forecast_df[['TimeIndex']]
y = forecast_df['Sales']

model = LinearRegression()
model.fit(X, y)

future_months = 6
last_index = forecast_df['TimeIndex'].max()

future_index = np.arange(
    last_index + 1,
    last_index + future_months + 1
)

future_sales = model.predict(future_index.reshape(-1, 1))

forecast_result = pd.DataFrame({
    "TimeIndex": future_index,
    "Forecasted Sales": future_sales
})

fig_forecast = px.line(title="📈 Sales Forecast")

fig_forecast.add_scatter(
    x=forecast_df['TimeIndex'],
    y=forecast_df['Sales'],
    mode='lines+markers',
    name='Historical Sales'
)

fig_forecast.add_scatter(
    x=forecast_result['TimeIndex'],
    y=forecast_result['Forecasted Sales'],
    mode='lines+markers',
    name='Forecasted Sales'
)

st.plotly_chart(fig_forecast, use_container_width=True)

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Data Preview")
st.dataframe(filtered_df.head(50))

# ---------------- INSIGHTS ----------------
st.subheader("📌 Business Insights")

st.write("""
✔ Sales show clear seasonal trends  
✔ A few products contribute major revenue  
✔ Certain regions consistently outperform others  
✔ Forecast helps in inventory and revenue planning
""")

st.success("✅ Dashboard loaded successfully!")
