import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Set page config
st.set_page_config(
    page_title="Sales Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Generate synthetic sales data
@st.cache_data
def generate_sales_data(n_records=500):
    """Generate realistic sales data"""
    np.random.seed(42)
    random.seed(42)
    
    regions = ['North', 'South', 'East', 'West']
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    customers = [f'Customer {i}' for i in range(1, 101)]
    
    data = []
    start_date = datetime(2024, 1, 1)
    
    for _ in range(n_records):
        date = start_date + timedelta(days=random.randint(0, 364))
        region = random.choice(regions)
        product = random.choice(products)
        quantity = random.randint(1, 50)
        unit_price = random.randint(50, 250)
        customer = random.choice(customers)
        
        data.append({
            'Date': date,
            'Region': region,
            'Product': product,
            'Quantity': quantity,
            'Unit_Price': unit_price,
            'Revenue': quantity * unit_price,
            'Customer': customer
        })
    
    df = pd.DataFrame(data)
    df['Month'] = df['Date'].dt.strftime('%b')
    df['Month_Num'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Year'] = df['Date'].dt.year
    
    return df.sort_values('Date')

# Load data
df = generate_sales_data()

# Sidebar
st.sidebar.title("📊 Dashboard Filters")
st.sidebar.markdown("---")

# Date range filter
min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Region filter
regions = ['All'] + sorted(df['Region'].unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", regions)

# Product filter
products = ['All'] + sorted(df['Product'].unique().tolist())
selected_product = st.sidebar.selectbox("Select Product", products)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use filters to drill down into specific segments of your sales data.")

# Filter data
filtered_df = df.copy()

if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df['Date'].dt.date >= date_range[0]) & 
        (filtered_df['Date'].dt.date <= date_range[1])
    ]

if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['Region'] == selected_region]

if selected_product != 'All':
    filtered_df = filtered_df[filtered_df['Product'] == selected_product]

# Main dashboard
st.title("📈 Sales Analysis Dashboard")
st.markdown("### Comprehensive Sales Performance Overview")
st.markdown("---")

# Calculate KPIs
total_revenue = filtered_df['Revenue'].sum()
total_orders = len(filtered_df)
avg_order_value = filtered_df['Revenue'].mean()
total_quantity = filtered_df['Quantity'].sum()
unique_customers = filtered_df['Customer'].nunique()

# Calculate growth (comparing first half vs second half)
midpoint = filtered_df['Date'].min() + (filtered_df['Date'].max() - filtered_df['Date'].min()) / 2
first_half = filtered_df[filtered_df['Date'] < midpoint]
second_half = filtered_df[filtered_df['Date'] >= midpoint]
first_half_revenue = first_half['Revenue'].sum()
second_half_revenue = second_half['Revenue'].sum()
growth = ((second_half_revenue - first_half_revenue) / first_half_revenue * 100) if first_half_revenue > 0 else 0

# Display KPIs
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="💰 Total Revenue",
        value=f"${total_revenue:,.0f}",
        delta=f"{growth:.1f}% growth"
    )

with col2:
    st.metric(
        label="🛒 Total Orders",
        value=f"{total_orders:,}",
        delta=f"Avg: ${avg_order_value:.0f}"
    )

with col3:
    st.metric(
        label="👥 Customers",
        value=f"{unique_customers:,}",
        delta="Unique"
    )

with col4:
    st.metric(
        label="📦 Units Sold",
        value=f"{total_quantity:,}",
        delta=f"{(total_quantity/total_orders):.1f} per order"
    )

with col5:
    st.metric(
        label="💵 Avg Order Value",
        value=f"${avg_order_value:.2f}",
        delta=None
    )

st.markdown("---")

# Row 1: Monthly Trend and Revenue by Region
col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Monthly Revenue Trend")
    monthly_revenue = filtered_df.groupby(['Month', 'Month_Num']).agg({
        'Revenue': 'sum',
        'Date': 'count'
    }).reset_index()
    monthly_revenue.columns = ['Month', 'Month_Num', 'Revenue', 'Orders']
    monthly_revenue = monthly_revenue.sort_values('Month_Num')
    
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(
        x=monthly_revenue['Month'],
        y=monthly_revenue['Revenue'],
        mode='lines+markers',
        name='Revenue',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8)
    ))
    fig_monthly.update_layout(
        height=350,
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig_monthly.update_yaxis(title="Revenue ($)")
    st.plotly_chart(fig_monthly, use_container_width=True)

with col2:
    st.subheader("🌍 Revenue by Region")
    revenue_by_region = filtered_df.groupby('Region')['Revenue'].sum().reset_index()
    
    fig_region = px.pie(
        revenue_by_region,
        values='Revenue',
        names='Region',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4
    )
    fig_region.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}'
    )
    fig_region.update_layout(
        height=350,
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_region, use_container_width=True)

st.markdown("---")

# Row 2: Product Performance
st.subheader("📊 Product Performance Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    product_performance = filtered_df.groupby('Product').agg({
        'Revenue': 'sum',
        'Quantity': 'sum',
        'Date': 'count'
    }).reset_index()
    product_performance.columns = ['Product', 'Revenue', 'Quantity', 'Orders']
    product_performance = product_performance.sort_values('Revenue', ascending=True)
    
    fig_products = go.Figure()
    fig_products.add_trace(go.Bar(
        y=product_performance['Product'],
        x=product_performance['Revenue'],
        name='Revenue',
        orientation='h',
        marker_color='#3b82f6',
        text=product_performance['Revenue'].apply(lambda x: f'${x:,.0f}'),
        textposition='auto',
    ))
    fig_products.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Revenue ($)"
    )
    st.plotly_chart(fig_products, use_container_width=True)

with col2:
    st.markdown("#### 🏆 Top Products")
    top_products = product_performance.sort_values('Revenue', ascending=False).reset_index(drop=True)
    for idx, row in top_products.head(5).iterrows():
        st.markdown(f"""
        **{idx+1}. {row['Product']}**  
        Revenue: ${row['Revenue']:,.0f}  
        Units: {row['Quantity']:,} | Orders: {row['Orders']:,}
        """)
        st.markdown("---")

st.markdown("---")

# Row 3: Customer and Regional Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("👥 Top 10 Customers by Revenue")
    top_customers = filtered_df.groupby('Customer')['Revenue'].sum().sort_values(ascending=False).head(10).reset_index()
    
    fig_customers = px.bar(
        top_customers,
        x='Revenue',
        y='Customer',
        orientation='h',
        color='Revenue',
        color_continuous_scale='Blues'
    )
    fig_customers.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig_customers.update_traces(
        text=top_customers['Revenue'].apply(lambda x: f'${x:,.0f}'),
        textposition='auto'
    )
    st.plotly_chart(fig_customers, use_container_width=True)

with col2:
    st.subheader("📈 Quarterly Performance")
    quarterly_data = filtered_df.groupby('Quarter').agg({
        'Revenue': 'sum',
        'Date': 'count'
    }).reset_index()
    quarterly_data.columns = ['Quarter', 'Revenue', 'Orders']
    quarterly_data['Quarter'] = 'Q' + quarterly_data['Quarter'].astype(str)
    
    fig_quarterly = go.Figure()
    fig_quarterly.add_trace(go.Bar(
        x=quarterly_data['Quarter'],
        y=quarterly_data['Revenue'],
        name='Revenue',
        marker_color='#10b981',
        text=quarterly_data['Revenue'].apply(lambda x: f'${x/1000:.1f}K'),
        textposition='auto',
    ))
    fig_quarterly.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
        yaxis_title="Revenue ($)"
    )
    st.plotly_chart(fig_quarterly, use_container_width=True)

st.markdown("---")

# Data Table
st.subheader("📋 Recent Transactions")

# Add download button
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Data as CSV",
    data=csv,
    file_name=f'sales_data_{datetime.now().strftime("%Y%m%d")}.csv',
    mime='text/csv',
)

# Display data
display_df = filtered_df[['Date', 'Product', 'Region', 'Customer', 'Quantity', 'Unit_Price', 'Revenue']].copy()
display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
display_df['Revenue'] = display_df['Revenue'].apply(lambda x: f'${x:,.2f}')
display_df['Unit_Price'] = display_df['Unit_Price'].apply(lambda x: f'${x:,.2f}')

st.dataframe(
    display_df.head(20),
    use_container_width=True,
    hide_index=True
)

st.info(f"📊 Showing 20 of {len(filtered_df)} total transactions")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Sales Analysis Dashboard | Built with Streamlit & Plotly</p>
        <p>Data generated for demonstration purposes</p>
    </div>
""", unsafe_allow_html=True)
