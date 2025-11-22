import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="NYC Taxi Analytics", layout="wide")

st.title("NYC Taxi Analytics Dashboard (2023-2024)")

# Load Gold Data
data_files = [
    'data/gold/daily_stats.parquet',
    'data/gold/hourly_stats.parquet',
    'data/gold/borough_stats.parquet'
]

missing_files = [f for f in data_files if not os.path.exists(f)]

if missing_files:
    st.error(f"Data not found: {missing_files}. Please run the pipeline first using 'python main.py'.")
    st.stop()

try:
    daily_stats = pd.read_parquet('data/gold/daily_stats.parquet')
    hourly_stats = pd.read_parquet('data/gold/hourly_stats.parquet')
    borough_stats = pd.read_parquet('data/gold/borough_stats.parquet')
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Top Level Metrics
total_trips = daily_stats['total_trips'].sum()
total_revenue = daily_stats['total_revenue'].sum()
avg_fare = total_revenue / total_trips if total_trips > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Total Trips", f"{total_trips:,}")
col2.metric("Total Revenue", f"${total_revenue:,.2f}")
col3.metric("Avg Fare", f"${avg_fare:.2f}")

# Charts
st.subheader("Daily Trends")
fig_daily = px.line(daily_stats, x='trip_date', y='total_trips', title='Daily Trip Counts')
st.plotly_chart(fig_daily, use_container_width=True)

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Hourly Demand")
    fig_hourly = px.bar(hourly_stats, x='pickup_hour', y='total_trips', title='Trips by Hour of Day')
    st.plotly_chart(fig_hourly, use_container_width=True)

with col_chart2:
    st.subheader("Borough Analysis")
    fig_borough = px.bar(borough_stats, x='Borough', y='total_trips', title='Trips by Borough')
    st.plotly_chart(fig_borough, use_container_width=True)

st.subheader("Borough Details")
st.dataframe(borough_stats)

st.dataframe(daily_stats.sort_values('trip_date', ascending=False))
