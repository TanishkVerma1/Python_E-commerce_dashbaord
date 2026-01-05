"""
SouqPlus Executive Dashboard
A production-ready analytics dashboard for UAE e-commerce operations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="SouqPlus Executive Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric label {
        color: #f0f0f0 !important;
        font-weight: 600;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.8rem !important;
        font-weight: bold;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #ffd700 !important;
    }
    h1 {
        color: #2c3e50;
        padding-bottom: 10px;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    h2 {
        color: #34495e;
        font-size: 1.4rem;
        padding-top: 10px;
        font-weight: 600;
    }
    .insight-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 15px;
        border-left: 5px solid #e17055;
        margin: 15px 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        color: #2d3436;
        font-weight: 500;
    }
    div[data-testid="stTabs"] {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 10px;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)


def load_order_items():
    """Load order_items.csv with special handling for quoted lines"""
    import io
    with open('order_items.csv', 'r', encoding='utf-8-sig') as f:
        content = f.read()
    lines = content.strip().split('\n')
    cleaned_lines = [line.strip().strip('"') for line in lines]
    cleaned_content = '\n'.join(cleaned_lines)
    return pd.read_csv(io.StringIO(cleaned_content))


@st.cache_data
def load_data():
    """Load and prepare all data with proper date parsing"""
    try:
        # Load datasets
        customers = pd.read_csv('customers.csv')
        orders = pd.read_csv('orders.csv', parse_dates=['order_date'])
        order_items = load_order_items()
        fulfillment = pd.read_csv('fullfillment.csv', parse_dates=['actual_delivery_date'])
        returns = pd.read_csv('returns.csv', parse_dates=['return_date'])
        
        # Merge orders with order_items to get complete order data
        orders_full = orders.merge(order_items, on='order_id', how='left')
        
        # Merge with fulfillment data
        orders_full = orders_full.merge(
            fulfillment[['order_id', 'delivery_status', 'actual_delivery_date', 
                        'warehouse_hub', 'delivery_zone', 'delay_reason']],
            on='order_id', 
            how='left'
        )
        
        # Merge with customer data
        orders_full = orders_full.merge(
            customers[['customer_id', 'city']],
            on='customer_id',
            how='left'
        )
        
        # Merge with returns data
        returns_agg = returns.groupby('order_id').agg({
            'refund_amount': 'sum',
            'return_date': 'first'
        }).reset_index()
        
        orders_full = orders_full.merge(
            returns_agg,
            on='order_id',
            how='left'
        )
        
        # Fill NaN values
        orders_full['refund_amount'] = orders_full['refund_amount'].fillna(0)
        orders_full['discount_amount'] = orders_full['discount_amount'].fillna(0)
        orders_full['delay_reason'] = orders_full['delay_reason'].fillna('No Delay')
        
        return orders_full, customers, orders, order_items, fulfillment, returns
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None


def format_currency(value):
    """Format value as AED currency"""
    return f"AED {value:,.0f}"


def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.1f}%"


def create_insight_box(text):
    """Create a styled insight box"""
    st.markdown(f'<div class="insight-box">💡 <strong>Insight:</strong> {text}</div>', 
                unsafe_allow_html=True)


def apply_filters(df, date_range, cities, channels):
    """Apply sidebar filters to dataframe"""
    filtered_df = df.copy()
    
    # Date filter
    if date_range:
        filtered_df = filtered_df[
            (filtered_df['order_date'].dt.date >= date_range[0]) &
            (filtered_df['order_date'].dt.date <= date_range[1])
        ]
    
    # City filter
    if cities and 'All' not in cities:
        filtered_df = filtered_df[filtered_df['city'].isin(cities)]
    
    # Channel filter
    if channels and 'All' not in channels:
        filtered_df = filtered_df[filtered_df['order_channel'].isin(channels)]
    
    return filtered_df


def executive_view(df):
    """Executive View: Revenue, Growth, and Strategic Insights"""
    
    st.header("📊 Executive View")
    st.markdown("---")
    
    # KPIs Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_revenue = df['net_amount'].sum()
        st.metric("Total Revenue", format_currency(total_revenue))
    
    with col2:
        total_orders = df['order_id'].nunique()
        st.metric("Total Orders", f"{total_orders:,}")
    
    with col3:
        aov = total_revenue / total_orders if total_orders > 0 else 0
        st.metric("Average Order Value", format_currency(aov))
    
    with col4:
        total_discount = df['discount_amount'].sum()
        st.metric("Total Discount Given", format_currency(total_discount))
    
    with col5:
        total_refund = df['refund_amount'].sum()
        st.metric("Total Refund Amount", format_currency(total_refund))
    
    st.markdown("---")
    
    # Daily Revenue Trend
    st.subheader("📈 Daily Revenue Trend")
    daily_revenue = df.groupby(df['order_date'].dt.date)['net_amount'].sum().reset_index()
    daily_revenue.columns = ['Date', 'Revenue']
    
    fig_daily = px.line(
        daily_revenue, 
        x='Date', 
        y='Revenue',
        title='Daily Revenue Trend (AED)',
        labels={'Revenue': 'Revenue (AED)', 'Date': 'Order Date'}
    )
    fig_daily.update_traces(line_color='#1f77b4', line_width=2)
    fig_daily.update_layout(hovermode='x unified', height=400)
    st.plotly_chart(fig_daily, use_container_width=True)
    
    create_insight_box("Track daily revenue patterns to identify peak sales days and plan inventory accordingly.")
    
    # Two columns for next row
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by City
        st.subheader("🏙️ Revenue by City")
        city_revenue = df.groupby('city')['net_amount'].sum().reset_index()
        city_revenue = city_revenue.sort_values('net_amount', ascending=True)
        
        fig_city = px.bar(
            city_revenue,
            x='net_amount',
            y='city',
            orientation='h',
            title='Revenue Distribution by City (AED)',
            labels={'net_amount': 'Revenue (AED)', 'city': 'City'}
        )
        fig_city.update_traces(marker_color='#2ecc71')
        fig_city.update_layout(height=350)
        st.plotly_chart(fig_city, use_container_width=True)
        
        create_insight_box("Focus marketing efforts on high-performing cities while exploring growth opportunities in underperforming regions.")
    
    with col2:
        # Revenue by Order Channel
        st.subheader("📱 Revenue by Order Channel")
        channel_revenue = df.groupby('order_channel')['net_amount'].sum().reset_index()
        
        fig_channel = px.pie(
            channel_revenue,
            values='net_amount',
            names='order_channel',
            title='Revenue Share by Order Channel',
            hole=0.4
        )
        fig_channel.update_traces(textposition='inside', textinfo='percent+label')
        fig_channel.update_layout(height=350)
        st.plotly_chart(fig_channel, use_container_width=True)
        
        create_insight_box("Optimize investment in channels with highest revenue contribution and customer acquisition cost efficiency.")
    
    # Two columns for category analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by Product Category
        st.subheader("🏷️ Revenue by Product Category")
        category_revenue = df.groupby('product_category')['net_amount'].sum().reset_index()
        category_revenue = category_revenue.sort_values('net_amount', ascending=True)
        
        fig_category = px.bar(
            category_revenue,
            x='net_amount',
            y='product_category',
            orientation='h',
            title='Revenue by Product Category (AED)',
            labels={'net_amount': 'Revenue (AED)', 'product_category': 'Category'}
        )
        fig_category.update_traces(marker_color='#3498db')
        fig_category.update_layout(height=350)
        st.plotly_chart(fig_category, use_container_width=True)
        
        create_insight_box("Top categories drive majority of revenue. Consider cross-selling strategies to boost lower-performing categories.")
    
    with col2:
        # Discount % by Category (Margin Proxy)
        st.subheader("💰 Discount % by Category (Margin Proxy)")
        category_discount = df.groupby('product_category').agg({
            'discount_amount': 'sum',
            'net_amount': 'sum'
        }).reset_index()
        category_discount['discount_pct'] = (
            category_discount['discount_amount'] / 
            (category_discount['net_amount'] + category_discount['discount_amount']) * 100
        )
        category_discount = category_discount.sort_values('discount_pct', ascending=True)
        
        fig_discount = px.bar(
            category_discount,
            x='discount_pct',
            y='product_category',
            orientation='h',
            title='Discount % by Category',
            labels={'discount_pct': 'Discount %', 'product_category': 'Category'},
            color='discount_pct',
            color_continuous_scale='Reds'
        )
        fig_discount.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_discount, use_container_width=True)
        
        create_insight_box("High discount categories may indicate pricing pressure or promotional intensity. Monitor for margin impact.")


def operations_view(df, fulfillment, returns):
    """Operations View: Fulfillment, Delivery Performance, Returns"""
    
    st.header("🚚 Operations View")
    st.markdown("---")
    
    # Calculate operational KPIs
    total_deliveries = df['delivery_status'].notna().sum()
    on_time = (df['delivery_status'] == 'Delivered').sum()
    delayed = (df['delivery_status'] == 'Delayed').sum()
    failed = (df['delivery_status'] == 'Failed').sum()
    
    on_time_pct = (on_time / total_deliveries * 100) if total_deliveries > 0 else 0
    delayed_pct = (delayed / total_deliveries * 100) if total_deliveries > 0 else 0
    failed_pct = (failed / total_deliveries * 100) if total_deliveries > 0 else 0
    
    total_items = len(df)
    returned_items = (df['refund_amount'] > 0).sum()
    return_rate = (returned_items / total_items * 100) if total_items > 0 else 0
    
    # KPIs Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("On-time Delivery %", format_percentage(on_time_pct), 
                 delta="Good" if on_time_pct >= 80 else "Needs Attention",
                 delta_color="normal" if on_time_pct >= 80 else "inverse")
    
    with col2:
        st.metric("Delayed Delivery %", format_percentage(delayed_pct),
                 delta="High" if delayed_pct > 15 else "Acceptable",
                 delta_color="inverse" if delayed_pct > 15 else "normal")
    
    with col3:
        st.metric("Failed Delivery %", format_percentage(failed_pct),
                 delta="Critical" if failed_pct > 5 else "Acceptable",
                 delta_color="inverse" if failed_pct > 5 else "normal")
    
    with col4:
        st.metric("Return Rate %", format_percentage(return_rate),
                 delta="High" if return_rate > 10 else "Normal",
                 delta_color="inverse" if return_rate > 10 else "normal")
    
    st.markdown("---")
    
    # Delivery Status Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Delivery Status Distribution")
        status_counts = df['delivery_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig_status = px.pie(
            status_counts,
            values='Count',
            names='Status',
            title='Delivery Status Breakdown',
            color='Status',
            color_discrete_map={
                'Delivered': '#2ecc71',
                'Delayed': '#f39c12',
                'Failed': '#e74c3c'
            }
        )
        fig_status.update_traces(textposition='inside', textinfo='percent+label')
        fig_status.update_layout(height=350)
        st.plotly_chart(fig_status, use_container_width=True)
        
        create_insight_box("Monitor failed deliveries closely - they directly impact customer satisfaction and operational costs.")
    
    with col2:
        st.subheader("⏱️ On-time vs Delayed by Warehouse Hub")
        hub_performance = df[df['delivery_status'].isin(['Delivered', 'Delayed'])].groupby(
            ['warehouse_hub', 'delivery_status']
        ).size().reset_index(name='Count')
        
        fig_hub = px.bar(
            hub_performance,
            x='warehouse_hub',
            y='Count',
            color='delivery_status',
            title='Delivery Performance by Warehouse Hub',
            labels={'warehouse_hub': 'Warehouse Hub', 'Count': 'Number of Orders'},
            color_discrete_map={
                'Delivered': '#2ecc71',
                'Delayed': '#e74c3c'
            },
            barmode='group'
        )
        fig_hub.update_layout(height=350)
        st.plotly_chart(fig_hub, use_container_width=True)
        
        create_insight_box("Identify underperforming hubs for capacity planning and process improvement initiatives.")
    
    # Delays and Reasons
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗺️ Delays by Delivery Zone")
        delayed_df = df[df['delivery_status'] == 'Delayed']
        zone_delays = delayed_df['delivery_zone'].value_counts().reset_index()
        zone_delays.columns = ['Zone', 'Delayed Orders']
        zone_delays = zone_delays.sort_values('Delayed Orders', ascending=True)
        
        fig_zone = px.bar(
            zone_delays,
            x='Delayed Orders',
            y='Zone',
            orientation='h',
            title='Delayed Deliveries by Zone',
            labels={'Delayed Orders': 'Number of Delays', 'Zone': 'Delivery Zone'}
        )
        fig_zone.update_traces(marker_color='#e67e22')
        fig_zone.update_layout(height=350)
        st.plotly_chart(fig_zone, use_container_width=True)
        
        create_insight_box("High-delay zones may need route optimization or additional delivery partners.")
    
    with col2:
        st.subheader("⚠️ Top Delay Reasons")
        delay_reasons = df[df['delivery_status'] == 'Delayed']['delay_reason'].value_counts().head(10).reset_index()
        delay_reasons.columns = ['Reason', 'Count']
        delay_reasons = delay_reasons.sort_values('Count', ascending=True)
        
        fig_reasons = px.bar(
            delay_reasons,
            x='Count',
            y='Reason',
            orientation='h',
            title='Top 10 Reasons for Delivery Delays',
            labels={'Count': 'Number of Incidents', 'Reason': 'Delay Reason'}
        )
        fig_reasons.update_traces(marker_color='#c0392b')
        fig_reasons.update_layout(height=350)
        st.plotly_chart(fig_reasons, use_container_width=True)
        
        create_insight_box("Address top delay reasons systematically to improve overall delivery performance.")
    
    # Returns Analysis
    st.subheader("🔄 Returns Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        returns_by_category = df[df['refund_amount'] > 0].groupby('product_category').size().reset_index(name='Returns')
        returns_by_category = returns_by_category.sort_values('Returns', ascending=True)
        
        fig_returns_cat = px.bar(
            returns_by_category,
            x='Returns',
            y='product_category',
            orientation='h',
            title='Returns by Product Category',
            labels={'Returns': 'Number of Returns', 'product_category': 'Category'}
        )
        fig_returns_cat.update_traces(marker_color='#8e44ad')
        fig_returns_cat.update_layout(height=350)
        st.plotly_chart(fig_returns_cat, use_container_width=True)
        
        create_insight_box("High return rates in specific categories may signal quality issues or sizing problems.")
    
    with col2:
        returns_by_city = df[df['refund_amount'] > 0].groupby('city').size().reset_index(name='Returns')
        returns_by_city = returns_by_city.sort_values('Returns', ascending=True)
        
        fig_returns_city = px.bar(
            returns_by_city,
            x='Returns',
            y='city',
            orientation='h',
            title='Returns by City',
            labels={'Returns': 'Number of Returns', 'city': 'City'}
        )
        fig_returns_city.update_traces(marker_color='#16a085')
        fig_returns_city.update_layout(height=350)
        st.plotly_chart(fig_returns_city, use_container_width=True)
        
        create_insight_box("Geographic return patterns can reveal logistics or customer expectation issues in specific markets.")


def main():
    """Main application entry point"""
    
    # Header
    st.title("🛍️ SouqPlus Executive Dashboard")
    st.markdown("**Real-time insights for data-driven decision making**")
    
    # Load data
    with st.spinner("Loading data..."):
        orders_full, customers, orders, order_items, fulfillment, returns = load_data()
    
    if orders_full is None:
        st.error("Failed to load data. Please ensure all CSV files are in the correct location.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    st.sidebar.markdown("---")
    
    # Date range filter
    min_date = orders_full['order_date'].min().date()
    max_date = orders_full['order_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Ensure date_range is a tuple with two dates
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    
    # City filter
    all_cities = ['All'] + sorted(orders_full['city'].dropna().unique().tolist())
    selected_cities = st.sidebar.multiselect(
        "Select Cities",
        options=all_cities,
        default=['All']
    )
    
    # Channel filter
    all_channels = ['All'] + sorted(orders_full['order_channel'].dropna().unique().tolist())
    selected_channels = st.sidebar.multiselect(
        "Select Order Channels",
        options=all_channels,
        default=['All']
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this Dashboard**
    
    This dashboard provides real-time insights into SouqPlus operations across the UAE.
    
    - **Executive View**: Revenue, growth, and strategic metrics
    - **Operations View**: Fulfillment, delivery, and returns analysis
    
    Use filters to drill down into specific periods, cities, or channels.
    """)
    
    # Apply filters
    filtered_df = apply_filters(orders_full, (start_date, end_date), selected_cities, selected_channels)
    
    # Show filter summary
    st.markdown(f"**Showing data from {start_date} to {end_date}** | "
               f"**{len(filtered_df):,} order items** | "
               f"**{filtered_df['order_id'].nunique():,} unique orders**")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Executive View", "🚚 Operations View"])
    
    with tab1:
        executive_view(filtered_df)
    
    with tab2:
        operations_view(filtered_df, fulfillment, returns)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d;'>"
        "SouqPlus Executive Dashboard | Built with Streamlit | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
