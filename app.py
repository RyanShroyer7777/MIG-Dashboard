import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import numpy as np
import traceback
from style import apply_custom_style

# Import dependencies with error handling
try:
    from cache import (
        CachedDatabaseInterface,
        calculate_cumulative_returns,
        calculate_equity_returns,
        calculate_portfolio_allocation
    )
    from config import DATABASE_URL, DEFAULT_FUND_ID
except ImportError as e:
    st.error(f"Failed to import required modules: {str(e)}")
    st.stop()

def init_database():
    """Initialize database connection with error handling."""
    try:
        return CachedDatabaseInterface(DATABASE_URL)
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        if st.checkbox("Show error details"):
            st.code(traceback.format_exc())
        return None

def init_page():
    """Initialize the Streamlit page with custom styling."""
    st.set_page_config(
        page_title="UO MIG Portfolio Analytics",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown(apply_custom_style(), unsafe_allow_html=True)
    st.markdown(
        '''
        <div class="custom-header">
            <h1>UO Masters Investment Group</h1>
            <p>Portfolio Analytics Dashboard</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

def plot_cumulative_returns(data: pd.DataFrame):
    st.markdown("## Portfolio Returns: DADCO & IVV", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['PORTFOLIO'],
        name='Portfolio',
        line=dict(color='#154733', width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['BENCHMARK'],
        name='Benchmark',
        line=dict(color='#FEE123', width=2.5)
    ))
    fig.update_layout(
        height=450,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
        yaxis=dict(title='Cumulative Return', tickformat='.1%'),
        xaxis=dict(title='Date'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def create_allocation_pie(holdings: pd.DataFrame, prices: pd.DataFrame, cash_balance: float):
    allocation = calculate_portfolio_allocation(holdings, prices, cash_balance)
    colors = px.colors.qualitative.Set3[:len(allocation)]
    fig = go.Figure(data=[go.Pie(
        labels=allocation['stock_symbol'],
        values=allocation['market_value'],
        hole=.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside'
    )])
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)


def display_holdings_table(holdings: pd.DataFrame, returns: pd.DataFrame, prices: pd.DataFrame):
    merged_data = holdings.merge(
        prices[['stock_symbol', 'current_price']], 
        on='stock_symbol', how='left'
    )
    merged_data['market_value'] = merged_data['shares_held'] * merged_data['current_price']
    total_value = merged_data['market_value'].sum()
    merged_data['weight'] = merged_data['market_value'] / total_value
    available_return_cols = ['stock_symbol']
    for col in ['weekly_return', 'monthly_return', 'fytd_return']:
        if col in returns.columns:
            available_return_cols.append(col)
    merged_data = merged_data.merge(returns[available_return_cols], on='stock_symbol', how='left')
    display_data = pd.DataFrame({
        'Symbol': merged_data['stock_symbol'],
        'Shares': merged_data['shares_held'].map('{:,.0f}'.format),
        'Avg Cost': merged_data['average_cost'].map('${:,.2f}'.format),
        'Current': merged_data['current_price'].map('${:,.2f}'.format),
        'Market Value': merged_data['market_value'].map('${:,.2f}'.format),
        'Weight': merged_data['weight'].map('{:.2%}'.format)
    })
    if 'fytd_return' in merged_data.columns:
        display_data['FYTD Return'] = merged_data['fytd_return'].map('{:.2%}'.format)
    st.dataframe(display_data, use_container_width=True, hide_index=True)

def display_top_bottom_performers(returns: pd.DataFrame):
    metric = 'weekly_return'  # Change to 'fytd_return' or 'monthly_return' as needed
    top_performers = returns.nlargest(5, metric)
    bottom_performers = returns.nsmallest(5, metric)
    
    # Add a label for Top Performers
    st.markdown("### Top 5 Performers")
    st.dataframe(
        top_performers[['stock_symbol', metric]]
        .rename(columns={metric: 'Weekly Return'})
        .style.format({'Weekly Return': '{:.2%}'}),
        use_container_width=True,
        hide_index=True
    )
    
    # Add a label for Bottom Performers
    st.markdown("### Bottom 5 Performers")
    st.dataframe(
        bottom_performers[['stock_symbol', metric]]
        .rename(columns={metric: 'Weekly Return'})
        .style.format({'Weekly Return': '{:.2%}'}),
        use_container_width=True,
        hide_index=True
    )

def main():
    init_page()
    try:
        cached_db = init_database()
        if cached_db is None:
            return
        fiscal_start_date = date(2024, 3, 29)
        today = date.today()
        with st.spinner("Loading portfolio data..."):
            processor = cached_db.create_processor(fiscal_start_date, today, fund_id=DEFAULT_FUND_ID)
            if processor is None:
                st.error("Failed to initialize data processor.")
                return
            risk_metrics = processor.calculate_risk_metrics(fiscal_start_date)
            returns_data = processor.calculate_cumulative_returns(fiscal_start_date)
            holdings = processor.holdings
            prices = processor.stock_prices
            equity_returns = processor.calculate_equity_returns(fiscal_start_date)

        # Tabs for better organization
        tab1, tab2, tab3 = st.tabs(["Performance", "Holdings", "Risk Analysis"])
        
        with tab1:
            plot_cumulative_returns(returns_data['fytd'])
            
            st.subheader("Performance Summary")
            perf_data = pd.DataFrame({
                'Period': ['Weekly', 'Monthly', 'FYTD'],
                'Portfolio': [returns_data[p.lower()]['PORTFOLIO'].iloc[-1] for p in ['weekly', 'monthly', 'fytd']],
                'Benchmark': [returns_data[p.lower()]['BENCHMARK'].iloc[-1] for p in ['weekly', 'monthly', 'fytd']],
                'Excess Return': [
                    returns_data[p.lower()]['PORTFOLIO'].iloc[-1] - returns_data[p.lower()]['BENCHMARK'].iloc[-1]
                    for p in ['weekly', 'monthly', 'fytd']
                ]
            })
            st.dataframe(perf_data.style.format({
                'Portfolio': '{:.2%}', 'Benchmark': '{:.2%}', 'Excess Return': '{:.2%}'
            }), use_container_width=True, hide_index=True)
        
        with tab2:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Portfolio Allocation")
                create_allocation_pie(holdings, prices, processor.cash_balance)
            with col2:
                st.subheader("Holdings Details")
                display_holdings_table(holdings, equity_returns, prices)
                
            st.subheader("Performance Highlights")
            display_top_bottom_performers(equity_returns)

        with tab3:
            st.subheader("Risk Analytics")
            risk_data = pd.DataFrame({
                'Metric': ['Alpha (Annual)', 'Beta', 'Tracking Error (FYTD)'],
                'Value': [
                    f"{risk_metrics['alpha']:.2%}",
                    f"{risk_metrics['beta']:.2f}",
                    f"{risk_metrics['tracking_error']['fytd']:.2%}"
                ]
            })
            st.dataframe(risk_data, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error("An unexpected error occurred.")
        if st.checkbox("Show detailed error message"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
