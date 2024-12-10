import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import numpy as np
import traceback
from style import apply_custom_style, format_performance_table, format_holdings_table

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="UO MIG Portfolio Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    st.markdown(apply_custom_style(), unsafe_allow_html=True)
    
    # Header with logo
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("""
        <div class="custom-header">
            <h1>UO Masters Investment Group</h1>
            <p>Portfolio Analytics Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.image("uo_logo.jpg", width=80)

def plot_cumulative_returns(data: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['PORTFOLIO'],
        name='Portfolio',
        line=dict(color='#004F2F', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['BENCHMARK'],
        name='Benchmark',
        line=dict(color='#FEE123', width=3)
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=20, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
        yaxis=dict(title='Cumulative Return', tickformat='.1%'),
        xaxis=dict(title='Date'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    return fig

def plot_portfolio_and_cash(balances: pd.DataFrame):
    balances['balance_date'] = pd.to_datetime(balances['balance_date'])
    balances = balances.sort_values('balance_date')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=balances['balance_date'],
        y=balances['total_portfolio_value'],
        name='Portfolio Value',
        line=dict(color='#004F2F', width=3),
        hovertemplate='<b>%{x|%b %d}</b><br>Portfolio: $%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=balances['balance_date'],
        y=balances['cash_balance'],
        name='Cash Balance',
        line=dict(color='#FEE123', width=2),
        fill='tozeroy',
        fillcolor='rgba(254, 225, 35, 0.1)',
        hovertemplate='<b>%{x|%b %d}</b><br>Cash: $%{y:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=40, r=40, t=10, b=20),
        paper_bgcolor='rgba(255,255,255,0.95)',
        plot_bgcolor='rgba(255,255,255,0.95)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(title='Value ($)', tickformat='$,.0f', gridcolor='rgba(0,79,47,0.1)'),
        xaxis=dict(title='Date', gridcolor='rgba(0,79,47,0.1)')
    )
    return fig

def create_allocation_chart(holdings: pd.DataFrame, prices: pd.DataFrame, cash_balance: float):
    """Create enhanced allocation pie chart with more space."""
    allocation = calculate_portfolio_allocation(holdings, prices, cash_balance)
    colors = ['#2E75B6', '#70AD47', '#4472C4', '#ED7D31', '#5B9BD5', 
              '#A5A5A5', '#FFC000', '#9E480E', '#997300', '#43682B']
    
    fig = go.Figure(data=[go.Pie(
        labels=allocation['stock_symbol'],
        values=allocation['market_value'],
        hole=.4,
        marker=dict(colors=colors[:len(allocation)]),
        textinfo='label+percent',
        textposition='outside',
        textfont_size=12,
        hovertemplate="<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        height=400,  # Increased height
        margin=dict(
            l=80,    # Left margin
            r=80,    # Right margin
            t=40,    # Top margin
            b=120     # Increased bottom margin
        ),
        showlegend=False,
        autosize=True,
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        uniformtext=dict(mode='hide', minsize=12)  # Ensure consistent text size
    )
    
    return fig
def display_holdings_table(holdings: pd.DataFrame, returns: pd.DataFrame, prices: pd.DataFrame):
    merged_data = holdings.merge(
        prices[['stock_symbol', 'current_price']], 
        on='stock_symbol', 
        how='left'
    )
    
    merged_data['market_value'] = merged_data['shares_held'] * merged_data['current_price']
    total_value = merged_data['market_value'].sum()
    merged_data['weight'] = merged_data['market_value'] / total_value
    
    merged_data = merged_data.merge(
        returns[['stock_symbol', 'weekly_return', 'monthly_return', 'fytd_return']], 
        on='stock_symbol', 
        how='left'
    )
    
    display_data = pd.DataFrame({
        'Symbol': merged_data['stock_symbol'],
        'Shares': merged_data['shares_held'],
        'Price': merged_data['current_price'],
        'Market Value': merged_data['market_value'],
        'Weight': merged_data['weight'],
        'Weekly Return': merged_data['weekly_return'],
        'Monthly Return': merged_data['monthly_return'],
        'FYTD Return': merged_data['fytd_return']
    })
    
    styled_df = display_data.style.format({
        'Shares': '{:,.0f}',
        'Price': '${:,.2f}',
        'Market Value': '${:,.2f}',
        'Weight': '{:.2%}',
        'Weekly Return': '{:+.2%}',
        'Monthly Return': '{:+.2%}',
        'FYTD Return': '{:+.2%}'
    }).applymap(
        lambda x: 'color: #0C9B6A' if isinstance(x, float) and x > 0 else 'color: #DC2626',
        subset=['Weekly Return', 'Monthly Return', 'FYTD Return']
    ).set_table_styles([{
        'selector': 'th',
        'props': [
            ('background-color', '#1A2B32'),
            ('color', 'white'),
            ('font-weight', '600'),
            ('text-align', 'left'),
            ('padding', '1rem'),
            ('border-bottom', '3px solid #FEE123')
        ]
    }])
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

def display_top_bottom_performers(returns: pd.DataFrame):
    """Display top and bottom performing stocks."""
    metric = 'weekly_return'
    
    top_performers = returns.nlargest(5, metric)
    bottom_performers = returns.nsmallest(5, metric)
    
    top_df = (
        top_performers[['stock_symbol', metric]]
        .rename(columns={
            'stock_symbol': 'Symbol',
            metric: 'Weekly Return'
        })
    )
    
    bottom_df = (
        bottom_performers[['stock_symbol', metric]]
        .rename(columns={
            'stock_symbol': 'Symbol',
            metric: 'Weekly Return'
        })
    )
    
    top_styled = top_df.style.format({
        'Weekly Return': '{:+.2%}'
    }).applymap(
        lambda x: 'color: #0C9B6A' if isinstance(x, float) and x > 0 else 'color: #DC2626',
        subset=['Weekly Return']
    ).set_table_styles([{
        'selector': 'th',
        'props': [
            ('background-color', '#1A2B32'),
            ('color', 'white'),
            ('font-weight', '600'),
            ('text-align', 'left'),
            ('padding', '1rem'),
            ('border-bottom', '3px solid #FEE123')
        ]
    }])
    
    bottom_styled = bottom_df.style.format({
        'Weekly Return': '{:+.2%}'
    }).applymap(
        lambda x: 'color: #0C9B6A' if isinstance(x, float) and x > 0 else 'color: #DC2626',
        subset=['Weekly Return']
    ).set_table_styles([{
        'selector': 'th',
        'props': [
            ('background-color', '#1A2B32'),
            ('color', 'white'),
            ('font-weight', '600'),
            ('text-align', 'left'),
            ('padding', '1rem'),
            ('border-bottom', '3px solid #FEE123')
        ]
    }])
    
    st.markdown("### Top 5 Performers")
    st.dataframe(top_styled, use_container_width=True, hide_index=True)
    
    st.markdown("### Bottom 5 Performers")
    st.dataframe(bottom_styled, use_container_width=True, hide_index=True)
def display_metrics(processor, balances, returns_data):
    """Display enhanced metric cards."""
    latest_total = balances['total_portfolio_value'].iloc[-1]
    current_cash = processor.cash_balance
    ytd_return = returns_data['fytd']['PORTFOLIO'].iloc[-1] * 100

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-label">Portfolio Value</div>
            <div class="metric-value">${latest_total:,.0f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Cash Position</div>
            <div class="metric-value">${current_cash:,.0f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">FYTD Return</div>
            <div class="metric-value {'positive' if ytd_return >= 0 else 'negative'}">
                {ytd_return:+.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def format_risk_metrics(df):
    """Format risk metrics with consistent styling."""
    return df.style.apply(lambda x: [
        'background-color: var(--card-bg); padding: 1rem !important; font-weight: 500;' 
        if i == 0 else 
        'background-color: var(--card-bg); padding: 1rem !important;' 
        for i in range(len(x))
    ], axis=1)

def display_risk_metrics(risk_metrics):
    """Display risk metrics with enhanced styling."""
    st.markdown("### Risk Analysis")
    
    # Create DataFrame with proper formatting
    risk_data = pd.DataFrame({
        'Metric': ['Alpha (Annual)', 'Beta', 'Tracking Error (FYTD)'],
        'Value': [
            risk_metrics['alpha'],  # Keep numeric for proper formatting
            risk_metrics['beta'], 
            risk_metrics['tracking_error']['fytd']
        ]
    })

    # Force clean formatting with proper sizing
    styled_risk_data = risk_data.style.format({
        'Value': '{:.2%}'  # Formats numeric values as percentages
    }).set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('background-color', '#004F2F !important'),  # Dark green header
                ('color', 'white !important'),              # White text
                ('font-size', '16px'),                      # Larger font
                ('font-weight', 'bold'),                    # Bold text
                ('text-align', 'left'),                     # Align left
                ('padding', '10px'),                        # Add padding
                ('border-bottom', '2px solid #1E3932'),     # Border below header
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('padding', '10px'),
                ('border-bottom', '1px solid #e0e0e0'),     # Light row borders
                ('text-align', 'right'),                   # Right-align values
            ]
        },
        {
            'selector': 'tbody tr:nth-child(even) td',
            'props': [
                ('background-color', '#F8F8F8'),            # Alternating row background
            ]
        }
    ])

    # Display the table
    st.dataframe(
        styled_risk_data, 
        use_container_width=True, 
        height=200,  # Explicitly set height to avoid excess space
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
            balances = cached_db.get_balances(
                DEFAULT_FUND_ID, 
                fiscal_start_date.strftime('%Y-%m-%d'), 
                today.strftime('%Y-%m-%d')
            )

        # Display metrics at the top
        display_metrics(processor, balances, returns_data)

        # Period selector for the main cumulative returns chart
        period = st.radio(
            "Select Return Period:",
            options=["Weekly", "Monthly", "FYTD"],
            horizontal=True,
            index=2,
            key="period_selector"
        )

        # Horizontal split for main charts (2:1 ratio)
        st.markdown("### Portfolio Overview")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Cumulative Returns")
            st.plotly_chart(
                plot_cumulative_returns(returns_data[period.lower()]), 
                use_container_width=True,
                key="main_returns_chart"
            )
            
        with col2:
            st.markdown("#### Total Portfolio Value & Cash Balance")
            st.plotly_chart(
                plot_portfolio_and_cash(balances), 
                use_container_width=True,
                key="main_cash_chart"
            )

        # Tabs for the rest of the content
        tabs = st.tabs(["Performance", "Holdings", "Risk Analysis"])

        # Holdings Tab
        with tabs[1]:
            col3, col4 = st.columns([1, 2])
            with col3:
                st.markdown("### Portfolio Allocation")
                st.plotly_chart(
                    create_allocation_chart(holdings, prices, processor.cash_balance), 
                    use_container_width=True,
                    key="allocation_pie_chart"
                )
            with col4:
                st.markdown("### Holdings Detail")
                display_holdings_table(holdings, equity_returns, prices)
            
            # Add performers section under the main holdings content
            st.markdown("### Performance Analysis")
            col5, col6 = st.columns(2)
            with col5:
                display_top_bottom_performers(equity_returns)

        # Performance Tab
        with tabs[0]:
            st.markdown("### Performance Summary")
            perf_data = pd.DataFrame({
                'Period': ['Weekly', 'Monthly', 'FYTD'],
                'Portfolio': [returns_data[p.lower()]['PORTFOLIO'].iloc[-1] for p in ['weekly', 'monthly', 'fytd']],
                'Benchmark': [returns_data[p.lower()]['BENCHMARK'].iloc[-1] for p in ['weekly', 'monthly', 'fytd']],
                'Excess Return': [
                    returns_data[p.lower()]['PORTFOLIO'].iloc[-1] - returns_data[p.lower()]['BENCHMARK'].iloc[-1]
                    for p in ['weekly', 'monthly', 'fytd']
                ]
            })
            
            # Apply color formatting to all numeric columns
            st.dataframe(
                perf_data.style.format({
                    'Portfolio': '{:+.2%}',
                    'Benchmark': '{:+.2%}',
                    'Excess Return': '{:+.2%}'
                }).applymap(
                    lambda x: 'color: #0C9B6A' if isinstance(x, float) and x > 0 else 'color: #DC2626',
                    subset=['Portfolio', 'Benchmark', 'Excess Return']
                ),
                use_container_width=True,
                hide_index=True,
                key="performance_summary_table"
            )

        # Risk Analysis Tab
        with tabs[2]:
            display_risk_metrics(risk_metrics)

    except Exception as e:
        st.error("An unexpected error occurred.")
        if st.checkbox("Show detailed error message"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
