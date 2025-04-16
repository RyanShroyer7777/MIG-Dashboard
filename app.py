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
# ----- UPDATE FISCAL_YEARS MANUALLY WHEN ADDING NEW FISCAL YEARS -----
FISCAL_YEARS = {
    "FY 2023": date(2023, 4, 1),
    "FY 2024": date(2024, 4, 1),
    "FY 2025": date(2025, 4, 1)
}
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


def init_page(latest_date: pd.Timestamp):
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
        # Add last updated timestamp below title - now only showing the date
        st.markdown(
            f"<p style='color: #666; font-size: 0.9em; margin-top: -0.5em;'>Last Updated: {latest_date.strftime('%B %d, %Y')}</p>",
            unsafe_allow_html=True)
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
        height=500,  # Increased height for main graph
        margin=dict(l=40, r=40, t=20, b=40),
        paper_bgcolor='white',
        plot_bgcolor='white',
        yaxis=dict(title='Cumulative Return', tickformat='.1%'),
        xaxis=dict(title='Date'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
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
            l=80,  # Left margin
            r=80,  # Right margin
            t=40,  # Top margin
            b=120  # Increased bottom margin
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

    # Use 'shares' if available, otherwise 'shares_held'
    shares_col = 'shares' if 'shares' in holdings.columns else 'shares_held'

    # Calculate market value using current price (don't use DB market_value)
    merged_data['market_value'] = merged_data[shares_col] * merged_data['current_price']
    total_value = merged_data['market_value'].sum()
    merged_data['weight'] = merged_data['market_value'] / total_value

    merged_data = merged_data.merge(
        returns[['stock_symbol', 'weekly_return', 'monthly_return', 'fytd_return']],
        on='stock_symbol',
        how='left'
    )

    display_data = pd.DataFrame({
        'Symbol': merged_data['stock_symbol'],
        'Shares': merged_data[shares_col],
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
    """
    Display top and bottom performing stocks with period selection.
    Bottom performers are sorted with worst performers (most negative returns) at the top.

    Args:
        returns (pd.DataFrame): DataFrame containing return data with columns:
            - stock_symbol
            - weekly_return
            - monthly_return
            - fytd_return
    """
    # Add period selector
    period = st.radio(
        "Select Performance Period:",
        options=["Weekly", "Monthly", "FYTD"],
        horizontal=True,
        key="performers_period_selector"
    )

    # Map selected period to column name
    metric_map = {
        "Weekly": "weekly_return",
        "Monthly": "monthly_return",
        "FYTD": "fytd_return"
    }
    metric = metric_map[period]

    # Make sure we're working with numeric values
    returns[metric] = pd.to_numeric(returns[metric], errors='coerce')

    # Sort the returns - descending for top performers, ascending for bottom performers
    top_performers = returns.nlargest(5, metric)
    bottom_performers = returns.nsmallest(5, metric)

    # Create display dataframes
    top_df = top_performers[['stock_symbol', metric]].copy()
    bottom_df = bottom_performers[['stock_symbol', metric]].copy()

    # Rename columns
    column_rename = {
        'stock_symbol': 'Symbol',
        metric: f'{period} Return'
    }

    top_df.rename(columns=column_rename, inplace=True)
    bottom_df.rename(columns=column_rename, inplace=True)

    # Style the dataframes
    def style_dataframe(df):
        return df.style.format({
            f'{period} Return': '{:+.2%}'
        }).applymap(
            lambda x: 'color: #0C9B6A' if isinstance(x, float) and x > 0 else 'color: #DC2626',
            subset=[f'{period} Return']
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

    # Apply styling
    top_styled = style_dataframe(top_df)
    bottom_styled = style_dataframe(bottom_df)

    # Create two columns for display
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Top 5 {period} Performers")
        st.dataframe(
            top_styled,
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown(f"### Bottom 5 {period} Performers")
        st.dataframe(
            bottom_styled,
            use_container_width=True,
            hide_index=True
        )


def display_metrics(processor, balances, returns_data):
    """Display enhanced metric cards with FYTD fallback if 'PORTFOLIO' is missing."""
    latest_total = balances['total_portfolio_value'].iloc[-1]
    current_cash = processor.cash_balance

    # Handle missing 'PORTFOLIO' FYTD data
    if 'PORTFOLIO' in returns_data['fytd'].columns:
        ytd_return = returns_data['fytd']['PORTFOLIO'].iloc[-1] * 100
    else:
        st.warning("No FYTD data for 'PORTFOLIO'. Check your daily_returns data.")
        ytd_return = 0

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


def display_risk_metrics(risk_metrics: dict):
    """
    Display risk metrics in a clean table format with descriptions.

    Args:
        risk_metrics (dict): Dictionary containing risk metrics
    """
    try:
        # Create metrics data
        metrics_data = {
            'Metric': [
                'Sharpe Ratio (FYTD)',
                'Market-Adjusted Alpha (Annual)',
                'Raw Alpha (Annual)',
                'Rolling Beta',
                'Tracking Error (FYTD)',
                'Treynor Ratio (FYTD)'
            ],
            'Value': [
                risk_metrics.get("sharpe", {}).get("fytd", None),
                risk_metrics.get("alpha", None),
                risk_metrics.get("raw_alpha", None),
                risk_metrics.get("beta", None),
                risk_metrics.get("tracking_error", {}).get("fytd", None),
                risk_metrics.get("treynor", {}).get("fytd", None)
            ],
            'Description': [
                'Risk-adjusted return metric measuring excess return per unit of risk using standard deviation.',
                'Portfolio\'s excess return after adjusting for market risk premium and beta.',
                'Simple excess return over risk-free rate without market adjustment.',
                'Measures portfolio sensitivity to market movements. Beta > 1 indicates higher market sensitivity.',
                'Measures how consistently the portfolio follows its benchmark. Lower values indicate closer benchmark tracking.',
                'Excess return per unit of systematic risk (beta).'
            ]
        }

        df = pd.DataFrame(metrics_data)

        # Format the values with consistent decimal places
        formatted_values = []
        for metric, value in zip(df['Metric'], df['Value']):
            if value is None:
                formatted_values.append('N/A')
            elif 'Alpha' in metric or 'Tracking Error' in metric:
                formatted_values.append(f'{value:+.2%}')
            elif 'Beta' in metric:
                formatted_values.append(f'{value:.2f}')
            elif 'Ratio' in metric:
                formatted_values.append(f'{value:.2f}')
            else:
                formatted_values.append(f'{value:.2f}')

        df['Value'] = formatted_values

        # Create styled table
        styled_df = df.style.set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('background-color', '#004F2F'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('padding', '12px 15px'),
                    ('text-align', 'left'),
                    ('border-bottom', '3px solid #FEE123')
                ]
            },
            {
                'selector': 'td',
                'props': [
                    ('padding', '12px 15px'),
                    ('border-bottom', '1px solid #e0e0e0')
                ]
            },
            {
                'selector': 'td:nth-child(2)',
                'props': [
                    ('text-align', 'center'),
                    ('font-family', 'monospace')
                ]
            }
        ]).apply(lambda x: [
            'background-color: #f8f9fa' if i % 2 == 0 else ''
            for i in range(len(x))
        ], axis=0)

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error("An error occurred while displaying risk metrics.")
        if st.checkbox("Show detailed error message"):
            st.code(traceback.format_exc())


def main():
    try:
        cached_db = init_database()
        if cached_db is None:
            return

        fiscal_start_date = date(2024, 3, 29)
        today = date.today()

        with st.spinner("Loading portfolio data..."):
            # Changed parameter from fund_id to fund
            processor = cached_db.create_processor(fiscal_start_date, today, fund=DEFAULT_FUND_ID)
            if processor is None:
                st.error("Failed to initialize data processor.")
                return

            # Get latest date from processor
            latest_date = processor.daily_returns['return_date'].max()

            # Initialize page with latest date
            init_page(latest_date)

            risk_metrics = processor.calculate_risk_metrics(fiscal_start_date)
            sharpe_ratio = processor.calculate_sharpe_ratio(fiscal_start_date)
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

        # Main cumulative returns chart - now full width
        st.markdown("### Cumulative Returns")
        st.plotly_chart(
            plot_cumulative_returns(returns_data[period.lower()]),
            use_container_width=True,
            key="main_returns_chart"
        )

        # Tabs for the rest of the content
        tabs = st.tabs(["Performance", "Holdings", "Risk Analysis"])

        # Holdings tab
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

            # Performance Analysis section with new switchable display
            st.markdown("### Performance Analysis")
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
            st.markdown("### Risk Metrics Overview")

            risk_metrics = processor.calculate_risk_metrics(fiscal_start_date)
            sharpe_metrics = processor.calculate_sharpe_ratio(fiscal_start_date)

            combined_metrics = {
                **risk_metrics,
                **sharpe_metrics
            }

            display_risk_metrics(combined_metrics)

    except Exception as e:
        st.error("An unexpected error occurred.")
        if st.checkbox("Show detailed error message"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()