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

# ----- FISCAL YEARS CONFIGURATION -----
# Using start and end dates for each fiscal year
FISCAL_YEARS = {
    "FY 2024-25": {"start": date(2024, 4, 1), "end": date(2025, 3, 31)},
    "FY 2025-26": {"start": date(2025, 4, 1), "end": date(2026, 3, 31)}
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
    # Check if DataFrame is empty or lacks required columns
    if data.empty or 'PORTFOLIO' not in data.columns or 'BENCHMARK' not in data.columns:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected period",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#666")
        )
        fig.update_layout(height=500)
        return fig

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

    # Fixed: Use .map instead of .applymap
    styled_df = display_data.style.format({
        'Shares': '{:,.0f}',
        'Price': '${:,.2f}',
        'Market Value': '${:,.2f}',
        'Weight': '{:.2%}',
        'Weekly Return': '{:+.2%}',
        'Monthly Return': '{:+.2%}',
        'FYTD Return': '{:+.2%}'
    }).map(
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
        }).map(  # Fixed: Use .map instead of .applymap
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


def display_metrics(processor, balances, returns_data, selected_fiscal_year):
    """Display enhanced metric cards with FYTD fallback if 'PORTFOLIO' is missing."""
    latest_total = balances['total_portfolio_value'].iloc[-1]
    current_cash = processor.cash_balance

    # Handle missing 'PORTFOLIO' FYTD data
    if 'fytd' in returns_data and 'PORTFOLIO' in returns_data['fytd'].columns and not returns_data['fytd'].empty:
        ytd_return = returns_data['fytd']['PORTFOLIO'].iloc[-1] * 100
    else:
        ytd_return = 0

    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-label">PORTFOLIO VALUE</div>
            <div class="metric-value">${latest_total:,.0f}</div>
            <div class="metric-subtitle">{selected_fiscal_year}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">CASH POSITION</div>
            <div class="metric-value">${current_cash:,.0f}</div>
            <div class="metric-subtitle">{(current_cash / latest_total * 100):.1f}% of Portfolio</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">FYTD RETURN</div>
            <div class="metric-value {'positive' if ytd_return >= 0 else 'negative'}">
                {ytd_return:+.1f}%
            </div>
            <div class="metric-subtitle">{selected_fiscal_year}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def plot_rolling_volatility(volatility_data: pd.DataFrame, period_name: str):
    """
    Create an enhanced rolling volatility chart.

    Args:
        volatility_data: DataFrame with columns 'return_date', 'PORTFOLIO', 'BENCHMARK'
        period_name: String indicating the calculation period (e.g., 'FYTD' or 'TTM')

    Returns:
        Plotly figure object
    """
    if volatility_data.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for volatility analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(height=400)
        return fig

    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(
        x=volatility_data['return_date'],
        y=volatility_data['PORTFOLIO'],
        name='Portfolio Volatility',
        line=dict(color='#004F2F', width=3),
        hovertemplate='%{x|%b %d, %Y}<br>Volatility: %{y:.2%}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=volatility_data['return_date'],
        y=volatility_data['BENCHMARK'],
        name='Benchmark Volatility',
        line=dict(color='#FEE123', width=3),
        hovertemplate='%{x|%b %d, %Y}<br>Volatility: %{y:.2%}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=f"30-Day Rolling Volatility ({period_name})",
        title_font=dict(size=18, color='#1A2B32'),
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor='white',
        plot_bgcolor='#F8F9FA',
        yaxis=dict(
            title='Annualized Volatility',
            tickformat='.1%',
            gridcolor='#E0E0E0',
            zerolinecolor='#E0E0E0'
        ),
        xaxis=dict(
            title='',
            gridcolor='#E0E0E0'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        hovermode='x unified'
    )

    return fig


def plot_return_distribution(distribution_data: dict, period_name: str):
    """
    Create return distribution chart showing histograms and density plots.

    Args:
        distribution_data: Dictionary with PORTFOLIO and BENCHMARK return distribution data
        period_name: String indicating the calculation period (e.g., 'FYTD' or 'TTM')

    Returns:
        Plotly figure object
    """
    if not distribution_data or 'PORTFOLIO' not in distribution_data:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for return distribution analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#666")
        )
        fig.update_layout(height=400)
        return fig

    # Create figure
    fig = go.Figure()

    # Colors
    colors = {
        'PORTFOLIO': '#004F2F',
        'BENCHMARK': '#FEE123'
    }

    # Add histogram traces
    for source in ['PORTFOLIO', 'BENCHMARK']:
        if source not in distribution_data:
            continue

        returns = distribution_data[source]['return_data']

        # Create histogram trace
        fig.add_trace(go.Histogram(
            x=returns,
            name=f"{source} Returns",
            opacity=0.6,
            marker_color=colors[source],
            xbins=dict(
                start=returns.min() - 0.005,
                end=returns.max() + 0.005,
                size=0.005
            ),
            histnorm='probability density',
            hovertemplate='Return: %{x:.2%}<br>Density: %{y:.2f}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title=f"Daily Return Distribution ({period_name})",
        title_font=dict(size=18, color='#1A2B32'),
        height=380,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor='white',
        plot_bgcolor='#F8F9FA',
        barmode='overlay',
        yaxis=dict(
            title='Density',
            gridcolor='#E0E0E0',
            zerolinecolor='#E0E0E0'
        ),
        xaxis=dict(
            title='Daily Return',
            tickformat='.1%',
            gridcolor='#E0E0E0'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        hovermode='x unified'
    )

    return fig


def display_risk_metrics(risk_metrics: dict):
    """
    Display enhanced risk metrics with visual elements and better descriptions.
    """
    try:
        # Check if we're using extended calculation period
        is_extended = risk_metrics.get('calculation_period') in ['extended', 'extended_long']
        days_used = risk_metrics.get('days_used', 0)
        period_name = risk_metrics.get('period_name', 'FYTD')
        is_reliable = risk_metrics.get('is_reliable', True)

        # Show calculation period banner with appropriate styling
        if is_extended:
            extended_start = risk_metrics.get('extended_start_date')
            extended_start_str = extended_start.strftime('%B %d, %Y') if extended_start else "previous year"

            if not is_reliable and days_used < 60:
                # Warning for very limited data
                st.warning(f"""
                ⚠️ **Limited Data Alert**: Only {days_used} trading days available.

                Risk metrics are calculated using data from {extended_start_str} to present, 
                but may not be statistically reliable. Use with caution.
                """)
            else:
                # Normal info for extended period
                st.markdown(f"""
                <div style="background-color: #E8F4F9; border-left: 4px solid #2E86C1; padding: 15px; border-radius: 4px; margin-bottom: 20px;">
                    <span style="font-weight: bold; color: #2E86C1;">📊 Extended Calculation Period</span><br>
                    Risk metrics are calculated using data from <b>{extended_start_str}</b> to present 
                    ({days_used} trading days) for statistical reliability.
                </div>
                """, unsafe_allow_html=True)

        # Create metrics data
        metrics_data = {
            'Metric': [
                'Sharpe Ratio',
                'CAPM Alpha',
                'Historical Beta',
                'Current Portfolio Beta',
                'Tracking Error'
            ],
            'Value': [
                risk_metrics.get("sharpe", {}).get("fytd", None),
                risk_metrics.get("capm_alpha", None),
                risk_metrics.get("beta", None),
                risk_metrics.get("holdings_beta", None),
                risk_metrics.get("tracking_error", {}).get("fytd", None)
            ],
            'Description': [
                'Portfolio excess return (above risk-free rate) divided by return standard deviation. Higher values indicate better risk-adjusted returns.',
                'Portfolio excess return beyond what would be predicted by the CAPM model.',
                'Historical sensitivity to market movements calculated from past returns.',
                'Current portfolio sensitivity based on weighted average of individual holdings\' betas.',
                'Standard deviation of portfolio returns relative to benchmark returns.'
            ]
        }

        df = pd.DataFrame(metrics_data)

        # Format the values with consistent decimal places
        formatted_values = []
        for metric, value in zip(df['Metric'], df['Value']):
            if value is None:
                formatted_values.append('N/A')
            elif 'Alpha' in metric or 'Tracking Error' in metric:
                formatted_values.append(f'{value:+.2%}' if value is not None else 'N/A')
            elif 'Beta' in metric:
                formatted_values.append(f'{value:.2f}' if value is not None else 'N/A')
            elif 'Ratio' in metric:
                formatted_values.append(f'{value:.2f}' if value is not None else 'N/A')
            else:
                formatted_values.append(f'{value:.2f}' if value is not None else 'N/A')

        df['Value'] = formatted_values

        # Add period indicators
        metric_periods = []
        for metric in df['Metric']:
            if metric == 'Current Portfolio Beta':
                metric_periods.append('Current')
            else:
                metric_periods.append(period_name)

        df['Period'] = metric_periods

        # Apply styles to the dataframe
        styled_df = df.style.apply(lambda x: [
            'background-color: #f8f9fa' if i % 2 == 0 else ''
            for i in range(len(x))
        ], axis=0).set_table_styles([
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
                    ('border-bottom', '1px solid #e0e0e0'),
                    ('font-size', '14px')
                ]
            },
            {
                'selector': 'td:nth-child(2)',  # Value column
                'props': [
                    ('text-align', 'center'),
                    ('font-family', 'monospace'),
                    ('font-weight', 'bold'),
                    ('font-size', '16px')
                ]
            },
            {
                'selector': 'td:nth-child(3)',  # Period column
                'props': [
                    ('text-align', 'center'),
                    ('color', '#666'),
                    ('font-style', 'italic'),
                    ('font-size', '13px')
                ]
            }
        ])

        # First render the container opening
        st.markdown("""
        <div class="risk-metrics-container">
            <h3 class="risk-metrics-title">Risk Metrics Overview</h3>
        """, unsafe_allow_html=True)

        # Then render the data table
        st.dataframe(styled_df, hide_index=True, use_container_width=True)

        # Finally close the container - NO LEGEND HERE
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error("An error occurred while displaying risk metrics.")
        if st.checkbox("Show detailed error message"):
            st.code(traceback.format_exc())

def display_fiscal_year_selector():
    """Create a subtle fiscal year selector at the top."""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 20px; margin-bottom: 10px;">
            <span style="color: #004F2F; font-size: 1rem;">📅 Select Fiscal Year</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Simple dropdown that only takes up needed space - Fixed label warning
        selected_fiscal_year = st.selectbox(
            "Fiscal Year",  # Added a label to fix warning
            options=list(FISCAL_YEARS.keys()),
            index=0,  # Default to current fiscal year (2024-25)
            key="fiscal_year_selector",
            label_visibility="collapsed"  # Hide label completely
        )

    return selected_fiscal_year


def main():
    try:
        cached_db = init_database()
        if cached_db is None:
            return

        # Add a subtle fiscal year selector at the top
        selected_fiscal_year = display_fiscal_year_selector()

        # Get the start and end dates for the selected fiscal year
        fiscal_start_date = FISCAL_YEARS[selected_fiscal_year]["start"]
        fiscal_end_date = FISCAL_YEARS[selected_fiscal_year]["end"]

        # If end date is in the future, use today's date instead
        today = date.today()
        fiscal_end_date = min(fiscal_end_date, today)

        # Get the true latest return date across all data (not limited to fiscal year)
        latest_date = cached_db.get_latest_return_date() or fiscal_end_date

        # Initialize page UI with latest data date
        init_page(latest_date)

        # Convert dates to strings for processor functions that expect strings
        fiscal_start_str = fiscal_start_date.strftime('%Y-%m-%d')
        fiscal_end_str = fiscal_end_date.strftime('%Y-%m-%d')

        with st.spinner(f"Loading portfolio data..."):
            # Create processor with fiscal year date range
            processor = cached_db.create_processor(fiscal_start_date, fiscal_end_date, fund=DEFAULT_FUND_ID)
            if processor is None:
                st.error("Failed to initialize data processor.")
                return

            # Calculate metrics using adaptive methods (60 days minimum data requirement)
            min_required_days = 60
            risk_metrics = processor.calculate_adaptive_risk_metrics(fiscal_start_str, fiscal_end_str, min_required_days)
            sharpe_ratio = processor.calculate_adaptive_sharpe_ratio(fiscal_start_str, fiscal_end_str, min_required_days)

            # Other calculations
            returns_data = processor.calculate_cumulative_returns(fiscal_start_str, fiscal_end_str)
            holdings = processor.holdings
            prices = processor.stock_prices
            equity_returns = processor.calculate_equity_returns(fiscal_start_str, fiscal_end_str)

            # Volatility & return distribution (match extended period if used)
            if risk_metrics.get('calculation_period') in ['extended', 'extended_long'] and risk_metrics.get('extended_start_date'):
                extended_start = risk_metrics['extended_start_date']
                extended_start_str = extended_start.strftime('%Y-%m-%d')
                rolling_volatility = processor.calculate_rolling_volatility(30, extended_start_str, fiscal_end_str)
                return_distribution = processor.calculate_return_distribution(extended_start_str, fiscal_end_str)
            else:
                rolling_volatility = processor.calculate_rolling_volatility(30, fiscal_start_str, fiscal_end_str)
                return_distribution = processor.calculate_return_distribution(fiscal_start_str, fiscal_end_str)

            balances = cached_db.get_balances(DEFAULT_FUND_ID, fiscal_start_str, fiscal_end_str)

        # All your existing display logic remains unchanged
        display_metrics(processor, balances, returns_data, selected_fiscal_year)

        period = st.radio("Select Return Period:", options=["Weekly", "Monthly", "FYTD"], horizontal=True, index=2, key="period_selector")
        st.markdown("### Cumulative Returns")
        if period.lower() in returns_data and not returns_data[period.lower()].empty:
            st.plotly_chart(plot_cumulative_returns(returns_data[period.lower()]), use_container_width=True, key="main_returns_chart")
        else:
            st.info(f"No {period} returns data available for {selected_fiscal_year}")

        tabs = st.tabs(["Performance", "Holdings", "Risk Analysis"])

        # Tab 1 – Holdings
        with tabs[1]:
            col3, col4 = st.columns([1, 2])
            with col3:
                st.markdown("### Portfolio Allocation")
                st.plotly_chart(create_allocation_chart(holdings, prices, processor.cash_balance), use_container_width=True, key="allocation_pie_chart")
            with col4:
                st.markdown("### Holdings Detail")
                if not holdings.empty and not equity_returns.empty:
                    display_holdings_table(holdings, equity_returns, prices)
                else:
                    st.info("No holdings data available for the selected period")
            st.markdown("### Performance Analysis")
            if not equity_returns.empty:
                display_top_bottom_performers(equity_returns)
            else:
                st.info("No performance data available for the selected period")

        # Tab 0 – Performance
        with tabs[0]:
            st.markdown("### Performance Summary")
            if all(not returns_data.get(p.lower(), pd.DataFrame()).empty for p in ['weekly', 'monthly', 'fytd']):
                perf_data = pd.DataFrame({
                    'Period': ['Weekly', 'Monthly', 'FYTD'],
                    'Portfolio': [returns_data[p.lower()]['PORTFOLIO'].iloc[-1] for p in ['weekly', 'monthly', 'fytd']],
                    'Benchmark': [returns_data[p.lower()]['BENCHMARK'].iloc[-1] for p in ['weekly', 'monthly', 'fytd']],
                    'Excess Return': [returns_data[p.lower()]['PORTFOLIO'].iloc[-1] - returns_data[p.lower()]['BENCHMARK'].iloc[-1] for p in ['weekly', 'monthly', 'fytd']]
                })

                st.dataframe(
                    perf_data.style.format({
                        'Portfolio': '{:+.2%}',
                        'Benchmark': '{:+.2%}',
                        'Excess Return': '{:+.2%}'
                    }).map(
                        lambda x: 'color: #0C9B6A' if isinstance(x, float) and x > 0 else 'color: #DC2626',
                        subset=['Portfolio', 'Benchmark', 'Excess Return']
                    ),
                    use_container_width=True,
                    hide_index=True,
                    key="performance_summary_table"
                )
            else:
                st.info("Insufficient performance data available for the selected fiscal year")

        # Tab 2 – Risk Analysis
        with tabs[2]:
            if risk_metrics and sharpe_ratio:
                combined_metrics = {**risk_metrics, **sharpe_ratio}
                col1, col2 = st.columns(2)

                with col1:
                    display_risk_metrics(combined_metrics)

                with col2:
                    viz_option = st.radio("Select Risk Visualization:", options=["Volatility", "Return Distribution"], horizontal=True, key="risk_viz_selector")
                    period_name = combined_metrics.get('period_name', 'FYTD')

                    if viz_option == "Volatility":
                        if not rolling_volatility.empty:
                            st.plotly_chart(plot_rolling_volatility(rolling_volatility, period_name), use_container_width=True, key="volatility_chart")
                        else:
                            st.info(f"Insufficient data for volatility analysis in {selected_fiscal_year}. Need at least 30 days.")
                    else:
                        if return_distribution and 'PORTFOLIO' in return_distribution:
                            st.plotly_chart(plot_return_distribution(return_distribution, period_name), use_container_width=True, key="distribution_chart")
                        else:
                            st.info(f"Insufficient data for return distribution in {selected_fiscal_year}. Need at least 30 days.")

                with st.expander("📊 Risk Metrics Interpretation"):
                    st.markdown("""
                    ### Understanding Risk Metrics

                    **Sharpe Ratio**: Measures risk-adjusted return. Higher values indicate better return per unit of risk.
                    - Value > 1: Good
                    - Value > 2: Very good
                    - Value < 0: Investment underperformed the risk-free rate

                    **CAPM Alpha**: Excess return above what would be predicted based on market exposure (beta).
                    - Positive alpha indicates outperformance
                    - Negative alpha indicates underperformance

                    **Beta**: Measures sensitivity to market movements.
                    - Beta = 1: Moves with the market
                    - Beta > 1: More volatile than market
                    - Beta < 1: Less volatile than market

                    **Tracking Error**: Measures how closely the portfolio follows the benchmark.
                    - Lower values indicate closer adherence to benchmark
                    - Higher values suggest active management or style drift
                    """)
            else:
                st.info("Risk metrics not available for the selected fiscal year")

    except Exception as e:
        st.error("An unexpected error occurred.")
        if st.checkbox("Show detailed error message"):
            st.code(traceback.format_exc())


# Add enhanced custom CSS for all components
def add_custom_css():
    st.markdown("""
    <style>
    /* Make fiscal year selector more compact */
    [data-testid="stSelectbox"] {
        max-width: 200px !important;
    }

    /* Update metric styles to show fiscal year in subtitle */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #004F2F;
        border-radius: 8px;
        padding: 20px;
        flex: 1;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: bold;
        margin-bottom: 8px;
        color: #FEE123;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .metric-value.positive {
        color: #6EE7B7;
    }
    .metric-value.negative {
        color: #FCA5A5;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        opacity: 0.8;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0 0;
        border: 1px solid #eee;
        border-bottom: none;
        color: #1A2B32;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #004F2F !important;
        color: white !important;
    }

    /* Table header styling */
    thead th {
        background-color: #1A2B32 !important;
        color: white !important;
        font-weight: 600 !important;
        text-align: left !important;
        padding: 1rem !important;
        border-bottom: 3px solid #FEE123 !important;
    }

    /* Header styling */
    h3 {
        color: #1A2B32;
        border-bottom: 2px solid #004F2F;
        padding-bottom: 8px;
        margin-top: 25px;
    }

    /* Radio button styling */
    .stRadio > div {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 8px 15px;
        margin-bottom: 15px;
    }

    /* Risk metrics styling */
    .risk-metrics-container {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .risk-metrics-title {
        color: #1A2B32;
        margin-bottom: 15px;
        font-weight: 600;
        border-bottom: 2px solid #004F2F;
        padding-bottom: 8px;
    }
    .risk-metrics-table {
        margin-bottom: 15px;
    }
    .risk-metrics-table table {
        width: 100%;
        border-collapse: collapse;
    }
    .risk-metrics-legend {
        display: flex;
        justify-content: flex-start;
        gap: 20px;
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px solid #eee;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .legend-bullet {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    .legend-text {
        color: #666;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    add_custom_css()
    main()