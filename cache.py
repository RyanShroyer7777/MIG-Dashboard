import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any
from processor import DataProcessor
from database import DatabaseInterface
import traceback
import logging

class CacheConfig:
    """Configuration for cache TTLs."""
    DB_CACHE_TTL = 7 * 24 * 3600  # 1 week for raw database queries
    PROCESSED_CACHE_TTL = 24 * 3600  # 1 day for processed results
    PRICE_CACHE_TTL = 24 * 3600  # 1 day for stock prices

    # Keys for storing last update timestamps
    LAST_DB_UPDATE = "last_database_update"
    LAST_PROCESSED_UPDATE = "last_processed_update"

@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_combined_returns(_db: DatabaseInterface, start_date: date, end_date: date, fund: str = "DADCO") -> pd.DataFrame:
    portfolio_returns = _db.get_portfolio_returns(start_date, end_date, fund)
    benchmark_returns = _db.get_benchmark_returns(start_date, end_date)
    return _db.combine_portfolio_benchmark_returns(portfolio_returns, benchmark_returns)

@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def check_data_availability(_db: DatabaseInterface, start_date: date, end_date: date, fund: str = "DADCO") -> tuple:
    """
    Check if sufficient data is available for the specified date range.
    Returns (boolean, count) indicating if enough data is available and how many days.
    """
    portfolio_returns = _db.get_portfolio_returns(start_date, end_date, fund)
    unique_days = len(portfolio_returns.drop_duplicates(subset=['return_date']))
    # Consider enough data as at least 60 trading days
    return (unique_days >= 60, unique_days)


@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_stock_daily_returns(_db: DatabaseInterface, start_date: date, end_date: date) -> pd.DataFrame:
    return _db.get_stock_daily_returns(start_date, end_date)

@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_holdings(_db: DatabaseInterface, fund: str = "DADCO") -> pd.DataFrame:
    return _db.get_holdings(fund)

@st.cache_data(ttl=CacheConfig.PRICE_CACHE_TTL)
def get_stock_prices(_db: DatabaseInterface) -> pd.DataFrame:
    return _db.get_stock_prices()

@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_risk_free_rates(_db: DatabaseInterface, start_date: date, end_date: date) -> pd.DataFrame:
    return _db.get_risk_free_rates(start_date, end_date)

@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_balances(_db: DatabaseInterface, fund: str = "DADCO", start_date: str = None, end_date: str = None) -> pd.DataFrame:
    return _db.get_balances(fund, start_date, end_date)

@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_cash_balance(_db: DatabaseInterface, fund: str = "DADCO", start_date: date = None, end_date: date = None) -> float:
    """
    Get the latest cash balance for a fund.
    If no data is available, warn and return 0.0.
    """
    balances = get_balances(_db, fund, start_date, end_date)
    if balances.empty or 'cash_balance' not in balances.columns:
        st.warning("No cash balance data available; defaulting to 0.")
        return 0.0
    return balances.iloc[-1]['cash_balance']

@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_benchmark_weights(_db: DatabaseInterface, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch benchmark weights for the given date range."""
    return _db.get_benchmark_weights(start_date, end_date)

@st.cache_data(ttl=CacheConfig.PROCESSED_CACHE_TTL)
def calculate_cumulative_returns(daily_returns: pd.DataFrame, fiscal_start_date: str) -> Dict[str, pd.DataFrame]:
    daily_returns = daily_returns.copy()
    daily_returns['return_date'] = pd.to_datetime(daily_returns['return_date'])
    results = {}
    periods = {
        'weekly': 7,
        'monthly': 30,
        'fytd': None  # Will use fiscal_start_date
    }
    for period, days in periods.items():
        if days:
            start_date = datetime.now().date() - timedelta(days=days)
        else:
            start_date = pd.to_datetime(fiscal_start_date).date()
        mask = daily_returns['return_date'].dt.date >= start_date
        period_returns = daily_returns[mask].copy()
        if not period_returns.empty:
            results[period] = period_returns.pivot(
                index='return_date',
                columns='source',
                values='return_value'
            ).fillna(0)
            results[period].columns = ['portfolio', 'benchmark']
    return results

@st.cache_data(ttl=CacheConfig.PROCESSED_CACHE_TTL)
def calculate_equity_returns(stock_daily_returns: pd.DataFrame, holdings: pd.DataFrame, fiscal_start_date: str) -> pd.DataFrame:
    stock_daily_returns = stock_daily_returns.copy()
    stock_daily_returns['return_date'] = pd.to_datetime(stock_daily_returns['return_date'])
    results = []
    periods = {
        'weekly': 7,
        'monthly': 30,
        'fytd': None
    }
    for stock in holdings['stock_symbol'].unique():
        stock_data = stock_daily_returns[stock_daily_returns['stock_symbol'] == stock]
        stock_results = {'stock_symbol': stock}
        for period, days in periods.items():
            if days:
                start_date = datetime.now().date() - timedelta(days=days)
            else:
                start_date = pd.to_datetime(fiscal_start_date).date()
            mask = stock_data['return_date'].dt.date >= start_date
            period_data = stock_data[mask]
            cumulative_return = (1 + period_data['return_value']).prod() - 1 if not period_data.empty else 0
            stock_results[f'{period}_return'] = cumulative_return
        results.append(stock_results)
    return pd.DataFrame(results)

@st.cache_data(ttl=CacheConfig.PROCESSED_CACHE_TTL)
def calculate_portfolio_allocation(holdings: pd.DataFrame, stock_prices: pd.DataFrame, cash_balance: float) -> pd.DataFrame:
    allocation = holdings.merge(
        stock_prices[['stock_symbol', 'current_price']],
        on='stock_symbol',
        how='left'
    )
    shares_col = 'shares' if 'shares' in allocation.columns else 'shares_held'
    allocation['market_value'] = allocation[shares_col] * allocation['current_price']
    allocation = pd.concat([
        allocation,
        pd.DataFrame({'stock_symbol': ['CASH'], 'market_value': [cash_balance]})
    ])
    total_value = allocation['market_value'].sum()
    allocation['weight'] = allocation['market_value'] / total_value
    return allocation[['stock_symbol', 'market_value', 'weight']]

def calculate_active_weights(portfolio_allocation: pd.DataFrame, benchmark_weights: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """
    Calculate active weights by comparing portfolio allocation with benchmark weights for a given target date.
    """
    benchmark_on_date = benchmark_weights[benchmark_weights['date'] == pd.to_datetime(target_date)].rename(
        columns={'weight': 'benchmark_weight'}
    )
    merged = portfolio_allocation.merge(
        benchmark_on_date[['stock_symbol', 'benchmark_weight']],
        on='stock_symbol',
        how='left'
    )
    merged['active_weight'] = merged['weight'] - merged['benchmark_weight']
    return merged

class CachedDatabaseInterface:
    """Manages cached data for database operations."""
    def __init__(self, db_url: str):
        self.db = DatabaseInterface(db_url)
        if CacheConfig.LAST_DB_UPDATE not in st.session_state:
            st.session_state[CacheConfig.LAST_DB_UPDATE] = datetime.now()
        if CacheConfig.LAST_PROCESSED_UPDATE not in st.session_state:
            st.session_state[CacheConfig.LAST_PROCESSED_UPDATE] = datetime.now()

    def create_processor(self, fiscal_start_date: date, fiscal_end_date: date, fund: str = "DADCO") -> Optional[
        DataProcessor]:
        """
        Create a data processor with extended data range to ensure enough data for risk metrics.
        """
        try:
            # Default end date to today if not specified
            end_date = fiscal_end_date if fiscal_end_date else datetime.now().date()

            # Always load at least 12 months of data (enough for risk calculations)
            extended_start_date = fiscal_start_date - timedelta(days=365)  # 12 months
            logging.info(f"Loading extended data range: {extended_start_date} to {end_date}")

            # Load data based on determined date range
            daily_returns = get_combined_returns(self.db, extended_start_date, end_date, fund)
            stock_daily_returns = get_stock_daily_returns(self.db, extended_start_date, end_date)
            holdings = get_holdings(self.db, fund)
            risk_free_rates = get_risk_free_rates(self.db, extended_start_date, end_date)
            stock_prices = get_stock_prices(self.db)
            cash_balance = get_cash_balance(self.db, fund, fiscal_start_date, end_date)

            processor = DataProcessor(
                daily_returns=daily_returns,
                stock_daily_returns=stock_daily_returns,
                holdings=holdings,
                risk_free_rates=risk_free_rates,
                stock_prices=stock_prices,
                fund=fund,
                cash_balance=cash_balance
            )

            # Add fiscal year information for UI
            processor.fiscal_start_date = fiscal_start_date
            processor.fiscal_end_date = end_date

            return processor

        except Exception as e:
            st.error(f"Error creating processor: {str(e)}")
            st.code(traceback.format_exc())
            return None

    def get_balances(self, fund: str = "DADCO", start_date: str = None, end_date: str = None) -> pd.DataFrame:
        return get_balances(self.db, fund, start_date, end_date)

    def get_benchmark_weights(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Return benchmark weights using the cached function."""
        return get_benchmark_weights(self.db, start_date, end_date)

    def force_refresh_database(self) -> None:
        st.cache_data.clear()
        st.session_state[CacheConfig.LAST_DB_UPDATE] = datetime.now()
        st.session_state[CacheConfig.LAST_PROCESSED_UPDATE] = datetime.now()

    def get_cache_status(self) -> Dict[str, Any]:
        return {
            'last_db_update': st.session_state[CacheConfig.LAST_DB_UPDATE],
            'last_processed_update': st.session_state[CacheConfig.LAST_PROCESSED_UPDATE],
            'next_db_update': st.session_state[CacheConfig.LAST_DB_UPDATE] + timedelta(seconds=CacheConfig.DB_CACHE_TTL),
            'next_processed_update': st.session_state[CacheConfig.LAST_PROCESSED_UPDATE] + timedelta(seconds=CacheConfig.PROCESSED_CACHE_TTL)
        }

    def get_latest_return_date(self) -> Optional[date]:
        try:
            returns = self.db.get_portfolio_returns(
                start_date=date(2000, 1, 1),
                end_date=date.today(),
                fund="DADCO"
            )
            if not returns.empty:
                latest = returns['return_date'].max()
                return latest
            else:
                st.warning("Portfolio returns query returned an empty DataFrame.")
                return None
        except Exception as e:
            st.warning(f"Could not fetch latest return date: {e}")
            return None
