import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any
from processor import DataProcessor
from database import DatabaseInterface
import traceback


class CacheConfig:
    """Configuration for cache TTLs."""
    DB_CACHE_TTL = 7 * 24 * 3600  # 1 week for raw database queries
    PROCESSED_CACHE_TTL = 24 * 3600  # 1 day for processed results
    PRICE_CACHE_TTL = 24 * 3600  # 1 day for stock prices

    # Keys for storing last update timestamps
    LAST_DB_UPDATE = "last_database_update"
    LAST_PROCESSED_UPDATE = "last_processed_update"


@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_daily_returns(_db: DatabaseInterface, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch daily returns from the database."""
    return _db.get_daily_returns(start_date, end_date)


@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_stock_daily_returns(_db: DatabaseInterface, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch stock daily returns from the database."""
    return _db.get_stock_daily_returns(start_date, end_date)


@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_holdings(_db: DatabaseInterface, fund_id: str) -> pd.DataFrame:
    """Fetch holdings for a specific fund."""
    return _db.get_holdings(fund_id)


@st.cache_data(ttl=CacheConfig.PRICE_CACHE_TTL)
def get_stock_prices(_db: DatabaseInterface) -> pd.DataFrame:
    """Fetch the latest stock prices."""
    return _db.get_stock_prices()


@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_risk_free_rates(_db: DatabaseInterface, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch risk-free rates for the given date range."""
    return _db.get_risk_free_rates(start_date, end_date)


@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_balances(_db: DatabaseInterface, fund_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch balances data for a specific fund."""
    return _db.get_balances(fund_id, start_date, end_date)


@st.cache_data(ttl=CacheConfig.DB_CACHE_TTL)
def get_cash_balance(_db: DatabaseInterface, fund_id: str, start_date: date, end_date: date) -> float:
    balances = get_balances(_db, fund_id, start_date, end_date)
    if balances.empty or 'cash_balance' not in balances.columns:
        st.warning("No cash balance data available; defaulting to 0.")
        return 0.0
    return balances.iloc[-1]['cash_balance']



@st.cache_data(ttl=CacheConfig.PROCESSED_CACHE_TTL)
def calculate_cumulative_returns(daily_returns: pd.DataFrame, fiscal_start_date: str) -> Dict[str, pd.DataFrame]:
    """Calculate cumulative returns for different time periods."""
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
    """Calculate equity returns for different time periods."""
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
    """Calculate portfolio allocation including cash position."""
    allocation = holdings.merge(
        stock_prices[['stock_symbol', 'current_price']],
        on='stock_symbol',
        how='left'
    )
    
    allocation['market_value'] = allocation['shares_held'] * allocation['current_price']
    allocation = pd.concat([
        allocation,
        pd.DataFrame({'stock_symbol': ['CASH'], 'market_value': [cash_balance]})
    ])
    
    total_value = allocation['market_value'].sum()
    allocation['weight'] = allocation['market_value'] / total_value
    
    return allocation[['stock_symbol', 'market_value', 'weight']]


class CachedDatabaseInterface:
    """Manages cached data for database operations."""
    def __init__(self, db_url: str):
        """Initialize with database connection."""
        self.db = DatabaseInterface(db_url)

        # Initialize session state for tracking updates
        if CacheConfig.LAST_DB_UPDATE not in st.session_state:
            st.session_state[CacheConfig.LAST_DB_UPDATE] = datetime.now()
        if CacheConfig.LAST_PROCESSED_UPDATE not in st.session_state:
            st.session_state[CacheConfig.LAST_PROCESSED_UPDATE] = datetime.now()

    def create_processor(self, start_date: date, end_date: date, fund_id: str) -> Optional[DataProcessor]:
        """Create a DataProcessor instance with cached data."""
        try:
            daily_returns = get_daily_returns(self.db, start_date, end_date)
            stock_daily_returns = get_stock_daily_returns(self.db, start_date, end_date)
            holdings = get_holdings(self.db, fund_id)
            risk_free_rates = get_risk_free_rates(self.db, start_date, end_date)
            stock_prices = get_stock_prices(self.db)
            cash_balance = get_cash_balance(self.db, fund_id, start_date, end_date)

            return DataProcessor(
                daily_returns=daily_returns,
                stock_daily_returns=stock_daily_returns,
                holdings=holdings,
                risk_free_rates=risk_free_rates,
                stock_prices=stock_prices,
                fund_id=fund_id,
                cash_balance=cash_balance
            )
        except Exception as e:
            st.error(f"Error creating processor: {str(e)}")
            st.code(traceback.format_exc())
            return None

    def get_balances(self, fund_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get cached balances data."""
        return get_balances(self.db, fund_id, start_date, end_date)

    def force_refresh_database(self) -> None:
        """Force refresh all database caches."""
        st.cache_data.clear()
        st.session_state[CacheConfig.LAST_DB_UPDATE] = datetime.now()
        st.session_state[CacheConfig.LAST_PROCESSED_UPDATE] = datetime.now()

    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status information."""
        return {
            'last_db_update': st.session_state[CacheConfig.LAST_DB_UPDATE],
            'last_processed_update': st.session_state[CacheConfig.LAST_PROCESSED_UPDATE],
            'next_db_update': st.session_state[CacheConfig.LAST_DB_UPDATE] + timedelta(seconds=CacheConfig.DB_CACHE_TTL),
            'next_processed_update': st.session_state[CacheConfig.LAST_PROCESSED_UPDATE] + timedelta(seconds=CacheConfig.PROCESSED_CACHE_TTL)
        }

