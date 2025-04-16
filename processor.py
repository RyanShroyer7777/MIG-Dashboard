import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from datetime import datetime, date, timedelta
from typing import Tuple, Dict
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import traceback
import pandas_market_calendars as mcal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DataProcessor:
    def __init__(
            self,
            daily_returns: pd.DataFrame,
            stock_daily_returns: pd.DataFrame,
            holdings: pd.DataFrame,
            risk_free_rates: pd.DataFrame,
            stock_prices: pd.DataFrame,
            fund: str,
            cash_balance: float
    ):
        self.fund = fund
        self.cash_balance = max(cash_balance, 0)
        self.nyse = mcal.get_calendar('NYSE')

        # Convert date columns to tz-aware datetime (UTC)
        daily_returns['return_date'] = pd.to_datetime(daily_returns['return_date']).dt.tz_localize('UTC')
        stock_daily_returns['return_date'] = pd.to_datetime(stock_daily_returns['return_date']).dt.tz_localize('UTC')
        risk_free_rates['date'] = pd.to_datetime(risk_free_rates['date']).dt.tz_localize('UTC')
        stock_prices['price_date'] = pd.to_datetime(stock_prices['price_date']).dt.tz_localize('UTC')

        self.daily_returns = daily_returns
        self.stock_daily_returns = stock_daily_returns
        self.holdings = holdings
        self.risk_free_rates = risk_free_rates
        self.stock_prices = stock_prices

        self._validate_data()

    def _get_period_dates(self, current_date: pd.Timestamp) -> dict:
        if current_date.tzinfo is None or current_date.tzinfo.utcoffset(current_date) is None:
            current_date = current_date.tz_localize('UTC')

        trading_days = self.nyse.valid_days(
            start_date=current_date - pd.Timedelta(days=35),
            end_date=current_date
        )

        latest_trading_day = trading_days[trading_days <= current_date][-1]

        week_end = latest_trading_day
        while week_end.weekday() != 4:
            week_end -= pd.Timedelta(days=1)

        week_start = self.nyse.valid_days(
            start_date=week_end - pd.Timedelta(days=7),
            end_date=week_end
        )[0]

        month_end = latest_trading_day
        month_start = self.nyse.valid_days(
            start_date=month_end.replace(day=1) - pd.Timedelta(days=1),
            end_date=month_end
        )[0]

        return {
            'weekly': (week_start, week_end),
            'monthly': (month_start, month_end),
            'latest': latest_trading_day
        }

    def _validate_data(self) -> None:
        required_columns = {
            'daily_returns': ['return_date', 'return_value', 'source'],
            'stock_daily_returns': ['return_date', 'return_value', 'stock_symbol'],
            'holdings': ['stock_symbol', 'shares', 'cost_basis', 'fund', 'date'],
            'risk_free_rates': ['date', 'rate'],
            'stock_prices': ['stock_symbol', 'current_price', 'price_date']
        }

        column_mappings = {
            'holdings': {
                'shares_held': 'shares',
                'average_cost': 'cost_basis',
                'fund_id': 'fund',
                'open_date': 'date'
            }
        }

        for df_name, columns in required_columns.items():
            df = getattr(self, df_name)
            missing_cols = set(columns) - set(df.columns)

            if missing_cols and df_name in column_mappings:
                for missing_col in list(missing_cols):
                    for old_col, new_col in column_mappings[df_name].items():
                        if new_col == missing_col and old_col in df.columns:
                            df[new_col] = df[old_col]
                            missing_cols.remove(new_col)

            if missing_cols:
                raise ValueError(f"{df_name} is missing required columns: {missing_cols}")

    def calculate_portfolio_allocation(self) -> pd.DataFrame:
        allocation = self.holdings.merge(
            self.stock_prices[['stock_symbol', 'current_price']],
            on='stock_symbol',
            how='left'
        )
        allocation['market_value'] = allocation['shares'] * allocation['current_price']

        cash_allocation = pd.DataFrame({'stock_symbol': ['CASH'], 'market_value': [self.cash_balance]})
        allocation = pd.concat([allocation, cash_allocation], ignore_index=True)

        total_value = allocation['market_value'].sum()
        allocation['weight'] = allocation['market_value'] / total_value

        return allocation[['stock_symbol', 'market_value', 'weight']]

    def calculate_equity_returns(self, fiscal_start_date: str) -> pd.DataFrame:
        latest_date = self.stock_daily_returns['return_date'].max()
        periods = {
            "weekly": latest_date - pd.Timedelta(days=7),
            "monthly": latest_date - pd.Timedelta(days=30),
            "fytd": pd.Timestamp(fiscal_start_date).tz_localize('UTC')
        }

        results = []
        for stock in self.holdings['stock_symbol'].unique():
            stock_data = self.stock_daily_returns[self.stock_daily_returns['stock_symbol'] == stock]
            acquisition_date = pd.to_datetime(
                self.holdings[self.holdings['stock_symbol'] == stock]['date'].iloc[0]
            ).tz_localize('UTC')

            stock_results = {"stock_symbol": stock}

            for period, start_date in periods.items():
                period_start = max(start_date, acquisition_date)
                period_data = stock_data[
                    (stock_data['return_date'] >= period_start) & (stock_data['return_date'] <= latest_date)]

                period_data = period_data.drop_duplicates(subset=['return_date'])
                cumulative_return = (1 + period_data['return_value']).prod() - 1 if not period_data.empty else np.nan
                stock_results[f"{period}_return"] = cumulative_return

            results.append(stock_results)

        return pd.DataFrame(results)

    def calculate_tracking_error(self, fiscal_start_date: str) -> dict:
        returns = self.daily_returns.pivot(index='return_date', columns='source', values='return_value').dropna()
        today = pd.Timestamp.now(tz='UTC').normalize()
        periods = {
            'weekly': today - pd.Timedelta(days=7),
            'monthly': today - pd.Timedelta(days=30),
            'fytd': pd.Timestamp(fiscal_start_date).tz_localize('UTC')
        }

        tracking_errors = {}
        for period_name, start_date in periods.items():
            subset = returns[(returns.index >= start_date) & (returns.index <= today)]
            if subset.empty or len(subset) < 2:
                tracking_errors[period_name] = None
                continue

            diff = subset['PORTFOLIO'] - subset['BENCHMARK']
            tracking_errors[period_name] = diff.std() * np.sqrt(252)

        return tracking_errors

    def _get_latest_risk_free_rate(self) -> tuple[float, float]:
        try:
            if self.risk_free_rates.empty:
                return 0.0, 0.0

            latest_rate = self.risk_free_rates.sort_values('date', ascending=False).iloc[0]['rate']
            annual_rate = latest_rate
            daily_rate = (1 + annual_rate) ** (1 / 252) - 1
            return annual_rate, daily_rate

        except Exception as e:
            return 0.0, 0.0

    def calculate_risk_metrics(self, fiscal_start_date: str) -> dict:
        try:
            annual_rf_rate, daily_rf_rate = self._get_latest_risk_free_rate()
            returns = self.daily_returns.pivot(index='return_date', columns='source', values='return_value').fillna(0)

            portfolio_returns = returns['PORTFOLIO']
            benchmark_returns = returns['BENCHMARK']

            excess_portfolio = portfolio_returns - daily_rf_rate
            excess_benchmark = benchmark_returns - daily_rf_rate

            reg = LinearRegression()
            reg.fit(excess_benchmark.values.reshape(-1, 1), excess_portfolio.values)
            beta = reg.coef_[0]
            market_alpha = reg.intercept_ * 252

            ann_portfolio_return = (1 + portfolio_returns.mean()) ** 252 - 1
            raw_alpha = (ann_portfolio_return - annual_rf_rate) / 10

            tracking_error = (
                (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
                if len(portfolio_returns) > 1 else 0
            )

            return {
                'alpha': market_alpha,
                'raw_alpha': raw_alpha,
                'beta': beta,
                'tracking_error': {'fytd': tracking_error},
                'r_squared': reg.score(excess_benchmark.values.reshape(-1, 1), excess_portfolio.values)
            }
        except Exception as e:
            return {
                'alpha': 0,
                'raw_alpha': 0,
                'beta': 0,
                'tracking_error': {'fytd': 0},
                'r_squared': 0
            }

    def calculate_sharpe_ratio(self, fiscal_start_date: date) -> dict:
        try:
            annual_rf_rate, daily_rf_rate = self._get_latest_risk_free_rate()
            fiscal_start = pd.Timestamp(fiscal_start_date).tz_localize('UTC')

            portfolio_data = self.daily_returns[
                (self.daily_returns['source'] == 'PORTFOLIO') &
                (self.daily_returns['return_date'] >= fiscal_start)
                ]

            if portfolio_data.empty:
                return {'sharpe': {'fytd': None}, 'treynor': {'fytd': None}}

            portfolio_returns = portfolio_data['return_value']
            excess_returns = portfolio_returns - daily_rf_rate

            sharpe = (
                excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                if excess_returns.std() > 0 else None
            )

            risk_metrics = self.calculate_risk_metrics(fiscal_start_date)
            beta = risk_metrics['beta']
            treynor_ratio = (excess_returns.mean() * 252) / beta if beta != 0 else None

            return {
                'sharpe': {'fytd': sharpe},
                'treynor': {'fytd': treynor_ratio}
            }

        except Exception as e:
            return {'sharpe': {'fytd': None}, 'treynor': {'fytd': None}}

    def calculate_cumulative_returns(self, fiscal_start_date: str) -> dict:
        """
        Calculate cumulative returns for different time periods.
        Returns a dictionary with period names as keys and DataFrames of cumulative returns as values.

        Args:
            fiscal_start_date: Starting date of the fiscal year

        Returns:
            Dictionary with keys 'weekly', 'monthly', 'fytd', each containing a DataFrame of cumulative returns
        """
        daily_returns = self.daily_returns.copy()

        # Validate data
        if 'return_date' not in daily_returns.columns or 'source' not in daily_returns.columns or 'return_value' not in daily_returns.columns:
            raise ValueError("daily_returns must have 'return_date', 'source', and 'return_value' columns")

        # Verify required sources exist
        required_sources = ['PORTFOLIO', 'BENCHMARK']
        available_sources = daily_returns['source'].unique()
        missing_sources = set(required_sources) - set(available_sources)
        if missing_sources:
            logging.warning(f"Missing required sources: {missing_sources}")

        # Sort by date to ensure correct cumulative calculation
        daily_returns = daily_returns.sort_values('return_date')

        current_date = daily_returns['return_date'].max()
        periods = self._get_period_dates(current_date)
        results = {}

        # Define reusable calculation function
        def calculate_period_returns(period_data):
            if period_data.empty:
                logging.warning("No data available for the specified period")
                return pd.DataFrame()

            period_pivot = period_data.pivot(
                index='return_date',
                columns='source',
                values='return_value'
            )
            # Don't fillna with 0, which could artificially deflate returns
            # Instead, let cumprod handle NaN values naturally
            return (1 + period_pivot).cumprod() - 1

        # Calculate weekly and monthly returns
        for period_name in ['weekly', 'monthly']:
            start_date, end_date = periods[period_name]
            period_data = daily_returns[
                (daily_returns['return_date'] >= start_date) &
                (daily_returns['return_date'] <= end_date)
                ]

            # Log warning if data appears incomplete
            expected_days = len(self.nyse.valid_days(start_date=start_date, end_date=end_date))
            if len(period_data) < expected_days * 0.95:
                logging.warning(f"Incomplete data for {period_name}: {len(period_data)} vs expected {expected_days}")

            results[period_name] = calculate_period_returns(period_data)

        # Calculate FYTD returns
        fytd_start = pd.to_datetime(fiscal_start_date).tz_localize('UTC')
        fytd_data = daily_returns[daily_returns['return_date'] >= fytd_start]
        results['fytd'] = calculate_period_returns(fytd_data)

        return results