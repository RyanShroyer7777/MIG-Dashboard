import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from datetime import datetime, date, timedelta
from typing import Tuple, Dict, Optional, List, Union
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

    def calculate_equity_returns(self, fiscal_start_date: str, fiscal_end_date: str = None) -> pd.DataFrame:
        """
        Calculate equity returns for different time periods with end date support.

        Args:
            fiscal_start_date: Starting date of the fiscal year
            fiscal_end_date: Ending date of the fiscal year (optional)

        Returns:
            DataFrame with equity returns data
        """
        latest_date = self.stock_daily_returns['return_date'].max()

        # Use provided end date if specified, otherwise use latest date
        if fiscal_end_date:
            end_date = pd.Timestamp(fiscal_end_date).tz_localize('UTC')
            # Make sure end_date doesn't exceed latest available data
            end_date = min(end_date, latest_date)
        else:
            end_date = latest_date

        periods = {
            "weekly": end_date - pd.Timedelta(days=7),
            "monthly": end_date - pd.Timedelta(days=30),
            "fytd": pd.Timestamp(fiscal_start_date).tz_localize('UTC')
        }

        results = []
        for stock in self.holdings['stock_symbol'].unique():
            stock_data = self.stock_daily_returns[self.stock_daily_returns['stock_symbol'] == stock]
            acquisition_date = pd.to_datetime(
                self.holdings[self.holdings['stock_symbol'] == stock]['date'].iloc[0]
            ).tz_localize('UTC')

            # Filter data to respect end date
            stock_data = stock_data[stock_data['return_date'] <= end_date]

            stock_results = {"stock_symbol": stock}

            for period, start_date in periods.items():
                period_start = max(start_date, acquisition_date)
                period_data = stock_data[
                    (stock_data['return_date'] >= period_start) & (stock_data['return_date'] <= end_date)]

                period_data = period_data.drop_duplicates(subset=['return_date'])
                cumulative_return = (1 + period_data['return_value']).prod() - 1 if not period_data.empty else np.nan
                stock_results[f"{period}_return"] = cumulative_return

            results.append(stock_results)

        return pd.DataFrame(results)

    def calculate_tracking_error(self, fiscal_start_date: str, fiscal_end_date: str = None) -> dict:
        """
        Calculate tracking error with end date support.

        Args:
            fiscal_start_date: Starting date of the fiscal year
            fiscal_end_date: Ending date of the fiscal year (optional)

        Returns:
            Dictionary of tracking errors by period
        """
        returns = self.daily_returns.pivot(index='return_date', columns='source', values='return_value').dropna()
        today = pd.Timestamp.now(tz='UTC').normalize()

        # Use provided end date if specified, otherwise use today
        if fiscal_end_date:
            end_date = pd.Timestamp(fiscal_end_date).tz_localize('UTC')
            end_date = min(end_date, today)  # Don't exceed today
        else:
            end_date = today

        periods = {
            'weekly': end_date - pd.Timedelta(days=7),
            'monthly': end_date - pd.Timedelta(days=30),
            'fytd': pd.Timestamp(fiscal_start_date).tz_localize('UTC')
        }

        tracking_errors = {}
        for period_name, start_date in periods.items():
            subset = returns[(returns.index >= start_date) & (returns.index <= end_date)]
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
            logging.error(f"Error getting risk-free rate: {str(e)}")
            return 0.0, 0.0

    def calculate_risk_metrics(self, fiscal_start_date: str, fiscal_end_date: str = None) -> dict:
        """
        Calculate risk metrics with end date support.

        Args:
            fiscal_start_date: Starting date of the fiscal year
            fiscal_end_date: Ending date of the fiscal year (optional)

        Returns:
            Dictionary of risk metrics
        """
        try:
            annual_rf_rate, daily_rf_rate = self._get_latest_risk_free_rate()
            returns = self.daily_returns.pivot(index='return_date', columns='source', values='return_value').fillna(0)

            # Apply date filters
            start_date = pd.Timestamp(fiscal_start_date).tz_localize('UTC')

            if fiscal_end_date:
                end_date = pd.Timestamp(fiscal_end_date).tz_localize('UTC')
                returns = returns[(returns.index >= start_date) & (returns.index <= end_date)]
            else:
                returns = returns[returns.index >= start_date]

            # Add minimum data requirement
            if len(returns) < 30:  # Require at least 30 days of data
                logging.warning(f"Insufficient data for risk metrics: only {len(returns)} days available")
                return {
                    'capm_alpha': None,
                    'beta': None,
                    'holdings_beta': self.calculate_holdings_beta(),  # This should work regardless
                    'tracking_error': {'fytd': None},
                    'r_squared': None,
                    'days_used': len(returns)
                }

            portfolio_returns = returns['PORTFOLIO']
            benchmark_returns = returns['BENCHMARK']

            excess_portfolio = portfolio_returns - daily_rf_rate
            excess_benchmark = benchmark_returns - daily_rf_rate

            reg = LinearRegression()
            reg.fit(excess_benchmark.values.reshape(-1, 1), excess_portfolio.values)
            beta = reg.coef_[0]
            capm_alpha = reg.intercept_ * 252

            tracking_error = (
                (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
                if len(portfolio_returns) > 1 else 0
            )

            return {
                'capm_alpha': capm_alpha,
                'beta': beta,
                'holdings_beta': self.calculate_holdings_beta(),
                'tracking_error': {'fytd': tracking_error},
                'r_squared': reg.score(excess_benchmark.values.reshape(-1, 1), excess_portfolio.values),
                'days_used': len(returns)
            }
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'capm_alpha': None,
                'beta': None,
                'holdings_beta': self.calculate_holdings_beta(),
                'tracking_error': {'fytd': None},
                'r_squared': None,
                'days_used': 0
            }

    def calculate_stock_beta(self, stock_symbol: str, lookback_days: int = 252) -> float:
        """
        Calculate beta for an individual stock using historical returns.

        Args:
            stock_symbol: Symbol of the stock
            lookback_days: Number of trading days to look back for calculation

        Returns:
            Beta value for the stock
        """
        try:
            # Get stock daily returns
            stock_returns = self.stock_daily_returns[
                self.stock_daily_returns['stock_symbol'] == stock_symbol
                ]

            # Get benchmark returns for the same period
            benchmark_returns = self.daily_returns[
                self.daily_returns['source'] == 'BENCHMARK'
                ]

            # Merge the data
            merged_data = stock_returns.merge(
                benchmark_returns,
                left_on='return_date',
                right_on='return_date',
                how='inner',
                suffixes=('_stock', '_benchmark')
            )

            # Sort by date and limit to lookback period
            merged_data = merged_data.sort_values('return_date', ascending=False).head(lookback_days)

            if len(merged_data) < 30:  # Require at least 30 data points for regression
                return 1.0  # Default to market beta if insufficient data

            # Calculate beta using regression
            X = merged_data['return_value_benchmark'].values.reshape(-1, 1)
            y = merged_data['return_value_stock'].values

            reg = LinearRegression().fit(X, y)
            beta = reg.coef_[0]

            # Clip to reasonable range
            beta = max(0, min(3, beta))

            return beta
        except Exception as e:
            logging.warning(f"Error calculating beta for {stock_symbol}: {str(e)}")
            return 1.0  # Default to market beta on error

    def calculate_holdings_beta(self) -> float:
        """
        Calculate portfolio beta by summing weighted betas of individual holdings.

        Returns:
            Float representing the portfolio beta
        """
        try:
            # Get portfolio allocation
            allocation = self.calculate_portfolio_allocation()

            portfolio_beta = 0

            for _, row in allocation.iterrows():
                if row['stock_symbol'] == 'CASH':
                    # Cash has a beta of 0
                    continue

                # Calculate beta for this stock
                stock_beta = self.calculate_stock_beta(row['stock_symbol'])

                # Add weighted beta to portfolio beta
                portfolio_beta += row['weight'] * stock_beta

            return portfolio_beta
        except Exception as e:
            logging.error(f"Error calculating holdings beta: {str(e)}")
            return 1.0  # Default to market beta on error

    def calculate_sharpe_ratio(self, fiscal_start_date: str, fiscal_end_date: str = None) -> dict:
        """
        Calculate Sharpe ratio with end date support.

        Args:
            fiscal_start_date: Starting date of the fiscal year
            fiscal_end_date: Ending date of the fiscal year (optional)

        Returns:
            Dictionary with Sharpe ratio
        """
        try:
            annual_rf_rate, daily_rf_rate = self._get_latest_risk_free_rate()
            fiscal_start = pd.Timestamp(fiscal_start_date).tz_localize('UTC')

            # Handle end date if provided
            if fiscal_end_date:
                fiscal_end = pd.Timestamp(fiscal_end_date).tz_localize('UTC')
            else:
                fiscal_end = pd.Timestamp.now(tz='UTC')

            portfolio_data = self.daily_returns[
                (self.daily_returns['source'] == 'PORTFOLIO') &
                (self.daily_returns['return_date'] >= fiscal_start) &
                (self.daily_returns['return_date'] <= fiscal_end)
                ]

            if portfolio_data.empty or len(portfolio_data) < 30:
                return {'sharpe': {'fytd': None}, 'days_used': len(portfolio_data)}

            portfolio_returns = portfolio_data['return_value']
            excess_returns = portfolio_returns - daily_rf_rate

            sharpe = (
                excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                if excess_returns.std() > 0 else None
            )

            return {
                'sharpe': {'fytd': sharpe},
                'days_used': len(portfolio_data)
            }

        except Exception as e:
            logging.error(f"Error calculating Sharpe ratio: {str(e)}")
            return {'sharpe': {'fytd': None}, 'days_used': 0}

    def calculate_adaptive_risk_metrics(self, fiscal_start_date: str, fiscal_end_date: str,
                                        min_data_points: int = 120) -> dict:
        """
        Calculate risk metrics with adaptive time window that extends to previous fiscal year if needed.

        Args:
            fiscal_start_date: Starting date of the current fiscal year
            fiscal_end_date: Ending date of the current fiscal year
            min_data_points: Minimum data points required for reliable metrics (default: 120 trading days)

        Returns:
            Dictionary of risk metrics with metadata about the calculation period
        """
        try:
            # Convert dates to timestamps
            current_fy_start = pd.Timestamp(fiscal_start_date).tz_localize('UTC')
            current_fy_end = pd.Timestamp(fiscal_end_date).tz_localize('UTC')

            # Get data for current fiscal year
            current_fy_returns = self.daily_returns[
                (self.daily_returns['return_date'] >= current_fy_start) &
                (self.daily_returns['return_date'] <= current_fy_end)
                ]

            # Count unique trading days in current FY
            current_fy_days = len(current_fy_returns.drop_duplicates(subset=['return_date']))

            # If we have enough data points, just use current FY
            if current_fy_days >= min_data_points:
                metrics = self.calculate_risk_metrics(fiscal_start_date, fiscal_end_date)
                metrics['calculation_period'] = 'current_fy'
                metrics['period_name'] = 'Fiscal Year to Date'
                return metrics

            # Otherwise, calculate start date for extended period - use full previous fiscal year
            # Start from exactly one year before the fiscal start date
            extended_start = current_fy_start - pd.DateOffset(years=1)

            # Get data for extended period
            extended_returns = self.daily_returns[
                (self.daily_returns['return_date'] >= extended_start) &
                (self.daily_returns['return_date'] <= current_fy_end)
                ]

            # Count unique trading days in extended period
            extended_days = len(extended_returns.drop_duplicates(subset=['return_date']))

            # Check if even the extended period has enough data
            if extended_days < min_data_points:
                logging.warning(
                    f"Even extended period has insufficient data: {extended_days} days available, {min_data_points} required")

                # Try with an even longer period - 2 years if needed
                if extended_days < 60:  # If we have less than ~3 months of data
                    longer_start = extended_start - pd.DateOffset(years=1)
                    longer_start_str = longer_start.strftime('%Y-%m-%d')

                    # Calculate with whatever data we have, but mark it as potentially unreliable
                    metrics = self.calculate_risk_metrics(longer_start_str, fiscal_end_date)
                    metrics['calculation_period'] = 'extended_long'
                    metrics['extended_start_date'] = longer_start
                    metrics['days_used'] = extended_days
                    metrics['period_name'] = 'Extended Historical'
                    metrics['is_reliable'] = False
                    return metrics

            # Calculate metrics using extended period
            extended_start_str = extended_start.strftime('%Y-%m-%d')
            metrics = self.calculate_risk_metrics(extended_start_str, fiscal_end_date)

            # Add metadata about the calculation period
            metrics['calculation_period'] = 'extended'
            metrics['extended_start_date'] = extended_start
            metrics['days_used'] = extended_days
            metrics['period_name'] = 'Trailing 12 Months'
            metrics['is_reliable'] = extended_days >= min_data_points

            return metrics

        except Exception as e:
            logging.error(f"Error calculating adaptive risk metrics: {str(e)}")
            return {
                'capm_alpha': None,
                'beta': None,
                'holdings_beta': self.calculate_holdings_beta(),
                'tracking_error': {'fytd': None},
                'r_squared': None,
                'calculation_period': 'error',
                'days_used': 0,
                'period_name': 'Error',
                'is_reliable': False
            }

    def calculate_adaptive_sharpe_ratio(self, fiscal_start_date: str, fiscal_end_date: str,
                                        min_data_points: int = 120) -> dict:
        """
        Calculate Sharpe ratio with adaptive time window.

        Args:
            fiscal_start_date: Starting date of the current fiscal year
            fiscal_end_date: Ending date of the current fiscal year
            min_data_points: Minimum data points required for reliable metrics

        Returns:
            Dictionary with Sharpe ratio and metadata
        """
        try:
            # Convert dates to timestamps
            current_fy_start = pd.Timestamp(fiscal_start_date).tz_localize('UTC')
            current_fy_end = pd.Timestamp(fiscal_end_date).tz_localize('UTC')

            # Get data for current fiscal year
            current_fy_data = self.daily_returns[
                (self.daily_returns['source'] == 'PORTFOLIO') &
                (self.daily_returns['return_date'] >= current_fy_start) &
                (self.daily_returns['return_date'] <= current_fy_end)
                ]

            # Count unique trading days in current FY
            current_fy_days = len(current_fy_data.drop_duplicates(subset=['return_date']))

            # If we have enough data points, just use current FY
            if current_fy_days >= min_data_points:
                sharpe = self.calculate_sharpe_ratio(fiscal_start_date, fiscal_end_date)
                sharpe['calculation_period'] = 'current_fy'
                sharpe['period_name'] = 'Fiscal Year to Date'
                return sharpe

            # Otherwise, calculate start date for extended period
            # Previous fiscal year starts 1 year before current FY
            extended_start = current_fy_start - pd.DateOffset(years=1)
            extended_start_str = extended_start.strftime('%Y-%m-%d')

            # Calculate using extended period
            sharpe = self.calculate_sharpe_ratio(extended_start_str, fiscal_end_date)

            # Get extended data for metadata
            extended_data = self.daily_returns[
                (self.daily_returns['source'] == 'PORTFOLIO') &
                (self.daily_returns['return_date'] >= extended_start) &
                (self.daily_returns['return_date'] <= current_fy_end)
                ]

            # Count unique trading days in extended period
            extended_days = len(extended_data.drop_duplicates(subset=['return_date']))

            # Add metadata
            sharpe['calculation_period'] = 'extended'
            sharpe['extended_start_date'] = extended_start
            sharpe['period_name'] = 'Trailing 12 Months'

            return sharpe

        except Exception as e:
            logging.error(f"Error calculating adaptive Sharpe ratio: {str(e)}")
            return {
                'sharpe': {'fytd': None},
                'calculation_period': 'error',
                'days_used': 0,
                'period_name': 'Error'
            }

    def calculate_rolling_volatility(self, window: int = 30, fiscal_start_date: str = None,
                                     fiscal_end_date: str = None) -> pd.DataFrame:
        """
        Calculate rolling volatility for portfolio and benchmark.

        Args:
            window: Rolling window size in trading days
            fiscal_start_date: Start date for data filtering (optional)
            fiscal_end_date: End date for data filtering (optional)

        Returns:
            DataFrame with rolling volatility data
        """
        try:
            # Get returns data
            returns = self.daily_returns.pivot(index='return_date', columns='source', values='return_value').fillna(0)

            # Apply date filters if provided
            if fiscal_start_date:
                start_date = pd.Timestamp(fiscal_start_date).tz_localize('UTC')
                returns = returns[returns.index >= start_date]

            if fiscal_end_date:
                end_date = pd.Timestamp(fiscal_end_date).tz_localize('UTC')
                returns = returns[returns.index <= end_date]

            # Ensure we have enough data points
            if len(returns) < window:
                logging.warning(
                    f"Insufficient data for rolling volatility: {len(returns)} days available, {window} required")
                return pd.DataFrame()

            # Calculate rolling volatility (annualized)
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)

            # Add date index as column for easier plotting
            rolling_vol = rolling_vol.reset_index()

            return rolling_vol

        except Exception as e:
            logging.error(f"Error calculating rolling volatility: {str(e)}")
            return pd.DataFrame()

    def calculate_return_distribution(self, fiscal_start_date: str = None, fiscal_end_date: str = None) -> dict:
        """
        Calculate return distribution statistics for portfolio and benchmark.

        Args:
            fiscal_start_date: Start date for data filtering (optional)
            fiscal_end_date: End date for data filtering (optional)

        Returns:
            Dictionary with return distribution data
        """
        try:
            # Get returns data
            returns = self.daily_returns.pivot(index='return_date', columns='source', values='return_value')

            # Apply date filters if provided
            if fiscal_start_date:
                start_date = pd.Timestamp(fiscal_start_date).tz_localize('UTC')
                returns = returns[returns.index >= start_date]

            if fiscal_end_date:
                end_date = pd.Timestamp(fiscal_end_date).tz_localize('UTC')
                returns = returns[returns.index <= end_date]

            # Ensure we have enough data points
            if len(returns) < 30:
                logging.warning(
                    f"Insufficient data for return distribution: {len(returns)} days available, 30 required")
                return {}

            result = {}
            for source in ['PORTFOLIO', 'BENCHMARK']:
                if source not in returns.columns:
                    continue

                source_returns = returns[source].dropna()

                result[source] = {
                    'mean': source_returns.mean() * 252,  # Annualized mean return
                    'median': source_returns.median() * 252,  # Annualized median return
                    'std': source_returns.std() * np.sqrt(252),  # Annualized volatility
                    'skew': source_returns.skew(),  # Skewness
                    'kurtosis': source_returns.kurtosis(),  # Excess kurtosis
                    'min': source_returns.min(),  # Worst daily return
                    'max': source_returns.max(),  # Best daily return
                    'histogram': np.histogram(source_returns, bins=20),  # Histogram data
                    'return_data': source_returns  # Raw return data for plotting
                }

            return result

        except Exception as e:
            logging.error(f"Error calculating return distribution: {str(e)}")
            return {}

    def calculate_cumulative_returns(self, fiscal_start_date: str, fiscal_end_date: str = None) -> dict:
        """
        Calculate cumulative returns for different time periods with end date support.

        Args:
            fiscal_start_date: Starting date of the fiscal year
            fiscal_end_date: Ending date of the fiscal year (optional)

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

        # Use the provided end date or get the maximum date from the data
        if fiscal_end_date:
            end_date = pd.Timestamp(fiscal_end_date).tz_localize('UTC')
            # Make sure we don't exceed available data
            end_date = min(end_date, daily_returns['return_date'].max())
        else:
            end_date = daily_returns['return_date'].max()

        # Filter data to respect the end date
        daily_returns = daily_returns[daily_returns['return_date'] <= end_date]

        # Get the periods based on the end date, not the current date
        periods = self._get_period_dates(end_date)
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
            start_date, period_end_date = periods[period_name]
            period_data = daily_returns[
                (daily_returns['return_date'] >= start_date) &
                (daily_returns['return_date'] <= period_end_date)
                ]

            # Log warning if data appears incomplete
            expected_days = len(self.nyse.valid_days(start_date=start_date, end_date=period_end_date))
            if len(period_data) < expected_days * 0.95:
                logging.warning(f"Incomplete data for {period_name}: {len(period_data)} vs expected {expected_days}")

            results[period_name] = calculate_period_returns(period_data)

        # Calculate FYTD returns
        fytd_start = pd.to_datetime(fiscal_start_date).tz_localize('UTC')
        fytd_data = daily_returns[
            (daily_returns['return_date'] >= fytd_start) &
            (daily_returns['return_date'] <= end_date)
            ]
        results['fytd'] = calculate_period_returns(fytd_data)

        return results