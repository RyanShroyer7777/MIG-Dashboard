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
        fund_id: str,
        cash_balance: float
    ):
        self.fund_id = fund_id
        self.cash_balance = max(cash_balance, 0)

        # Convert date columns to datetime
        daily_returns['return_date'] = pd.to_datetime(daily_returns['return_date'])
        stock_daily_returns['return_date'] = pd.to_datetime(stock_daily_returns['return_date'])
        risk_free_rates['date'] = pd.to_datetime(risk_free_rates['date'])
        stock_prices['price_date'] = pd.to_datetime(stock_prices['price_date'])

        # Assign data to the class
        self.daily_returns = daily_returns
        self.stock_daily_returns = stock_daily_returns
        self.holdings = holdings
        self.risk_free_rates = risk_free_rates
        self.stock_prices = stock_prices

        # Validate the data
        self._validate_data()

    def _validate_data(self) -> None:
        required_columns = {
            'daily_returns': ['return_date', 'return_value', 'source', 'fund_id'],
            'stock_daily_returns': ['return_date', 'return_value', 'stock_symbol'],
            'holdings': ['stock_symbol', 'shares_held', 'average_cost', 'fund_id'],
            'risk_free_rates': ['date', 'rate'],
            'stock_prices': ['stock_symbol', 'current_price', 'price_date']
        }
        for df_name, columns in required_columns.items():
            df = getattr(self, df_name)
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"{df_name} is missing required columns: {missing_cols}")

    def calculate_portfolio_allocation(self) -> pd.DataFrame:
        # Merge stock prices to get current prices
        allocation = self.holdings.merge(
            self.stock_prices[['stock_symbol', 'current_price']],
            on='stock_symbol',
            how='left'
        )
        # Calculate market value of stocks
        allocation['market_value'] = allocation['shares_held'] * allocation['current_price']
        
        # Add cash allocation
        cash_allocation = pd.DataFrame({'stock_symbol': ['CASH'], 'market_value': [self.cash_balance]})
        allocation = pd.concat([allocation, cash_allocation], ignore_index=True)
    
        # Total portfolio value
        total_value = allocation['market_value'].sum()
        allocation['weight'] = allocation['market_value'] / total_value
    
        return allocation[['stock_symbol', 'market_value', 'weight']]



    def calculate_equity_returns(self, fiscal_start_date: str) -> pd.DataFrame:
        """
        Calculate cumulative returns for each stock (weekly, monthly, FYTD).
        """
        latest_date = self.stock_daily_returns['return_date'].max()

        periods = {
            "weekly": latest_date - pd.Timedelta(days=7),
            "monthly": latest_date - pd.Timedelta(days=30),
            "fytd": pd.Timestamp(fiscal_start_date)
        }

        results = []
        for stock in self.holdings['stock_symbol']:
            stock_data = self.stock_daily_returns[self.stock_daily_returns['stock_symbol'] == stock]
            stock_results = {"stock_symbol": stock}

            for period, start_date in periods.items():
                period_data = stock_data[(stock_data['return_date'] >= start_date) &
                                         (stock_data['return_date'] <= latest_date)]

                cumulative_return = (1 + period_data['return_value']).prod() - 1 if not period_data.empty else 0
                stock_results[f"{period}_return"] = cumulative_return

            results.append(stock_results)

        return pd.DataFrame(results)






    def calculate_tracking_error(self, fiscal_start_date: str) -> dict:
        """
        Calculate tracking error for different periods.
        """
        returns = self.daily_returns.pivot(
            index='return_date',
            columns='source',
            values='return_value'
        ).dropna()

        today = pd.Timestamp.now().normalize()
        periods = {
            'weekly': today - pd.Timedelta(days=7),
            'monthly': today - pd.Timedelta(days=30),
            'fytd': pd.Timestamp(fiscal_start_date)
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
                logging.warning("No risk-free rates available, using 0")
                return 0.0, 0.0
                
            latest_rate = (
                self.risk_free_rates
                .sort_values('date', ascending=False)
                .iloc[0]['rate']
            )
            
            annual_rate = latest_rate  # Remove the /100
            daily_rate = (1 + annual_rate) ** (1/252) - 1
            
            return annual_rate, daily_rate
            
        except Exception as e:
            logging.error(f"Error getting risk-free rate: {str(e)}")
            return 0.0, 0.0
    
    def calculate_risk_metrics(self, fiscal_start_date: str) -> dict:
        """Calculate FYTD risk metrics."""
        try:
            annual_rf_rate, daily_rf_rate = self._get_latest_risk_free_rate()
            
            returns = self.daily_returns.pivot(
                index='return_date',
                columns='source',
                values='return_value'
            ).fillna(0)
            
            portfolio_returns = returns['PORTFOLIO']
            benchmark_returns = returns['BENCHMARK']
            
            excess_portfolio = portfolio_returns - daily_rf_rate
            excess_benchmark = benchmark_returns - daily_rf_rate
            
            # Current market-adjusted alpha calculation
            reg = LinearRegression()
            reg.fit(excess_benchmark.values.reshape(-1, 1), excess_portfolio.values)
            beta = reg.coef_[0]
            market_alpha = reg.intercept_ * 252
            
            # Raw alpha calculation (just portfolio excess return)
            # Annualize both returns first
            ann_portfolio_return = (1 + portfolio_returns.mean()) ** 252 - 1
            raw_alpha = ann_portfolio_return - annual_rf_rate
            
            # Convert raw_alpha to match scaling of other metrics
            raw_alpha = raw_alpha / 10  # Adjust the scaling to match other metrics
            
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
            logging.error(f"Error in calculate_risk_metrics: {str(e)}")
            return {
                'alpha': 0,
                'raw_alpha': 0,
                'beta': 0,
                'tracking_error': {'fytd': 0},
                'r_squared': 0
            }
    def calculate_sharpe_ratio(self, fiscal_start_date: date) -> dict:
        """
        Calculate Sharpe ratio (risk-adjusted returns metric)
        """
        try:
            # Get risk-free rate and portfolio data
            annual_rf_rate, daily_rf_rate = self._get_latest_risk_free_rate()
            fiscal_start = pd.Timestamp(fiscal_start_date)
            
            portfolio_data = self.daily_returns[
                (self.daily_returns['source'] == 'PORTFOLIO') & 
                (self.daily_returns['return_date'] >= fiscal_start)
            ]
            
            if portfolio_data.empty:
                return {
                    'sharpe': {'fytd': None},
                    'treynor': {'fytd': None}
                }
                
            # Calculate returns
            portfolio_returns = portfolio_data['return_value']
            excess_returns = portfolio_returns - daily_rf_rate
            
            # Sharpe Ratio
            sharpe = (
                excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                if excess_returns.std() > 0 else None
            )
            
            # Treynor ratio
            risk_metrics = self.calculate_risk_metrics(fiscal_start_date)
            beta = risk_metrics['beta']
            treynor_ratio = (excess_returns.mean() * 252) / beta if beta != 0 else None
    
            return {
                'sharpe': {'fytd': sharpe},
                'treynor': {'fytd': treynor_ratio}
            }
            
        except Exception as e:
            logging.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'sharpe': {'fytd': None},
                'treynor': {'fytd': None}
            }
    def calculate_cumulative_returns(self, fiscal_start_date: str) -> dict:
        """
        Calculate cumulative returns for weekly, monthly, and fiscal YTD periods.
        """
        # Convert fiscal_start_date to datetime
        fiscal_start_date = pd.to_datetime(fiscal_start_date)
        
        # Initialize the result dictionary
        results = {
            'weekly': pd.DataFrame(),
            'monthly': pd.DataFrame(),
            'fytd': pd.DataFrame()
        }
        
        if self.daily_returns.empty:
            logging.error("Daily returns DataFrame is empty.")
            return results
        
        # Ensure return_date is datetime and sorted
        daily_returns = self.daily_returns.copy()
        daily_returns['return_date'] = pd.to_datetime(daily_returns['return_date'])
        daily_returns = daily_returns.sort_values(by='return_date')
    
        # Define periods for filtering
        today = pd.Timestamp.now().normalize()
        periods = {
            'weekly': today - pd.Timedelta(days=7),
            'monthly': today - pd.Timedelta(days=30),
            'fytd': fiscal_start_date
        }
    
        for period_name, start_date in periods.items():
            try:
                # Filter data for the given period
                period_data = daily_returns[daily_returns['return_date'] >= start_date]
    
                if period_data.empty:
                    logging.warning(f"No data available for {period_name}.")
                    continue
    
                # Pivot the data to separate portfolio and benchmark
                period_pivot = period_data.pivot(
                    index='return_date',
                    columns='source',
                    values='return_value'
                ).fillna(0)
    
                # Calculate cumulative returns
                cumul_returns = (1 + period_pivot).cumprod() - 1
                results[period_name] = cumul_returns
    
            except Exception as e:
                logging.error(f"Error processing {period_name} cumulative returns: {e}")
                continue
    
        return results
