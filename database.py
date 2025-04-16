from sqlalchemy import create_engine
import pandas as pd
import logging


class DatabaseInterface:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)

    def fetch_data(self, query, params=None):
        """
        Execute a query and return the results as a pandas DataFrame.
        """
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params=params)

    def get_portfolio_returns(self, start_date, end_date, fund="DADCO"):
        """
        Get portfolio returns from portfolio_returns table.
        Uses the total_return column for portfolio data.
        """
        query = """
        SELECT date as return_date, total_return as return_value, fund
        FROM portfolio_returns
        WHERE date BETWEEN %s AND %s
        AND fund = %s
        """
        return self.fetch_data(query, (start_date, end_date, fund))

    def get_benchmark_returns(self, start_date, end_date):
        """
        Get benchmark returns using IVV from stock_daily_returns table.
        Uses the original return column for stock data.
        """
        query = """
        SELECT date as return_date, return as return_value 
        FROM stock_daily_returns
        WHERE date BETWEEN %s AND %s
        AND stock_symbol = 'IVV'
        AND source = 'DADCO'
        """
        return self.fetch_data(query, (start_date, end_date))

    def combine_portfolio_benchmark_returns(self, portfolio_returns, benchmark_returns):
        """
        Combine portfolio and benchmark returns into a format matching
        the previous schema's daily_returns format.
        """
        portfolio_returns['return_date'] = pd.to_datetime(portfolio_returns['return_date'])
        benchmark_returns['return_date'] = pd.to_datetime(benchmark_returns['return_date'])
        portfolio_returns['source'] = 'PORTFOLIO'
        benchmark_returns['source'] = 'BENCHMARK'
        if 'fund' in portfolio_returns.columns and 'fund' not in benchmark_returns.columns:
            benchmark_returns['fund'] = portfolio_returns['fund'].iloc[0] if not portfolio_returns.empty else "DADCO"
        combined_returns = pd.concat([portfolio_returns, benchmark_returns])
        return combined_returns

    def get_stock_daily_returns(self, start_date, end_date):
        """
        Get stock daily returns from stock_daily_returns table.
        Uses the original return column for stock data.
        """
        query = """
        SELECT date as return_date, return as return_value, stock_symbol
        FROM stock_daily_returns
        WHERE date BETWEEN %s AND %s
        """
        return self.fetch_data(query, (start_date, end_date))

    def get_holdings(self, fund="DADCO"):
        """
        Get holdings for a specific fund.
        """
        query = """
        SELECT 
            stock_symbol, 
            shares as shares_held, 
            cost_basis as average_cost,
            date as open_date,
            fund,
            market_value
        FROM holdings
        WHERE fund = %s
        """
        return self.fetch_data(query, (fund,))

    def get_risk_free_rates(self, start_date, end_date):
        """
        Get risk-free rates, using the 3-month rate.
        """
        query = """
        SELECT date, rate_3m as rate
        FROM risk_free_rates
        WHERE date BETWEEN %s AND %s
        """
        return self.fetch_data(query, (start_date, end_date))

    def get_stock_prices(self):
        """
        Get the latest stock prices for all stocks.
        """
        query = """
        SELECT 
            stock_symbol, 
            adjusted_close as current_price,
            date as price_date
        FROM stock_prices sp1
        WHERE date = (
            SELECT MAX(date)
            FROM stock_prices sp2
            WHERE sp2.stock_symbol = sp1.stock_symbol
        )
        """
        return self.fetch_data(query)

    def get_balances(self, fund="DADCO", start_date=None, end_date=None):
        """
        Get balance data for a specific fund within a date range.
        """
        query = """
        SELECT 
            fund,
            date as balance_date,
            total_portfolio_value,
            cash_balance,
            net_external_cash_flow
        FROM balances
        WHERE fund = %s 
        AND date BETWEEN %s AND %s
        ORDER BY date
        """
        return self.fetch_data(query, (fund, start_date, end_date))

    def get_benchmark_weights(self, start_date, end_date):
        """
        Get benchmark weights for a given date range.
        """
        query = """
        SELECT date, stock_symbol, weight
        FROM benchmark_weights
        WHERE date BETWEEN %s AND %s
        ORDER BY date, stock_symbol
        """
        return self.fetch_data(query, (start_date, end_date))
