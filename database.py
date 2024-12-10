from sqlalchemy import create_engine
import pandas as pd

class DatabaseInterface:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)

    def fetch_data(self, query, params=None):
        """
        Execute a query and return the results as a pandas DataFrame.
        """
        with self.engine.connect() as conn:
            return pd.read_sql(query, conn, params=params)

    def get_daily_returns(self, start_date, end_date):
        query = """
        SELECT return_date, return_value, source, fund_id
        FROM daily_returns
        WHERE return_date BETWEEN %s AND %s
        """
        return self.fetch_data(query, (start_date, end_date))

    def get_stock_daily_returns(self, start_date, end_date):
        query = """
        SELECT return_date, return_value, stock_symbol
        FROM stock_daily_returns
        WHERE return_date BETWEEN %s AND %s
        """
        return self.fetch_data(query, (start_date, end_date))

    def get_holdings(self, fund_id):
        query = """
        SELECT stock_symbol, shares_held, average_cost, open_date, fund_id, permno
        FROM holdings
        WHERE fund_id = %s
        """
        return self.fetch_data(query, (fund_id,))

    def get_risk_free_rates(self, start_date, end_date):
        query = """
        SELECT date, rate
        FROM risk_free_rates
        WHERE date BETWEEN %s AND %s
        """
        return self.fetch_data(query, (start_date, end_date))

    def get_stock_prices(self):
        """
        Get the latest stock prices for all stocks.
        Returns a DataFrame with stock symbols and their current prices.
        """
        query = """
        SELECT stock_symbol, 
               price as current_price,
               return_date as price_date
        FROM stock_prices sp1
        WHERE return_date = (
            SELECT MAX(return_date)
            FROM stock_prices sp2
            WHERE sp2.stock_symbol = sp1.stock_symbol
        )
        """
        return self.fetch_data(query)

    def get_balances(self, fund_id, start_date, end_date):
        """
        Get balance data for a specific fund within a date range.
        
        Parameters:
            fund_id (str): The fund identifier
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: Balance data including portfolio value, cash balance,
                            cash flows, and verification status
        """
        query = """
        SELECT 
            fund_id,
            balance_date,
            total_portfolio_value,
            cash_balance,
            net_external_cash_flow,
            verified,
            interim_calculated,
            pending_settlements
        FROM balances
        WHERE fund_id = %s 
        AND balance_date BETWEEN %s AND %s
        ORDER BY balance_date
        """
        return self.fetch_data(query, (fund_id, start_date, end_date))

if __name__ == "__main__":
    # Replace with your actual database URL
    DATABASE_URL = 'postgresql+psycopg2://MIGRisk:MIGRISK1234@dbinstance.cnucw2ecmp84.us-east-2.rds.amazonaws.com:5432/MIG'
    db = DatabaseInterface(DATABASE_URL)

    # Test all queries with a known date range
    test_start = "2024-03-29"
    test_end = "2024-12-31"
    test_fund = "DADCO"

   # print("Testing Daily Returns Query...")
  #  daily_returns = db.get_daily_returns(test_start, test_end)
#    print(daily_returns.head())

  #  print("\nTesting Stock Daily Returns Query...")
  #  stock_returns = db.get_stock_daily_returns(test_start, test_end)
  #  print(stock_returns.head())

#    print("\nTesting Holdings Query...")
#    holdings = db.get_holdings(test_fund)
 #   print(holdings.head())

  #  print("\nTesting Risk-Free Rates Query...")
  #  risk_free_rates = db.get_risk_free_rates(test_start, test_end)
  #  print(risk_free_rates.head())

#    print("\nTesting Balances Query...")
 #   balances = db.get_balances(test_fund, test_start, test_end)
   # print(balances.head())