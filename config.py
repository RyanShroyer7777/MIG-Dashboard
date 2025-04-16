 
# config.py

# config.py
from datetime import date

DATABASE_URL = 'postgresql+psycopg2://MIGRisk:MIGRISK1234@dbinstance.cnucw2ecmp84.us-east-2.rds.amazonaws.com:5432/MIG'

DEFAULT_FUND_ID = 'DADCO'
DEFAULT_START_DATE = date(2024, 1, 1)
DEFAULT_END_DATE = date.today()  # Use today's date instead of hard-coded future date