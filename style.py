def apply_custom_style():
    return """
    <style>
        /* Main container styling */
        .main {
            padding: 0 !important;
            background-color: #f8faf9 !important;
        }
        
        .block-container {
            padding: 2rem 3rem 3rem 3rem !important;
            max-width: 100% !important;
        }
        
        /* Header styling */
        .custom-header {
            background: linear-gradient(90deg, #154733 70%, #0D2B1F 100%);
            padding: 1.5rem 3rem;
            margin: -2rem -3rem 2rem -3rem;
            color: white;
            border-bottom: 4px solid #FEE123;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            position: relative;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .custom-header h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 600;
            font-family: 'Helvetica Neue', sans-serif;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .custom-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Section headers */
        h2, h3, h4 {
            color: #154733 !important;
            font-weight: 600 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 1rem !important;
        }
        
        /* Tab styling */
        [data-testid="stTabs"] {
            background-color: white !important;
            padding: 1.5rem !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
            margin-top: 2rem !important;
            border: 1px solid rgba(21, 71, 51, 0.1) !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: transparent;
            padding: 0.5rem 1rem;
            border-bottom: 2px solid rgba(21, 71, 51, 0.1);
        }

        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            color: #4B5563 !important;
            background-color: rgba(21, 71, 51, 0.05) !important;
            border-radius: 8px 8px 0 0;
            font-weight: 500;
            padding: 0.5rem 1.5rem;
            margin-right: 1rem;
            transition: all 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, #154733 0%, #1A5840 100%) !important;
            color: white !important;
            border-bottom: 3px solid #FEE123 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Table styling */
        [data-testid="stDataFrame"] {
            background-color: white !important;
            padding: 1.25rem !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
            border: 1px solid rgba(21, 71, 51, 0.1) !important;
        }

        .dataframe {
            border: none !important;
            border-collapse: separate !important;
            border-spacing: 0 !important;
            width: 100% !important;
            margin-bottom: 0 !important;
            background-color: white !important;
        }
        
        .dataframe thead tr th {
            background: linear-gradient(90deg, #154733 0%, #1A5840 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            text-align: left !important;
            padding: 1rem !important;
            border: none !important;
            border-bottom: 3px solid #FEE123 !important;
            font-size: 0.95rem !important;
        }
        
        .dataframe tbody tr {
            border-bottom: 1px solid rgba(21, 71, 51, 0.1) !important;
            transition: background-color 0.2s ease !important;
        }
        
        .dataframe tbody tr td {
            padding: 0.875rem 1rem !important;
            color: #1f2937 !important;
            background-color: white !important;
            border-bottom: 1px solid rgba(21, 71, 51, 0.1) !important;
            font-size: 0.95rem !important;
        }
        
        .dataframe tbody tr:nth-child(odd) td {
            background-color: rgba(21, 71, 51, 0.02) !important;
        }
        
        .dataframe tbody tr:hover td {
            background-color: rgba(21, 71, 51, 0.05) !important;
        }
        
        /* Chart container styling */
        [data-testid="stPlotlyChart"] {
            background-color: white !important;
            padding: 1.25rem !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
            border: 1px solid rgba(21, 71, 51, 0.1) !important;
        }

        /* Metric styling */
        [data-testid="stMetric"] {
            background-color: white !important;
            padding: 1.25rem 1.5rem !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
            border: 1px solid rgba(21, 71, 51, 0.1) !important;
            transition: transform 0.2s ease;
        }
        
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            color: #4B5563 !important;
            font-weight: 500 !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 600 !important;
            color: #154733 !important;
        }

        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Spinner styling */
        .stSpinner > div {
            border-top-color: #154733 !important;
        }

        /* Alert styling */
        .stAlert {
            background-color: #FEF2F2;
            color: #991B1B;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #FCA5A5;
        }
    </style>
    """
