def apply_custom_style():
    return """
    <style>
        /* Main container styling */
        .main {
            padding: 0 !important;
        }
        
        .block-container {
            padding: 2rem 3rem 3rem 3rem !important;
            max-width: 100% !important;
        }
        
        /* Header styling */
        .custom-header {
            background-color: #154733;
            padding: 1.5rem 3rem;
            margin: -2rem -3rem 2rem -3rem;
            color: white;
            border-bottom: 4px solid #FEE123;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .custom-header h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 600;
            font-family: 'Helvetica Neue', sans-serif;
        }

        .custom-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Metric container styling */
        [data-testid="stMetric"] {
            background-color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e5e7eb;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            color: #4B5563;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 600 !important;
            color: #154733;
        }
        
        [data-testid="stMetricDelta"] {
            font-size: 0.9rem !important;
            color: #4B5563;
        }

        /* Tab styling */
        [data-testid="stTabs"] {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-top: 2rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: transparent;
            padding: 0 1rem;
            border-bottom: 2px solid #e5e7eb;
        }

        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            color: #374151 !important;
            background-color: #E5E7EB !important;
            border-radius: 6px 6px 0 0;
            font-weight: 500;
            padding: 0.5rem 1rem;
            margin-right: 1rem;
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            color: white !important;
            background-color: #154733 !important;
            border-bottom: 3px solid #154733 !important;
        }
        
        /* Table styling */
        [data-testid="stDataFrame"] {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .dataframe {
            border: none !important;
            border-collapse: collapse !important;
            width: 100% !important;
            margin-bottom: 0 !important;
        }
        
        .dataframe thead th {
            background-color: #154733 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            text-align: left !important;
            padding: 0.75rem 1rem !important;
            border-top: none !important;
            border-bottom: 2px solid #e5e7eb !important;
        }
        
        .dataframe tbody tr {
            border-bottom: 1px solid #e5e7eb !important;
        }
        
        .dataframe tbody td {
            padding: 0.75rem 1rem !important;
            color: #374151 !important;
            background-color: white !important;
        }
        
        .dataframe tbody tr:hover td {
            background-color: #f9fafb !important;
        }
        
        /* Chart container styling */
        [data-testid="stPlotlyChart"] {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }

        .js-plotly-plot {
            border-radius: 8px !important;
        }

        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Loading spinner styling */
        .stSpinner > div {
            border-top-color: #154733 !important;
        }

        /* Error message styling */
        .stAlert {
            background-color: #FEF2F2;
            color: #991B1B;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #FCA5A5;
        }

        /* Success message styling */
        .stSuccess {
            background-color: #F0FDF4;
            color: #166534;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #86EFAC;
        }
    </style>
    """
