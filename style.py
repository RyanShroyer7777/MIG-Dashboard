def apply_custom_style():
    return """
    <style>
        /* General Reset */
        body, .stApp {
            background-color: white;
        }

        /* Header Area */
        .custom-header-container {
            background: linear-gradient(90deg, #004F2F 0%, #195C3C 100%);
            padding: 1rem 2rem;
            margin: -1rem -1rem 2rem -1rem;
            border-bottom: 4px solid #FEE123;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .custom-header-container h1 {
            color: white !important;
            margin: 0;
            font-size: 28px;
            font-weight: 600;
        }
        .custom-header-container p {
            color: rgba(255, 255, 255, 0.9) !important;
            margin: 4px 0 0 0;
            font-size: 16px;
        }

        /* Simplified Metric Cards */
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            flex: 1;
            min-width: 200px;
            background: #004F2F;
            color: white;
            padding: 1rem;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-label {
            font-size: 14px;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #FEE123;
        }
        .metric-value {
            font-size: 22px;
            font-weight: bold;
        }

        /* Tables */
        .dataframe {
            width: 100%;
            margin: 10px 0 30px 0; /* Extra space below tables */
            border-collapse: collapse;
        }
        .dataframe th {
            background-color: #1E3932 !important; /* Darker green for header */
            color: white !important;
            padding: 10px;
            text-align: left;
            font-weight: bold;
        }
        .dataframe td {
            background-color: white;
            padding: 10px;
            border-bottom: 1px solid #e0e0e0;
        }
        .dataframe tr:nth-child(even) td {
            background-color: #F8F8F8; /* Light gray for alternating rows */
        }

        /* Tab Design */
        .stTabs [data-baseweb="tab"] {
            background: #E8F5E9;
            border: 1px solid #004F2F;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            margin: 0 0.5rem;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: #A5D6A7;
            cursor: pointer;
        }
        .stTabs [aria-selected="true"] {
            background: #004F2F !important;
            color: white !important;
        }

        /* Hide Streamlit's default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """


def format_performance_table(df):
    """Format performance table with strong dark header styling."""
    return df.style.format({
        'Portfolio': '{:+.2%}',
        'Benchmark': '{:+.2%}',
        'Excess Return': '{:+.2%}'
    }).set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('background-color', '#004F2F !important'),  # Force dark green
                ('color', 'white !important'),              # Force white text
                ('font-size', '16px'),                      # Make font larger
                ('font-weight', 'bold'),                    # Bold text
                ('text-align', 'center'),                   # Center align
                ('padding', '10px'),                        # Add padding
                ('border', '1px solid #004F2F'),            # Border for headers
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('padding', '10px'),
                ('border-bottom', '1px solid #e0e0e0'),
            ]
        },
        {
            'selector': 'tbody tr:nth-child(even) td',
            'props': [
                ('background-color', '#F8F8F8'),
            ]
        }
    ])

def format_holdings_table(df):
    """Format holdings table with strong dark header styling."""
    return df.style.format({
        'Shares': '{:,.0f}',
        'Price': '${:,.2f}',
        'Market Value': '${:,.2f}',
        'Weight': '{:.2%}',
        'Weekly Return': '{:+.2%}',
        'Monthly Return': '{:+.2%}',
        'FYTD Return': '{:+.2%}'
    }).set_table_styles([
        {
            'selector': 'th',
            'props': [
                ('background-color', '#004F2F !important'),  # Force dark green
                ('color', 'white !important'),              # Force white text
                ('font-size', '16px'),                      # Make font larger
                ('font-weight', 'bold'),                    # Bold text
                ('text-align', 'center'),                   # Center align
                ('padding', '10px'),                        # Add padding
                ('border', '1px solid #004F2F'),            # Border for headers
            ]
        },
        {
            'selector': 'td',
            'props': [
                ('padding', '10px'),
                ('border-bottom', '1px solid #e0e0e0'),
            ]
        },
        {
            'selector': 'tbody tr:nth-child(even) td',
            'props': [
                ('background-color', '#F8F8F8'),
            ]
        }
    ]).applymap(
        lambda x: 'color: #0C9B6A' if isinstance(x, float) and x > 0 else 'color: #DC2626',
        subset=['Weekly Return', 'Monthly Return', 'FYTD Return']
    )

