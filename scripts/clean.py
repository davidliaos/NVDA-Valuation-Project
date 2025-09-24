import pandas as pd
import numpy as np
import os
import warnings
import glob

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration
BASE_OUTPUT_FOLDER = "data"
COMPANIES = ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "TSM"]

# Global Mappings & Helpers
FINANCIAL_STATEMENT_COL_MAPPING = {
    'fiscalDateEnding': 'Date',
    'reportedCurrency': 'Currency',
    'totalRevenue': 'Revenue',
    'costOfRevenue': 'Cost of Revenue',
    'costofGoodsAndServicesSold': 'Cost of Goods Sold',
    'grossProfit': 'Gross Profit',
    'operatingExpenses': 'Operating Expenses',
    'sellingGeneralAndAdministrative': 'SG&A',
    'researchAndDevelopment': 'Research and Development',
    'operatingIncome': 'Operating Income',
    'interestIncome': 'Interest Income',
    'interestExpense': 'Interest Expense',
    'netInterestIncome': 'Net Interest Income',
    'incomeTaxExpense': 'Income Tax Expense',
    'incomeBeforeTax': 'Income Before Tax',
    'netIncome': 'Net Income',
    'netIncomeFromContinuingOperations': 'Net Income Continuing',
    'ebit': 'EBIT',
    'ebitda': 'EBITDA',
    'depreciationAndAmortization': 'Depreciation & Amortization',
    'comprehensiveIncomeNetOfTax': 'Comprehensive Income',

    # Balance Sheet
    'cashAndCashEquivalentsAtCarryingValue': 'Cash & Equivalents',
    'cashAndShortTermInvestments': 'Cash & Short Term Investments',
    'shortTermInvestments': 'Short Term Investments',
    'inventory': 'Inventory',
    'currentNetReceivables': 'Accounts Receivable',
    'totalCurrentAssets': 'Total Current Assets',
    'propertyPlantEquipment': 'Property Plant & Equipment',
    'accumulatedDepreciationAmortizationPPE': 'Accumulated Depreciation',
    'intangibleAssets': 'Intangible Assets',
    'intangibleAssetsExcludingGoodwill': 'Intangible Assets Excluding Goodwill',
    'goodwill': 'Goodwill',
    'longTermInvestments': 'Long Term Investments',
    'totalNonCurrentAssets': 'Total Non-Current Assets',
    'totalAssets': 'Total Assets',
    'currentAccountsPayable': 'Accounts Payable',
    'shortTermDebt': 'Short Term Debt',
    'currentDebt': 'Current Debt',
    'totalCurrentLiabilities': 'Total Current Liabilities',
    'longTermDebt': 'Long Term Debt',
    'longTermDebtNoncurrent': 'Long Term Debt Noncurrent',
    'totalNonCurrentLiabilities': 'Total Non-Current Liabilities',
    'totalLiabilities': 'Total Liabilities',
    'commonStock': 'Common Stock',
    'retainedEarnings': 'Retained Earnings',
    'treasuryStock': 'Treasury Stock',
    'totalShareholderEquity': 'Total Shareholder Equity',
    'commonStockSharesOutstanding': 'Shares Outstanding',

    # Cash Flow Statement
    'operatingCashflow': 'Operating Cash Flow',
    'paymentsForOperatingActivities': 'Payments for Operating Activities',
    'proceedsFromOperatingActivities': 'Proceeds from Operating Activities',
    'changeInOperatingAssets': 'Change in Operating Assets',
    'changeInOperatingLiabilities': 'Change in Operating Liabilities',
    'depreciationDepletionAndAmortization': 'Depreciation & Amortization',
    'capitalExpenditures': 'Capital Expenditures',
    'changeInReceivables': 'Change in Receivables',
    'changeInInventory': 'Change in Inventory',
    'profitLoss': 'Profit/Loss',
    'cashflowFromInvestment': 'Investing Cash Flow',
    'cashflowFromFinancing': 'Financing Cash Flow',
    'dividendPayout': 'Dividends Paid',
    'dividendPayoutCommonStock': 'Dividends Paid Common',
    'proceedsFromRepurchaseOfEquity': 'Stock Repurchase',
    'paymentsForRepurchaseOfCommonStock': 'Payments for Stock Repurchase',
    'proceedsFromIssuanceOfCommonStock': 'Stock Issuance',
    'proceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet': 'Debt Issued',
    'changeInCashAndCashEquivalents': 'Net Change In Cash',
    'netIncome': 'Net Income'
}

VALUES_TO_CONVERT = [
    'Revenue', 'Cost of Revenue', 'Cost of Goods Sold', 'Gross Profit', 'Operating Expenses', 
    'Research and Development', 'SG&A', 'Operating Income', 'Interest Income', 'Interest Expense',
    'Net Interest Income', 'Income Tax Expense', 'Income Before Tax', 'Net Income', 'Net Income Continuing',
    'EBIT', 'EBITDA', 'Depreciation & Amortization', 'Comprehensive Income',
    'Cash & Equivalents', 'Cash & Short Term Investments', 'Short Term Investments', 'Inventory',
    'Accounts Receivable', 'Total Current Assets', 'Property Plant & Equipment', 'Accumulated Depreciation',
    'Intangible Assets', 'Intangible Assets Excluding Goodwill', 'Goodwill', 'Long Term Investments',
    'Total Non-Current Assets', 'Total Assets', 'Accounts Payable', 'Short Term Debt', 'Current Debt',
    'Total Current Liabilities', 'Long Term Debt', 'Long Term Debt Noncurrent', 'Total Non-Current Liabilities',
    'Total Liabilities', 'Common Stock', 'Retained Earnings', 'Treasury Stock', 'Total Shareholder Equity',
    'Shares Outstanding', 'Operating Cash Flow', 'Payments for Operating Activities', 
    'Proceeds from Operating Activities', 'Change in Operating Assets', 'Change in Operating Liabilities',
    'Capital Expenditures', 'Change in Receivables', 'Change in Inventory', 'Profit/Loss',
    'Investing Cash Flow', 'Financing Cash Flow', 'Dividends Paid', 'Dividends Paid Common',
    'Stock Repurchase', 'Payments for Stock Repurchase', 'Stock Issuance', 'Debt Issued', 'Net Change In Cash'
]

def find_financial_statement_file(symbol, statement_type, frequency):
    """
    Find financial statement files with flexible filename matching.
    """
    data_dir = os.path.join(BASE_OUTPUT_FOLDER, symbol)
    
    if not os.path.exists(data_dir):
        return None
    
    # Pattern to match files
    patterns = [
        f"{symbol}_AV_{statement_type}_{frequency}.csv",  # Correct format
        f"{symbol}_AV_{statement_type}_{frequency}.*",    # Any extension
        f"*{symbol}_AV_{statement_type}_{frequency}*",    # Any prefix/suffix
    ]
    
    for pattern in patterns:
        matches = glob.glob(os.path.join(data_dir, pattern))
        if matches:
            return matches[0]
    
    return None

def load_and_clean_financial_statement(symbol, statement_type, frequency):
    """
    Loads, cleans, and standardizes a single financial statement for a given symbol.
    """
    filepath = find_financial_statement_file(symbol, statement_type, frequency)
    if not filepath:
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        
        if df.empty:
            return pd.DataFrame()

        # 1. Rename columns for consistency - only rename columns that exist
        existing_columns = df.columns
        rename_dict = {}
        for old_col, new_col in FINANCIAL_STATEMENT_COL_MAPPING.items():
            if old_col in existing_columns:
                rename_dict[old_col] = new_col
        df = df.rename(columns=rename_dict)

        # 2. Convert 'Date' column to datetime and set as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            df = df.set_index('Date').sort_index()
        else:
            return pd.DataFrame()

        # 3. Convert relevant columns to numeric
        for col in df.columns:
            if col != 'Currency' and col in VALUES_TO_CONVERT:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. Handle cash flow statement specific conversions
        if statement_type == 'cash_flow':
            for col in ['Capital Expenditures', 'Dividends Paid', 'Dividends Paid Common', 
                       'Payments for Stock Repurchase']:
                if col in df.columns:
                    df[col] = df[col].abs() * -1

        # 5. Fill NaN values with 0 for financial data
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

def load_and_clean_yfinance_prices(symbol):
    """Loads and cleans yfinance historical stock prices."""
    filepath = os.path.join(BASE_OUTPUT_FOLDER, symbol, f"{symbol}_yfinance_prices.csv")
    if not os.path.exists(filepath):
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.set_index('Date').sort_index()
        
        # Convert price columns to numeric
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.fillna(0)
    except Exception as e:
        return pd.DataFrame()

def load_and_clean_yfinance_analyst_data(symbol):
    """Loads and cleans yfinance analyst recommendations and targets."""
    analyst_data = {
        'recommendations': pd.DataFrame(),
        'targets': pd.DataFrame(),
        'growth_estimates': pd.DataFrame()
    }

    # Recommendations
    rec_filepath = os.path.join(BASE_OUTPUT_FOLDER, symbol, f"{symbol}_yfinance_recommendations.csv")
    if os.path.exists(rec_filepath):
        try:
            rec_df = pd.read_csv(rec_filepath)
            if not rec_df.empty and 'Date' in rec_df.columns:
                rec_df['Date'] = pd.to_datetime(rec_df['Date'], utc=True)
                rec_df = rec_df.set_index('Date').sort_index()
                analyst_data['recommendations'] = rec_df
        except:
            pass

    # Analyst Targets
    targets_filepath = os.path.join(BASE_OUTPUT_FOLDER, symbol, f"{symbol}_yfinance_analyst_targets.csv")
    if os.path.exists(targets_filepath):
        try:
            targets_df = pd.read_csv(targets_filepath)
            analyst_data['targets'] = targets_df
        except:
            pass

    # Growth Estimates
    growth_filepath = os.path.join(BASE_OUTPUT_FOLDER, symbol, f"{symbol}_yfinance_growth_estimates.csv")
    if os.path.exists(growth_filepath):
        try:
            growth_df = pd.read_csv(growth_filepath)
            analyst_data['growth_estimates'] = growth_df
        except:
            pass

    return analyst_data

def consolidate_financial_data():
    """
    Main function to consolidate and clean all financial data for all companies.
    """
    all_companies_data = {}

    for symbol in COMPANIES:
        print(f"\n--- Cleaning and consolidating data for {symbol} ---")
        
        company_data = {}

        # Define the mapping between statement types and dictionary keys
        statement_mapping = {
            'income_statement': 'IS',
            'balance_sheet': 'BS', 
            'cash_flow': 'CF'
        }
        
        # Load financial statements
        for statement_type in ['income_statement', 'balance_sheet', 'cash_flow']:
            for frequency in ['annual', 'quarterly']:
                key_prefix = statement_mapping[statement_type]
                key = f"{key_prefix}_{frequency.capitalize()}"
                
                df = load_and_clean_financial_statement(symbol, statement_type, frequency)
                company_data[key] = df
                
                if not df.empty:
                    print(f"✓ Loaded {key} ({len(df)} records, {len(df.columns)} columns)")
                    if key == 'IS_Annual':
                        print(f"  Available columns: {list(df.columns)}")
                else:
                    print(f"✗ Missing {key}")

        # Load yfinance data
        company_data['Stock_Prices'] = load_and_clean_yfinance_prices(symbol)
        if not company_data['Stock_Prices'].empty:
            print(f"✓ Loaded stock prices ({len(company_data['Stock_Prices'])} records)")
        
        analyst_data = load_and_clean_yfinance_analyst_data(symbol)
        company_data['Analyst_Recommendations'] = analyst_data['recommendations']
        company_data['Analyst_Targets'] = analyst_data['targets']
        company_data['Analyst_Growth_Estimates'] = analyst_data['growth_estimates']
        
        if not analyst_data['growth_estimates'].empty:
            print(f"✓ Loaded analyst growth estimates")

        all_companies_data[symbol] = company_data
        print(f"Finished consolidating data for {symbol}.")

    return all_companies_data

def print_data_summary(cleaned_data):
    """Print a summary of the cleaned data"""
    print(f"\n=== DATA CLEANING SUMMARY ===")
    print(f"Cleaned data for {len(cleaned_data)} companies")
    
    for symbol, data_dict in cleaned_data.items():
        print(f"\n{symbol}:")
        
        # Financial statements
        for key in ['IS_Annual', 'BS_Annual', 'CF_Annual', 'IS_Quarterly', 'BS_Quarterly', 'CF_Quarterly']:
            df = data_dict.get(key, pd.DataFrame())
            if not df.empty:
                print(f"  {key}: {len(df)} records, {len(df.columns)} columns")
            else:
                print(f"  {key}: No data available")
        
        # Stock prices
        prices_df = data_dict.get('Stock_Prices', pd.DataFrame())
        if not prices_df.empty:
            print(f"  Stock Prices: {len(prices_df)} records")
        
        # Analyst data
        growth_df = data_dict.get('Analyst_Growth_Estimates', pd.DataFrame())
        if not growth_df.empty:
            print(f"  Growth Estimates: Available")

if __name__ == "__main__":
    cleaned_data = consolidate_financial_data()
    
    print_data_summary(cleaned_data)
    
    if cleaned_data.get('NVDA'):
        nvda_data = cleaned_data['NVDA']
        
        # Check annual income statement with correct column names
        if nvda_data.get('IS_Annual') is not None and not nvda_data['IS_Annual'].empty:
            print(f"\nNVDA Annual Income Statement ({len(nvda_data['IS_Annual'])} records):")
            
            # Check which columns are actually available
            available_cols = nvda_data['IS_Annual'].columns.tolist()
            print(f"Available columns: {available_cols}")
            
            # Use columns that actually exist
            display_cols = []
            for col in ['Revenue', 'Net Income', 'ebitda', 'grossProfit']:
                if col in available_cols:
                    display_cols.append(col)
            
            if display_cols:
                print(nvda_data['IS_Annual'][display_cols].tail())
            else:
                print("No common financial columns found")
        else:
            print(f"\nNVDA Annual Income Statement: No data available")
        
        # Check stock prices
        if nvda_data.get('Stock_Prices') is not None and not nvda_data['Stock_Prices'].empty:
            print(f"\nNVDA Stock Prices (latest 5):")
            print(nvda_data['Stock_Prices'][['Close', 'Volume']].tail())

    print("\n--- Data Cleaning and Consolidation Complete ---")