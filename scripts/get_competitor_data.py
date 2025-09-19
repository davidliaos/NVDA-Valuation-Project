import requests
import pandas as pd
import os
from dotenv import load_dotenv
import time
import yfinance as yf

load_dotenv()
#Constants 
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
COMPETITOR_SYMBOLS = ["AMD", "INTC", "AVGO", "QCOM", "TSM"] # List of competitor tickers
BASE_OUTPUT_FOLDER = "data" # Main data directory

#Constants for Retry Logic
MAX_RETRIES = 5
RETRY_DELAY_ALPHA_VANTAGE = 15 
RETRY_DELAY_YFINANCE = 5     
LONG_API_LIMIT_WAIT = 65    
GLOBAL_COMP_WAIT = 60       



def fetch_alpha_vantage_data(function, symbol, max_retries=MAX_RETRIES):
    """
    Fetches data from Alpha Vantage API for financial statements with retry logic.
    """
    url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={API_KEY}"
    
    retries = 0
    while retries < max_retries:
        print(f"Fetching {function} for {symbol} (Attempt {retries + 1}/{max_retries})...")
        try:
            r = requests.get(url)
            r.raise_for_status()

            data = r.json()
            if "Error Message" in data:
                print(f"API Error for {function} ({symbol}): {data['Error Message']}")
                retries += 1
                time.sleep(RETRY_DELAY_ALPHA_VANTAGE)
                continue
            if "Note" in data:
                print(f"API Note for {function}: {data['Note']}")
                print(f"Likely hit API rate limit. Waiting {LONG_API_LIMIT_WAIT} seconds and retrying...")
                time.sleep(LONG_API_LIMIT_WAIT)
                continue
            return data
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {function} ({symbol}): {e}")
            retries += 1
            time.sleep(RETRY_DELAY_ALPHA_VANTAGE)
    print(f"Failed to fetch {function} for {symbol} after {max_retries} attempts.")
    return None

def save_financial_data(data, filename_suffix, entry_key, symbol, output_folder, max_retries=MAX_RETRIES):
    """Saves financial statement data from Alpha Vantage JSON response to CSV with retry logic."""
    if not data or entry_key not in data:
        print(f"No valid data to save for {filename_suffix} with key {entry_key} for {symbol}.")
        return False

    filepath = os.path.join(output_folder, f"{symbol}_AV_{filename_suffix}.csv")
    retries = 0
    while retries < max_retries:
        try:
            df = pd.DataFrame(data[entry_key])
            if 'fiscalDateEnding' in df.columns:
                df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
            df.to_csv(filepath, index=False)
            print(f"Saved {filepath}")
            return True
        except Exception as e:
            print(f"Error saving {filepath} (Attempt {retries + 1}/{max_retries}): {e}")
            retries += 1
            time.sleep(RETRY_DELAY_ALPHA_VANTAGE)
    print(f"Failed to save {filepath} after {max_retries} attempts.")
    return False

def fetch_yfinance_stock_data(symbol, output_folder, period="max", max_retries=MAX_RETRIES):
    """
    Fetches historical stock prices from Yahoo Finance using yfinance with retry logic.
    """
    filepath = os.path.join(output_folder, f"{symbol}_yfinance_prices.csv")
    retries = 0
    while retries < max_retries:
        print(f"Fetching {period} historical stock prices for {symbol} from yfinance (Attempt {retries + 1}/{max_retries})...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            if not df.empty:
                df.index.name = "Date"
                df.to_csv(filepath, index=True)
                print(f"Saved {filepath}")
                return True
            else:
                print(f"No data fetched for {symbol} with period {period} from yfinance (empty DataFrame).")
                retries += 1
                time.sleep(RETRY_DELAY_YFINANCE)
        except Exception as e:
            print(f"Error fetching yfinance stock data for {symbol} (Attempt {retries + 1}/{max_retries}): {e}")
            retries += 1
            time.sleep(RETRY_DELAY_YFINANCE)
    print(f"Failed to fetch yfinance stock data for {symbol} after {max_retries} attempts.")
    return False

def fetch_yfinance_analyst_data(symbol, output_folder, max_retries=MAX_RETRIES):
    """
    Fetches analyst recommendations, price targets, and estimates from Yahoo Finance using yfinance with retry logic.
    """
    retries = 0
    while retries < max_retries:
        print(f"Fetching analyst data for {symbol} from yfinance (Attempt {retries + 1}/{max_retries})...")
        try:
            ticker = yf.Ticker(symbol)

            recommendations = ticker.recommendations
            if recommendations is not None and not recommendations.empty:
                recommendations.index.name = "Date"
                recommendations_path = os.path.join(output_folder, f"{symbol}_yfinance_recommendations.csv")
                recommendations.to_csv(recommendations_path, index=True)
                print(f"Saved {recommendations_path}")
            else:
                print(f"No recommendations found for {symbol}.")

            # Basic Stock Info
            info = ticker.info
            analyst_info = {}
            if 'targetHighPrice' in info:
                analyst_info['Target High Price'] = info['targetHighPrice']
            if 'targetLowPrice' in info:
                analyst_info['Target Low Price'] = info['targetLowPrice']
            if 'targetMeanPrice' in info:
                analyst_info['Target Mean Price'] = info['targetMeanPrice']
            if 'numberOfAnalystOpinions' in info:
                analyst_info['Number of Analyst Opinions'] = info['numberOfAnalystOpinions']
            
            if analyst_info:
                analyst_df = pd.DataFrame([analyst_info])
                analyst_df.to_csv(os.path.join(output_folder, f"{symbol}_yfinance_analyst_targets.csv"), index=False)
                print(f"Saved {os.path.join(output_folder, f'{symbol}_yfinance_analyst_targets.csv')}")
            else:
                print(f"No analyst target price info found for {symbol}.")
                
            #Earnings and Revenue Growth Estimates
            analyst_estimates_data = {}
            if 'earningsQuarterlyGrowth' in info:
                analyst_estimates_data['Earnings Quarterly Growth'] = info['earningsQuarterlyGrowth']
            if 'earningsAnnualGrowth' in info:
                analyst_estimates_data['Earnings Annual Growth'] = info['earningsAnnualGrowth']
            if 'revenueQuarterlyGrowth' in info:
                analyst_estimates_data['Revenue Quarterly Growth'] = info['revenueQuarterlyGrowth']
            if 'revenueAnnualGrowth' in info:
                analyst_estimates_data['Revenue Annual Growth'] = info['revenueAnnualGrowth']
            
            if analyst_estimates_data:
                estimates_df = pd.DataFrame([analyst_estimates_data])
                estimates_df.to_csv(os.path.join(output_folder, f"{symbol}_yfinance_growth_estimates.csv"), index=False)
                print(f"Saved {os.path.join(output_folder, f'{symbol}_yfinance_growth_estimates.csv')}")
            else:
                print(f"No detailed growth estimates found for {symbol} in yfinance.info.")
            
            return True
        except Exception as e:
            print(f"Error fetching yfinance analyst data for {symbol} (Attempt {retries + 1}/{max_retries}): {e}")
            retries += 1
            time.sleep(RETRY_DELAY_YFINANCE)
    print(f"Failed to fetch yfinance analyst data for {symbol} after {max_retries} attempts.")
    return False


#Main execution for competitor data
def main_competitors():
    print("Starting data acquisition for competitors...")
    for symbol in COMPETITOR_SYMBOLS:
        print(f"\n--- Processing {symbol} ---")
        
        # Create symbol-specific folder for competitor
        current_symbol_output_folder = os.path.join(BASE_OUTPUT_FOLDER, symbol)
        if not os.path.exists(current_symbol_output_folder):
            os.makedirs(current_symbol_output_folder)

        # Alpha Vantage fetches with retry
        data_functions = {
            "INCOME_STATEMENT": {"annual": "annualReports", "quarterly": "quarterlyReports"},
            "BALANCE_SHEET": {"annual": "annualReports", "quarterly": "quarterlyReports"},
            "CASH_FLOW": {"annual": "annualReports", "quarterly": "quarterlyReports"}
        }

        for func_name, reports in data_functions.items():
            data = fetch_alpha_vantage_data(func_name, symbol)
            if data:
                save_financial_data(data, "income_statement_annual", reports["annual"], symbol, current_symbol_output_folder)
                save_financial_data(data, "income_statement_quarterly", reports["quarterly"], symbol, current_symbol_output_folder)
            
            # A short pause between different statements
            time.sleep(RETRY_DELAY_ALPHA_VANTAGE / 2) 
        
        # yfinance fetches with retry
        fetch_yfinance_stock_data(symbol, current_symbol_output_folder, period="max")
        fetch_yfinance_analyst_data(symbol, current_symbol_output_folder)
        
        print(f"--- Finished {symbol} ---\n")
        time.sleep(GLOBAL_COMP_WAIT) # Long pause before starting next competitor

    print("Competitor data acquisition complete!")

if __name__ == "__main__":
    main_competitors()