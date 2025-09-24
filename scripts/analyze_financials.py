import pandas as pd
import numpy as np
import yfinance as yf
from clean import consolidate_financial_data, COMPANIES

# --- Configuration ---
REPORTING_DATE = pd.Timestamp.now(tz='UTC')
RISK_FREE_RATE = 0.04
MARKET_RISK_PREMIUM = 0.055

def calculate_ttm(df, metric_column, periods=4):
    """Calculates Trailing Twelve Months (TTM) for a given metric from quarterly data."""
    if df.empty or metric_column not in df.columns:
        return np.nan, pd.Series()
    
    df = df.sort_index()
    
    if len(df) >= periods:
        last_4_quarters = df[metric_column].tail(periods)
        ttm_value = last_4_quarters.sum()
        return ttm_value, pd.Series()
    else:
        return np.nan, pd.Series()

def format_currency(value, decimals=2):
    """Format currency values with appropriate suffixes"""
    if pd.isna(value):
        return "N/A"
    
    if abs(value) >= 1000:  # Trillions
        return f"${value/1000:.{decimals}f}T"
    elif abs(value) >= 1:   # Billions
        return f"${value:.{decimals}f}B"
    else:                   # Millions
        return f"${value*1000:.0f}M"

def format_ratio(value, decimals=2):
    """Format ratio values"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"

def format_percentage(value, decimals=1):
    """Format percentage values"""
    if pd.isna(value):
        return "N/A"
    return f"{value*100:.{decimals}f}%"

def calculate_key_metrics(company_data):
    """Calculates TTM metrics, current market values, and initial valuation multiples."""
    metrics = {}

    for symbol, data_dict in company_data.items():
        sym_metrics = {'symbol': symbol}

        # --- YFINANCE DATA ---
        prices_df = data_dict.get('Stock_Prices')
        analyst_targets_df = data_dict.get('Analyst_Targets')
        
        if not prices_df.empty:
            latest_price = prices_df['Close'].iloc[-1]
            sym_metrics['Latest_Stock_Price'] = latest_price
            
            # Shares Outstanding
            bs_quarterly = data_dict.get('BS_Quarterly')
            bs_annual = data_dict.get('BS_Annual')

            shares_outstanding = np.nan
            shares_cols = ['Shares Outstanding', 'commonStockSharesOutstanding']
            
            if not bs_quarterly.empty:
                for col in shares_cols:
                    if col in bs_quarterly.columns:
                        shares_outstanding = bs_quarterly[col].iloc[-1]
                        break
            
            if pd.isna(shares_outstanding) and not bs_annual.empty:
                for col in shares_cols:
                    if col in bs_annual.columns:
                        shares_outstanding = bs_annual[col].iloc[-1]
                        break
            
            if not pd.isna(shares_outstanding) and shares_outstanding > 0:
                shares_outstanding_b = shares_outstanding / 1e9
                sym_metrics['Shares_Outstanding_B'] = shares_outstanding_b
                market_cap_b = latest_price * shares_outstanding_b
                sym_metrics['Market_Cap_B'] = market_cap_b

        # Analyst Targets
        if not analyst_targets_df.empty and 'Target Mean Price' in analyst_targets_df.columns:
            sym_metrics['Target_Mean_Price'] = analyst_targets_df['Target Mean Price'].iloc[0]

        # --- FINANCIAL METRICS ---
        is_quarterly = data_dict.get('IS_Quarterly')
        bs_quarterly = data_dict.get('BS_Quarterly')
        cf_quarterly = data_dict.get('CF_Quarterly')
        
        # TTM Calculations
        if not is_quarterly.empty:
            sym_metrics['TTM_Revenue_B'], _ = calculate_ttm(is_quarterly, 'Revenue')
            sym_metrics['TTM_Net_Income_B'], _ = calculate_ttm(is_quarterly, 'Net Income')
            sym_metrics['TTM_EBITDA_B'], _ = calculate_ttm(is_quarterly, 'EBITDA')
            sym_metrics['TTM_EBIT_B'], _ = calculate_ttm(is_quarterly, 'EBIT')
            sym_metrics['TTM_Gross_Profit_B'], _ = calculate_ttm(is_quarterly, 'Gross Profit')
            
            # Convert to billions
            for key in ['TTM_Revenue_B', 'TTM_Net_Income_B', 'TTM_EBITDA_B', 'TTM_EBIT_B', 'TTM_Gross_Profit_B']:
                if not pd.isna(sym_metrics[key]):
                    sym_metrics[key] = sym_metrics[key] / 1e9

        # Cash Flow
        if not cf_quarterly.empty:
            sym_metrics['TTM_Operating_Cash_Flow_B'], _ = calculate_ttm(cf_quarterly, 'Operating Cash Flow')
            sym_metrics['TTM_Capital_Expenditures_B'], _ = calculate_ttm(cf_quarterly, 'Capital Expenditures')
            
            for key in ['TTM_Operating_Cash_Flow_B', 'TTM_Capital_Expenditures_B']:
                if not pd.isna(sym_metrics[key]):
                    sym_metrics[key] = sym_metrics[key] / 1e9
            
            if not pd.isna(sym_metrics.get('TTM_Operating_Cash_Flow_B')) and not pd.isna(sym_metrics.get('TTM_Capital_Expenditures_B')):
                sym_metrics['TTM_FCF_B'] = sym_metrics['TTM_Operating_Cash_Flow_B'] + sym_metrics['TTM_Capital_Expenditures_B']

        # Balance Sheet for EV Calculation
        if not bs_quarterly.empty:
            long_term_debt = bs_quarterly.get('Long Term Debt', pd.Series([0])).iloc[-1]
            short_term_debt = bs_quarterly.get('Short Term Debt', pd.Series([0])).iloc[-1]
            cash_equivalents = bs_quarterly.get('Cash & Equivalents', pd.Series([0])).iloc[-1]
            
            sym_metrics['Latest_Total_Debt_B'] = (long_term_debt + short_term_debt) / 1e9
            sym_metrics['Latest_Cash_Equivalents_B'] = cash_equivalents / 1e9
            
            if not pd.isna(sym_metrics.get('Market_Cap_B')) and not pd.isna(sym_metrics['Latest_Total_Debt_B']) and not pd.isna(sym_metrics['Latest_Cash_Equivalents_B']):
                sym_metrics['Enterprise_Value_B'] = (sym_metrics['Market_Cap_B'] + 
                                                   sym_metrics['Latest_Total_Debt_B'] - 
                                                   sym_metrics['Latest_Cash_Equivalents_B'])
            else:
                sym_metrics['Enterprise_Value_B'] = np.nan

        # --- VALUATION MULTIPLES ---
        # P/E Ratio
        if not pd.isna(sym_metrics.get('Market_Cap_B')) and not pd.isna(sym_metrics.get('TTM_Net_Income_B')):
            if sym_metrics['TTM_Net_Income_B'] > 0:
                sym_metrics['P_E'] = sym_metrics['Market_Cap_B'] / sym_metrics['TTM_Net_Income_B']
        
        # EV/EBITDA
        if not pd.isna(sym_metrics.get('Enterprise_Value_B')) and not pd.isna(sym_metrics.get('TTM_EBITDA_B')):
            if sym_metrics['TTM_EBITDA_B'] > 0:
                sym_metrics['EV_EBITDA'] = sym_metrics['Enterprise_Value_B'] / sym_metrics['TTM_EBITDA_B']
        
        # P/S Ratio
        if not pd.isna(sym_metrics.get('Market_Cap_B')) and not pd.isna(sym_metrics.get('TTM_Revenue_B')):
            if sym_metrics['TTM_Revenue_B'] > 0:
                sym_metrics['P_S'] = sym_metrics['Market_Cap_B'] / sym_metrics['TTM_Revenue_B']

        # EV/Sales
        if not pd.isna(sym_metrics.get('Enterprise_Value_B')) and not pd.isna(sym_metrics.get('TTM_Revenue_B')):
            if sym_metrics['TTM_Revenue_B'] > 0:
                sym_metrics['EV_Sales'] = sym_metrics['Enterprise_Value_B'] / sym_metrics['TTM_Revenue_B']

        # Profit Margins
        if not pd.isna(sym_metrics.get('TTM_Net_Income_B')) and not pd.isna(sym_metrics.get('TTM_Revenue_B')):
            if sym_metrics['TTM_Revenue_B'] > 0:
                sym_metrics['Net_Margin'] = sym_metrics['TTM_Net_Income_B'] / sym_metrics['TTM_Revenue_B']

        metrics[symbol] = sym_metrics
    
    return pd.DataFrame.from_dict(metrics, orient='index')

def calculate_wacc(symbol, data_dict, market_risk_premium, risk_free_rate):
    """Calculates the Weighted Average Cost of Capital (WACC)."""
    bs_annual = data_dict.get('BS_Annual')
    is_annual = data_dict.get('IS_Annual')
    prices_df = data_dict.get('Stock_Prices')
    
    cost_of_equity = np.nan
    cost_of_debt_pre_tax = np.nan
    effective_tax_rate = np.nan
    market_cap = np.nan
    total_debt = np.nan
    beta = np.nan

    # Cost of Equity (CAPM)
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        beta = info.get('beta', np.nan)
        if not pd.isna(beta):
            cost_of_equity = risk_free_rate + (beta * market_risk_premium)
    except:
        pass

    # Cost of Debt
    if not bs_annual.empty and not is_annual.empty:
        try:
            long_term_debt = bs_annual.get('Long Term Debt', pd.Series([0])).iloc[-1]
            short_term_debt = bs_annual.get('Short Term Debt', pd.Series([0])).iloc[-1]
            latest_total_debt = long_term_debt + short_term_debt
            latest_interest_expense = is_annual.get('Interest Expense', pd.Series([0])).iloc[-1]
            
            if latest_total_debt > 0 and latest_interest_expense > 0:
                cost_of_debt_pre_tax = latest_interest_expense / latest_total_debt
                cost_of_debt_pre_tax = min(cost_of_debt_pre_tax, 0.15)
        except:
            pass

    # Effective Tax Rate
    if not is_annual.empty:
        try:
            income_before_tax = is_annual.get('Income Before Tax', pd.Series([0])).iloc[-1]
            income_tax_expense = is_annual.get('Income Tax Expense', pd.Series([0])).iloc[-1]
            
            if income_before_tax > 0:
                effective_tax_rate = income_tax_expense / income_before_tax
                effective_tax_rate = max(0, min(1, effective_tax_rate))
            else:
                effective_tax_rate = 0.25
        except:
            pass

    # Market Values
    if not prices_df.empty and not bs_annual.empty:
        try:
            latest_price = prices_df['Close'].iloc[-1]
            shares_cols = ['Shares Outstanding', 'commonStockSharesOutstanding']
            shares_outstanding = None
            for col in shares_cols:
                if col in bs_annual.columns:
                    shares_outstanding = bs_annual[col].iloc[-1]
                    break
            
            if shares_outstanding and shares_outstanding > 0:
                market_cap = latest_price * shares_outstanding
                total_debt = bs_annual.get('Long Term Debt', pd.Series([0])).iloc[-1] + bs_annual.get('Short Term Debt', pd.Series([0])).iloc[-1]
        except:
            pass

    # Calculate WACC
    if (not pd.isna(cost_of_equity) and not pd.isna(cost_of_debt_pre_tax) and 
        not pd.isna(effective_tax_rate) and not pd.isna(market_cap) and 
        not pd.isna(total_debt) and (market_cap + total_debt) > 0):
        
        equity_weight = market_cap / (market_cap + total_debt)
        debt_weight = total_debt / (market_cap + total_debt)
        
        wacc = (equity_weight * cost_of_equity) + \
               (debt_weight * cost_of_debt_pre_tax * (1 - effective_tax_rate))
        
        return wacc, cost_of_equity, cost_of_debt_pre_tax, effective_tax_rate, beta
    else:
        return np.nan, cost_of_equity, cost_of_debt_pre_tax, effective_tax_rate, beta

def print_formatted_results(company_metrics_df, wacc_results):
    """Print formatted results"""
    
    print("\n" + "="*100)
    print("FINANCIAL ANALYSIS RESULTS")
    print("="*100)
    
    # Company Metrics Table
    print("\n📊 COMPANY VALUATION METRICS")
    print("-" * 100)
    
    headers = ["Symbol", "Price", "Market Cap", "EV", "Revenue", "Net Income", "EBITDA", "P/E", "EV/EBITDA", "P/S", "Net Margin"]
    format_funcs = [None, "${:.2f}", format_currency, format_currency, format_currency, format_currency, 
                   format_currency, format_ratio, format_ratio, format_ratio, format_percentage]
    
    # Create formatted rows
    rows = []
    for symbol in COMPANIES:
        if symbol in company_metrics_df.index:
            metrics = company_metrics_df.loc[symbol]
            row = [
                symbol,
                metrics.get('Latest_Stock_Price', np.nan),
                metrics.get('Market_Cap_B', np.nan),
                metrics.get('Enterprise_Value_B', np.nan),
                metrics.get('TTM_Revenue_B', np.nan),
                metrics.get('TTM_Net_Income_B', np.nan),
                metrics.get('TTM_EBITDA_B', np.nan),
                metrics.get('P_E', np.nan),
                metrics.get('EV_EBITDA', np.nan),
                metrics.get('P_S', np.nan),
                metrics.get('Net_Margin', np.nan)
            ]
            rows.append(row)
    
    # Print header
    header_fmt = "{:<8} {:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<8} {:<10} {:<8} {:<10}"
    print(header_fmt.format(*headers))
    print("-" * 100)
    
    # Print rows
    for row in rows:
        formatted_row = []
        for i, (value, fmt_func) in enumerate(zip(row, format_funcs)):
            if fmt_func is None:
                formatted_row.append(str(value) if not pd.isna(value) else "N/A")
            else:
                if fmt_func == "${:.2f}": 
                    formatted_value = f"${value:.2f}" if not pd.isna(value) else "N/A"
                else:
                    formatted_value = fmt_func(value)
                formatted_row.append(formatted_value)
        print(header_fmt.format(*formatted_row))
    
    # WACC Results Table
    print("\n📈 WEIGHTED AVERAGE COST OF CAPITAL (WACC)")
    print("-" * 80)
    
    wacc_headers = ["Symbol", "Beta", "Cost of Equity", "Cost of Debt", "Tax Rate", "WACC"]
    wacc_format_funcs = [None, format_ratio, format_percentage, format_percentage, format_percentage, format_percentage]
    
    wacc_rows = []
    for symbol in COMPANIES:
        if symbol in wacc_results:
            wacc_data = wacc_results[symbol]
            row = [
                symbol,
                wacc_data.get('Beta', np.nan),
                wacc_data.get('Cost_of_Equity', np.nan),
                wacc_data.get('Cost_of_Debt_PreTax', np.nan),
                wacc_data.get('Effective_Tax_Rate', np.nan),
                wacc_data.get('WACC', np.nan)
            ]
            wacc_rows.append(row)
    
    # Print WACC header
    wacc_header_fmt = "{:<8} {:<6} {:<14} {:<12} {:<10} {:<8}"
    print(wacc_header_fmt.format(*wacc_headers))
    print("-" * 80)
    
    # Print WACC rows
    for row in wacc_rows:
        formatted_row = []
        for i, (value, fmt_func) in enumerate(zip(row, wacc_format_funcs)):
            if fmt_func is None:
                formatted_row.append(str(value) if not pd.isna(value) else "N/A")
            else:
                formatted_row.append(fmt_func(value))
        print(wacc_header_fmt.format(*formatted_row))
    
    # Key Insights
    print("\n💡 KEY INSIGHTS")
    print("-" * 50)
    
    # Find company with highest/lowest multiples
    if not company_metrics_df.empty:
        valid_pe = company_metrics_df[company_metrics_df['P_E'] > 0]['P_E'] if 'P_E' in company_metrics_df.columns else pd.Series()
        valid_ev_ebitda = company_metrics_df[company_metrics_df['EV_EBITDA'] > 0]['EV_EBITDA'] if 'EV_EBITDA' in company_metrics_df.columns else pd.Series()
        
        if not valid_pe.empty:
            highest_pe = valid_pe.idxmax()
            lowest_pe = valid_pe.idxmin()
            print(f"• Highest P/E Ratio: {highest_pe} ({format_ratio(company_metrics_df.loc[highest_pe, 'P_E'])})")
            print(f"• Lowest P/E Ratio: {lowest_pe} ({format_ratio(company_metrics_df.loc[lowest_pe, 'P_E'])})")
        
        if not valid_ev_ebitda.empty:
            highest_ev_ebitda = valid_ev_ebitda.idxmax()
            print(f"• Highest EV/EBITDA: {highest_ev_ebitda} ({format_ratio(company_metrics_df.loc[highest_ev_ebitda, 'EV_EBITDA'])})")
    
    if wacc_results:
        valid_wacc = {k: v for k, v in wacc_results.items() if not pd.isna(v['WACC'])}
        if valid_wacc:
            highest_wacc = max(valid_wacc.items(), key=lambda x: x[1]['WACC'])
            lowest_wacc = min(valid_wacc.items(), key=lambda x: x[1]['WACC'])
            
            print(f"• Highest WACC: {highest_wacc[0]} ({format_percentage(highest_wacc[1]['WACC'])})")
            print(f"• Lowest WACC: {lowest_wacc[0]} ({format_percentage(lowest_wacc[1]['WACC'])})")

if __name__ == "__main__":
    # Load Cleaned Data
    all_companies_data = consolidate_financial_data()
    print("\n--- Starting Financial Analysis ---")

    # Calculate Key Metrics
    company_metrics_df = calculate_key_metrics(all_companies_data)

    # Calculate WACC
    wacc_results = {}
    for symbol in COMPANIES:
        if symbol in all_companies_data:
            wacc, coe, cod_pretax, tax_rate, beta = calculate_wacc(
                symbol, 
                all_companies_data[symbol], 
                MARKET_RISK_PREMIUM, 
                RISK_FREE_RATE
            )
            wacc_results[symbol] = {
                'WACC': wacc,
                'Cost_of_Equity': coe,
                'Cost_of_Debt_PreTax': cod_pretax,
                'Effective_Tax_Rate': tax_rate,
                'Beta': beta
            }

    # Print formatted results
    print_formatted_results(company_metrics_df, wacc_results)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE ✅")
    print("="*100)