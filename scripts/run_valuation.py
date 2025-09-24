import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from three_statement_model import ThreeStatementModel, run_three_statement_analysis
from clean import consolidate_financial_data
from analyze_financials import calculate_key_metrics, calculate_wacc
import yfinance as yf

# Configuration
RISK_FREE_RATE = 0.04
MARKET_RISK_PREMIUM = 0.055

def create_output_directory():
    """Create output directory if it doesn't exist"""
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/charts', exist_ok=True)
    os.makedirs('output/reports', exist_ok=True)

def run_dcf_valuation(model, wacc: float, symbol: str):
    """Run Discounted Cash Flow valuation"""
    print(f"\n{'='*50}")
    print(f"DCF VALUATION - {symbol}")
    print(f"{'='*50}")
    
    # Get free cash flow projections
    valuation_data = model.get_valuation_data()
    if not valuation_data:
        print("No valuation data available")
        return None
    
    free_cash_flows = valuation_data['free_cash_flows']
    terminal_growth = valuation_data['terminal_growth_rate']
    projection_years = valuation_data['projection_years']
    
    print(f"WACC used: {wacc:.2%}")
    print(f"Terminal growth rate: {terminal_growth:.2%}")
    
    # Discount projected free cash flows
    discounted_cash_flows = []
    for year, fcf in enumerate(free_cash_flows, 1):
        discount_factor = 1 / ((1 + wacc) ** year)
        discounted_fcf = fcf * discount_factor
        discounted_cash_flows.append(discounted_fcf)
        print(f"Year {year}: FCF = ${fcf/1e9:.2f}B, Discounted = ${discounted_fcf/1e9:.2f}B")
    
    # Calculate terminal value
    final_year_fcf = free_cash_flows.iloc[-1]
    terminal_value = final_year_fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    discounted_terminal_value = terminal_value / ((1 + wacc) ** projection_years)
    
    print(f"\nTerminal Value: ${terminal_value/1e9:.2f}B")
    print(f"Discounted Terminal Value: ${discounted_terminal_value/1e9:.2f}B")
    
    # Calculate enterprise value
    enterprise_value = sum(discounted_cash_flows) + discounted_terminal_value
    
    # Get current market data for comparison
    try:
        ticker = yf.Ticker(symbol)
        current_price = ticker.history(period='1d')['Close'].iloc[-1]
        info = ticker.info
        shares_outstanding = info.get('sharesOutstanding', 0)
        
        if shares_outstanding > 0:
            market_cap = current_price * shares_outstanding
            print(f"\nCurrent Market Cap: ${market_cap/1e9:.2f}B")
            print(f"DCF Enterprise Value: ${enterprise_value/1e9:.2f}B")
            
            # Calculate equity value per share
            # Simplified: EV = Market Cap + Debt - Cash
            # For proper calculation, we'd need up to date  debt and cash
            implied_equity_value = enterprise_value  # Simplified assumption
            implied_share_price = implied_equity_value / shares_outstanding
            
            print(f"Current Share Price: ${current_price:.2f}")
            print(f"Implied DCF Share Price: ${implied_share_price:.2f}")
            
            premium_discount = (implied_share_price / current_price - 1) * 100
            print(f"Valuation: {premium_discount:+.1f}% vs current price")
            
            return {
                'enterprise_value': enterprise_value,
                'implied_share_price': implied_share_price,
                'current_price': current_price,
                'premium_discount': premium_discount,
                'wacc': wacc,
                'terminal_growth': terminal_growth
            }
    
    except Exception as e:
        print(f"Error getting current market data: {e}")
    
    return None

def run_sensitivity_analysis(model, symbol: str, base_wacc: float, base_growth: float):
    """Run sensitivity analysis on key DCF assumptions"""
    print(f"\n{'='*50}")
    print(f"SENSITIVITY ANALYSIS - {symbol}")
    print(f"{'='*50}")
    
    # Test different WACC and growth rate scenarios
    wacc_scenarios = [base_wacc - 0.01, base_wacc, base_wacc + 0.01]  # -1%, base, +1%
    growth_scenarios = [base_growth - 0.01, base_growth, base_growth + 0.01]  # -1%, base, +1%
    
    sensitivity_matrix = pd.DataFrame(index=growth_scenarios, columns=wacc_scenarios)
    
    valuation_data = model.get_valuation_data()
    if not valuation_data:
        return None
    
    free_cash_flows = valuation_data['free_cash_flows']
    projection_years = valuation_data['projection_years']
    
    for growth_rate in growth_scenarios:
        for wacc_rate in wacc_scenarios:
            if wacc_rate <= growth_rate:  # Avoid division by zero
                sensitivity_matrix.loc[growth_rate, wacc_rate] = np.nan
                continue
            
            # Calculate DCF with these parameters
            discounted_cash_flows = []
            for year, fcf in enumerate(free_cash_flows, 1):
                discount_factor = 1 / ((1 + wacc_rate) ** year)
                discounted_cash_flows.append(fcf * discount_factor)
            
            final_year_fcf = free_cash_flows.iloc[-1]
            terminal_value = final_year_fcf * (1 + growth_rate) / (wacc_rate - growth_rate)
            discounted_terminal_value = terminal_value / ((1 + wacc_rate) ** projection_years)
            
            enterprise_value = sum(discounted_cash_flows) + discounted_terminal_value
            sensitivity_matrix.loc[growth_rate, wacc_rate] = enterprise_value / 1e9  # Billions
    
    print("Enterprise Value Sensitivity Matrix ($B):")
    print("Rows: Terminal Growth Rate, Columns: WACC")
    print(sensitivity_matrix.round(2))
    
    return sensitivity_matrix

def generate_valuation_report(model, dcf_results, sensitivity_results, symbol: str):
    """Generate a comprehensive valuation report"""
    report_path = f'output/reports/{symbol}_valuation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"COMPREHENSIVE VALUATION REPORT - {symbol}\n")
        f.write(f"{'='*60}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 50 + "\n")
        
        if dcf_results:
            f.write(f"Current Price: ${dcf_results['current_price']:.2f}\n")
            f.write(f"DCF Implied Price: ${dcf_results['implied_share_price']:.2f}\n")
            f.write(f"Valuation Gap: {dcf_results['premium_discount']:+.1f}%\n")
            f.write(f"WACC: {dcf_results['wacc']:.2%}\n")
            f.write(f"Terminal Growth: {dcf_results['terminal_growth']:.2%}\n\n")
        
        f.write("FINANCIAL PROJECTIONS\n")
        f.write("-" * 50 + "\n")
        
        # Add projection summary
        if hasattr(model, 'projected_data') and 'income_statement' in model.projected_data:
            revenues = model.projected_data['income_statement']['Revenue']
            f.write("Revenue Projections:\n")
            for i, (date, revenue) in enumerate(revenues.items()):
                f.write(f"  Year {i+1}: ${revenue/1e9:.2f}B\n")
        
        f.write("\nASSUMPTIONS\n")
        f.write("-" * 50 + "\n")
        for assumption, value in model.assumptions.items():
            if isinstance(value, list):
                f.write(f"{assumption}: {[f'{v:.2%}' for v in value]}\n")
            else:
                f.write(f"{assumption}: {value:.2%}\n")
        
        f.write("\nSENSITIVITY ANALYSIS\n")
        f.write("-" * 50 + "\n")
        if sensitivity_results is not None:
            f.write("Enterprise Value under different scenarios ($B):\n")
            f.write(str(sensitivity_results.round(2)))
        
        f.write(f"\n\nReport generated on: {pd.Timestamp.now()}\n")
    
    print(f"Valuation report saved to: {report_path}")

def plot_valuation_comparison(dcf_results, symbol: str):
    """Create visualization comparing DCF valuation to current price"""
    if not dcf_results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Price comparison
    prices = [dcf_results['current_price'], dcf_results['implied_share_price']]
    labels = ['Current Price', 'DCF Implied Price']
    colors = ['red' if dcf_results['premium_discount'] < 0 else 'green', 'blue']
    
    ax1.bar(labels, prices, color=colors, alpha=0.7)
    ax1.set_ylabel('Share Price ($)')
    ax1.set_title(f'{symbol} - Price Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(prices):
        ax1.text(i, v + max(prices)*0.01, f'${v:.2f}', ha='center', va='bottom')
    
    # Premium/Discount chart
    premium = dcf_results['premium_discount']
    ax2.bar(['Valuation Gap'], [premium], 
            color='green' if premium > 0 else 'red', alpha=0.7)
    ax2.set_ylabel('Premium/Discount (%)')
    ax2.set_title(f'DCF vs Current Price')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    
    # Add value label
    ax2.text(0, premium + (1 if premium > 0 else -1), f'{premium:+.1f}%', 
             ha='center', va='bottom' if premium > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(f'output/charts/{symbol}_valuation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the complete valuation analysis"""
    print("INITIATING COMPREHENSIVE VALUATION ANALYSIS")
    print("="*60)
    
    # Create output directories
    create_output_directory()
    
    # Load cleaned data
    print("Loading financial data...")
    all_companies_data = consolidate_financial_data()
    
    # Focus on NVIDIA for detailed analysis
    symbol = 'NVDA'
    
    if symbol not in all_companies_data:
        print(f"No data available for {symbol}")
        return
    
    print(f"\nAnalyzing {symbol} in detail...")
    
    try:
        # Calculate current metrics
        print("\n1. CALCULATING CURRENT METRICS")
        company_metrics = calculate_key_metrics({symbol: all_companies_data[symbol]})
        print(company_metrics.round(2))
        
        # Calculate WACC
        print("\n2. CALCULATING WACC")
        wacc, coe, cod_pretax, tax_rate, beta = calculate_wacc(
            symbol, 
            all_companies_data[symbol], 
            MARKET_RISK_PREMIUM, 
            RISK_FREE_RATE
        )
        
        print(f"WACC for {symbol}: {wacc:.2%}")
        print(f"Cost of Equity: {coe:.2%}")
        print(f"Cost of Debt: {cod_pretax:.2%}")
        print(f"Tax Rate: {tax_rate:.2%}")
        print(f"Beta: {beta:.2f}")
        
        # Create 3-statement model
        print("\n3. BUILDING 3-STATEMENT FINANCIAL MODEL")
        model = ThreeStatementModel(symbol, all_companies_data[symbol])
        
        # Run with custom assumptions if needed
        custom_assumptions = {
            'revenue_growth': [0.40, 0.30, 0.20, 0.15, 0.10],  # Realistic growth curve
            'terminal_growth': 0.03,
        }
        
        projections = model.run_complete_model(custom_assumptions)
        model.display_summary()
        
        # Run DCF valuation
        print("\n4. DISCOUNTED CASH FLOW VALUATION")
        dcf_results = run_dcf_valuation(model, wacc, symbol)
        
        # Run sensitivity analysis
        print("\n5. SENSITIVITY ANALYSIS")
        sensitivity_results = run_sensitivity_analysis(model, symbol, wacc, 0.03)
        
        # Generate reports and visualizations
        print("\n6. GENERATING REPORTS")
        generate_valuation_report(model, dcf_results, sensitivity_results, symbol)
        
        # Create visualizations
        model.plot_projections()
        if dcf_results:
            plot_valuation_comparison(dcf_results, symbol)
        
        print(f"\n{'='*60}")
        print("VALUATION ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        
        if dcf_results:
            status = "OVERVALUED" if dcf_results['premium_discount'] < 0 else "UNDERVALUED"
            print(f"CONCLUSION: {symbol} appears {status}")
            print(f"DCF Implied Price: ${dcf_results['implied_share_price']:.2f}")
            print(f"Current Price: ${dcf_results['current_price']:.2f}")
            print(f"Valuation Gap: {dcf_results['premium_discount']:+.1f}%")
        
    except Exception as e:
        print(f"Error in valuation analysis: {e}")
        import traceback
        traceback.print_exc()

def quick_multi_company_analysis():
    """Quick analysis for all companies"""
    print("\nQUICK MULTI-COMPANY ANALYSIS")
    print("="*50)
    
    all_companies_data = consolidate_financial_data()
    
    for symbol in ['NVDA', 'AMD', 'INTC', 'AVGO', 'QCOM', 'TSM']:
        if symbol in all_companies_data:
            try:
                print(f"\n{symbol}:")
                
                # Quick metrics
                metrics = calculate_key_metrics({symbol: all_companies_data[symbol]})
                if not metrics.empty and symbol in metrics.index:
                    current_price = metrics.loc[symbol, 'Latest_Stock_Price']
                    pe_ratio = metrics.loc[symbol, 'P_E'] if 'P_E' in metrics.columns else np.nan
                    ev_ebitda = metrics.loc[symbol, 'EV_EBITDA'] if 'EV_EBITDA' in metrics.columns else np.nan
                    
                    print(f"  Price: ${current_price:.2f}")
                    print(f"  P/E: {pe_ratio:.1f}" if not pd.isna(pe_ratio) else "  P/E: N/A")
                    print(f"  EV/EBITDA: {ev_ebitda:.1f}" if not pd.isna(ev_ebitda) else "  EV/EBITDA: N/A")
                
                # Quick WACC
                wacc, _, _, _, _ = calculate_wacc(symbol, all_companies_data[symbol], MARKET_RISK_PREMIUM, RISK_FREE_RATE)
                print(f"  WACC: {wacc:.2%}" if not pd.isna(wacc) else "  WACC: N/A")
                
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    # Run detailed analysis for NVIDIA
    main()
    
    # Uncomment below to run quick analysis for all companies
    # quick_multi_company_analysis()