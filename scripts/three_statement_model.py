

import pandas as pd
import numpy as np
import yfinance as yf
from clean import consolidate_financial_data, COMPANIES
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Configuration style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ThreeStatementModel:
    """
    A comprehensive 3-statement financial model for DCF valuation
    Integrates historical data with forward projections
    """
    
    def __init__(self, symbol: str, company_data: Dict, projection_years: int = 5):
        self.symbol = symbol
        self.company_data = company_data
        self.projection_years = projection_years
        self.historical_years = 5  # Use last 5 years for historical analysis
        
        # Initialize data structures
        self.assumptions = {}
        self.historical_data = {}
        self.projected_data = {}
        
        # Load and prepare data
        self._load_historical_data()
        self._set_default_assumptions()
        
    def _load_historical_data(self):
        """Load and organize historical financial data"""
        print(f"Loading historical data for {self.symbol}...")
        
        # Income Statement Data
        is_annual = self.company_data.get('IS_Annual', pd.DataFrame())
        if not is_annual.empty:
            # Get last 5 years of data
            historical_is = is_annual.tail(self.historical_years).copy()
            self.historical_data['income_statement'] = historical_is
            
            # Calculate key metrics
            self._calculate_historical_metrics(historical_is)
        
        # Balance Sheet Data
        bs_annual = self.company_data.get('BS_Annual', pd.DataFrame())
        if not bs_annual.empty:
            historical_bs = bs_annual.tail(self.historical_years).copy()
            self.historical_data['balance_sheet'] = historical_bs
        
        # Cash Flow Data
        cf_annual = self.company_data.get('CF_Annual', pd.DataFrame())
        if not cf_annual.empty:
            historical_cf = cf_annual.tail(self.historical_years).copy()
            self.historical_data['cash_flow'] = historical_cf
            
    def _calculate_historical_metrics(self, income_statement: pd.DataFrame):
        """Calculate historical growth rates and margins"""
        if len(income_statement) < 2:
            return
            
        # Revenue growth rates
        revenues = income_statement.get('Revenue', pd.Series())
        if len(revenues) > 1:
            revenue_growth = revenues.pct_change().dropna()
            self.historical_data['revenue_growth'] = revenue_growth
            print(f"  Historical revenue growth: {revenue_growth.mean():.2%} avg")
        
        # Margin analysis
        if 'Gross Profit' in income_statement.columns and 'Revenue' in income_statement.columns:
            gross_margins = income_statement['Gross Profit'] / income_statement['Revenue']
            self.historical_data['gross_margins'] = gross_margins
            print(f"  Gross margins: {gross_margins.mean():.2%} avg")
            
        if 'Operating Income' in income_statement.columns and 'Revenue' in income_statement.columns:
            operating_margins = income_statement['Operating Income'] / income_statement['Revenue']
            self.historical_data['operating_margins'] = operating_margins
            print(f"  Operating margins: {operating_margins.mean():.2%} avg")
            
        if 'Net Income' in income_statement.columns and 'Revenue' in income_statement.columns:
            net_margins = income_statement['Net Income'] / income_statement['Revenue']
            self.historical_data['net_margins'] = net_margins
            print(f"  Net margins: {net_margins.mean():.2%} avg")
    
    def _set_default_assumptions(self):
        """Set reasonable default assumptions for projections"""
        
        # Revenue Growth Assumptions
        historical_growth = self.historical_data.get('revenue_growth', pd.Series([0.05]))  # Default 5%
        avg_historical_growth = historical_growth.mean() if not historical_growth.empty else 0.05
        
        # Start with historical average, then gradually decline to terminal growth
        initial_growth = avg_historical_growth
        terminal_growth = 0.03  # Long-term GDP-like growth
        
        # Create declining growth curve
        growth_rates = []
        for year in range(self.projection_years):
            # Linear decline from historical to terminal growth
            decline_factor = year / (self.projection_years - 1) if self.projection_years > 1 else 1
            year_growth = initial_growth * (1 - decline_factor) + terminal_growth * decline_factor
            growth_rates.append(max(0.01, year_growth))  # Minimum 1% growth
            
        self.assumptions['revenue_growth'] = growth_rates
        
        # Margin Assumptions (maintain historical averages)
        gross_margins = self.historical_data.get('gross_margins', pd.Series([0.4]))  # Default 40%
        operating_margins = self.historical_data.get('operating_margins', pd.Series([0.15]))  # Default 15%
        net_margins = self.historical_data.get('net_margins', pd.Series([0.1]))  # Default 10%
        
        self.assumptions['gross_margin'] = gross_margins.mean() if not gross_margins.empty else 0.4
        self.assumptions['operating_margin'] = operating_margins.mean() if not operating_margins.empty else 0.15
        self.assumptions['net_margin'] = net_margins.mean() if not net_margins.empty else 0.10
        
        # Working Capital Assumptions (as % of revenue)
        self.assumptions['working_capital_pct'] = 0.15  # 15% of revenue
        
        # Capital Expenditure Assumptions (as % of revenue)
        self.assumptions['capex_pct'] = 0.05  # 5% of revenue
        
        # Depreciation & Amortization (as % of PPE)
        self.assumptions['depreciation_pct'] = 0.10  # 10% of net PPE
        
        # Tax Rate
        self.assumptions['tax_rate'] = 0.21  # Corporate tax rate
        
        print(f"Default assumptions set for {self.symbol}")
        
    def set_custom_assumptions(self, assumptions_dict: Dict):
        """Allow custom assumption overrides"""
        self.assumptions.update(assumptions_dict)
        print("Custom assumptions applied")
        
    def project_income_statement(self):
        """Project future income statements"""
        print(f"Projecting income statement for {self.symbol}...")
        
        historical_is = self.historical_data.get('income_statement', pd.DataFrame())
        if historical_is.empty:
            print("No historical income statement data available")
            return
            
        # Get latest historical data
        latest_year = historical_is.index[-1]
        latest_revenue = historical_is.loc[latest_year, 'Revenue']
        
        # Initialize projection dataframe
        projection_dates = pd.date_range(
            start=latest_year + pd.DateOffset(years=1),
            periods=self.projection_years,
            freq='Y'
        )
        
        projected_is = pd.DataFrame(index=projection_dates)
        
        # Project Revenue
        revenues = [latest_revenue]
        for i, growth_rate in enumerate(self.assumptions['revenue_growth']):
            next_revenue = revenues[-1] * (1 + growth_rate)
            revenues.append(next_revenue)
        
        projected_is['Revenue'] = revenues[1:]  # Exclude latest historical
        
        # Project Margins
        projected_is['Gross Profit'] = projected_is['Revenue'] * self.assumptions['gross_margin']
        projected_is['Operating Income'] = projected_is['Revenue'] * self.assumptions['operating_margin']
        projected_is['Net Income'] = projected_is['Revenue'] * self.assumptions['net_margin']
        
        # Calculate derived metrics
        projected_is['Cost of Revenue'] = projected_is['Revenue'] - projected_is['Gross Profit']
        projected_is['Operating Expenses'] = projected_is['Gross Profit'] - projected_is['Operating Income']
        
        # Taxes and Net Income 
        projected_is['Income Tax Expense'] = projected_is['Operating Income'] * self.assumptions['tax_rate']
        projected_is['Net Income'] = projected_is['Operating Income'] - projected_is['Income Tax Expense']
        
        self.projected_data['income_statement'] = projected_is
        return projected_is
        
    def project_balance_sheet(self):
        """Project future balance sheets"""
        print(f"Projecting balance sheet for {self.symbol}...")
        
        historical_bs = self.historical_data.get('balance_sheet', pd.DataFrame())
        if historical_bs.empty:
            print("No historical balance sheet data available")
            return
            
        # Get latest historical data
        latest_year = historical_bs.index[-1]
        
        # Initialize projection dataframe
        projection_dates = pd.date_range(
            start=latest_year + pd.DateOffset(years=1),
            periods=self.projection_years,
            freq='Y'
        )
        
        projected_bs = pd.DataFrame(index=projection_dates)
        
        # Get projected revenue
        projected_revenue = self.projected_data['income_statement']['Revenue']
        
        # Project Working Capital
        projected_bs['Working Capital'] = projected_revenue * self.assumptions['working_capital_pct']
        
        # Project Property, Plant & Equipment (PPE)
        # Assume PPE grows with capex and depreciation
        if 'Property Plant & Equipment Net' in historical_bs.columns:
            latest_ppe = historical_bs.loc[latest_year, 'Property Plant & Equipment Net']
            projected_capex = projected_revenue * self.assumptions['capex_pct']
            projected_depreciation = latest_ppe * self.assumptions['depreciation_pct']
            
            ppe_values = [latest_ppe]
            for i in range(self.projection_years):
                new_ppe = ppe_values[-1] + projected_capex.iloc[i] - projected_depreciation
                ppe_values.append(new_ppe)
            
            projected_bs['Property Plant & Equipment Net'] = ppe_values[1:]
        
        # Project Total Assets
        projected_bs['Total Assets'] = projected_bs['Working Capital'] + projected_bs.get('Property Plant & Equipment Net', 0)
        
        # Project Shareholders' Equity (grows with retained earnings)
        if 'Total Shareholder Equity' in historical_bs.columns:
            latest_equity = historical_bs.loc[latest_year, 'Total Shareholder Equity']
            projected_ni = self.projected_data['income_statement']['Net Income']
            
            # Assume no dividends for simplicity(all earnings retained), dividend given out is also negligible for NVDA
            equity_values = [latest_equity]
            for i in range(self.projection_years):
                new_equity = equity_values[-1] + projected_ni.iloc[i]
                equity_values.append(new_equity)
            
            projected_bs['Total Shareholder Equity'] = equity_values[1:]
        
        # Balance sheet equation: A = L + S/E
        projected_bs['Total Liabilities'] = projected_bs['Total Assets'] - projected_bs['Total Shareholder Equity']
        
        self.projected_data['balance_sheet'] = projected_bs
        return projected_bs
        
    def project_cash_flow_statement(self):
        """Project future cash flow statements"""
        print(f"Projecting cash flow statement for {self.symbol}...")
        
        if 'income_statement' not in self.projected_data or 'balance_sheet' not in self.projected_data:
            print("Need income statement and balance sheet projections first")
            return
            
        projection_dates = self.projected_data['income_statement'].index
        
        projected_cf = pd.DataFrame(index=projection_dates)
        
        # Get projected data
        projected_ni = self.projected_data['income_statement']['Net Income']
        projected_revenue = self.projected_data['income_statement']['Revenue']
        
        # Operating Cash Flow
        # OCF = Net Income + Depreciation - Change in Working Capital
        projected_depreciation = projected_revenue * self.assumptions['capex_pct'] * 0.5  # Rough estimate
        projected_wc_change = projected_revenue.diff().fillna(0) * self.assumptions['working_capital_pct']
        
        projected_cf['Operating Cash Flow'] = projected_ni + projected_depreciation - projected_wc_change
        
        # Investing Cash Flow (primarily CapEx)
        projected_cf['Capital Expenditures'] = -projected_revenue * self.assumptions['capex_pct']  # Negative for outflow
        projected_cf['Investing Cash Flow'] = projected_cf['Capital Expenditures']
        
        # Financing Cash Flow (simplified - assume no new financing)
        projected_cf['Financing Cash Flow'] = 0
        
        # Net Change in Cash
        projected_cf['Net Change in Cash'] = (
            projected_cf['Operating Cash Flow'] + 
            projected_cf['Investing Cash Flow'] + 
            projected_cf['Financing Cash Flow']
        )
        
        # Free Cash Flow (key for DCF)
        projected_cf['Free Cash Flow'] = projected_cf['Operating Cash Flow'] + projected_cf['Capital Expenditures']
        
        self.projected_data['cash_flow'] = projected_cf
        return projected_cf
        
    def run_complete_model(self, custom_assumptions: Dict = None):
        """Run the complete 3-statement model"""
        print(f"\n{'='*60}")
        print(f"RUNNING 3-STATEMENT MODEL FOR {self.symbol}")
        print(f"{'='*60}")
        
        if custom_assumptions:
            self.set_custom_assumptions(custom_assumptions)
        
        # Run projections in order (IS -> BS -> CF)
        self.project_income_statement()
        self.project_balance_sheet()
        self.project_cash_flow_statement()
        
        print(f"\nModel completed for {self.symbol}")
        return self.projected_data
        
    def get_valuation_data(self):
        """Extract key data for DCF valuation"""
        if 'cash_flow' not in self.projected_data:
            return None
            
        cash_flows = self.projected_data['cash_flow']['Free Cash Flow']
        terminal_growth = self.assumptions['revenue_growth'][-1]  # Use final year growth as terminal
        
        return {
            'free_cash_flows': cash_flows,
            'terminal_growth_rate': terminal_growth,
            'projection_years': self.projection_years
        }
        
    def display_summary(self):
        """Display a summary of the projections"""
        if not self.projected_data:
            print("No projections available. Run model first.")
            return
            
        print(f"\n{'='*60}")
        print(f"PROJECTION SUMMARY - {self.symbol}")
        print(f"{'='*60}")
        
        # Revenue Projections
        if 'income_statement' in self.projected_data:
            revenues = self.projected_data['income_statement']['Revenue']
            print(f"\nRevenue Projections:")
            for i, (date, revenue) in enumerate(revenues.items()):
                growth_pct = (revenue / revenues.iloc[i-1] - 1) * 100 if i > 0 else 0
                print(f"  Year {i+1}: ${revenue/1e9:.2f}B ({growth_pct:.1f}%)")
        
        # Free Cash Flow Projections
        if 'cash_flow' in self.projected_data:
            fcfs = self.projected_data['cash_flow']['Free Cash Flow']
            print(f"\nFree Cash Flow Projections:")
            for i, (date, fcf) in enumerate(fcfs.items()):
                print(f"  Year {i+1}: ${fcf/1e9:.2f}B")
                
        # Key Assumptions
        print(f"\nKey Assumptions:")
        print(f"  Terminal Growth Rate: {self.assumptions['revenue_growth'][-1]:.2%}")
        print(f"  Gross Margin: {self.assumptions['gross_margin']:.2%}")
        print(f"  Operating Margin: {self.assumptions['operating_margin']:.2%}")
        print(f"  Net Margin: {self.assumptions['net_margin']:.2%}")
        
    def plot_projections(self):
        """Create visualization of projections"""
        if not self.projected_data:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Financial Projections - {self.symbol}', fontsize=16, fontweight='bold')
        
        # Revenue Projection
        if 'income_statement' in self.projected_data:
            revenues = self.projected_data['income_statement']['Revenue'] / 1e9  # Convert to billions
            axes[0,0].plot(revenues.index.year, revenues.values, marker='o', linewidth=2)
            axes[0,0].set_title('Revenue Projection')
            axes[0,0].set_ylabel('Revenue ($B)')
            axes[0,0].grid(True, alpha=0.3)
        
        # Free Cash Flow Projection
        if 'cash_flow' in self.projected_data:
            fcfs = self.projected_data['cash_flow']['Free Cash Flow'] / 1e9
            axes[0,1].plot(fcfs.index.year, fcfs.values, marker='s', color='green', linewidth=2)
            axes[0,1].set_title('Free Cash Flow Projection')
            axes[0,1].set_ylabel('FCF ($B)')
            axes[0,1].grid(True, alpha=0.3)
        
        # Margin Projections
        if 'income_statement' in self.projected_data:
            is_data = self.projected_data['income_statement']
            gross_margins = (is_data['Gross Profit'] / is_data['Revenue']) * 100
            operating_margins = (is_data['Operating Income'] / is_data['Revenue']) * 100
            net_margins = (is_data['Net Income'] / is_data['Revenue']) * 100
            
            axes[1,0].plot(gross_margins.index.year, gross_margins.values, marker='^', label='Gross Margin')
            axes[1,0].plot(operating_margins.index.year, operating_margins.values, marker='s', label='Operating Margin')
            axes[1,0].plot(net_margins.index.year, net_margins.values, marker='o', label='Net Margin')
            axes[1,0].set_title('Margin Projections')
            axes[1,0].set_ylabel('Margin (%)')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Balance Sheet Growth
        if 'balance_sheet' in self.projected_data:
            bs_data = self.projected_data['balance_sheet']
            assets = bs_data.get('Total Assets', pd.Series([0])) / 1e9
            equity = bs_data.get('Total Shareholder Equity', pd.Series([0])) / 1e9
            
            if not assets.empty:
                axes[1,1].plot(assets.index.year, assets.values, marker='o', label='Total Assets')
                axes[1,1].plot(equity.index.year, equity.values, marker='s', label='Shareholders Equity')
                axes[1,1].set_title('Balance Sheet Projection')
                axes[1,1].set_ylabel('Value ($B)')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'output/{self.symbol}_projections.png', dpi=300, bbox_inches='tight')
        plt.show()

def run_three_statement_analysis(symbols: List[str] = None):
    """Run 3-statement analysis for multiple companies"""
    if symbols is None:
        symbols = COMPANIES
    
    # Load cleaned data
    print("Loading financial data...")
    all_companies_data = consolidate_financial_data()
    
    models = {}
    
    for symbol in symbols:
        if symbol in all_companies_data:
            print(f"\n{'='*50}")
            print(f"ANALYZING: {symbol}")
            print(f"{'='*50}")
            
            try:
                # Create and run model
                model = ThreeStatementModel(symbol, all_companies_data[symbol])
                projections = model.run_complete_model()
                
                if projections:
                    models[symbol] = model
                    model.display_summary()
                    
                    # Save to Excel
                    save_projections_to_excel(model, symbol)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
    
    return models

def save_projections_to_excel(model: ThreeStatementModel, symbol: str):
    """Save projections to Excel file"""
    try:
        with pd.ExcelWriter(f'output/{symbol}_financial_projections.xlsx', engine='openpyxl') as writer:
            # Save each statement
            for statement_type, data in model.projected_data.items():
                data.to_excel(writer, sheet_name=statement_type.capitalize())
            
            # Save assumptions
            assumptions_df = pd.DataFrame.from_dict(model.assumptions, orient='index', columns=['Value'])
            assumptions_df.to_excel(writer, sheet_name='Assumptions')
            
        print(f"Projections saved to output/{symbol}_financial_projections.xlsx")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs('output', exist_ok=True)
    
    # Run analysis for all companies
    models = run_three_statement_analysis()
    
    print(f"\n{'='*60}")
    print("3-STATEMENT MODEL ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Models created for {len(models)} companies")
    
    # Generate visualizations for the first company as example
    if models:
        first_symbol = list(models.keys())[0]
        models[first_symbol].plot_projections()