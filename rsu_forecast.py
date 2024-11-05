import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import os


# Read CSV of census
census_df = pd.read_csv('/home/josh/monte_carlo/original_census.csv')

# Convert tenure from days to years
census_df['Tenure_Years'] = census_df['Tenure'] / 365

# Prompt user for CAGR of employee growth
cagr = float(input("Please enter the Compound Annual Growth Rate (CAGR) of employee growth (in percentage): ")) / 100

# Prompt user for CAGR of salary growth
sal_cagr = float(input("Please enter the Compound Annual Growth Rate (CAGR) of salary growth (in percentage): ")) / 100

# Read turnover improvement data
turnover_improvement_df = pd.read_csv('/home/josh/monte_carlo/improvement_inputs.csv')
turnover_improvement = turnover_improvement_df['Turnover Improvement'].values.flatten()
productivity_gains = turnover_improvement_df['Productivity Gains'].values.flatten()
reduced_safety_incidents = turnover_improvement_df['Reduced Safety Incidents'].values.flatten()
decrease_in_claims = turnover_improvement_df['Decrease in Claims'].values.flatten()
cpi = turnover_improvement_df['CPI'].values.flatten()

# Calculate the number of employees over the years based on CAGR
initial_employees = len(census_df)
years = np.arange(1, 11) # Example: 10 years
employee_growth = [initial_employees * (1 + cagr) ** year for year in years]

# Print the estimated number of employees over the years
print("Estimated number of employees over the years based on CAGR:")
for year, employees in zip(years, employee_growth):
    print(f"Year {year}: {employees:.0f} employees")

# Estimate the tenure and salaries of every employee each year and create a data frame for each year
data_frames = {}
for year in range(1, 11):
    num_employees = int(employee_growth[year - 1])
    # Sample rows directly from the census DataFrame
    sampled_employees = census_df.sample(n=num_employees, replace=True)
    # Apply the salary growth rate for the current year
    sampled_employees['Adjusted_Salary'] = sampled_employees['Salary'] * (1 + sal_cagr) ** (year - 1)
    # Adjust tenure based on turnover improvement
    improvement_factor = 1 + turnover_improvement[year - 1]
    sampled_employees['Adjusted_Tenure_Years'] = sampled_employees['Tenure_Years'] * improvement_factor
    df = pd.DataFrame({
        'Index': range(1, num_employees + 1),
        'Tenure_Years': sampled_employees['Adjusted_Tenure_Years'].values,
        'Salary': sampled_employees['Adjusted_Salary'].values
    })
    data_frames[f'Year_{year}'] = df
    # Calculate and print the mean and median of the salaries
    mean_salary = df['Salary'].mean()
    median_salary = df['Salary'].median()
    print(f"Year {year} - Mean Salary: ${mean_salary:.2f}, Median Salary: ${median_salary:.2f}")

# Current timestamp
current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create folders for current run
folder_path = os.path.join("/home/josh/monte_carlo/simulation_results", current_timestamp)
os.makedirs(folder_path)

# Create subfolder paths
workforce_results_path = os.path.join(f'{folder_path}/workforce_results')
input_summary_path = os.path.join(f'{folder_path}/input_summary')

# Create subfolders
os.makedirs(workforce_results_path)
os.makedirs(input_summary_path)

# Save the data frames for each year as CSV files
for year, df in data_frames.items():
    file_name = f"{workforce_results_path}/workforce_simulation_{year}.csv"
    df.to_csv(file_name, index=False)
    print(f"Data Frame for {year} saved as {file_name}")

# Calculate savings for each category
savings = {
    'Year': [],
    'Turnover Improvement': [],
    'Productivity Gains': [],
    'Reduced Safety Incidents': [],
    'Decrease in Claims': []
}

for year in range(1, 11):
    num_employees = int(employee_growth[year - 1])
    # Turnover Improvement
    turnover_savings = turnover_improvement[year - 1] * num_employees * 7000 * (1 + cpi[year - 1])
    # Productivity Gains
    direct_labor_benefits_per_hr = turnover_improvement_df['Direct Labor and Benefits per hr'][year - 1]
    productivity_savings = 0
    if direct_labor_benefits_per_hr != 0:
        hrs_with_improvement = (1-productivity_gains[year - 1]) * 40 * num_employees # 40 * number of employees will not give us the number of hours. Investigate how the hours from the BBP model were made.
        productivity_savings = direct_labor_benefits_per_hr * (hrs_with_improvement - (40 * num_employees)) * -1
    # Reduced Safety Incidents
    safety_savings = 21e6 * reduced_safety_incidents[year - 1]
    # Decrease in Claims
    claims_savings = 9e6 * (1 + cpi[year - 1]) * decrease_in_claims[year - 1]
    
    savings['Year'].append(f'Year {year}')
    savings['Turnover Improvement'].append(turnover_savings)
    savings['Productivity Gains'].append(productivity_savings)
    savings['Reduced Safety Incidents'].append(safety_savings)
    savings['Decrease in Claims'].append(claims_savings)

# Add total row
savings['Year'].append('Total')
savings['Turnover Improvement'].append(sum(savings['Turnover Improvement']))
savings['Productivity Gains'].append(sum(savings['Productivity Gains']))
savings['Reduced Safety Incidents'].append(sum(savings['Reduced Safety Incidents']))
savings['Decrease in Claims'].append(sum(savings['Decrease in Claims']))

# Create DataFrame and save as CSV
savings_df = pd.DataFrame(savings)
savings_file_path = os.path.join(folder_path, 'savings_summary.csv')
savings_df.to_csv(savings_file_path, index=False)
print(f"Savings summary saved as {savings_file_path}")

# Pull samples for comparison against original
year_5_df = data_frames['Year_5']
year_10_df = data_frames['Year_10']

# Compare the tenure distribution from the original census, year 5 simulation data, and year 10 simulation data
plt.hist(year_10_df['Tenure_Years'], bins=30, alpha=1, label='Year 10 Simulation', edgecolor='black', color='white')
plt.hist(year_5_df['Tenure_Years'], bins=30, alpha=1, label='Year 5 Simulation', edgecolor='black', color='blue')
plt.hist(census_df['Tenure_Years'], bins=30, alpha=1, label='Original Census', edgecolor='black', color='black')
plt.title('Comparison of Tenure Distribution')
plt.xlabel('Tenure (Years)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Compare the salary distribution from the original census, year 5 simulation data, and year 10 simulation data
plt.hist(year_10_df['Salary'], bins=30, alpha=0.5, label='Year 10 Simulation', edgecolor='black', color='white')
plt.hist(year_5_df['Salary'], bins=30, alpha=0.5, label='Year 5 Simulation', edgecolor='black', color='blue')
plt.hist(census_df['Salary'], bins=30, alpha=0.5, label='Original Census', edgecolor='black', color='black')
plt.title('Comparison of Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Calculate average salary by tenure for original census data
original_avg_salary_by_tenure = census_df.groupby('Tenure_Years')['Salary'].mean()
# Calculate average salary by tenure for year 5 and year 10 simulation data
year_5_avg_salary_by_tenure = year_5_df.groupby('Tenure_Years')['Salary'].mean()
year_10_avg_salary_by_tenure = year_10_df.groupby('Tenure_Years')['Salary'].mean()

# Plot the average salary by tenure
plt.figure(figsize=(10, 6))
plt.plot(original_avg_salary_by_tenure, label='Original Census', marker='o')
plt.plot(year_5_avg_salary_by_tenure, label='Year 5 Simulation', marker='o')
plt.plot(year_10_avg_salary_by_tenure, label='Year 10 Simulation', marker='o')
plt.title('Average Salary by Tenure')
plt.xlabel('Tenure (Years)')
plt.ylabel('Average Salary')
plt.legend()
plt.grid(True)
plt.show()

# Parameters for simulation
simulation_period = 10  # in years

LINE_starting_price = float(input("What is the price at the start of the period? "))

# Read CSV of peer company stock data
peer_df = pd.read_csv('/home/josh/monte_carlo/peer_tickers.csv')

# Extract tickers from peer_df
tickers = peer_df['Ticker'].tolist()

# Fetch stock prices for the tickers
peer_stock_data = yf.download(tickers, period='10y')['Adj Close']

# Calculate daily log returns
log_returns = np.log(peer_stock_data / peer_stock_data.shift(1)).dropna()

# Calculate the mean and standard deviation of the log returns
mu = log_returns.mean().mean()  # average return of peer companies over 10 years
sigma = log_returns.std().mean()  # average volatility of peer companies over 10 years

# Annualize the mean and standard deviation
annualized_mu = mu * 252  # Annualized mean return
annualized_sigma = sigma * np.sqrt(252)  # Annualized volatility

# Calculate daily mean and volatility
daily_mu = annualized_mu / 252
daily_sigma = annualized_sigma / np.sqrt(252)
num_simulations = 10000  # Number of simulations
end_prices = []
days_per_year = 252  # Number of trading days in a year

for sim in range(num_simulations):
    price = LINE_starting_price
    period_prices = [price]
    for year in range(simulation_period):
        daily_returns = np.random.normal(daily_mu, daily_sigma, days_per_year)
        price_series = price * np.exp(np.cumsum(daily_returns))  # Brownian motion model
        price = price_series[-1]
        period_prices.append(price)
    end_prices.append(period_prices)

average_end_price = np.mean([prices[-1] for prices in end_prices])
print(f'Average annualized return of peer companies: {annualized_mu}')
print(f'Average annualized volatility of peer companies: {annualized_sigma}')
print(f'Average Price at End of Simulation over {num_simulations} simulations: {average_end_price}')

# Store the median, average, upper quartile, and lower quartile price at the end of each year for all simulations
median_prices = []
average_prices = []
fifth_percentiles = []
upper_quartiles = []

for year in range(simulation_period + 1):
    year_prices = [prices[year] for prices in end_prices]
    median_prices.append(np.median(year_prices))
    average_prices.append(np.mean(year_prices))
    fifth_percentiles.append(np.percentile(year_prices, 5))
    upper_quartiles.append(np.percentile(year_prices, 75))

# Print the lists
print(f'Median Prices: {median_prices}')
print(f'Average Prices: {average_prices}')
print(f'Fifth Percentiles: {fifth_percentiles}')
print(f'Upper Quartiles: {upper_quartiles}')

# Determine size of grants for multiple iterations
grant_sizes = input('Enter the grant sizes as percentages of base, separated by commas: ').split(',')
grant_sizes = [float(size.strip()) / 100 for size in grant_sizes]

# Determine vesting length
vesting_length = int(input('What is the vesting length (in years)? '))

# Base directory for saving folders
base_dir = folder_path

# Function to process DataFrame and save as CSV
def process_and_save_df(prices, folder_name, grant_size):
    # Create folder if it doesn't exist
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Dictionary to store DataFrames
    dataframes = {}
    # Loop through numbers 1 to 10 to read each CSV file
    for year in range(1, 11):
        # Read workforce data from each year
        file_path = f'{workforce_results_path}/workforce_simulation_Year_{year}.csv'
        df = pd.read_csv(file_path)
        
        # Loop through each grant year to calculate tranches
        for grant_year in range(1, year + 1):
            tranche_number = year - grant_year + 1
            if tranche_number <= vesting_length:
                tranche_column = f"Grant {grant_year} Tranche {tranche_number} Vested"
                granted_column = f"Grant {grant_year} Tranche {tranche_number} Granted"
                price_column = f"Grant {grant_year} Price"
                grant_price = prices[grant_year - 1]  # Use the appropriate price for the grant year
                
                # Calculate shares vested
                df[tranche_column] = np.floor(((df['Salary'] * grant_size) / vesting_length) / grant_price)
                df[price_column] = grant_price
                
                # Calculate shares granted (ignoring tenure)
                df[granted_column] = np.floor(((df['Salary'] * grant_size) / vesting_length) / grant_price)
                
                # Update Tranche Size based on Tenure_Years
                df[tranche_column] = df.apply(
                    lambda row: row[tranche_column] if row["Tenure_Years"] >= tranche_number else 0, axis=1
                )
        
        # Adjust tranches based on tenure
        for index, row in df.iterrows():
            tenure_years = row["Tenure_Years"]
            for grant_year in range(1, year + 1):
                for tranche in range(vesting_length, 0, -1):
                    tranche_column = f"Grant {grant_year} Tranche {tranche} Vested"
                    if tranche_column in df.columns:
                        if tenure_years >= tranche:
                            df.at[index, tranche_column] = row[tranche_column]
                        else:
                            df.at[index, tranche_column] = 0
        
        # Add Vest Price, Value at Vest, and Cost columns
        vest_price = prices[year]  # Use the price for the current year
        df['Vest Price'] = vest_price
        tranche_columns = [col for col in df.columns if 'Tranche' in col and 'Vested' in col]
        df['Value at Vest'] = df[tranche_columns].sum(axis=1) * vest_price
        
        # Shares vesting that year
        df['Total Shares Vesting'] = df.apply(lambda row: sum(row[f"Grant {grant_year} Tranche {year - grant_year + 1} Vested"]
                                                              for grant_year in range(1, year + 1) if f"Grant {grant_year} Tranche {year - grant_year + 1} Vested" in df.columns), axis=1)
        
        # Calculate the cost as the sum of each tranche size times their corresponding grant price
        df['Cost'] = df.apply(lambda row: sum(row[f"Grant {grant_year} Tranche {year - grant_year + 1} Vested"] * row[f"Grant {grant_year} Price"]
                                              for grant_year in range(1, year + 1) if f"Grant {grant_year} Tranche {year - grant_year + 1} Vested" in df.columns), axis=1)
        
        # Calculate dividends
        initial_dividend = 2.00
        df['Annualized Dividend'] = df.apply(lambda row: sum(row[f"Grant {grant_year} Tranche {year - grant_year + 1} Vested"] * (initial_dividend * (1.05 ** (year - 1)))
                                                              for grant_year in range(1, year + 1) if f"Grant {grant_year} Tranche {year - grant_year + 1} Vested" in df.columns), axis=1)
        # Create dividend per share column
        df['Annualized Dividend Price Per Share'] = initial_dividend * (1.05 ** (year - 1))
        
        # Store the DataFrame in the dictionary
        dataframes[f'Year_{year}'] = df
    
    # Save each DataFrame as a CSV
    for year, df in dataframes.items():
        csv_path = os.path.join(folder_path, f'{year}.csv')
        df.to_csv(csv_path, index=False)
    
    # Create summary file
    summary_data = {
        'Year': [],
        'Shares Granted': [],
        'Share Utilization': [],
        'RSU Expense': [],
        'DER Expense': [],
        'Total Expense': [],
        'Total Employee Benefit': [],
        'Number of Employees': []
    }
    
    for year, df in dataframes.items():
        summary_data['Year'].append(year)
        summary_data['Shares Granted'].append(df[[col for col in df.columns if 'Granted' in col]].sum().sum())
        summary_data['Share Utilization'].append(df[[col for col in df.columns if 'Vested' in col]].sum().sum())
        summary_data['RSU Expense'].append(df['Cost'].sum())
        summary_data['DER Expense'].append(df['Annualized Dividend'].sum())
        summary_data['Total Expense'].append(df['Annualized Dividend'].sum() + df['Cost'].sum())
        summary_data['Total Employee Benefit'].append(df['Value at Vest'].sum() + df['Annualized Dividend'].sum())
        summary_data['Number of Employees'].append(len(df))
    
    # Add total row
    summary_data['Year'].append('Total')
    summary_data['Shares Granted'].append(sum(summary_data['Shares Granted']))
    summary_data['Share Utilization'].append(sum(summary_data['Share Utilization']))
    summary_data['RSU Expense'].append(sum(summary_data['RSU Expense']))
    summary_data['DER Expense'].append(sum(summary_data['DER Expense']))
    summary_data['Total Expense'].append(sum(summary_data['Total Expense']))
    summary_data['Total Employee Benefit'].append(sum(summary_data['Total Employee Benefit']))
    summary_data['Number of Employees'].append(sum(summary_data['Number of Employees']))
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(folder_path, 'summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)

# Process and save DataFrames for each price list and grant size
for grant_size in grant_sizes:
    process_and_save_df(median_prices, f'median_{grant_size}', grant_size)
    process_and_save_df(average_prices, f'average_{grant_size}', grant_size)
    process_and_save_df(fifth_percentiles, f'fifth_percentile_{grant_size}', grant_size)
    process_and_save_df(upper_quartiles, f'upper_quartile_{grant_size}', grant_size)

# Create file summarizing inputs
inputs = []
inputs.append(f'Employee growth rate: {cagr}')
inputs.append(f'Salary growth rate: {sal_cagr}')
inputs.append(f'Price at start of simulation: {LINE_starting_price}')
inputs.append(f'Grant sizes as % of base: {grant_sizes}')
inputs.append(f'Vesting length (years): {vesting_length}')
inputs.append(f'Average annualized return of peer companies: {annualized_mu}')
inputs.append(f'Average annualized volatility of peer companies: {annualized_sigma}')

# List to DataFrame
inputs_df = pd.DataFrame(inputs, columns=['Parameters'])
inputs_df.to_csv(f'{input_summary_path}/inputs.csv', index=False)