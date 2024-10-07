# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Load the dataset
file_path = 'https://www.kaggle.com/datasets/gokulrajkmv/unemployment-in-india'
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Drop any missing values (optional based on data quality)
data = data.dropna()

# Plot the unemployment rate over time for different regions
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', hue='Region', data=data)
plt.title('Unemployment Rate in India by Region')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot labor participation rate for different areas (Urban vs Rural)
plt.figure(figsize=(10, 5))
sns.boxplot(x='Area', y='Estimated Labour Participation Rate (%)', data=data)
plt.title('Labour Participation Rate by Area (Urban vs Rural)')
plt.xlabel('Area')
plt.ylabel('Labour Participation Rate (%)')
plt.show()

# Calculate and display summary statistics of unemployment rate by region
unemployment_stats = data.groupby('Region')['Estimated Unemployment Rate (%)'].describe()
print(unemployment_stats)

# Optional: Save cleaned data
# data.to_csv('Cleaned_Unemployment_Data.csv', index=False)
