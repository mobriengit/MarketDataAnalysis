import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Setting a random seed for reproducibility
np.random.seed(0)

# Generate dates
dates = pd.date_range(start='2015-01-01', end='2020-12-31', freq='M')

# Simulating marketing spend
marketing_spend = np.random.normal(loc=20000, scale=5000, size=len(dates))

# Simulating ROI with varying impacts from external factors
economic_index = np.random.normal(loc=0, scale=1, size=len(dates))  # Economic stability index
seasonality = np.cos(2 * np.pi * dates.month / 12)  # Seasonal effect over the year
covid_impact = np.where(dates >= '2020-01', -np.random.normal(loc=5000, scale=1000, size=len(dates)), 0)  # Negative impact due to COVID-19

# Calculate ROI as influenced by marketing spend and external factors
roi = marketing_spend * (0.1 + 0.05 * economic_index + 0.1 * seasonality) + covid_impact

# Create DataFrame
data = pd.DataFrame({
    'Date': dates,
    'Marketing_Spend': marketing_spend,
    'Economic_Index': economic_index,
    'Seasonality': seasonality,
    'Covid_Impact': covid_impact,
    'ROI': roi
})
data.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(data['Marketing_Spend'], label='Marketing Spend')
plt.plot(data['ROI'], label='ROI', color='red')
plt.title('Marketing Spend vs. ROI')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data['Economic_Index'], label='Economic Index')
plt.plot(data['Seasonality'], label='Seasonal Effect', color='green')
plt.plot(data['Covid_Impact'], label='COVID-19 Impact', color='purple')
plt.title('External Factors Impact')
plt.legend()
plt.tight_layout()
plt.show()

from statsmodels.tsa.vector_ar.var_model import VAR

# Fitting VAR model
model = VAR(data[['Marketing_Spend', 'Economic_Index', 'Seasonality', 'Covid_Impact', 'ROI']])
results = model.fit(maxlags=15, ic='aic')
print(results.summary())

# Forecasting
lag_order = results.k_ar
forecasted_values = results.forecast(data[['Marketing_Spend', 'Economic_Index', 'Seasonality', 'Covid_Impact', 'ROI']].values[-lag_order:], steps=12)
forecast_index = pd.date_range(start=data.index[-1], periods=13, freq='M')[1:]

forecast_df = pd.DataFrame(forecasted_values, index=forecast_index, columns=['Marketing_Spend', 'Economic_Index', 'Seasonality', 'Covid_Impact', 'ROI'])
plt.figure(figsize=(10, 5))
plt.plot(data['ROI'], label='Historical ROI')
plt.plot(forecast_df['ROI'], label='Forecasted ROI', color='red')
plt.title('ROI Forecast')
plt.legend()
plt.show()

