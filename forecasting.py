## Forecasting for comanies

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('esg_sentiment_score.csv')
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

forecast_horizon = 2
results = []

# Traverse each company to make predictions
for company in df['Company'].unique():
    company_data = df[df['Company'] == company].sort_values('Year')
    ts = company_data.set_index('Year')['compound']

    try:
        model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=forecast_horizon)

        forecast_years = pd.date_range(start=ts.index[-1] + pd.DateOffset(years=1),
                                       periods=forecast_horizon, freq='Y')
        result_df = pd.DataFrame({
            'Company': company,
            'Year': forecast_years.year,
            'Predicted_compound': forecast.values
        })
        results.append(result_df)

    except Exception as e:
        print(f"Model failed for {company}: {e}")

# Merge results
final_forecast = pd.concat(results, ignore_index=True)


# Add ESG risk level classification
def classify_esg(score):
    if score > 0.5:
        return 'Low risk'
    elif score < -0.5:
        return 'High risk'
    else:
        return 'Medium risk'


final_forecast['Risk Level'] = final_forecast['Predicted_compound'].apply(classify_esg)
# Visualize ESG risk level distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=final_forecast, x='Year', hue='Risk Level')
plt.title('ESG risk level distribution in 2025-2026')
plt.xlabel('year')
plt.ylabel('Number of companies')
plt.legend(title='ESG Risk Level')
plt.tight_layout()
plt.show()

final_forecast.to_csv('esg_forecast_with_risk_levels.csv', index=False)


## Forecasting for industries

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from plotly.offline import plot


df = pd.read_csv("esg_sentiment_score.csv")

industry_map = {
    'ASML': 'Semiconductors', 'Airbus': 'Aerospace', 'Allianz': 'Insurance', 'AB InBev': 'Food & Beverage',
    'AstraZeneca': 'Pharmaceuticals', 'BASF': 'Chemicals', 'BMW': 'Automotive', 'BNP Paribas': 'Banking',
    'Bayer': 'Pharmaceuticals', 'CRH': 'Building Materials', 'Daimler': 'Automotive', 'Danone': 'Food & Beverage',
    'Deutsche Bank': 'Banking', 'Deutsche Börse': 'Financial Services', 'Deutsche Post DHL': 'Logistics',
    'Deutsche Telekom': 'Telecommunications', 'Enel': 'Energy', 'Engie': 'Energy', 'Eurazeo': 'Investment',
    'Eutelsat': 'Telecommunications', 'Ferrari': 'Automotive', 'GSK': 'Pharmaceuticals', 'Hermès': 'Luxury Goods',
    'ING Group': 'Banking', 'Iberdrola': 'Energy', 'Inditex': 'Apparel Retail', 'Intesa Sanpaolo': 'Banking',
    'Kering': 'Luxury Goods', 'L’Oréal': 'Cosmetics', 'LVMH': 'Luxury Goods', 'Luxottica': 'Luxury Goods',
    'Merck KGaA': 'Pharmaceuticals', 'Munich Re': 'Insurance', 'Nestlé': 'Food & Beverage',
    'Novartis': 'Pharmaceuticals', 'Pernod Ricard': 'Food & Beverage', 'Prosus': 'Internet',
    'Publicis Groupe': 'Advertising', 'Repsol': 'Energy', 'Roche': 'Pharmaceuticals', 'SAP': 'Software',
    'STMicroelectronics': 'Semiconductors', 'Safran': 'Aerospace', 'Sanofi': 'Pharmaceuticals',
    'Schneider Electric': 'Industrial Manufacturing', 'Siemens': 'Industrial Manufacturing',
    'Société Générale': 'Banking', 'TotalEnergies': 'Energy', 'Unilever': 'Consumer Goods', 'Volkswagen': 'Automotive'
}

df['Industry'] = df['Company'].map(industry_map)
df = df.dropna(subset=['Industry'])
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

df_filtered = df[(df['Year'].dt.year >= 2020) & (df['Year'].dt.year <= 2024)]
industry_ts = df_filtered.groupby(['Industry', 'Year'])['compound'].mean().reset_index()

forecast_horizon = 2
industry_forecasts = []

for industry in industry_ts['Industry'].unique():
    sub_df = industry_ts[industry_ts['Industry'] == industry].sort_values('Year')
    ts = sub_df.set_index('Year')['compound']

    try:
        model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=forecast_horizon)

        forecast_years = pd.date_range(start=ts.index[-1] + pd.DateOffset(years=1),
                                       periods=forecast_horizon, freq='Y')
        clipped_forecast = forecast.clip(lower=-1, upper=1)
        forecast_df = pd.DataFrame({
            'Industry': industry,
            'Year': forecast_years,
            'compound': forecast.values
        })
        industry_forecasts.append(forecast_df)

    except Exception as e:
        print(f"Forecast failed for {industry}: {e}")

forecast_df = pd.concat(industry_forecasts, ignore_index=True)
full_df = pd.concat([industry_ts, forecast_df], ignore_index=True)
full_df['Year'] = full_df['Year'].dt.year  
full_df['compound'] = full_df['compound']

'''

fig = px.line(full_df, x="Year", y="compound", color="Industry", markers=True,
              title="2020–2026 ESG Forecast by Industry (Interactive)")

# HTML Dashboard (old version)
plot(fig, filename="esg_forecast_plot.html", auto_open=True)
'''
'''
# Plot industries
plt.figure(figsize=(14, 7))
sns.lineplot(data=full_df, x='Year', y='compound', hue='Industry', marker='o')
plt.title("2020–2026 ESG Sentiment Forecast per Industry (Compound × 50)")
plt.ylabel("ESG Score × 50")
plt.xlabel("Year")
plt.xticks(np.arange(2020, 2027), rotation=45)
plt.yticks(np.arange(0, 55 + 10, 10))  
plt.tight_layout()
plt.show()
'''
# FacetGrid 
g = sns.FacetGrid(full_df, col="Industry", col_wrap=4, height=3.5, sharey=True)
g.map_dataframe(sns.lineplot, x="Year", y="compound", marker="o")
g.set_titles(col_template="{col_name}")
g.set_axis_labels("Year", "ESG × 50")
plt.subplots_adjust(top=0.93)
g.fig.suptitle("2020–2026 ESG Forecast by Industry (Scaled)", fontsize=16)
plt.show()
 
