from os import write
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import statsmodels.api as sm

# Set display options
pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid")

# read esg_sentiment_score.csv
df = pd.read_csv('esg_sentiment_score.csv')

# 1. Clustering based on compound
# Aggregate data by company to calculate the average emotional scores for each company in all years
df_company_aggregated = df.groupby("Company")[["compound"]].mean().reset_index()

# Select the feature used for clustering
features = ["compound"]
X = df_company_aggregated[features]

# Standardized data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(pd.DataFrame(X_scaled, columns=features).head())

#Calculate SSE (sum of squared errors) at different k values
sse = []
k_range = range(1, 10)  

for k in k_range:
     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
     kmeans.fit(X_scaled)
     sse.append(kmeans.inertia_) 

 # Draw the elbow method chart and select the best k value (k=3)
plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker="o", linestyle="-", color="b")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("SSE")
plt.title("Elbow Method for Optimal k")
plt.show()

sil_scores = []
for k in k_range[1:]:  # silhouette_score 
     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
     labels = kmeans.fit_predict(X_scaled)
     sil_scores.append(silhouette_score(X_scaled, labels))

# Draw Silhouette Score The biggest thing is k=4
plt.figure(figsize=(8, 5))
plt.plot(k_range[1:], sil_scores, marker="o", linestyle="-", color="g")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal k")
plt.show()

# Select k=3 for clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_company_aggregated["Cluster"] = kmeans.fit_predict(X_scaled)

print(df_company_aggregated[["Company","Cluster"]].head())
print(df_company_aggregated.groupby("Cluster")[features].mean())


# 2. Clustering based on Yahoo ESG score

# 50 companies ticker name in Yahoo Finance
ticker_mapping = {
    "ASML": "ASML.AS",
    "Prosus": "PRX.AS",
    "ING Group": "INGA.AS",
    "Airbus": "AIR.PA",
    "BNP Paribas": "BNP.PA",
    "L'Oreal": "OR.PA",
    "LVMH": "MC.PA",
    "Sanofi": "SAN.PA",
    "TotalEnergies": "TTE.PA",
    "Schneider Electric": "SU.PA",
    "Société Générale": "GLE.PA",
    "Publicis": "PUB.PA",
    "Safran": "SAF.PA",
    "Engie": "ENGI.PA",
    "Danone": "BN.PA",
    "Hermès": "RMS.PA",
    "Kering": "KER.PA",
    "STMicroelectronics": "STMPA.PA",
    "Eutelsat": "ETL.PA",
    "Eurazeo": "RF.PA",
    "Pernod Ricard": "RI.PA",
    "Allianz": "ALV.DE",
    "SAP": "SAP.DE",
    "Siemens": "SIE.DE",
    "Deutsche Bank": "DBK.DE",
    "Deutsche Börse": "DB1.DE",
    "Deutsche Post": "DHL.DE",
    "Deutsche Telekom": "DTE.DE",
    "Volkswagen": "VOW3.DE",
    "Merck": "MRK.DE",
    "BMW": "BMW.DE",
    "BASF": "BAS.DE",
    "Bayer": "BAYN.DE",
    "Munich Re": "MUV2.DE",
    "Daimler": "MBG.DE",  
    "Enel": "ENEL.MI",
    "Intesa Sanpaolo": "ISP.MI",
    "Ferrari": "RACE.MI",
    "Luxottica": "EL.PA",  
    "Iberdrola": "IBE.MC",
    "Inditex": "ITX.MC",
    "Repsol": "REP.MC",
    "Nestlé": "NESN.SW",
    "Novartis": "NOVN.SW",
    "Roche": "ROG.SW",
    "AstraZeneca": "AZN.L",
    "Unilever": "ULVR.L",
    "GlaxoSmithKline": "GSK.L",
    "CRH": "CRH.L",
    "Anheuser-Busch InBev": "ABI.BR",
}

# Use “yfinance” function to get stock data: Download 2019-12-31 to 2024-12-31 data 
data = yf.download(
    tickers=list(ticker_mapping.values()),
    start="2020-01-01",
    end="2024-12-31",
    group_by="ticker"
 )
    
company_esg = df_company_aggregated.copy()

def get_yahoo_esg(ticker):
    """get ESG total score"""
    try:
        stock = yf.Ticker(ticker)
        esg_data = stock.sustainability
        if esg_data is not None and "totalEsg" in esg_data.index:
            esg_value = esg_data.loc["totalEsg"] 
        # Extract number parts using regular expressions
            match = re.search(r'(\d+\.\d+)', str(esg_value))
            if match:
                return float(match.group(1)) 
        return np.nan
    except:
        return np.nan
    
print(get_yahoo_esg('ASML.AS'))

# Apply functions
company_esg["yahoo_esg"] = company_esg["Company"].map(
    lambda x: get_yahoo_esg(ticker_mapping.get(x, None)))
yahoo_esg = company_esg[["yahoo_esg"]].dropna()  # 去掉缺失值

print(company_esg.head())


# Standardized data
scaler = StandardScaler()
yahoo_esg_scaled = scaler.fit_transform(yahoo_esg)

# choose k=3
kmeans_yahoo_esg = KMeans(n_clusters=3, random_state=42, n_init=10)
company_esg["Cluster_ESG"] = kmeans_yahoo_esg.fit_predict(yahoo_esg_scaled)

print(company_esg[["Company", "Cluster_ESG", "yahoo_esg"]].head())

# 3. Clustering based on compound + volatility

# Calculate annualized volatility according to Yahoo Finance's stock data
df_company_esg_vol = company_esg.copy()
volatility_data = []

for company in df_company_esg_vol['Company']:
    ticker = ticker_mapping.get(company, None)
    
    if ticker:
        try:
            stock_data = data[ticker]
            # Calculate daily rate of return
            stock_data['daily_return'] = stock_data['Close'].pct_change()
            
            # Calculate the standard deviation of daily yield
            daily_volatility = stock_data['daily_return'].std()
            
            # Annualize the daily volatility, annualized volatility = daily volatility * √252 (252 is the number of trading days)
            annualized_volatility = daily_volatility * np.sqrt(252)
            
            volatility_data.append(annualized_volatility)
        except KeyError:
            print(f"can't find {company} ({ticker}) data")
            volatility_data.append(None)
    else:
        print(f"{company} not exist in ticker_mapping")
        volatility_data.append(None)

# Add volatility data to df_company_aggregated
df_company_esg_vol['volatility'] = volatility_data
print(df_company_esg_vol)

# Select the feature use for cluster
features = ["compound", "volatility"]
X = df_company_esg_vol[features]

# Use “StandarScaler” to standardized data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_company_esg_vol["Cluster_Compound_Volatility"] = kmeans.fit_predict(X_scaled)

print(df_company_esg_vol)

# 4. Clustering based on Yahoo ESG + volatility

# Select the feature use for cluster
features_esg_volatility = ["yahoo_esg", "volatility"]
X_esg_volatility = df_company_esg_vol[features_esg_volatility]

# Use “StandarScaler” to standardized data
scaler_esg_volatility = StandardScaler()
X_esg_volatility_scaled = scaler_esg_volatility.fit_transform(X_esg_volatility)

# Select k=3 for clustering
kmeans_esg_volatility = KMeans(n_clusters=3, random_state=42, n_init=10)
df_company_esg_vol["Cluster_ESG_Volatility"] = kmeans_esg_volatility.fit_predict(X_esg_volatility_scaled)

print(df_company_esg_vol[["Company", "Cluster_ESG_Volatility", "yahoo_esg", "volatility"]].head())


# 5. Clustering based on compound + Yahoo ESG + volatility

# Select the feature use for cluster
features_esg_volatility = ["yahoo_esg", "volatility","compound"]
X_esg_volatility_compound = df_company_esg_vol[features_esg_volatility]

# Use “StandarScaler” to standardized data
scaler_esg_volatility = StandardScaler()
X_esg_vol_com_scaled = scaler_esg_volatility.fit_transform(X_esg_volatility_compound)

# Select k=3 for clustering
kmeans_esg_vol_com = KMeans(n_clusters=3, random_state=42, n_init=10)
df_company_esg_vol["Cluster_ESG_Volatility_compound"] = kmeans_esg_vol_com.fit_predict(X_esg_vol_com_scaled)

print(df_company_esg_vol[["Company", "Cluster_ESG_Volatility_compound", "yahoo_esg", "volatility","compound"]].head())

# print cluster_result.csv
df_company_esg_vol.to_csv("cluster_result.csv", index=False)

# 1. The impact of emotional ratings on ESG ratings
df_company_esg_vol['sentiment_category'] = df_company_esg_vol['Cluster'].map({2: 'Negative', 1: 'Neutral', 0: 'Positive'})
plt.figure(figsize=(10, 6))
palette = {'Negative': '#86C6F0', 'Neutral': '#FFDD8C', 'Positive': '#A0E6A1'}
sns.boxplot(x='sentiment_category', y='yahoo_esg', data=df_company_esg_vol, palette=palette)
plt.title('Distribution of ESG Scores by Sentiment Category')
plt.show()

# 2. Research on the Correlation between ESG Scores and Stock Volatility
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_company_esg_vol, x='yahoo_esg', y='volatility', hue='Cluster_ESG_Volatility', palette='Set2', style='Cluster_ESG', s=100, alpha=0.7)
plt.title('ESG Score vs Volatility by Cluster', fontsize=14)
plt.xlabel('Yahoo ESG Score', fontsize=12)
plt.ylabel('Volatility', fontsize=12)
plt.legend(title='Cluster', title_fontsize='13', loc='lower right')
plt.show()

# 3. Analysis of Stock Price Volatility by Sentiment Category
df_company_esg_vol['sentiment_category'] = df_company_esg_vol['Cluster_Compound_Volatility'].map({2: 'Negative', 1: 'Neutral', 0: 'Positive'})  
palette = {'Negative': '#86C6F0', 'Neutral': '#FFDD8C', 'Positive': '#A0E6A1'}  
plt.figure(figsize=(10, 6))
sns.boxplot(x='sentiment_category', y='volatility', data=df_company_esg_vol, palette=palette)
plt.title('Distribution of Volatility by Sentiment Category', fontsize=14)
plt.xlabel('Sentiment Category', fontsize=12)
plt.ylabel('Volatility', fontsize=12)
plt.show()

#4.Impact of Sentiment, ESG, and Volatility on Company Performance
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Set up a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
scatter = ax.scatter(df_company_esg_vol['compound'], df_company_esg_vol['yahoo_esg'], df_company_esg_vol['volatility'], 
                     c=df_company_esg_vol['Cluster_ESG_Volatility_compound'], cmap='viridis', s=100)

# Set labels and title
ax.set_xlabel('Sentiment (Compound)')
ax.set_ylabel('ESG Score')
ax.set_zlabel('Volatility')
ax.set_title('3D Scatter Plot: Compound, ESG, and Volatility by Cluster')

# Add a colorbar
plt.colorbar(scatter, label='Cluster')
plt.show()

# Set up a stacked bar chart
cluster_summary = df_company_esg_vol.groupby('Cluster_ESG_Volatility_compound')[['compound', 'yahoo_esg', 'volatility']].mean()

# Plot the stacked bar chart
# Set up a stacked bar chart with softer colors
cluster_summary.plot(kind='bar', stacked=True, figsize=(12, 8), 
                     color=['#B3D9FF', '#FFDD8C', '#A0E6A1'])  # Softer colors

# Update the title and labels
plt.title('Stacked Bar Chart: Mean Sentiment, ESG, and Volatility by Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Mean Values', fontsize=12)
plt.xticks(rotation=0)

# Update the legend title and labels
plt.legend(title='Features', labels=['Sentiment Score', 'ESG Score', 'Volatility'])

# Show the plot
plt.show()
