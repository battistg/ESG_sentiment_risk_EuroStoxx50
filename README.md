# ESG Risk Profiling of Euro Stoxx 50 Companies using Sentiment Analysis

## Project Overview
This project proposes a data-driven framework to assess **ESG risk** for Euro Stoxx 50 companies using sentiment analysis from ESG-related news headlines (2020–2024). The aim is to offer a scalable alternative to traditional ESG ratings using natural language processing and machine learning techniques.

## Key Features
- ESG sentiment scoring using VADER
- Forecasting of ESG sentiment trends using SARIMA
- Clustering companies by sentiment, third party ESG rating, and volatility
- Classification models (Decision Tree, Random Forest) to predict ESG risk clusters
- Analysis at both company and industry level

## Project Structure

```bash
├── Data_sourcing_sentiment.py     
├── clustering.py                  
├── decision_tree.py               
├── random_forest.py               
├── forecasting.py                 
```
## Requirements

```bash
pip install pandas \
numpy \
matplotlib \
scikit-learn \
statsmodels \
nltk \
seaborn \
plotly
```

## How to Run
1. **Data Collection & Sentiment Scoring**
- Collect news headlines from Google News RSS feeds
- Clean and preprocess text
- Apply VADER sentiment analysis and generate annual sentiment scores
2. **Forecast ESG Sentiment**
- Generate sentiment forecasts for 2025–2026 using SARIMA
3. **Cluster Companies**
- Perform K-means clustering on sentiment, ESG scores, and stock volatility
4. **Classify ESG Risk Profiles**
- Perform decision tree analysis and random forest (with a bigger sample size, decision tree is preferable).

## Contributors
- Catarina Martins
- Dhruv Choudhary
- Giovanni Battistella
- Mansi Soni
- Sayesha Kakkar
- Xinyuan Hou
- Yinxiu Song

⸻

Disclaimer: This project is academic and intended for educational use.
