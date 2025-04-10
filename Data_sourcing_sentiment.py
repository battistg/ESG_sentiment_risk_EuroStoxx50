# Data Sourcing

import xmltodict
import requests
import pandas as pd
from datetime import datetime # To extract the year
import time
import os
import numpy as np
import re
import nltk
import string
import matplotlib.pyplot as plt


# Article limit
ARTICLES_PER_YEAR_PER_COMPANY = 50

# Companies list form Euro Stoxx 50
euro_stoxx_50_companies = [
    "Airbus", "Allianz", "Anheuser-Busch InBev", "ASML", "AstraZeneca", "BASF", "Bayer", "BMW", "BNP Paribas",
    "CRH", "Daimler", "Danone", "Deutsche Bank", "Deutsche Börse", "Deutsche Post", "Deutsche Telekom", "Enel",
    "Engie", "Luxottica", "Eurazeo", "Eutelsat", "Ferrari", "GlaxoSmithKline", "Hermès", "Iberdrola",
    "Inditex", "ING Group", "Intesa Sanpaolo", "Kering", "L'Oreal", "LVMH", "Merck", "Munich Re", "Nestlé",
    "Novartis", "Pernod Ricard", "Prosus", "Publicis", "Repsol", "Roche", "Safran", "Sanofi", "SAP", "Schneider Electric",
    "Siemens", "Société Générale", "STMicroelectronics", "TotalEnergies", "Unilever", "Volkswagen"
]

# Keywords
esg_keywords = [
    "sustainability", "climate", "carbon", "renewable", "energy", "circular",
    "environmental","eco", "CSR", "ethical", "ethics", "diversity", "greenwashing", 
    "ESG", "sustainable", "human rights", "inclusion", "equality", "pollution", "emission", "deforestation", "toxic",
    "contamination", "waste", "oil spill", "exploitation", "corruption", "bribe", "wildlife", "unsustainable", "harassment", "violation",
    "strike", "protest", "unsafe", "abuse", "labor", "bias", "fraud", "misconduct", "scandal"
]

# Google News RSS template
google_news_rss_url_template = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

def getRSS(url: str) -> dict:
    """Download and convert RSS feed into a Python Dictionary"""
    response = requests.get(url)
    return xmltodict.parse(response.content)

# Nested Dictionary
filtered_articles = {
    company: {year: [] for year in range(2020, 2025)}
    for company in euro_stoxx_50_companies
}

# Downloading the articles (for each comany and year)
for company in euro_stoxx_50_companies:
    for keyword in esg_keywords:
        query = f"{company} {keyword}".replace(" ", "+")
        rss_url = google_news_rss_url_template.format(query=query)

        print(f"Fetching RSS from Google News for: {company} - {keyword}")

        try:
            data = getRSS(rss_url)

            # Analysing articles
            if ("rss" in data and "channel" in data["rss"] and 
                "item" in data["rss"]["channel"]):
                
                for item in data["rss"]["channel"]["item"]:
                    pub_date_str = item.get("pubDate", "")
                    try:
                        pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
                        year = pub_date.year
                    except ValueError:
                        continue  # Ignoring articles where "date" is not valid

                    # Filtering years (2020-2024)
                    if 2020 <= year <= 2024:
                        # Check if title or description contains at least a keyword
                        title = item["title"].lower()
                        description = item.get("description", "").lower()
                        if any(k in title or k in description for k in esg_keywords):
                            # limit number of articles
                            if len(filtered_articles[company][year]) < ARTICLES_PER_YEAR_PER_COMPANY:
                                filtered_articles[company][year].append({
                                    "Company": company,
                                    "Keyword": keyword,
                                    "Title": item["title"],
                                    "Description": item.get("description", ""),
                                    "URL": item["link"],
                                    "Published At": pub_date_str
                                })

        except Exception as e:
            print(f"Errore con Google News RSS per {company} - {keyword}: {e}")

        # Break the request to prevent limit exceeding 
        time.sleep(1)

# Transform the nested dictionary in a list
final_articles = []
for company, years_data in filtered_articles.items():
    for year, articles_list in years_data.items():
        final_articles.extend(articles_list)

# Creating data frame
df = pd.DataFrame(final_articles)


# Save articles into csv
csv_path = "/Users/gio/Desktop/TRINITY COLLEGE/MODULES/ESG ANALYTICS/Group Project/articles.csv"
df.to_csv(csv_path, index=False)

###-----------------------------
# Data cleaning

# Removing duplicates

df_cleaned = df.drop_duplicates(subset=["Title"])

# Drop Description, Keyword and URL columns

df_cleaned = df.drop(columns=["Keyword", "Description", "URL"])

# Convert date and extract the year

df_cleaned["Published At"] = pd.to_datetime(df_cleaned["Published At"], errors="coerce", utc=True)
df_cleaned["Year"] = df_cleaned["Published At"].dt.year

# Grouping by (Company, Year) and aggregate the titles
agg_df = df_cleaned.groupby(["Company", "Year"])["Title"].apply(lambda x: " ".join(x)).reset_index()

# Text data cleaning
nltk.download('stopwords')
stop = set(nltk.corpus.stopwords.words('english'))
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Lowercase and strip
    text = text.lower().strip()
    # Remove stopwords
    words = [w for w in text.split() if w not in stop]
    text = " ".join(words)
    # Remove punctuation and special characters
    text = re.sub(r"[{}]".format(string.punctuation), "", text)
    text = re.sub(r"[^\w\s']", "", text)  # remove other symbols
    text = re.sub(r"\d+", "", text)       # remove digits
    text = re.sub(r"@\w+", "", text)      # remove mentions
    text = re.sub(r"#\w+", "", text)      # remove hashtags
    # Lemmatization
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    text = " ".join(tokens)
    return text

agg_df["Cleaned_Title"] = agg_df["Title"].apply(clean_text)

###------------------------
# Sentiment Analysis

# Sentiment Analysis with VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    scores = analyzer.polarity_scores(text)
    return scores  # dictionary with {neg, neu, pos, compound}

agg_df["VaderScores"] = agg_df["Cleaned_Title"].apply(analyze_sentiment_vader)
# Expand the dictionary into separate columns
agg_df = pd.concat([agg_df, agg_df["VaderScores"].apply(pd.Series)], axis=1)
agg_df.drop(columns=["VaderScores"], inplace=True)

def interpret_sentiment(compound):
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

agg_df["VaderSentiment"] = agg_df["compound"].apply(interpret_sentiment)

# Removing Title and Cleaned Title columns

agg_df = agg_df.drop(columns=["Title", "Cleaned_Title"])

# Show Result
agg_df.head()

# Save the result
agg_df.to_csv("esg_sentiment_score.csv", index=False)

# Plot top 10 companies by sentiment score
# Calculate the overall average compound sentiment score per company (across all years)
company_avg = agg_df.groupby("Company")["compound"].mean().reset_index()

# Top 10 companies with the highest average sentiment scores
top10 = company_avg.nlargest(10, "compound")

plt.figure(figsize=(10,6))
bars = plt.bar(top10["Company"], top10["compound"], color='steelblue')
plt.title("Top 10 Companies by Average Sentiment Score (2020-2024)")
plt.xlabel("Company")
plt.ylabel("Average Compound Sentiment Score")

# Narrow the y-axis 
plt.ylim([0.985, 1.0])  
plt.xticks(rotation=45)

# Add data labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.0005,        # small offset above the bar
        f"{height:.4f}",       # 4 decimal places
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()


# Bottom 10 companies with the lowest average sentiment scores
bottom10 = company_avg.nsmallest(10, "compound")

plt.figure(figsize=(10,6))
bars = plt.bar(bottom10["Company"], bottom10["compound"], color='firebrick')
plt.title("Bottom 10 Companies by Average Sentiment Score (2020-2024)")
plt.xlabel("Company")
plt.ylabel("Average Compound Sentiment Score")

plt.xticks(rotation=45)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.0005,   # piccola distanza sopra la barra
        f"{height:.4f}",  # 4 cifre decimali
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()

# Industry based sentiment plots


industry_map = {
    'ASML': 'Semiconductors',
    'Airbus': 'Aerospace',
    'Allianz': 'Insurance',
    'AB InBev': 'Food & Beverage',
    'AstraZeneca': 'Pharmaceuticals',
    'BASF': 'Chemicals',
    'BMW': 'Automotive',
    'BNP Paribas': 'Banking',
    'Bayer': 'Pharmaceuticals',
    'CRH': 'Building Materials',
    'Daimler': 'Automotive',
    'Danone': 'Food & Beverage',
    'Deutsche Bank': 'Banking',
    'Deutsche Börse': 'Financial Services',
    'Deutsche Post DHL': 'Logistics',
    'Deutsche Telekom': 'Telecommunications',
    'Enel': 'Energy',
    'Engie': 'Energy',
    'Eurazeo': 'Investment',
    'Eutelsat': 'Telecommunications',
    'Ferrari': 'Automotive',
    'GSK': 'Pharmaceuticals',
    'Hermès': 'Luxury Goods',
    'ING Group': 'Banking',
    'Iberdrola': 'Energy',
    'Inditex': 'Apparel Retail',
    'Intesa Sanpaolo': 'Banking',
    'Kering': 'Luxury Goods',
    'L’Oréal': 'Cosmetics',
    'LVMH': 'Luxury Goods',
    'Luxottica': 'Luxury Goods',
    'Merck KGaA': 'Pharmaceuticals',
    'Munich Re': 'Insurance',
    'Nestlé': 'Food & Beverage',
    'Novartis': 'Pharmaceuticals',
    'Pernod Ricard': 'Food & Beverage',
    'Prosus': 'Internet',
    'Publicis Groupe': 'Advertising',
    'Repsol': 'Energy',
    'Roche': 'Pharmaceuticals',
    'SAP': 'Software',
    'STMicroelectronics': 'Semiconductors',
    'Safran': 'Aerospace',
    'Sanofi': 'Pharmaceuticals',
    'Schneider Electric': 'Industrial Manufacturing',
    'Siemens': 'Industrial Manufacturing',
    'Société Générale': 'Banking',
    'TotalEnergies': 'Energy',
    'Unilever': 'Consumer Goods',
    'Volkswagen': 'Automotive'
}

agg_df['Industry'] = agg_df['Company'].map(industry_map)

agg_df = agg_df.dropna(subset=['Industry'])

agg_df['Year'] = pd.to_datetime(agg_df['Year'], format='%Y')

#Average compound score per industry per year
industry_ts = agg_df.groupby(['Industry', 'Year'])['compound'].mean().reset_index()

# Average compound score per industry overall
industry_overall = agg_df.groupby(['Industry'])['compound'].mean().reset_index()

# Top 5 industries with the highest average sentiment scores
top10_ind = industry_overall.nlargest(5, "compound")

plt.figure(figsize=(10,6))
bars = plt.bar(top10_ind["Industry"], top10_ind["compound"], color='steelblue')
plt.title("Top 5 Industries by Average Sentiment Score (2020-2024)")
plt.xlabel("Industry")
plt.ylabel("Average Compound Sentiment Score")

# Narrow the y-axis 
plt.ylim([0.80, 1.0])  
plt.xticks(rotation=45)

# Add data labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.0005,        # small offset above the bar
        f"{height:.4f}",       # 4 decimal places
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()


# Bottom 5 industries with the lowest average sentiment scores
bottom10_ind = industry_overall.nsmallest(5, "compound")

plt.figure(figsize=(10,6))
bars = plt.bar(bottom10_ind["Industry"], bottom10_ind["compound"], color='firebrick')
plt.title("Bottom 5 Industries by Average Sentiment Score (2020-2024)")
plt.xlabel("Industry")
plt.ylabel("Average Compound Sentiment Score")

plt.xticks(rotation=45)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.0005,   # piccola distanza sopra la barra
        f"{height:.4f}",  # 4 cifre decimali
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()
d