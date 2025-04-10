import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"cluster_result.csv")

# Select relevant columns
df = df[['Company', 'compound', 'yahoo_esg', 'volatility', 'Cluster_Compound_Volatility']]
df['Cluster'] = df['Cluster_Compound_Volatility'].astype(str)

# Print cluster distribution
print("Cluster distribution:\n", df['Cluster'].value_counts())

# Split dataset into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.3, random_state=123, stratify=df['Cluster'])

# Define features and target
X_train = train_df[['compound',  'volatility']]
y_train = train_df['Cluster']
X_valid = valid_df[['compound', 'volatility']]
y_valid = valid_df['Cluster']

# Train Random Forest model with class weights to handle imbalance
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=123
)
rf_model.fit(X_train, y_train)

# Predict on validation set
y_pred = rf_model.predict(X_valid)

# Evaluate model performance
conf_matrix = confusion_matrix(y_valid, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_valid, y_pred))

# Plot confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()
