import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("cluster_result.csv")

# Select relevant columns
df = df[['Company', 'compound', 'yahoo_esg', 'volatility', 'Cluster_Compound_Volatility']]
df['Cluster'] = df['Cluster_Compound_Volatility'].astype(str)  # Convert Cluster to categorical

# Print cluster distribution
print(df['Cluster'].value_counts())

# Split dataset into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.3, random_state=123, stratify=df['Cluster'])

# Define features and target
X_train = train_df[['compound', 'volatility']]
y_train = train_df['Cluster']
X_valid = valid_df[['compound', 'volatility']]
y_valid = valid_df['Cluster']

# Train Decision Tree model with class weights to handle imbalance
unique_classes = y_train.unique()
class_weights = {cls: weight for cls, weight in zip(unique_classes, [0.4, 0.4, 0.2])}

tree_model = DecisionTreeClassifier(criterion='gini', min_samples_split=2, min_samples_leaf=1, class_weight=class_weights, random_state=123)
tree_model.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(12, 6))
plot_tree(tree_model, feature_names=['compound', 'volatility'], class_names=tree_model.classes_, filled=True, rounded=True)
plt.title("Decision Tree for ESG-Based Cluster Classification (Balanced)")
plt.show()

# Predict on validation set
y_pred = tree_model.predict(X_valid)

# Evaluate model performance
conf_matrix = confusion_matrix(y_valid, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_valid, y_pred))
