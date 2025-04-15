import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels
feature_names = iris.feature_names

# Train a model using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Apply Recursive Feature Elimination (RFE) to select top 2 features
rfe = RFE(estimator=model, n_features_to_select=2)
fit = rfe.fit(X, y)

# Create a DataFrame for feature ranking
feature_ranking = pd.DataFrame({"Feature": feature_names, "Ranking": fit.ranking_})
feature_ranking = feature_ranking.sort_values(by="Ranking")

# Display feature ranking
print("Feature Selection using RFE:\n", feature_ranking)