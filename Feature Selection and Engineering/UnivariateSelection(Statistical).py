import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels
feature_names = iris.feature_names

# Apply SelectKBest with chi2 to select top 2 best features
bestfeatures = SelectKBest(score_func=chi2, k=2)
fit = bestfeatures.fit(X, y)

# Create a DataFrame for feature scores
feature_scores = pd.DataFrame({"Feature": feature_names, "Score": fit.scores_})
feature_scores = feature_scores.sort_values(by="Score", ascending=False)

# Display feature importance
print("Top Features in the Iris Dataset:\n", feature_scores)

# Select the best two features and transform the dataset
X_new = bestfeatures.transform(X)
print("\nTransformed Dataset with Selected Features:\n", X_new[:5])  # Display first 5 rows