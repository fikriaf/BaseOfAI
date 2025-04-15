import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier

# Load Iris dataset
iris = load_iris()
x = iris.data  # Features
y = iris.target  # Target (species)
feature_names = iris.feature_names

# Train ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(x, y)

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame to display feature importance
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Display the feature importance
print(importance_df)

# Plot feature importance
plt.bar(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance using ExtraTreesClassifier")
plt.gca().invert_yaxis()
plt.show()