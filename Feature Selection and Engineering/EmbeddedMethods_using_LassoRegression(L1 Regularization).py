from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
import pandas as pd

# Load dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
feature_names = diabetes.feature_names

# Train Lasso model
lasso = Lasso(alpha=0.1)  # L1 Regularization
lasso.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({"Feature": feature_names, "Coefficient": lasso.coef_})
feature_importance = feature_importance.sort_values(by="Coefficient", ascending=False)

# Display results
print("Lasso Regression Feature Selection:\n", feature_importance)