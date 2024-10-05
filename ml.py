import joblib
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('malware.csv', sep='|')
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)  # For numerical columns
df.fillna(df.select_dtypes(include=[object]).mode().iloc[0], inplace=True)  # For categorical columns

# Prepare features and labels
X = df.drop(['Name', 'md5', 'legitimate'], axis=1).values
y = df['legitimate'].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection
extratrees = ExtraTreesClassifier().fit(X_scaled, y)
feature_selector = SelectFromModel(extratrees, prefit=True)
X_new = feature_selector.transform(X_scaled)

# Save the feature selector
joblib.dump(feature_selector, 'feature_selector.pkl')

# Save the classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_new, y)

joblib.dump(decision_tree, 'classifier.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Save feature names
selected_features = [df.columns[i] for i in feature_selector.get_support(indices=True)]
with open('features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)
