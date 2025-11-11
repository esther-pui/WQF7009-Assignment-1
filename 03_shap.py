# %% Imports
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import shap
import numpy as np

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()

# Oversample
X_train, y_train = data_loader.oversample(X_train, y_train)

print(X_train.shape)
print(X_test.shape)

# %% Fit model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Task 2: Create XAI  explainer  (SHAP)
# %% SHAP for binary classification
explainer = shap.TreeExplainer(rf, feature_names=X_train.columns)
shap_values = explainer.shap_values(X_test)

print("SHAP output shape:", shap_values.shape)

# %% Local explanation
shap.initjs()

patient_index = 1
patient_shap = shap_values[patient_index, :, 1]
patient_features = X_test.iloc[patient_index]
base_value = explainer.expected_value[1]

print("Patient features:")
print(patient_features)

prediction = rf.predict(X_test.iloc[[patient_index]])[0]
print("Prediction:", prediction)

# Task 3
# Local Explainer (Force plot)
patient_index = 1

patient_shap = shap_values[patient_index, :, 1]  # class 1
patient_features = X_test.iloc[patient_index]

base_value = explainer.expected_value[1]
# shap.plots.force(base_value, patient_shap, patient_features, matplotlib=True)

# Task 4
# %% # Global explanation (summary plot)
sample = X_test.sample(200, random_state=42)
global_shap = explainer.shap_values(sample)

# For binary classification: use class 1 (stroke)
# shap.summary_plot(global_shap[..., 1], sample)
