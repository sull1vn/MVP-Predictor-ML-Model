import joblib  # For loading the serialized model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# Load the serialized model
model = joblib.load('mvp_model.pkl')

dataset = '/Users/sullivanmotley/Desktop/ml_dataset/shuffled_data.csv'

# Load your test data (replace with your own data loading process)
test_data = pd.read_csv(dataset)

feature_columns = ['PYARDS', 'TD', 'INT', 'CMP%', 'RYARD', 'RYA', 'RTD', 'REC', 'Y/A', 'YEAR']
target_column = 'MVP'

# Separate features (X_test) and labels (y_test)
X_test = test_data[feature_columns]
y_test = test_data[target_column]

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Generate a classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
