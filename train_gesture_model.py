import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the data
data = pd.read_csv('gesture_data.csv', header=None)

# Features are all columns except the last, label is the last column
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data (20% test, stratified by label)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Save the model
joblib.dump(clf, 'gesture_model.joblib')
print('Model saved as gesture_model.joblib')
