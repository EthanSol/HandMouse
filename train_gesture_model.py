import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
import joblib

# Load and split data from all files in gesture_data
train_data = []
test_data = []
data_folder = 'gesture_data'
for file in glob.glob(os.path.join(data_folder, '*.csv')):
    data_frame = pd.read_csv(file, header=None)
    data_frame = shuffle(data_frame, random_state=42).reset_index(drop=True)
    n_train = int(0.7 * len(data_frame))
    train_data.append(data_frame.iloc[:n_train])
    test_data.append(data_frame.iloc[n_train:])

train_df = pd.concat(train_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

clf = RandomForestClassifier(n_estimators=70, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

joblib.dump(clf, 'gesture_model.pkl')
print('Model saved as gesture_model.pkl')
