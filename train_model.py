import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Titanic dataset (you can use a CSV or a similar dataset)
df = pd.read_csv('titanic.csv')

# Preprocess the dataset (dummy code, customize according to your dataset)
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Feature selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features]
y = df['Survived']

# Train a model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save the model
joblib.dump(model, 'titanic_model.pkl')
