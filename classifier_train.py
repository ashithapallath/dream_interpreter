import pandas as pd

df = pd.read_csv('data/rsos_dream_data.csv', usecols=['text_dream', 'emotions_code'])

print("Dataset Loaded!")
print(df.info())  # Check for missing values
print(df.head())  # Show first few rows
df = df.dropna(subset=['text_dream', 'emotions_code'])  
df = df[df['emotions_code'].str.strip() != '']  # Remove empty labels

print(f"After cleaning, dataset has {df.shape[0]} rows")
print(df.head())
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text_dream'].str.lower().values)

print(f"Vectorized shape: {X.shape}")
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss='log_loss', max_iter=300, tol=1e-3)
clf.fit(X, df['emotions_code'].values)

print("Model trained successfully!")
import joblib

joblib.dump((vectorizer, clf), 'classifier_cpu_fast.pkl')
print("Classifier saved successfully!")

