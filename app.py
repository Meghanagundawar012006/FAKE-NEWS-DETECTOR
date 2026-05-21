# Fake News Detector using Machine Learning
# ----------------------------------------
# Install required libraries before running:
# pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Load Dataset
# -------------------------------
# Make sure news.csv is in the same folder as this Python file

data = pd.read_csv("news.csv")

# Display first 5 rows
print("Dataset Preview:")
print(data.head())

# -------------------------------
# Check Dataset Columns
# -------------------------------
print("\nColumns in Dataset:")
print(data.columns)

# -------------------------------
# Features and Labels
# -------------------------------
# Dataset should contain:
# text  -> news article
# label -> FAKE or REAL

X = data['text']
y = data['label']

# -------------------------------
# Split Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Convert Text into Numbers
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# -------------------------------
# Train Model
# -------------------------------
model = LogisticRegression()

model.fit(X_train_vectorized, y_train)

# -------------------------------
# Model Testing
# -------------------------------
y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# -------------------------------
# Fake News Prediction Function
# -------------------------------
def predict_news(news_text):
    news_vector = vectorizer.transform([news_text])
    prediction = model.predict(news_vector)

    if prediction[0] == 'FAKE':
        return "The News is FAKE"
    else:
        return "The News is REAL"

# -------------------------------
# User Input
# -------------------------------
print("\n----- Fake News Detector -----")

user_news = input("Enter a news article or headline:\n")

result = predict_news(user_news)

print("\nPrediction:", result)