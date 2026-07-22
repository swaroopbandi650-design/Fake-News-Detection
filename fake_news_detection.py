# Step 1: Import libraries

import pandas as pd
import numpy as np
import re
import pickle
# Step 2: Read the datasets

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
print(fake.head())
print(true.head())
# Step 4: Add labels

fake["label"] = 0
true["label"] = 1
print(fake.head())
print(true.head())
# Step 5: Combine both datasets

data = pd.concat([fake, true], ignore_index=True)

print(data.head())
print(data.shape)
# Step 6: Check missing values

print(data.isnull().sum())
# Step 7: Combine title and text

data["content"] = data["title"] + " " + data["text"]

print(data[["content", "label"]].head())
# Step 8: Clean the text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text
# Apply cleaning

data["content"] = data["content"].apply(clean_text)

print(data[["content", "label"]].head())
# Step 9: Separate input and output

X = data["content"]
y = data["label"]

print(X.head())
print(y.head())
# Step 10: Split the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data :", X_train.shape)
print("Testing Data  :", X_test.shape)
# Step 11: Convert text into numbers

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train.shape)
print(X_test.shape)
# Step 12: Train Logistic Regression Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Model and Vectorizer saved successfully!")

print("Model trained successfully!")
# Step 13: Test the model

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100)
# Step 14: Test your own news

news = input("Enter News: ")

news = clean_text(news)

news_vector = vectorizer.transform([news])

prediction = model.predict(news_vector)

if prediction[0] == 0:
    print("Fake News")
else:
    print("Real News")
    