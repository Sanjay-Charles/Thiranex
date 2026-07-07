import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion


############################################################
# Load Dataset
############################################################

df = pd.read_csv("emails.zip")

# Replace missing values
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")

# Combine subject and body into one text column
df["text"] = df["subject"] + " " + df["body"]

# Features and labels
X = df["text"]
y = df["label"]


############################################################
# URL Feature Extractor
############################################################

class URLFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        phishing_words = [
            "verify",
            "bank",
            "login",
            "password",
            "urgent",
            "account",
            "click",
            "winner",
            "prize",
            "update",
            "confirm",
            "security",
            "paypal",
            "invoice",
            "gift",
            "free"
        ]

        feature_list = []

        for email in X:

            email = str(email)

            urls = re.findall(r"http[s]?://\S+", email)

            url_count = len(urls)

            has_ip = int(
                any(
                    re.search(r"\d+\.\d+\.\d+\.\d+", url)
                    for url in urls
                )
            )

            long_url = int(any(len(url) > 40 for url in urls))

            keyword_count = sum(
                email.lower().count(word)
                for word in phishing_words
            )

            feature_list.append([
                url_count,
                has_ip,
                long_url,
                keyword_count
            ])

        return np.array(feature_list)


############################################################
# Feature Extraction
############################################################

combined_features = FeatureUnion([

    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )
    ),

    (
        "url_features",
        URLFeatures()
    )

])


############################################################
# Build Model
############################################################

model = Pipeline([

    ("features", combined_features),

    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )

])


############################################################
# Train/Test Split
############################################################

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

############################################################
# Train
############################################################

print("Training model...\n")

model.fit(X_train, y_train)

############################################################
# Predict
############################################################

predictions = model.predict(X_test)

############################################################
# Accuracy
############################################################

accuracy = accuracy_score(y_test, predictions)

print("Accuracy : {:.2f}%".format(accuracy * 100))

print("\nClassification Report\n")

print(classification_report(y_test, predictions))

############################################################
# Confusion Matrix
############################################################

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Phishing"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.show()

############################################################
# Test Sample Email
############################################################

while True:

    print("\n-------------------------------------")
    print("Enter an email to test.")
    print("Type 'exit' to quit.")
    print("-------------------------------------")

    sample = input("\nEmail: ")

    if sample.lower() == "exit":
        break

    prediction = model.predict([sample])[0]

    if prediction == 1:
        print("\nPrediction : PHISHING EMAIL")
    else:
        print("\nPrediction : SAFE EMAIL")
