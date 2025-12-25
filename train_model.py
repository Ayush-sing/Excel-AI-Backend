import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

TRAINING_FILE_PRIMARY = "excel_intents_extended.csv"
TRAINING_FILE_FALLBACK = "excel_intents.csv"
MODEL_FILE = "excel_intent_model.pkl"

train_file = TRAINING_FILE_PRIMARY if os.path.exists(TRAINING_FILE_PRIMARY) else TRAINING_FILE_FALLBACK
if not os.path.exists(train_file):
    raise FileNotFoundError("No training data found. Provide excel_intents.csv or run generator.")

df = pd.read_csv(train_file)
if "message" not in df.columns or "intent" not in df.columns:
    raise ValueError("Training CSV must have 'message' and 'intent' columns.")

X = df["message"]
y = df["intent"]

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X, y)
joblib.dump(model, MODEL_FILE)
print(f"✅ Trained on {len(df)} samples from {train_file} → saved {MODEL_FILE}")
