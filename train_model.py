"""
train_model.py
--------------
Train a TF-IDF + Logistic Regression classifier on historical GitHub tickets
and save the pipeline to model/classifier.pkl for use by the GitHub Action.

Usage:
    python scripts/train_model.py --data data/github_tickets_10k.csv
"""

import argparse
import json
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"title", "body", "cluster"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df["text"] = df["title"].fillna("") + " " + df["body"].fillna("")
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10_000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents="unicode",
                    analyzer="word",
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=5.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train(data_path: str, model_dir: str = "model") -> None:
    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)

    X, y = df["text"], df["cluster"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, evaluating on {len(X_test)} ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = report["accuracy"]
    print(f"\nAccuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "classifier.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}")

    # Save class labels and accuracy for reference
    meta = {
        "classes": sorted(y.unique().tolist()),
        "accuracy": round(accuracy, 4),
        "trained_on": len(X_train),
    }
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Metadata saved to model/meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/github_tickets_10k.csv",
        help="Path to labelled CSV",
    )
    parser.add_argument(
        "--model-dir", default="model", help="Directory to save trained model"
    )
    args = parser.parse_args()
    train(args.data, args.model_dir)
