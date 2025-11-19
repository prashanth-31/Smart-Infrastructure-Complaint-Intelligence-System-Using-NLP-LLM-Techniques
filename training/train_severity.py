from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from training.config import PATHS
from utils.severity_features import SeverityFeatureBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SVM severity classifier")
    parser.add_argument("--csv", default=str(PATHS.data_csv), help="CSV path with complaint_text and severity columns")
    parser.add_argument("--output", default=str(PATHS.severity_path), help="Output pickle path")
    parser.add_argument("--embedding_model", default="distilbert-base-uncased", help="Transformer backbone for embeddings")
    parser.add_argument("--max_length", type=int, default=256, help="Max token length for embeddings")
    parser.add_argument("--c", type=float, default=1.0, help="SVM C parameter")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "complaint_text" not in df.columns or "severity" not in df.columns:
        raise ValueError("CSV must contain 'complaint_text' and 'severity' columns")
    df = df.dropna(subset=["complaint_text", "severity"]).reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["severity"], random_state=42)

    feature_builder = SeverityFeatureBuilder(embedding_model=args.embedding_model, max_length=args.max_length)
    X_train = feature_builder.fit_transform(train_df["complaint_text"].tolist())
    y_train = train_df["severity"].tolist()

    clf = SVC(C=args.c, probability=True, kernel="rbf")
    clf.fit(X_train, y_train)

    X_test = feature_builder.transform(test_df["complaint_text"].tolist())
    y_test = test_df["severity"].tolist()
    report = classification_report(y_test, clf.predict(X_test))
    print(report)

    payload = {
        "model": clf,
        "vectorizer": feature_builder,
    }
    joblib.dump(payload, args.output)
    print(f"Saved severity model to {args.output}")


if __name__ == "__main__":
    main()
