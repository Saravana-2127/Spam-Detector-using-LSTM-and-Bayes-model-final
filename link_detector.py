"""
Link/URL Spam & Phishing Classifier
=====================================
Trains a classifier that looks at a URL's STRUCTURE (length, use of IP
addresses, suspicious TLDs, keyword stuffing, etc. — see url_features.py)
rather than natural-language word frequency, since URLs aren't sentences.

This is deliberately a different model family (Random Forest over engineered
features) from the Naive Bayes / LSTM text models, for two reasons:
  1. It matches how real phishing-URL research approaches this exact problem.
  2. Random Forests give free feature-importance numbers, so we get an
     explainability story here too, consistent with the rest of the project.

Dataset: phishing_site_urls.csv (~550k URLs, labelled "good"/"bad"),
sourced from a public phishing-detection research project.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from urllib.parse import urlparse

from url_features import extract_features


def load_dataset(path="data/phishing_site_urls.csv", sample_per_class=25000, seed=42):
    """
    The full dataset has ~550k rows. For a student project, training on a
    balanced subsample is both fast and avoids the model just learning to
    always predict the majority class ("good").
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["URL", "Label"])

    good = df[df["Label"] == "good"].sample(n=sample_per_class, random_state=seed)
    bad = df[df["Label"] == "bad"].sample(n=min(sample_per_class, (df["Label"] == "bad").sum()), random_state=seed)

    balanced = pd.concat([good, bad]).sample(frac=1, random_state=seed).reset_index(drop=True)

    # In this dataset, long deep URLs are mostly "good" and short bare domains mostly "bad", so a model
    # learns "short homepage = phishing" (it called gmail.com 99% malicious). Fix: also train on the
    # ROOT-ONLY form of every URL with the same label, so path length/depth can't be the shortcut.
    def root_only(u):
        u = u if "://" in u else "http://" + u
        return urlparse(u).netloc
    roots = balanced.copy()
    roots["URL"] = roots["URL"].map(root_only)
    roots = roots[roots["URL"].str.len() > 3]
    balanced = pd.concat([balanced, roots]).drop_duplicates(subset="URL", keep="first")
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced


def build_feature_matrix(urls):
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows)


def train():
    print("Loading dataset...")
    df = load_dataset()
    print(f"Using {len(df)} URLs ({(df['Label']=='bad').sum()} bad, {(df['Label']=='good').sum()} good)")

    print("Extracting lexical features (this takes a moment for ~50k URLs)...")
    X = build_feature_matrix(df["URL"])
    y = (df["Label"] == "bad").astype(int)  # 1 = malicious, 0 = legitimate

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print(f"\nTest set size: {len(y_test)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (malicious): {prec:.4f}")
    print(f"Recall (malicious):    {rec:.4f}")
    print(f"F1 score (malicious):  {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")

    print("\nFeature importances (which URL properties matter most):")
    importances = sorted(
        zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    for name, score in importances:
        print(f"  {name}: {score:.4f}")

    # Sample predictions
    examples = [
        "http://192.168.1.5/login/verify-account",
        "https://www.google.com/search?q=probability+theory",
        "bit.ly/x7f2Ks9qLp",
        "www.gmail.com",
        "https://www.facebook.com/",
        "secure-login-paypal-verify.tk/account/update",
    ]
    print("\nSample predictions:")
    for url in examples:
        feats = pd.DataFrame([extract_features(url)])
        prob = model.predict_proba(feats)[0][1]
        print(f"  {url} -> malicious probability: {prob:.3f}")

    with open("link_model.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_names": list(X.columns)}, f)
    print("\nModel saved to link_model.pkl")


if __name__ == "__main__":
    train()