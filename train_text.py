"""
python3 train_text.py
Reads data/sms.tsv  +  data/extra_*.tsv (label<TAB>text, e.g. your own Indian SMS/email samples)
+ feedback.csv (your corrections win over older labels). De-duplicates BEFORE splitting
(the SMS Spam Collection has many near-duplicates that inflate test accuracy).
"""
import csv, glob, os, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve)
from text_model import build_pipeline, message_key, has_text


FEEDBACK_WEIGHT = 5.0   # your corrections count 5x


def load_corpus():
    d = {}
    for p in ["data/sms.tsv"] + sorted(glob.glob("data/extra_*.tsv")):
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) == 2 and row[0] in ("spam", "ham"):
                    d[message_key(row[1])] = (row[1], row[0], 1.0)
    if os.path.exists("feedback.csv"):
        with open("feedback.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("corrected_label") in ("spam", "ham") and has_text(r.get("message", "")):
                    # link-only feedback is learned by link_feedback.py, not the text models
                    d[message_key(r["message"])] = (r["message"], r["corrected_label"], FEEDBACK_WEIGHT)
    X = [v[0] for v in d.values()]
    y = np.array([1 if v[1] == "spam" else 0 for v in d.values()])
    w = np.array([v[2] for v in d.values()])
    return X, y, w


def main():
    X, y, w = load_corpus()
    print(f"{len(X)} unique messages, {y.sum()} spam")
    Xtr, Xte, ytr, yte, wtr, _ = train_test_split(X, y, w, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipeline().fit(Xtr, ytr, clf__sample_weight=wtr)
    p = pipe.predict_proba(Xte)[:, 1]
    print(f"PR-AUC: {average_precision_score(yte, p):.4f}")
    print(classification_report(yte, p >= 0.5, target_names=["ham", "spam"], digits=4))
    print(confusion_matrix(yte, p >= 0.5))
    prec, rec, thr = precision_recall_curve(yte, p)
    ok = np.where(prec[:-1] >= 0.98)[0]
    t = float(thr[ok[0]]) if len(ok) else 0.5      # lowest threshold with precision >= 98%
    print(f"threshold for >=98% precision: {t:.3f} (recall {rec[ok[0]] if len(ok) else float('nan'):.3f})")
    final = build_pipeline().fit(X, y, clf__sample_weight=w)                # refit on everything for deployment
    with open("text_model.pkl", "wb") as f:
        pickle.dump({"pipeline": final, "threshold": t}, f)
    print("saved text_model.pkl")

if __name__ == "__main__":
    main()