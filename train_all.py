"""
python train_all.py     — trains Naive Bayes, LSTM, TF-IDF+LR, and the stacked ensemble.
70% train / 15% validation (fits the ensemble weights) / 15% test (reported). Then refits
everything on ALL data (sms + data/extra_*.tsv + your feedback) and saves for the app.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

import ensemble
from train_text import load_corpus
from text_model import build_pipeline
from naive_bayes import NaiveBayesSpamClassifier
from lstm_model import fit_lstm, save_lstm, predict_lstm_prob


def lab(y): return ["spam" if v else "ham" for v in y]

def fit_models(X, y, w, verbose=True):
    nb = NaiveBayesSpamClassifier().fit(X, lab(y))
    lr = build_pipeline().fit(X, y, clf__sample_weight=w)
    lstm = fit_lstm(X, y, verbose=verbose)
    return nb, lr, lstm

def probs(models, X):
    nb, lr, lstm = models
    return (np.array([nb.predict_proba(m).get("spam", 0.0) for m in X]),
            np.array([predict_lstm_prob(lstm, m) for m in X]),
            lr.predict_proba(X)[:, 1])

def report(name, y, p):
    pr = p >= 0.5
    print(f"{name:22} P={precision_score(y, pr):.3f}  R={recall_score(y, pr):.3f}  "
          f"F1={f1_score(y, pr):.3f}  PR-AUC={average_precision_score(y, p):.3f}  FP={int(((pr==1)&(y==0)).sum())} FN={int(((pr==0)&(y==1)).sum())}")


PROBES = [  # modern/Indian-style messages the 2005 SMS data never saw
    ("spam", "Dear customer your SBI account will be blocked today. Update KYC now http://sbi-kyc-update.xyz/login"),
    ("spam", "Your electricity power will be disconnected tonight 9:30PM. Call 9876543210 immediately to update bill"),
    ("spam", "Earn Rs 5000 daily working from home part time. Message on WhatsApp +919812345678 now"),
    ("spam", "Your Amazon order is on hold. Verify your payment details: amaz0n-verify.tk/confirm"),
    ("spam", "Congratulations you have won iPhone 15 Pro. Claim reward at bit.ly/3xYz12 within 2 hours"),
    ("spam", "HDFC Bank: Your PAN card is not updated. Account will be suspended. Click hdfc-pan-update.top"),
    ("ham", "Your OTP for login is 482913. Do not share it with anyone."),
    ("ham", "Hey are we still on for the lab at 5? Bring the Verilog sheet"),
    ("ham", "Your Swiggy order will arrive in 10 minutes. Track here: https://www.swiggy.com/track/12345"),
    ("ham", "Rs 450.00 debited from A/c XX1234 on 18-Oct at Zomato. Avl bal Rs 10,230.55"),
    ("ham", "Dont forget assignment submission on vtop tomorrow"),
]

def main():
    X, y, w = load_corpus()
    X = np.array(X, dtype=object)
    print(f"{len(X)} unique messages, {y.sum()} spam")
    Xtr, Xtmp, ytr, ytmp, wtr, _ = train_test_split(X, y, w, test_size=0.3, stratify=y, random_state=42)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=42)

    print("\nTraining on 70% ...")
    models = fit_models(list(Xtr), ytr, wtr)
    nb_v, ls_v, lr_v = probs(models, list(Xva))
    meta = ensemble.fit(nb_v, ls_v, lr_v, yva)
    print("ensemble weights (NB, LSTM, LR):", np.round(meta.weights, 2))

    nb_t, ls_t, lr_t = probs(models, list(Xte))
    ens_t = ensemble.predict_many(meta, nb_t, ls_t, lr_t)
    print(f"\n=== held-out TEST ({len(yte)} msgs, never used for training or weighting) ===")
    report("Naive Bayes", yte, nb_t); report("LSTM", yte, ls_t)
    report("TF-IDF + LR", yte, lr_t); report("ENSEMBLE (used)", yte, ens_t)
    print(confusion_matrix(yte, ens_t >= 0.5))

    print("\nRefitting on ALL data for deployment ...")
    nb, lr, lstm = fit_models(list(X), y, w, verbose=False)
    nb.save("naive_bayes_model.pkl")
    save_lstm(*lstm)
    import pickle
    with open("text_model.pkl", "wb") as f:
        pickle.dump({"pipeline": lr, "threshold": 0.5}, f)
    ensemble.save(meta)

    print("\n=== sanity probes (modern messages) ===")
    pn, pl, pr_ = probs((nb, lr, lstm), [t for _, t in PROBES])
    pe = ensemble.predict_many(meta, pn, pl, pr_)
    for (truth, t), p in zip(PROBES, pe):
        ok = "OK " if (p >= 0.5) == (truth == "spam") else "MISS"
        print(f"  {ok} true={truth:4} p={p:.2f} | {t[:70]}")
    print("\nSaved: naive_bayes_model.pkl lstm_model.pt text_model.pkl ensemble.pkl")

if __name__ == "__main__":
    main()
