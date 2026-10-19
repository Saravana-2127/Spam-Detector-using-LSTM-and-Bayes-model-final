"""
Flask web app — paste a message (with links) -> SPAM/HAM verdict + reasons; correct it -> it learns.

Run order:  python train_all.py   (once, and again whenever you add data)   then   python app.py
"""
import threading
from flask import Flask, render_template, request

import ensemble
import link_feedback
import text_model
from feedback_store import log_feedback, feedback_count, corrected_label_for
from link_analysis import analyze_links as _analyze_links
from link_scan import extract_urls
from lstm_model import load_lstm, predict_lstm_prob
from naive_bayes import NaiveBayesSpamClassifier
from risk import risk_level

app = Flask(__name__)

nb_model = NaiveBayesSpamClassifier.load("naive_bayes_model.pkl")
lstm_bundle = load_lstm("lstm_model.pt")
text_model.load("text_model.pkl")
meta = ensemble.load("ensemble.pkl")
nb_lock = threading.Lock()          # Flask runs threaded=True; NB gets updated by feedback


def build_result(message):
    words = text_model.has_text(message)
    if words:
        with nb_lock:
            nb_prob = nb_model.predict_proba(message).get("spam", 0.0)
        lstm_prob = predict_lstm_prob(lstm_bundle, message)
        lr_prob = text_model.predict_spam_prob(message)
        text_prob = ensemble.predict(meta, nb_prob, lstm_prob, lr_prob)
    else:                       # only a link: text models have nothing to judge
        nb_prob = lstm_prob = lr_prob = text_prob = 0.0

    link_results = _analyze_links(message, fetch=True, max_urls=5)
    worst_link = max((r["malicious_prob"] / 100 for r in link_results), default=0.0)
    combined = max(text_prob, worst_link)

    remembered = corrected_label_for(message)       # you already told us the answer for this one
    if remembered:
        combined = 0.99 if remembered == "spam" else 0.01

    risk_score = round(combined * 100, 1)
    reasons = text_model.explain(message, top_n=6) if words else []

    # plain-English summary of WHY
    parts = []
    if remembered:
        parts.append(f"You previously marked this message as {remembered.upper()}.")
    else:
        if not words:
            parts.append("Message is only a link, so the verdict comes from link analysis alone.")
        spammy = [r["word"] for r in reasons if r["direction"] == "spam-like"][:4]
        hammy = [r["word"] for r in reasons if r["direction"] == "ham-like"][:3]
        if spammy and text_prob >= 0.3:
            parts.append("Spam-like wording: " + ", ".join(spammy) + ".")
        if hammy and text_prob < 0.5:
            parts.append("Normal wording: " + ", ".join(hammy) + ".")
        bad = [l for l in link_results if l["label"] == "MALICIOUS"]
        for l in bad[:2]:
            parts.append(f"Link {l['url']} looks malicious: " + "; ".join(l["reasons"][:2]) + ".")
        if worst_link > text_prob and bad:
            parts.append("The verdict is driven mainly by the link, not the text.")
        if not parts or (not words and len(parts) == 1 and not link_results):
            parts.append("No spam wording or suspicious links found.")
        elif not words and not any(l["label"] == "MALICIOUS" for l in link_results):
            parts.append("The link looks fine.")

    return {
        "message": message,
        "nb_prob": round(nb_prob * 100, 2),
        "lstm_prob": round(lstm_prob * 100, 2),
        "lr_prob": round(lr_prob * 100, 2),
        "text_prob": round(text_prob * 100, 2),
        "nb_label": "SPAM" if nb_prob >= 0.5 else "HAM",
        "lstm_label": "SPAM" if lstm_prob >= 0.5 else "HAM",
        "lr_label": "SPAM" if lr_prob >= 0.5 else "HAM",
        "risk_score": risk_score,
        "level": risk_level(risk_score),
        "reasons": reasons,
        "summary": " ".join(parts),
        "remembered": remembered,
        "links": link_results,
        "predicted_label": "spam" if risk_score >= 50 else "ham",
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result, message = None, ""
    if request.method == "POST":
        message = request.form.get("message", "")
        if message.strip():
            result = build_result(message)
    return render_template("index.html", result=result, message=message, feedback_count=feedback_count())


@app.route("/feedback", methods=["POST"])
def feedback():
    message = request.form.get("message", "")
    predicted_label = request.form.get("predicted_label", "")
    corrected_label = request.form.get("corrected_label", "")

    just_learned = False
    if message.strip() and corrected_label in ("spam", "ham"):
        try:
            log_feedback(message, predicted_label, corrected_label)               # 1. remember it
            if text_model.has_text(message):                                      # words to learn from?
                with nb_lock:                                                     # 2. NB: exact instant update
                    nb_model.partial_fit(message, corrected_label)
                    try:
                        nb_model.save("naive_bayes_model.pkl")
                    except Exception as err:
                        print(f"[Warning] Could not save naive_bayes_model.pkl: {err}")
                text_model.retrain_async()                                        # 4. main model (background)
            link_feedback.record(extract_urls(message), corrected_label)          # 3. link domains
            just_learned = True
        except Exception as err:
            print(f"[Error] Adaptive learning step failed: {err}")

    try:
        result = build_result(message) if message.strip() else None
    except Exception as err:
        print(f"[Error] Could not rebuild result: {err}")
        result = None

    return render_template("index.html", result=result, message=message,
                           feedback_count=feedback_count(), just_learned=just_learned)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000, threaded=True)