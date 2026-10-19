"""
text_model.py — TF-IDF (word + char n-grams) + calibrated Logistic Regression.
Shared preprocess() MUST be used for training AND inference (v1 stripped URLs only at
inference, but trained on raw text -> train/serve mismatch).
"""
import os, re, pickle, threading
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from link_scan import mask_urls, extract_urls

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
_MONEY = re.compile(r"(?:₹|rs\.?|inr|\$|£|€)\s*\d[\d,]*(?:\.\d+)?", re.I)
_LONGNUM = re.compile(r"\d{5,}")


def preprocess(text):
    t = mask_urls(text or "", " urltok ")
    t = _EMAIL.sub(" emailtok ", t)
    t = _PHONE.sub(" phonetok ", t)
    t = _MONEY.sub(" moneytok ", t)
    t = _LONGNUM.sub(" longnumtok ", t)
    return t.lower()


def message_key(message):
    """Identity of a message = its words + the registered domains of its links.
    (Link-only messages like 'facebook.com' vs 'facboxx.in' must NOT collide.)"""
    from brand_impersonation import host_of, split_domain
    doms = sorted({split_domain(host_of(u))[0] for u in extract_urls(message)})
    return " ".join(preprocess(message).split()) + "||" + ",".join(doms)


def has_text(message):
    """True if there are real words besides links/numbers (else only the link analyser can judge it)."""
    words = [w for w in re.findall(r"[a-z']{2,}", preprocess(message)) if w not in
             {"urltok", "phonetok", "moneytok", "emailtok", "longnumtok"}]
    return len(words) >= 2


def build_pipeline():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=preprocess, token_pattern=r"[a-z']{2,}",
                                 ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=preprocess, analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=3, sublinear_tf=True, max_features=200_000)),
    ])
    lr = LogisticRegression(C=10, class_weight="balanced", max_iter=3000)
    return Pipeline([("feats", feats), ("clf", CalibratedClassifierCV(lr, cv=5, method="sigmoid"))])


_bundle = None

def load(path="text_model.pkl"):
    global _bundle
    with open(path, "rb") as f:
        _bundle = pickle.load(f)
    return _bundle

def predict_spam_prob(text):
    if _bundle is None:
        load()
    return float(_bundle["pipeline"].predict_proba([text])[0][1])


# ---------------- explanations ----------------
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
_PRETTY = {"urltok": "[a link]", "phonetok": "[phone number]", "moneytok": "[money amount]",
           "emailtok": "[email address]", "longnumtok": "[long number]"}
_xcache = {}

def _explain_assets(pipe):
    if _xcache.get("id") != id(pipe):
        clf = pipe.named_steps["clf"]
        coefs = []
        for cc in clf.calibrated_classifiers_:
            est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator")
            coefs.append(est.coef_.ravel())
        _xcache.update(id=id(pipe), coef=np.mean(coefs, axis=0),
                       names=pipe.named_steps["feats"].get_feature_names_out())
    return _xcache["coef"], _xcache["names"]


def explain(text, top_n=6):
    """Which words/phrases pushed the main model toward spam (+) or ham (-)."""
    if _bundle is None:
        load()
    pipe = _bundle["pipeline"]
    coef, names = _explain_assets(pipe)
    row = pipe.named_steps["feats"].transform([text]).multiply(coef).tocoo()
    items = [(names[j][6:], v) for j, v in zip(row.col, row.data) if names[j].startswith("word__")]
    items = [(w, v) for w, v in items if not all(p in ENGLISH_STOP_WORDS for p in w.split())]
    items.sort(key=lambda t: abs(t[1]), reverse=True)
    pretty = lambda w: " ".join(_PRETTY.get(p, p) for p in w.split())
    return [{"word": pretty(w), "direction": "spam-like" if v > 0 else "ham-like",
             "strength": round(abs(float(v)), 2)} for w, v in items[:top_n]]


# ---------------- learn from feedback (background retrain) ----------------
_lock = threading.Lock()
_state = {"running": False, "again": False}

def retrain_async(path="text_model.pkl"):
    """Retrain on sms + extra data + ALL feedback (feedback weighted x5). Never blocks a request."""
    with _lock:
        if _state["running"]:
            _state["again"] = True
            return
        _state["running"] = True
    threading.Thread(target=_worker, args=(path,), daemon=True).start()

def _worker(path):
    global _bundle
    try:
        while True:
            from train_text import load_corpus
            X, y, w = load_corpus()
            pipe = build_pipeline()
            pipe.fit(X, y, clf__sample_weight=w)
            new = {"pipeline": pipe, "threshold": (_bundle or {}).get("threshold", 0.5)}
            with open(path + ".tmp", "wb") as f:
                pickle.dump(new, f)
            os.replace(path + ".tmp", path)
            _bundle = new
            print("[feedback] text model retrained")
            with _lock:
                if _state["again"]:
                    _state["again"] = False
                    continue
                _state["running"] = False
                return
    except Exception as err:
        print(f"[Warning] background retrain failed: {err}")
        with _lock:
            _state["running"] = False