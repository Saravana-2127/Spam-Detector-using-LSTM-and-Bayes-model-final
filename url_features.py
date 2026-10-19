"""
URL feature extraction for link/phishing detection.

This is a DIFFERENT technique from the text-based Naive Bayes / LSTM models.
Those work on ENGLISH WORDS (word frequency, word sequence). A URL is not
natural-language text — "bit.ly/x7f2Ks" has no "words" in the linguistic
sense. Instead, we extract LEXICAL / STRUCTURAL features: measurable
properties of the URL's shape and composition that tend to differ between
legitimate and malicious links. This is the standard approach used in
phishing-URL research (see PhishStorm, PhiUSIIL and similar published work).

Each feature below is also individually meaningful, which is what lets us
give a human-readable "reason" for a verdict (explainability), not just a
black-box score.
"""

import re
import math
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "work", "loan",
    "win", "review", "country", "stream", "gdn", "mom", "party", "trade",
    "date", "faith", "science", "accountant", "download",
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorte.st", "adf.ly", "bl.ink",
}

SUSPICIOUS_KEYWORDS = {
    "login", "verify", "secure", "account", "update", "confirm", "signin",
    "banking", "password", "suspended", "unlock", "billing", "webscr",
    "urgent", "limited", "click", "claim", "reward", "prize", "gift",
}


def _ensure_scheme(url):
    """Many URLs in the training data have no http:// prefix. Add one so
    urlparse can split host/path correctly."""
    if not re.match(r"^[a-zA-Z]+://", url):
        return "http://" + url
    return url


def _shannon_entropy(s):
    """Higher entropy = more random-looking string, common in generated
    malicious subdomains/paths (e.g. 'x7f2Ks9qLp')."""
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def _has_ip_address(host):
    return bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", host))


def extract_features(raw_url):
    """
    Returns a dict of numeric features (for the ML model) — call
    extract_reasons() separately to get human-readable flags for display.
    """
    url = _ensure_scheme(raw_url.strip())
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = parsed.path or ""
        scheme = parsed.scheme
    except ValueError:
        # Some real-world malformed/malicious URLs (e.g. broken IPv6-looking
        # brackets) aren't valid enough for urlparse to split cleanly.
        # Fall back to treating the whole string as an opaque blob — it's
        # still perfectly usable for length/character-based features, and a
        # URL that can't even be parsed is itself a mild red flag.
        host = ""
        path = url
        scheme = ""

    full = url.lower()

    domain_parts = host.split(".")
    tld = domain_parts[-1] if len(domain_parts) > 1 else ""
    num_subdomains = max(0, len(domain_parts) - 2)

    features = {
        "url_length": len(url),
        "host_length": len(host),
        "num_dots": full.count("."),
        "num_hyphens": full.count("-"),
        "num_at": full.count("@"),
        "num_digits": sum(c.isdigit() for c in full),
        "num_subdomains": num_subdomains,
        "has_ip": int(_has_ip_address(host)),
        "has_https": int(scheme == "https"),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "is_shortener": int(host in URL_SHORTENERS),
        "num_suspicious_keywords": sum(1 for k in SUSPICIOUS_KEYWORDS if k in full),
        "path_entropy": _shannon_entropy(path),
        "double_slash_in_path": int("//" in path),
        "digit_ratio": (sum(c.isdigit() for c in full) / len(full)) if full else 0,
    }
    return features


def extract_reasons(raw_url):
    """
    Human-readable, rule-level explanations (feature 2: explainability),
    independent of the trained model's internal weights — these are the
    same red flags a security-aware human would look for.
    """
    f = extract_features(raw_url)
    reasons = []

    if f["has_ip"]:
        reasons.append("uses a raw IP address instead of a domain name")
    if f["suspicious_tld"]:
        reasons.append("uses a domain ending commonly abused for spam/phishing")
    if f["is_shortener"]:
        reasons.append("uses a URL shortener, which hides the real destination")
    if f["num_at"] > 0:
        reasons.append("contains '@', a known trick to disguise the real destination")
    if f["num_subdomains"] >= 3:
        reasons.append("has an unusually large number of subdomains")
    if f["num_suspicious_keywords"] >= 2:
        reasons.append("contains multiple urgency/credential-related keywords (e.g. verify, login, secure)")
    if f["url_length"] > 75:
        reasons.append("is unusually long")
    if f["double_slash_in_path"]:
        reasons.append("contains a suspicious double-slash redirect pattern in the path")
    if not f["has_https"]:
        reasons.append("does not use HTTPS")

    return reasons


if __name__ == "__main__":
    examples = [
        "http://192.168.1.5/login/verify-account",
        "https://www.google.com/search?q=probability+theory",
        "bit.ly/x7f2Ks9qLp",
        "secure-login-paypal-verify.tk/account/update",
    ]
    for url in examples:
        print(url)
        print("  features:", extract_features(url))
        print("  reasons: ", extract_reasons(url))
        print()
