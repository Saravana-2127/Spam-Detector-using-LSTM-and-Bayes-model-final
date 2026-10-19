"""
link_analysis.py — link scoring shared by app.py and gmail_scanner.py.
fetch=False  -> purely lexical + brand checks, NO network requests. Use this for emails:
fetching links found in mail can trigger one-click unsubscribe/confirm URLs and tells
spammers your address is live.
"""
import pickle
import pandas as pd

from link_scan import extract_urls
from url_features import extract_features, extract_reasons
import link_feedback
from brand_impersonation import check_brand_impersonation, check_title_brand_impersonation

_link_model = _feature_names = None


def _load():
    global _link_model, _feature_names
    if _link_model is None:
        with open("link_model.pkl", "rb") as f:
            b = pickle.load(f)
        _link_model, _feature_names = b["model"], b["feature_names"]


def predict_link(url):
    _load()
    feats = pd.DataFrame([extract_features(url)])[_feature_names]
    return _link_model.predict_proba(feats)[0][1]


def analyze_links(message, fetch=True, urls=None, max_urls=8):
    urls = (urls if urls is not None else extract_urls(message))[:max_urls]
    results = []
    for original_url in urls:
        response, final_url, redirected = None, original_url, False
        if fetch:
            from url_expander import expand_url_chain
            chain = expand_url_chain(original_url)
            final_url, response, redirected = chain["final_url"], chain["response"], chain["is_redirected"]

        prob = predict_link(final_url)
        reasons = extract_reasons(final_url)

        # known-domain knowledge beats a lexical guess (the RF flags short legit homepages)
        st, why = link_feedback.status(final_url)
        if st == "trusted":
            prob = min(prob, 0.05 if why == "well-known legitimate domain" else 0.10)
            reasons = [r for r in reasons if "HTTPS" not in r]
            reasons.append(f"trusted domain: {why}")
        elif st == "blocked":
            prob = max(prob, 0.95)
            reasons.append(f"blocked domain: {why}")
        else:
            # The URL model learned on a dataset where short bare domains skew "bad", so it can't be
            # trusted when the URL shows no red flags at all. Unknown != malicious: cap below the spam line.
            flags = [r for r in reasons if "HTTPS" not in r and "unusually long" not in r]
            if not flags:
                prob = min(prob, 0.45)
                reasons.append("no red-flag patterns in the URL itself (the domain is just not well known)")
        if redirected:
            reasons.append(f"URL uses redirects/shorteners: {original_url} -> {final_url}")

        imp = check_brand_impersonation(final_url)
        if imp["is_impersonation"]:
            prob = max(prob, 0.85)
            reasons.append(f"domain imitates '{imp['brand']}' ({imp['matched_official']}): {imp['why']}")

        if response is not None:
            from web_scraper import inspect_landing_page
            scrape = inspect_landing_page(response, final_url)
            prob = max(prob, scrape["prob_boost"])
            reasons.extend(scrape["reasons"])
            t = check_title_brand_impersonation(scrape["page_title"], final_url, scrape["has_password_field"])
            if t["is_title_impersonation"]:
                prob = max(prob, 0.90)
                reasons.append(t["reason"])

        results.append({
            "url": original_url, "final_url": final_url,
            "malicious_prob": round(float(prob) * 100, 2),
            "label": "MALICIOUS" if prob >= 0.5 else "LOOKS OK",
            "reasons": reasons,
        })
    return results