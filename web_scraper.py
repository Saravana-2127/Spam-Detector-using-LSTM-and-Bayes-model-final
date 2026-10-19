"""
web_scraper.py
Inspects live HTML content for phishing signals, password forms, and platform abuse.
"""

from typing import Dict, List, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
from brand_impersonation import host_of, split_domain

LEGIT_HOSTING_PLATFORMS = [
    "forms.gle", "docs.google.com/forms", "typeform.com", "notion.site",
    "vercel.app", "netlify.app", "firebaseapp.com", "weebly.com",
    "wixsite.com", "github.io", "webflow.io", "blogspot.com", "godaddysites.com"
]

CREDENTIAL_KEYWORDS = [
    "verify account", "confirm password", "update payment", "security alert",
    "unauthorized access", "session expired", "banking login", "wallet seed phrase"
]


def inspect_landing_page(response: requests.Response, final_url: str) -> Dict[str, Any]:
    reasons: List[str] = []
    prob_boost: float = 0.0
    page_title: str = ""
    
    if response is None or getattr(response, "status_code", None) != 200:
        return {
            "reasons": reasons,
            "prob_boost": prob_boost,
            "page_title": page_title,
            "has_password_field": False
        }

    try:
        soup = BeautifulSoup(response.content[:500000], "html.parser")

        if soup.title and soup.title.string:
            page_title = soup.title.string.strip()

        password_inputs = soup.find_all("input", {"type": "password"})
        has_password_field = len(password_inputs) > 0

        host = host_of(final_url)
        path_l = (urlparse(final_url).path or "").lower()
        is_platform_abuse = any(
            host == p.split("/")[0] or host.endswith("." + p.split("/")[0])
            for p in LEGIT_HOSTING_PLATFORMS
        )

        if is_platform_abuse:
            reasons.append(f"hosted on a free/legitimate platform ({final_url})")
            if has_password_field:
                reasons.append("🚨 CRITICAL: Login/password form hosted on a free platform (credential harvesting)")
                prob_boost = max(prob_boost, 0.95)
            else:
                prob_boost = max(prob_boost, 0.40)

        page_reg = split_domain(host)[0]
        for form in soup.find_all("form"):
            target = urljoin(final_url, form.get("action", "") or "")
            tgt_reg = split_domain(host_of(target))[0]
            sensitive = form.find("input", {"type": "password"}) is not None
            if tgt_reg and tgt_reg != page_reg and sensitive:
                reasons.append(f"password form submits to a different domain ({tgt_reg})")
                prob_boost = max(prob_boost, 0.85)

        body_text = soup.get_text().lower()
        matched_keywords = [kw for kw in CREDENTIAL_KEYWORDS if kw in body_text]
        if matched_keywords and (has_password_field or len(matched_keywords) >= 2):
            reasons.append(f"page text contains urgent credential prompts: {', '.join(matched_keywords[:3])}")
            prob_boost = max(prob_boost, 0.65)

        return {
            "reasons": reasons,
            "prob_boost": prob_boost,
            "page_title": page_title,
            "has_password_field": has_password_field
        }

    except Exception as err:
        return {
            "reasons": [f"could not scrape page content ({str(err)})"],
            "prob_boost": 0.0,
            "page_title": "",
            "has_password_field": False
        }