"""
link_feedback.py — domain-level memory from your corrections.
Mark a message spam -> its link domains are blocked; mark it ham -> they're trusted
(link model score capped; brand-impersonation & page-scan checks still apply).
URL shorteners are never trusted/blocked as a whole (bit.ly hides who's behind it).
"""
import json, os, threading
from brand_impersonation import host_of, split_domain, KNOWN_BRANDS
from url_features import URL_SHORTENERS

PATH = "link_feedback.json"
_lock = threading.Lock()

BUILTIN_TRUSTED = {
    "google.com", "youtube.com", "wikipedia.org", "github.com", "microsoft.com", "apple.com",
    "amazon.com", "amazon.in", "flipkart.com", "swiggy.com", "zomato.com", "linkedin.com",
    "facebook.com", "instagram.com", "whatsapp.com", "x.com", "twitter.com", "paytm.com",
    "phonepe.com", "irctc.co.in", "uber.com", "olacabs.com", "netflix.com", "spotify.com",
    "zoom.us", "stackoverflow.com", "python.org", "pytorch.org", "vit.ac.in",
    # mail / big tech / reference
    "gmail.com", "googlemail.com", "outlook.com", "hotmail.com", "live.com", "office.com", "yahoo.com",
    "icloud.com", "proton.me", "protonmail.com", "zoho.com", "bing.com", "msn.com", "adobe.com",
    "dropbox.com", "slack.com", "notion.so", "canva.com", "openai.com", "anthropic.com", "claude.ai",
    "reddit.com", "quora.com", "medium.com", "wikimedia.org", "gitlab.com", "mozilla.org", "cloudflare.com",
    "stackexchange.com", "w3schools.com", "geeksforgeeks.org", "kaggle.com", "arxiv.org", "ieee.org",
    "sciencedirect.com", "springer.com", "nature.com", "tensorflow.org", "coursera.org", "udemy.com",
    "edx.org", "nptel.ac.in", "imdb.com", "spotify.com", "primevideo.com", "hotstar.com", "bbc.com",
    "nytimes.com", "thehindu.com", "ndtv.com", "indiatimes.com", "hindustantimes.com",
    # india: banks / payments / shopping / travel / telecom / jobs
    "sbi.co.in", "onlinesbi.sbi", "hdfcbank.com", "icicibank.com", "axisbank.com", "kotak.com",
    "razorpay.com", "npci.org.in", "bhimupi.org.in", "myntra.com", "bigbasket.com", "blinkit.com",
    "nykaa.com", "makemytrip.com", "goibibo.com", "redbus.in", "jio.com", "airtel.in", "bsnl.co.in",
    "naukri.com", "indeed.com", "linkedin.com", "meesho.com", "ajio.com", "cred.club", "zerodha.com",
} | {split_domain(o)[0] for offs in KNOWN_BRANDS.values() for o in offs}


def _read():
    try:
        with open(PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"trusted": [], "blocked": []}


def record(urls, label):
    with _lock:
        d = _read()
        for u in urls:
            host = host_of(u)
            if not host or host in URL_SHORTENERS:
                continue
            reg = split_domain(host)[0]
            add, rem = ("blocked", "trusted") if label == "spam" else ("trusted", "blocked")
            if reg not in d[add]: d[add].append(reg)
            if reg in d[rem]: d[rem].remove(reg)
        with open(PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)


def status(url):
    """-> ('blocked'|'trusted'|None, why)"""
    host = host_of(url)
    if not host or host in URL_SHORTENERS:
        return None, None
    reg = split_domain(host)[0]
    d = _read()
    if reg in d["blocked"]:
        return "blocked", "you previously marked a message with this domain as spam"
    if reg in d["trusted"]:
        return "trusted", "you previously marked a message with this domain as safe"
    if (host == reg or host == "www." + reg) and reg in BUILTIN_TRUSTED:
        return "trusted", "well-known legitimate domain"
    if reg.endswith((".gov.in", ".nic.in", ".ac.in", ".edu.in", ".edu", ".gov")):
        return "trusted", "well-known legitimate domain"
    return None, None