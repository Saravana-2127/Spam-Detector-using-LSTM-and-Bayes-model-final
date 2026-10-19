"""
brand_impersonation.py  (v2)
Fixes vs v1:
- v1 used `brand in label` substring matching: "vit" matched visit/invite/vitamins,
  "apple" matched applebees, etc. Now matches whole hyphen-separated TOKENS.
- v1 took label = first dot-part, so "paypal.com.evil.tk" gave label "paypal" but
  "login.paypal.com.evil.tk" gave "login" (missed). Now uses the real registered domain
  and checks subdomains separately.
- difflib ratio on the whole host (threshold 0.55) was far too loose. Now edit distance
  on the registered-domain label, with leetspeak (0->o, 1->l) and rn->m normalisation.
- Title check required only that the brand word appeared in <title> -> any news article
  mentioning Google got 0.90. Now also requires a password field on the page.
"""
import re
from urllib.parse import urlparse

KNOWN_BRANDS = {
    "vit": ["vit.ac.in", "vtop.vit.ac.in", "vtopcc.vit.ac.in", "vitap.ac.in", "vitbhopal.ac.in", "vitchennai.ac.in"],
    "google": ["google.com", "google.co.in", "google.in", "accounts.google.com", "docs.google.com"],
    "paypal": ["paypal.com"],
    "amazon": ["amazon.com", "amazon.in", "aws.amazon.com"],
    "microsoft": ["microsoft.com", "live.com", "office.com", "outlook.com"],
    "apple": ["apple.com", "icloud.com"],
    "netflix": ["netflix.com"],
    "instagram": ["instagram.com"],
    "facebook": ["facebook.com", "facebook.in", "fb.com"],
    "linkedin": ["linkedin.com"],
    "sbi": ["onlinesbi.sbi", "sbi.co.in", "sbicard.com"],
    "hdfc": ["hdfcbank.com"],
    "icici": ["icicibank.com"],
}
BRAND_ALIASES = {"vit": ["vtop"], "sbi": ["onlinesbi"], "hdfc": ["hdfcbank"], "icici": ["icicibank"]}
BAIT = ("login", "signin", "secure", "verify", "account", "support", "update", "kyc", "pay",
        "bank", "wallet", "service", "help", "portal", "care", "confirm", "billing", "refund",
        "reward", "offer", "alert", "suspend", "unlock")
MULTI_TLDS = {"co.in", "ac.in", "gov.in", "org.in", "net.in", "edu.in", "co.uk", "com.au", "com.br", "co.jp", "co.za"}
_LEET = str.maketrans("013457$@", "oleastsa")


def host_of(url):
    url = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url) else "http://" + url
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def split_domain(host):
    parts = host.split(".")
    n = 3 if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_TLDS else 2
    if len(parts) < n:
        return host, host, []
    return ".".join(parts[-n:]), parts[-n], parts[:-n]


def is_official(host, officials):
    return any(host == o or host.endswith("." + o) for o in officials)


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _tokens(s):
    return [t for t in re.split(r"[-_.]", s.translate(_LEET)) if t]


def check_brand_impersonation(raw_url):
    no = {"is_impersonation": False, "brand": None, "matched_official": None, "why": None}
    host = host_of(raw_url)
    if not host:
        return no
    reg, label, subs = split_domain(host)
    norm = label.translate(_LEET)
    toks = _tokens(label)
    sub_toks = [t for s in subs for t in _tokens(s)]
    variants = {norm, norm.replace("rn", "m"), norm.replace("vv", "w")}

    for brand, officials in KNOWN_BRANDS.items():
        if is_official(host, officials):
            continue
        names = [brand] + BRAND_ALIASES.get(brand, [])
        why = None
        if any(n in sub_toks for n in names):
            why = "brand name used as a subdomain of an unrelated domain"
        elif any(n in toks for n in names):
            why = "brand name used inside an unrelated domain"
        elif len(brand) >= 5 and brand in norm and any(b in norm for b in BAIT):
            why = "brand name combined with a login/security keyword"
        else:
            for off in {split_domain(o)[1] for o in officials}:
                if len(off) < 5:
                    continue
                thr = 0 if len(off) < 6 else (1 if len(off) < 8 else 2)
                if min(_lev(v, off) for v in variants) <= thr:
                    why = "domain is a lookalike/typo of the real brand domain"
                    break
        if why:
            return {"is_impersonation": True, "brand": brand, "matched_official": officials[0], "why": why}
    return no


def check_title_brand_impersonation(page_title, final_url, has_password_field=False):
    out = {"is_title_impersonation": False, "brand": None}
    if not page_title or not has_password_field:
        return out
    title, host = page_title.lower(), host_of(final_url)
    for brand, officials in KNOWN_BRANDS.items():
        if re.search(rf"\b{re.escape(brand)}\b", title) and not is_official(host, officials):
            return {
                "is_title_impersonation": True, "brand": brand, "official_domain": officials[0],
                "reason": f"login page titled '{page_title}' claims to be {brand.upper()}, "
                          f"but {host} is not an official {brand} domain",
            }
    return out