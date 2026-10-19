"""
link_scan.py  (v2)
Extract candidate URLs from raw message text.

Fixes vs v1:
- v1 matched "e.g", "10.50", "Mr.Smith", "report.pdf", emails etc. as URLs and then
  *fetched* them. Every bogus match could push the whole message to MALICIOUS
  (app.py takes max() over links). Now a scheme-less match must end in a real TLD.
- strip_urls() used str.replace, which left "/abc" behind when "bit.ly" matched before
  "bit.ly/abc". Now replaces by span.
"""
import re

TLDS = {
    "com", "net", "org", "in", "co", "io", "me", "app", "info", "biz", "edu", "gov",
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "live", "online",
    "site", "shop", "store", "club", "vip", "work", "loan", "win", "review", "stream",
    "download", "date", "party", "trade", "icu", "buzz", "cyou", "fun", "life", "one",
    "ly", "gl", "gle", "to", "cc", "us", "uk", "ai", "tv", "ws", "dev", "page", "sbi",
    "ru", "cn", "de", "fr", "au", "ca", "sg", "ae", "pk", "bd", "lk", "np", "nl", "br",
}

_HOST = r"(?:(?:\d{1,3}\.){3}\d{1,3}|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24})"
URL_PATTERN = re.compile(
    rf"(?i)(?<![\w@.\-])(?P<scheme>https?://)?(?P<www>www\.)?(?P<host>{_HOST})(?::\d+)?(?:/[^\s]*)?"
)


def _iter_urls(text):
    for m in URL_PATTERN.finditer(text):
        raw = m.group(0)
        cleaned = raw.rstrip(".,!?;:)'\"]")
        host = m.group("host").lower()
        is_ip = bool(re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", host))
        has_marker = bool(m.group("scheme") or m.group("www"))
        tld = host.rsplit(".", 1)[-1]
        if not (has_marker or is_ip or tld in TLDS):
            continue
        if len(cleaned) < 5:
            continue
        yield m.start(), m.start() + len(cleaned), cleaned


def extract_urls(text):
    seen, out = set(), []
    for _, _, u in _iter_urls(text):
        if u.lower() not in seen:
            seen.add(u.lower())
            out.append(u)
    return out


def mask_urls(text, token=" urltok "):
    spans = list(_iter_urls(text))
    for s, e, _ in reversed(spans):
        text = text[:s] + token + text[e:]
    return text


def strip_urls(text):
    return mask_urls(text, " ")


if __name__ == "__main__":
    for t in ["Claim prize at bit.ly/x7f2Ks9qLp now", "pay Rs.100.50 e.g. today Mr.Smith",
              "mail me a@b.com or see report.pdf", "go to http://192.168.1.5/login",
              "secure-login-paypal-verify.tk/account/update, ok?"]:
        print(t, "->", extract_urls(t))