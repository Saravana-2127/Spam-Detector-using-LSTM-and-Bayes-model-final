"""
url_expander.py
Unrolls shortened URLs and traces full HTTP redirect chains.
"""

from typing import Dict, Any
from urllib.parse import urlparse
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}


def _is_local_url(url: str) -> bool:
    """Check if URL points to localhost or private IP space to prevent deadlocks."""
    try:
        host = (urlparse(url if "://" in url else "http://" + url).netloc or "").split(":")[0].lower()
        if host in LOCAL_HOSTS or host.startswith(("192.168.", "10.", "172.16.", "172.17.", "172.18.", "172.19.")):
            return True
    except Exception:
        pass
    return False


def expand_url_chain(raw_url: str, timeout: int = 3) -> Dict[str, Any]:
    url = raw_url if raw_url.startswith(("http://", "https://")) else "http://" + raw_url
    chain = [url]
    
    if _is_local_url(url):
        return {
            "initial_url": url,
            "final_url": url,
            "redirect_chain": chain,
            "is_redirected": False,
            "response": None,
            "error": "Skipped local IP address check"
        }
    
    try:
        response = requests.get(
            url,
            headers=DEFAULT_HEADERS,
            timeout=timeout,
            allow_redirects=True,
            verify=False
        )
        
        if response.history:
            chain = [resp.url for resp in response.history] + [response.url]
            
        return {
            "initial_url": url,
            "final_url": response.url,
            "redirect_chain": chain,
            "is_redirected": len(chain) > 1,
            "response": response,
            "error": None
        }
        
    except Exception as err:
        return {
            "initial_url": url,
            "final_url": url,
            "redirect_chain": chain,
            "is_redirected": False,
            "response": None,
            "error": str(err)
        }