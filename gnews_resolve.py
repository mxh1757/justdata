"""
gnews_resolve.py
-----------------
Best-effort resolution of Google News RSS redirect links
(https://news.google.com/rss/articles/...) back to the original
publisher URL, so article_fetch.py can pull real article text for
headlines that came from the Google News RSS fallback in dashboard.py.

Why this is its own small module, not folded into article_fetch.py:
Google's redirect scheme is an internal, undocumented mechanism -- it
already changed once (pre-2024 it was an offline base64 decode; now it
requires a live request to Google's internal "batchexecute" RPC
instead). This can break again without notice, same category of risk
as the yfinance news endpoint. Isolating it here makes that risk
explicit and keeps it swappable/removable on its own.

Every function here degrades gracefully: on any failure (package
missing, network error, rate limit, Google changing the format again),
resolve() returns the original URL unchanged rather than raising -- so
a broken decoder never takes down article summarization. It just means
Google-News-sourced headlines fall back to the same headline-only
summary behavior that existed before this module.

Setup (optional -- everything works without this, just with fewer full
article summaries for Google-News-sourced headlines):
    pip install googlenewsdecoder
"""

try:
    from googlenewsdecoder import gnewsdecoder
    _DECODER_AVAILABLE = True
except ImportError:
    _DECODER_AVAILABLE = False


def is_google_news_link(url: str) -> bool:
    return bool(url) and "news.google.com" in url


def resolve(url: str, interval: float = 1.0) -> str:
    """
    Resolve a Google News RSS redirect link to its original publisher
    URL. Returns the input unchanged if:
      - it's not a Google News link (nothing to do)
      - the googlenewsdecoder package isn't installed (optional dep)
      - resolution fails for any reason (rate limit, format change,
        network error, timeout)

    `interval` is a small delay Google's internal endpoint expects
    between requests to avoid rate limiting -- only relevant when
    resolving several links back-to-back.
    """
    if not is_google_news_link(url):
        return url

    if not _DECODER_AVAILABLE:
        return url

    try:
        result = gnewsdecoder(url, interval=interval)
        if result and result.get("status") and result.get("decoded_url"):
            return result["decoded_url"]
    except Exception as e:
        print(f"[warn] Google News link resolution failed for {url}: {e}")

    return url
