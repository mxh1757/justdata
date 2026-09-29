"""
debug_news.py
-------------
Run this directly from your project folder (same venv as the dashboard):

    python debug_news.py

Paste the full output back. It checks, in order:
  1. yfinance version
  2. Whether basic yfinance calls work at all (history/info)
  3. Raw t.news output (length + first item, if any)
  4. Direct HTTP reachability to the two Yahoo hosts yfinance depends on
  5. Whether a browser-like User-Agent changes anything
"""

import sys

print("=" * 60)
print("1. yfinance version")
print("=" * 60)
try:
    import yfinance as yf
    print("yfinance:", yf.__version__)
except Exception as e:
    print("FAILED to import yfinance:", e)
    sys.exit(1)

TICKER = "AAPL"

print("\n" + "=" * 60)
print(f"2. Basic yfinance calls for {TICKER}")
print("=" * 60)
t = yf.Ticker(TICKER)

try:
    hist = t.history(period="5d")
    print("history() rows:", len(hist))
except Exception as e:
    print("history() FAILED:", repr(e))

try:
    info = t.info
    print("info longName:", info.get("longName", "N/A") if info else "EMPTY DICT")
except Exception as e:
    print("info FAILED:", repr(e))

print("\n" + "=" * 60)
print(f"3. Raw t.news for {TICKER}")
print("=" * 60)
try:
    news = t.news
    print("type:", type(news))
    print("length:", len(news) if news else 0)
    if news:
        print("first item keys:", list(news[0].keys()))
        print("first item:", news[0])
    else:
        print("news is empty/falsy:", repr(news))
except Exception as e:
    print("t.news FAILED:", repr(e))

print("\n" + "=" * 60)
print("4. Direct HTTP reachability (plain requests, default UA)")
print("=" * 60)
import requests

for host in [
    "https://query1.finance.yahoo.com",
    "https://query2.finance.yahoo.com",
]:
    try:
        r = requests.get(host, timeout=8)
        print(f"{host} -> status {r.status_code}, {len(r.content)} bytes")
    except Exception as e:
        print(f"{host} -> FAILED: {repr(e)}")

print("\n" + "=" * 60)
print("5. Same, with a browser-like User-Agent")
print("=" * 60)
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"}
for host in [
    "https://query1.finance.yahoo.com/v1/finance/search?q=AAPL",
    "https://query2.finance.yahoo.com/v1/finance/search?q=AAPL",
]:
    try:
        r = requests.get(host, headers=headers, timeout=8)
        print(f"{host} -> status {r.status_code}, {len(r.content)} bytes")
        print("  body preview:", r.text[:200].replace("\n", " "))
    except Exception as e:
        print(f"{host} -> FAILED: {repr(e)}")

print("\n" + "=" * 60)
print("6. curl_cffi impersonation (only if installed)")
print("=" * 60)
try:
    from curl_cffi import requests as cffi_requests
    session = cffi_requests.Session(impersonate="chrome")
    t2 = yf.Ticker(TICKER, session=session)
    news2 = t2.news
    print("curl_cffi news length:", len(news2) if news2 else 0)
except ImportError:
    print("curl_cffi not installed -- skip (pip install curl_cffi to test this path)")
except Exception as e:
    print("curl_cffi attempt FAILED:", repr(e))

print("\nDone. Paste this whole output back.")
