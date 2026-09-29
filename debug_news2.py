"""
debug_news2.py
--------------
Run from your project folder, same venv:

    python debug_news2.py

This turns off yfinance's default exception-hiding for the news call
(so we see the REAL error instead of a silent empty list), and also
hits the underlying endpoint (finance.yahoo.com/xhr/ncp) directly so
we can see exactly what Yahoo is sending back.
"""

import yfinance as yf

# Unhide exceptions -- by default yfinance swallows failures in get_news()
# and just returns [], which is exactly the symptom we're chasing.
yf.config.debug.hide_exceptions = False
yf.config.debug.logging = True

print("=" * 60)
print("1. t.news with exceptions unhidden")
print("=" * 60)
t = yf.Ticker("AAPL")
try:
    news = t.news
    print("news length:", len(news))
except Exception as e:
    print("REAL ERROR:", type(e).__name__, "-", e)

print("\n" + "=" * 60)
print("2. Direct POST to the news endpoint (what yfinance calls internally)")
print("=" * 60)
import requests

url = "https://finance.yahoo.com/xhr/ncp?queryRef=latestNews&serviceKey=ncp_fin"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Content-Type": "application/json",
}
payload = {
    "serviceConfig": {
        "snippetCount": 10,
        "s": ["AAPL"],
    }
}

try:
    r = requests.post(url, headers=headers, json=payload, timeout=10)
    print("status:", r.status_code)
    print("content-type:", r.headers.get("content-type"))
    print("body length:", len(r.content))
    print("body preview (first 500 chars):")
    print(r.text[:500])
except Exception as e:
    print("POST FAILED:", repr(e))

print("\nDone. Paste this whole output back.")
