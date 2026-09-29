"""
dashboard.py
------------
A local web dashboard (Streamlit) with two sections:

  1. Overall Market Sentiment -- a table of recent financial headlines
     (from broad RSS feeds), each scored positive/negative/neutral by FinBERT.

  2. Ticker Lookup -- search any ticker to see price, market cap, volume,
     P/E, revenue, EPS, a price chart, and a locally-generated plain-English
     explanation of that ticker's recent news, plus the underlying headlines.

Data sources (all free, no API key):
  - fetch_data.py (RSS feeds) -> broad market news
  - yfinance -> price, fundamentals, and per-ticker news
  - FinBERT (sentiment_model.py) -> sentiment scoring, runs locally
  - distilbart-cnn (news_summarizer.py) -> news explanation, runs locally

Run it:
    streamlit run dashboard.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import hashlib
import feedparser
from datetime import datetime, timezone

from sentiment_model import FinancialSentimentAnalyzer
from news_summarizer import NewsSummarizer, get_market_impact_note
from fetch_data import fetch_headlines
from article_fetch import fetch_article_paragraphs, fetch_og_image
import gnews_resolve
from llm_summarizer import is_available as llm_available, summarize_article_llm
import storage
import ner
import trends

st.set_page_config(page_title="Market Sentiment Dashboard", layout="wide")


# --- Cached resources / data loaders -----------------------------------

@st.cache_resource(show_spinner="Loading local FinBERT model...")
def load_analyzer():
    return FinancialSentimentAnalyzer()


@st.cache_resource(show_spinner="Loading local summarization model...")
def load_summarizer():
    return NewsSummarizer()


@st.cache_resource(show_spinner="Loading local NER model...")
def load_ner():
    # Triggers spaCy's model load on first use; ner.extract_organizations()
    # lazily loads internally too, so this just warms the cache up front.
    ner.extract_organizations("warmup")
    return ner


@st.cache_data(ttl=86400, show_spinner=False)
def get_cached_og_image(url: str):
    """Cached for a day -- an article's thumbnail image doesn't change."""
    return fetch_og_image(gnews_resolve.resolve(url))


@st.cache_data(ttl=900, show_spinner="Fetching overall market news...")
def get_market_headlines(limit_per_feed: int = 50):
    return fetch_headlines(limit_per_feed=limit_per_feed)


@st.cache_data(ttl=900, show_spinner="Scoring market sentiment...")
def get_scored_market_data(limit_per_feed: int = 50):
    """
    Fetch, score, and persist general market headlines. Cached at the same
    TTL as get_market_headlines so this only actually runs (and only
    writes to storage) once per refresh window, not on every Streamlit
    rerun triggered by button clicks elsewhere on the page.
    """
    headlines = get_market_headlines(limit_per_feed=limit_per_feed)
    if not headlines:
        return [], [], []

    analyzer = load_analyzer()
    titles = [h["title"] for h in headlines]
    results = analyzer.analyze(titles)
    load_ner()
    entities_list = [ner.extract_organizations(t) for t in titles]

    records = [
        {
            "ticker": None,
            "source": h["source"],
            "title": h["title"],
            "link": h.get("link", ""),
            "published": h.get("published", ""),
            "label": r["label"],
            "score": r["score"],
            "entities": ents,
        }
        for h, r, ents in zip(headlines, results, entities_list)
    ]
    storage.save_articles(records)

    return headlines, results, entities_list


@st.cache_data(ttl=300, show_spinner="Fetching ticker data...")
def get_ticker_data(ticker: str):
    t = yf.Ticker(ticker)
    info = t.info or {}
    hist = t.history(period="6mo")
    return info, hist


def _rss_fallback_news(ticker: str, company_name: str = "", limit: int = 10):
    """
    Fallback source for per-ticker news when yfinance's own news endpoint
    fails (as of Sept 2026, Yahoo's underlying xhr/ncp?queryRef=latestNews
    endpoint returns a server-side 500 -- confirmed via direct testing,
    not fixable client-side).

    Queries Google News' RSS search directly for this ticker/company --
    a targeted per-ticker search, not a filter over the generic market
    feed (which only has meaningful hit-rate for large, frequently-
    covered names). Uses feedparser, already a project dependency.

    Returns the same flat shape as the normalized yfinance path, so
    callers don't need to know which source actually served the data.

    Note: Google News RSS links are Google-redirect URLs, not direct
    publisher links -- this can reduce the hit-rate of the OG-image
    thumbnail fetch (article_fetch.fetch_og_image) for these items, since
    it's fetching a Google interstitial page rather than the original
    article page. Not fixable without resolving each redirect (extra
    network round-trip per headline), so left as-is for now.
    """
    from urllib.parse import quote

    query = f"{company_name or ticker} stock"
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"

    try:
        parsed = feedparser.parse(url)
    except Exception as e:
        print(f"[warn] Google News RSS fallback failed for {ticker}: {e}")
        return []

    items = []
    for entry in parsed.entries[:limit]:
        title = getattr(entry, "title", "").strip()
        if not title:
            continue
        link = getattr(entry, "link", "")

        source = "Google News"
        entry_source = getattr(entry, "source", None)
        if entry_source is not None:
            source = getattr(entry_source, "title", None) or getattr(entry_source, "value", None) or source

        published_iso = None
        if getattr(entry, "published_parsed", None):
            dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            published_iso = dt.isoformat()

        items.append(
            {
                "title": title,
                "link": link,
                "publisher": source,
                "providerPublishTime": None,
                "published_iso": published_iso,
            }
        )

    return items


@st.cache_data(ttl=300, show_spinner="Fetching ticker news...")
def get_ticker_news(ticker: str, limit: int = 10, company_name: str = ""):
    """
    Fetch and normalize per-ticker news from yfinance.

    yfinance has changed its news response shape across versions:
      - older versions: flat dict {"title", "link", "publisher", "providerPublishTime"}
      - newer versions: nested under "content" -> {"title", "canonicalUrl": {"url"},
        "provider": {"displayName"}, "pubDate" (ISO string)}

    This normalizes both into one flat shape so the rest of the app
    doesn't need to care which version is installed.

    If yfinance's news endpoint returns nothing (currently returning a
    server-side 500 from Yahoo as of Sept 2026 -- confirmed via direct
    testing, not something we can fix client-side), falls back to
    filtering the existing general-market RSS feed by ticker/company name.
    """
    t = yf.Ticker(ticker)
    try:
        raw = t.news or []
    except Exception as e:
        print(f"[warn] yfinance news fetch failed for {ticker}: {e}")
        raw = []

    if not raw:
        return _rss_fallback_news(ticker, company_name=company_name, limit=limit)

    normalized = []
    for item in raw:
        content = item.get("content", item)  # fall back to flat shape if no "content" key

        title = content.get("title") or item.get("title") or ""
        if not title:
            continue

        link = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or item.get("link")
            or ""
        )
        publisher = (content.get("provider") or {}).get("displayName") or item.get("publisher") or ""

        publish_time = None
        pub_date = content.get("pubDate")
        if pub_date:
            try:
                publish_time = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).timestamp()
            except Exception:
                publish_time = None
        if publish_time is None and item.get("providerPublishTime"):
            publish_time = item["providerPublishTime"]

        normalized.append(
            {
                "title": title,
                "link": link,
                "publisher": publisher,
                "providerPublishTime": publish_time,
            }
        )

    return normalized[:limit]


@st.cache_data(ttl=300, show_spinner="Scoring ticker news...")
def get_scored_ticker_news(ticker: str, limit: int = 10, company_name: str = ""):
    """
    Fetch, score, and persist news for a specific ticker. Cached at the
    same TTL as get_ticker_news so scoring + saving only happens once per
    refresh window per ticker, not on every rerun.
    """
    news_items = get_ticker_news(ticker, limit=limit, company_name=company_name)
    if not news_items:
        return [], [], []

    analyzer = load_analyzer()
    titles = [n.get("title", "") for n in news_items if n.get("title")]
    results = analyzer.analyze(titles) if titles else []
    load_ner()
    entities_list = [ner.extract_organizations(t) for t in titles]

    records = []
    for item, r, ents in zip(news_items, results, entities_list):
        publish_time = item.get("providerPublishTime")
        if publish_time:
            published = datetime.fromtimestamp(publish_time, tz=timezone.utc).isoformat()
        elif item.get("published_iso"):
            # RSS fallback path already gives an ISO string directly.
            published = item["published_iso"]
        else:
            published = ""
        records.append(
            {
                "ticker": ticker,
                "source": item.get("publisher", ""),
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "published": published,
                "label": r["label"],
                "score": r["score"],
                "entities": ents,
            }
        )
    storage.save_articles(records)

    return news_items, results, entities_list


def fmt_large_number(n):
    if n is None:
        return "N/A"
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:,.0f}"


SENTIMENT_ICON = {"positive": "🟢 Positive", "negative": "🔴 Negative", "neutral": "⚪ Neutral"}


# ==========================================================================
# SIDEBAR: Watchlist
# ==========================================================================

st.sidebar.subheader("⭐ My Watchlist")

with st.sidebar.form("add_watchlist_form", clear_on_submit=True):
    new_watch_ticker = st.text_input("Add a ticker", placeholder="e.g. MSFT")
    submitted = st.form_submit_button("Add")
    if submitted and new_watch_ticker.strip():
        storage.add_watchlist_ticker(new_watch_ticker)
        st.rerun()

watchlist_tickers = storage.get_watchlist()

if not watchlist_tickers:
    st.sidebar.caption(
        "No tickers saved yet. Add one above, or use the ☆ button next "
        "to the ticker lookup below."
    )
else:
    for wl_ticker in watchlist_tickers:
        wl_cols = st.sidebar.columns([3, 1])
        with wl_cols[0]:
            if st.button(wl_ticker, key=f"wl_select_{wl_ticker}", width='stretch'):
                st.session_state.ticker_input = wl_ticker
                st.rerun()
        with wl_cols[1]:
            if st.button("✕", key=f"wl_remove_{wl_ticker}", help=f"Remove {wl_ticker}"):
                storage.remove_watchlist_ticker(wl_ticker)
                st.rerun()

st.sidebar.divider()


# ==========================================================================
# SECTION 1: Overall Market Sentiment
# ==========================================================================

st.title("📊 Market Sentiment Dashboard")
st.header("🌐 Overall Market Sentiment")

col_a, col_b = st.columns([3, 1])
with col_b:
    n_rows = st.slider("Headlines to show", min_value=10, max_value=250, value=50, step=10)
    if st.button("Refresh market news"):
        st.cache_data.clear()

headlines, results, entities_list = get_scored_market_data()

if not headlines:
    st.info("No market headlines available right now.")
else:
    rows = []
    for h, r, entities in zip(headlines, results, entities_list):
        try:
            date_str = datetime.fromisoformat(h["published"]).strftime("%Y-%m-%d %H:%M")
        except Exception:
            date_str = h.get("published", "")
        rows.append(
            {
                "label": r["label"],
                "Sentiment": SENTIMENT_ICON.get(r["label"], r["label"]),
                "Confidence": f"{r['score']:.2f}",
                "Title": h["title"],
                "Companies": ", ".join(entities) if entities else "—",
                "Source": h["source"],
                "Date": date_str,
                "Link": h.get("link") or "",
            }
        )

    all_df = pd.DataFrame(rows).sort_values("Date", ascending=False)
    counts = all_df["label"].value_counts()

    if "market_sentiment_filter" not in st.session_state:
        st.session_state.market_sentiment_filter = None

    st.caption("Click a sentiment to filter the table below. Click it again to clear the filter.")
    filter_cols = st.columns(3)
    filter_config = [
        ("positive", "🟢 Positive"),
        ("negative", "🔴 Negative"),
        ("neutral", "⚪ Neutral"),
    ]
    for col, (label, display) in zip(filter_cols, filter_config):
        is_active = st.session_state.market_sentiment_filter == label
        with col:
            if st.button(
                f"{display} — {int(counts.get(label, 0))}",
                key=f"filter_{label}",
                width='stretch',
                type="primary" if is_active else "secondary",
            ):
                st.session_state.market_sentiment_filter = None if is_active else label
                st.rerun()

    active_filter = st.session_state.market_sentiment_filter
    df = all_df[all_df["label"] == active_filter] if active_filter else all_df
    df = df.drop(columns=["label"]).head(n_rows)

    if active_filter:
        st.caption(f"Showing **{active_filter}** headlines only ({len(df)} of {len(all_df)}).")

    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("Article", display_text="🔗 Open"),
        },
    )

st.divider()

# ==========================================================================
# SECTION 2: Ticker Lookup
# ==========================================================================

st.header("🔍 Ticker Lookup")

if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = "AAPL"

lookup_cols = st.columns([5, 1])
with lookup_cols[0]:
    ticker_input = st.text_input(
        "Enter a ticker symbol",
        key="ticker_input",
        help="Stocks: AAPL, TSLA, MSFT. Currency pairs: EURUSD=X, GBPUSD=X. Crypto: BTC-USD.",
    )
ticker = ticker_input.strip().upper()

with lookup_cols[1]:
    st.write("")  # vertical spacer to align button with text input
    st.write("")
    current_watchlist = storage.get_watchlist()
    if ticker in current_watchlist:
        if st.button("⭐ Remove", key="unstar_btn", help="Remove from watchlist"):
            storage.remove_watchlist_ticker(ticker)
            st.rerun()
    elif ticker:
        if st.button("☆ Add", key="star_btn", help="Add to watchlist"):
            storage.add_watchlist_ticker(ticker)
            st.rerun()

if ticker:
    info, hist = get_ticker_data(ticker)

    if not info or hist.empty:
        st.error(f"Couldn't find data for '{ticker}'. Check the symbol and try again.")
    else:
        name = info.get("longName") or info.get("shortName") or ticker
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        prev_close = info.get("previousClose")
        change = change_pct = None
        if current_price is not None and prev_close:
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100

        st.subheader(f"{name} ({ticker})")

        if current_price is not None:
            price_str = f"${current_price:,.2f}"
            delta_str = f"{change:+.2f} ({change_pct:+.2f}%)" if change is not None else None
            st.metric("Current Price", price_str, delta_str)

        cols = st.columns(5)
        cols[0].metric("Market Cap", fmt_large_number(info.get("marketCap")))
        cols[1].metric("Volume", fmt_large_number(info.get("volume")))
        cols[2].metric("P/E Ratio", f"{info.get('trailingPE'):.2f}" if info.get("trailingPE") else "N/A")
        cols[3].metric("Revenue (TTM)", fmt_large_number(info.get("totalRevenue")))
        cols[4].metric("EPS (trailing)", f"{info.get('trailingEps'):.2f}" if info.get("trailingEps") else "N/A")

        # Price history chart
        st.markdown("**Price History (6 months)**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close"))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Price ($)")
        st.plotly_chart(fig, width='stretch')

        # News + explanation + sentiment
        st.markdown("**Recent News & Explanation**")
        news_items, results, entities_list = get_scored_ticker_news(ticker, company_name=name)

        if not news_items:
            st.info("No recent news found for this ticker.")
        else:
            titles = [n.get("title", "") for n in news_items if n.get("title")]
            summarizer = load_summarizer()
            explanation = summarizer.summarize_headlines(titles[:8])
            st.info(f"**What's happening:** {explanation}")

            if results:
                counts = pd.Series([r["label"] for r in results]).value_counts()
                summary_cols = st.columns(3)
                for i, label in enumerate(["positive", "negative", "neutral"]):
                    summary_cols[i].metric(label.capitalize(), int(counts.get(label, 0)))

            st.divider()

            if "headline_details" not in st.session_state:
                st.session_state.headline_details = {}
            if "headline_visible" not in st.session_state:
                st.session_state.headline_visible = {}

            color_map = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
            for item, result, entities in zip(news_items, results, entities_list):
                publish_time = item.get("providerPublishTime")
                if publish_time:
                    time_str = datetime.fromtimestamp(publish_time).strftime("%Y-%m-%d %H:%M")
                elif item.get("published_iso"):
                    try:
                        time_str = datetime.fromisoformat(item["published_iso"]).strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        time_str = ""
                else:
                    time_str = ""
                key = hashlib.md5(
                    (item.get("link", "") + item.get("title", "")).encode()
                ).hexdigest()[:10]

                row_cols = st.columns([12, 1])
                with row_cols[0]:
                    entity_tags = " ".join(f"`{e}`" for e in entities) if entities else ""
                    st.markdown(
                        f"{color_map.get(result['label'], '')} **{result['label'].upper()}** "
                        f"({result['score']:.2f}) — "
                        f"[{item.get('title', '(no title)')}]({item.get('link', '#')})  \n"
                        f"<small>{item.get('publisher', '')} · {time_str}</small>"
                        + (f"  \n{entity_tags}" if entity_tags else ""),
                        unsafe_allow_html=True,
                    )
                with row_cols[1]:
                    if st.button("📄", key=f"btn_{key}", help="Show summary & market impact"):
                        st.session_state.headline_visible[key] = not st.session_state.headline_visible.get(
                            key, False
                        )

                if st.session_state.headline_visible.get(key):
                    if key not in st.session_state.headline_details:
                        with st.spinner("Summarizing article..."):
                            article_url = gnews_resolve.resolve(item.get("link", ""))
                            paragraphs = fetch_article_paragraphs(article_url)

                            if paragraphs:
                                full_text = " ".join(paragraphs)
                                summary = None
                                used_llm = False

                                if "ollama_available" not in st.session_state:
                                    st.session_state.ollama_available = llm_available()

                                if st.session_state.ollama_available:
                                    summary = summarize_article_llm(full_text)
                                    used_llm = summary is not None

                                if summary is None:
                                    summary = summarizer.summarize_headlines(
                                        [full_text], max_length=200, min_length=60
                                    )
                            else:
                                # Full article text wasn't retrievable (common
                                # for Google-News-sourced links, whose redirect
                                # couldn't be resolved, or paywalled/blocked
                                # sites). Fall back to synthesizing from the
                                # other headlines already fetched for this
                                # ticker, which is more useful than just
                                # repeating this one headline back.
                                other_titles = [t for t in titles if t and t != item.get("title", "")]
                                if other_titles:
                                    summary = summarizer.summarize_headlines(other_titles[:8])
                                    summary += (
                                        " (Full text of this specific article couldn't be "
                                        "retrieved -- this summary is based on related coverage instead.)"
                                    )
                                else:
                                    summary = (
                                        f"{item.get('title', '')} (full article text couldn't be "
                                        "retrieved -- showing headline only)"
                                    )
                                used_llm = False

                            impact = get_market_impact_note(result["label"], result["score"])
                            st.session_state.headline_details[key] = (summary, impact, used_llm)

                    summary, impact, used_llm = st.session_state.headline_details[key]
                    with st.container(border=True):
                        st.markdown(f"**Summary:** {summary}")
                        st.markdown(f"**Market interpretation:** {impact}")
                        source_note = (
                            "Summarized by local LLM (Ollama)."
                            if used_llm
                            else "Summarized by local distilbart model."
                        )
                        st.caption(
                            f"{source_note} This is a general interpretation of the sentiment "
                            "classification, not financial advice or a price prediction."
                        )

st.sidebar.caption(
    "Data: Yahoo Finance + free RSS feeds.\n"
    "Sentiment: FinBERT (local). Explanations: distilbart-cnn (local)."
)

# ==========================================================================
# SECTION 3: Topics (what's everyone talking about)
# ==========================================================================

st.divider()
st.header("🗂️ Topics")

topic_counts = storage.get_topic_counts()

if not topic_counts:
    st.info(
        "No topics computed yet. Topic modeling runs as a separate, "
        "occasional batch job (it uses a large embedding model, so it "
        "doesn't run on every page load). Once you've accumulated a good "
        "amount of history, run:\n\n"
        "`python analyze_topics.py`\n\n"
        "from the project folder, then refresh this page."
    )
else:
    topic_df = pd.DataFrame(topic_counts, columns=["Topic", "Articles"])
    st.dataframe(topic_df, width='stretch', hide_index=True)
    st.caption(
        "Topics are computed periodically via `analyze_topics.py`, not live -- "
        "rerun that script anytime to refresh these with newly accumulated articles."
    )

# ==========================================================================
# SECTION 4: Trends (is sentiment improving or declining, what's emerging)
# ==========================================================================

st.divider()
st.header("📈 Trends")


@st.cache_data(ttl=300, show_spinner="Building trend data...")
def get_trend_dataframe():
    articles = storage.get_all_articles()
    return trends.articles_to_dataframe(articles)


trend_df = get_trend_dataframe()

if trend_df.empty:
    st.info(
        "Not enough history yet to show trends. Keep using the dashboard "
        "(look up tickers, browse the market table) to accumulate data, "
        "then check back here."
    )
else:
    date_span_days = (trend_df["date"].max() - trend_df["date"].min()).days
    default_freq = "D" if date_span_days <= 21 else "W"

    trend_cols = st.columns([2, 1])
    with trend_cols[0]:
        scope_options = ["All articles", "(general market)"] + sorted(
            t for t in trend_df["ticker"].dropna().unique() if t
        )
        scope = st.selectbox("Show sentiment trend for:", scope_options)
    with trend_cols[1]:
        freq_label = st.radio(
            "Interval", ["Daily", "Weekly"],
            index=0 if default_freq == "D" else 1, horizontal=True,
        )
    freq = "D" if freq_label == "Daily" else "W"

    ticker_filter = None if scope == "All articles" else scope
    sent_trend = trends.sentiment_trend(trend_df, ticker=ticker_filter, freq=freq)

    if sent_trend.empty:
        st.info(f"No dated articles found for {scope}.")
    else:
        st.markdown("**Sentiment over time**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sent_trend.index, y=sent_trend["positive"],
            mode="lines", name="Positive", line=dict(color="#2ecc71"),
        ))
        fig.add_trace(go.Scatter(
            x=sent_trend.index, y=sent_trend["negative"],
            mode="lines", name="Negative", line=dict(color="#e74c3c"),
        ))
        fig.add_trace(go.Scatter(
            x=sent_trend.index, y=sent_trend["neutral"],
            mode="lines", name="Neutral", line=dict(color="#95a5a6"),
        ))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Articles")
        st.plotly_chart(fig, width='stretch')

        latest_net = sent_trend["net_sentiment"].dropna()
        if len(latest_net) >= 2:
            change = latest_net.iloc[-1] - latest_net.iloc[-2]
            direction = "improving" if change > 0.05 else "declining" if change < -0.05 else "holding steady"
            st.caption(
                f"Net sentiment (positive − negative, as a share of total) is "
                f"**{direction}** compared to the previous period "
                f"({latest_net.iloc[-2]:+.2f} → {latest_net.iloc[-1]:+.2f})."
            )

    st.markdown("**What's newly emerging**")
    topic_tr = trends.topic_trend(trend_df, freq="W")
    emerging = trends.detect_emerging_topics(topic_tr)

    if topic_tr.empty:
        st.caption(
            "No topic data yet -- run `python analyze_topics.py` to compute "
            "topics first, then emerging-topic detection can work off of that."
        )
    elif not emerging:
        st.caption("Nothing unusual -- topic volumes look stable week over week.")
    else:
        for e in emerging:
            growth_str = "new this week" if e["growth_ratio"] is None else f"{e['growth_ratio']}x above usual"
            st.markdown(f"🔺 **{e['topic']}** — {e['recent_count']} articles this week ({growth_str})")

# ==========================================================================
# SECTION 5: 3D Topic Explorer
# ==========================================================================

st.divider()
st.header("🌌 3D Topic Explorer")

coord_articles = storage.get_articles_with_coords()

if not coord_articles:
    st.info(
        "No 3D topic map yet. This is computed as part of the same batch "
        "job as Topics -- run:\n\n"
        "`python analyze_topics.py`\n\n"
        "from the project folder (needs enough accumulated history; see "
        "the Topics section above), then refresh this page."
    )
else:
    coord_df = pd.DataFrame(coord_articles)

    color_by = st.radio(
        "Color points by:", ["Sentiment", "Topic"], horizontal=True,
        help="Sentiment shows sentiment clusters; Topic shows related themes.",
    )

    if color_by == "Sentiment":
        color_map = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
        colors = coord_df["label"].map(color_map).fillna("#95a5a6")
        legend_note = "🟢 Positive · 🔴 Negative · ⚪ Neutral"
    else:
        topics_unique = coord_df["topic_label"].fillna("(uncategorized)").unique()
        palette = [
            "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
            "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b",
        ]
        topic_color_map = {t: palette[i % len(palette)] for i, t in enumerate(topics_unique)}
        colors = coord_df["topic_label"].fillna("(uncategorized)").map(topic_color_map)
        legend_note = "Each color is a distinct topic -- hover a point to see which."

    fig = go.Figure(data=[go.Scatter3d(
        x=coord_df["x"], y=coord_df["y"], z=coord_df["z"],
        mode="markers",
        marker=dict(
            size=5,
            color=colors,
            opacity=0.8,
        ),
        text=coord_df["title"],
        customdata=coord_df["id"],
        hovertemplate="%{text}<extra></extra>",
    )])
    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        ),
    )

    st.caption(f"{legend_note} — drag to rotate, scroll to zoom, click a point to see its topic cluster.")
    event = st.plotly_chart(
        fig, width='stretch', on_select="rerun", selection_mode="points", key="topic_explorer_3d",
    )

    selected_points = event.selection.get("points", []) if event and event.selection else []
    if selected_points:
        clicked_id = selected_points[0]["customdata"]
        clicked_row = coord_df[coord_df["id"] == clicked_id]
        if not clicked_row.empty:
            clicked_topic = clicked_row.iloc[0]["topic_label"]
            has_topic = pd.notna(clicked_topic)
            st.markdown(f"**Cluster: {clicked_topic if has_topic else '(uncategorized)'}**")

            related = coord_df[coord_df["topic_label"] == clicked_topic] if has_topic else clicked_row
            for _, row in related.head(10).iterrows():
                icon = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(row["label"], "")
                link = row.get("link") or "#"

                thumb_cols = st.columns([1, 5])
                with thumb_cols[0]:
                    image_url = get_cached_og_image(link) if link and link != "#" else None
                    if image_url:
                        try:
                            st.image(image_url, width=80)
                        except Exception:
                            st.write("🖼️")  # broken/unreachable image, don't break the layout
                    else:
                        st.write("🖼️")
                with thumb_cols[1]:
                    st.markdown(f"{icon} [{row['title']}]({link})")

            st.caption(f"{len(related)} article(s) in this cluster.")
    else:
        st.caption("Click a point above to see other articles in its topic cluster.")
