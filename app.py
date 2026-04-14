import streamlit as st
import yfinance as yf
import finnhub
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fredapi import Fred
from datetime import datetime, timedelta
import google.genai as genai
from google.genai import types

# --- 1. DATA STORAGE (SUPABASE CLOUD) ---
from supabase import create_client, Client

# Initialize Supabase Connection
@st.cache_resource
def init_connection():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.error("🚨 Supabase connection failed. Check your API keys in secrets.toml.")
        return None

supabase = init_connection()

def load_watchlist():
    if supabase:
        try:
            response = supabase.table("watchlist").select("ticker").execute()
            return [row['ticker'] for row in response.data]
        except Exception as e:
            pass
    return ["AAPL", "TSLA", "BTC-USD"]

def save_watchlist(watchlist):
    if not supabase:
        return
    try:
        backup = supabase.table("watchlist").select("ticker").execute()
        backup_data = [{"ticker": row["ticker"]} for row in backup.data]
        supabase.table("watchlist").delete().neq("ticker", "0").execute()
        if watchlist:
            data = [{"ticker": t} for t in watchlist]
            supabase.table("watchlist").insert(data).execute()
    except Exception as e:
        st.error(f"Database sync error: {e}. Attempting to restore previous watchlist...")
        try:
            supabase.table("watchlist").delete().neq("ticker", "0").execute()
            if backup_data:
                supabase.table("watchlist").insert(backup_data).execute()
            st.warning("Watchlist restored to previous state.")
        except Exception as restore_err:
            st.error(f"Restore also failed: {restore_err}. Please refresh the page.")

def load_portfolio():
    if supabase:
        try:
            response = supabase.table("portfolio").select("*").order("id").execute()
            return response.data
        except Exception as e:
            pass
    return []

# --- 2. SETUP & THEME ---
st.set_page_config(page_title="Market Command Center", layout="wide")

st.markdown("""
    <style>
    /* Pitch-Black Terminal with Cyan Accents */
    .stApp { background-color: #000000; color: #FFFFFF; }

    /* Kill Streamlit's teal primary color on inputs */
    :root {
        --primary-color: #FFFFFF !important;
    }
    
    /* =========================================
       PREVENTS SIDEBAR SQUISHING 
       ========================================= */
    [data-testid="stSidebar"] { 
        border-right: 1px solid #333333; 
        min-width: 330px !important; 
    }
    
    /* =========================================
       PULLS MAIN PAGE UP TO CEILING
       ========================================= */
    .block-container {
        padding-top: 1.5rem !important; 
    }

    [data-testid="stMetric"] {
        background-color: #000000 !important;
        border: 1px solid #333333 !important;
        border-left: 3px solid #00d2ff !important;
        padding: 15px !important;
        box-shadow: none !important;
    }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; font-family: monospace; }
    
    .stTabs [data-baseweb="tab-list"] { background-color: #0a0a0a; border-bottom: 1px solid #333333; }
    .stTabs [data-baseweb="tab"] { color: #888888; }
    .stTabs [aria-selected="true"] { color: #00d2ff !important; border-bottom: 2px solid #00d2ff !important; }

    a { color: #00d2ff !important; text-decoration: none; font-weight: bold; }
    a:hover { color: #FFFFFF !important; text-decoration: underline; }
    
    [data-testid="stInfo"], [data-testid="stError"], [data-testid="stSuccess"] { 
        background-color: #000000 !important; 
        border: 1px solid #333333 !important; 
        border-left: 3px solid #00d2ff !important; 
        color: #FFFFFF !important; 
    }
    
    /* Sleek Radio Buttons for Timeframes */
    div[role="radiogroup"] > label {
        background-color: #0a0a0a !important;
        border: 1px solid #333333 !important;
        border-radius: 5px !important;
        padding: 5px 15px !important;
        margin-right: 5px !important;
    }
    div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #00d2ff !important;
        color: #000000 !important;
        font-weight: bold;
    }

    /* Pushes sidebar content up to the top of the screen */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important; 
        padding-bottom: 1rem !important;
        gap: 0.5rem !important; 
    }
    
    /* Reduces wasted space around ALL horizontal lines */
    hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* White sharp border around the ticker search input box */
    [data-testid="stSidebar"] [data-testid="stTextInput"] > div > div > input {
        border: 1px solid #FFFFFF !important;
        border-radius: 0px !important;
        outline: none !important;
        box-shadow: none !important;
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] > div > div {
        border: 1px solid #FFFFFF !important;
        border-radius: 0px !important;
        box-shadow: none !important;
        background-color: #000000 !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] > div > div:focus-within {
        border: 1px solid #FFFFFF !important;
        border-radius: 0px !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] * {
        border-radius: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. API & CACHED DATA ENGINE ---
fred = Fred(api_key=st.secrets["fred_api_key"])
finnhub_client = finnhub.Client(api_key=st.secrets["finnhub_api_key"])

# Ordered fallback list — if Google retires the first, the next is tried automatically
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
]

try:
    ai_client = genai.Client(api_key=st.secrets["gemini_api_key"])
except KeyError:
    st.warning("⚠️ Gemini API key not found in secrets.toml — AI features disabled.")
    ai_client = None
except Exception as e:
    st.warning(f"⚠️ Gemini initialization failed: {e} — AI features disabled.")
    ai_client = None

def call_gemini(prompt):
    """Tries each model in GEMINI_MODELS until one works. Returns (text, model_used) or (None, None)."""
    if not ai_client:
        return None, None
    for model in GEMINI_MODELS:
        try:
            # Injecting the strict config here
            response = ai_client.models.generate_content(
                model=model, 
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1, # Forces strict, robotic consistency
                    max_output_tokens=300
                )
            )
            raw = response.text
            if raw and raw.strip():
                return raw.strip(), model
        except Exception as e:
            err = str(e)
            if "404" in err or "429" in err or "not found" in err.lower():
                continue
            return None, None
    return None, None

@st.cache_data(ttl=14400)  # 4 hours — reduces Yahoo Finance hits on shared hosting IPs
def get_stock_info(symbol):
    ticker_obj = yf.Ticker(symbol)
    try:
        info = ticker_obj.info
    except Exception:
        info = {}
    try:
        financials = ticker_obj.financials
    except Exception:
        financials = None
    return {"info": info, "financials": financials}

@st.cache_data(ttl=300)
def get_chart_data(symbol, period):
    interval = "1m" if period == "1d" else "5m" if period == "5d" else "1d"
    return yf.Ticker(symbol).history(period=period, interval=interval)

@st.cache_data(ttl=300)
def get_benchmark_data(period):
    interval = "1m" if period == "1d" else "5m" if period == "5d" else "1d"
    return yf.Ticker("SPY").history(period=period, interval=interval)

@st.cache_data(ttl=86400)
def get_macro_series(series_id):
    return fred.get_series(series_id).dropna()

@st.cache_data(ttl=3600)
def get_correlation_data(watchlist_tuple):
    return yf.download(list(watchlist_tuple), period="6mo", progress=False)['Close']

@st.cache_data(ttl=3600)
def get_ai_news_analysis(headlines_text):
    """Returns (briefing_text, sentiment_label, sentiment_reasoning) in one Gemini call."""
    if not ai_client or not headlines_text.strip():
        return None, None, None
    prompt = f"""
    Act as an expert financial analyst. Read the following recent news headlines.

    Your response must follow this EXACT format with these three labeled sections.
    DO NOT output any conversational filler. DO NOT use markdown bolding on the labels. 
    Output ONLY the labels and the text.

    BRIEFING: Write a cohesive 3-sentence executive summary highlighting key tailwinds and headwinds. Professional and objective tone.

    SENTIMENT: Write exactly one word: BULLISH, BEARISH, or NEUTRAL

    REASONING: Write one concise sentence explaining the sentiment label.

    Headlines:
    {headlines_text}
    """
    try:
        raw, _ = call_gemini(prompt)
        if not raw:
            return None, None, None

        import re
        text = raw.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\*\*\s*([A-Z]+)\s*\*\*\s*:', r'\1:', text)
        text = re.sub(r'\*\*\s*([A-Z]+:)\s*\*\*', r'\1', text)
        text = re.sub(r':\s*\[([^\]]+)\]', r': \1', text)

        briefing, sentiment, reasoning = None, None, None

        b_match = re.search(r'BRIEFING:\s*(.*?)(?=\n\s*SENTIMENT:|\Z)', text, re.DOTALL | re.IGNORECASE)
        if b_match:
            briefing = b_match.group(1).strip()

        s_match = re.search(r'SENTIMENT:\s*\[?\s*(BULLISH|BEARISH|NEUTRAL)\s*\]?', text, re.IGNORECASE)
        if s_match:
            sentiment = s_match.group(1).upper()

        r_match = re.search(r'REASONING:\s*(.*?)(?=\n\s*[A-Z]+:|\Z)', text, re.DOTALL | re.IGNORECASE)
        if r_match:
            reasoning = r_match.group(1).strip()

        return briefing, sentiment, reasoning
    except Exception:
        return None, None, None

def get_global_briefing(headlines_text):
    """Uses session-state TTL cache and call_gemini fallback chain."""
    import time

    cache_key = "global_briefing_result"
    cache_ts_key = "global_briefing_ts"
    ttl_seconds = 3600

    now = time.time()
    cached_result = st.session_state.get(cache_key)
    cached_ts = st.session_state.get(cache_ts_key, 0)
    cached_headlines = st.session_state.get("global_briefing_headlines", "")

    if (cached_result
            and (now - cached_ts) < ttl_seconds
            and cached_headlines == headlines_text):
        return cached_result

    if not ai_client or not headlines_text.strip():
        return None

    prompt = f"""You are an expert financial analyst. Read these news headlines and write a clear, 
3-sentence executive briefing summarizing the dominant macro narrative, key risks, and any bright spots.
Write in plain prose. Do not use bullet points or labels.

Headlines:
{headlines_text}"""

    result, _ = call_gemini(prompt)
    if result:
        st.session_state[cache_key] = result
        st.session_state[cache_ts_key] = now
        st.session_state["global_briefing_headlines"] = headlines_text
    return result

# Thin wrapper for ticker news tab (no separate cache needed)
def get_ai_summary(headlines_text):
    briefing, _, _ = get_ai_news_analysis(headlines_text)
    return briefing

@st.cache_data(ttl=300)
def build_portfolio_curve(trades_tuple, period):
    """
    Time-weighted portfolio reconstruction.
    trades_tuple: tuple of (ticker, shares, date_str) sorted by date.
    Returns a pd.Series of daily portfolio dollar value, starting from
    the earliest trade date. Each position only contributes from its
    own buy date forward — so adding/removing trades adjusts the curve correctly.
    """
    if not trades_tuple:
        return pd.Series(dtype=float)

    # Find earliest trade date and map the full price history needed
    all_tickers = list(set(t[0] for t in trades_tuple))
    earliest = min(pd.to_datetime(t[2]) for t in trades_tuple)

    # Download full price history for all tickers in one call
    raw = yf.download(all_tickers, start=earliest, progress=False)['Close']
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=all_tickers[0])
    raw = raw.ffill()

    # Apply period filter AFTER download so we have prices to anchor cost basis
    period_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "ALL": None}
    days = period_map.get(period)
    if days:
        cutoff = pd.Timestamp.now(tz=raw.index.tz) - pd.Timedelta(days=days)
        display_data = raw[raw.index >= cutoff]
    else:
        display_data = raw

    if display_data.empty:
        return pd.Series(dtype=float)

    # Build daily portfolio value: for each day, sum shares * price
    # for each position that was open on that day
    portfolio_value = pd.Series(0.0, index=display_data.index)
    for ticker, shares, date_str in trades_tuple:
        if ticker not in display_data.columns:
            continue
        buy_date = pd.Timestamp(date_str)
        if buy_date.tzinfo is None and display_data.index.tz is not None:
            buy_date = buy_date.tz_localize(display_data.index.tz)
        # Only add value from buy date onwards
        mask = display_data.index >= buy_date
        portfolio_value[mask] += display_data[ticker][mask] * float(shares)

    # Remove leading zeros (days before first trade)
    portfolio_value = portfolio_value[portfolio_value > 0]
    return portfolio_value

@st.cache_data(ttl=300)
def get_heatmap_data(index_name, tickers_tuple):
    heatmap_data = []
    for stock in tickers_tuple:
        try:
            info = yf.Ticker(stock).fast_info
            price = info['last_price']
            prev = info['previous_close']
            change = ((price - prev) / prev) * 100
            try:
                mcap = info['market_cap']
            except Exception:
                mcap = 1_000_000_000
            heatmap_data.append({
                "Ticker": stock,
                "Change": change,
                "Change_Str": f"{change:+.1f}%",
                "Market Cap": mcap
            })
        except Exception as e:
            continue
    return heatmap_data

@st.cache_data(ttl=30)
def get_watchlist_prices(watchlist_tuple):
    """Fetches all watchlist prices in one yf.download call instead of N sequential requests."""
    if not watchlist_tuple:
        return {}
    try:
        raw = yf.download(list(watchlist_tuple), period="2d", progress=False, auto_adjust=True)
        prices = {}
        close = raw['Close']
        if isinstance(close, pd.Series):
            close = close.to_frame(name=watchlist_tuple[0])
        for ticker in watchlist_tuple:
            if ticker in close.columns:
                series = close[ticker].dropna()
                if len(series) >= 2:
                    prices[ticker] = {
                        "last_price": float(series.iloc[-1]),
                        "previous_close": float(series.iloc[-2])
                    }
                elif len(series) == 1:
                    prices[ticker] = {
                        "last_price": float(series.iloc[-1]),
                        "previous_close": float(series.iloc[-1])
                    }
        return prices
    except Exception:
        return {}

@st.fragment(run_every="120s") 
def render_watchlist():
    st.markdown("<h4 style='margin-bottom: 5px; margin-top: 10px;'>📈 Watchlist Monitor</h4>", unsafe_allow_html=True)
    
    if st.button("🔄 Sync Market Data", use_container_width=True):
        st.cache_data.clear()
        for k in ["global_briefing_result", "global_briefing_ts", "global_briefing_headlines"]:
            st.session_state.pop(k, None)
        st.rerun()

    prices = get_watchlist_prices(tuple(st.session_state.watchlist))

    for stock in st.session_state.watchlist:
        try:
            data = prices.get(stock)
            if not data:
                st.caption(f"⚠️ {stock}: no data")
                continue
            s_price = data["last_price"]
            s_change = ((s_price - data["previous_close"]) / data["previous_close"]) * 100
            
            color = "#10b981" if s_change >= 0 else "#ef4444" 
            
            cols = st.columns([3, 3, 2], vertical_alignment="center")
            
            if cols[0].button(f"**{stock}**", key=f"nav_{stock}", use_container_width=True):
                st.session_state["ticker_search_widget"] = stock
                st.session_state.active_ticker = stock
                st.session_state["app_mode_widget"] = "🌍 Market Dashboard"
                st.rerun()         
            
            cols[1].markdown(f"<span style='color:#00d2ff; font-weight:bold;'>${s_price:.2f}</span> <span style='color:{color}; font-size:12px;'>{s_change:+.1f}%</span>", unsafe_allow_html=True)
            
            if cols[2].button("✕ Remove", key=f"remove_{stock}", use_container_width=True):
                st.session_state.watchlist.remove(stock)
                save_watchlist(st.session_state.watchlist)
                st.rerun()
        except Exception as e:
            st.caption(f"⚠️ {stock}: {e}")

def format_large_number(num):
    if num is None or pd.isna(num): return "N/A"
    if abs(num) >= 1e12: return f"{num / 1e12:.2f}T"
    if abs(num) >= 1e9: return f"{num / 1e9:.2f}B"
    if abs(num) >= 1e6: return f"{num / 1e6:.2f}M"
    return str(num)

def get_sentiment_score(news_list):
    bullish = ['surge', 'growth', 'profit', 'buy', 'upbeat', 'expansion', 'dividend']
    bearish = ['drop', 'lawsuit', 'miss', 'sell', 'risk', 'decline', 'investigation']
    score = 0
    for item in news_list:
        h = item['headline'].lower()
        score += sum(1 for w in bullish if w in h)
        score -= sum(1 for w in bearish if w in h)
    return score

# --- 4. ULTRA-COMPACT SIDEBAR NAVIGATION ---
st.sidebar.markdown("<h3 style='margin-top: 0px; margin-bottom: 0px;'>📊 STEPS CAPITAL</h3>", unsafe_allow_html=True)

if "app_mode_widget" not in st.session_state:
    st.session_state["app_mode_widget"] = "🌍 Market Dashboard"

app_mode = st.sidebar.radio(
    "Navigation", 
    ["🌍 Market Dashboard", "🏆 Portfolio Backtester"], 
    key="app_mode_widget",
    label_visibility="collapsed"
)

if 'active_ticker' not in st.session_state:
    st.session_state.active_ticker = "" 
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = load_watchlist()

sc1, sc2 = st.sidebar.columns([2, 1], vertical_alignment="center")

ticker_input = sc1.text_input(
    "Search", 
    placeholder="TICKER...",
    key="ticker_search_widget",
    label_visibility="collapsed"
).upper()

active_ticker = ticker_input
st.session_state.active_ticker = active_ticker

if sc2.button("➕ Add", use_container_width=True):
    if active_ticker and active_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(active_ticker)
        save_watchlist(st.session_state.watchlist)
        st.rerun()

days_to_look = st.sidebar.slider("News History (Days)", 1, 14, 7)
    
with st.sidebar:
    render_watchlist()

with st.sidebar:
    with st.expander("🔔 Price Alerts", expanded=False):
        if 'price_alerts' not in st.session_state:
            st.session_state.price_alerts = {}  # {ticker: {"above": float|None, "below": float|None}}

        alert_cols = st.columns([2, 1], vertical_alignment="bottom")
        alert_ticker = alert_cols[0].text_input("Ticker", placeholder="AAPL", key="alert_ticker_input", label_visibility="collapsed").upper()
        alert_direction = alert_cols[1].selectbox("Dir", ["Above ▲", "Below ▼"], label_visibility="collapsed")
        alert_price = st.number_input("Alert Price ($)", min_value=0.01, value=100.0, step=1.0, key="alert_price_input", label_visibility="collapsed")

        if st.button("➕ Set Alert", use_container_width=True):
            if alert_ticker:
                if alert_ticker not in st.session_state.price_alerts:
                    st.session_state.price_alerts[alert_ticker] = {"above": None, "below": None}
                if "Above" in alert_direction:
                    st.session_state.price_alerts[alert_ticker]["above"] = alert_price
                else:
                    st.session_state.price_alerts[alert_ticker]["below"] = alert_price
                st.rerun()

        if st.session_state.price_alerts:
            st.markdown("<hr style='margin:6px 0'>", unsafe_allow_html=True)
            alerts_to_remove = []
            for tkr, levels in list(st.session_state.price_alerts.items()):
                try:
                    live_price = yf.Ticker(tkr).fast_info['last_price']
                    for direction, threshold in [("above", levels.get("above")), ("below", levels.get("below"))]:
                        if threshold is None:
                            continue
                        triggered = (direction == "above" and live_price >= threshold) or \
                                    (direction == "below" and live_price <= threshold)
                        arrow = "▲" if direction == "above" else "▼"
                        color = "#10b981" if triggered else "#555555"
                        label = "🔔 TRIGGERED" if triggered else f"${live_price:.2f}"
                        st.markdown(
                            f"<div style='font-size:12px; font-family:monospace; padding:3px 0;'>"
                            f"<span style='color:#00d2ff'>{tkr}</span> "
                            f"<span style='color:{color}'>{arrow} ${threshold:.2f}</span> "
                            f"<span style='color:#888'>{label}</span></div>",
                            unsafe_allow_html=True
                        )
                except Exception:
                    pass

                a_col, b_col = st.columns([3, 1])
                a_col.caption(tkr)
                if b_col.button("✕", key=f"del_alert_{tkr}", use_container_width=True):
                    alerts_to_remove.append(tkr)

            for tkr in alerts_to_remove:
                del st.session_state.price_alerts[tkr]
            if alerts_to_remove:
                st.rerun()
        else:
            st.caption("No alerts set.")


# ==========================================
# 5. APP ROUTING (PORTFOLIO vs DASHBOARD)
# ==========================================

if app_mode == "🏆 Portfolio Backtester":
    # --- PORTFOLIO BACKTESTER PAGE ---
    st.title("🏆 Live Portfolio Tracker")
    st.markdown("Track exact positions by inputting your real share quantity and average cost basis.")
    
    col_in1, col_in2, col_in3, col_in4, col_in5 = st.columns([1.5, 1.5, 1.5, 1.5, 1], vertical_alignment="bottom")
    with col_in1:
        p_ticker = st.text_input("Ticker", placeholder="AAPL").upper()
    with col_in2:
        p_shares = st.number_input("Shares", min_value=0.01, value=10.0, step=1.0)
    with col_in3:
        p_price = st.number_input("Price Paid ($)", min_value=0.01, value=150.0, step=10.0)
    with col_in4:
        p_date = st.date_input("Date Bought", value=datetime.now() - timedelta(days=365))
    
    with col_in5:
        if st.button("➕ Add Trade", use_container_width=True):
            if p_ticker and supabase:
                new_trade = {"ticker": p_ticker, "shares": p_shares, "price": p_price, "date": str(p_date)}
                supabase.table("portfolio").insert(new_trade).execute()
                st.rerun()

    my_portfolio = load_portfolio()
    if my_portfolio:
        combined_value = 0.0
        initial_investment = 0.0
        perf_data = []
        legacy_detected = False
        
        with st.spinner("Fetching live prices..."):
            for trade in my_portfolio:
                try:
                    if "shares" not in trade or "price" not in trade:
                        legacy_detected = True
                        continue

                    t_obj = yf.Ticker(trade['ticker'])
                    current_price = t_obj.fast_info['last_price']
                    
                    trade_shares = float(trade['shares'])
                    trade_price = float(trade['price'])
                    
                    cost_basis = trade_shares * trade_price
                    current_val = trade_shares * current_price
                    profit_loss = current_val - cost_basis
                    return_pct = (profit_loss / cost_basis) * 100 if cost_basis > 0 else 0
                    
                    combined_value += current_val
                    initial_investment += cost_basis
                    
                    perf_data.append({
                        "Asset": trade['ticker'],
                        "Date": trade['date'],
                        "Shares": f"{trade_shares:,.4f}".rstrip('0').rstrip('.'),
                        "Avg Cost": f"${trade_price:,.2f}",
                        "Invested": f"${cost_basis:,.2f}",
                        "Current Val": f"${current_val:,.2f}",
                        "P/L $": f"{profit_loss:+,.2f}",
                        "Return %": f"{return_pct:+.2f}%"
                    })
                except Exception as e:
                    st.error(f"Could not fetch data for {trade.get('ticker', 'Unknown')}: {e}")

        if legacy_detected:
            st.error("🚨 Legacy data format detected. The old format is blocking the portfolio table from rendering.")
            if st.button("🗑️ Force Reset Portfolio", use_container_width=True):
                supabase.table("portfolio").delete().neq("id", 0).execute()
                st.rerun()

        if perf_data:
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            total_return_pct = ((combined_value - initial_investment) / initial_investment) * 100 if initial_investment > 0 else 0
            
            m1.metric("Total Portfolio Value", f"${combined_value:,.2f}")
            m2.metric("Total Invested", f"${initial_investment:,.2f}")
            m3.metric("Combined Return", f"{total_return_pct:+.2f}%")

            # --- PORTFOLIO PERFORMANCE CHART (TIME-WEIGHTED) ---
            st.markdown("---")
            st.subheader("📈 Portfolio Performance Over Time")

            # Timeframe selector + benchmark toggles in one row
            perf_c1, perf_c2, perf_c3, perf_c4 = st.columns([3, 1, 1, 1])
            with perf_c1:
                perf_tf = st.radio(
                    "Timeframe", ["1M", "3M", "6M", "1Y", "ALL"],
                    index=3, horizontal=True, key="perf_tf",
                    label_visibility="collapsed"
                )
            show_spy  = perf_c2.toggle("S&P 500", value=True,  key="perf_spy")
            show_qqq  = perf_c3.toggle("NASDAQ",  value=False, key="perf_qqq")
            show_dia  = perf_c4.toggle("Dow",     value=False, key="perf_dia")

            try:
                with st.spinner("Building performance chart..."):
                    # Build the canonical trades tuple for caching
                    trades_tuple = tuple(
                        (t['ticker'], t['shares'], t['date'])
                        for t in my_portfolio
                        if 'ticker' in t and 'shares' in t and 'date' in t
                    )
                    port_series = build_portfolio_curve(trades_tuple, perf_tf)

                if port_series.empty:
                    st.info("Not enough trade history to build a performance chart yet.")
                else:
                    # Anchor: cost basis on the first day the portfolio had value
                    cost_basis_start = initial_investment
                    port_pct = ((port_series - cost_basis_start) / cost_basis_start) * 100

                    fig_perf = go.Figure()

                    # Portfolio line
                    final_pct = port_pct.iloc[-1]
                    port_color = '#00d2ff'
                    fig_perf.add_trace(go.Scatter(
                        x=port_pct.index, y=port_pct,
                        name="My Portfolio",
                        fill='tozeroy',
                        fillcolor='rgba(0,210,255,0.07)',
                        line=dict(color=port_color, width=2.5),
                        hovertemplate="<b>Portfolio</b>: %{y:+.2f}%<extra></extra>"
                    ))

                    # Benchmark helper — normalizes benchmark to start at 0% on same date
                    def add_benchmark(fig, ticker, name, color, dash='dot'):
                        try:
                            b_data = yf.Ticker(ticker).history(
                                start=port_pct.index[0], end=port_pct.index[-1]
                            )['Close'].ffill()
                            if not b_data.empty:
                                b_pct = ((b_data - b_data.iloc[0]) / b_data.iloc[0]) * 100
                                fig.add_trace(go.Scatter(
                                    x=b_pct.index, y=b_pct, name=name,
                                    line=dict(color=color, width=1.5, dash=dash),
                                    hovertemplate=f"<b>{name}</b>: %{{y:+.2f}}%<extra></extra>"
                                ))
                        except Exception:
                            pass

                    if show_spy: add_benchmark(fig_perf, "SPY",  "S&P 500 (SPY)", "rgba(255,255,255,0.4)")
                    if show_qqq: add_benchmark(fig_perf, "QQQ",  "NASDAQ (QQQ)",  "#a78bfa")
                    if show_dia: add_benchmark(fig_perf, "DIA",  "Dow (DIA)",     "#FFA500")

                    # Zero line
                    fig_perf.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)

                    fig_perf.update_layout(
                        height=320, margin=dict(l=0, r=0, t=10, b=0),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        yaxis_title="Return (%)", hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        yaxis=dict(ticksuffix="%", gridcolor='rgba(255,255,255,0.05)', zeroline=False)
                    )
                    fig_perf.update_xaxes(showgrid=False, zeroline=False)
                    st.plotly_chart(fig_perf, use_container_width=True)

            except Exception as e:
                st.caption(f"⚠️ Performance chart unavailable: {e}")

            st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
            
            # --- SURGICAL INDIVIDUAL DELETE LOGIC ---
            st.markdown("---")
            st.subheader("⚙️ Manage Positions")
            
            man1, man2, man3 = st.columns([2, 1, 1], vertical_alignment="bottom")
            
            # Map valid trades into a dictionary using their database 'id' as the key
            valid_trades = {t['id']: t for t in my_portfolio if "id" in t}

            def format_dropdown(trade_id):
                t = valid_trades[trade_id]
                return f"{t['ticker']} - Bought on {t['date']} ({t['shares']} sh @ ${t['price']})"

            with man1:
                del_selection_id = st.selectbox(
                    "Select Trade to Remove", 
                    options=list(valid_trades.keys()), 
                    format_func=format_dropdown, 
                    label_visibility="collapsed"
                )
            with man2:
                if st.button("❌ Delete Trade", use_container_width=True):
                    if del_selection_id and supabase:
                        supabase.table("portfolio").delete().eq("id", del_selection_id).execute()
                        st.rerun()
            with man3:
                if st.button("🗑️ Reset All", use_container_width=True):
                    if supabase:
                        supabase.table("portfolio").delete().neq("id", 0).execute()
                    st.rerun()
            
            # --- PORTFOLIO RISK & ALLOCATION ---
            st.markdown("---")
            risk_col, alloc_col = st.columns([1.4, 1])

            with risk_col:
                st.subheader("🔗 Portfolio Risk: Correlation Matrix")
                owned_tickers = list(set([trade['ticker'] for trade in my_portfolio if 'ticker' in trade]))
                if len(owned_tickers) > 1:
                    try:
                        with st.spinner("Calculating risk correlation..."):
                            data_all = get_correlation_data(tuple(owned_tickers))
                            returns = data_all.ffill().pct_change().dropna()
                            corr_matrix = returns.corr()
                            fig_corr = px.imshow(
                                corr_matrix, text_auto=".2f", color_continuous_scale='Blues',
                                zmin=-1, zmax=1, aspect="auto"
                            )
                            fig_corr.update_layout(
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color="white", height=380, margin=dict(l=0, r=0, t=30, b=0)
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                    except Exception as e:
                        st.error(f"Matrix calculation failed: {str(e)}")
                else:
                    st.info("Add at least 2 different tickers to your portfolio to analyze risk correlation.")

            with alloc_col:
                st.subheader("🥧 Allocation")
                # Group by ticker, sum current value
                alloc_map = {}
                for row in perf_data:
                    ticker = row["Asset"]
                    val = float(row["Current Val"].replace("$", "").replace(",", ""))
                    alloc_map[ticker] = alloc_map.get(ticker, 0) + val
                if alloc_map:
                    fig_donut = go.Figure(go.Pie(
                        labels=list(alloc_map.keys()),
                        values=list(alloc_map.values()),
                        hole=0.55,
                        textinfo="label+percent",
                        textfont=dict(color="white", size=13),
                        marker=dict(
                            colors=['#00d2ff','#10b981','#FFA500','#ef4444','#a78bfa','#f472b6','#34d399'],
                            line=dict(color='#000000', width=2)
                        )
                    ))
                    fig_donut.update_layout(
                        height=380, margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=False,
                        annotations=[dict(
                            text=f"${combined_value:,.0f}",
                            x=0.5, y=0.5, font=dict(size=16, color='white', family='monospace'),
                            showarrow=False
                        )]
                    )
                    st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("Your tracker is empty. Input a trade above using your exact shares and cost basis.")

else:
    # --- MARKET DASHBOARD PAGE ---
    if active_ticker and active_ticker.strip() != "":
        # --- INDIVIDUAL STOCK VIEW ---
        try:
            stock_pkg = get_stock_info(active_ticker)
        except Exception as e:
            st.error(f"⚠️ Could not load data for **{active_ticker}** — Yahoo Finance may be rate limiting. Please wait a moment and try again.")
            st.caption(f"Technical detail: {e}")
            st.stop()

        info = stock_pkg["info"]

        if not info:
            st.warning(f"⚠️ No data returned for **{active_ticker}**. This may be a temporary rate limit from Yahoo Finance. Try again in a moment.")
            st.stop()
        
        fast_hist = yf.Ticker(active_ticker).history(period="5d")
        if not fast_hist.empty:
            current_price = fast_hist['Close'].iloc[-1]
            prev_close = fast_hist['Close'].iloc[-2] if len(fast_hist) > 1 else current_price
            change = ((current_price - prev_close) / prev_close) * 100
        else:
            current_price, change = 0, 0

        st.markdown(f"""
            <div style="background-color: #000000; padding: 15px; border: 1px solid #333333; border-top: 2px solid #00d2ff; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; font-family: monospace;">
                <span style="color: white; font-weight: bold; font-size: 20px;">> {info.get('longName', active_ticker).upper()} </span>
                <span style="color: #00d2ff; font-weight: bold;">SYS_STAT: OK | TCKR: {active_ticker}</span>
            </div>
        """, unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Price", f"${current_price:,.2f}", f"{change:+.2f}%")
        k2.metric("Market Cap", format_large_number(info.get('marketCap', 0)))
        k3.metric("Volume", format_large_number(info.get('regularMarketVolume', 0)))
        k4.metric("PE Ratio", f"{info.get('trailingPE', 'N/A')}")

        st.markdown("<br>", unsafe_allow_html=True)

        tab_chart, tab_news, tab_fin = st.tabs(["📊 Charting & Tech", "📰 News & Sentiment", "💰 Financials"])
        
        with tab_chart:
            tf_options = {"1D": "1d", "5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "YTD": "ytd", "1Y": "1y", "5Y": "5y", "MAX": "max"}
            selected_tf = st.radio("Timeframe", list(tf_options.keys()), index=6, horizontal=True, label_visibility="collapsed")
            mapped_period = tf_options[selected_tf]

            hist = get_chart_data(active_ticker, mapped_period)
            spy_hist = get_benchmark_data(mapped_period)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            chart_type = st.session_state.get('chart_style', 'Line')
            show_sma = st.session_state.get('toggle_sma', False)
            show_bb = st.session_state.get('toggle_bb', False)
            show_rsi = st.session_state.get('toggle_rsi', False)
            show_macd = st.session_state.get('toggle_macd', False)
            
            use_tech = show_sma or show_rsi or show_bb or show_macd or chart_type == "Candle"
            fig = go.Figure()

            if not use_tech:
                if not hist.empty and not spy_hist.empty:
                    ticker_norm = (hist['Close'] / hist['Close'].iloc[0]) * 100
                    spy_norm = (spy_hist['Close'] / spy_hist['Close'].iloc[0]) * 100
                    fig.add_trace(go.Scatter(x=ticker_norm.index, y=ticker_norm, name=active_ticker, line=dict(color='#00d2ff', width=3)))
                    fig.add_trace(go.Scatter(x=spy_norm.index, y=spy_norm, name="S&P 500 (SPY)", line=dict(color='rgba(255,255,255,0.4)', width=2, dash='dot')))
                y_title = "Normalized Growth (Base 100)"
            else:
                if chart_type == "Candle":
                    fig.add_trace(go.Candlestick(
                        x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'],
                        name="Price", increasing_line_color='#10b981', decreasing_line_color='#ef4444'
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=hist['Close'], name="Price", fill='tozeroy', 
                        fillcolor='rgba(0, 210, 255, 0.1)', line=dict(color='#00d2ff', width=2.5)
                    ))
                
                if show_sma:
                    hist['SMA20'] = hist['Close'].rolling(window=20).mean()
                    hist['SMA50'] = hist['Close'].rolling(window=50).mean()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA20'], name="20D SMA", line=dict(color='#FFA500', width=1.5)))
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA50'], name="50D SMA", line=dict(color='#10b981', width=1.5)))
                    
                if show_bb:
                    hist['BB_mid'] = hist['Close'].rolling(window=20).mean()
                    hist['BB_std'] = hist['Close'].rolling(window=20).std()
                    hist['BB_upper'] = hist['BB_mid'] + 2 * hist['BB_std']
                    hist['BB_lower'] = hist['BB_mid'] - 2 * hist['BB_std']
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_upper'], name="Upper BB", line=dict(color='rgba(255,255,255,0.3)', dash='dot', width=1)))
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['BB_lower'], name="Lower BB", fill='tonexty', fillcolor='rgba(255,255,255,0.05)', line=dict(color='rgba(255,255,255,0.3)', dash='dot', width=1)))

                y_title = "Price ($)"

            hide_breaks = []
            if mapped_period in ["1d", "5d"]:
                hide_breaks = [dict(bounds=["sat", "mon"])] 

            fig.update_layout(
                height=500, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_rangeslider_visible=False, yaxis_title=y_title, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(rangebreaks=hide_breaks)
            )
            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', zeroline=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            t1, t2, t3, t4, t5 = st.columns(5)
            
            t1.selectbox("Chart Style", ["Line", "Candle"], key="chart_style", label_visibility="collapsed")
            t2.toggle("SMA (20/50)", key="toggle_sma")
            t3.toggle("Bollinger Bands", key="toggle_bb")
            t4.toggle("RSI Indicator", key="toggle_rsi")
            t5.toggle("MACD Indicator", key="toggle_macd")

            if show_macd and len(hist) > 26:
                exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
                exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                histogram = macd - signal
                
                hist_colors = ['#10b981' if val >= 0 else '#ef4444' for val in histogram]
                
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Bar(x=hist.index, y=histogram, name="Hist", marker_color=hist_colors))
                fig_macd.add_trace(go.Scatter(x=hist.index, y=macd, name="MACD", line=dict(color='#00d2ff', width=1.5)))
                fig_macd.add_trace(go.Scatter(x=hist.index, y=signal, name="Signal", line=dict(color='#FFA500', width=1.5)))
                
                fig_macd.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="MACD", hovermode="x unified", xaxis=dict(rangebreaks=hide_breaks))
                fig_macd.update_xaxes(showgrid=False)
                fig_macd.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
                st.plotly_chart(fig_macd, use_container_width=True)

            if show_rsi and len(hist) > 14:
                delta = hist['Close'].diff()
                # Wilder's smoothing (alpha=1/14) — matches Bloomberg/TradingView standard
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                rs = avg_gain / avg_loss
                hist['RSI'] = 100 - (100 / (1 + rs))
                
                fig_rsi = px.line(hist, x=hist.index, y='RSI')
                fig_rsi.add_hline(y=70, line_dash="dot", line_color="#ef4444", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dot", line_color="#10b981", annotation_text="Oversold")
                
                fig_rsi.update_layout(
                    height=200, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis_title="RSI (14)", hovermode="x unified", xaxis=dict(rangebreaks=hide_breaks)
                )
                fig_rsi.update_xaxes(showgrid=False)
                fig_rsi.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
                st.plotly_chart(fig_rsi, use_container_width=True)

        with tab_news:
            try:
                start_date = (datetime.now() - timedelta(days=days_to_look)).strftime('%Y-%m-%d')
                news_items = finnhub_client.company_news(active_ticker, _from=start_date, to=datetime.now().strftime('%Y-%m-%d'))
            except Exception as e:
                news_items = []

            # --- SENTIMENT BADGE: AI-powered with keyword fallback ---
            ai_briefing, ai_sentiment, ai_reasoning = None, None, None
            if news_items:
                headlines_only = "\n".join([f"- {item['headline']}" for item in news_items[:10]])
                with st.spinner("🤖 AI is analyzing sentiment..."):
                    ai_briefing, ai_sentiment, ai_reasoning = get_ai_news_analysis(headlines_only)

            # Sentiment badge — renders independently of briefing
            if ai_sentiment:
                badge_color = "#10b981" if ai_sentiment == "BULLISH" else "#ef4444" if ai_sentiment == "BEARISH" else "#888888"
                badge_icon = "📈" if ai_sentiment == "BULLISH" else "📉" if ai_sentiment == "BEARISH" else "➡️"
                st.markdown(f"""
                    <div style="display:flex; align-items:center; gap:12px; padding:12px 16px;
                                background:#0a0a0a; border:1px solid #333; border-left:4px solid {badge_color};
                                border-radius:6px; margin-bottom:12px;">
                        <span style="font-size:22px;">{badge_icon}</span>
                        <div>
                            <span style="color:{badge_color}; font-weight:bold; font-family:monospace; font-size:16px;">
                                {ai_sentiment} SENTIMENT
                            </span>
                            {"<br><span style='color:#aaa; font-size:13px;'>" + ai_reasoning + "</span>" if ai_reasoning else ""}
                        </div>
                        <span style="margin-left:auto; color:#555; font-size:11px; font-family:monospace;">AI-POWERED</span>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Keyword fallback — only shown if AI sentiment parsing failed
                sent = get_sentiment_score(news_items)
                if sent > 0: st.success(f"**BULLISH SENTIMENT** (Score: {sent})")
                elif sent < 0: st.error(f"**BEARISH SENTIMENT** (Score: {sent})")
                else: st.info("**NEUTRAL SENTIMENT**")

            st.markdown("<br>", unsafe_allow_html=True)

            # Briefing renders independently — not gated on ai_sentiment being truthy
            if ai_briefing:
                st.markdown(f"""
                    <div style="background-color: #0a0a0a; padding: 20px; border-radius: 8px; border: 1px solid #333333; border-left: 4px solid #00d2ff; margin-bottom: 25px;">
                        <h4 style="color: #00d2ff; margin-top: 0; font-family: monospace;">🤖 AI EXECUTIVE BRIEFING</h4>
                        <p style="color: #FFFFFF; font-size: 15px; line-height: 1.6;">{ai_briefing}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            for item in news_items[:10]:
                st.markdown(f"""
                    <div style="background-color: #0a0a0a; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 3px solid #00d2ff;">
                        <a href="{item['url']}" target="_blank" style="color: #00d2ff; text-decoration: none; font-weight: bold; font-size: 16px;">{item['headline']}</a>
                        <br>
                        <span style="color: #888888; font-size: 12px;">{datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d %H:%M')} | {item['source']}</span>
                    </div>
                """, unsafe_allow_html=True)

        with tab_fin:
            raw_fin = stock_pkg.get("financials")
            
            if raw_fin is not None and not raw_fin.empty:
                all_metrics = raw_fin.index.tolist()

                categories = {
                    "Top Line (Growth)": ["Revenue", "Gross Profit", "Sales"],
                    "Bottom Line (Profit)": ["Net Income", "EBITDA", "Operating Income"],
                    "Cash & Debt (Risk)": ["Cash", "Debt", "Liability"],
                    "Efficiency": ["Expense", "Research", "Development", "Assets"]
                }

                btn_cols = st.columns(len(categories))

                if f'f_sel_{active_ticker}' not in st.session_state:
                    st.session_state[f'f_sel_{active_ticker}'] = all_metrics[:5]

                for i, (name, keywords) in enumerate(categories.items()):
                    if btn_cols[i].button(name, key=f"btn_{name}_{active_ticker}", use_container_width=True):
                        st.session_state[f'f_sel_{active_ticker}'] = [row for row in all_metrics if any(k.lower() in row.lower() for k in keywords)]
                        st.rerun()

                selected_metrics = st.multiselect(
                    "Search or refine metrics:", 
                    options=all_metrics, 
                    default=st.session_state[f'f_sel_{active_ticker}'],
                    key=f"ms_{active_ticker}" 
                )

                display_df = raw_fin.loc[selected_metrics] if selected_metrics else raw_fin.head(5)
                st.dataframe(display_df.map(format_large_number), use_container_width=True, height=400)

                exp_col1, exp_col2 = st.columns([1, 3])
                csv = display_df.to_csv().encode('utf-8')
                exp_col1.download_button("📥 Save CSV", data=csv, file_name=f"{active_ticker}_financials.csv")
            else:
                st.info(f"Financial statements are not applicable for {active_ticker} (likely an ETF or Cryptocurrency).")

    else:
        # --- HOME PAGE VIEW (NO TICKER SEARCHED) ---
        # Reuse the same cached batch fetch as the sidebar — zero extra API calls
        ticker_prices = get_watchlist_prices(tuple(st.session_state.watchlist))
        watchlist_data = []
        for ticker in st.session_state.watchlist:
            try:
                data = ticker_prices.get(ticker)
                if not data:
                    continue
                s_price = data["last_price"]
                s_prev_close = data["previous_close"]
                change = ((s_price - s_prev_close) / s_prev_close) * 100
                color = "#10b981" if change >= 0 else "#ef4444"
                watchlist_data.append(
                    f"<span style='color:white; font-weight:bold;'>{ticker}</span> "
                    f"<span style='color:#00d2ff;'>${s_price:.2f}</span> "
                    f"<span style='color:{color};'>({change:+.2f}%)</span>"
                )
            except Exception as e:
                continue

        ticker_html = " &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; ".join(watchlist_data)

        st.markdown(f"""
            <style>
                @keyframes scroll-left {{
                    0% {{ transform: translateX(100%); }}
                    100% {{ transform: translateX(-100%); }}
                }}
                .ticker-container {{
                    background-color: #000000;
                    padding: 12px 0;
                    border-top: 1px solid #333333;
                    border-bottom: 1px solid #333333;
                    margin-bottom: 10px;
                    overflow: hidden;
                    white-space: nowrap;
                    width: 100%;
                }}
                .ticker-text {{
                    display: inline-block;
                    font-family: monospace;
                    font-size: 18px;
                    letter-spacing: 1px;
                    animation: scroll-left 25s linear infinite;
                }}
                .ticker-text:hover {{
                    animation-play-state: paused;
                }}
            </style>
            <div class="ticker-container"><div class="ticker-text">{ticker_html}</div></div>
        """, unsafe_allow_html=True)

        st.subheader("🏛 Market Macro Pulse")
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.caption("Fed Funds Rate")
            st.line_chart(get_macro_series('FEDFUNDS').tail(24), color="#ef4444", height=150)
        with m2:
            st.caption("Inflation (CPI)")
            st.line_chart(get_macro_series('CPIAUCSL').tail(24), color="#10b981", height=150)
        with m3:
            st.caption("Yield Curve (10Y-2Y)")
            try:
                t10 = get_macro_series('DGS10') 
                t2 = get_macro_series('DGS2')
                spread = (t10 - t2).tail(24)
                st.line_chart(spread, color="#00d2ff", height=150)
            except Exception as e:
                st.caption("Yield data unavailable.")

        st.markdown("---")

        col_news, col_heat = st.columns([1, 1.5])
        
        with col_news:
            st.subheader("📰 Global Headlines")
            try:
                g_news = finnhub_client.general_news('general', min_id=0)

                if g_news:
                    global_headlines = "\n".join([f"- {item['headline']}" for item in g_news[:10]])

                    with st.spinner("🤖 AI is synthesizing global macro data..."):
                        global_briefing = get_global_briefing(global_headlines)

                    if global_briefing:
                        st.markdown(f"""
                            <div style="background-color: #000000; padding: 20px; border-radius: 8px; border: 1px solid #333333; border-left: 4px solid #00d2ff; margin-bottom: 10px;">
                                <h4 style="color: #00d2ff; margin-top: 0; font-family: monospace;">🤖 GLOBAL MACRO BRIEFING</h4>
                                <p style="color: #FFFFFF; font-size: 15px; line-height: 1.6;">{global_briefing}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    with st.container(height=400):
                        for item in g_news[:10]:
                            st.markdown(f"🔗 **[{item['headline']}]({item['url']})**")
                            st.markdown("---")

            except Exception as e:
                st.error(f"News unavailable: {e}")

        with col_heat:
            h_col1, h_col2 = st.columns([1.5, 1], vertical_alignment="center")
            
            with h_col1:
                st.subheader("🗺️ Market Heat Map")
            
            indices = {
                "S&P 500 (Top 30)": ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'BRK-B', 'LLY', 'AVGO', 'JPM', 'TSLA', 'UNH', 'V', 'XOM', 'MA', 'JNJ', 'PG', 'HD', 'COST', 'MRK', 'ABBV', 'CRM', 'CVX', 'AMD', 'BAC', 'PEP', 'LIN', 'KO', 'TMO'],
                "NASDAQ 100 (Top 30)": ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'AVGO', 'TSLA', 'COST', 'PEP', 'CSCO', 'TMUS', 'ADBE', 'TXN', 'QCOM', 'AMD', 'CMCSA', 'INTU', 'AMGN', 'HON', 'AMAT', 'ISRG', 'SBUX', 'BKNG', 'GILD', 'PANW', 'VRTX', 'MDLZ', 'REGN'],
                "Dow Jones (30)": ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT']
            }
            
            with h_col2:
                selected_index = st.selectbox("Select Market", list(indices.keys()), label_visibility="collapsed")
                
            target_tickers = indices[selected_index]
            
            with st.spinner(f"Rendering {selected_index} map..."):
                heatmap_data = get_heatmap_data(selected_index, tuple(target_tickers))
                
                if heatmap_data:
                    df_heat = pd.DataFrame(heatmap_data)
                    
                    fig_tree = px.treemap(
                        df_heat,
                        path=[px.Constant(selected_index), 'Ticker'],
                        values='Market Cap',
                        color='Change',
                        color_continuous_scale=['#ef4444', '#000000', '#10b981'], 
                        color_continuous_midpoint=0,
                        custom_data=['Change_Str'] 
                    )
                    
                    fig_tree.update_traces(
                        textinfo="label+text",
                        texttemplate="<b>%{label}</b><br>%{customdata[0]}",
                        textfont=dict(color="white", size=26), 
                        hovertemplate="<b>%{label}</b><br>Daily Change: %{customdata[0]}<br>Market Cap: $%{value:,.0f}<extra></extra>"
                    )
                    
                    fig_tree.update_layout(
                        margin=dict(t=10, l=0, r=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    
                    st.plotly_chart(fig_tree, use_container_width=True)