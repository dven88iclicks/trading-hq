"""
Trading & Portfolio Management System
The Watcher (scanner) + The Advisor (Telegram) + The HQ (Streamlit dashboard)

Start lokaal:  streamlit run main.py
Railway:       zie Procfile
"""

import os
import json
import time
import hmac
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def _secret(key: str, default: str = "") -> str:
    """Read from st.secrets first, fall back to environment variable."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


TELEGRAM_TOKEN     = _secret("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID   = _secret("TELEGRAM_CHAT_ID")
DASHBOARD_PASSWORD = _secret("DASHBOARD_PASSWORD")

PORTFOLIO_FILE = Path("portfolio.json")
SIGNALS_FILE   = Path("signals.json")
ADVICE_FILE    = Path("advice_log.json")
SCAN_INTERVAL  = 30 * 60  # seconds

# Top-50 volatiele aandelen watchlist (inclusief MRCY)
WATCHLIST = [
    "MRCY", "NVDA", "AMD",  "TSLA", "MSTR", "COIN", "PLTR", "RIVN",
    "SOFI", "HOOD", "GME",  "SPCE", "BB",   "SNDL", "TLRY", "ACB",
    "CGC",  "BYND", "DKNG", "PENN", "FUBO", "RBLX", "PATH", "AI",
    "SOUN", "HIMS", "CVNA", "OPEN", "LMND", "MARA", "RIOT", "HUT",
    "BITF", "SMCI", "IONQ", "QUBT", "RGTI", "LUNR", "RKLB", "ASTS",
    "JOBY", "ACHR", "EVGO", "CHPT", "PLUG", "FCEL", "ARRY", "STEM",
    "BLNK", "WKHS",
]

# ══════════════════════════════════════════════════════════════════════════════
# THREAD SAFETY
# ══════════════════════════════════════════════════════════════════════════════

_lock            = threading.Lock()
_watcher_started = False


# ══════════════════════════════════════════════════════════════════════════════
# DATA PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_portfolio() -> dict:
    with _lock:
        if PORTFOLIO_FILE.exists():
            try:
                return json.loads(PORTFOLIO_FILE.read_text())
            except Exception:
                pass
    # Default: jouw huidige MRCY-positie
    return {
        "MRCY": {
            "shares":    4.38356102,
            "avg_price": 89.43,
            "added":     datetime.now().isoformat(),
        }
    }


def save_portfolio(portfolio: dict) -> None:
    with _lock:
        PORTFOLIO_FILE.write_text(json.dumps(portfolio, indent=2, ensure_ascii=False))


def load_signals() -> list:
    with _lock:
        if SIGNALS_FILE.exists():
            try:
                return json.loads(SIGNALS_FILE.read_text())
            except Exception:
                pass
    return []


def append_signal(sig: dict) -> None:
    signals = load_signals()
    signals.insert(0, sig)
    with _lock:
        SIGNALS_FILE.write_text(json.dumps(signals[:200], indent=2))


def load_advice() -> list:
    with _lock:
        if ADVICE_FILE.exists():
            try:
                return json.loads(ADVICE_FILE.read_text())
            except Exception:
                pass
    return []


def save_advice(records: list) -> None:
    with _lock:
        ADVICE_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False))


def upsert_advice(record: dict) -> None:
    """Voeg toe of update op basis van advice_id."""
    records = load_advice()
    for i, r in enumerate(records):
        if r.get("advice_id") == record.get("advice_id"):
            records[i] = record
            save_advice(records)
            return
    records.insert(0, record)
    save_advice(records[:500])


# ══════════════════════════════════════════════════════════════════════════════
# TECHNISCHE ANALYSE
# ══════════════════════════════════════════════════════════════════════════════

def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=True).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=True).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def calc_bb(prices: pd.Series, period: int = 20, k: float = 2.0):
    """Return (upper, mid, lower) Bollinger Bands."""
    sma   = prices.rolling(period).mean()
    sigma = prices.rolling(period).std()
    return sma + k * sigma, sma, sma - k * sigma


def compute_signal(close: pd.Series) -> dict:
    """Compute BUY / HOLD / SELL signal for a Close price series."""
    if len(close) < 20:
        return {"signal": "DATA_ERROR", "rsi": None, "upper_bb": None, "lower_bb": None, "price": None}

    current   = float(close.iloc[-1])
    rsi_val   = float(calc_rsi(close).iloc[-1])
    upper, _, lower = calc_bb(close)
    upper_val = float(upper.iloc[-1])
    lower_val = float(lower.iloc[-1])

    if rsi_val < 30 or current <= lower_val:
        signal = "BUY"
    elif rsi_val > 70 or current >= upper_val:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal":   signal,
        "price":    round(current, 4),
        "rsi":      round(rsi_val, 1),
        "upper_bb": round(upper_val, 2),
        "lower_bb": round(lower_val, 2),
    }


def _get_close(data: pd.DataFrame, ticker: str, multi: bool) -> pd.Series:
    """Safely extract Close series from a yfinance download result."""
    if not multi:
        return data["Close"].dropna()
    try:
        # yfinance 0.2.x group_by='ticker': df[ticker][field]
        return data[ticker]["Close"].dropna()
    except (KeyError, TypeError):
        try:
            # Alternative structure: df[field][ticker]
            return data["Close"][ticker].dropna()
        except Exception:
            return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════════════════

def telegram_send(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# WATCHER  (draait als achtergrond-thread)
# ══════════════════════════════════════════════════════════════════════════════

def run_scan() -> list:
    """
    Scan alle watchlist-tickers.
    Stuurt Telegram-alerts voor portfolio-posities met KOOP/VERKOOP signaal.
    Geeft lijst van signalen terug (gesorteerd op RSI).
    """
    try:
        raw = yf.download(
            WATCHLIST, period="3mo", interval="1d",
            group_by="ticker", progress=False, threads=True,
        )
    except Exception:
        return []

    multi    = len(WATCHLIST) > 1
    results  = []
    portfolio = load_portfolio()

    for ticker in WATCHLIST:
        try:
            close = _get_close(raw, ticker, multi)
            if len(close) < 20:
                continue
            sig             = compute_signal(close)
            sig["ticker"]   = ticker
            sig["scanned_at"] = datetime.now().isoformat()
            results.append(sig)
        except Exception:
            continue

    # Sorteer op RSI (laagste = meest oversold)
    results.sort(key=lambda x: x.get("rsi") or 50)

    # Bewaar signalen en stuur Telegram-berichten
    for sig in results:
        ticker = sig["ticker"]
        signal = sig.get("signal")

        if signal in ("BUY", "SELL"):
            append_signal(sig)

        if ticker in portfolio and signal in ("BUY", "SELL"):
            pos     = portfolio[ticker]
            pnl_usd = (sig["price"] - pos["avg_price"]) * pos["shares"]
            pnl_pct = ((sig["price"] - pos["avg_price"]) / pos["avg_price"]) * 100
            emoji   = "🔴 VERKOPEN" if signal == "SELL" else "🟢 BIJKOPEN"

            msg = (
                f"<b>{emoji}: {ticker}</b>\n"
                f"Prijs: <b>${sig['price']:.2f}</b>\n"
                f"RSI: {sig['rsi']}\n"
                f"Jouw positie: {pos['shares']} aandelen @ ${pos['avg_price']}\n"
                f"P&amp;L: <b>${pnl_usd:+.2f} ({pnl_pct:+.1f}%)</b>\n"
            )

            # Smart Signal: vervangingsadvies bij SELL
            if signal == "SELL":
                candidates = [
                    s for s in results
                    if s["ticker"] not in portfolio
                    and s.get("signal") != "SELL"
                    and s.get("rsi") is not None
                ]
                if candidates:
                    best = candidates[0]  # laagste RSI
                    msg += (
                        f"\n\U0001f4a1 <b>Vervangings-advies:</b> {best['ticker']}\n"
                        f"RSI: {best['rsi']} · Prijs: ${best['price']:.2f} · "
                        f"Signaal: {best['signal']}"
                    )

            telegram_send(msg)

        # Extra alert voor sterke oversold kansen buiten portfolio
        elif ticker not in portfolio and (sig.get("rsi") or 50) < 25:
            telegram_send(
                f"\U0001f7e2 <b>KOOPKANS: {ticker}</b>\n"
                f"RSI: {sig['rsi']} (sterk oversold)\n"
                f"Prijs: ${sig['price']:.2f}  ·  Onderste BB: ${sig['lower_bb']:.2f}"
            )

    return results


def _watcher_loop() -> None:
    time.sleep(10)  # Kleine vertraging zodat Streamlit eerst opstart
    while True:
        try:
            run_scan()
        except Exception:
            pass
        time.sleep(SCAN_INTERVAL)


def start_watcher_once() -> None:
    global _watcher_started
    with _lock:
        if not _watcher_started:
            t = threading.Thread(target=_watcher_loop, daemon=True, name="watcher")
            t.start()
            _watcher_started = True


# ══════════════════════════════════════════════════════════════════════════════
# GECACHDE YFINANCE FUNCTIES  (module-niveau, zodat cache werkt over reruns)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def fetch_quotes(tickers: tuple) -> dict:
    """Haal meest recente slotkoersen op voor een lijst tickers."""
    tlist = list(tickers)
    if not tlist:
        return {}
    try:
        raw    = yf.download(tlist, period="5d", interval="1d", progress=False, group_by="ticker")
        multi  = len(tlist) > 1
        prices = {}
        for t in tlist:
            try:
                close = _get_close(raw, t, multi)
                prices[t] = float(close.dropna().iloc[-1])
            except Exception:
                prices[t] = None
        return prices
    except Exception:
        return {}


@st.cache_data(ttl=300)
def fetch_history(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """Haal OHLCV-data op voor één ticker."""
    try:
        return yf.download(ticker, period=period, interval="1d", progress=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def fetch_scan() -> list:
    """Scan de volledige watchlist (gecached 30 min)."""
    return run_scan()


@st.cache_data(ttl=3600)
def fetch_eur_usd() -> float:
    """Haal live EUR/USD koers op via yfinance. Fallback: 1.08."""
    try:
        data = yf.download("EURUSD=X", period="1d", interval="5m", progress=False)
        return float(data["Close"].dropna().iloc[-1])
    except Exception:
        return 1.08


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT  —  PAGINA CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Trading HQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

start_watcher_once()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background:#0f1117; color:#e2e8f0; font-family:system-ui,sans-serif; }
section[data-testid="stSidebar"] { background:#0c0f16 !important; border-right:1px solid #1e2436; }
section[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
h1,h2,h3,h4 { color:#f1f5f9 !important; }
.stButton>button { background:#007BFF !important; color:#fff !important; border:none !important;
    border-radius:6px !important; font-weight:600 !important; }
.stButton>button:hover { opacity:.85 !important; }
.stTextInput input, .stNumberInput input {
    background:#12161f !important; border:1px solid #1e2436 !important;
    color:#e2e8f0 !important; border-radius:6px !important; }
.stTabs [data-baseweb="tab"] { color:#64748b !important; }
.stTabs [aria-selected="true"] { color:#007BFF !important; border-bottom-color:#007BFF !important; }
div[data-testid="stMetricValue"] { color:#007BFF !important; font-weight:800 !important; }
.pnl-pos { color:#4ade80; font-weight:700; }
.pnl-neg { color:#f87171; font-weight:700; }
.card { background:#12161f; border:1px solid #1e2436; border-radius:8px; padding:14px 18px; }
.badge-buy  { background:#14532d; color:#4ade80; padding:3px 10px;
    border-radius:999px; font-size:.75rem; font-weight:700; }
.badge-sell { background:#450a0a; color:#f87171; padding:3px 10px;
    border-radius:999px; font-size:.75rem; font-weight:700; }
.badge-hold { background:#1e2d3d; color:#60a5fa; padding:3px 10px;
    border-radius:999px; font-size:.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN GATE
# ══════════════════════════════════════════════════════════════════════════════

def _get_active_password() -> str:
    """Actief wachtwoord: lokaal bestand heeft prioriteit boven secrets/env."""
    pwd_file = Path("dashboard_password.txt")
    if pwd_file.exists():
        stored = pwd_file.read_text().strip()
        if stored:
            return stored
    return DASHBOARD_PASSWORD


def check_login() -> bool:
    if st.session_state.get("authenticated"):
        return True

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🔒 Trading HQ")
        st.markdown("<p style='color:#64748b'>Beveiligd portaal</p>", unsafe_allow_html=True)
        with st.form("login_form"):
            pwd = st.text_input("Wachtwoord", type="password",
                                label_visibility="collapsed", placeholder="Wachtwoord")
            submitted = st.form_submit_button("Inloggen →", use_container_width=True)
            if submitted:
                active_pwd = _get_active_password()
                if not active_pwd:
                    st.warning("Geen wachtwoord geconfigureerd. Stel DASHBOARD_PASSWORD in via Railway Variables.")
                elif hmac.compare_digest(pwd, active_pwd):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Onjuist wachtwoord.")
    return False


if not check_login():
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📈 Trading HQ")
    st.divider()
    page = st.radio(
        "nav",
        ["Portfolio & P&L", "Markt Scan", "Investeer Advies", "Signaalgeschiedenis", "Transacties"],
        label_visibility="collapsed",
    )
    st.divider()

    if st.button("🔄 Nu Scannen", use_container_width=True):
        fetch_scan.clear()
        with st.spinner("Scannen..."):
            run_scan()
        st.success("Scan voltooid.")
        st.rerun()

    st.caption("Automatische scan elke 30 min.")
    st.divider()

    if st.button("🚪 Uitloggen", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()


portfolio = load_portfolio()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Portfolio & P&L
# ══════════════════════════════════════════════════════════════════════════════

if page == "Portfolio & P&L":
    st.markdown("## 📊 Portfolio & P&L")

    if not portfolio:
        st.info("Geen posities. Voeg ze toe via 'Transacties'.")
        st.stop()

    tickers = list(portfolio.keys())
    prices  = fetch_quotes(tuple(sorted(tickers)))

    # ── Samenvatting ─────────────────────────────────────────────────────────
    total_cost  = sum(portfolio[t]["shares"] * portfolio[t]["avg_price"] for t in tickers)
    total_value = sum(
        portfolio[t]["shares"] * (prices.get(t) or portfolio[t]["avg_price"])
        for t in tickers
    )
    total_pnl     = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Totale Waarde",    f"${total_value:,.2f}")
    c2.metric("Kosten Basis",     f"${total_cost:,.2f}")
    c3.metric("Unrealized P&L",   f"${total_pnl:+,.2f}", f"{total_pnl_pct:+.1f}%")
    c4.metric("Posities",         len(tickers))

    st.divider()

    # ── Per-positie detail ────────────────────────────────────────────────────
    for ticker in tickers:
        pos           = portfolio[ticker]
        current_price = prices.get(ticker)
        hist          = fetch_history(ticker, "1mo")

        with st.expander(f"**{ticker}** — {pos['shares']} aandelen", expanded=True):
            col_chart, col_stats = st.columns([3, 1])

            with col_chart:
                if not hist.empty:
                    close_s      = hist["Close"].squeeze()
                    upper, mid, lower = calc_bb(close_s)
                    rsi_s        = calc_rsi(close_s)

                    fig = make_subplots(
                        rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04,
                        subplot_titles=[f"{ticker} — Laatste 30 dagen", "RSI (14)"],
                    )

                    # Candlestick
                    fig.add_trace(go.Candlestick(
                        x=hist.index,
                        open=hist["Open"].squeeze(), high=hist["High"].squeeze(),
                        low=hist["Low"].squeeze(),   close=close_s,
                        name=ticker,
                        increasing_line_color="#4ade80",
                        decreasing_line_color="#f87171",
                    ), row=1, col=1)

                    # Bollinger Bands
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=upper, name="BB Hoog",
                        line=dict(color="#60a5fa", dash="dash", width=1),
                    ), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=lower, name="BB Laag",
                        line=dict(color="#f59e0b", dash="dash", width=1),
                        fill="tonexty", fillcolor="rgba(96,165,250,0.06)",
                    ), row=1, col=1)

                    # Inkoop-prijs lijn
                    fig.add_hline(
                        y=pos["avg_price"], line_color="#a78bfa", line_dash="dot",
                        row=1, col=1, annotation_text=f"Inkoop ${pos['avg_price']}",
                        annotation_font_color="#a78bfa",
                    )

                    # RSI
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=rsi_s, name="RSI",
                        line=dict(color="#fb923c", width=1.5),
                    ), row=2, col=1)
                    fig.add_hline(y=70, line_color="#f87171", line_dash="dash",
                                  line_width=1, row=2, col=1)
                    fig.add_hline(y=30, line_color="#4ade80", line_dash="dash",
                                  line_width=1, row=2, col=1)

                    fig.update_layout(
                        height=400, paper_bgcolor="#12161f", plot_bgcolor="#12161f",
                        font=dict(color="#e2e8f0", size=11),
                        xaxis_rangeslider_visible=False,
                        showlegend=False,
                        margin=dict(l=0, r=0, t=30, b=0),
                    )
                    fig.update_xaxes(gridcolor="#1e2436", showgrid=True)
                    fig.update_yaxes(gridcolor="#1e2436", showgrid=True)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Geen historische data beschikbaar voor {ticker}.")

            with col_stats:
                if current_price and not hist.empty:
                    pnl_usd   = (current_price - pos["avg_price"]) * pos["shares"]
                    pnl_pct   = ((current_price - pos["avg_price"]) / pos["avg_price"]) * 100
                    sig_data  = compute_signal(hist["Close"].squeeze())
                    signal    = sig_data.get("signal", "?")

                    badge_map = {"BUY": "buy", "SELL": "sell", "HOLD": "hold"}
                    label_map = {"BUY": "BIJKOPEN", "SELL": "VERKOPEN", "HOLD": "VASTHOUDEN"}
                    badge_cls = badge_map.get(signal, "hold")
                    label     = label_map.get(signal, signal)
                    pnl_cls   = "pnl-pos" if pnl_usd >= 0 else "pnl-neg"

                    st.markdown(f"""
                    <div class="card">
                        <div style="margin-bottom:12px">
                            <span class="badge-{badge_cls}">{label}</span>
                        </div>
                        <div style="font-size:.7rem;color:#64748b">Huidige prijs</div>
                        <div style="font-size:1.3rem;font-weight:800;color:#f1f5f9">${current_price:.2f}</div>
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">Gemiddelde inkoop</div>
                        <div style="color:#94a3b8">${pos['avg_price']:.2f}</div>
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">Aandelen</div>
                        <div style="color:#94a3b8">{pos['shares']}</div>
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">Unrealized P&L</div>
                        <div class="{pnl_cls}">${pnl_usd:+.2f}<br>{pnl_pct:+.1f}%</div>
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">RSI (14)</div>
                        <div style="font-size:1rem;color:#fb923c;font-weight:700">{sig_data.get('rsi', '—')}</div>
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">BB Hoog / Laag</div>
                        <div style="font-size:.8rem;color:#60a5fa">${sig_data.get('upper_bb', '—')} / ${sig_data.get('lower_bb', '—')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Prijsdata niet beschikbaar.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Markt Scan
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Markt Scan":
    st.markdown("## 🔍 Markt Scan — Top 50 Volatile")
    st.caption("Gesorteerd op RSI · laagste RSI = meest oversold = potentieel koopsignaal")

    with st.spinner("Data ophalen (kan 30 seconden duren)..."):
        scan_results = fetch_scan()

    if not scan_results:
        st.warning("Geen scanresultaten. Probeer opnieuw via 'Nu Scannen' in de sidebar.")
    else:
        emoji_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🔵"}
        rows = []
        for s in scan_results:
            signal = s.get("signal", "?")
            rows.append({
                " ":            emoji_map.get(signal, "⚪"),
                "Ticker":       s.get("ticker", "?"),
                "Prijs":        f"${s['price']:.2f}"    if s.get("price")    else "—",
                "RSI":          f"{s['rsi']:.1f}"       if s.get("rsi")      else "—",
                "BB Hoog":      f"${s['upper_bb']:.2f}" if s.get("upper_bb") else "—",
                "BB Laag":      f"${s['lower_bb']:.2f}" if s.get("lower_bb") else "—",
                "Signaal":      signal,
                "Portfolio":    "✓" if s.get("ticker") in portfolio else "",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Gescand: {len(scan_results)} tickers  ·  "
                   f"{sum(1 for r in rows if r['Signaal']=='BUY')} KOOP  ·  "
                   f"{sum(1 for r in rows if r['Signaal']=='SELL')} VERKOOP")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Investeer Advies
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Investeer Advies":
    st.markdown("## 💰 Investeer Advies")
    st.caption(
        "Voer je beschikbare budget in. De tool berekent welke aandelen uit de watchlist "
        "het beste koopsignaal hebben en hoeveel je kunt kopen. Geef daarna aan of je het advies hebt gevolgd."
    )

    # ── EUR/USD koers ─────────────────────────────────────────────────────────
    eur_usd = fetch_eur_usd()

    col_budget, col_rate = st.columns([2, 1])
    with col_budget:
        budget_eur = st.number_input(
            "Beschikbaar budget (€)", min_value=1.0, value=200.0,
            step=10.0, format="%.2f",
        )
    with col_rate:
        eur_usd_input = st.number_input(
            "EUR/USD koers", min_value=0.5, value=round(eur_usd, 4),
            step=0.001, format="%.4f",
            help="Live koers via yfinance. Pas handmatig aan indien gewenst.",
        )

    budget_usd = budget_eur * eur_usd_input
    st.markdown(
        f"**Budget:** €{budget_eur:,.2f} → "
        f"<span style='color:#007BFF;font-weight:700'>${budget_usd:,.2f}</span>",
        unsafe_allow_html=True,
    )

    if st.button("🔍 Bereken Advies", use_container_width=False):
        st.session_state["show_advice"] = True
        fetch_scan.clear()

    st.divider()

    # ── Adviezen tonen ────────────────────────────────────────────────────────
    if st.session_state.get("show_advice"):
        with st.spinner("Signalen ophalen..."):
            scan = fetch_scan()

        buy_signals = [s for s in scan if s.get("signal") == "BUY" and s.get("price")]
        buy_signals = sorted(buy_signals, key=lambda x: x.get("rsi") or 50)[:5]

        if not buy_signals:
            st.info("Geen BUY-signalen op dit moment. Probeer later opnieuw.")
        else:
            st.markdown(f"### Top {len(buy_signals)} koopsignalen voor €{budget_eur:,.0f}")
            st.caption("Gerangschikt op RSI (laagste = sterkst oversold)")

            for rank, sig in enumerate(buy_signals, start=1):
                ticker     = sig["ticker"]
                price_usd  = sig["price"]
                price_eur  = price_usd / eur_usd_input
                max_shares = budget_usd / price_usd
                cost_usd   = max_shares * price_usd
                cost_eur   = cost_usd / eur_usd_input

                advice_id  = f"{ticker}_{datetime.now().strftime('%Y%m%d')}"
                in_port    = ticker in portfolio

                # Bestaande log-entry ophalen
                existing = next(
                    (r for r in load_advice() if r.get("advice_id") == advice_id), None
                )
                outcome = existing.get("outcome", "pending") if existing else "pending"

                with st.container():
                    # Header rij
                    hc1, hc2, hc3, hc4, hc5 = st.columns([1, 2, 2, 2, 2])
                    hc1.markdown(f"**#{rank}**")
                    hc2.markdown(f"**{ticker}**{'  ✓ portfolio' if in_port else ''}")
                    hc3.markdown(f"Prijs: **${price_usd:.2f}** (€{price_eur:.2f})")
                    hc4.markdown(f"RSI: **{sig['rsi']}**")

                    if outcome == "gevolgt":
                        hc5.markdown(
                            '<span style="color:#4ade80;font-weight:700">✓ Gevolgt</span>',
                            unsafe_allow_html=True,
                        )
                    elif outcome == "niet_gevolgt":
                        hc5.markdown(
                            '<span style="color:#f87171;font-weight:700">✗ Niet gevolgt</span>',
                            unsafe_allow_html=True,
                        )

                    # Advies-box
                    st.markdown(
                        f'<div class="card" style="margin-bottom:4px">'
                        f'Met <b>€{budget_eur:,.2f}</b> kun je maximaal '
                        f'<b>{max_shares:.4f} aandelen {ticker}</b> kopen '
                        f'(≈ ${cost_usd:,.2f} / €{cost_eur:,.2f}) '
                        f'· BB Laag: ${sig["lower_bb"]:.2f} · BB Hoog: ${sig["upper_bb"]:.2f}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Opvolg-knoppen (alleen zichtbaar als nog pending)
                    if outcome == "pending":
                        btn_col1, btn_col2, _ = st.columns([1.5, 1.5, 4])
                        follow  = btn_col1.button("✓ Advies gevolgt",     key=f"follow_{advice_id}")
                        no_follow = btn_col2.button("✗ Niet gevolgt",     key=f"nope_{advice_id}")

                        if no_follow:
                            upsert_advice({
                                "advice_id":      advice_id,
                                "timestamp":      datetime.now().isoformat(),
                                "ticker":         ticker,
                                "budget_eur":     budget_eur,
                                "budget_usd":     round(budget_usd, 2),
                                "advised_price_usd": price_usd,
                                "advised_price_eur": round(price_eur, 4),
                                "advised_shares": round(max_shares, 8),
                                "rsi_at_advice":  sig["rsi"],
                                "outcome":        "niet_gevolgt",
                                "actual_price_usd": None,
                                "actual_shares":    None,
                            })
                            st.rerun()

                        if follow:
                            st.session_state[f"enter_price_{advice_id}"] = True

                    # Prijsinvulformulier na "Gevolgt" klik
                    if st.session_state.get(f"enter_price_{advice_id}") and outcome == "pending":
                        with st.form(f"form_actual_{advice_id}"):
                            st.markdown(f"**Transactie invoeren — {ticker}**")
                            fc1, fc2 = st.columns(2)
                            with fc1:
                                actual_price = st.number_input(
                                    "Aankooprijs ($)", min_value=0.01,
                                    value=round(price_usd, 2), step=0.01, format="%.4f",
                                )
                            with fc2:
                                actual_shares = st.number_input(
                                    "Aantal aandelen gekocht", min_value=0.00000001,
                                    value=round(max_shares, 8), step=0.00000001, format="%.8f",
                                )
                            if st.form_submit_button("Opslaan & toevoegen aan portfolio"):
                                # Sla advies op als gevolgt
                                upsert_advice({
                                    "advice_id":         advice_id,
                                    "timestamp":         datetime.now().isoformat(),
                                    "ticker":            ticker,
                                    "budget_eur":        budget_eur,
                                    "budget_usd":        round(budget_usd, 2),
                                    "advised_price_usd": price_usd,
                                    "advised_price_eur": round(price_eur, 4),
                                    "advised_shares":    round(max_shares, 8),
                                    "rsi_at_advice":     sig["rsi"],
                                    "outcome":           "gevolgt",
                                    "actual_price_usd":  actual_price,
                                    "actual_shares":     actual_shares,
                                })
                                # Voeg toe aan portfolio (gemiddeld als al aanwezig)
                                if ticker in portfolio:
                                    old   = portfolio[ticker]
                                    total_shares = old["shares"] + actual_shares
                                    avg_price    = (
                                        (old["shares"] * old["avg_price"] + actual_shares * actual_price)
                                        / total_shares
                                    )
                                    portfolio[ticker]["shares"]    = round(total_shares, 8)
                                    portfolio[ticker]["avg_price"] = round(avg_price, 4)
                                else:
                                    portfolio[ticker] = {
                                        "shares":    round(actual_shares, 8),
                                        "avg_price": round(actual_price, 4),
                                        "added":     datetime.now().isoformat(),
                                    }
                                save_portfolio(portfolio)
                                fetch_quotes.clear()
                                del st.session_state[f"enter_price_{advice_id}"]
                                st.success(
                                    f"✓ {actual_shares:.8f} {ticker} @ ${actual_price:.4f} "
                                    f"toegevoegd aan portfolio."
                                )
                                st.rerun()

                    st.markdown("<hr style='border-color:#1e2436;margin:8px 0'>", unsafe_allow_html=True)

    # ── Adviesgeschiedenis ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Adviesgeschiedenis")
    advice_log = load_advice()
    if not advice_log:
        st.info("Nog geen adviezen gelogd.")
    else:
        rows = []
        for a in advice_log[:100]:
            outcome_label = {
                "gevolgt":      "✓ Gevolgt",
                "niet_gevolgt": "✗ Niet gevolgt",
                "pending":      "⏳ Nog open",
            }.get(a.get("outcome", "pending"), "—")

            actual_price = a.get("actual_price_usd")
            advised_price = a.get("advised_price_usd", 0)
            diff_str = ""
            if actual_price and advised_price:
                diff = actual_price - advised_price
                diff_str = f"${diff:+.2f}"

            rows.append({
                "Datum":         a.get("timestamp", "")[:10],
                "Ticker":        a.get("ticker", "?"),
                "Budget":        f"€{a.get('budget_eur', 0):,.0f}",
                "Geadv. prijs":  f"${advised_price:.2f}" if advised_price else "—",
                "Werkelijk":     f"${actual_price:.4f}" if actual_price else "—",
                "Verschil":      diff_str,
                "Aandelen":      f"{a.get('actual_shares', a.get('advised_shares', 0)):.4f}",
                "RSI":           a.get("rsi_at_advice", "—"),
                "Status":        outcome_label,
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Signaalgeschiedenis
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Signaalgeschiedenis":
    st.markdown("## 📋 Signaalgeschiedenis")

    signals = load_signals()
    if not signals:
        st.info("Nog geen signalen gelogd. Wacht op de volgende scan of klik 'Nu Scannen'.")
    else:
        label_map = {"BUY": "🟢 KOPEN", "SELL": "🔴 VERKOPEN", "HOLD": "🔵 HOLD"}
        rows = []
        for s in signals[:150]:
            ts = s.get("scanned_at", "")[:16].replace("T", " ")
            signal = s.get("signal", "?")
            rows.append({
                "Tijdstip": ts,
                "Ticker":   s.get("ticker", "?"),
                "Signaal":  label_map.get(signal, signal),
                "Prijs":    f"${s['price']:.2f}" if s.get("price") else "—",
                "RSI":      f"{s['rsi']:.1f}"    if s.get("rsi")   else "—",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"{len(signals)} signalen opgeslagen")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Transacties
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Transacties":
    st.markdown("## 💼 Portefeuille Beheer")

    # ── Positie toevoegen / wijzigen ─────────────────────────────────────────
    st.markdown("### Positie Toevoegen of Wijzigen")
    with st.form("add_position"):
        c1, c2, c3 = st.columns(3)
        with c1:
            new_ticker = st.text_input("Ticker (bijv. NVDA)").strip().upper()
        with c2:
            new_shares = st.number_input("Aantal aandelen", min_value=0.0,
                                         value=0.0, step=0.00000001, format="%.8f")
        with c3:
            new_price  = st.number_input("Gemiddelde inkoopprijs ($)", min_value=0.0,
                                         value=0.0, step=0.01, format="%.2f")
        if st.form_submit_button("Opslaan", use_container_width=True):
            if new_ticker and new_shares > 0 and new_price > 0:
                portfolio[new_ticker] = {
                    "shares":    new_shares,
                    "avg_price": new_price,
                    "added":     datetime.now().isoformat(),
                }
                save_portfolio(portfolio)
                fetch_quotes.clear()
                st.success(f"✓ {new_ticker}: {new_shares} aandelen @ ${new_price:.2f} opgeslagen.")
                st.rerun()
            else:
                st.error("Vul alle velden correct in (ticker, aandelen > 0, prijs > 0).")

    st.divider()

    # ── Wachtwoord wijzigen ───────────────────────────────────────────────────
    st.markdown("### Wachtwoord Wijzigen")
    st.caption(
        "Het nieuwe wachtwoord wordt opgeslagen in `dashboard_password.txt`. "
        "Op Railway moet je na een wijziging ook de Environment Variable bijwerken "
        "zodat het wachtwoord na een herstart bewaard blijft."
    )
    PWD_FILE = Path("dashboard_password.txt")
    with st.form("change_password"):
        cp1, cp2 = st.columns(2)
        with cp1:
            pwd_old = st.text_input("Huidig wachtwoord", type="password")
        with cp2:
            pwd_new = st.text_input("Nieuw wachtwoord", type="password")
        pwd_confirm = st.text_input("Bevestig nieuw wachtwoord", type="password")
        if st.form_submit_button("Wachtwoord opslaan"):
            current_pwd = PWD_FILE.read_text().strip() if PWD_FILE.exists() else DASHBOARD_PASSWORD
            if not current_pwd:
                st.error("Geen huidig wachtwoord geconfigureerd.")
            elif not hmac.compare_digest(pwd_old, current_pwd):
                st.error("Huidig wachtwoord onjuist.")
            elif len(pwd_new) < 8:
                st.error("Nieuw wachtwoord moet minimaal 8 tekens bevatten.")
            elif pwd_new != pwd_confirm:
                st.error("Wachtwoorden komen niet overeen.")
            else:
                PWD_FILE.write_text(pwd_new)
                st.success("Wachtwoord opgeslagen. Je wordt uitgelogd — log opnieuw in.")
                st.session_state.authenticated = False
                st.rerun()

    st.divider()

    # ── Huidige posities ─────────────────────────────────────────────────────
    st.markdown("### Huidige Posities")
    if not portfolio:
        st.info("Nog geen posities.")
    else:
        for ticker in list(portfolio.keys()):
            pos = portfolio[ticker]
            c_a, c_b, c_c, c_d = st.columns([2, 2, 2, 1])
            c_a.markdown(f"**{ticker}**")
            c_b.markdown(f"{pos['shares']} aandelen")
            c_c.markdown(f"@ ${pos['avg_price']:.2f}")
            if c_d.button("Verwijder", key=f"del_{ticker}"):
                del portfolio[ticker]
                save_portfolio(portfolio)
                fetch_quotes.clear()
                st.rerun()
