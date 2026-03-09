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
import secrets
import threading
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
    """Environment variable first (Railway), then st.secrets (lokaal)."""
    env_val = os.getenv(key)
    if env_val:
        return env_val
    try:
        val = st.secrets.get(key, default)
        return val if val else default
    except Exception:
        return default


TELEGRAM_TOKEN     = _secret("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID   = _secret("TELEGRAM_CHAT_ID")
DASHBOARD_PASSWORD = _secret("DASHBOARD_PASSWORD")

PORTFOLIO_FILE      = Path("portfolio.json")
SIGNALS_FILE        = Path("signals.json")
ADVICE_FILE         = Path("advice_log.json")
LAST_SIGNALS_FILE   = Path("last_signals.json")
SETTINGS_FILE       = Path("settings.json")
SESSIONS_FILE       = Path("sessions.json")
SCAN_INTERVAL       = 15 * 60  # seconds
ALERT_COOLDOWN_H    = 6        # minimaal 6 uur tussen dezelfde alert per ticker
SESSION_TIMEOUT_MIN = 10       # automatisch uitloggen na X minuten inactiviteit

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
# MARKTUREN  (NYSE / NASDAQ — ET-tijdzone)
# ══════════════════════════════════════════════════════════════════════════════

_ET = ZoneInfo("America/New_York")
_NL = ZoneInfo("Europe/Amsterdam")

# NYSE feestdagen 2025-2027 (officieel gesloten)
_US_HOLIDAYS = {
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}


def _is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _US_HOLIDAYS


def _next_open(from_dt: datetime) -> datetime:
    """Geeft de eerstvolgende marktopening terug als datetime (ET)."""
    d = from_dt.date()
    # Als vandaag een handelsdag is en de markt nog niet open was, is het vandaag 09:30
    if _is_trading_day(d):
        open_today = from_dt.replace(hour=9, minute=30, second=0, microsecond=0)
        if from_dt < open_today:
            return open_today
    # Zoek volgende handelsdag
    d += timedelta(days=1)
    for _ in range(10):
        if _is_trading_day(d):
            return datetime(d.year, d.month, d.day, 9, 30, tzinfo=_ET)
        d += timedelta(days=1)
    return datetime(d.year, d.month, d.day, 9, 30, tzinfo=_ET)


def market_status() -> dict:
    """Geeft live marktstatusinfo terug in ET én NL-tijd."""
    now_et    = datetime.now(_ET)
    today     = now_et.date()
    open_et   = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_et  = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)

    trading_day = _is_trading_day(today)

    if trading_day and open_et <= now_et < close_et:
        status   = "open"
        sluit_in = close_et - now_et
        m        = int(sluit_in.total_seconds() // 60)
        msg      = f"Markt OPEN  ·  sluit over {m // 60}u {m % 60}m (ET 16:00)"
        color    = "#4ade80"
    elif trading_day and now_et < open_et:
        status   = "pre-market"
        opent_in = open_et - now_et
        m        = int(opent_in.total_seconds() // 60)
        msg      = f"Pre-market  ·  markt opent over {m // 60}u {m % 60}m"
        color    = "#fb923c"
    else:
        if today.weekday() >= 5:
            status = "weekend"
        elif today in _US_HOLIDAYS:
            status = "feestdag"
        else:
            status = "nabeurs"
        nxt      = _next_open(now_et)
        nxt_nl   = nxt.astimezone(_NL)
        dag_nl   = ["ma", "di", "wo", "do", "vr", "za", "zo"][nxt_nl.weekday()]
        msg      = f"Markt GESLOTEN  ·  opent {dag_nl} {nxt_nl.strftime('%d %b om %H:%M')} NL-tijd"
        color    = "#f87171"

    next_open_et = _next_open(now_et)
    next_open_nl = next_open_et.astimezone(_NL)
    now_nl       = now_et.astimezone(_NL)

    return {
        "is_open":       status == "open",
        "status":        status,
        "msg":           msg,
        "color":         color,
        "now_nl":        now_nl,
        "now_et":        now_et,
        "next_open_nl":  next_open_nl,
        "next_open_et":  next_open_et,
    }


# ══════════════════════════════════════════════════════════════════════════════
# THREAD SAFETY
# ══════════════════════════════════════════════════════════════════════════════

_lock            = threading.Lock()
_scan_lock       = threading.Lock()   # voorkomt gelijktijdige scan-runs
_watcher_started = False


# Module-level earnings cache (thread-safe via _lock, TTL 24 uur)
_earnings_cache: dict = {}   # {ticker: (fetched_at: datetime, earnings_date: date | None)}


def _get_earnings_date(ticker: str):
    """
    Haal de eerstvolgende earnings date op via yfinance calendar.
    Gecached 24 uur in geheugen. Geeft None terug als onbekend of fout.
    """
    with _lock:
        cached = _earnings_cache.get(ticker)
    if cached:
        fetched_at, ed = cached
        if (datetime.now() - fetched_at).total_seconds() < 86400:
            return ed
    ed = None
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None:
            pass
        elif isinstance(cal, dict):
            dates = cal.get("Earnings Date") or cal.get("earningsDate") or []
            if dates:
                ed = pd.Timestamp(dates[0]).date()
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            col = cal.get("Earnings Date") or cal.iloc[:, 0]
            if col is not None and len(col):
                ed = pd.Timestamp(col.iloc[0]).date()
    except Exception:
        ed = None
    with _lock:
        _earnings_cache[ticker] = (datetime.now(), ed)
    return ed


def _earnings_within_days(ticker: str, days: int = 7) -> bool:
    """True als er een earnings-datum bekend is binnen 'days' kalenderdagen."""
    try:
        ed = _get_earnings_date(ticker)
        if ed is None:
            return False
        delta = (ed - date.today()).days
        return 0 <= delta <= days
    except Exception:
        return False


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
    # Werkelijk portfolio — exacte posities per 2026-03-09 (uit transactie-screenshots)
    _now = datetime.now().isoformat()
    return {
        "MRCY": {"shares": 4.38356102,   "avg_price": 89.43, "buy_date": "2024-09-01",
                 "added": _now, "status": "ACTIEF"},
        "FUBO": {"shares": 198.27586206, "avg_price": 1.16,  "buy_date": "2026-03-09",
                 "added": _now, "status": "PENDING"},
        "EVGO": {"shares": 89.93667860,  "avg_price": 2.169, "buy_date": "2026-03-09",
                 "added": _now, "status": "PENDING"},
        "ARRY": {"shares": 17.71771771,  "avg_price": 6.66,  "buy_date": "2026-03-09",
                 "added": _now, "status": "PENDING"},
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


def load_last_signals() -> dict:
    """Laad het laatste verzonden signaal per ticker {ticker: {signal, price, sent_at}}."""
    with _lock:
        if LAST_SIGNALS_FILE.exists():
            try:
                return json.loads(LAST_SIGNALS_FILE.read_text())
            except Exception:
                pass
    return {}


def save_last_signals(data: dict) -> None:
    with _lock:
        LAST_SIGNALS_FILE.write_text(json.dumps(data, indent=2))


def load_settings() -> dict:
    with _lock:
        if SETTINGS_FILE.exists():
            try:
                return json.loads(SETTINGS_FILE.read_text())
            except Exception:
                pass
    return {"budget_eur": 0.0}


def save_settings(s: dict) -> None:
    with _lock:
        SETTINGS_FILE.write_text(json.dumps(s, indent=2, ensure_ascii=False))


# ── Session management ────────────────────────────────────────────────────────

def _load_sessions() -> dict:
    with _lock:
        if SESSIONS_FILE.exists():
            try:
                return json.loads(SESSIONS_FILE.read_text())
            except Exception:
                pass
    return {}


def _save_sessions(data: dict) -> None:
    with _lock:
        SESSIONS_FILE.write_text(json.dumps(data, indent=2))


def _create_session() -> str:
    """Maak een nieuw sessie-token, sla op, verwijder verlopen sessies."""
    token    = secrets.token_urlsafe(32)
    now_str  = datetime.now().isoformat()
    sessions = _load_sessions()
    sessions[token] = {"created": now_str, "last_activity": now_str}
    # Opruimen: verwijder sessies ouder dan 24 uur
    cutoff = datetime.now().timestamp() - 86400
    sessions = {
        k: v for k, v in sessions.items()
        if datetime.fromisoformat(v.get("last_activity", now_str)).timestamp() > cutoff
    }
    sessions[token] = {"created": now_str, "last_activity": now_str}
    _save_sessions(sessions)
    return token


def _is_session_valid(token: str) -> bool:
    """Controleer of het token bestaat én de inactiviteitsgrens niet overschreden is."""
    if not token:
        return False
    sessions = _load_sessions()
    session  = sessions.get(token)
    if not session:
        return False
    try:
        elapsed_min = (
            datetime.now() - datetime.fromisoformat(session["last_activity"])
        ).total_seconds() / 60
        return elapsed_min < SESSION_TIMEOUT_MIN
    except Exception:
        return False


def _touch_session(token: str) -> None:
    """Ververs de last_activity timestamp (bij elke pagina-interactie)."""
    sessions = _load_sessions()
    if token in sessions:
        sessions[token]["last_activity"] = datetime.now().isoformat()
        _save_sessions(sessions)


def _delete_session(token: str) -> None:
    """Verwijder sessie bij uitloggen."""
    sessions = _load_sessions()
    sessions.pop(token, None)
    _save_sessions(sessions)


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


def forecast_48h(close: pd.Series) -> dict:
    """
    Schat de verwachte prijsbeweging in de komende 48 uur op basis van:
    - Kortetermijn momentum (5-daags)
    - RSI-richting (stijgend of dalend)
    - Bollinger Band positie (mean reversion)
    - Recente volatiliteit (ATR-gebaseerd prijsbereik)
    """
    if len(close) < 10:
        return {"richting": "onbekend", "target_laag": None, "target_hoog": None,
                "verwachting_pct": 0.0, "onderbouwing": "Onvoldoende data."}

    current   = float(close.iloc[-1])
    rsi_val   = float(calc_rsi(close).iloc[-1])
    rsi_prev  = float(calc_rsi(close).iloc[-2])
    upper, mid, lower = calc_bb(close)

    # 5-daags momentum: gemiddelde dagelijkse return
    returns_5d    = close.pct_change().iloc[-5:]
    momentum      = float(returns_5d.mean())        # gem. dagelijks rendement
    recent_vol    = float(returns_5d.std())          # dagelijkse volatiliteit

    # 48u bereik op basis van volatiliteit (2 handelsdagen ≈ √2 × dagvol)
    vol_48h = recent_vol * (2 ** 0.5)
    range_h = current * (1 + vol_48h)
    range_l = current * (1 - vol_48h)

    # Richtingsbepaling: combineer momentum + RSI-richting + BB-positie
    score = 0

    # Momentum
    if momentum > 0.005:
        score += 2
    elif momentum > 0:
        score += 1
    elif momentum < -0.005:
        score -= 2
    else:
        score -= 1

    # RSI trend (stijgend of dalend)
    if rsi_val > rsi_prev:
        score += 1
    else:
        score -= 1

    # BB mean reversion: prijs dicht bij onderste band = verwacht herstel
    bb_range = float(upper.iloc[-1]) - float(lower.iloc[-1])
    if bb_range > 0:
        bb_pos = (current - float(lower.iloc[-1])) / bb_range  # 0=onderaan, 1=bovenaan
        if bb_pos < 0.2:
            score += 2   # oversold → verwacht herstel omhoog
        elif bb_pos > 0.8:
            score -= 2   # overbought → verwacht correctie

    # Verwacht rendement: schaal score naar % beweging
    verwachting_pct = round(score * (recent_vol * 100 * 0.4), 2)
    target_mid      = current * (1 + verwachting_pct / 100)
    target_laag     = round(min(range_l, target_mid * 0.995), 4)
    target_hoog     = round(max(range_h, target_mid * 1.005), 4)

    if score >= 2:
        richting = "omhoog"
    elif score <= -2:
        richting = "omlaag"
    else:
        richting = "neutraal"

    redenen = []
    if momentum > 0:
        redenen.append(f"positief 5d-momentum (+{momentum*100:.2f}%/dag)")
    else:
        redenen.append(f"negatief 5d-momentum ({momentum*100:.2f}%/dag)")
    if rsi_val > rsi_prev:
        redenen.append(f"RSI stijgend ({rsi_prev:.0f}→{rsi_val:.0f})")
    else:
        redenen.append(f"RSI dalend ({rsi_prev:.0f}→{rsi_val:.0f})")
    if bb_range > 0:
        if bb_pos < 0.3:
            redenen.append("prijs nabij BB-onderkant (herstelkans)")
        elif bb_pos > 0.7:
            redenen.append("prijs nabij BB-bovenkant (correctierisico)")

    return {
        "richting":        richting,
        "verwachting_pct": verwachting_pct,
        "target_laag":     target_laag,
        "target_hoog":     target_hoog,
        "onderbouwing":    " · ".join(redenen),
    }


def compute_signal(close: pd.Series, volume: pd.Series = None) -> dict:
    """
    Institutional-grade signaalberekening:
    - STRONG BUY : RSI < 30 + prijs ≤ BB-laag + prijs ≥ $5 + prijs > SMA-200
                   + volume > 1.2× 10-daags gemiddelde  (volume-bevestiging)
    - BUY        : zelfde als STRONG BUY maar zonder volume-bevestiging
    - SELL       : RSI > 70 ÉN prijs ≥ bovenste BB
    - HOLD       : alles daartussenin of filters niet gehaald
    """
    if len(close) < 20:
        return {"signal": "DATA_ERROR", "rsi": None, "upper_bb": None, "lower_bb": None,
                "price": None, "sma200": None, "vol_surge": False}

    current   = float(close.iloc[-1])
    rsi_val   = float(calc_rsi(close).iloc[-1])
    upper, _, lower = calc_bb(close)
    upper_val = float(upper.iloc[-1])
    lower_val = float(lower.iloc[-1])

    # SMA-200
    _sma200_raw = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    sma200_val  = _sma200_raw if (_sma200_raw and not np.isnan(_sma200_raw)) else None

    # Kwaliteitsfilter: geen penny stocks
    price_ok = current >= 5.0
    # Trend filter: prijs boven SMA-200
    trend_ok = (sma200_val is None) or (current > sma200_val)

    # Volume-validatie: volume > 1.2× 10-daags gemiddelde
    vol_surge = False
    if volume is not None and len(volume) >= 11:
        avg_vol_10d = float(volume.iloc[-11:-1].mean())
        today_vol   = float(volume.iloc[-1])
        if avg_vol_10d > 0:
            vol_surge = today_vol > avg_vol_10d * 1.2

    if rsi_val < 30 and current <= lower_val and price_ok and trend_ok:
        signal = "STRONG BUY" if vol_surge else "BUY"
    elif rsi_val > 70 and current >= upper_val:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal":    signal,
        "price":     round(current, 4),
        "rsi":       round(rsi_val, 1),
        "upper_bb":  round(upper_val, 2),
        "lower_bb":  round(lower_val, 2),
        "sma200":    round(sma200_val, 2) if sma200_val else None,
        "vol_surge": vol_surge,
    }


def _get_close(data: pd.DataFrame, ticker: str, multi: bool) -> pd.Series:
    """Safely extract Close series from a yfinance download result."""
    if not multi:
        return data["Close"].dropna()
    try:
        return data[ticker]["Close"].dropna()
    except (KeyError, TypeError):
        try:
            return data["Close"][ticker].dropna()
        except Exception:
            return pd.Series(dtype=float)


def _get_volume(data: pd.DataFrame, ticker: str, multi: bool) -> pd.Series:
    """Safely extract Volume series from a yfinance download result."""
    if not multi:
        return data["Volume"].dropna()
    try:
        return data[ticker]["Volume"].dropna()
    except (KeyError, TypeError):
        try:
            return data["Volume"][ticker].dropna()
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
    # Voorkom dat twee threads tegelijk scannen (knop + achtergrond-thread)
    if not _scan_lock.acquire(blocking=False):
        return []   # al bezig — stil weggaan
    try:
        return _run_scan_inner()
    finally:
        _scan_lock.release()


def _run_scan_inner() -> list:
    """Interne scan-logica — aanroepen via run_scan() (heeft de lock)."""
    try:
        # 1 jaar data zodat SMA-200 berekend kan worden
        raw = yf.download(
            WATCHLIST, period="1y", interval="1d",
            group_by="ticker", progress=False, threads=True,
        )
    except Exception:
        return []

    multi    = len(WATCHLIST) > 1
    results  = []
    portfolio = load_portfolio()

    for ticker in WATCHLIST:
        try:
            close  = _get_close(raw, ticker, multi)
            volume = _get_volume(raw, ticker, multi)
            if len(close) < 20:
                continue
            sig = compute_signal(close, volume)

            # Earnings Protector: blokkeer BUY-signalen als earnings binnen 7 dagen
            if sig["signal"] in ("BUY", "STRONG BUY") and _earnings_within_days(ticker, 7):
                sig["signal"]           = "HOLD"
                sig["earnings_warning"] = True

            sig["ticker"]     = ticker
            sig["scanned_at"] = datetime.now().isoformat()
            results.append(sig)
        except Exception:
            continue

    # Sorteer op RSI (laagste = meest oversold)
    results.sort(key=lambda x: x.get("rsi") or 50)

    ms             = market_status()
    last_signals   = load_last_signals()
    settings       = load_settings()
    market_ctx     = fetch_market_context()
    market_bearish = market_ctx.get("sentiment") == "bearish"
    budget       = float(settings.get("budget_eur", 0))
    now          = datetime.now()

    def _recently_alerted(ticker: str, alert_type: str) -> bool:
        """True als dit exact alert-type al verstuurd is binnen ALERT_COOLDOWN_H uur."""
        prev = last_signals.get(ticker, {})
        if prev.get("last_telegram") != alert_type:
            return False
        sent_str = prev.get("sent_at")
        if not sent_str:
            return False
        try:
            hours_ago = (now - datetime.fromisoformat(sent_str)).total_seconds() / 3600
            return hours_ago < ALERT_COOLDOWN_H
        except Exception:
            return False

    def _cooldown_ok(ticker: str, signal: str) -> bool:
        """True als de laatste alert langer dan ALERT_COOLDOWN_H geleden is."""
        return not _recently_alerted(ticker, signal)

    def _is_extreme(sig: dict) -> bool:
        """
        Extreem zeldzame koopkans: RSI < 15 én prijs ≥ 15% onder onderste Bollinger Band.
        Kwaliteitsfilter: geen penny stocks.
        """
        rsi   = sig.get("rsi") or 50
        price = sig.get("price") or 0
        lb    = sig.get("lower_bb") or price
        return rsi < 15 and lb > 0 and price < lb * 0.85 and price >= 5.0

    for sig in results:
        ticker = sig["ticker"]
        signal = sig.get("signal")

        if signal in ("BUY", "STRONG BUY", "SELL"):
            append_signal(sig)

        prev         = last_signals.get(ticker, {})
        prev_signal  = prev.get("signal")
        prev_alerted = prev.get("alerted", False)

        # Bijwerken staat — bewaar last_telegram zodat cooldown-check blijft werken
        last_signals[ticker] = {
            "signal":        signal,
            "price":         sig.get("price"),
            "updated":       now.isoformat(),
            "sent_at":       prev.get("sent_at"),
            "alerted":       prev_alerted,
            "last_telegram": prev.get("last_telegram"),   # NIET overschrijven
        }

        # ── Advies verlopen: was actief + gealerteerd, nu HOLD ───────────────
        if prev_signal in ("BUY", "SELL") and prev_alerted and signal == "HOLD":
            if _cooldown_ok(ticker, "VERLOPEN"):
                telegram_send(
                    f"⚠️ <b>Advies {ticker}: Verlopen</b> (Prijs is gestabiliseerd)\n"
                    f"RSI: {sig['rsi']}  ·  Prijs: ${sig['price']:.2f}"
                )
                last_signals[ticker].update({
                    "alerted":       False,
                    "sent_at":       now.isoformat(),
                    "last_telegram": "VERLOPEN",
                })
            save_last_signals(last_signals)
            continue

        # ── Portfolio-positie: PENDING = stil; ACTIEF = VERKOOP bij RSI > 70 ───
        if ticker in portfolio:
            pos    = portfolio[ticker]
            status = pos.get("status", "ACTIEF")

            # PENDING-posities: nooit alerteren (order nog niet uitgevoerd)
            if status != "ACTIEF":
                continue

            rsi_val    = sig.get("rsi") or 0
            cur_price  = sig.get("price") or 0
            avg_price  = float(pos.get("avg_price") or 0)
            profit_pct = ((cur_price - avg_price) / avg_price * 100) if avg_price > 0 else 0

            # Verkoop-trigger: RSI > 70 EN prijs boven minimale winstdrempel per ticker
            _sell_floor = {"MRCY": 94.00}.get(ticker, 0.0)  # per-ticker minimumprijs
            should_sell = rsi_val > 70 and (cur_price > _sell_floor if _sell_floor else True)

            if should_sell and not _recently_alerted(ticker, "VERKOOP"):
                pnl_usd     = (cur_price - avg_price) * float(pos.get("shares", 0))
                _floor_txt  = f" · prijs > ${_sell_floor:.2f}" if _sell_floor else ""
                sell_reason = f"RSI overbought ({rsi_val:.0f}){_floor_txt}"
                markt = (
                    "⏸ Markt gesloten — uitvoering bij opening"
                    if not ms["is_open"]
                    else "⬤ Markt open"
                )

                # Top-3 alternatieven: prijs ≥ $5, trend omhoog (prijs > SMA-200)
                alts = [
                    s for s in results
                    if s["ticker"] not in portfolio
                    and (s.get("rsi") or 100) < 35
                    and (s.get("price") or 0) >= 5.0
                    and (
                        s.get("sma200") is None          # onvoldoende data → benefit of doubt
                        or (s.get("price") or 0) > (s.get("sma200") or 0)
                    )
                ][:3]

                msg = (
                    f"🔴 <b>VERKOPEN: {ticker}</b>\n"
                    f"Reden: {sell_reason}\n"
                    f"Prijs: <b>${cur_price:.2f}</b>  ·  RSI: {rsi_val}\n"
                    f"Positie: {pos.get('shares')} aandelen @ ${avg_price:.2f}\n"
                    f"P&amp;L: <b>${pnl_usd:+.2f} ({profit_pct:+.1f}%)</b>\n"
                    f"{markt}"
                )
                if alts:
                    msg += "\n\n\U0001f4a1 <b>Alternatieven (↑ trend, RSI &lt; 35):</b>"
                    for alt in alts:
                        msg += f"\n• <b>{alt['ticker']}</b> — RSI: {alt['rsi']} · ${alt['price']:.2f}"

                telegram_send(msg)
                last_signals[ticker].update({
                    "alerted":       True,
                    "sent_at":       now.isoformat(),
                    "last_telegram": "VERKOOP",
                })
                save_last_signals(last_signals)
            continue   # bot blijft stil over portfolio-posities — geen BUY-alerts

        # ── Niet in portfolio — BUY/KOOPKANS logica ──────────────────────────
        # Markt dicht + geen actief signaal → overslaan
        if not ms["is_open"] and signal not in ("BUY", "STRONG BUY", "SELL"):
            continue

        # ── BUY/STRONG BUY: budget = 0 → alleen bij extreme kans ─────────────
        if signal in ("BUY", "STRONG BUY") and budget <= 0:
            if not _is_extreme(sig) or _recently_alerted(ticker, "EXTREME BUY"):
                continue
            telegram_send(
                f"🚨 <b>EXTREME KOOPKANS: {ticker}</b>\n"
                f"RSI: {sig['rsi']} (extreem oversold) · Prijs: ${sig['price']:.2f}\n"
                f"Prijs ligt ver onder Bollinger Band — overweeg bij te storten.\n"
                f"Budget staat op €0 — dit is een uitzonderlijke situatie."
            )
            last_signals[ticker].update({"alerted": True, "sent_at": now.isoformat(), "last_telegram": "EXTREME BUY"})
            save_last_signals(last_signals)
            continue

        # ── Koopkans / Strong Buy buiten portfolio (budget > 0, markt open) ──
        # Bij bearish markt (SPY/ITA rood > 1%): alleen STRONG BUY adviezen
        if market_bearish and signal == "BUY":
            continue
        if (budget > 0 and ms["is_open"]
                and signal in ("BUY", "STRONG BUY")
                and (sig.get("price") or 0) >= 5.0
                and not _recently_alerted(ticker, signal)):
            if signal == "STRONG BUY":
                telegram_send(
                    f"💎 <b>STRONG BUY: {ticker}</b>\n"
                    f"RSI: {sig['rsi']} (sterk oversold) + volume-surge bevestigd\n"
                    f"Prijs: ${sig['price']:.2f}  ·  BB Laag: ${sig['lower_bb']:.2f}\n"
                    f"Volume > 1.2× 10-daags gemiddelde — institutioneel koopsignaal."
                )
            else:
                telegram_send(
                    f"\U0001f7e2 <b>KOOPKANS: {ticker}</b>\n"
                    f"RSI: {sig['rsi']} (sterk oversold)\n"
                    f"Prijs: ${sig['price']:.2f}  ·  Onderste BB: ${sig['lower_bb']:.2f}"
                )
            last_signals[ticker].update({"alerted": True, "sent_at": now.isoformat(), "last_telegram": signal})
            save_last_signals(last_signals)

    # Sla altijd de laatste bekende state op (ook niet-gealerteerde tickers bijwerken)
    save_last_signals(last_signals)
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
def fetch_price_targets(ticker: str) -> dict:
    """
    Haal live analistenkoersdoelen op via Yahoo Finance (yfinance).
    - 1 jaar  : analistenconsensus (gemiddeld koersdoel)
    - 3 maanden: interpolatie tussen huidig en 1-jaarsdoel (25% van de weg)
    - 1 maand : technische trendprojectie op basis van 3-maands momentum
    Gecached 1 uur.
    """
    result = {}
    try:
        info    = yf.Ticker(ticker).fast_info
        current = float(info.last_price) if hasattr(info, "last_price") else None
    except Exception:
        current = None

    try:
        full_info       = yf.Ticker(ticker).info
        target_1y       = full_info.get("targetMeanPrice")
        target_1y_high  = full_info.get("targetHighPrice")
        target_1y_low   = full_info.get("targetLowPrice")
        n_analysts      = full_info.get("numberOfAnalystOpinions") or 0
        rec_key         = (full_info.get("recommendationKey") or "").lower()

        rec_nl = {
            "strong buy":  "Sterk kopen",
            "buy":         "Kopen",
            "hold":        "Houden",
            "underperform":"Onderpresteren",
            "sell":        "Verkopen",
        }.get(rec_key, rec_key.capitalize() if rec_key else "—")

        if not current:
            current = full_info.get("currentPrice") or full_info.get("regularMarketPrice")

        beta = full_info.get("beta")
        result["n_analysts"]    = int(n_analysts)
        result["recommendation"] = rec_nl
        result["target_1y"]     = round(target_1y, 2)      if target_1y      else None
        result["target_1y_high"]= round(target_1y_high, 2) if target_1y_high else None
        result["target_1y_low"] = round(target_1y_low, 2)  if target_1y_low  else None
        result["beta"]          = round(float(beta), 2)    if beta is not None else None

        if current and target_1y:
            move = target_1y - current
            result["target_3m"]     = round(current + move * 0.25, 2)
            result["target_3m_pct"] = round((move * 0.25 / current) * 100, 1)
            result["target_1y_pct"] = round((move / current) * 100, 1)

    except Exception:
        pass

    # 1-maand target: technisch momentum (3-maands data)
    try:
        hist = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if not hist.empty:
            close   = hist["Close"].squeeze()
            fc      = forecast_48h(close)
            # 48u verwachting × ~10 (20 handelsdagen / 2) = ruwe 1-maandsprognose
            pct_1m  = (fc.get("verwachting_pct") or 0) * 10
            cur     = float(close.iloc[-1])
            result["target_1m"]     = round(cur * (1 + pct_1m / 100), 2)
            result["target_1m_pct"] = round(pct_1m, 1)
            if not current:
                current = cur
    except Exception:
        pass

    result["current"] = current
    return result


@st.cache_data(ttl=300)
def fetch_market_context() -> dict:
    """
    Haal marktcontext op: S&P 500 (SPY) en Defense Sector (ITA).
    Geeft dagelijkse % verandering terug.  Gecached 5 min.
    """
    result = {}
    try:
        raw = yf.download(
            ["SPY", "ITA"], period="5d", interval="1d",
            group_by="ticker", progress=False, threads=True,
        )
        multi = True
        for sym in ["SPY", "ITA"]:
            try:
                close = _get_close(raw, sym, multi).dropna()
                if len(close) >= 2:
                    prev  = float(close.iloc[-2])
                    last  = float(close.iloc[-1])
                    chg   = (last - prev) / prev * 100
                    result[sym] = {
                        "price":      round(last, 2),
                        "change_pct": round(chg, 2),
                        "is_up":      chg >= 0,
                    }
            except Exception:
                result[sym] = None
    except Exception:
        pass
    # Bepaal overall marktsentiment
    spy = result.get("SPY")
    ita = result.get("ITA")
    if spy and ita:
        avg_chg = (spy["change_pct"] + ita["change_pct"]) / 2
        if avg_chg <= -1.0:
            sentiment = "bearish"
        elif avg_chg >= 0.5:
            sentiment = "bullish"
        else:
            sentiment = "neutraal"
    elif spy:
        sentiment = "bearish" if spy["change_pct"] <= -1.0 else ("bullish" if spy["change_pct"] >= 0.5 else "neutraal")
    else:
        sentiment = "onbekend"
    result["sentiment"] = sentiment
    return result


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

# ── PWA / iPhone "Voeg toe aan beginscherm" meta-tags ─────────────────────────
st.markdown("""
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Trading HQ">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#0f1117">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<link rel="manifest" href="/app/static/manifest.json">
<link rel="apple-touch-icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%230f1117'/><text y='.9em' font-size='80'>📈</text></svg>">
""", unsafe_allow_html=True)

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

/* Expander: donkere achtergrond + leesbare tekst */
details { background:#12161f !important; border:1px solid #1e2436 !important; border-radius:8px !important; }
details summary { background:#12161f !important; color:#f1f5f9 !important; font-weight:600 !important; }
details summary:hover { background:#1e2436 !important; }
details summary p { color:#f1f5f9 !important; }
div[data-testid="stExpander"] { background:#12161f !important; border:1px solid #1e2436 !important; border-radius:8px !important; }
div[data-testid="stExpander"] summary { color:#f1f5f9 !important; }
div[data-testid="stExpander"] > div { background:#12161f !important; }
.badge-buy  { background:#14532d; color:#4ade80; padding:3px 10px;
    border-radius:999px; font-size:.75rem; font-weight:700; }
.badge-sell { background:#450a0a; color:#f87171; padding:3px 10px;
    border-radius:999px; font-size:.75rem; font-weight:700; }
.badge-hold { background:#1e2d3d; color:#60a5fa; padding:3px 10px;
    border-radius:999px; font-size:.75rem; font-weight:700; }
.badge-strong { background:#052e16; color:#4ade80; padding:3px 10px;
    border-radius:999px; font-size:.75rem; font-weight:700;
    border:1px solid #4ade80; letter-spacing:.04em; }
.risk-bar { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:2px; }
.market-bull { color:#4ade80; font-weight:700; }
.market-bear { color:#f87171; font-weight:700; }
.market-neutral { color:#94a3b8; font-weight:700; }
/* Form submit buttons — dark theme */
[data-testid="stFormSubmitButton"] button {
    background:#1e2d3d !important; color:#e2e8f0 !important;
    border:1px solid #334155 !important; border-radius:6px !important;
    font-weight:600 !important; }
[data-testid="stFormSubmitButton"] button:hover {
    background:#263548 !important; border-color:#60a5fa !important;
    color:#f1f5f9 !important; }
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
    # ── Al ingelogd in deze sessie ────────────────────────────────────────────
    if st.session_state.get("authenticated"):
        token = st.session_state.get("_session_token")
        if token:
            _touch_session(token)   # activiteit bijhouden → timeout verlengen
        return True

    # ── Sessie-token uit URL uitlezen (persistent login na refresh) ───────────
    url_token = st.query_params.get("s")
    if url_token and _is_session_valid(url_token):
        st.session_state["authenticated"]    = True
        st.session_state["_session_token"]   = url_token
        _touch_session(url_token)
        return True

    # Token bestaat maar is verlopen → URL opschonen en opnieuw inloggen tonen
    if url_token and not _is_session_valid(url_token):
        _delete_session(url_token)
        st.query_params.clear()

    # ── Login-scherm ──────────────────────────────────────────────────────────
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
                    new_token = _create_session()
                    st.session_state["authenticated"]  = True
                    st.session_state["_session_token"] = new_token
                    st.query_params["s"] = new_token   # token in URL → persistent
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

    # ── Budget instelling ─────────────────────────────────────────────────────
    _settings = load_settings()
    _budget   = float(_settings.get("budget_eur", 0))

    st.markdown("**Beschikbaar budget**")
    _new_budget = st.number_input(
        "€ investeerbaar", min_value=0.0, value=_budget,
        step=50.0, format="%.2f", label_visibility="collapsed",
        key="sidebar_budget",
    )
    if _new_budget != _budget:
        save_settings({**_settings, "budget_eur": _new_budget})
        st.rerun()

    if _budget <= 0:
        st.markdown('<span style="color:#f87171;font-size:.75rem">⛔ Budget €0 — BUY alerts uit</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span style="color:#4ade80;font-size:.75rem">✓ Budget €{_budget:,.0f} actief</span>', unsafe_allow_html=True)

    st.divider()

    if st.button("🔄 Nu Scannen", use_container_width=True):
        with st.spinner("Scannen..."):
            fetch_scan.clear()
            fetch_scan()   # één aanroep, result wordt gecached — geen tweede run via rerun()
        st.success("Scan voltooid.")
        st.rerun()

    st.caption("Automatische scan elke 30 min.")
    st.divider()

    # ── Market Context widget ─────────────────────────────────────────────────
    st.markdown("**Market Context**")
    _mc = fetch_market_context()
    _sentiment = _mc.get("sentiment", "onbekend")
    _sent_color = {"bullish": "#4ade80", "bearish": "#f87171",
                   "neutraal": "#94a3b8", "onbekend": "#475569"}.get(_sentiment, "#475569")
    _sent_icon  = {"bullish": "▲", "bearish": "▼", "neutraal": "►"}.get(_sentiment, "?")
    st.markdown(
        f'<div style="background:#0c0f16;border:1px solid #1e2436;border-radius:6px;padding:8px 10px;font-size:.78rem">'
        f'<div style="color:{_sent_color};font-weight:700;margin-bottom:4px">{_sent_icon} {_sentiment.upper()}</div>',
        unsafe_allow_html=True,
    )
    for _sym in ["SPY", "ITA"]:
        _d = _mc.get(_sym)
        if _d:
            _c = "#4ade80" if _d["is_up"] else "#f87171"
            _arrow = "▲" if _d["is_up"] else "▼"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;font-size:.72rem">'
                f'<span style="color:#64748b">{_sym}</span>'
                f'<span style="color:{_c}">${_d["price"]:,.2f} {_arrow}{abs(_d["change_pct"]):.2f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<div style="font-size:.72rem;color:#475569">{_sym}: —</div>', unsafe_allow_html=True)
    if _sentiment == "bearish":
        st.markdown('<div style="margin-top:4px;font-size:.68rem;color:#f87171">⚠️ Rode markt — bot extra voorzichtig</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()

    if st.button("🚪 Uitloggen", use_container_width=True):
        _tok = st.session_state.get("_session_token")
        if _tok:
            _delete_session(_tok)
        st.session_state.authenticated = False
        st.session_state.pop("_session_token", None)
        st.query_params.clear()
        st.rerun()


portfolio = load_portfolio()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Portfolio & P&L
# ══════════════════════════════════════════════════════════════════════════════

if page == "Portfolio & P&L":
    _hdr_l, _hdr_r = st.columns([3, 1])
    _hdr_l.markdown("## 📊 Portfolio & P&L")
    with _hdr_r:
        _eur_on = st.toggle("🇪🇺 Euro's", key="eur_toggle", value=st.session_state.get("eur_toggle", False))

    if not portfolio:
        st.info("Geen posities. Voeg ze toe via 'Transacties'.")
        st.stop()

    # ── Valuta-hulpfuncties ───────────────────────────────────────────────────
    _eur_rate = fetch_eur_usd() if _eur_on else 1.0
    _ccy      = "€" if _eur_on else "$"

    def _c(usd: float) -> float:
        """Converteer USD naar geselecteerde valuta."""
        return usd / _eur_rate

    def _fmt(usd: float, sign: bool = False) -> str:
        v = _c(usd)
        return (f"{_ccy}{v:+,.2f}" if sign else f"{_ccy}{v:,.2f}")

    if _eur_on:
        st.caption(f"Wisselkoers: 1 € = ${_eur_rate:.4f}  ·  Alle bedragen omgerekend naar euro's")

    tickers = [k for k in portfolio.keys() if not k.startswith("_")]
    prices  = fetch_quotes(tuple(sorted(tickers)))

    # ── Samenvatting ─────────────────────────────────────────────────────────
    # PENDING-posities tellen mee in waarde, maar NIET in P&L
    _actief   = [t for t in tickers if portfolio[t].get("status", "ACTIEF") == "ACTIEF"]
    _pending  = [t for t in tickers if portfolio[t].get("status", "ACTIEF") != "ACTIEF"]

    total_cost  = sum(portfolio[t]["shares"] * portfolio[t]["avg_price"] for t in tickers)
    total_value = sum(
        portfolio[t]["shares"] * (prices.get(t) or portfolio[t]["avg_price"])
        for t in tickers
    )
    # P&L alleen over ACTIEVE posities
    pnl_cost  = sum(portfolio[t]["shares"] * portfolio[t]["avg_price"] for t in _actief)
    pnl_value = sum(
        portfolio[t]["shares"] * (prices.get(t) or portfolio[t]["avg_price"])
        for t in _actief
    )
    total_pnl     = pnl_value - pnl_cost
    total_pnl_pct = (total_pnl / pnl_cost * 100) if pnl_cost else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Totale Waarde",    _fmt(total_value))
    c2.metric("Kosten Basis",     _fmt(total_cost))
    c3.metric("Unrealized P&L",   _fmt(total_pnl, sign=True), f"{total_pnl_pct:+.1f}%")
    c4.metric("Posities",         f"{len(_actief)} actief  ·  {len(_pending)} pending" if _pending else str(len(tickers)))

    st.divider()

    # ── Snelle transactie toevoegen ───────────────────────────────────────────
    with st.expander("➕ Snelle Transactie Toevoegen", expanded=False):
        with st.form("quick_add_portfolio", clear_on_submit=True):
            qc1, qc2, qc3, qc4, qc5 = st.columns(5)
            with qc1:
                qt = st.text_input("Ticker (bijv. AAPL)").strip().upper()
            with qc2:
                qa = st.number_input("Aantal aandelen", min_value=0.0, step=0.0001, format="%.4f", value=0.0)
            with qc3:
                qp = st.number_input("Prijs ($)", min_value=0.0, step=0.01, format="%.2f", value=0.0)
            with qc4:
                qd = st.date_input("Aankoopdatum", value=date.today())
            with qc5:
                qs = st.selectbox("Status", ["PENDING", "ACTIEF"])
            if st.form_submit_button("+ Toevoegen aan Portfolio", use_container_width=True):
                if qt and qa > 0 and qp > 0:
                    if qt in portfolio:
                        old_pos   = portfolio[qt]
                        tot       = old_pos["shares"] + qa
                        new_avg   = (old_pos["shares"] * old_pos["avg_price"] + qa * qp) / tot
                        portfolio[qt]["shares"]    = round(tot, 8)
                        portfolio[qt]["avg_price"] = round(new_avg, 4)
                    else:
                        portfolio[qt] = {
                            "shares":    round(qa, 8),
                            "avg_price": round(qp, 4),
                            "buy_date":  qd.isoformat(),
                            "added":     datetime.now().isoformat(),
                            "status":    qs,
                        }
                    save_portfolio(portfolio)
                    fetch_quotes.clear()
                    st.success(f"✓ {qt}: {qa:.4f} aandelen @ ${qp:.2f} [{qs}] opgeslagen.")
                    st.rerun()
                else:
                    st.error("Vul ticker, aantal (> 0) en prijs (> 0) in.")

    st.divider()

    # ── Per-positie detail ────────────────────────────────────────────────────
    for ticker in tickers:
        pos           = portfolio[ticker]
        current_price = prices.get(ticker)
        hist          = fetch_history(ticker, "1mo")

        _pos_status  = pos.get("status", "ACTIEF")
        _status_tag  = "  ⏳ PENDING" if _pos_status == "PENDING" else ""
        with st.expander(f"**{ticker}** — {pos['shares']} aandelen{_status_tag}", expanded=True):
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
                # Stats card heeft hist NIET nodig — alleen de chart doet dat.
                # Gebruik avg_price als fallback wanneer live koers nog niet beschikbaar is.
                _display_price = current_price or pos["avg_price"]
                if _display_price:
                    _status        = pos.get("status", "ACTIEF")
                    _is_pending    = _status == "PENDING"
                    huidige_waarde = _display_price * pos["shares"]
                    # Prefer stored scan result (uses 1y data) over recomputing from 1mo hist
                    _stored_sig = load_last_signals().get(ticker, {})
                    if _stored_sig and _stored_sig.get("signal") not in (None, "DATA_ERROR", "?"):
                        sig_data = _stored_sig
                    elif not hist.empty:
                        sig_data = compute_signal(hist["Close"].squeeze())
                    else:
                        sig_data = {"signal": "DATA_ERROR", "rsi": None,
                                    "upper_bb": None, "lower_bb": None}
                    current_price  = _display_price  # gebruik fallback voor verdere berekeningen
                    signal         = sig_data.get("signal", "?")

                    # Earnings Protector — waarschuwing als cijfers binnen 7 dagen
                    _earnings_soon = _earnings_within_days(ticker, 7)
                    _earnings_date = _get_earnings_date(ticker)
                    if _earnings_soon and _earnings_date:
                        st.warning(
                            f"⚠️ **Earnings binnen 7 dagen** — {ticker} publiceert "
                            f"cijfers op **{_earnings_date}**. BUY-alerts worden geblokkeerd.",
                            icon=None,
                        )

                    # P&L: toon $0.00 voor PENDING (order nog niet bevestigd)
                    if _is_pending:
                        pnl_usd = 0.0
                        pnl_pct = 0.0
                    else:
                        pnl_usd = (current_price - pos["avg_price"]) * pos["shares"]
                        pnl_pct = ((current_price - pos["avg_price"]) / pos["avg_price"]) * 100

                    # Dagen in bezit
                    _buy_str = pos.get("buy_date") or pos.get("added", "")
                    try:
                        _bought_date = datetime.fromisoformat(_buy_str[:10]).date()
                        _dagen = (date.today() - _bought_date).days
                    except Exception:
                        _dagen = 0

                    badge_map = {"BUY": "buy", "STRONG BUY": "strong", "SELL": "sell", "HOLD": "hold"}
                    # PENDING-posities tonen altijd "IN AFWACHTING" badge
                    if _is_pending:
                        badge_cls = "hold"
                        label     = "IN AFWACHTING"
                    elif signal in ("DATA_ERROR", "?", None):
                        badge_cls = "hold"
                        label     = "LADEN..."
                    else:
                        label_map = {"BUY": "BIJKOPEN", "STRONG BUY": "💎 STRONG BUY",
                                     "SELL": "VERKOPEN", "HOLD": "VASTHOUDEN"}
                        badge_cls = badge_map.get(signal, "hold")
                        label     = label_map.get(signal, signal)
                    pnl_cls = "pnl-pos" if pnl_usd >= 0 else "pnl-neg"

                    _pnl_display = (
                        '<div style="font-size:.72rem;color:#fb923c;font-style:italic">'
                        'Wacht op marktopening —<br>berekening start bij eerste koersbeweging.</div>'
                        if _is_pending
                        else f'<div class="{pnl_cls}">{_fmt(pnl_usd, sign=True)} ({pnl_pct:+.1f}%)</div>'
                    )

                    # Pre-fetch price targets for beta / upside display in card
                    _pt_card   = fetch_price_targets(ticker)
                    _beta      = (_pt_card or {}).get("beta")
                    _upside    = (_pt_card or {}).get("target_1y_pct")
                    # Risk score: beta → number of filled bars (1–5)
                    if _beta is not None:
                        _bars     = min(5, max(1, round(abs(_beta) * 2)))
                        _bar_html = "".join(
                            f'<span class="risk-bar" style="background:{"#f87171" if i < _bars else "#1e2436"}"></span>'
                            for i in range(5)
                        )
                        _beta_lbl = "Laag" if abs(_beta) < 0.8 else ("Hoog" if abs(_beta) > 1.5 else "Gemiddeld")
                        _risk_row = (
                            f'<div style="font-size:.7rem;color:#64748b;margin-top:8px">Risico (Beta {_beta:.1f})</div>'
                            f'<div style="margin-top:2px">{_bar_html} '
                            f'<span style="font-size:.7rem;color:#94a3b8">{_beta_lbl}</span></div>'
                        )
                    else:
                        _risk_row = ""
                    # Analyst upside
                    if _upside is not None:
                        _up_color  = "#4ade80" if _upside >= 0 else "#f87171"
                        _up_arrow  = "▲" if _upside >= 0 else "▼"
                        _upside_row = (
                            f'<div style="font-size:.7rem;color:#64748b;margin-top:8px">Analist koersdoel (1j)</div>'
                            f'<div style="font-size:.9rem;font-weight:700;color:{_up_color}">'
                            f'{_up_arrow} {abs(_upside):.1f}% upside</div>'
                        )
                    else:
                        _upside_row = ""

                    _pending_tag = "&nbsp;<span style='font-size:.65rem;color:#fb923c;font-weight:600'>⏳ PENDING</span>" if _is_pending else ""
                    st.markdown(f"""
                    <div class="card">
                        <div style="margin-bottom:12px"><span class="badge-{badge_cls}">{label}</span>{_pending_tag}</div>
                        <div style="font-size:.7rem;color:#64748b">Huidige prijs</div>
                        <div style="font-size:1.3rem;font-weight:800;color:#f1f5f9">{_fmt(current_price)}</div>
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">Huidige waarde</div>
                        <div style="font-size:1rem;font-weight:700;color:#60a5fa">{_fmt(huidige_waarde)}</div>
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">Gemiddelde inkoop</div>
                        <div style="color:#94a3b8">{_fmt(pos['avg_price'])}</div>
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">Aandelen</div>
                        <div style="color:#94a3b8">{pos['shares']}</div>
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">Totale winst/verlies</div>
                        {_pnl_display}
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">Dagen in bezit</div>
                        <div style="color:#94a3b8">{_dagen} dagen</div>
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">RSI (14)</div>
                        <div style="font-size:1rem;color:#fb923c;font-weight:700">{sig_data.get('rsi') or '—'}</div>
                        <div style="font-size:.7rem;color:#64748b;margin-top:8px">BB Hoog / Laag</div>
                        <div style="font-size:.8rem;color:#60a5fa">{_fmt(sig_data['upper_bb']) if sig_data.get('upper_bb') else '—'} / {_fmt(sig_data['lower_bb']) if sig_data.get('lower_bb') else '—'}</div>
                        {_risk_row}
                        {_upside_row}
                    </div>
                    """, unsafe_allow_html=True)

                    # Telegram sync badge
                    _ls   = load_last_signals().get(ticker, {})
                    _tg   = _ls.get("last_telegram")
                    _sent = (_ls.get("sent_at") or "")[:16].replace("T", " ")
                    if _tg:
                        _tg_color = {
                            "KOOP":       "#4ade80",
                            "VERKOOP":    "#f87171",
                            "VERLOPEN":   "#94a3b8",
                            "EXTREME BUY":"#facc15",
                            "KOOPKANS":   "#60a5fa",
                        }.get(_tg, "#64748b")
                        st.markdown(
                            f'<div style="background:#0c1220;border:1px solid #1e2436;'
                            f'border-radius:6px;padding:6px 10px;margin-top:6px;font-size:.72rem">'
                            f'<span style="color:#475569">Telegram: </span>'
                            f'<span style="color:{_tg_color};font-weight:700">{_tg}</span>'
                            f'{"  <span style=\\'color:#334155\\'>" + _sent + "</span>" if _sent else ""}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    # Koersdoelen sectie
                    pt = _pt_card  # already fetched above
                    if pt:
                        def _tgt_row(label, price_usd, pct, color):
                            if price_usd is None:
                                return f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #1e2436"><span style="font-size:.72rem;color:#64748b">{label}</span><span style="font-size:.72rem;color:#475569">—</span></div>'
                            arrow = "▲" if (pct or 0) >= 0 else "▼"
                            return (
                                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid #1e2436">'
                                f'<span style="font-size:.72rem;color:#64748b">{label}</span>'
                                f'<span style="font-size:.78rem;color:{color};font-weight:600">'
                                f'{_fmt(price_usd)} <span style="font-size:.68rem">{arrow}{abs(pct or 0):.1f}%</span></span>'
                                f'</div>'
                            )

                        t1m  = pt.get("target_1m");  p1m  = pt.get("target_1m_pct", 0)
                        t3m  = pt.get("target_3m");  p3m  = pt.get("target_3m_pct", 0)
                        t1y  = pt.get("target_1y");  p1y  = pt.get("target_1y_pct", 0)
                        t1yh = pt.get("target_1y_high")
                        t1yl = pt.get("target_1y_low")
                        rec  = pt.get("recommendation", "—")
                        na   = pt.get("n_analysts", 0)

                        c1m  = "#4ade80" if (p1m or 0) >= 0 else "#f87171"
                        c3m  = "#4ade80" if (p3m or 0) >= 0 else "#f87171"
                        c1y  = "#4ade80" if (p1y or 0) >= 0 else "#f87171"

                        rec_color = {"Kopen": "#4ade80", "Sterk kopen": "#4ade80",
                                     "Houden": "#fb923c", "Verkopen": "#f87171",
                                     "Onderpresteren": "#f87171"}.get(rec, "#94a3b8")

                        analyst_range = (
                            f"{_fmt(t1yl)} – {_fmt(t1yh)}" if t1yl and t1yh else "—"
                        )

                        st.markdown(
                            f'<div class="card" style="margin-top:8px">'
                            f'<div style="font-size:.65rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Koersdoelen</div>'
                            + _tgt_row("1 maand (technisch)", t1m, p1m, c1m)
                            + _tgt_row("3 maanden (analisten)", t3m, p3m, c3m)
                            + _tgt_row("1 jaar (consensus)", t1y, p1y, c1y)
                            + f'<div style="padding:5px 0;font-size:.7rem;color:#64748b">Bandbreedte 1j: <span style="color:#94a3b8">{analyst_range}</span></div>'
                            f'<div style="padding:3px 0;font-size:.7rem;color:#64748b">Aanbeveling: <span style="color:{rec_color};font-weight:700">{rec}</span>'
                            f'{"  <span style=\\'color:#475569\\'>(" + str(na) + " analisten)</span>" if na else ""}'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("Koersdata nog niet beschikbaar — wordt geladen bij volgende refresh.")

            # ── Verkopen sectie (onder chart + stats) ────────────────────────
            st.markdown(
                '<div style="border-top:1px solid #1e2436;margin-top:10px;padding-top:10px">'
                '<span style="font-size:.75rem;color:#64748b;text-transform:uppercase;'
                'letter-spacing:.06em">Positie verkopen</span></div>',
                unsafe_allow_html=True,
            )
            with st.form(f"sell_pos_{ticker}"):
                sv1, sv2 = st.columns(2)
                with sv1:
                    sell_qty = st.number_input(
                        "Aantal te verkopen",
                        min_value=0.0,
                        max_value=float(pos["shares"]),
                        value=float(pos["shares"]),
                        step=0.0001,
                        format="%.4f",
                    )
                with sv2:
                    _default_sell_price = float(current_price) if current_price else float(pos["avg_price"])
                    sell_pr = st.number_input(
                        "Verkoopprijs ($)",
                        min_value=0.0,
                        value=_default_sell_price,
                        step=0.01,
                        format="%.2f",
                    )
                sell_b1, sell_b2 = st.columns(2)
                with sell_b1:
                    if st.form_submit_button("Deels Verkopen", use_container_width=True):
                        if sell_qty > 0:
                            remaining = round(float(pos["shares"]) - sell_qty, 8)
                            if remaining > 0.00000001:
                                portfolio[ticker]["shares"] = remaining
                                save_portfolio(portfolio)
                                fetch_quotes.clear()
                                st.success(f"✓ {sell_qty:.4f} {ticker} verkocht @ ${sell_pr:.2f}. {remaining:.4f} resterend.")
                            else:
                                del portfolio[ticker]
                                save_portfolio(portfolio)
                                fetch_quotes.clear()
                                st.success(f"✓ {ticker} volledig verkocht @ ${sell_pr:.2f}.")
                            st.rerun()
                with sell_b2:
                    if st.form_submit_button("Volledig Verkopen", use_container_width=True):
                        del portfolio[ticker]
                        save_portfolio(portfolio)
                        fetch_quotes.clear()
                        st.success(f"✓ {ticker} volledig verkocht @ ${sell_pr:.2f}.")
                        st.rerun()


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
        emoji_map = {"BUY": "🟢", "STRONG BUY": "💎", "SELL": "🔴", "HOLD": "🔵"}

        # Haal analyst-data op voor BUY/STRONG BUY tickers (gecached, max ~10 calls)
        _buy_tickers = [s["ticker"] for s in scan_results
                        if s.get("signal") in ("BUY", "STRONG BUY") and s.get("ticker")]
        _pt_cache: dict = {}
        for _bt in _buy_tickers[:10]:
            _pt_cache[_bt] = fetch_price_targets(_bt)

        rows = []
        for s in scan_results:
            signal   = s.get("signal", "?")
            sma200   = s.get("sma200")
            price    = s.get("price")
            trend_ok = (sma200 is None) or (price and price > sma200)
            vol_tag  = " ⚡" if s.get("vol_surge") else ""
            _pt      = _pt_cache.get(s.get("ticker"), {})
            _rec     = (_pt.get("recommendation") or "—") if _pt else "—"
            _upside  = _pt.get("target_1y_pct") if _pt else None
            _upside_str = (f"{'▲' if _upside >= 0 else '▼'}{abs(_upside):.0f}%"
                           if _upside is not None else "—")
            rows.append({
                " ":            emoji_map.get(signal, "⚪"),
                "Ticker":       s.get("ticker", "?"),
                "Prijs":        f"${price:.2f}"          if price             else "—",
                "RSI":          f"{s['rsi']:.1f}"        if s.get("rsi")      else "—",
                "SMA-200":      f"${sma200:.2f}"         if sma200            else "—",
                "Trend":        "↑" if trend_ok else "↓",
                "BB Hoog":      f"${s['upper_bb']:.2f}"  if s.get("upper_bb") else "—",
                "BB Laag":      f"${s['lower_bb']:.2f}"  if s.get("lower_bb") else "—",
                "Signaal":      signal + vol_tag,
                "Aanbeveling":  _rec,
                "Upside 1j":    _upside_str,
                "Portfolio":    "✓" if s.get("ticker") in portfolio else "",
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        _n_strong = sum(1 for r in rows if "STRONG BUY" in r["Signaal"])
        _n_buy    = sum(1 for r in rows if r["Signaal"].startswith("BUY"))
        st.caption(f"Gescand: {len(scan_results)} tickers  ·  "
                   f"{_n_strong} STRONG BUY  ·  {_n_buy} KOOP  ·  "
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

    # ── Marktstatus banner ────────────────────────────────────────────────────
    ms = market_status()
    dag_map = ["ma","di","wo","do","vr","za","zo"]
    now_str = ms["now_nl"].strftime("%H:%M")
    st.markdown(
        f'<div style="background:#12161f;border:1.5px solid {ms["color"]}33;'
        f'border-radius:8px;padding:10px 16px;margin-bottom:12px;'
        f'display:flex;align-items:center;gap:16px;flex-wrap:wrap">'
        f'<span style="color:{ms["color"]};font-weight:700;font-size:.9rem">⬤ {ms["status"].upper()}</span>'
        f'<span style="color:#e2e8f0;font-size:.85rem">{ms["msg"]}</span>'
        f'<span style="color:#475569;font-size:.75rem">NL-tijd: {now_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
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

                    # 48u forecast berekenen
                    hist_sig  = fetch_history(ticker, "1mo")
                    fc        = {}
                    if not hist_sig.empty:
                        fc = forecast_48h(hist_sig["Close"].squeeze())

                    richting      = fc.get("richting", "onbekend")
                    verwacht_pct  = fc.get("verwachting_pct", 0.0)
                    t_laag        = fc.get("target_laag")
                    t_hoog        = fc.get("target_hoog")
                    onderbouwing  = fc.get("onderbouwing", "")

                    richting_icon  = {"omhoog": "▲", "omlaag": "▼", "neutraal": "►"}.get(richting, "?")
                    richting_color = {"omhoog": "#4ade80", "omlaag": "#f87171", "neutraal": "#60a5fa"}.get(richting, "#94a3b8")
                    bereik_str     = f"${t_laag:.2f} – ${t_hoog:.2f}" if t_laag and t_hoog else "—"

                    # Advies-box
                    st.markdown(
                        f'<div class="card" style="margin-bottom:4px">'
                        f'<div style="margin-bottom:6px">'
                        f'Met <b>€{budget_eur:,.2f}</b> kun je maximaal '
                        f'<b>{max_shares:.4f} aandelen {ticker}</b> kopen '
                        f'(≈ ${cost_usd:,.2f} / €{cost_eur:,.2f}) '
                        f'· BB Laag: ${sig["lower_bb"]:.2f} · BB Hoog: ${sig["upper_bb"]:.2f}'
                        f'</div>'
                        f'<div style="margin-top:6px;font-size:.78rem;'
                        f'color:{"#4ade80" if ms["is_open"] else "#fb923c"};font-weight:600">'
                        f'{"⬤ Markt open — order wordt direct uitgevoerd" if ms["is_open"] else "⏸ Markt gesloten — order wordt uitgevoerd op " + ms["next_open_nl"].strftime("%a %d %b om %H:%M") + " NL-tijd"}'
                        f'</div>'
                        f'<div style="border-top:1px solid #1e2436;padding-top:8px;margin-top:4px">'
                        f'<span style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em">48u verwachting</span>&nbsp;&nbsp;'
                        f'<span style="color:{richting_color};font-weight:700">{richting_icon} {richting.upper()}</span>'
                        f'&nbsp;<span style="color:{richting_color}">{verwacht_pct:+.1f}%</span>'
                        f'&nbsp;&nbsp;<span style="color:#64748b;font-size:.8rem">Bereik: {bereik_str}</span>'
                        f'<br><span style="font-size:.72rem;color:#475569">{onderbouwing}</span>'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Opvolg-sectie
                    if outcome == "pending":
                        st.markdown(
                            '<div style="background:#0c1a0c;border:1px solid #14532d;'
                            'border-radius:8px;padding:14px 16px;margin:6px 0">',
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Heb je dit advies gevolgd?**")

                        with st.form(f"form_actual_{advice_id}"):
                            fc1, fc2 = st.columns(2)
                            with fc1:
                                actual_price = st.number_input(
                                    f"Aankooprijs {ticker} ($)",
                                    min_value=0.0001,
                                    value=round(price_usd, 4),
                                    step=0.01,
                                    format="%.4f",
                                )
                            with fc2:
                                actual_shares = st.number_input(
                                    "Aantal aandelen gekocht",
                                    min_value=0.00000001,
                                    value=round(max_shares, 8),
                                    step=0.00000001,
                                    format="%.8f",
                                )

                            sb1, sb2 = st.columns([2, 1])
                            with sb1:
                                if st.form_submit_button(
                                    f"✓ Ja, ik heb {ticker} gekocht — opslaan",
                                    use_container_width=True,
                                ):
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
                                        old          = portfolio[ticker]
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
                                    st.success(
                                        f"✓ {actual_shares:.8f} {ticker} @ ${actual_price:.4f} "
                                        f"toegevoegd aan portfolio."
                                    )
                                    st.rerun()

                            with sb2:
                                if st.form_submit_button("✗ Niet gevolgd", use_container_width=True):
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
                                        "outcome":           "niet_gevolgt",
                                        "actual_price_usd":  None,
                                        "actual_shares":     None,
                                    })
                                    st.rerun()

                        st.markdown('</div>', unsafe_allow_html=True)

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
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            new_ticker = st.text_input("Ticker (bijv. NVDA)").strip().upper()
        with c2:
            new_shares = st.number_input("Aantal aandelen", min_value=0.0,
                                         value=0.0, step=0.00000001, format="%.8f")
        with c3:
            new_price  = st.number_input("Inkoopprijs ($)", min_value=0.0,
                                         value=0.0, step=0.01, format="%.2f")
        with c4:
            new_date   = st.date_input("Aankoopdatum", value=date.today())
        with c5:
            new_status = st.selectbox("Status", ["PENDING", "ACTIEF"])
        if st.form_submit_button("Opslaan", use_container_width=True):
            if new_ticker and new_shares > 0 and new_price > 0:
                if new_ticker in portfolio:
                    old_p     = portfolio[new_ticker]
                    tot       = old_p["shares"] + new_shares
                    new_avg   = (old_p["shares"] * old_p["avg_price"] + new_shares * new_price) / tot
                    portfolio[new_ticker]["shares"]    = round(tot, 8)
                    portfolio[new_ticker]["avg_price"] = round(new_avg, 4)
                else:
                    portfolio[new_ticker] = {
                        "shares":    new_shares,
                        "avg_price": new_price,
                        "buy_date":  new_date.isoformat(),
                        "added":     datetime.now().isoformat(),
                        "status":    new_status,
                    }
                save_portfolio(portfolio)
                fetch_quotes.clear()
                st.success(f"✓ {new_ticker}: {new_shares} aandelen @ ${new_price:.2f} [{new_status}] opgeslagen.")
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

    # ── Portfolio herstellen ──────────────────────────────────────────────────
    st.markdown("### Portfolio Herstellen")
    st.caption("Overschrijf de huidige posities met de correcte waarden uit de transactie-screenshots.")
    if st.button("🔄 Herstel naar correcte posities", use_container_width=True):
        budget = portfolio.get("_budget_eur", 0)
        _now = datetime.now().isoformat()
        correct = {
            "MRCY": {"shares": 4.38356102,   "avg_price": 89.43, "buy_date": "2024-09-01", "added": _now, "status": "ACTIEF"},
            "FUBO": {"shares": 198.27586206,  "avg_price": 1.16,  "buy_date": "2026-03-09", "added": _now, "status": "PENDING"},
            "EVGO": {"shares": 89.93667860,   "avg_price": 2.169, "buy_date": "2026-03-09", "added": _now, "status": "PENDING"},
            "ARRY": {"shares": 17.71771771,   "avg_price": 6.66,  "buy_date": "2026-03-09", "added": _now, "status": "PENDING"},
            "_budget_eur": budget,
        }
        save_portfolio(correct)
        fetch_quotes.clear()
        st.success("✓ Portfolio hersteld naar correcte waarden.")
        st.rerun()

    st.divider()

    # ── Huidige posities ─────────────────────────────────────────────────────
    st.markdown("### Huidige Posities")
    if not portfolio:
        st.info("Nog geen posities.")
    else:
        for ticker in [k for k in portfolio.keys() if not k.startswith("_")]:
            pos = portfolio[ticker]

            _edit_status = pos.get("status", "ACTIEF")
            _tag = "  ⏳ PENDING" if _edit_status == "PENDING" else ""
            with st.expander(f"**{ticker}** — {pos['shares']} aandelen @ ${pos['avg_price']:.4f}{_tag}", expanded=False):
                with st.form(f"edit_{ticker}"):
                    ec1, ec2, ec3, ec4 = st.columns(4)
                    with ec1:
                        edit_shares = st.number_input(
                            "Totaal aantal aandelen",
                            min_value=0.0,
                            value=float(pos["shares"]),
                            step=0.00000001,
                            format="%.8f",
                        )
                    with ec2:
                        edit_price = st.number_input(
                            "Gemiddelde inkoopprijs ($)",
                            min_value=0.0,
                            value=float(pos["avg_price"]),
                            step=0.0001,
                            format="%.4f",
                        )
                    with ec3:
                        _bd_str = pos.get("buy_date") or pos.get("added", "")[:10]
                        try:
                            _bd_default = date.fromisoformat(_bd_str)
                        except Exception:
                            _bd_default = date.today()
                        edit_date = st.date_input("Aankoopdatum", value=_bd_default)
                    with ec4:
                        _status_idx = 0 if _edit_status == "ACTIEF" else 1
                        edit_status = st.selectbox("Status", ["ACTIEF", "PENDING"],
                                                   index=_status_idx)

                    sb1, sb2 = st.columns([2, 1])
                    with sb1:
                        if st.form_submit_button("Opslaan", use_container_width=True):
                            if edit_shares > 0 and edit_price > 0:
                                portfolio[ticker]["shares"]    = round(edit_shares, 8)
                                portfolio[ticker]["avg_price"] = round(edit_price, 4)
                                portfolio[ticker]["buy_date"]  = edit_date.isoformat()
                                portfolio[ticker]["status"]    = edit_status
                                save_portfolio(portfolio)
                                fetch_quotes.clear()
                                st.success(f"✓ {ticker} bijgewerkt: {edit_shares} aandelen @ ${edit_price:.4f} [{edit_status}]")
                                st.rerun()
                            else:
                                st.error("Aandelen en prijs moeten groter zijn dan 0.")
                    with sb2:
                        if st.form_submit_button("Verwijder positie", use_container_width=True):
                            del portfolio[ticker]
                            save_portfolio(portfolio)
                            fetch_quotes.clear()
                            st.rerun()
