import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from html import escape  # für sichere Tooltips

# Optional: Auto-Refresh (falls Paket installiert ist)
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# ---------------------------------------------------------
# BASIS-KONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crypto Live Ticker – TradingView Style V5",
    layout="wide",
)

# Bitfinex Public API (ohne API-Key)
BITFINEX_BASE_URL = "https://api-pub.bitfinex.com/v2"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CryptoTV-V5/1.0; +https://streamlit.io)"
}

# Symbole auf Bitfinex
SYMBOLS = {
    "BTC": "tBTCUSD",
    "ETH": "tETHUSD",
    "XRP": "tXRPUSD",
    "SOL": "tSOLUSD",
    "DOGE": "tDOGE:USD",
}

# Anzeige-Labels → interne Timeframes (Bitfinex: 1m..1D)
TIMEFRAMES = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",  # Bitfinex schreibt 1D
}

DEFAULT_TIMEFRAME = "1d"
VALID_SIGNALS = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]

# Wie viele Jahre Historie sollen ungefähr geladen werden?
YEARS_HISTORY = 3.0


def candles_for_history(interval_internal: str, years: float = YEARS_HISTORY) -> int:
    """Rechnet ungefähr aus, wie viele Kerzen für X Jahre gebraucht werden."""
    candles_per_day_map = {
        "1m": 60 * 24,   # 1440
        "5m": 12 * 24,   # 288
        "15m": 4 * 24,   # 96
        "1h": 24,        # 24
        "4h": 6,         # 6
        "1D": 1,         # 1
    }
    candles_per_day = candles_per_day_map.get(interval_internal, 24)
    return int(candles_per_day * 365 * years)


# ---------------------------------------------------------
# THEME CSS
# ---------------------------------------------------------
DARK_CSS = """
<style>
body, .main {
    background-color: #020617;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.tv-card {
    background: #020617;
    border-radius: 0.75rem;
    border: 1px solid #1f2933;
    padding: 0.75rem 1rem;
}
.tv-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: #9ca3af;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.signal-badge {
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
</style>
"""

LIGHT_CSS = """
<style>
body, .main {
    background-color: #F3F4F6;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.tv-card {
    background: #FFFFFF;
    border-radius: 0.75rem;
    border: 1px solid #E5E7EB;
    padding: 0.75rem 1rem;
}
.tv-title {
    font-weight: 600;
    font-size: 0.9rem;
    color: #6B7280;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.signal-badge {
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-weight: 600;
    display: inline-block;
}
</style>
"""

# ---------------------------------------------------------
# API FUNKTIONEN – BITFINEX
# ---------------------------------------------------------
def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    timeframe = interval  # z.B. "1m", "1h", "1D"
    key = f"trade:{timeframe}:{symbol}"
    url = f"{BITFINEX_BASE_URL}/candles/{key}/hist"

    params = {"limit": limit, "sort": -1}

    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Candles HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        raw = resp.json()
    except ValueError:
        raise RuntimeError(f"Candles: Ungültige JSON-Antwort: {resp.text[:200]}")

    if not isinstance(raw, list) or len(raw) == 0:
        return pd.DataFrame()

    rows = []
    for c in raw:
        # [MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]
        if len(c) < 6:
            continue
        rows.append(
            {
                "open_time": pd.to_datetime(c[0], unit="ms"),
                "open": float(c[1]),
                "close": float(c[2]),
                "high": float(c[3]),
                "low": float(c[4]),
                "volume": float(c[5]),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("open_time")
    df.sort_index(inplace=True)
    return df


@st.cache_data(ttl=60)
def cached_fetch_klines(symbol: str, interval: str, limit: int = 200):
    """Gecachter Candle-Abruf – reduziert Last & Rate-Limits."""
    return fetch_klines(symbol, interval, limit)


def fetch_ticker_24h(symbol: str):
    url = f"{BITFINEX_BASE_URL}/ticker/{symbol}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Ticker HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        d = resp.json()
    except ValueError:
        raise RuntimeError(f"Ticker: Ungültige JSON-Antwort: {resp.text[:200]}")

    if not isinstance(d, (list, tuple)) or len(d) < 7:
        raise RuntimeError(f"Ticker: Unerwartetes Format: {d}")

    last_price = float(d[6])
    change_pct = float(d[5]) * 100.0
    return last_price, change_pct


# ---------------------------------------------------------
# INDIKATOREN
# ---------------------------------------------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()

    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMA20/EMA50, MA200, Bollinger 20, RSI14.
    MA200 = klassischer Bitcoin-Makrotrendfilter.
    """
    if df.empty:
        return df

    close = df["close"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ma200"] = close.rolling(200).mean()

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    df["bb_mid"] = sma20
    df["bb_up"] = sma20 + 2 * std20
    df["bb_lo"] = sma20 - 2 * std20

    df["rsi14"] = compute_rsi(close)

    return df


# ---------------------------------------------------------
# SIGNAL-LOGIK (mit Begründung)
# ---------------------------------------------------------
def _signal_core_with_reason(last, prev):
    """
    Kernlogik:
    - Adaptive Bollinger
    - RSI Trend Confirmation
    - Blow-Off-Top Detector
    Liefert (signal, reason).
    """

    close = last["close"]
    prev_close = prev["close"]

    ema50 = last["ema50"]
    ma200 = last["ma200"]

    rsi_now = last["rsi14"]
    rsi_prev = prev["rsi14"]

    bb_up = last["bb_up"]
    bb_lo = last["bb_lo"]
    bb_mid = last["bb_mid"]

    high = last["high"]
    low = last["low"]
    body = abs(close - last["open"])
    candle_range = high - low
    upper_wick = high - max(close, last["open"])

    # Adaptive Volatility → passt Bollinger-Sensitivität an
    vol = (bb_up - bb_lo) / bb_mid if bb_mid != 0 else 0
    is_low_vol = vol < 0.06
    is_high_vol = vol > 0.12

    # MA200 fehlt → nicht traden
    if pd.isna(ma200):
        return "HOLD", "MA200 noch nicht verfügbar – zu wenig Historie, daher kein Trade."

    # Nur Long-Trading in Bullen-Trends
    if close < ma200:
        return "HOLD", "Kurs liegt unter MA200 – System handelt nur Long im Bullenmarkt."

    # -------------------------------------------------------
    # Blow-Off-Top Detector (Bitcoin-spezifisch)
    # -------------------------------------------------------
    blowoff = (
        candle_range > 0
        and upper_wick > candle_range * 0.45  # langer oberer Docht
        and close < prev_close                # Umkehrkerze
        and close > bb_up                     # über dem oberen BB
        and rsi_now > 73                      # RSI hoch
    )

    if blowoff:
        return (
            "STRONG SELL",
            "Blow-Off-Top: langer oberer Docht, Kurs über oberem Bollinger-Band "
            "und RSI > 73 mit Umkehrkerze – hohes Top-Risiko."
        )

    # -------------------------------------------------------
    # Adaptive STRONG BUY – tiefer Dip
    # -------------------------------------------------------
    deep_dip = (
        close <= bb_lo
        and rsi_now < 35
        and rsi_now > rsi_prev
    )

    if deep_dip:
        if is_low_vol and close < bb_lo * 0.995:
            return (
                "STRONG BUY",
                "Tiefer Dip: Kurs an/unter unterem Bollinger-Band in ruhiger Phase, "
                "RSI < 35 dreht nach oben – aggressiver Rebound-Einstieg."
            )
        return (
            "STRONG BUY",
            "Tiefer Dip: Kurs am unteren Bollinger-Band, RSI < 35 und steigt wieder – "
            "kräftiges Long-Signal."
        )

    # -------------------------------------------------------
    # BUY – normale gesunde Pullbacks
    # -------------------------------------------------------
    buy_price_cond = (
        close <= bb_lo * (1.01 if is_high_vol else 1.00)
        or close <= ema50 * 0.96
    )

    buy_rsi_cond = (
        30 < rsi_now <= 48
        and rsi_now > rsi_prev
    )

    if buy_price_cond and buy_rsi_cond:
        return (
            "BUY",
            "Gesunder Pullback: Kurs im Bereich unteres Bollinger-Band bzw. leicht unter EMA50, "
            "RSI zwischen 30 und 48 und dreht nach oben."
        )

    # -------------------------------------------------------
    # STRONG SELL – extreme Überhitzung
    # -------------------------------------------------------
    strong_sell_cond = (
        close > ema50 * 1.12
        and close > bb_up
        and rsi_now > 80
        and rsi_now < rsi_prev
    )

    if strong_sell_cond:
        return (
            "STRONG SELL",
            "Extreme Überhitzung: Kurs deutlich über EMA50 und oberem Bollinger-Band, "
            "RSI > 80 und fällt bereits – starkes Abverkaufsrisiko."
        )

    # -------------------------------------------------------
    # SELL – normale Übertreibung
    # -------------------------------------------------------
    sell_cond = (
        close > bb_up
        and rsi_now > 72
        and rsi_now < rsi_prev
    )

    if sell_cond:
        return (
            "SELL",
            "Übertreibung: Kurs über dem oberen Bollinger-Band, RSI > 72 und dreht nach unten – "
            "Gewinnmitnahme / Short-Signal."
        )

    # Nichts erkannt
    return "HOLD", "Keine klare Übertreibung oder Dip – System wartet (HOLD)."


def signal_for_pair(last, prev):
    """Alte einfache Schnittstelle: nur das Signal."""
    sig, _ = _signal_core_with_reason(last, prev)
    return sig


def signal_with_reason(last, prev):
    """Neue Schnittstelle: (signal, reason)."""
    return _signal_core_with_reason(last, prev)


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wendet signal_with_reason() an und gibt nur neue Signale aus,
    wenn sich die Richtung ändert → keine gespammten Wiederholungssignale.
    Zusätzlich Spalte 'signal_reason'.
    """
    if df.empty or len(df) < 2:
        df["signal"] = "NO DATA"
        df["signal_reason"] = "Nicht genug Daten für ein Signal."
        return df

    signals = []
    reasons = []
    last_sig = "NO DATA"

    for i in range(len(df)):
        if i == 0:
            signals.append("NO DATA")
            reasons.append("Erste Candle – keine Historie für Signalberechnung.")
            continue

        sig_raw, reason_raw = signal_with_reason(df.iloc[i], df.iloc[i - 1])

        # nur neues Signal, wenn Richtung wechselt
        if sig_raw == last_sig:
            sig_display = "HOLD"
            reason_display = f"Signal '{sig_raw}' besteht weiter – kein neues Signal generiert."
        else:
            sig_display = sig_raw
            reason_display = reason_raw

        signals.append(sig_display)
        reasons.append(reason_display)

        if sig_raw in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
            last_sig = sig_raw

    df["signal"] = signals
    df["signal_reason"] = reasons
    return df


# ---------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------
def latest_signal(df: pd.DataFrame) -> str:
    if "signal" not in df.columns or df.empty:
        return "NO DATA"
    valid = df[df["signal"].isin(VALID_SIGNALS)]
    return valid["signal"].iloc[-1] if not valid.empty else "NO DATA"


def compute_backtest_trades(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Erzeugt eine Backtest-Tabelle:
    entry_time, exit_time, signal, reason, entry_price, exit_price, ret_pct, correct
    """
    if df.empty or "signal" not in df.columns:
        return pd.DataFrame()

    rows = []
    closes = df["close"].values
    signals = df["signal"].values
    idx = df.index

    has_reason = "signal_reason" in df.columns

    for i in range(len(df) - horizon):
        sig = signals[i]
        if sig not in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
            continue

        entry = closes[i]
        exit_ = closes[i + horizon]
        if entry == 0:
            continue

        ret = (exit_ - entry) / entry * 100
        direction = 1 if sig in ["BUY", "STRONG BUY"] else -1
        correct = (np.sign(ret) * direction) > 0
        reason = df["signal_reason"].iloc[i] if has_reason else ""

        rows.append(
            {
                "entry_time": idx[i],
                "exit_time": idx[i + horizon],
                "signal": sig,
                "reason": reason,
                "entry_price": entry,
                "exit_price": exit_,
                "ret_pct": float(ret),
                "correct": bool(correct),
            }
        )

    return pd.DataFrame(rows)


def summarize_backtest(df_bt: pd.DataFrame):
    if df_bt.empty:
        return {}

    summary = {
        "total_trades": int(len(df_bt)),
        "overall_avg_return": float(df_bt["ret_pct"].mean()),
        "overall_hit_rate": float(df_bt["correct"].mean() * 100),
    }

    per = []
    for sig in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
        sub = df_bt[df_bt["signal"] == sig]
        if sub.empty:
            continue
        per.append(
            {
                "Signal": sig,
                "Trades": len(sub),
                "Avg Return %": float(sub["ret_pct"].mean()),
                "Hit Rate %": float(sub["correct"].mean() * 100),
            }
        )

    summary["per_type"] = per
    return summary


def signal_color(signal: str) -> str:
    return {
        "STRONG BUY": "#00C853",
        "BUY": "#64DD17",
        "HOLD": "#9E9E9E",
        "SELL": "#FF5252",
        "STRONG SELL": "#D50000",
        "NO DATA": "#757575",
    }.get(signal, "#9E9E9E")


# ---------------------------------------------------------
# PLOTLY CHARTS – GEMEINSAMER PRICE+RSI FIGURE
# ---------------------------------------------------------
def base_layout_kwargs(theme: str):
    if theme == "Dark":
        bg, fg = "#020617", "#E5E7EB"
    else:
        bg, fg = "#FFFFFF", "#111827"

    return dict(
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=fg),
    )


def grid_color_for_theme(theme: str) -> str:
    return "#111827" if theme == "Dark" else "#E5E7EB"


def create_price_rsi_figure(df, symbol_label, timeframe_label, theme):
    """
    Ein gemeinsamer Plot mit 2 Reihen:
    - oben: Price + EMA + Bollinger + Volume
    - unten: RSI (14)
    shared_xaxes=True → Zoom & Range sind synchron.
    """

    # --- Farb-Setup (TradingView-like) ---
    BULL_COLOR = "#22c55e"   # grüne Candles
    BEAR_COLOR = "#ef4444"   # rote Candles

    EMA20_COLOR = "#FF9800"  # Orange – EMA20
    EMA50_COLOR = "#2196F3"  # Blau – EMA50

    if theme == "Dark":
        # Dezentes Grau/Weiß für Bollinger in dunklem Chart
        BB_LINE_COLOR = "#d1d5db"                          # hellgraue Linie
        BB_FILL_COLOR = "rgba(209,213,219,0.10)"           # super sanftes Grau
        BB_MID_COLOR = "#9ca3af"                           # Mittelband: graublau
    else:
        # Dezentes Hellgrau/Graublau für helles Chart
        BB_LINE_COLOR = "#94a3b8"                          # graublau
        BB_FILL_COLOR = "rgba(148,163,184,0.07)"           # sehr leichtes Grau
        BB_MID_COLOR = "#6b7280"                           # dunkleres Grau

    layout_kwargs = base_layout_kwargs(theme)
    bg = layout_kwargs["plot_bgcolor"]
    fg = layout_kwargs["font"]["color"]
    grid = grid_color_for_theme(theme)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        subplot_titles=(f"{symbol_label}/USD — {timeframe_label}", "RSI (14)"),
    )

    fig.update_layout(
        height=720,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=fg),
        margin=dict(l=10, r=10, t=60, b=40),
        xaxis_rangeslider_visible=False,
    )

    # --- OBERES PANEL: BOLLINGER + PRICE + VOLUME ---

    # 1) Bollinger-Band-Fläche als eigenes Polygon (liegt sicher HINTER allem)
    if {"bb_up", "bb_lo"}.issubset(df.columns):
        band_x = list(df.index) + list(df.index[::-1])
        band_y = list(df["bb_up"]) + list(df["bb_lo"][::-1])

        fig.add_trace(
            go.Scatter(
                x=band_x,
                y=band_y,
                name="BB Area",
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor=BB_FILL_COLOR,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    # 2) Candles (liegen über dem Band)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_fillcolor=BULL_COLOR,
            increasing_line_color=BULL_COLOR,
            decreasing_fillcolor=BEAR_COLOR,
            decreasing_line_color=BEAR_COLOR,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # 3) Bollinger-Linien (Upper/Lower/Mid) über Candles & Band
    if "bb_up" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_up"],
                name="BB Upper",
                mode="lines",
                line=dict(width=1, color=BB_LINE_COLOR),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    if "bb_lo" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_lo"],
                name="BB Lower",
                mode="lines",
                line=dict(width=1, color=BB_LINE_COLOR),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    if "bb_mid" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_mid"],
                name="BB Basis",
                mode="lines",
                line=dict(width=1, dash="dot", color=BB_MID_COLOR),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    # 4) EMA20 / EMA50 / EMA200 (TradingView-Standardfarben)
    if "ema20" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["ema20"],
                name="EMA20",
                mode="lines",
                line=dict(width=1.5, color=EMA20_COLOR),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    if "ema50" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["ema50"],
                name="EMA50",
                mode="lines",
                line=dict(width=1.5, color=EMA50_COLOR),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    # MA200 einzeichnen (wichtige Trendlinie)
    if "ma200" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["ma200"],
                name="MA200",
                mode="lines",
                line=dict(width=1.8, color="#e5e7eb"),  # hellgrau/weiß wie in TradingView
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    # 5) Volume auf zweiter Y-Achse (ohne Label-Text)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            opacity=0.3,
            marker=dict(color="#f59e0b"),
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # --- UNTERES PANEL: RSI (14) ---

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["rsi14"],
            mode="lines",
            name="RSI14",
            line=dict(width=1.5, color="#a855f7"),  # violett
        ),
        row=2,
        col=1,
    )

    # RSI Level-Linien (nur im unteren Panel)
    line_color = "#e5e7eb" if theme == "Dark" else "#6B7280"
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color=line_color,
        line_width=1,
        row=2,
        col=1,
    )
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color=line_color,
        line_width=1,
        row=2,
        col=1,
    )

    # --- Layout / Achsen ---

    # Price-Achse links
    fig.update_yaxes(
        title_text="Price",
        showgrid=True,
        gridcolor=grid,
        row=1,
        col=1,
        secondary_y=False,
    )

    # Volume-Achse rechts – kein Titel, nur Skala
    fig.update_yaxes(
        title_text="",
        showgrid=False,
        row=1,
        col=1,
        secondary_y=True,
    )

    # RSI-Achse
    fig.update_yaxes(
        title_text="RSI",
        range=[0, 100],
        showgrid=True,
        gridcolor=grid,
        row=2,
        col=1,
    )

    # X-Achse nur unten beschriften
    fig.update_xaxes(
        title_text="Time",
        showgrid=False,
        row=2,
        col=1,
    )
    fig.update_xaxes(showgrid=False, row=1, col=1)

    return fig


def create_signal_history_figure(df, allowed, theme):
    """Signal-Historie als eigener Chart – mit Begründung im Hover."""
    fig = go.Figure()

    levels = {
        "STRONG SELL": -2,
        "SELL": -1,
        "HOLD": 0,
        "BUY": 1,
        "STRONG BUY": 2,
    }

    if "signal" not in df.columns:
        df = df.copy()
        df["signal"] = "NO DATA"

    if "signal_reason" not in df.columns:
        df = df.copy()
        df["signal_reason"] = ""

    df2 = df[df["signal"].isin(levels.keys())].copy()
    df2["lvl"] = df2["signal"].map(levels)
    df2 = df2[df2["signal"].isin(allowed)]

    for sig, lvl in levels.items():
        if sig not in allowed:
            continue
        sub = df2[df2["signal"] == sig]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub.index,
                y=[lvl] * len(sub),
                mode="markers",
                name=sig,
                marker=dict(size=8),
                text=sub["signal_reason"],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"Signal: {sig}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        )

    layout_kwargs = base_layout_kwargs(theme)
    bg = layout_kwargs["plot_bgcolor"]
    fg = layout_kwargs["font"]["color"]
    grid = grid_color_for_theme(theme)

    fig.update_layout(
        title="Signal History",
        height=220,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=fg),
    )

    fig.update_yaxes(
        tickvals=[-2, -1, 0, 1, 2],
        ticktext=list(levels.keys()),
        range=[-2.5, 2.5],
        showgrid=True,
        gridcolor=grid,
    )

    return fig



def create_signal_history_figure(df, allowed, theme):
    """Signal-Historie als eigener Chart – mit Begründung im Hover."""
    fig = go.Figure()

    levels = {
        "STRONG SELL": -2,
        "SELL": -1,
        "HOLD": 0,
        "BUY": 1,
        "STRONG BUY": 2,
    }

    if "signal" not in df.columns:
        df = df.copy()
        df["signal"] = "NO DATA"

    if "signal_reason" not in df.columns:
        df = df.copy()
        df["signal_reason"] = ""

    df2 = df[df["signal"].isin(levels.keys())].copy()
    df2["lvl"] = df2["signal"].map(levels)
    df2 = df2[df2["signal"].isin(allowed)]

    for sig, lvl in levels.items():
        if sig not in allowed:
            continue
        sub = df2[df2["signal"] == sig]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub.index,
                y=[lvl] * len(sub),
                mode="markers",
                name=sig,
                marker=dict(size=8),
                text=sub["signal_reason"],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"Signal: {sig}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        )

    layout_kwargs = base_layout_kwargs(theme)
    bg = layout_kwargs["plot_bgcolor"]
    fg = layout_kwargs["font"]["color"]
    grid = grid_color_for_theme(theme)

    fig.update_layout(
        title="Signal History",
        height=220,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=fg),
    )

    fig.update_yaxes(
        tickvals=[-2, -1, 0, 1, 2],
        ticktext=list(levels.keys()),
        range=[-2.5, 2.5],
        showgrid=True,
        gridcolor=grid,
    )

    return fig



def create_signal_history_figure(df, allowed, theme):
    """Signal-Historie als eigener Chart – mit Begründung im Hover."""
    fig = go.Figure()

    levels = {
        "STRONG SELL": -2,
        "SELL": -1,
        "HOLD": 0,
        "BUY": 1,
        "STRONG BUY": 2,
    }

    if "signal" not in df.columns:
        df = df.copy()
        df["signal"] = "NO DATA"

    if "signal_reason" not in df.columns:
        df = df.copy()
        df["signal_reason"] = ""

    df2 = df[df["signal"].isin(levels.keys())].copy()
    df2["lvl"] = df2["signal"].map(levels)
    df2 = df2[df2["signal"].isin(allowed)]

    for sig, lvl in levels.items():
        if sig not in allowed:
            continue
        sub = df2[df2["signal"] == sig]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub.index,
                y=[lvl] * len(sub),
                mode="markers",
                name=sig,
                marker=dict(size=8),
                text=sub["signal_reason"],
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"Signal: {sig}<br>"
                    "%{text}<extra></extra>"
                ),
            )
        )

    layout_kwargs = base_layout_kwargs(theme)
    bg = layout_kwargs["plot_bgcolor"]
    fg = layout_kwargs["font"]["color"]
    grid = grid_color_for_theme(theme)

    fig.update_layout(
        title="Signal History",
        height=220,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=fg),
    )

    fig.update_yaxes(
        tickvals=[-2, -1, 0, 1, 2],
        ticktext=list(levels.keys()),
        range=[-2.5, 2.5],
        showgrid=True,
        gridcolor=grid,
    )

    return fig


# ---------------------------------------------------------
# SESSION STATE INITIALISIERUNG
# ---------------------------------------------------------
def init_state():
    st.session_state.setdefault("selected_symbol", "BTC")
    st.session_state.setdefault("selected_timeframe", DEFAULT_TIMEFRAME)
    st.session_state.setdefault("theme", "Dark")
    st.session_state.setdefault("backtest_horizon", 5)
    st.session_state.setdefault("backtest_trades", pd.DataFrame())


# ---------------------------------------------------------
# HAUPT UI / STREAMLIT APP
# ---------------------------------------------------------
def main():
    init_state()

    # Auto-Refresh (TradingView Feel)
    if st_autorefresh is not None:
        st_autorefresh(interval=60 * 1000, key="refresh")

    # Sidebar: Theme Toggle
    st.sidebar.title("⚙️ Einstellungen")
    theme = st.sidebar.radio(
        "Theme",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1,
    )
    st.session_state.theme = theme

    st.markdown(DARK_CSS if theme == "Dark" else LIGHT_CSS, unsafe_allow_html=True)

    # Header Bar
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(
        f"""
        <div class="tv-card" style="margin-bottom: 0.6rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="tv-title">Crypto Live Ticker</div>
                    <div style="font-size:1.1rem; font-weight:600;">
                        TradingView Style • Desktop V5
                    </div>
                </div>
                <div style="text-align:right; font-size:0.85rem; opacity:0.8;">
                    Datenquelle: Bitfinex Spot<br/>
                    Letztes Update: {now}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Layout: Watchlist links, Charts rechts
    col_left, col_right = st.columns([2, 5], gap="medium")

    # ---------------------------------------------------------
    # WATCHLIST (LEFT PANEL)
    # ---------------------------------------------------------
    with col_left:
        with st.container():
            st.markdown('<div class="tv-card">', unsafe_allow_html=True)
            st.markdown('<div class="tv-title">Watchlist</div>', unsafe_allow_html=True)

            sel = st.radio(
                "Symbol",
                list(SYMBOLS.keys()),
                index=list(SYMBOLS.keys()).index(st.session_state.selected_symbol),
                label_visibility="collapsed",
            )
            st.session_state.selected_symbol = sel

            rows = []
            selected_tf_label = st.session_state.selected_timeframe
            selected_tf_internal = TIMEFRAMES[selected_tf_label]
            limit_watch = candles_for_history(selected_tf_internal, years=YEARS_HISTORY)

            for label, sym in SYMBOLS.items():
                try:
                    price, chg_pct = fetch_ticker_24h(sym)
                    try:
                        df_tmp = cached_fetch_klines(sym, selected_tf_internal, limit=limit_watch)
                        df_tmp = compute_indicators(df_tmp)
                        df_tmp = compute_signals(df_tmp)
                        sig = latest_signal(df_tmp)
                    except Exception:
                        sig = "NO DATA"

                    rows.append(
                        {
                            "Symbol": label,
                            "Price": price,
                            "Change %": chg_pct,
                            "Signal": sig,
                        }
                    )
                except Exception:
                    rows.append(
                        {
                            "Symbol": label,
                            "Price": np.nan,
                            "Change %": np.nan,
                            "Signal": "NO DATA",
                        }
                    )

            df_watch = pd.DataFrame(rows).set_index("Symbol")

            def highlight(row):
                theme_local = st.session_state.theme
                if row.name == st.session_state.selected_symbol:
                    bg = "#111827" if theme_local == "Dark" else "#D1D5DB"
                    fg = "white" if theme_local == "Dark" else "black"
                    return [f"background-color:{bg}; color:{fg}"] * len(row)
                return [""] * len(row)

            styled = df_watch.style.apply(highlight, axis=1).format(
                {"Price": "{:,.2f}", "Change %": "{:+.2f}"}
            )

            st.dataframe(styled, use_container_width=True, height=270)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        with st.container():
            st.markdown('<div class="tv-card">', unsafe_allow_html=True)
            st.markdown('<div class="tv-title">System</div>', unsafe_allow_html=True)
            st.write("🖥️ Modus: Desktop TradingView-Style V5")
            st.write("📡 Feed: Bitfinex Spot (REST API, Public)")
            st.write("📏 Panels: Price+Volume, RSI, Signals, Backtest, Trades")
            st.write("🎨 Theme: Dark / Light (Sidebar)")
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # MAIN CHART AREA (RIGHT PANEL)
    # ---------------------------------------------------------
    with col_right:
        with st.container():
            st.markdown('<div class="tv-card">', unsafe_allow_html=True)

            symbol_label = st.session_state.selected_symbol
            symbol = SYMBOLS[symbol_label]
            tf_label = st.session_state.selected_timeframe
            interval_internal = TIMEFRAMES[tf_label]

            st.markdown('<div class="tv-title">Chart</div>', unsafe_allow_html=True)

            # Timeframe-Buttons
            cols_tf = st.columns(len(TIMEFRAMES))
            for i, tf in enumerate(TIMEFRAMES.keys()):
                with cols_tf[i]:
                    if st.button(tf, key=f"tf_{tf}"):
                        st.session_state.selected_timeframe = tf
                        st.rerun()

            # Daten abrufen
            try:
                limit_main = candles_for_history(interval_internal, years=YEARS_HISTORY)
                df_raw = cached_fetch_klines(symbol, interval_internal, limit=limit_main)
                df = compute_indicators(df_raw.copy())
                df = compute_signals(df)

                sig = latest_signal(df)
                last = df.iloc[-1]
                prev = df.iloc[-2]

                last_price = last["close"]
                change_abs = last_price - prev["close"]
                change_pct = (change_abs / prev["close"]) * 100 if prev["close"] != 0 else 0
                last_time = df.index[-1]
                signal_reason = last.get("signal_reason", "")

                feed_ok = True
                error_msg = ""
            except Exception as e:
                df = pd.DataFrame()
                sig = "NO DATA"
                last_price = 0
                change_abs = 0
                change_pct = 0
                last_time = None
                signal_reason = ""
                feed_ok = False
                error_msg = str(e)

            # Top-Kennzahlen
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.caption("Preis")
                st.markdown(f"**{last_price:,.2f} USD**" if feed_ok else "–")

            with k2:
                st.caption("Change letzte Candle")
                if feed_ok:
                    s = "+" if change_abs >= 0 else "-"
                    st.markdown(f"**{s}{abs(change_abs):.2f} ({s}{abs(change_pct):.2f}%)**")
                else:
                    st.markdown("–")

            with k3:
                st.caption("Signal")
                reason_html = escape(signal_reason, quote=True)
                st.markdown(
                    f'<span class="signal-badge" style="background-color:{signal_color(sig)};" '
                    f'title="{reason_html}">{sig}</span>',
                    unsafe_allow_html=True,
                )

            with k4:
                st.caption("Status")
                if feed_ok:
                    st.markdown("🟢 **Live**")
                    st.caption(f"Letzte Candle: {last_time}")
                else:
                    st.markdown("🔴 **Fehler**")
                    st.caption(error_msg[:80])

            st.markdown("---")

            # Gemeinsamer Price+RSI-Chart
            if not df.empty:
                fig_price_rsi = create_price_rsi_figure(df, symbol_label, tf_label, theme)
                st.plotly_chart(fig_price_rsi, use_container_width=True)
            else:
                st.warning("Keine Daten geladen – API/Internet prüfen.")

            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # SIGNAL-HISTORY + BACKTEST PANELS
        # ---------------------------------------------------------
        st.markdown("")
        col_hist, col_bt = st.columns([3, 2])

        # Signal-History Panel
        with col_hist:
            with st.container():
                st.markdown('<div class="tv-card">', unsafe_allow_html=True)
                st.markdown('<div class="tv-title">Signal History</div>', unsafe_allow_html=True)

                if df.empty:
                    st.info("Keine Signale verfügbar.")
                else:
                    allow = st.multiselect(
                        "Signale anzeigen",
                        VALID_SIGNALS,
                        default=VALID_SIGNALS,
                    )
                    st.plotly_chart(
                        create_signal_history_figure(df, allow, theme),
                        use_container_width=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

        # Backtest Panel
        with col_bt:
            with st.container():
                st.markdown('<div class="tv-card">', unsafe_allow_html=True)
                st.markdown('<div class="tv-title">Backtest</div>', unsafe_allow_html=True)

                if df.empty:
                    st.info("Keine Daten.")
                else:
                    horizon = st.slider(
                        "Halte-Dauer (Kerzen)",
                        1,
                        20,
                        value=st.session_state.backtest_horizon,
                    )
                    st.session_state.backtest_horizon = horizon

                    bt = compute_backtest_trades(df, horizon)
                    st.session_state.backtest_trades = bt

                    stats = summarize_backtest(bt)

                    if not stats:
                        st.info("Keine verwertbaren Trades.")
                    else:
                        st.markdown(f"**Trades gesamt:** {stats['total_trades']}")
                        st.markdown(f"**Ø Return:** {stats['overall_avg_return']:.2f}%")
                        st.markdown(f"**Trefferquote:** {stats['overall_hit_rate']:.1f}%")

                        if stats.get("per_type"):
                            st.markdown("---")
                            st.caption("Pro Signal:")
                            st.table(pd.DataFrame(stats["per_type"]))

                st.markdown("</div>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # TRADES LIST – MIT CSV EXPORT
        # ---------------------------------------------------------
        st.markdown("")
        with st.container():
            st.markdown('<div class="tv-card">', unsafe_allow_html=True)
            st.markdown('<div class="tv-title">Trades List (Backtest)</div>', unsafe_allow_html=True)

            bt = st.session_state.backtest_trades

            if bt.empty:
                st.info("Noch keine Trades.")
            else:
                df_show = bt.copy()
                df_show["entry_time"] = df_show["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
                df_show["exit_time"] = df_show["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
                df_show["ret_pct"] = df_show["ret_pct"].map(lambda x: f"{x:.2f}")
                df_show["correct"] = df_show["correct"].map(lambda x: "✅" if x else "❌")

                cols = [
                    "entry_time",
                    "exit_time",
                    "signal",
                    "reason",
                    "entry_price",
                    "exit_price",
                    "ret_pct",
                    "correct",
                ]
                df_show = df_show[[c for c in cols if c in df_show.columns]]

                st.dataframe(df_show, use_container_width=True, height=260)

                csv = bt.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 CSV Export",
                    csv,
                    file_name=f"trades_{symbol_label}_{tf_label}.csv",
                    mime="text/csv",
                )

            st.markdown("</div>", unsafe_allow_html=True)

        # Refresh Button
        st.markdown("")
        r1, r2 = st.columns([1, 5])
        with r1:
            if st.button("🔄 Refresh"):
                st.rerun()

        with r2:
            st.caption("Charts aktualisieren automatisch alle 60 Sekunden (oder manuell per Button).")


# ---------------------------------------------------------
# LAUNCH
# ---------------------------------------------------------
if __name__ == "__main__":
    main()







