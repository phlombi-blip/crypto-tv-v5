import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# BASIS-KONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crypto Live Ticker – TradingView Style V5",
    layout="wide",
)

BINANCE_BASE_URL = "https://api.binance.com"

SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
    "SOL": "SOLUSDT",
    "DOGE": "DOGEUSDT",
}

TIMEFRAMES = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

DEFAULT_TIMEFRAME = "1h"
VALID_SIGNALS = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]

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
# API FUNKTIONEN
# ---------------------------------------------------------
def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Direkter Binance-Kline Abruf."""
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    rows = [
        {
            "open_time": pd.to_datetime(c[0], unit="ms"),
            "open": float(c[1]),
            "high": float(c[2]),
            "low": float(c[3]),
            "close": float(c[4]),
            "volume": float(c[5]),
        }
        for c in raw
    ]

    df = pd.DataFrame(rows).set_index("open_time")
    return df


@st.cache_data(ttl=60)
def cached_fetch_klines(symbol: str, interval: str, limit: int = 200):
    """Gecachter Binance Call – verbessert Performance & Rate Limits."""
    return fetch_klines(symbol, interval, limit)


def fetch_ticker_24h(symbol: str):
    url = f"{BINANCE_BASE_URL}/api/v3/ticker/24hr"
    params = {"symbol": symbol}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    d = r.json()
    return float(d["lastPrice"]), float(d["priceChangePercent"])

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
    if df.empty:
        return df

    close = df["close"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()

    sma = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    df["bb_middle"] = sma
    df["bb_upper"] = sma + std * 2
    df["bb_lower"] = sma - std * 2

    df["rsi14"] = compute_rsi(close)

    return df

# ---------------------------------------------------------
# SIGNAL-LOGIK
# ---------------------------------------------------------
def signal_for_pair(last, prev):
    """Regelbasierte technische Signal-Engine."""
    ema20, ema50 = last["ema20"], last["ema50"]
    close = last["close"]
    bb_u, bb_l, bb_m = last["bb_upper"], last["bb_lower"], last["bb_middle"]
    rsi_now, rsi_prev = last["rsi14"], prev["rsi14"]

    # STRONG BUY
    if ema20 > ema50 and close <= bb_l and rsi_prev < 30 and rsi_now > 30:
        return "STRONG BUY"

    # BUY
    if (
        ema20 > ema50
        and bb_l < close <= bb_m
        and 30 <= rsi_now <= 50
        and rsi_now > rsi_prev
    ):
        return "BUY"

    # STRONG SELL
    if ema20 < ema50 and close >= bb_u and rsi_prev > 70 and rsi_now < 70:
        return "STRONG SELL"

    # SELL
    if (
        ema20 < ema50
        and bb_m <= close < bb_u
        and 50 <= rsi_now <= 70
        and rsi_now < rsi_prev
    ):
        return "SELL"

    return "HOLD"


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 21:
        df["signal"] = "NO DATA"
        return df

    signals = []
    for i in range(len(df)):
        if i < 20:
            signals.append("NO DATA")
            continue
        signals.append(signal_for_pair(df.iloc[i], df.iloc[i - 1]))

    df["signal"] = signals
    return df
# ---------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------
def latest_signal(df: pd.DataFrame) -> str:
    valid = df[df["signal"].isin(VALID_SIGNALS)]
    return valid["signal"].iloc[-1] if not valid.empty else "NO DATA"


def compute_backtest_trades(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Erzeugt eine Backtest-Tabelle:
    entry_time, exit_time, signal, entry_price, exit_price, ret_pct, correct
    """
    if df.empty or "signal" not in df.columns:
        return pd.DataFrame()

    rows = []
    closes = df["close"].values
    signals = df["signal"].values
    idx = df.index

    for i in range(len(df) - horizon):
        sig = signals[i]
        if sig not in ["STRONG BUY", "BUY", "SELL", "STRONG SELL"]:
            continue

        entry = closes[i]
        exit = closes[i + horizon]
        if entry == 0:
            continue

        ret = (exit - entry) / entry * 100
        direction = 1 if sig in ["BUY", "STRONG BUY"] else -1
        correct = (np.sign(ret) * direction) > 0

        rows.append(
            {
                "entry_time": idx[i],
                "exit_time": idx[i + horizon],
                "signal": sig,
                "entry_price": entry,
                "exit_price": exit,
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
# PLOTLY CHARTS
# ---------------------------------------------------------
def base_layout_kwargs(theme: str):
    if theme == "Dark":
        bg, fg, grid = "#020617", "#E5E7EB", "#111827"
    else:
        bg, fg, grid = "#FFFFFF", "#111827", "#E5E7EB"

    return dict(
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(color=fg),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=grid),
    )


def create_price_figure(df, symbol_label, timeframe_label, theme):
    fig = go.Figure()

    # Candles
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )

    # EMA20/50
    if "ema20" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema20"], name="EMA20", mode="lines"))
    if "ema50" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema50"], name="EMA50", mode="lines"))

    # Bollinger
    if "bb_upper" in df:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper", mode="lines", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_middle"], name="BB Mid", mode="lines", line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower", mode="lines", line=dict(width=1)))

    fig.update_layout(
        title=f"{symbol_label}/USDT — {timeframe_label}",
        height=520,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        **base_layout_kwargs(theme),
    )
    return fig


def create_volume_figure(df, theme):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume"))
    fig.update_layout(
        title="Volume",
        height=200,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        **base_layout_kwargs(theme),
    )
    return fig


def create_rsi_figure(df, theme):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["rsi14"], mode="lines", name="RSI14"))

    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")

    fig.update_layout(
        title="RSI (14)",
        height=200,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 100]),
        **base_layout_kwargs(theme),
    )
    return fig


def create_signal_history_figure(df, allowed, theme):
    fig = go.Figure()

    levels = {
        "STRONG SELL": -2,
        "SELL": -1,
        "HOLD": 0,
        "BUY": 1,
        "STRONG BUY": 2,
    }

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
            )
        )

    fig.update_layout(
        title="Signal History",
        height=220,
        hovermode="x unified",
        yaxis=dict(
            tickvals=[-2, -1, 0, 1, 2],
            ticktext=list(levels.keys()),
            range=[-2.5, 2.5],
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        **base_layout_kwargs(theme),
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

    # --------------------------------------
    # Auto-Refresh (TradingView Feel)
    # --------------------------------------
    st_autorefresh(interval=60 * 1000, key="refresh")

    # --------------------------------------
    # Sidebar: Theme Toggle
    # --------------------------------------
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
                    Datenquelle: Binance Spot<br/>
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

            # Symbol-Auswahl
            sel = st.radio(
                "Symbol",
                list(SYMBOLS.keys()),
                index=list(SYMBOLS.keys()).index(st.session_state.selected_symbol),
                label_visibility="collapsed",
            )
            st.session_state.selected_symbol = sel

            # Watchlist-Daten
            rows = []
            selected_tf = st.session_state.selected_timeframe
            for label, sym in SYMBOLS.items():
                try:
                    price, chg = fetch_ticker_24h(sym)

                    try:
                        df_tmp = cached_fetch_klines(sym, TIMEFRAMES[selected_tf], limit=120)
                        df_tmp = compute_indicators(df_tmp)
                        df_tmp = compute_signals(df_tmp)
                        sig = latest_signal(df_tmp)
                    except:
                        sig = "NO DATA"

                    rows.append({"Symbol": label, "Price": price, "Change %": chg, "Signal": sig})
                except:
                    rows.append({"Symbol": label, "Price": np.nan, "Change %": np.nan, "Signal": "NO DATA"})

            df_watch = pd.DataFrame(rows).set_index("Symbol")

            def highlight(row):
                if row.name == st.session_state.selected_symbol:
                    return ["background-color:#111827; color:white"] * len(row)
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
            st.write("📡 Feed: Binance Spot (REST API)")
            st.write("📏 Panels: Price, Volume, RSI, Signals, Backtest, Trades")
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
            interval = TIMEFRAMES[tf_label]

            st.markdown('<div class="tv-title">Chart</div>', unsafe_allow_html=True)

            # Timeframe-Buttons
            cols_tf = st.columns(len(TIMEFRAMES))
            for i, tf in enumerate(TIMEFRAMES.keys()):
                with cols_tf[i]:
                    if st.button(tf, key=f"tf_{tf}"):
                        st.session_state.selected_timeframe = tf
                        st.experimental_rerun()

            # Daten abrufen
            try:
                df_raw = cached_fetch_klines(symbol, interval, limit=240)
                df = compute_indicators(df_raw.copy())
                df = compute_signals(df)

                sig = latest_signal(df)
                last = df.iloc[-1]
                prev = df.iloc[-2]

                last_price = last["close"]
                change_abs = last_price - prev["close"]
                change_pct = (change_abs / prev["close"]) * 100 if prev["close"] != 0 else 0
                last_time = df.index[-1]

                feed_ok = True
            except Exception as e:
                df = pd.DataFrame()
                sig = "NO DATA"
                last_price = 0
                change_abs = 0
                change_pct = 0
                last_time = None
                feed_ok = False
                error_msg = str(e)

            # Top-Kennzahlen
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.caption("Preis")
                st.markdown(f"**{last_price:,.2f} USDT**" if feed_ok else "–")

            with k2:
                st.caption("Change letzte Candle")
                if feed_ok:
                    s = "+" if change_abs >= 0 else "-"
                    st.markdown(f"**{s}{abs(change_abs):.2f} ({s}{abs(change_pct):.2f}%)**")
                else:
                    st.markdown("–")

            with k3:
                st.caption("Signal")
                st.markdown(
                    f'<span class="signal-badge" style="background-color:{signal_color(sig)};">{sig}</span>',
                    unsafe_allow_html=True,
                )

            with k4:
                st.caption("Status")
                if feed_ok:
                    st.markdown("🟢 **Live**")
                    st.caption(f"Letzte Candle: {last_time}")
                else:
                    st.markdown("🔴 **Fehler**")
                    st.caption(error_msg[:60])

            st.markdown("---")

            # Charts rendern
            if not df.empty:
                st.plotly_chart(create_price_figure(df, symbol_label, tf_label, theme), use_container_width=True)
                st.plotly_chart(create_volume_figure(df, theme), use_container_width=True)
                st.plotly_chart(create_rsi_figure(df, theme), use_container_width=True)
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
                    st.plotly_chart(create_signal_history_figure(df, allow, theme), use_container_width=True)

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
                        1, 20,
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
                st.experimental_rerun()
        with r2:
            st.caption("Charts aktualisieren automatisch alle 60 Sekunden.")


# ---------------------------------------------------------
# LAUNCH
# ---------------------------------------------------------
if __name__ == "__main__":
    main()