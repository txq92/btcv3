#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Advanced cryptocurrency trend detector with reversal signals."""

import os
import sys
import time

# Fix encoding for Windows
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
import numpy as np
import pandas as pd
import pytz
import requests
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Tuple, List, Optional
from datetime import datetime


# ============== CONFIG ==============
SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT", 
    "SUIUSDT",
    "SOLUSDT",
    "WLDUSDT",
    "ADAUSDT",
    "ENAUSDT",
    "TRUMPUSDT",
    "TONUSDT"
]
INTERVAL = "3m"
LIMIT = 120

# API Configs
BASE = "https://api.binance.com"
TZ = pytz.timezone("Asia/Ho_Chi_Minh")
API_TIMEOUT = 20               # Timeout for API calls (seconds)
TELEGRAM_TIMEOUT = 10         # Timeout for Telegram API (seconds)
RETRY_DELAY = 10             # Wait time between retries on error (seconds)
SYMBOL_DELAY = 1             # Delay between fetching data for each coin (seconds)

# Operational parameters
LOOP_SLEEP_SECONDS = 15         # Sleep between main loops
SEND_IMAGES = False              # Send images with signals
MTF_CONFIRM = True              # Confirm according to 15m trend
SEND_MARKET_DIRECTION = False    # Enable/Disable Market Direction alerts

# Enhanced R:R Configuration
SYMBOL_SPECIFIC_RR = {
    "BTCUSDT": {"min_rr": 1.2, "risk_percent": 1.0, "atr_sl_mult": 1.2, "atr_tp_mult": 2.5},
    "ETHUSDT": {"min_rr": 1.5, "risk_percent": 0.8, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0},
    "SUIUSDT": {"min_rr": 1.8, "risk_percent": 0.5, "atr_sl_mult": 1.8, "atr_tp_mult": 3.5},
    "SOLUSDT": {"min_rr": 1.8, "risk_percent": 0.5, "atr_sl_mult": 1.8, "atr_tp_mult": 3.5},
    "WLDUSDT": {"min_rr": 1.8, "risk_percent": 0.5, "atr_sl_mult": 1.8, "atr_tp_mult": 3.5},
    "ADAUSDT": {"min_rr": 1.5, "risk_percent": 0.8, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0},
    "ENAUSDT": {"min_rr": 1.8, "risk_percent": 0.5, "atr_sl_mult": 1.8, "atr_tp_mult": 3.5},
    "TRUMPUSDT": {"min_rr": 2.0, "risk_percent": 0.5, "atr_sl_mult": 2.0, "atr_tp_mult": 4.0},
    "TONUSDT": {"min_rr": 1.8, "risk_percent": 0.5, "atr_sl_mult": 1.8, "atr_tp_mult": 3.5}
}
VOLUME_THRESHOLD = 1.2          # Volume ratio for confidence boost
VOLATILITY_THRESHOLD = 2.0      # High volatility threshold

# Telegram configuration
#TELEGRAM_CHAT_ID = "-4706073326"
TELEGRAM_BOT_TOKEN = "8226246719:AAHXDggFiFYpsgcq1vwTAWv7Gsz1URP4KEU"
TELEGRAM_CHAT_ID = "-4985792721"

# Global state
error_counts: Dict[str, int] = {s: 0 for s in SYMBOLS}

def get_signal_performance_stats(symbol: str = None, days: int = 7) -> dict:
    """Get signal performance statistics."""
    # Since MongoDB is removed, return default values
    return {
        "total_signals": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "avg_rr": 0.0,
        "long_signals": 0,
        "long_wins": 0,
        "long_win_rate": 0.0,
        "long_avg_rr": 0.0,
        "short_signals": 0,
        "short_wins": 0,
        "short_win_rate": 0.0,
        "short_avg_rr": 0.0,
        "symbol": symbol or "ALL",
        "period_days": days
    }

def get_save_paths(symbol: str) -> dict:
    """Get file paths for saving charts."""
    return {
        "price_3m": f"{symbol.lower()}_3m_price.png",
        "price_15m": f"{symbol.lower()}_15m_price.png"
    }

# ================= Indicators =================
def ema(series: pd.Series, length: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    d = series.diff()
    up = d.clip(lower=0.0)
    down = -d.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/length, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    e_fast = ema(series, fast)
    e_slow = ema(series, slow)
    macd_line = e_fast - e_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calculate_atr_old(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range for volatility measurement."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, prev_close = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    # Wilder's smoothing
    return tr.ewm(alpha=1/period, adjust=False).mean()


def stochastic_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic RSI indicator.
    
    Returns:
        Tuple of (%K, %D) lines
    """
    # Calculate RSI first
    rsi_values = rsi(series, rsi_period)
    
    # Calculate Stochastic of RSI
    lowest_rsi = rsi_values.rolling(window=stoch_period).min()
    highest_rsi = rsi_values.rolling(window=stoch_period).max()
    
    stoch_rsi = 100 * (rsi_values - lowest_rsi) / (highest_rsi - lowest_rsi)
    stoch_rsi = stoch_rsi.fillna(50)
    
    # Smooth the Stochastic RSI
    k_percent = stoch_rsi.rolling(window=k_period).mean()
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent.fillna(50), d_percent.fillna(50)

def calculate_volume_profile(df: pd.DataFrame, price_range_pct: float = 0.5, num_bins: int = 10) -> Dict:
    """Calculate volume profile to identify volume clusters and Point of Control (POC).
    
    Args:
        df: DataFrame with price and volume data
        price_range_pct: Percentage of price range around current price to analyze
        num_bins: Number of price bins to divide the range into
        
    Returns:
        Dictionary with volume profile analysis
    """
    if len(df) < 20:  # Need enough data for meaningful volume profile
        return {"success": False, "error": "Not enough data points"}
    
    try:
        current_price = df.iloc[-1]['close']
        price_range = current_price * price_range_pct / 100
        
        # Define price range around current price
        min_price = current_price - price_range
        max_price = current_price + price_range
        
        # Filter data within the range
        mask = (df['high'] >= min_price) & (df['low'] <= max_price)
        df_subset = df[mask].copy()
        
        if len(df_subset) < 5:
            return {"success": False, "error": "Not enough data points in price range"}
        
        # Create price bins
        price_bins = np.linspace(min_price, max_price, num_bins + 1)
        bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
        bin_width = price_bins[1] - price_bins[0]
        
        # Initialize volume by price level
        volume_by_price = np.zeros(num_bins)
        
        # Calculate volume contribution to each price bin
        for idx, row in df_subset.iterrows():
            # Determine price range for this candle
            #candle_min = min(row['open'], row['close'])
            #candle_max = max(row['open'], row['close'])
            candle_min = row['low']
            candle_max = row['high']
            
            candle_volume = row['volume']
            
            # Allocate volume to bins that this candle spans
            for i in range(num_bins):
                bin_low = price_bins[i]
                bin_high = price_bins[i+1]
                
                # Check if candle overlaps with this bin
                if candle_max >= bin_low and candle_min <= bin_high:
                    # Calculate overlap
                    overlap_low = max(candle_min, bin_low)
                    overlap_high = min(candle_max, bin_high)
                    overlap_pct = (overlap_high - overlap_low) / (candle_max - candle_min) if candle_max > candle_min else 1.0
                    
                    # Allocate volume based on overlap percentage
                    volume_by_price[i] += candle_volume * overlap_pct
        
        # Find Point of Control (price level with highest volume)
        poc_idx = np.argmax(volume_by_price)
        poc_price = bin_centers[poc_idx]
        poc_volume = volume_by_price[poc_idx]
        
        # Find Value Area (70% of total volume)
        total_volume = np.sum(volume_by_price)
        target_volume = total_volume * 0.7
        
        # Sort bins by volume in descending order
        sorted_idx = np.argsort(volume_by_price)[::-1]
        
        # Take bins until we reach target volume
        cumulative_volume = 0
        value_area_bins = []
        for i in sorted_idx:
            value_area_bins.append(i)
            cumulative_volume += volume_by_price[i]
            if cumulative_volume >= target_volume:
                break
        
        # Find value area boundaries
        va_bins = sorted(value_area_bins)
        if len(va_bins) > 0:
            va_high_idx = va_bins[-1]
            va_low_idx = va_bins[0]
            va_high = bin_centers[va_high_idx] + bin_width/2
            va_low = bin_centers[va_low_idx] - bin_width/2
        else:
            va_high = poc_price + bin_width/2
            va_low = poc_price - bin_width/2
        
        # Identify high volume nodes (local maxima)
        high_vol_nodes = []
        for i in range(1, num_bins-1):
            if (volume_by_price[i] > volume_by_price[i-1] and 
                volume_by_price[i] > volume_by_price[i+1] and
                volume_by_price[i] > total_volume / num_bins):
                high_vol_nodes.append({
                    "price": bin_centers[i],
                    "volume": volume_by_price[i],
                    "volume_pct": volume_by_price[i] / total_volume * 100
                })
        
        # Identify low volume nodes (local minima)
        low_vol_nodes = []
        for i in range(1, num_bins-1):
            if (volume_by_price[i] < volume_by_price[i-1] and 
                volume_by_price[i] < volume_by_price[i+1] and
                volume_by_price[i] < total_volume / num_bins / 2):  # Less than half average
                low_vol_nodes.append({
                    "price": bin_centers[i],
                    "volume": volume_by_price[i],
                    "volume_pct": volume_by_price[i] / total_volume * 100
                })
        
        return {
            "success": True,
            "current_price": current_price,
            "point_of_control": poc_price,
            "poc_volume_pct": poc_volume / total_volume * 100,
            "value_area_high": va_high,
            "value_area_low": va_low,
            "high_volume_nodes": high_vol_nodes,
            "low_volume_nodes": low_vol_nodes,
            "bins": list(bin_centers),
            "volumes": list(volume_by_price),
            "volume_distribution": [{"price": p, "volume": v} 
                                   for p, v in zip(bin_centers, volume_by_price)],
            "price_in_value_area": va_low <= current_price <= va_high,
            "price_near_poc": abs(current_price - poc_price) / current_price < 0.005  # Within 0.5%
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def only_closed(df: pd.DataFrame) -> pd.DataFrame:
    now = datetime.now(TZ)
    return df[df["close_time"] <= now].copy()


def analyze_market_structure(df: pd.DataFrame) -> Dict:
    """Comprehensive market structure analysis.
    
    Returns:
        Dictionary with market structure quality assessment
    """
    try:
        if len(df) < 20:
            return {"success": False, "error": "Not enough data for structure analysis"}
        
        # Get swing points data
        h_idx = list(df.index[df["swing_high"]])
        l_idx = list(df.index[df["swing_low"]])
        
        structure_quality = 0
        structure_notes = []
        
        # 1. Trend Structure Assessment
        if len(h_idx) >= 2 and len(l_idx) >= 2:
            recent_highs = [df.loc[i, "high"] for i in h_idx[-2:]]
            recent_lows = [df.loc[i, "low"] for i in l_idx[-2:]]
            
            # Check for higher highs and higher lows (uptrend structure)
            if recent_highs[1] > recent_highs[0] and recent_lows[1] > recent_lows[0]:
                structure_quality += 2
                structure_notes.append("HH + HL (bullish structure)")
            # Check for lower highs and lower lows (downtrend structure)
            elif recent_highs[1] < recent_highs[0] and recent_lows[1] < recent_lows[0]:
                structure_quality += 2
                structure_notes.append("LH + LL (bearish structure)")
            # Mixed structure (consolidation)
            else:
                structure_quality += 1
                structure_notes.append("Mixed structure (consolidation)")
        
        # 2. Support/Resistance Quality
        current_price = df.iloc[-1]['close']
        
        # Find nearby key levels (swing highs/lows within 2% of current price)
        nearby_levels = []
        price_tolerance = current_price * 0.02  # 2% tolerance
        
        for idx in h_idx[-5:]:  # Last 5 swing highs
            level = df.loc[idx, "high"]
            if abs(current_price - level) <= price_tolerance:
                nearby_levels.append({"type": "resistance", "price": level, "strength": 1})
        
        for idx in l_idx[-5:]:  # Last 5 swing lows
            level = df.loc[idx, "low"]
            if abs(current_price - level) <= price_tolerance:
                nearby_levels.append({"type": "support", "price": level, "strength": 1})
        
        # 3. Level Confluence (multiple touches increase strength)
        confluence_levels = []
        for level in nearby_levels:
            similar_levels = [l for l in nearby_levels 
                            if abs(l["price"] - level["price"]) <= current_price * 0.005]  # 0.5% tolerance
            if len(similar_levels) > 1:
                avg_price = sum([l["price"] for l in similar_levels]) / len(similar_levels)
                confluence_levels.append({
                    "type": level["type"],
                    "price": avg_price,
                    "strength": len(similar_levels),
                    "confluence": True
                })
        
        # Remove duplicates and keep strongest levels
        unique_levels = []
        for level in confluence_levels:
            exists = False
            for existing in unique_levels:
                if abs(existing["price"] - level["price"]) <= current_price * 0.005:
                    if level["strength"] > existing["strength"]:
                        unique_levels.remove(existing)
                        unique_levels.append(level)
                    exists = True
                    break
            if not exists:
                unique_levels.append(level)
        
        # Bonus points for confluence
        strong_levels = [l for l in unique_levels if l["strength"] >= 2]
        if strong_levels:
            structure_quality += len(strong_levels)
            structure_notes.append(f"Found {len(strong_levels)} confluence level(s)")
        
        # 4. Trend Strength Assessment
        # Calculate recent price momentum
        recent_closes = df['close'].tail(10)
        price_momentum = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]
        
        if abs(price_momentum) > 0.02:  # Strong momentum (>2%)
            structure_quality += 1
            direction = "up" if price_momentum > 0 else "down"
            structure_notes.append(f"Strong {direction} momentum ({price_momentum:.2%})")
        
        # 5. Volume Structure Quality
        recent_volumes = df['volume'].tail(10)
        volume_trend = recent_volumes.corr(pd.Series(range(len(recent_volumes))))
        
        if abs(volume_trend) > 0.3:  # Strong volume trend correlation
            structure_quality += 1
            trend_type = "increasing" if volume_trend > 0 else "decreasing"
            structure_notes.append(f"Volume {trend_type} with price")
        
        # 6. Breakout Potential
        # Check if price is near consolidation boundaries
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        price_range = recent_high - recent_low
        
        # Price near boundaries indicates potential breakout
        near_high = (current_price - recent_low) / price_range > 0.8
        near_low = (current_price - recent_low) / price_range < 0.2
        
        if near_high or near_low:
            structure_quality += 1
            boundary = "resistance" if near_high else "support"
            structure_notes.append(f"Near key {boundary} level")
        
        # Overall structure assessment
        if structure_quality >= 5:
            structure_rating = "EXCELLENT"
            confidence_multiplier = 1.3
        elif structure_quality >= 3:
            structure_rating = "GOOD"
            confidence_multiplier = 1.1
        elif structure_quality >= 1:
            structure_rating = "FAIR"
            confidence_multiplier = 1.0
        else:
            structure_rating = "POOR"
            confidence_multiplier = 0.8
        
        return {
            "success": True,
            "structure_quality": structure_quality,
            "structure_rating": structure_rating,
            "confidence_multiplier": confidence_multiplier,
            "structure_notes": structure_notes,
            "key_levels": unique_levels,
            "price_momentum": price_momentum,
            "volume_trend_correlation": volume_trend,
            "near_boundary": near_high or near_low,
            "breakout_direction": "up" if near_high else "down" if near_low else "none"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def calculate_signal_confluence(signals: List[Dict], current_price: float) -> Dict:
    """Calculate confluence score based on multiple signals occurring near current price and time.
    
    Args:
        signals: List of detected signals
        current_price: Current market price
        
    Returns:
        Dictionary with confluence analysis
    """
    if not signals:
        return {"confluence_score": 0, "bullish_signals": 0, "bearish_signals": 0, "net_bias": "neutral"}
    
    # Group signals by recency (last 10 periods) and proximity to current price
    recent_signals = signals[-10:] if len(signals) >= 10 else signals
    price_tolerance = current_price * 0.01  # 1% price tolerance
    
    bullish_signals = []
    bearish_signals = []
    
    for sig in recent_signals:
        sig_price = sig.get("price", sig.get("level", current_price))
        
        # Check if signal is near current price
        if abs(sig_price - current_price) <= price_tolerance:
            if sig["side"] == "bullish":
                bullish_signals.append(sig)
            else:
                bearish_signals.append(sig)
    
    # Calculate confluence score
    bull_count = len(bullish_signals)
    bear_count = len(bearish_signals)
    
    # Weight different signal types
    signal_weights = {
        "BOS": 3,
        "RSI Divergence": 2,
        "BB Breakout": 2,
        "Stoch RSI": 1.5,
        "EMA Cross": 2,
        "MACD Cross": 1.5,
        "Engulfing": 1,
        "BB Bounce": 1,
        "Pin Bar": 1.8
    }
    
    weighted_bull_score = sum([signal_weights.get(sig["type"], 1) for sig in bullish_signals])
    weighted_bear_score = sum([signal_weights.get(sig["type"], 1) for sig in bearish_signals])
    
    confluence_score = max(weighted_bull_score, weighted_bear_score)
    net_bias = "bullish" if weighted_bull_score > weighted_bear_score else "bearish" if weighted_bear_score > weighted_bull_score else "neutral"
    
    return {
        "confluence_score": confluence_score,
        "bullish_signals": bull_count,
        "bearish_signals": bear_count,
        "weighted_bull_score": weighted_bull_score,
        "weighted_bear_score": weighted_bear_score,
        "net_bias": net_bias,
        "signal_types": [sig["type"] for sig in recent_signals]
    }

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle_band = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band

def bb_squeeze_detector(df: pd.DataFrame, bb_period: int = 20, kc_period: int = 20, kc_mult: float = 1.5) -> pd.Series:
    """Detect Bollinger Band squeeze (BB inside Keltner Channel).
    
    Returns:
        Boolean series indicating squeeze periods
    """
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = bollinger_bands(df['close'], bb_period)
    
    # Keltner Channel (using ATR)
    kc_middle = df['close'].rolling(window=kc_period).mean()
    atr = calculate_atr(df, kc_period)
    kc_upper = kc_middle + (atr * kc_mult)
    kc_lower = kc_middle - (atr * kc_mult)
    
    # Squeeze occurs when BB is inside KC
    squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
    
    return squeeze.fillna(False)

# ================= Swings =================
def swing_points(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Identify swing high and low points."""
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    k = window
    sh = np.full(n, False)
    sl = np.full(n, False)
    
    for i in range(k, n-k):
        if highs[i] == np.max(highs[i-k:i+k+1]):
            sh[i] = True
        if lows[i] == np.min(lows[i-k:i+k+1]):
            sl[i] = True
    
    out = df.copy()
    out["swing_high"] = sh
    out["swing_low"] = sl
    return out

def last_two_swing_idx(df: pd.DataFrame):
    """Get indices of last two swing highs and lows."""
    h_idx = list(df.index[df["swing_high"]])
    l_idx = list(df.index[df["swing_low"]])
    return h_idx[-2:], l_idx[-2:]

# ================= Fetch Data =================
def get_klines(symbol: str, interval=INTERVAL, limit=LIMIT) -> pd.DataFrame:
    """Fetch candlestick data from Binance API."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            url = f"{BASE}/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            r = requests.get(url, params=params, timeout=API_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            
            cols = ["open_time","open","high","low","close","volume",
                   "close_time","qav","trades","taker_base","taker_quote","ignore"]
            df = pd.DataFrame(data, columns=cols)
            
            for c in ["open","high","low","close","volume"]:
                df[c] = df[c].astype(float)
                
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(TZ)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.tz_convert(TZ)
            
            return df[["open_time","open","high","low","close","volume","close_time"]]
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching klines for {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise

def get_klines_15m(symbol: str, limit=120) -> pd.DataFrame:
    """Get 15m timeframe data."""
    return get_klines(symbol=symbol, interval="15m", limit=limit)

# ================= Trend Score =================
def score_trend(df: pd.DataFrame) -> dict:
    """Score the current trend based on multiple indicators."""
    last = df.iloc[-1]
    score = 0
    reasons = []

    # EMA alignment & price location (EMA50 vs EMA100)
    if last["ema50"] > last["ema100"]:
        score += 1
        reasons.append("EMA50 > EMA100")
    elif last["ema50"] < last["ema100"]:
        score -= 1
        reasons.append("EMA50 < EMA100")

    if last["close"] > last["ema50"]:
        score += 1
        reasons.append("Close > EMA50")
    elif last["close"] < last["ema50"]:
        score -= 1
        reasons.append("Close < EMA50")

    # RSI regime
    if last["rsi"] > 55:
        score += 1
        reasons.append("RSI > 55")
    elif last["rsi"] < 45:
        score -= 1
        reasons.append("RSI < 45")

    # MACD momentum
    if last["macd_hist"] > 0 and last["macd_line"] > last["macd_signal"]:
        score += 1
        reasons.append("MACD up")
    elif last["macd_hist"] < 0 and last["macd_line"] < last["macd_signal"]:
        score -= 1
        reasons.append("MACD down")

    # Volume vs VolMA20
    if last["volume"] > last["vol_ma20"]:
        score += (1 if last["close"] >= df.iloc[-2]["close"] else -1)

    # Trend classification
    if score >= 2 and last["close"] > last["ema50"] > last["ema100"]:
        label = "UPTREND"
    elif score <= -2 and last["close"] < last["ema50"] < last["ema100"]:
        label = "DOWNTREND"
    else:
        label = "SIDEWAYS / MIXED"

    return {
        "score": score, "label": label, "last_time": last["open_time"],
        "last_close": last["close"], "ema50": last["ema50"], "ema100": last["ema100"],
        "rsi": last["rsi"], "macd_line": last["macd_line"],
        "macd_signal": last["macd_signal"], "macd_hist": last["macd_hist"],
        "volume": last["volume"], "vol_ma20": last["vol_ma20"], "reasons": reasons
    }

# ================= Reversal Signals =================
def detect_bos(df: pd.DataFrame) -> List[Dict]:
    """Detect Break of Structure signals."""
    sig = []
    h_idx, l_idx = last_two_swing_idx(df)
    if len(h_idx)==2 and len(l_idx)==2:
        h1,h2 = h_idx[-1], h_idx[-2]
        l1,l2 = l_idx[-1], l_idx[-2]
        hh_hl = (df.loc[h1,"high"]>df.loc[h2,"high"]) and (df.loc[l1,"low"]>df.loc[l2,"low"])
        lh_ll = (df.loc[h1,"high"]<df.loc[h2,"high"]) and (df.loc[l1,"low"]<df.loc[l2,"low"])
        last_close = df.iloc[-1]["close"]
        if hh_hl and last_close < df.loc[l1,"low"]:
            sig.append({"type":"BOS","side":"bearish","at":df.iloc[-1]["open_time"],
                        "level":float(df.loc[l1,"low"]), "idx":int(l1),
                        "note":f"Break HL {df.loc[l1,'low']:.2f}"})
        if lh_ll and last_close > df.loc[h1,"high"]:
            sig.append({"type":"BOS","side":"bullish","at":df.iloc[-1]["open_time"],
                        "level":float(df.loc[h1,"high"]), "idx":int(h1),
                        "note":f"Break LH {df.loc[h1,'high']:.2f}"})
    return sig

def detect_rsi_divergence(df: pd.DataFrame) -> List[Dict]:
    """Detect RSI divergence patterns."""
    sig = []
    h_idx, l_idx = last_two_swing_idx(df)
    # Bearish: Price HH, RSI LH
    if len(h_idx)==2:
        h2,h1 = h_idx[-2],h_idx[-1]
        if df.loc[h1,"high"]>df.loc[h2,"high"] and df.loc[h1,"rsi"]<df.loc[h2,"rsi"]:
            sig.append({"type":"RSI Divergence","side":"bearish","at":df.loc[h1,"open_time"],
                        "price_pts":[(df.loc[h2,"open_time"],float(df.loc[h2,"high"])),
                                     (df.loc[h1,"open_time"],float(df.loc[h1,"high"]))],
                        "rsi_pts":[(df.loc[h2,"open_time"],float(df.loc[h2,"rsi"])),
                                   (df.loc[h1,"open_time"],float(df.loc[h1,"rsi"]))],
                        "note":"Price HH vs RSI LH"})
    # Bullish: Price LL, RSI HL
    if len(l_idx)==2:
        l2,l1 = l_idx[-2],l_idx[-1]
        if df.loc[l1,"low"]<df.loc[l2,"low"] and df.loc[l1,"rsi"]>df.loc[l2,"rsi"]:
            sig.append({"type":"RSI Divergence","side":"bullish","at":df.loc[l1,"open_time"],
                        "price_pts":[(df.loc[l2,"open_time"],float(df.loc[l2,"low"])),
                                     (df.loc[l1,"open_time"],float(df.loc[l1,"low"]))],
                        "rsi_pts":[(df.loc[l2,"open_time"],float(df.loc[l2,"rsi"])),
                                   (df.loc[l1,"open_time"],float(df.loc[l1,"rsi"]))],
                        "note":"Price LL vs RSI HL"})
    return sig

def detect_engulfing(df: pd.DataFrame, lookback: int = 10) -> List[Dict]:
    """Detect engulfing candlestick patterns."""
    sig = []
    sub = df.tail(lookback+1).copy()
    for i in range(1,len(sub)):
        o1,c1 = sub.iloc[i-1]["open"], sub.iloc[i-1]["close"]
        o2,c2 = sub.iloc[i]["open"],  sub.iloc[i]["close"]
        # bullish engulfing near swing low
        is_bull = (c2>o2) and not (c1>o1) and (max(o2,c2)>=max(o1,c1)) and (min(o2,c2)<=min(o1,c1))
        near_sl = bool(sub.iloc[i]["swing_low"]) or bool(sub.iloc[i-1]["swing_low"])
        if is_bull and near_sl:
            sig.append({"type":"Engulfing","side":"bullish","at":sub.iloc[i]["open_time"],
                        "price":float(c2), "note":"Bullish engulfing @ swing-low"})
        # bearish engulfing near swing high
        is_bear = (c2<o2) and not (c1<o1) and (max(o2,c2)>=max(o1,c1)) and (min(o2,c2)<=min(o1,c1))
        near_sh = bool(sub.iloc[i]["swing_high"]) or bool(sub.iloc[i-1]["swing_high"])
        if is_bear and near_sh:
            sig.append({"type":"Engulfing","side":"bearish","at":sub.iloc[i]["open_time"],
                        "price":float(c2), "note":"Bearish engulfing @ swing-high"})
    return sig

def detect_ema_cross(df: pd.DataFrame, within:int=20)->List[Dict]:
    """Detect EMA crossover signals."""
    sig=[]
    e50=df["ema50"].values
    e100=df["ema100"].values
    n=len(df)
    start=max(1,n-within)
    for i in range(start,n):
        if (e50[i-1]<=e100[i-1]) and (e50[i]>e100[i]):
            sig.append({"type":"EMA Cross","side":"bullish","at":df.iloc[i]["open_time"],
                        "price":float(df.iloc[i]["close"]), "note":"Golden cross (50>100)"})
        if (e50[i-1]>=e100[i-1]) and (e50[i]<e100[i]):
            sig.append({"type":"EMA Cross","side":"bearish","at":df.iloc[i]["open_time"],
                        "price":float(df.iloc[i]["close"]), "note":"Death cross (50<100)"})
    return sig

def detect_macd_cross(df: pd.DataFrame, within:int=20)->List[Dict]:
    """Detect MACD crossover signals."""
    sig=[]
    n=len(df)
    start=max(1,n-within)
    ml=df["macd_line"].values
    sg=df["macd_signal"].values
    for i in range(start,n):
        if (ml[i-1]<=sg[i-1]) and (ml[i]>sg[i]):
            sig.append({"type":"MACD Cross","side":"bullish","at":df.iloc[i]["open_time"],
                        "value":float(ml[i]), "note":"MACD up-cross"})
        if (ml[i-1]>=sg[i-1]) and (ml[i]<sg[i]):
            sig.append({"type":"MACD Cross","side":"bearish","at":df.iloc[i]["open_time"],
                        "value":float(ml[i]), "note":"MACD down-cross"})
    return sig

def detect_stochastic_rsi_signals(df: pd.DataFrame, within: int = 10) -> List[Dict]:
    """Detect Stochastic RSI oversold/overbought reversal signals."""
    sig = []
    
    if len(df) < 30:  # Need enough data for Stochastic RSI calculation
        return sig
    
    k_percent, d_percent = stochastic_rsi(df['close'])
    df_temp = df.copy()
    df_temp['stoch_rsi_k'] = k_percent
    df_temp['stoch_rsi_d'] = d_percent
    
    n = len(df_temp)
    start = max(1, n - within)
    
    for i in range(start, n):
        if i < 1:
            continue
            
        current_k = df_temp.iloc[i]['stoch_rsi_k']
        current_d = df_temp.iloc[i]['stoch_rsi_d']
        prev_k = df_temp.iloc[i-1]['stoch_rsi_k']
        prev_d = df_temp.iloc[i-1]['stoch_rsi_d']
        
        # Skip if NaN values
        if pd.isna(current_k) or pd.isna(current_d) or pd.isna(prev_k) or pd.isna(prev_d):
            continue
        
        # Bullish signal: oversold crossover (K crosses above D from oversold)
        if (prev_k <= prev_d and current_k > current_d and 
            current_k < 30 and current_d < 30):  # Both in oversold territory
            sig.append({
                "type": "Stoch RSI", 
                "side": "bullish", 
                "at": df_temp.iloc[i]["open_time"],
                "price": float(df_temp.iloc[i]["close"]),
                "stoch_k": float(current_k),
                "stoch_d": float(current_d),
                "note": f"Oversold crossover K:{current_k:.1f} D:{current_d:.1f}"
            })
        
        # Bearish signal: overbought crossover (K crosses below D from overbought)
        elif (prev_k >= prev_d and current_k < current_d and 
              current_k > 70 and current_d > 70):  # Both in overbought territory
            sig.append({
                "type": "Stoch RSI", 
                "side": "bearish", 
                "at": df_temp.iloc[i]["open_time"],
                "price": float(df_temp.iloc[i]["close"]),
                "stoch_k": float(current_k),
                "stoch_d": float(current_d),
                "note": f"Overbought crossover K:{current_k:.1f} D:{current_d:.1f}"
            })
    
    return sig

def detect_bollinger_signals(df: pd.DataFrame, within: int = 10) -> List[Dict]:
    """Detect Bollinger Band breakout and mean reversion signals."""
    sig = []
    
    if len(df) < 25:  # Need enough data for BB calculation
        return sig
    
    bb_upper, bb_middle, bb_lower = bollinger_bands(df['close'], period=20)
    squeeze = bb_squeeze_detector(df)
    
    df_temp = df.copy()
    df_temp['bb_upper'] = bb_upper
    df_temp['bb_middle'] = bb_middle
    df_temp['bb_lower'] = bb_lower
    df_temp['bb_squeeze'] = squeeze
    
    n = len(df_temp)
    start = max(1, n - within)
    
    for i in range(start, n):
        if i < 2:
            continue
            
        current = df_temp.iloc[i]
        prev = df_temp.iloc[i-1]
        prev2 = df_temp.iloc[i-2]
        
        # Skip if NaN values
        if (pd.isna(current['bb_upper']) or pd.isna(current['bb_lower']) or 
            pd.isna(prev['bb_upper']) or pd.isna(prev['bb_lower'])):
            continue
        
        # Bollinger Band Squeeze Breakout
        if (prev2['bb_squeeze'] == True and prev['bb_squeeze'] == True and 
            current['bb_squeeze'] == False):
            
            # Determine breakout direction
            if current['close'] > current['bb_upper']:
                sig.append({
                    "type": "BB Breakout", 
                    "side": "bullish", 
                    "at": current["open_time"],
                    "price": float(current["close"]),
                    "bb_level": float(current['bb_upper']),
                    "note": f"Squeeze breakout above upper band {current['bb_upper']:.2f}"
                })
            elif current['close'] < current['bb_lower']:
                sig.append({
                    "type": "BB Breakout", 
                    "side": "bearish", 
                    "at": current["open_time"],
                    "price": float(current["close"]),
                    "bb_level": float(current['bb_lower']),
                    "note": f"Squeeze breakout below lower band {current['bb_lower']:.2f}"
                })
        
        # Mean reversion signals
        # Bullish: Price touches lower band and bounces back
        elif (prev['close'] <= prev['bb_lower'] and current['close'] > prev['bb_lower'] and
              current['close'] > prev['close']):  # Bounce from lower band
            sig.append({
                "type": "BB Bounce", 
                "side": "bullish", 
                "at": current["open_time"],
                "price": float(current["close"]),
                "bb_level": float(prev['bb_lower']),
                "note": f"Bounce from lower band {prev['bb_lower']:.2f}"
            })
        
        # Bearish: Price touches upper band and reverses
        elif (prev['close'] >= prev['bb_upper'] and current['close'] < prev['bb_upper'] and
              current['close'] < prev['close']):  # Rejection from upper band
            sig.append({
                "type": "BB Bounce", 
                "side": "bearish", 
                "at": current["open_time"],
                "price": float(current["close"]),
                "bb_level": float(prev['bb_upper']),
                "note": f"Rejection from upper band {prev['bb_upper']:.2f}"
            })
    
    return sig

def collect_reversal_signals(df: pd.DataFrame) -> List[Dict]:
    """Collect all reversal signals including new advanced indicators."""
    sig = []
    
    # Original signals
    sig += detect_bos(df)
    sig += detect_rsi_divergence(df)
    sig += detect_engulfing(df, lookback=min(10, len(df)-2))
    sig += detect_ema_cross(df, within=min(20, len(df)-1))
    sig += detect_macd_cross(df, within=min(20, len(df)-1))
    
    # New advanced signals
    sig += detect_stochastic_rsi_signals(df, within=min(10, len(df)-1))
    sig += detect_bollinger_signals(df, within=min(10, len(df)-1))

    # Pin Bar detection
    sig += detect_pin_bar(df, lookback=min(12, len(df)-1))
    
    return sorted(sig, key=lambda s: s["at"])

# ================= Plotting =================
def plot_price(df: pd.DataFrame, signals: List[Dict], save_path: str, interval: str = "3m", symbol: str = "BTCUSDT"):
    """Plot price chart with indicators and signals."""
    plt.figure()
    plt.plot(df["open_time"], df["close"], linewidth=1.0, label="Close")
    plt.plot(df["open_time"], df["ema50"], linewidth=1.0, label="EMA50")
    plt.plot(df["open_time"], df["ema100"], linewidth=1.0, label="EMA100")

    # Plot signals
    for s in signals:
        t = s["at"]
        if s["type"] in ("EMA Cross","Engulfing","Pin Bar"):
            close_match = df.loc[df["open_time"]<=t, "close"]
            y = s.get("price", float(close_match.iloc[-1]) if len(close_match) > 0 else df.iloc[-1]["close"])
            marker = "^" if s["side"]=="bullish" else "v"
            plt.scatter([t],[y], marker=marker, s=60)
            plt.annotate(s["type"], (t,y), xytext=(6,10), textcoords="offset points", fontsize=8)
        elif s["type"]=="BOS":
            level = s.get("level")
            plt.axhline(level, linestyle="--", linewidth=0.8)
            marker = "v" if s["side"]=="bearish" else "^"
            y = level
            plt.scatter([t],[y], marker=marker, s=60)
            plt.annotate(f"BOS {s['side']}", (t,y), xytext=(6,10), textcoords="offset points", fontsize=8)
        elif s["type"]=="RSI Divergence":
            pts = s.get("price_pts")
            if pts and len(pts)==2:
                xs=[pts[0][0], pts[1][0]]
                ys=[pts[0][1], pts[1][1]]
                plt.plot(xs, ys, linewidth=1.0)
                m = "v" if s["side"]=="bearish" else "^"
                plt.scatter([xs[-1]],[ys[-1]], marker=m, s=60)
                plt.annotate("Div", (xs[-1],ys[-1]), xytext=(6,10), textcoords="offset points", fontsize=8)

    plt.title(f"{symbol} – {interval} Price with EMA50/EMA100 (+Signals)")
    plt.xlabel("Time (Asia/Ho_Chi_Minh)")
    plt.ylabel("Price")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=TZ))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ================= Messaging =================
def send_telegram_message(message: str, max_retries: int = 3) -> bool:
    """Send message to Telegram."""
    if not TELEGRAM_BOT_TOKEN or "YOUR_TELEGRAM_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_notification": True
    }
    
    for attempt in range(max_retries):
        try:
            r = requests.post(url, data=payload, timeout=TELEGRAM_TIMEOUT)
            r.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            print(f"Telegram send error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
    
    return False

def send_telegram_photo(photo_path: str, caption: str = "", max_retries: int = 3) -> bool:
    """Send photo to Telegram."""
    if not TELEGRAM_BOT_TOKEN or "YOUR_TELEGRAM_BOT_TOKEN" in TELEGRAM_BOT_TOKEN:
        return False
        
    if not os.path.exists(photo_path):
        print(f"Photo file not found: {photo_path}")
        return False
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    
    for attempt in range(max_retries):
        try:
            with open(photo_path, "rb") as photo:
                files = {"photo": photo}
                data = {
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption,
                    "parse_mode": "HTML",
                    "disable_notification": True
                }
                r = requests.post(url, data=data, files=files, timeout=TELEGRAM_TIMEOUT)
                r.raise_for_status()
                return True
        except (requests.exceptions.RequestException, IOError) as e:
            print(f"Telegram photo send error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(RETRY_DELAY)
    
    return False

# ================= SL/TP & Helpers =================
def compute_sl_tp(dfp: pd.DataFrame, side: str, symbol: str = "BTCUSDT") -> dict:
    """Enhanced SL/TP computation with ATR buffer and multiple TP levels."""
    try:
        # Get symbol-specific configuration
        config = SYMBOL_SPECIFIC_RR.get(symbol, {
            "min_rr": 1.5, "atr_sl_mult": 1.5, "atr_tp_mult": 3.0
        })
        
        h_idx, l_idx = last_two_swing_idx(dfp)
        entry = float(dfp.iloc[-1]["close"])
        
        # Calculate ATR for dynamic SL/TP
        atr_series = calculate_atr(dfp)
        current_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else (entry * 0.02)
        
        sl = tp1 = tp2 = tp3 = None
        strategy_used = "Enhanced_Swing_ATR"
        confidence = 0.7
        
        if side == "bullish":
            # Enhanced SL calculation with ATR buffer
            if l_idx:
                swing_low = float(dfp.loc[l_idx[-1], "low"])
                atr_buffer = current_atr * 0.5  # ATR buffer
                sl = swing_low - atr_buffer
            else:
                # Fallback to ATR-based SL
                sl = entry - (current_atr * config["atr_sl_mult"])
                strategy_used = "ATR_Based"
                confidence = 0.6
            
            # Enhanced TP calculation - multiple levels
            if h_idx:
                swing_high = float(dfp.loc[h_idx[-1], "high"])
                tp1 = swing_high  # Primary target at swing high
                
                # Calculate swing range for extensions
                if l_idx:
                    swing_range = swing_high - float(dfp.loc[l_idx[-1], "low"])
                    tp2 = swing_high + (swing_range * 0.618)  # Fibonacci extension
                    tp3 = swing_high + (swing_range * 1.0)    # 100% extension
                else:
                    tp2 = entry + (current_atr * config["atr_tp_mult"])
                    tp3 = entry + (current_atr * config["atr_tp_mult"] * 1.5)
            else:
                # Fallback to ATR-based TP
                tp1 = entry + (current_atr * config["atr_tp_mult"])
                tp2 = entry + (current_atr * config["atr_tp_mult"] * 1.2)
                tp3 = entry + (current_atr * config["atr_tp_mult"] * 1.5)
                strategy_used = "ATR_Based"
        
        else:  # bearish
            # Enhanced SL calculation with ATR buffer
            if h_idx:
                swing_high = float(dfp.loc[h_idx[-1], "high"])
                atr_buffer = current_atr * 0.5
                sl = swing_high + atr_buffer
            else:
                sl = entry + (current_atr * config["atr_sl_mult"])
                strategy_used = "ATR_Based"
                confidence = 0.6
            
            # Enhanced TP calculation - multiple levels
            if l_idx:
                swing_low = float(dfp.loc[l_idx[-1], "low"])
                tp1 = swing_low
                
                if h_idx:
                    swing_range = float(dfp.loc[h_idx[-1], "high"]) - swing_low
                    tp2 = swing_low - (swing_range * 0.618)
                    tp3 = swing_low - (swing_range * 1.0)
                else:
                    tp2 = entry - (current_atr * config["atr_tp_mult"])
                    tp3 = entry - (current_atr * config["atr_tp_mult"] * 1.5)
            else:
                tp1 = entry - (current_atr * config["atr_tp_mult"])
                tp2 = entry - (current_atr * config["atr_tp_mult"] * 1.2)
                tp3 = entry - (current_atr * config["atr_tp_mult"] * 1.5)
                strategy_used = "ATR_Based"
        
        # Calculate R:R ratio
        rr = None
        if sl is not None and tp1 is not None:
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)
            rr = reward / risk if risk > 0 else None
        
        # Volume and volatility analysis for confidence adjustment
        volume_ratio = dfp.iloc[-1]["volume"] / dfp["volume"].tail(20).mean()
        if volume_ratio > VOLUME_THRESHOLD:
            confidence *= 1.1  # Boost confidence with high volume
        elif volume_ratio < 0.8:
            confidence *= 0.9  # Reduce confidence with low volume
        
        # Volatility adjustment
        avg_atr = atr_series.tail(20).mean()
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        if volatility_ratio > VOLATILITY_THRESHOLD:
            confidence *= 0.8  # Reduce confidence in high volatility
        
        return {
            "entry": entry,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "rr": rr,
            "strategy": strategy_used,
            "confidence": confidence,
            "atr": current_atr,
            "volume_ratio": volume_ratio,
            "volatility_ratio": volatility_ratio
        }
        
    except Exception as e:
        print(f"Error in enhanced SL/TP calculation: {e}")
        # Fallback to simple calculation
        h_idx, l_idx = last_two_swing_idx(dfp)
        entry = float(dfp.iloc[-1]["close"])
        
        sl = tp1 = None
        if side == "bullish":
            if l_idx: sl = float(dfp.loc[l_idx[-1], "low"])
            if h_idx: tp1 = float(dfp.loc[h_idx[-1], "high"])
        else:
            if h_idx: sl = float(dfp.loc[h_idx[-1], "high"])
            if l_idx: tp1 = float(dfp.loc[l_idx[-1], "low"])
        
        rr = None
        if sl is not None and tp1 is not None:
            if side == "bullish" and entry > sl:
                rr = (tp1 - entry) / (entry - sl)
            elif side == "bearish" and sl > entry:
                rr = (entry - tp1) / (sl - entry)
        
        return {
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": None, "tp3": None,
            "rr": rr, "strategy": "Fallback_Simple", "confidence": 0.5,
            "atr": None, "volume_ratio": 1.0, "volatility_ratio": 1.0
        }

def aligned_with_15m(side: str, label15: str) -> Tuple[bool, str]:
    """Determine alignment with 15m frame; return with descriptive tag."""
    if label15 == "UPTREND" and side == "bullish":
        return True, "WITH 15m trend"
    if label15 == "DOWNTREND" and side == "bearish":
        return True, "WITH 15m trend"
    if label15 == "SIDEWAYS / MIXED":
        return True, "15m SIDEWAYS"
    return False, "COUNTER-TREND 15m"

def fmt(n: Optional[float], digits=2) -> str:
    """Format number with specified digits."""
    return "—" if n is None else f"{n:.{digits}f}"

def generate_trading_recommendation(symbol: str, result3: dict, result15: dict, 
                                   market_structure: dict, volume_profile: dict, 
                                   confluence_analysis: dict) -> Dict:
    """Generate Long/Short recommendation based on comprehensive analysis.
    
    Returns:
        Dictionary with trading recommendation and confidence
    """
    recommendation = {
        "symbol": symbol,
        "action": "HOLD",  # Default to HOLD
        "confidence": 0.0,
        "bias_score": 0,
        "reasons": [],
        "risk_level": "MEDIUM",
        "timeframe_alignment": False,
        "structure_supports": False,
        "volume_supports": False
    }
    
    bias_score = 0
    reasons = []
    
    # 1. Timeframe Alignment (Weight: 3 points)
    if result3['label'] == "UPTREND" and result15['label'] == "UPTREND":
        bias_score += 3
        reasons.append("3m & 15m both UPTREND")
        recommendation["timeframe_alignment"] = True
    elif result3['label'] == "DOWNTREND" and result15['label'] == "DOWNTREND":
        bias_score -= 3
        reasons.append("3m & 15m both DOWNTREND")
        recommendation["timeframe_alignment"] = True
    elif result3['label'] == "UPTREND" and result15['label'] != "DOWNTREND":
        bias_score += 2
        reasons.append("3m UPTREND, 15m non-bearish")
    elif result3['label'] == "DOWNTREND" and result15['label'] != "UPTREND":
        bias_score -= 2
        reasons.append("3m DOWNTREND, 15m non-bullish")
    
    # 2. Market Structure Assessment (Weight: 2 points)
    if market_structure.get("success", False):
        structure_rating = market_structure.get("structure_rating", "FAIR")
        breakout_direction = market_structure.get("breakout_direction", "none")
        
        if structure_rating in ["EXCELLENT", "GOOD"]:
            recommendation["structure_supports"] = True
            if breakout_direction == "up":
                bias_score += 2
                reasons.append(f"Strong structure supports upward breakout")
            elif breakout_direction == "down":
                bias_score -= 2
                reasons.append(f"Strong structure supports downward breakout")
            else:
                # General structure quality boost
                momentum = market_structure.get("price_momentum", 0)
                if momentum > 0.01:
                    bias_score += 1
                    reasons.append("Strong structure + upward momentum")
                elif momentum < -0.01:
                    bias_score -= 1
                    reasons.append("Strong structure + downward momentum")
    
    # 3. Volume Profile Analysis (Weight: 1 point)
    if volume_profile.get("success", False):
        poc_price = volume_profile.get("point_of_control")
        current_price = volume_profile.get("current_price")
        
        if poc_price and current_price:
            if volume_profile.get("price_near_poc", False):
                recommendation["volume_supports"] = True
                # POC acts as support/resistance
                if current_price > poc_price:
                    bias_score += 1
                    reasons.append("Price above POC (volume support)")
                else:
                    bias_score -= 1
                    reasons.append("Price below POC (volume resistance)")
        
        # High volume nodes boost
        hvn_count = len(volume_profile.get("high_volume_nodes", []))
        if hvn_count >= 2:
            bias_score += 0.5
            reasons.append(f"Strong volume clusters ({hvn_count} HVN)")
    
    # 4. Signal Confluence (Weight: 2 points)
    confluence_score = confluence_analysis.get("confluence_score", 0)
    net_bias = confluence_analysis.get("net_bias", "neutral")
    
    if confluence_score >= 3:
        if net_bias == "bullish":
            bias_score += 2
            reasons.append(f"Strong bullish confluence ({confluence_score:.1f})")
        elif net_bias == "bearish":
            bias_score -= 2
            reasons.append(f"Strong bearish confluence ({confluence_score:.1f})")
    elif confluence_score >= 2:
        if net_bias == "bullish":
            bias_score += 1
            reasons.append(f"Moderate bullish confluence ({confluence_score:.1f})")
        elif net_bias == "bearish":
            bias_score -= 1
            reasons.append(f"Moderate bearish confluence ({confluence_score:.1f})")
    
    # 5. Technical Indicators (Weight: 1 point each)
    rsi_value = result3.get('rsi', 50)
    if rsi_value > 60:
        bias_score += 0.5
        reasons.append("RSI > 60 (bullish)")
    elif rsi_value < 40:
        bias_score -= 0.5
        reasons.append("RSI < 40 (bearish)")
    
    macd_hist = result3.get('macd_hist', 0)
    if macd_hist > 0:
        bias_score += 0.5
        reasons.append("MACD histogram positive")
    elif macd_hist < 0:
        bias_score -= 0.5
        reasons.append("MACD histogram negative")
    
    # Generate final recommendation
    recommendation["bias_score"] = bias_score
    recommendation["reasons"] = reasons
    
    if bias_score >= 4:
        recommendation["action"] = "STRONG LONG"
        recommendation["confidence"] = min(95, 70 + abs(bias_score) * 5)
        recommendation["risk_level"] = "LOW"
    elif bias_score >= 2:
        recommendation["action"] = "LONG"
        recommendation["confidence"] = min(85, 60 + abs(bias_score) * 5)
        recommendation["risk_level"] = "MEDIUM"
    elif bias_score <= -4:
        recommendation["action"] = "STRONG SHORT"
        recommendation["confidence"] = min(95, 70 + abs(bias_score) * 5)
        recommendation["risk_level"] = "LOW"
    elif bias_score <= -2:
        recommendation["action"] = "SHORT"
        recommendation["confidence"] = min(85, 60 + abs(bias_score) * 5)
        recommendation["risk_level"] = "MEDIUM"
    else:
        recommendation["action"] = "HOLD"
        recommendation["confidence"] = 50
        recommendation["risk_level"] = "HIGH"
        reasons.append("Mixed signals, wait for clearer direction")
    
    return recommendation

def send_performance_summary_to_telegram(period_days: int = 7):
    """Send performance summary to Telegram."""
    try:
        # Get overall performance
        overall_stats = get_signal_performance_stats(days=period_days)
        
        if overall_stats["total_signals"] == 0:
            return False  # No data to report
        
        # Get individual symbol performance
        symbol_stats = {}
        for symbol in SYMBOLS:
            symbol_stats[symbol] = get_signal_performance_stats(symbol=symbol, days=period_days)
        
        # Build performance message
        msg_parts = [
            f"<b>📈 Performance Report ({period_days} Days)</b>",
            f"━━━━━━━━━━━━━━━━━━━━━━━━",
            f"<b>🎯 Overall Performance:</b>",
            f"Total Signals: {overall_stats['total_signals']}",
            f"Win Rate: <b>{overall_stats['win_rate']:.1f}%</b> ({overall_stats['wins']}W/{overall_stats['losses']}L)",
            f"Average R:R: <b>{overall_stats['avg_rr']:.2f}</b>",
            f"",
            f"<b>📊 Long vs Short Performance:</b>",
            f"🟢 LONG: {overall_stats['long_win_rate']:.1f}% ({overall_stats['long_wins']}W/{overall_stats['long_signals'] - overall_stats['long_wins']}L) | Avg R:R: {overall_stats['long_avg_rr']:.2f}",
            f"🔴 SHORT: {overall_stats['short_win_rate']:.1f}% ({overall_stats['short_wins']}W/{overall_stats['short_signals'] - overall_stats['short_wins']}L) | Avg R:R: {overall_stats['short_avg_rr']:.2f}",
            f"",
            f"<b>💰 Symbol Performance:</b>"
        ]
        
        # Add individual symbol performance
        for symbol in SYMBOLS:
            stats = symbol_stats[symbol]
            if stats["total_signals"] > 0:
                win_rate_emoji = "✅" if stats["win_rate"] >= 60 else "⚠️" if stats["win_rate"] >= 40 else "❌"
                long_short_info = f"L:{stats['long_win_rate']:.0f}%" if stats['long_signals'] > 0 else "L:N/A"
                long_short_info += f" S:{stats['short_win_rate']:.0f}%" if stats['short_signals'] > 0 else " S:N/A"
                msg_parts.append(
                    f"{win_rate_emoji} {symbol}: {stats['win_rate']:.1f}% ({stats['total_signals']} signals) | {long_short_info}"
                )
        
        # Add recommendations based on performance
        msg_parts.extend([
            f"",
            f"<b>💡 Strategy Recommendations:</b>"
        ])
        
        # Analyze which direction performs better
        if overall_stats['long_win_rate'] > overall_stats['short_win_rate'] + 10:
            msg_parts.append("🟢 <b>Focus on LONG signals</b> (higher win rate)")
        elif overall_stats['short_win_rate'] > overall_stats['long_win_rate'] + 10:
            msg_parts.append("🔴 <b>Focus on SHORT signals</b> (higher win rate)")
        else:
            msg_parts.append("⚖️ Balanced approach: Both LONG & SHORT viable")
        
        # Risk management advice based on overall win rate
        if overall_stats['win_rate'] >= 70:
            msg_parts.append("🎯 Excellent performance! Consider increasing position size")
        elif overall_stats['win_rate'] >= 50:
            msg_parts.append("✅ Solid performance! Maintain current strategy")
        elif overall_stats['win_rate'] >= 30:
            msg_parts.append("⚠️ Below average. Consider stricter filters")
        else:
            msg_parts.append("❌ Poor performance. Review strategy parameters")
        
        msg = "\n".join(msg_parts)
        return send_telegram_message(msg)
        
    except Exception as e:
        print(f"Error sending performance summary: {e}")
        return False

def detect_pin_bar(df: pd.DataFrame, lookback: int = 12,
                   tail_ratio: float = 2.0,        # đuôi >= 2x thân
                   body_frac_max: float = 0.30,     # thân <= 30% range
                   dom_tail_frac_min: float = 0.55, # đuôi chính >= 55% range
                   min_atr_frac: float = 0.8,       # range >= 0.8*ATR
                   min_vol_frac: float = 0.9,       # vol >= 0.9*VolMA20
                   clv_gate: float = 0.60           # close location gate
                   ) -> List[Dict]:
    """
    Pin bar chất lượng:
      - Thân nhỏ; 1 đuôi dài vượt trội; đuôi >= tail_ratio*thân
      - Close Location Value (CLV) ủng hộ hướng:
          + Bullish pin: close ở top >= 60% range
          + Bearish pin: close ở bottom <= 40% range
      - Range nến đủ lớn (>= 0.8*ATR)
      - Volume không quá yếu (>= 0.9*VolMA20)
      - Ưu tiên bối cảnh: gần swing và theo EMA regime
    """
    sig = []
    if len(df) < max(30, lookback+1):
        return sig

    atr_series = calculate_atr(df)
    vol_ma20 = df['volume'].rolling(20).mean()

    sub = df.tail(lookback).copy()
    for i in range(len(sub)):
        idx = sub.index[i]
        row = sub.loc[idx]

        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        rng = h - l
        if rng <= 0:
            continue

        body = abs(c - o)
        upper_tail = h - max(o, c)
        lower_tail = min(o, c) - l

        # điều kiện hình thái
        if body > body_frac_max * rng:
            continue

        # close location value: 0 = close tại low, 1 = close tại high
        clv = (c - l) / rng

        bull_shape = (
            lower_tail >= tail_ratio * body and
            lower_tail >= dom_tail_frac_min * rng and
            upper_tail <= (1 - dom_tail_frac_min) * rng and
            clv >= clv_gate
        )
        bear_shape = (
            upper_tail >= tail_ratio * body and
            upper_tail >= dom_tail_frac_min * rng and
            lower_tail <= (1 - dom_tail_frac_min) * rng and
            clv <= (1 - clv_gate)
        )

        # bối cảnh
        atr_ok = True
        atr_val = atr_series.loc[idx]
        if not np.isnan(atr_val):
            atr_ok = (rng >= min_atr_frac * atr_val)

        vol_ok = True
        vm = vol_ma20.loc[idx]
        if not np.isnan(vm):
            vol_ok = (row["volume"] >= min_vol_frac * vm)

        near_sw_low  = bool(row.get("swing_low", False))
        near_sw_high = bool(row.get("swing_high", False))

        ema50 = df.loc[idx, "ema50"] if "ema50" in df.columns else np.nan
        ema100 = df.loc[idx, "ema100"] if "ema100" in df.columns else np.nan

        bull_ema_ok = (not np.isnan(ema100)) and (c >= ema100)   # có thể nâng lên: c>ema50 & ema50>=ema100
        bear_ema_ok = (not np.isnan(ema100)) and (c <= ema100)

        # FINAL
        if bull_shape and atr_ok and vol_ok and bull_ema_ok and near_sw_low:
            sig.append({
                "type": "Pin Bar",
                "side": "bullish",
                "at": row["open_time"],
                "price": float(c),
                "note": "Bullish pin (long lower wick, CLV high, ATR/Vol ok)"
            })
        elif bear_shape and atr_ok and vol_ok and bear_ema_ok and near_sw_high:
            sig.append({
                "type": "Pin Bar",
                "side": "bearish",
                "at": row["open_time"],
                "price": float(c),
                "note": "Bearish pin (long upper wick, CLV low, ATR/Vol ok)"
            })

    return sig


# ================= Main Processing =================
def process_symbol(symbol: str, first_run: bool, last_signal_ids: dict) -> Tuple[bool, str]:
    """Process a single symbol to detect and report trading signals.
    
    Args:
        symbol: The trading pair symbol
        first_run: Whether this is the first iteration
        last_signal_ids: Dict tracking the last signal sent for each symbol
        
    Returns:
        Tuple of (success, message)
    """
    try:
        save_paths = get_save_paths(symbol)
        
        # ===== 3m =====
        df = get_klines(symbol=symbol)
        
        df = only_closed(df)

        df["ema50"] = ema(df["close"], 50)
        df["ema100"] = ema(df["close"], 100)
        df["rsi"] = rsi(df["close"], 14)
        macd_line, macd_signal, macd_hist = macd(df["close"], 12, 26, 9)
        df["macd_line"] = macd_line
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df = swing_points(df, window=3)
        dfp = df.tail(120).copy()

        result3 = score_trend(dfp)
        signals = collect_reversal_signals(dfp)

        #plot_price(dfp, signals, save_paths["price_3m"], interval="3m", symbol=symbol)

        # ===== 15m =====
        df15 = get_klines_15m(symbol=symbol)
        df15 = only_closed(df15)
        df15["ema50"] = ema(df15["close"], 50)
        df15["ema100"] = ema(df15["close"], 100)
        df15["rsi"] = rsi(df15["close"], 14)
        macd_line15, macd_signal15, macd_hist15 = macd(df15["close"], 12, 26, 9)
        df15["macd_line"] = macd_line15
        df15["macd_signal"] = macd_signal15
        df15["macd_hist"] = macd_hist15
        df15["vol_ma20"] = df15["volume"].rolling(20).mean()
        df15 = swing_points(df15, window=3)
        dfp15 = df15.tail(120).copy()
        result15 = score_trend(dfp15)

        # ===== Console log =====
        print(f"\n=== {symbol} Trend Detector ===")
        print(f"Time: {result3['last_time'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"3m Close: {result3['last_close']:.2f} | EMA50: {result3['ema50']:.2f} | EMA100: {result3['ema100']:.2f}")
        print(f"3m Score: {result3['score']}  =>  Trend: {result3['label']}")
        print(f"15m Trend: {result15['label']}")

        # Plot 15m chart
        signals15 = collect_reversal_signals(dfp15)
        #plot_price(dfp15, signals15, save_paths["price_15m"], interval="15m", symbol=symbol)

        # ===== Advanced Analysis =====
        # Volume Profile Analysis
        volume_profile = calculate_volume_profile(dfp, price_range_pct=2.0)
        
        # Market Structure Analysis
        market_structure = analyze_market_structure(dfp)
        
        # Signal Confluence Analysis
        current_price = result3['last_close']
        confluence_analysis = calculate_signal_confluence(signals, current_price)
        
        # Generate Trading Recommendation
        recommendation = generate_trading_recommendation(
            symbol, result3, result15, market_structure, volume_profile, confluence_analysis
        )
        
        # Display recommendation in console
        print(f"💡 {symbol} Trading Recommendation:")
        print(f"   Action: {recommendation['action']} (Confidence: {recommendation['confidence']:.0f}%)")
        print(f"   Risk Level: {recommendation['risk_level']} | Bias Score: {recommendation['bias_score']:.1f}")
        print(f"   Key Factors: {', '.join(recommendation['reasons'][:2])}")  # Show first 2 reasons
        
        if signals:
            latest = signals[-1]
            signal_id = f"{symbol}_{latest['type']}_{latest['side']}_{latest['at']}"
            
            if first_run or signal_id != last_signal_ids.get(symbol):
                side = latest['side']
                rr_data = compute_sl_tp(dfp, side, symbol)
                
                # Get symbol-specific minimum R:R
                symbol_config = SYMBOL_SPECIFIC_RR.get(symbol, {"min_rr": 1.5})
                symbol_min_rr = symbol_config["min_rr"]
                
                ok_15m, tag_15m = aligned_with_15m(side, result15['label'])
                mtf_ok = (not MTF_CONFIRM) or ok_15m
                rr_ok = (rr_data["rr"] is not None) and (rr_data["rr"] >= symbol_min_rr)
                
                # Enhanced quality filters using new analysis
                base_confidence_ok = rr_data["confidence"] >= 0.5
                
                # Apply market structure confidence multiplier
                if market_structure.get("success", False):
                    rr_data["confidence"] *= market_structure["confidence_multiplier"]
                
                # Volume profile enhancement
                vp_boost = 0
                if volume_profile.get("success", False):
                    if volume_profile.get("price_near_poc", False):
                        vp_boost += 0.1  # Boost for trading near POC
                    if volume_profile.get("price_in_value_area", False):
                        vp_boost += 0.05  # Small boost for value area
                    if volume_profile.get("high_volume_nodes"):
                        vp_boost += len(volume_profile["high_volume_nodes"]) * 0.05
                
                rr_data["confidence"] += vp_boost
                
                # Signal confluence boost
                if confluence_analysis["confluence_score"] >= 3:
                    rr_data["confidence"] *= 1.2  # 20% boost for high confluence
                elif confluence_analysis["confluence_score"] >= 2:
                    rr_data["confidence"] *= 1.1  # 10% boost for medium confluence
                
                # Enhanced confidence check
                enhanced_confidence_ok = rr_data["confidence"] >= 0.6  # Raised threshold
                
                print(f"📊 {symbol} Enhanced Signal Analysis:")
                rr_text = f"{rr_data['rr']:.2f}" if rr_data['rr'] is not None else "N/A"
                print(f"   R:R: {rr_text} (min: {symbol_min_rr})")
                print(f"   15m: {ok_15m} | Base Confidence: {base_confidence_ok} | Enhanced: {enhanced_confidence_ok}")
                print(f"   Volume: {rr_data['volume_ratio']:.2f}x | Volatility: {rr_data['volatility_ratio']:.2f}x")
                
                if market_structure.get("success", False):
                    print(f"   Market Structure: {market_structure['structure_rating']} ({market_structure['structure_quality']}/7)")
                    print(f"   Structure Notes: {', '.join(market_structure['structure_notes'][:2])}")  # First 2 notes
                
                if volume_profile.get("success", False):
                    vp_status = []
                    if volume_profile.get("price_near_poc"):
                        vp_status.append("Near POC")
                    if volume_profile.get("price_in_value_area"):
                        vp_status.append("In Value Area")
                    if volume_profile.get("high_volume_nodes"):
                        vp_status.append(f"{len(volume_profile['high_volume_nodes'])} HVN")
                    print(f"   Volume Profile: {' | '.join(vp_status) if vp_status else 'Normal distribution'}")
                
                if confluence_analysis["confluence_score"] > 0:
                    print(f"   Signal Confluence: {confluence_analysis['confluence_score']:.1f} ({confluence_analysis['net_bias']})")
                    print(f"   Signals: {confluence_analysis['bullish_signals']}🟢 vs {confluence_analysis['bearish_signals']}🔴")
                
                if mtf_ok and rr_ok and enhanced_confidence_ok:
                    
                    #Chặn trade sát POC nếu không phải breakout
                    no_trade_near_poc = volume_profile.get("success") \
                        and volume_profile.get("price_near_poc") \
                        and latest["type"] not in ("BB Breakout", "BOS")

                    if no_trade_near_poc:
                        print(f"❌ {symbol}: Chặn trade sát POC nếu không phải breakout")
                        return True, "Filtered near POC"

                    signal_time = latest['at'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Enhanced Entry/SL/TP display with risk calculation
                    entry_price = rr_data['entry']
                    sl_price = rr_data['sl']
                    tp1_price = rr_data['tp1']
                    
                    # Calculate risk and reward amounts
                    if sl_price and tp1_price:
                        risk_amount = abs(entry_price - sl_price)
                        reward_amount = abs(tp1_price - entry_price)
                        risk_percent = (risk_amount / entry_price) * 100
                        reward_percent = (reward_amount / entry_price) * 100
                    else:
                        risk_percent = reward_percent = 0
                    
                    # Build enhanced TP levels text with percentages
                    tp_text = f"🎯 <b>TP1: {fmt(rr_data['tp1'])}</b>"
                    if tp1_price:
                        tp1_percent = abs((tp1_price - entry_price) / entry_price) * 100
                        tp_text += f" (+{tp1_percent:.1f}%)"
                    
                    if rr_data['tp2']:
                        tp2_percent = abs((rr_data['tp2'] - entry_price) / entry_price) * 100
                        tp_text += f"\n🎯 TP2: {fmt(rr_data['tp2'])} (+{tp2_percent:.1f}%)"
                    if rr_data['tp3']:
                        tp3_percent = abs((rr_data['tp3'] - entry_price) / entry_price) * 100
                        tp_text += f"\n🎯 TP3: {fmt(rr_data['tp3'])} (+{tp3_percent:.1f}%)"
                    
                    # Enhanced market condition indicators
                    market_indicators = []
                    if rr_data['volume_ratio'] > VOLUME_THRESHOLD:
                        market_indicators.append("🔥 High Volume")
                    elif rr_data['volume_ratio'] < 0.8:
                        market_indicators.append("⚠️ Low Volume")
                    
                    if rr_data['volatility_ratio'] > VOLATILITY_THRESHOLD:
                        market_indicators.append("📈 High Volatility")
                    
                    # Add market structure info
                    if market_structure.get("success", False):
                        market_indicators.append(f"🏗️ {market_structure['structure_rating']}")
                    
                    # Add volume profile info
                    if volume_profile.get("success", False) and volume_profile.get("price_near_poc"):
                        market_indicators.append("📊 Near POC")
                    
                    # Add confluence info
                    if confluence_analysis["confluence_score"] >= 2:
                        market_indicators.append(f"🎯 Confluence: {confluence_analysis['confluence_score']:.1f}")
                    
                    market_text = " | ".join(market_indicators) if market_indicators else "📊 Normal Conditions"

                    # Get current time for signal sending
                    current_time = datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')
                    
                    msg_parts = [
                        f"<b>🚨 {symbol} {latest['type']} Signal</b>",
                        f"📊 Direction: <b>{side.upper()}</b> | Confidence: <b>{rr_data['confidence']:.0%}</b>",
                        f"⏰ Detection : {signal_time}",
                        f"📤 Signal : {current_time}",
                        f"",
                        f"📈 <b>TRADE SETUP:</b>",
                        f"💰 Entry: <b>{fmt(entry_price)}</b>",
                        f"🔴 Stop Loss: <b>{fmt(sl_price)}</b> ({risk_percent:.1f}%)",
                        tp_text,
                        f"",
                        f"📊 Risk:Reward = <b>1:{fmt(rr_data['rr'], 2)}</b>",
                        f"🎯 Strategy: {rr_data['strategy']}",
                        f"",
                        f"📈 <b>MARKET ANALYSIS:</b>",
                        f"🕐 3m Trend: {result3['label']} | 15m: {result15['label']}",
                        f"📋 Current: {result3['last_close']:.2f} | EMA50: {result3['ema50']:.2f} | RSI: {result3['rsi']:.1f}",
                        f"🔍 {market_text}"
                    ]
                    
                    # Add ATR info if available
                    if rr_data['atr']:
                        atr_percent = (rr_data['atr'] / rr_data['entry']) * 100
                        msg_parts.append(f"📊 ATR: {rr_data['atr']:.4f} ({atr_percent:.2f}%)")
                    
                    msg = "\n".join(msg_parts)
                    
                    print(f"✅ {symbol}: Sending ENHANCED signal (R:R={rr_data['rr']:.2f}, Final Confidence={rr_data['confidence']:.2f})")
                    

                    send_telegram_message(msg)
                    if SEND_IMAGES:
                        send_telegram_photo(save_paths["price_3m"], f"{symbol} Price 3m")
                        send_telegram_photo(save_paths["price_15m"], f"{symbol} Price 15m")
                    
                    last_signal_ids[symbol] = signal_id
                else:
                    filter_reasons = []
                    if not rr_ok: 
                        rr_text = f"{rr_data['rr']:.2f}" if rr_data['rr'] is not None else "N/A"
                        filter_reasons.append(f"R:R {rr_text} < {symbol_min_rr}")
                    if not mtf_ok: filter_reasons.append("15m misalignment")
                    if not enhanced_confidence_ok: filter_reasons.append(f"Low enhanced confidence {rr_data['confidence']:.2f}")
                    
                    print(f"❌ {symbol}: Signal filtered - {', '.join(filter_reasons)}")
        
        return True, "Success"
    except Exception as e:
        print(f"[ERROR] Processing {symbol}: {e}")
        error_counts[symbol] += 1
        return False, str(e)



def main():
    """Main loop that processes all configured symbols."""
    last_signal_ids = {symbol: None for symbol in SYMBOLS}
    last_strong_recommendations = {symbol: None for symbol in SYMBOLS}
    last_performance_report_time = datetime.now()
    performance_report_interval = 6 * 3600  # 6 hours in seconds
    
    first_run = True
    loop_count = 0
    
    print("\nStarting Enhanced Crypto Signal Detector...")
    try:
        print("🔧 Features: MongoDB, Volume Profile, Market Structure, Signal Confluence")
    except UnicodeEncodeError:
        print("Features: MongoDB, Volume Profile, Market Structure, Signal Confluence")
    
    # Send startup notification
    try:
        if os.path.exists('start_server.png'):
            send_telegram_photo('start_server.png', "🚀 Enhanced Signal Detector Started")
    except Exception as e:
        print(f"Error sending startup photo: {e}")
    
    while True:
        try:
            loop_count += 1
            all_recommendations = []
            
            for symbol in SYMBOLS:
                success, message = process_symbol(symbol, first_run, last_signal_ids)
                if success:
                    # Get current recommendation for this symbol
                    # We need to generate it again (simplified version)
                    try:
                        # Chỉ dựng lại dữ liệu / khuyến nghị khi bật cờ
                        if SEND_MARKET_DIRECTION:
                            df = get_klines(symbol=symbol)
                            df = only_closed(df)
                            df["ema50"] = ema(df["close"], 50)
                            df["ema100"] = ema(df["close"], 100)
                            df["rsi"] = rsi(df["close"], 14)
                            macd_line, macd_signal, macd_hist = macd(df["close"], 12, 26, 9)
                            df["macd_line"] = macd_line
                            df["macd_signal"] = macd_signal
                            df["macd_hist"] = macd_hist
                            df["vol_ma20"] = df["volume"].rolling(20).mean()
                            df = swing_points(df, window=3)
                            dfp = df.tail(120).copy()

                            df15 = get_klines_15m(symbol=symbol)
                            df15 = only_closed(df15)
                            df15["ema50"] = ema(df15["close"], 50)
                            df15["ema100"] = ema(df15["close"], 100)
                            df15["rsi"] = rsi(df15["close"], 14)
                            macd_line15, macd_signal15, macd_hist15 = macd(df15["close"], 12, 26, 9)
                            df15["macd_line"] = macd_line15
                            df15["macd_signal"] = macd_signal15
                            df15["macd_hist"] = macd_hist15
                            df15["vol_ma20"] = df15["volume"].rolling(20).mean()
                            df15 = swing_points(df15, window=3)
                            dfp15 = df15.tail(120).copy()

                            result3 = score_trend(dfp)
                            result15 = score_trend(dfp15)
                            signals = collect_reversal_signals(dfp)

                            volume_profile = calculate_volume_profile(dfp, price_range_pct=2.0)
                            market_structure = analyze_market_structure(dfp)
                            confluence_analysis = calculate_signal_confluence(signals, result3['last_close'])

                            recommendation = generate_trading_recommendation(
                                symbol, result3, result15, market_structure, volume_profile, confluence_analysis
                            )

                            # luôn append khi đã có recommendation
                            all_recommendations.append(recommendation)

                            # gửi thông báo strong recommendation (nếu có)
                            if recommendation['action'] in ['STRONG LONG', 'STRONG SHORT']:
                                last_rec = last_strong_recommendations.get(symbol)
                                current_rec_id = f"{symbol}_{recommendation['action']}_{recommendation['confidence']:.0f}"
                                if last_rec != current_rec_id:
                                    action_emoji = "📈🟢" if "LONG" in recommendation['action'] else "📉🔴"
                                    rec_msg_parts = [
                                        f"<b>{action_emoji} {symbol} {recommendation['action']}</b>",
                                        f"🎯 Confidence: <b>{recommendation['confidence']:.0f}%</b> | Risk: {recommendation['risk_level']}",
                                        f"📊 Bias Score: {recommendation['bias_score']:.1f}/10",
                                        f"💡 Key Reasons:",
                                    ]
                                    for reason in recommendation['reasons'][:3]:
                                        rec_msg_parts.append(f"   • {reason}")
                                    rec_msg_parts.extend([
                                        "",
                                        "⚡ Market Conditions:",
                                        f"   📊 Structure: {'✅' if recommendation['structure_supports'] else '❌'} | "
                                        f"Volume: {'✅' if recommendation['volume_supports'] else '❌'} | "
                                        f"MTF: {'✅' if recommendation['timeframe_alignment'] else '❌'}",
                                        f"📋 Price: {result3['last_close']:.2f} | RSI: {result3['rsi']:.1f}",
                                    ])
                                    rec_msg = "\n".join(rec_msg_parts)
                                    send_telegram_message(rec_msg)
                                    last_strong_recommendations[symbol] = current_rec_id
                                    print(f"📢 Sent {recommendation['action']} recommendation for {symbol}")
                    except Exception as e:
                        print(f"Error generating recommendation for {symbol}: {e}")
                    
                    time.sleep(SYMBOL_DELAY)  # Delay only after successful processing
                else:
                    print(f"Failed to process {symbol}: {message}")
            
            # Send periodic performance report (every 6 hours)
            current_time = datetime.now()
            if (current_time - last_performance_report_time).total_seconds() >= performance_report_interval:
                print("📊 Sending periodic performance report...")
                if send_performance_summary_to_telegram(period_days=7):
                    last_performance_report_time = current_time
                    print("✅ Performance report sent successfully")
                else:
                    print("⚠️ Performance report not sent (no data or error)")
            
            # Summary of recommendations every 10 loops
            if loop_count % 10 == 0 and all_recommendations:
                strong_longs = [r for r in all_recommendations if r['action'] in ['STRONG LONG', 'LONG']]
                strong_shorts = [r for r in all_recommendations if r['action'] in ['STRONG SHORT', 'SHORT']]
                
                print(f"\n📋 Current Market Overview (Loop {loop_count}):")
                if strong_longs:
                    long_symbols = [r['symbol'] for r in strong_longs]
                    print(f"   🟢 LONG Bias: {', '.join(long_symbols)}")
                if strong_shorts:
                    short_symbols = [r['symbol'] for r in strong_shorts]
                    print(f"   🔴 SHORT Bias: {', '.join(short_symbols)}")
                
                holds = [r for r in all_recommendations if r['action'] == 'HOLD']
                if holds:
                    hold_symbols = [r['symbol'] for r in holds]
                    print(f"   ⏸️ HOLD: {', '.join(hold_symbols)}")
            
            first_run = False
            time.sleep(LOOP_SLEEP_SECONDS)
            
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            # Send final performance report before shutdown
            print("📊 Sending final performance report...")
            send_performance_summary_to_telegram(period_days=1)  # Last 24 hours
            break
        except Exception as e:
            print(f"[CRITICAL ERROR] {e}")
            time.sleep(RETRY_DELAY)

if __name__ == "__main__":
    try:
        print("\nStarting ...")
        main()
    except KeyboardInterrupt:
        print("\nDetected Ctrl+C, shutting down gracefully...")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
