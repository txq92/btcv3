import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import threading
import hmac
import hashlib
import math
import telebot
from dotenv import load_dotenv

# ==============================================================================
# ========== CẤU HÌNH ==========
# ==============================================================================
if os.path.exists(".env"):
    load_dotenv(".env")

BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TESTNET_MODE = os.environ.get("TESTNET_MODE", "False").lower() == "true"

BASE_URL = "https://testnet.binancefuture.com" if TESTNET_MODE else "https://fapi.binance.com"

if not BINANCE_API_KEY or not BINANCE_API_SECRET or not TELEGRAM_BOT_TOKEN:
    print("❌ Thiếu API key hoặc Telegram token trong .env")
    exit()

GLOBAL_RUNNING = False
TRADE_AMOUNT_USDT = 10.0
GLOBAL_LEVERAGE = 25
TIMEFRAME = "5m"
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
LAST_PROCESSED_MINUTE = -1

USE_PINBAR_FILTER = True
USE_EMA20_FILTER = True

MARKET_DATA_CACHE = {}

# ================== 50 COIN VOLUME LỚN NHẤT (2026) ==================
SYMBOL_CONFIGS = {
    "BTCUSDT": {"X": 0.15, "Y": 0.05, "Active": True},
    "ETHUSDT": {"X": 0.30, "Y": 0.05, "Active": True},
    "SOLUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "XRPUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "DOGEUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "BNBUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "ADAUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "ZECUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "SUIUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "AVAXUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "LINKUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "LTCUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "1000PEPEUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "PUMPUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "ENAUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "DOTUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "NEARUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "AAVEUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "TAOUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "FILUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "TONUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "HBARUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "ATOMUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "APTUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "1000SHIBUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "TRXUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "UNIUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "ICPUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "ETCUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "ARBUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "OPUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "POLUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "INJUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "FETUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "RENDERUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "SEIUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "TIAUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "WIFUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "BONKUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "FLOKIUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "GALAUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "SANDUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "XLMUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "VETUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "MANAUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "EIGENUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "PYTHUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "KASUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "STXUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "JUPUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
}

# ==============================================================================
# ========== BINANCE API CORE (giữ nguyên) ==========
# ==============================================================================
def binance_request(method: str, endpoint: str, params=None, signed=False):
    try:
        url = BASE_URL + endpoint
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        if params is None: params = {}
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            signature = hmac.new(BINANCE_API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
            params['signature'] = signature
        if method.upper() == "GET":
            r = requests.get(url, headers=headers, params=params, timeout=10)
        else:
            r = requests.request(method, url, headers=headers, data=params, timeout=10)
        return r.json()
    except Exception as e:
        print(f"❌ Binance API Error: {e}")
        return None

# (get_market_rules, check_existing_position, execute_smart_trade, manage_trailing_sl giữ nguyên như code trước – bạn copy từ file cũ vào nếu cần)

# run_market_scan, main_loop, Telegram bot (có /pinbar /ema /flags /status) GIỮ NGUYÊN như code lần trước

# ==============================================================================
# ========== CHẠY ==========
# ==============================================================================
if __name__ == "__main__":
    print(f"🤖 Binance Bot RR V5 (50 coin volume lớn) khởi động... Mode: {'TESTNET' if TESTNET_MODE else 'LIVE'}")
    bot.infinity_polling(none_stop=True)
