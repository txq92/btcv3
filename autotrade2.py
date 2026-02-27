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
import json
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
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")  # Để trống trước, sau đó điền
TESTNET_MODE = os.environ.get("TESTNET_MODE", "False").lower() == "true"

BASE_URL = "https://testnet.binancefuture.com" if TESTNET_MODE else "https://fapi.binance.com"

if not BINANCE_API_KEY or not BINANCE_API_SECRET or not TELEGRAM_BOT_TOKEN:
    print("❌ Thiếu BINANCE_API_KEY, BINANCE_API_SECRET hoặc TELEGRAM_BOT_TOKEN trong .env")
    exit()

GLOBAL_RUNNING = False
TRADE_AMOUNT_USDT = 10.0
GLOBAL_LEVERAGE = 25
TIMEFRAME = "5m"
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
LAST_PROCESSED_MINUTE = -1

MARKET_DATA_CACHE = {}

SYMBOL_CONFIGS = {
    "BTCUSDT": {"X": 0.15, "Y": 0.05, "Active": True},
    "ETHUSDT": {"X": 0.3, "Y": 0.05, "Active": True},
    "SOLUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "BNBUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "XRPUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "DOGEUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "ADAUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "AVAXUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "SHIBUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "DOTUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "LINKUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "TRXUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "UNIUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "ATOMUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "ICPUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "ETCUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "FILUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "NEARUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
    "APTUSDT": {"X": 0.35, "Y": 0.05, "Active": True},
}

# ==============================================================================
# ========== BINANCE API CORE ==========
# ==============================================================================
def binance_request(method: str, endpoint: str, params=None, signed=False):
    try:
        url = BASE_URL + endpoint
        headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
        if params is None:
            params = {}

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

def get_market_rules(symbol):
    if symbol in MARKET_DATA_CACHE:
        return MARKET_DATA_CACHE[symbol]
    res = binance_request("GET", "/fapi/v1/exchangeInfo")
    if res and "symbols" in res:
        for s in res["symbols"]:
            if s["symbol"] == symbol:
                tick_sz = lot_sz = min_notional = 0
                for f in s["filters"]:
                    if f["filterType"] == "PRICE_FILTER":
                        tick_sz = float(f["tickSize"])
                    elif f["filterType"] == "LOT_SIZE":
                        lot_sz = float(f["stepSize"])
                    elif f["filterType"] == "MIN_NOTIONAL":
                        min_notional = float(f.get("minNotional", 5))
                prec = len(str(tick_sz).split(".")[-1])
                data = {"tickSz": tick_sz, "lotSz": lot_sz, "prec": prec, "minNotional": min_notional}
                MARKET_DATA_CACHE[symbol] = data
                return data
    return None

def check_existing_position(symbol):
    res = binance_request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)
    if isinstance(res, list):
        for pos in res:
            if float(pos.get("positionAmt", 0)) != 0:
                return "LONG" if float(pos["positionAmt"]) > 0 else "SHORT"
    return None

def execute_smart_trade(symbol, side, entry_price, low, high):
    try:
        if check_existing_position(symbol):
            return None, "0", 0, 0, "Đã có vị thế"

        rules = get_market_rules(symbol)
        if not rules:
            return None, "0", 0, 0, "Không lấy rules"

        # Tính quantity
        notional = TRADE_AMOUNT_USDT * GLOBAL_LEVERAGE
        qty = math.floor(notional / entry_price / rules["lotSz"]) * rules["lotSz"]
        if qty * entry_price < rules["minNotional"]:
            qty = math.ceil(rules["minNotional"] / entry_price / rules["lotSz"]) * rules["lotSz"]
        qty_str = f"{qty:.8f}".rstrip("0").rstrip(".")

        # Set leverage + isolated
        binance_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": GLOBAL_LEVERAGE}, signed=True)
        binance_request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": "ISOLATED"}, signed=True)

        # Lệnh market
        order_side = side.upper()
        res = binance_request("POST", "/fapi/v1/order", {
            "symbol": symbol,
            "side": order_side,
            "type": "MARKET",
            "quantity": qty_str
        }, signed=True)

        if res.get("code") != 200:
            return None, "0", 0, 0, res.get("msg", "Market fail")

        # Tính SL & TP
        prec = rules["prec"]
        if side == "buy":
            sl = round(low * 0.998, prec)
            tp = round(entry_price + (entry_price - sl) * 2, prec)
            sl_side = "SELL"
            tp_side = "SELL"
        else:
            sl = round(high * 1.002, prec)
            tp = round(entry_price - (entry_price - sl) * 2, prec)
            sl_side = "BUY"
            tp_side = "BUY"

        # Đặt SL (STOP_MARKET)
        binance_request("POST", "/fapi/v1/order", {
            "symbol": symbol,
            "side": sl_side,
            "type": "STOP_MARKET",
            "quantity": qty_str,
            "stopPrice": str(sl),
            "reduceOnly": "true",
            "workingType": "MARK_PRICE",
            "timeInForce": "GTC"
        }, signed=True)

        # Đặt TP (TAKE_PROFIT_MARKET)
        binance_request("POST", "/fapi/v1/order", {
            "symbol": symbol,
            "side": tp_side,
            "type": "TAKE_PROFIT_MARKET",
            "quantity": qty_str,
            "stopPrice": str(tp),
            "reduceOnly": "true",
            "workingType": "MARK_PRICE",
            "timeInForce": "GTC"
        }, signed=True)

        return res, qty_str, sl, tp, ""
    except Exception as e:
        return None, "0", 0, 0, str(e)

# Trailing SL (polling)
def manage_trailing_sl():
    try:
        positions = binance_request("GET", "/fapi/v2/positionRisk", signed=True)
        if not isinstance(positions, list):
            return
        for pos in positions:
            if float(pos["positionAmt"]) == 0:
                continue
            sym = pos["symbol"]
            if sym not in SYMBOL_CONFIGS:
                continue
            entry_px = float(pos["entryPrice"])
            pos_side = "LONG" if float(pos["positionAmt"]) > 0 else "SHORT"

            # Lấy close gần nhất
            c_res = requests.get(f"{BASE_URL}/fapi/v1/klines?symbol={sym}&interval={TIMEFRAME}&limit=5").json()
            if not c_res:
                continue
            last_close = float(c_res[-2][4])

            # Lấy SL hiện tại (tìm order STOP_MARKET reduceOnly)
            orders = binance_request("GET", "/fapi/v1/openOrders", {"symbol": sym}, signed=True)
            sl_order = None
            for o in orders:
                if o["type"] in ("STOP_MARKET", "STOP") and o["reduceOnly"] == "true":
                    sl_order = o
                    break
            if not sl_order:
                continue
            current_sl = float(sl_order["stopPrice"])
            algo_id = sl_order["orderId"]

            risk = abs(entry_px - current_sl)
            rr1 = entry_px + risk if pos_side == "LONG" else entry_px - risk
            rr2 = entry_px + risk * 2 if pos_side == "LONG" else entry_px - risk * 2

            new_sl = None
            if pos_side == "LONG":
                if last_close >= rr2 and current_sl < rr1:
                    new_sl = round(rr1, 2)
                elif last_close >= rr1 and current_sl < entry_px:
                    new_sl = round(entry_px, 2)
            else:
                if last_close <= rr2 and current_sl > rr1:
                    new_sl = round(rr1, 2)
                elif last_close <= rr1 and current_sl > entry_px:
                    new_sl = round(entry_px, 2)

            if new_sl:
                # Cancel old SL
                binance_request("DELETE", "/fapi/v1/order", {"symbol": sym, "orderId": algo_id}, signed=True)
                # Place new SL
                binance_request("POST", "/fapi/v1/order", {
                    "symbol": sym,
                    "side": "SELL" if pos_side == "LONG" else "BUY",
                    "type": "STOP_MARKET",
                    "quantity": abs(float(pos["positionAmt"])),
                    "stopPrice": str(new_sl),
                    "reduceOnly": "true",
                    "workingType": "MARK_PRICE"
                }, signed=True)
                print(f"🛡️ Trail SL {sym} -> {new_sl}")
    except:
        pass

def run_market_scan():
    for sym, cfg in SYMBOL_CONFIGS.items():
        if not cfg.get("Active"):
            continue
        try:
            url = f"{BASE_URL}/fapi/v1/klines?symbol={sym}&interval={TIMEFRAME}&limit=50"
            resp = requests.get(url, timeout=10).json()
            df = pd.DataFrame(resp, columns=['ts', 'o', 'h', 'l', 'c', 'v', 'ct', 'q', 'n', 'tb', 'tq', 'i'])
            df[['o', 'h', 'l', 'c']] = df[['o', 'h', 'l', 'c']].astype(float)
            df = df.sort_values('ts').reset_index(drop=True)
            df['ema20'] = df['c'].ewm(span=20, adjust=False).mean()

            s = df.iloc[-2]
            max_oc = max(s['o'], s['c'])
            min_oc = min(s['o'], s['c'])
            up_wick = ((s['h'] - max_oc) / max_oc) * 100
            lo_wick = ((min_oc - s['l']) / min_oc) * 100

            side = None
            if (s['c'] > s['o']) and (s['c'] > s['ema20']) and (lo_wick >= cfg['X']) and (up_wick <= cfg['Y']):
                side = "buy"
            elif (s['c'] < s['o']) and (s['c'] < s['ema20']) and (up_wick >= cfg['X']) and (lo_wick <= cfg['Y']):
                side = "sell"

            if side:
                res, sz, sl, tp, err = execute_smart_trade(sym, side, s['c'], s['l'], s['h'])
                total_vol = TRADE_AMOUNT_USDT * GLOBAL_LEVERAGE

                if res and res.get("orderId"):
                    msg = f"✅ OK | {side.upper()} {sym}\nVol: {total_vol} USDT | SL: {sl} | TP: {tp}"
                else:
                    msg = f"❌ LỖI: {err or 'Fail'} | {side.upper()} {sym}\nVol: {total_vol} USDT"

                print(msg)
                if TELEGRAM_CHAT_ID:
                    try:
                        bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
                    except:
                        pass
        except Exception as e:
            print(f"Scan error {sym}: {e}")

def main_loop():
    global LAST_PROCESSED_MINUTE
    while True:
        if GLOBAL_RUNNING:
            now = datetime.now(VIETNAM_TZ)
            if now.minute % 5 == 0 and now.minute != LAST_PROCESSED_MINUTE:
                time.sleep(5)
                run_market_scan()
                manage_trailing_sl()
                LAST_PROCESSED_MINUTE = now.minute
        time.sleep(1)

threading.Thread(target=main_loop, daemon=True).start()

# ==============================================================================
# ========== TELEGRAM BOT (có bảo mật Chat ID + alert) ==========
# ==============================================================================
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

def is_authorized(message):
    if not TELEGRAM_CHAT_ID:
        return True
    return str(message.chat.id) == TELEGRAM_CHAT_ID

@bot.message_handler(commands=['getid'])
def send_chat_id(message):
    bot.reply_to(message, f"🆔 **Chat ID của bạn:**\n`{message.chat.id}`\n\nDán vào `.env` rồi restart bot nhé!")

@bot.message_handler(commands=['help'])
def send_help(message):
    if not is_authorized(message):
        return
    help_text = """📋 **HƯỚNG DẪN BINANCE BOT RR V5**

/start - Chào mừng
/getid - Lấy Chat ID của bạn
/status - Xem trạng thái
/volume 15 - Đặt vốn (USDT)
/leverage 20 - Đặt đòn bẩy
/run - Bật bot
/stop - Tắt bot
/testnet - Bật Testnet (demo)
/mainnet - Bật Live
/mode - Xem mode hiện tại
/help - Xem hướng dẫn này"""
    bot.reply_to(message, help_text, parse_mode='Markdown')

@bot.message_handler(commands=['status', 'mode'])
def send_status(message):
    if not is_authorized(message):
        return
    mode_text = "🧪 **TESTNET MODE** (demo)" if TESTNET_MODE else "🔴 **LIVE MODE**"
    active = sum(1 for v in SYMBOL_CONFIGS.values() if v.get("Active"))
    text = f"""📊 **TRẠNG THÁI BOT**

{mode_text}
💰 Volume: **{TRADE_AMOUNT_USDT} USDT**
🔥 Leverage: **{GLOBAL_LEVERAGE}x**
🟢 Trạng thái: **{'ĐANG CHẠY' if GLOBAL_RUNNING else 'DỪNG'}**
📊 Coin active: **{active}**

✅ Sẵn sàng!"""
    bot.reply_to(message, text, parse_mode='Markdown')

@bot.message_handler(commands=['volume'])
def set_volume(message):
    if not is_authorized(message):
        return
    try:
        global TRADE_AMOUNT_USDT
        TRADE_AMOUNT_USDT = float(message.text.split()[1])
        bot.reply_to(message, f"✅ Đã đặt Volume = {TRADE_AMOUNT_USDT} USDT")
    except:
        bot.reply_to(message, "❌ Sai cú pháp! Ví dụ: `/volume 15`")

@bot.message_handler(commands=['leverage'])
def set_leverage(message):
    if not is_authorized(message):
        return
    try:
        global GLOBAL_LEVERAGE
        GLOBAL_LEVERAGE = int(message.text.split()[1])
        bot.reply_to(message, f"✅ Đã đặt Leverage = {GLOBAL_LEVERAGE}x")
    except:
        bot.reply_to(message, "❌ Sai cú pháp! Ví dụ: `/leverage 20`")

@bot.message_handler(commands=['run'])
def run_bot(message):
    if not is_authorized(message):
        return
    global GLOBAL_RUNNING
    GLOBAL_RUNNING = True
    bot.reply_to(message, f"🚀 **BOT ĐÃ KHỞI ĐỘNG!**\nVolume: {TRADE_AMOUNT_USDT} | Leverage: {GLOBAL_LEVERAGE}x")

@bot.message_handler(commands=['stop'])
def stop_bot(message):
    if not is_authorized(message):
        return
    global GLOBAL_RUNNING
    GLOBAL_RUNNING = False
    bot.reply_to(message, "⛔ **BOT ĐÃ DỪNG**")

@bot.message_handler(commands=['testnet'])
def set_testnet(message):
    if not is_authorized(message):
        return
    # Lưu ý: thay đổi TESTNET_MODE chỉ có hiệu lực khi restart
    bot.reply_to(message, "🧪 Đặt TESTNET_MODE=True trong .env và restart bot để chuyển sang demo!")

@bot.message_handler(commands=['mainnet'])
def set_mainnet(message):
    if not is_authorized(message):
        return
    bot.reply_to(message, "🔴 Đặt TESTNET_MODE=False trong .env và restart bot để chạy thật!")

if __name__ == "__main__":
    print(f"🤖 Binance Bot RR V5 khởi động... Mode: {'TESTNET' if TESTNET_MODE else 'LIVE'}")
    bot.infinity_polling(none_stop=True)
