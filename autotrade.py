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
import base64
import math
import telebot
from dotenv import load_dotenv

# ==============================================================================
# ========== CẤU HÌNH ==========
# ==============================================================================
if os.path.exists(".env"):
    load_dotenv(".env")

OKX_API_KEY = os.environ.get("OKX_API_KEY")
OKX_SECRET_KEY = os.environ.get("OKX_SECRET_KEY")
OKX_PASSPHRASE = os.environ.get("OKX_PASSPHRASE")
OKX_BASE_URL = "https://www.okx.com"
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

if not TELEGRAM_BOT_TOKEN:
    print("❌ Thiếu TELEGRAM_BOT_TOKEN trong .env")
    exit()

DEMO_MODE = False                    # True = test ảo, False = tiền thật
GLOBAL_RUNNING = False
TRADE_AMOUNT_USDT = 10.0
GLOBAL_LEVERAGE = 25
TIMEFRAME = "5m"
VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
LAST_PROCESSED_MINUTE = -1

MARKET_DATA_CACHE = {}

SYMBOL_CONFIGS = {
    "XAG-USDT-SWAP": {"X": 0.5, "Y": 0.05, "Active": False},
    "BTC-USDT-SWAP": {"X": 0.15, "Y": 0.05, "Active": True},
    "ETH-USDT-SWAP": {"X": 0.3, "Y": 0.05, "Active": True},
    "SOL-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "BNB-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "XRP-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "DOGE-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "ADA-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "AVAX-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "SHIB-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "DOT-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "LINK-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "TRX-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "UNI-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "ATOM-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "ICP-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "ETC-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "FIL-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "NEAR-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "APT-USDT-SWAP": {"X": 0.35, "Y": 0.05, "Active": True},
    "XAU-USDT-SWAP": {"X": 0.1, "Y": 0.05, "Active": False},
}

# ==============================================================================
# ========== API CORE (có DEMO MODE) ==========
# ==============================================================================

def okx_request(method, endpoint, body=None):
    try:
        ts = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
        body_str = json.dumps(body) if body else ""
        message = ts + method + endpoint + body_str
        mac = hmac.new(bytes(OKX_SECRET_KEY, 'utf-8'), bytes(message, 'utf-8'), hashlib.sha256)
        sign = base64.b64encode(mac.digest()).decode()
        
        headers = {
            'OK-ACCESS-KEY': OKX_API_KEY,
            'OK-ACCESS-SIGN': sign,
            'OK-ACCESS-TIMESTAMP': ts,
            'OK-ACCESS-PASSPHRASE': OKX_PASSPHRASE,
            'Content-Type': 'application/json'
        }
        if DEMO_MODE:
            headers['x-simulated-trading'] = '1'
        
        res = requests.request(method, OKX_BASE_URL + endpoint, headers=headers, data=body_str, timeout=10)
        return res.json()
    except Exception as e:
        print(f"❌ API Error: {e}")
        return None

def get_market_rules(symbol):
    if symbol in MARKET_DATA_CACHE: return MARKET_DATA_CACHE[symbol]
    try:
        url = f"{OKX_BASE_URL}/api/v5/public/instruments?instType=SWAP&instId={symbol}"
        res = requests.get(url, timeout=10).json()
        if res.get('code') == '0' and res.get('data'):
            inst = res['data'][0]
            data = {
                "lotSz": float(inst['lotSz']),
                "tickSz": float(inst['tickSz']),
                "prec": len(inst['tickSz'].split('.')[-1]) if '.' in inst['tickSz'] else 0,
                "minSz": float(inst['minSz']),
                "ctVal": float(inst['ctVal'])
            }
            MARKET_DATA_CACHE[symbol] = data
            return data
    except: pass
    return None

def check_existing_position(symbol):
    res = okx_request("GET", f"/api/v5/account/positions?instId={symbol}")
    if res and res.get('code') == '0' and res.get('data'):
        for pos in res['data']:
            if pos['pos'] != '0': return pos['posSide']
    return None

def execute_smart_trade(symbol, side, entry_price, low, high):
    try:
        if check_existing_position(symbol):
            return None, "0", 0, 0, "Đã có vị thế"
        rules = get_market_rules(symbol)
        if not rules: return None, "0", 0, 0, "Không lấy được rules"

        ct_val = rules['ctVal']
        lot_sz = rules['lotSz']
        prec = rules['prec']
        min_sz = rules['minSz']

        total_notional_usdt = TRADE_AMOUNT_USDT * GLOBAL_LEVERAGE
        raw_sz = total_notional_usdt / (entry_price * ct_val)
        size = math.floor(raw_sz / lot_sz) * lot_sz
        if size < min_sz: size = min_sz
        sz_str = format(size, 'f').rstrip('0').rstrip('.')

        pos_side = "long" if side == "buy" else "short"
        if side == "buy":
            sl = round(low * 0.998, prec)
        else:
            sl = round(high * 1.002, prec)

        risk = abs(entry_price - sl)
        tp = round(entry_price + (risk * 2), prec) if side == "buy" else round(entry_price - (risk * 2), prec)

        okx_request("POST", "/api/v5/account/set-leverage", {
            "instId": symbol, "lever": str(GLOBAL_LEVERAGE), "mgnMode": "isolated", "posSide": pos_side
        })

        body = {
            "instId": symbol, "tdMode": "isolated", "side": side, "posSide": pos_side,
            "ordType": "market", "sz": sz_str,
            "attachAlgoOrds": [
                {"attachAlgoOrdType": "sl", "slTriggerPx": str(sl), "slOrdPx": "-1"},
                {"attachAlgoOrdType": "tp", "tpTriggerPx": str(tp), "tpOrdPx": "-1"}
            ]
        }
        res = okx_request("POST", "/api/v5/trade/order", body)
        return res, sz_str, sl, tp, res.get('msg') if res and res.get('code') != '0' else ""
    except Exception as e:
        return None, "0", 0, 0, str(e)

def manage_trailing_sl():
    try:
        pos_res = okx_request("GET", "/api/v5/account/positions")
        if not pos_res or pos_res.get('code') != '0': return
        for pos in pos_res.get('data', []):
            if pos['pos'] == '0': continue
            sym = pos['instId']
            if sym not in SYMBOL_CONFIGS: continue
            entry_px = float(pos['avgPx'])
            pos_side = pos['posSide']

            c_res = requests.get(f"{OKX_BASE_URL}/api/v5/market/history-candles?instId={sym}&bar={TIMEFRAME}&limit=5").json()
            if not c_res.get('data'): continue
            last_close = float(c_res['data'][1][4])

            algo_res = okx_request("GET", f"/api/v5/trade/orders-algo?instId={sym}&ordType=conditional")
            current_sl = algo_id = None
            for algo in algo_res.get('data', []):
                if algo.get('slTriggerPx'):
                    current_sl, algo_id = float(algo['slTriggerPx']), algo['algoId']
                    break
            if not algo_id: continue

            risk = abs(entry_px - current_sl)
            rr1 = entry_px + risk if pos_side == 'long' else entry_px - risk
            rr2 = entry_px + risk*2 if pos_side == 'long' else entry_px - risk*2
            prec = get_market_rules(sym)['prec']

            new_sl = None
            if pos_side == 'long':
                if last_close >= rr2 and current_sl < rr1: new_sl = round(rr1, prec)
                elif last_close >= rr1 and current_sl < entry_px: new_sl = round(entry_px, prec)
            else:
                if last_close <= rr2 and current_sl > rr1: new_sl = round(rr1, prec)
                elif last_close <= rr1 and current_sl > entry_px: new_sl = round(entry_px, prec)

            if new_sl:
                okx_request("POST", "/api/v5/trade/amend-algos", {"instId": sym, "algoId": algo_id, "newSlTriggerPx": str(new_sl)})
    except: pass

def run_market_scan():
    for sym, cfg in SYMBOL_CONFIGS.items():
        if not cfg.get("Active"): continue
        try:
            resp = requests.get(f"{OKX_BASE_URL}/api/v5/market/history-candles?instId={sym}&bar={TIMEFRAME}&limit=50", timeout=10).json()
            data = resp.get('data', [])
            if not data: continue
            df = pd.DataFrame(data, columns=['ts','o','h','l','c','v','volCcy','volCcyQuote','confirm'])
            df[['o','h','l','c']] = df[['o','h','l','c']].astype(float)
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
                if res and res.get('code') == '0':
                    msg = f"✅ OK | {side.upper()} {sym}\nVol: {total_vol} USDT | SL: {sl} | TP: {tp}"
                else:
                    msg = f"❌ LỖI: {err or 'Fail'} | {side.upper()} {sym}\nVol: {total_vol} USDT | SL: {sl} | TP: {tp}"
                print(msg)
        except: pass

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
# ========== TELEGRAM BOT ==========
# ==============================================================================
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "🤖 **OKX Bot RR V5** đã sẵn sàng!\nGõ /help để xem đầy đủ cách dùng.")

@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = """📋 **HƯỚNG DẪN SỬ DỤNG OKX BOT RR V5**

✅ **Lệnh cơ bản:**
• /start          → Khởi động & chào mừng
• /help           → Xem hướng dẫn này
• /status         → Xem trạng thái bot (volume, leverage, mode...)
• /mode           → Xem mode hiện tại (Demo hay Live)

💰 **Cài đặt:**
• /volume 15      → Đặt vốn mỗi lệnh = 15 USDT
• /leverage 20    → Đặt đòn bẩy = 20x

▶️ **Điều khiển bot:**
• /run            → Bật bot (tự động quét mỗi 5 phút)
• /stop           → Tắt bot ngay lập tức

🧪 **Demo / Live:**
• /demo           → Chuyển sang DEMO (test tiền ảo - an toàn 100%)
• /live           → Chuyển sang LIVE (tiền thật)
• /mode           → Kiểm tra đang Demo hay Live

📌 **Lưu ý quan trọng:**
• Bot chỉ quét nến đã đóng (không dùng nến đang hình thành)
• Mỗi 5 phút bot sẽ tự động kiểm tra tất cả coin active
• Khi bật /run, bot chạy ngầm 24/7
• Demo mode dùng header x-simulated-trading → không mất tiền thật
• Có thể chuyển Demo ↔ Live bất kỳ lúc nào

Gõ lệnh ngay để bắt đầu! 🚀"""
    bot.reply_to(message, help_text, parse_mode='Markdown')

@bot.message_handler(commands=['status', 'mode'])
def send_status(message):
    mode_text = "🧪 **DEMO MODE** (tiền ảo)" if DEMO_MODE else "🔴 **LIVE MODE** (tiền thật)"
    active_count = sum(1 for v in SYMBOL_CONFIGS.values() if v.get("Active"))
    text = f"""📊 **TRẠNG THÁI BOT**

{mode_text}

💰 Volume: **{TRADE_AMOUNT_USDT} USDT**
🔥 Leverage: **{GLOBAL_LEVERAGE}x**
📈 Notional: **{TRADE_AMOUNT_USDT * GLOBAL_LEVERAGE} USDT**
🟢 Trạng thái: **{'ĐANG CHẠY' if GLOBAL_RUNNING else 'DỪNG'}**
📊 Coin active: **{active_count}/20**

✅ Sẵn sàng nhận lệnh!"""
    bot.reply_to(message, text, parse_mode='Markdown')

@bot.message_handler(commands=['volume'])
def set_volume(message):
    try:
        amt = float(message.text.split()[1])
        global TRADE_AMOUNT_USDT
        TRADE_AMOUNT_USDT = amt
        bot.reply_to(message, f"✅ Đã đặt **Volume = {amt} USDT**")
    except:
        bot.reply_to(message, "❌ Sai cú pháp!\nVí dụ: `/volume 15`", parse_mode='Markdown')

@bot.message_handler(commands=['leverage'])
def set_leverage(message):
    try:
        lev = int(message.text.split()[1])
        global GLOBAL_LEVERAGE
        GLOBAL_LEVERAGE = lev
        bot.reply_to(message, f"✅ Đã đặt **Leverage = {lev}x**")
    except:
        bot.reply_to(message, "❌ Sai cú pháp!\nVí dụ: `/leverage 20`", parse_mode='Markdown')

@bot.message_handler(commands=['run'])
def run_bot(message):
    global GLOBAL_RUNNING
    if GLOBAL_RUNNING:
        bot.reply_to(message, "⚠️ Bot đã đang chạy!")
    else:
        GLOBAL_RUNNING = True
        bot.reply_to(message, f"🚀 **BOT ĐÃ KHỞI ĐỘNG!**\nVolume: {TRADE_AMOUNT_USDT} USDT | Leverage: {GLOBAL_LEVERAGE}x\nBot sẽ quét mỗi 5 phút.")

@bot.message_handler(commands=['stop'])
def stop_bot(message):
    global GLOBAL_RUNNING
    if not GLOBAL_RUNNING:
        bot.reply_to(message, "⚠️ Bot đã dừng rồi!")
    else:
        GLOBAL_RUNNING = False
        bot.reply_to(message, "⛔ **BOT ĐÃ DỪNG AN TOÀN**")

@bot.message_handler(commands=['demo'])
def set_demo(message):
    global DEMO_MODE
    DEMO_MODE = True
    bot.reply_to(message, "🧪 **ĐÃ BẬT DEMO MODE**\nTất cả lệnh sau dùng tiền ảo - an toàn 100%!")

@bot.message_handler(commands=['live'])
def set_live(message):
    global DEMO_MODE
    DEMO_MODE = False
    bot.reply_to(message, "🔴 **ĐÃ BẬT LIVE MODE**\nCảnh báo: Sẽ dùng tiền thật!")

# ==============================================================================
# ========== CHẠY ==========
# ==============================================================================
if __name__ == "__main__":
    print(f"🤖 OKX Bot RR V5 khởi động... Mode: {'DEMO' if DEMO_MODE else 'LIVE'}")
    bot.infinity_polling(none_stop=True)
