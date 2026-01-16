# scripts/token_utils.py

import os
import json
import aiohttp
import asyncio
import pandas as pd
from io import StringIO
from collections import deque
from datetime import datetime, timezone

from dotenv import load_dotenv
from typing import List

load_dotenv()

PROGRESS_FILE = "../scripts/progress.json"
PAGE_DELAY = float(os.getenv("PAGE_DELAY", 1))
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
HELIUS_BASE_URL = "https://mainnet.helius-rpc.com/"
GMGN_COOKIES_PATH = "cookies/gmgn_cookies.txt"
GMGN_BASE = "https://gmgn.ai/pf/api/v1/wallet/sol"


# ------------------- Cookies -------------------

def load_netscape_cookies(path: str) -> dict:
    cookies = {}
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) == 7:
                _, _, _, _, _, name, value = parts
                cookies[name] = value
    return cookies


# ------------------- Solscan -------------------

async def fetch_page(session, address, from_time: int, to_time: int = 0):
    url = "https://api-v2.solscan.io/v2/account/defi/export"
    params = {"address": address, "from_time": from_time, "to_time": to_time}
    
    async with session.get(url, params=params) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise Exception(f"HTTP {resp.status}: {text}")

        csv_raw = await resp.text()
        df = pd.read_csv(StringIO(csv_raw))
        if not df.empty:
            df["Human Time TS"] = pd.to_datetime(df["Human Time"], utc=True).astype(int) // 10**9
        return df


async def load_all_defi_activity(address: str, cookies_path="cookies/solscan_cookies.txt",
                                 save_dir="pages", output_file=None, from_time: int | None = None):
    """
    Загружает страницы DeFi активности Solscan с from_time до текущего времени.
    """
    if output_file is None:
        output_file = f"data/{address}_all_defi_activity.csv"

    os.makedirs(save_dir, exist_ok=True)
    cookies = load_netscape_cookies(cookies_path)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://solscan.io/",
        "Origin": "https://solscan.io",
        "Accept": "application/json, text/plain, */*"
    }

    page_num = 1
    to_time = int(datetime.now(timezone.utc).timestamp())  # верхняя граница = текущее время UTC
    from_time = from_time or 0  # если не указано — с начала

    async with aiohttp.ClientSession(cookies=cookies, headers=headers) as session:
        while to_time > from_time:
            print(f"Downloading page {page_num}, from_time={from_time}, to_time={to_time}")
            df = await fetch_page(session, address, from_time=from_time, to_time=to_time)

            if df.empty:
                print("No more data, finishing.")
                break

            filename = os.path.join(save_dir, f"page_{page_num}.csv")
            df.to_csv(filename, index=False)

            # Для следующей страницы уменьшаем to_time на min timestamp страницы минус 1
            min_ts = df["Human Time TS"].min()
            if min_ts <= from_time:
                print("Reached from_time limit, stopping.")
                break
            to_time = int(min_ts - 1)
            page_num += 1
            await asyncio.sleep(PAGE_DELAY)

    # объединяем страницы
    saved_files = [os.path.join(save_dir, f) for f in os.listdir(save_dir)
                   if f.startswith("page_") and f.endswith(".csv")]
    if not saved_files:
        return pd.DataFrame()
    all_dfs = [pd.read_csv(f) for f in sorted(saved_files)]
    result = pd.concat(all_dfs).drop_duplicates().reset_index(drop=True)

    result.to_csv(output_file, index=False)
    for f in saved_files:
        os.remove(f)

    return result


async def load_token_trades_solscan(token: str, from_time: int | None = None) -> pd.DataFrame:
    df = await load_all_defi_activity(address=token, from_time=from_time, save_dir="pages")
    if df.empty:
        return df
    return df[df["Action"] == "ACTIVITY_TOKEN_SWAP"].reset_index(drop=True)


async def get_wallet_history(wallet: str, stop_after_ts: int | None = None):
    """
    Возвращает уникальные токены, которые КОШЕЛЕК ПОКУПАЛ (Token2) 
    на основе Solscan DeFi Export.
    Можно ограничить выборку до определенного timestamp (stop_after_ts).
    """
    # Загружаем все страницы (CSV -> pandas) с ограничением по дате
    df = await load_all_defi_activity(wallet, from_time=stop_after_ts)

    if df.empty:
        return []

    # Только свапы
    df = df[df["Action"] == "ACTIVITY_TOKEN_SWAP"]

    if df.empty:
        return []

    # Token2 — купленный токен
    tokens = df["Token2"].dropna().unique().tolist()

    # убираем SOL wrapper (опционально)
    tokens = [t for t in tokens if t != "So11111111111111111111111111111111111111111" and t.endswith('pump')]

    return tokens


# ------------------- Wallet analysis -------------------

def analyze_wallets_fifo(df: pd.DataFrame, token_mint=None) -> pd.DataFrame:
    df["Human Time"] = pd.to_datetime(df["Human Time"], utc=True)

    if token_mint is None:
        token_mint = df["Token1"].iloc[0]

    wallets = df["From"].unique()
    wallets_df = pd.DataFrame(wallets, columns=["wallet"])

    def wallet_metrics(wallet):
        df_wallet = df[df["From"] == wallet].sort_values("Human Time")
        buys_queue = deque()
        pnl = 0.0
        num_buys = 0
        num_sells = 0
        trades = []

        for _, row in df_wallet.iterrows():
            ts = int(pd.Timestamp(row["Human Time"]).timestamp())

            if row["Token2"] == token_mint:
                amount = float(row["Amount2"])
                usd_cost = float(row["Value"])
                buys_queue.append([amount, usd_cost])
                num_buys += 1
                trades.append({"side": "buy", "base_amount": amount, "usd_value": usd_cost, "ts": ts})
            elif row["Token1"] == token_mint:
                amount = float(row["Amount1"])
                usd_revenue = float(row["Value"])
                num_sells += 1
                trades.append({"side": "sell", "base_amount": amount, "usd_value": usd_revenue, "ts": ts})

                sell_left = amount
                while sell_left > 0 and buys_queue:
                    buy_amount, buy_cost = buys_queue.popleft()
                    if buy_amount <= sell_left:
                        part = buy_amount / amount
                        pnl += usd_revenue * part - buy_cost
                        sell_left -= buy_amount
                    else:
                        used_ratio = sell_left / buy_amount
                        pnl += usd_revenue * (sell_left / amount) - buy_cost * used_ratio
                        remaining_amount = buy_amount - sell_left
                        remaining_cost = buy_cost * (remaining_amount / buy_amount)
                        buys_queue.appendleft([remaining_amount, remaining_cost])
                        sell_left = 0

        return pd.Series({"num_buys": num_buys, "num_sells": num_sells, "pnl": pnl, "trades": trades})

    stats_df = wallets_df["wallet"].apply(wallet_metrics)
    wallets_df = pd.concat([wallets_df, stats_df], axis=1)
    wallets_df["trades"] = wallets_df["trades"].apply(lambda v: json.loads(v) if isinstance(v, str) else v)
    return wallets_df.sort_values("pnl", ascending=False).reset_index(drop=True)


def filter_wallets_by_min_buy(df_wallets: pd.DataFrame, min_usd: float):
    result = []
    for _, row in df_wallets.iterrows():
        trades = row["trades"]
        if isinstance(trades, str):
            trades = json.loads(trades)
        if any(t.get("side") == "buy" and t.get("usd_value", 0) >= min_usd for t in trades):
            result.append({"wallet": row["wallet"], "num_buys": row["num_buys"], "num_sells": row["num_sells"], "pnl": row["pnl"]})
    return result


# ------------------- Helius -------------------

async def fetch_token_metadata(token_mint: str) -> dict:
    url = f"{HELIUS_BASE_URL}?api-key={HELIUS_API_KEY}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAsset",
        "params": {"id": token_mint, "options": {"showFungible": True}}
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                raise Exception(f"Helius returned status {resp.status}")

            data = await resp.json()
            result = data.get("result")
            if not result:
                return {"token": token_mint, "symbol": None, "content": []}

            token_info = result.get("token_info", {})
            content = result.get("content", {})
            symbol = token_info.get("symbol") or token_mint
            return {"token": token_mint, "symbol": symbol, "content": content}

