import os
import aiohttp
import asyncio
from dotenv import load_dotenv

from db import token_to_table_name

load_dotenv()
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")
HELIUS_BASE_URL = "https://mainnet.helius-rpc.com/"

async def get_token_safe_name(token_mint: str) -> str:
    url = f"{HELIUS_BASE_URL}?api-key={HELIUS_API_KEY}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAsset",
        "params": {
            "id": token_mint,
            "options": {"showFungible": True}
        }
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                print(f'if resp.status != 200: {token_to_table_name(token_mint)}')
                return token_to_table_name(token_mint)
            data = await resp.json()
            result = data.get("result")
            if not result:
                print(f'if not result: {token_to_table_name(token_mint)}')
                return token_to_table_name(token_mint)
            metadata = result.get("content", {}).get("metadata", {})
            name = metadata.get("symbol") or metadata.get("name") or token_mint
            safe_name = token_to_table_name(name)
            print(safe_name)
            return safe_name

async def main():
    token_mint = "QxSK4nJG2TQYoJoTyjjhcePAy1vgE9HbD6inTKWpump"  # пример
    info = await get_token_safe_name(token_mint)
    print("Info:", info)

if __name__ == "__main__":
    asyncio.run(main())
