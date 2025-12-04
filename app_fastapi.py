from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import numpy as np
from pathlib import Path
import time
from datetime import datetime, timedelta
import json

# 初始化FastAPI应用
app = FastAPI(title="加密货币价格预测")


# 请求模型
class PredictionRequest(BaseModel):
    crypto_name: str
    period: str


# 响应模型
class PredictionResponse(BaseModel):
    current_price: float
    predicted_price: float
    price_change: float
    period_days: int


# 图表数据请求模型
class ChartDataRequest(BaseModel):
    crypto_name: str


# 图表数据响应模型
class ChartDataResponse(BaseModel):
    dates: list
    prices: list


# 加密货币符号映射
CRYPTO_SYMBOLS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "ripple": "XRP",
    "cardano": "ADA",
    "solana": "SOL",
    "litecoin": "LTC",
    "dogecoin": "DOGE",
    "btc": "BTC",
    "eth": "ETH",
    "xrp": "XRP",
    "ada": "ADA",
    "sol": "SOL",
    "ltc": "LTC",
    "doge": "DOGE",
}


def get_binance_klines(symbol: str, interval: str = "1d", limit: int = 730):
    """从Binance获取历史K线数据"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": f"{symbol}USDT",
            "interval": interval,
            "limit": limit,
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()

        klines = response.json()
        if not klines or len(klines) < 100:
            return None

        # 提取收盘价
        prices = np.array([float(kline[4]) for kline in klines])
        return prices
    except Exception as e:
        print(f"Binance API错误: {e}")
        return None


def get_coingecko_data(crypto_name: str):
    """从CoinGecko获取历史数据"""
    try:
        url = (
            f"https://api.coingecko.com/api/v3/coins/{crypto_name.lower()}/market_chart"
        )
        params = {
            "vs_currency": "usd",
            "days": 730,
            "interval": "daily",
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, params=params, headers=headers, timeout=15)

        if response.status_code == 429:
            # 速率限制，等待后重试
            time.sleep(2)
            response = requests.get(url, params=params, headers=headers, timeout=15)

        response.raise_for_status()
        data = response.json()

        if "prices" not in data or len(data["prices"]) < 100:
            return None

        prices = np.array([price[1] for price in data["prices"]])
        return prices
    except Exception as e:
        print(f"CoinGecko API错误: {e}")
        return None


# 基于过去两年数据的价格预测函数
def predict_price(crypto_name: str, period_days: int):
    """
    使用真实API数据进行预测
    优先使用Binance，备选CoinGecko
    """
    try:
        crypto_lower = crypto_name.lower()
        prices = None

        # 首先尝试Binance API
        if crypto_lower in CRYPTO_SYMBOLS:
            symbol = CRYPTO_SYMBOLS[crypto_lower]
            prices = get_binance_klines(symbol, limit=730)

        # 如果Binance失败，尝试CoinGecko
        if prices is None:
            prices = get_coingecko_data(crypto_name)

        # 如果都失败，报错
        if prices is None:
            raise ValueError(
                f"无法获取 {crypto_name} 的数据。请检查加密货币名称是否正确。"
            )

        if len(prices) < 100:
            raise ValueError("数据不足，无法进行可靠预测")

        # 获取当前价格
        current_price = prices[-1]

        # 使用多项式回归进行预测（使用2次多项式，避免过度拟合）
        x = np.arange(len(prices))

        # 多项式拟合
        coefficients = np.polyfit(x, prices, 2)
        poly = np.poly1d(coefficients)

        # 计算预测未来的日期
        future_x = len(prices) + period_days
        predicted_price = poly(future_x)

        # 确保预测价格为正数
        if predicted_price <= 0:
            # 如果多项式预测为负，使用线性回归
            linear_coef = np.polyfit(x[-180:], prices[-180:], 1)
            linear_poly = np.poly1d(linear_coef)
            predicted_price = linear_poly(future_x)

            # 如果线性也为负，取当前价格的80%
            if predicted_price <= 0:
                predicted_price = current_price * 0.8

        # 计算价格变化百分比
        price_change = ((predicted_price - current_price) / current_price) * 100

        return {
            "current_price": round(current_price, 2),
            "predicted_price": round(max(predicted_price, 0.01), 2),
            "price_change": round(price_change, 2),
            "period_days": period_days,
        }

    except Exception as e:
        raise ValueError(f"预测错误: {str(e)}")


@app.get("/")
async def root():
    """返回主页HTML"""
    html_file = Path(__file__).parent / "templates" / "index.html"
    return FileResponse(html_file, media_type="text/html; charset=utf-8")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """预测加密货币价格"""

    # 验证输入
    crypto_name = request.crypto_name.strip()
    period = request.period.strip()

    if not crypto_name:
        raise HTTPException(status_code=400, detail="请输入加密货币名称")

    if not period:
        raise HTTPException(status_code=400, detail="请选择预测期间")

    # 将期间转换为天数
    period_map = {"1周": 7, "1个月": 30, "3个月": 90}

    if period not in period_map:
        raise HTTPException(status_code=400, detail="无效的预测期间")

    period_days = period_map[period]

    try:
        # 进行预测
        result = predict_price(crypto_name, period_days)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@app.post("/chart-data")
async def get_chart_data(data: dict = Body(...)):
    """获取图表数据"""
    try:
        crypto_name = data.get("crypto_name", "").strip()

        print(f"DEBUG: 收到请求: {data}")

        if not crypto_name:
            raise HTTPException(status_code=400, detail="请输入加密货币名称")

        # 获取价格数据
        prices = None
        crypto_lower = crypto_name.lower()

        # 首先尝试Binance API
        if crypto_lower in CRYPTO_SYMBOLS:
            symbol = CRYPTO_SYMBOLS[crypto_lower]
            prices = get_binance_klines(symbol, limit=730)

        # 如果Binance失败，尝试CoinGecko
        if prices is None:
            prices = get_coingecko_data(crypto_name)

        if prices is None:
            raise ValueError(f"无法获取 {crypto_name} 的数据")

        # 生成日期标签（最后730天）
        today = datetime.now()
        dates = [
            (today - timedelta(days=730 - i)).strftime("%Y-%m-%d")
            for i in range(len(prices))
        ]

        # 只显示关键日期以避免过于拥挤
        display_dates = []
        display_prices = []
        step = max(1, len(prices) // 30)  # 最多显示30个标签

        for i in range(0, len(prices), step):
            display_dates.append(dates[i])
            display_prices.append(float(prices[i]))

        # 确保最后一个点显示
        if len(prices) - 1 not in range(0, len(prices), step):
            display_dates.append(dates[-1])
            display_prices.append(float(prices[-1]))

        result = {
            "dates": display_dates,
            "prices": display_prices,
        }
        print(
            f"DEBUG: 返回数据 - 日期数: {len(display_dates)}, 价格数: {len(display_prices)}"
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取图表数据失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5000)
