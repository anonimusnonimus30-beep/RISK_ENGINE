#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


OUTPUT_DIR = Path("output")
DATA_DIR = Path("data")

SYMBOLS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "IWM": "IWM",
    "TLT": "TLT",
}

START_DATE = "2007-01-01"
OUT_OF_SAMPLE_DATE = "2018-01-01"
TRANSACTION_COST = 0.0005
INITIAL_CAPITAL = 1000.0


def normalize_download(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError(f"No se obtuvieron datos para {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})

    if "Date" not in df.columns:
        raise RuntimeError(f"No existe columna Date para {symbol}")

    price_column = "Adj Close" if "Adj Close" in df.columns else "Close"

    if price_column not in df.columns:
        raise RuntimeError(f"No existe precio para {symbol}")

    result = df[["Date", price_column]].copy()
    result.columns = ["Date", symbol]

    result["Date"] = pd.to_datetime(result["Date"]).dt.tz_localize(None)
    result[symbol] = pd.to_numeric(result[symbol], errors="coerce")

    result = (
        result.dropna()
        .drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .set_index("Date")
    )

    return result


def download_data() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)

    frames = []

    for name, ticker in SYMBOLS.items():
        print(f"Descargando {name} ({ticker})...")

        raw = yf.download(
            ticker,
            start=START_DATE,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        frame = normalize_download(raw, name)
        frame.to_csv(DATA_DIR / f"{name}.csv")
        frames.append(frame)

    data = pd.concat(frames, axis=1, join="inner").dropna()

    if len(data) < 500:
        raise RuntimeError(
            f"Historial común insuficiente: {len(data)} filas"
        )

    print(
        f"Periodo común: {data.index.min().date()} "
        f"a {data.index.max().date()} | {len(data)} sesiones"
    )

    return data


def rolling_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    return series.pct_change().rolling(window).std() * math.sqrt(252)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    for symbol in ["SPY", "QQQ"]:
        df[f"{symbol}_RET1"] = df[symbol].pct_change()
        df[f"{symbol}_RET5"] = df[symbol].pct_change(5)
        df[f"{symbol}_SMA50"] = df[symbol].rolling(50).mean()
        df[f"{symbol}_SMA100"] = df[symbol].rolling(100).mean()
        df[f"{symbol}_SMA200"] = df[symbol].rolling(200).mean()
        df[f"{symbol}_VOL20"] = rolling_volatility(df[symbol], 20)

    df["VIX_RET1"] = df["VIX"].pct_change()
    df["VIX_RET5"] = df["VIX"].pct_change(5)
    df["VIX_SMA20"] = df["VIX"].rolling(20).mean()

    df["IWM_SPY"] = df["IWM"] / df["SPY"]
    df["IWM_SPY_SMA50"] = df["IWM_SPY"].rolling(50).mean()
    df["IWM_SPY_RET20"] = df["IWM_SPY"].pct_change(20)

    df["TLT_RET20"] = df["TLT"].pct_change(20)

    return df.dropna()


def calculate_risk_score(
    row: pd.Series,
    symbol: str,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    price = float(row[symbol])
    ret1 = float(row[f"{symbol}_RET1"])
    ret5 = float(row[f"{symbol}_RET5"])
    vol20 = float(row[f"{symbol}_VOL20"])

    sma50 = float(row[f"{symbol}_SMA50"])
    sma100 = float(row[f"{symbol}_SMA100"])
    sma200 = float(row[f"{symbol}_SMA200"])

    vix = float(row["VIX"])
    vix_ret1 = float(row["VIX_RET1"])
    vix_ret5 = float(row["VIX_RET5"])

    if price < sma200:
        score += 25
        reasons.append("Precio bajo SMA200")

    if price < sma100:
        score += 15
        reasons.append("Precio bajo SMA100")

    if price < sma50:
        score += 10
        reasons.append("Precio bajo SMA50")

    if sma50 < sma200:
        score += 10
        reasons.append("SMA50 bajo SMA200")

    if ret1 <= -0.025:
        score += 15
        reasons.append("Caída diaria superior a 2.5%")
    elif ret1 <= -0.015:
        score += 8
        reasons.append("Caída diaria superior a 1.5%")

    if ret5 <= -0.06:
        score += 15
        reasons.append("Caída de cinco sesiones superior a 6%")
    elif ret5 <= -0.035:
        score += 8
        reasons.append("Caída de cinco sesiones superior a 3.5%")

    if vix >= 35:
        score += 25
        reasons.append("VIX superior a 35")
    elif vix >= 25:
        score += 15
        reasons.append("VIX superior a 25")
    elif vix >= 20:
        score += 7
        reasons.append("VIX superior a 20")

    if vix_ret1 >= 0.20:
        score += 15
        reasons.append("VIX sube más de 20% en una sesión")
    elif vix_ret1 >= 0.10:
        score += 8
        reasons.append("VIX sube más de 10% en una sesión")

    if vix_ret5 >= 0.30:
        score += 10
        reasons.append("VIX sube más de 30% en cinco sesiones")

    if row["IWM_SPY"] < row["IWM_SPY_SMA50"]:
        score += 7
        reasons.append("IWM débil frente a SPY")

    if row["IWM_SPY_RET20"] < -0.04:
        score += 8
        reasons.append("Deterioro fuerte de IWM/SPY")

    if vol20 >= 0.35:
        score += 10
        reasons.append("Volatilidad realizada elevada")
    elif vol20 >= 0.25:
        score += 5
        reasons.append("Volatilidad realizada moderadamente elevada")

    return min(score, 100), reasons


def risk_level(score: int) -> str:
    if score >= 70:
        return "CRITICAL"
    if score >= 55:
        return "HIGH"
    if score >= 35:
        return "WATCH"
    return "NORMAL"


def exposure_factor(score: int) -> float:
    if score >= 70:
        return 0.00
    if score >= 55:
        return 0.25
    if score >= 35:
        return 0.50
    return 1.00


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def annualized_return(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25

    if years <= 0:
        return 0.0

    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series) -> float:
    returns = returns.dropna()

    if returns.std() == 0:
        return 0.0

    return float(
        returns.mean() / returns.std() * math.sqrt(252)
    )


def backtest_symbol(
    features: pd.DataFrame,
    symbol: str,
) -> tuple[pd.DataFrame, dict]:
    rows = []

    for date, row in features.iterrows():
        score, reasons = calculate_risk_score(row, symbol)

        rows.append({
            "Date": date,
            "risk_score": score,
            "risk_level": risk_level(score),
            "exposure_factor": exposure_factor(score),
            "reasons": " | ".join(reasons),
        })

    result = pd.DataFrame(rows).set_index("Date")

    result["asset_return"] = features[symbol].pct_change()
    result["applied_exposure"] = result["exposure_factor"].shift(1).fillna(0)

    result["turnover"] = (
        result["applied_exposure"]
        .diff()
        .abs()
        .fillna(result["applied_exposure"].abs())
    )

    result["strategy_return"] = (
        result["asset_return"] * result["applied_exposure"]
        - result["turnover"] * TRANSACTION_COST
    )

    result["buy_hold_return"] = result["asset_return"]

    result["strategy_equity"] = (
        INITIAL_CAPITAL * (1 + result["strategy_return"]).cumprod()
    )

    result["buy_hold_equity"] = (
        INITIAL_CAPITAL * (1 + result["buy_hold_return"]).cumprod()
    )

    oos = result.loc[result.index >= OUT_OF_SAMPLE_DATE].copy()

    metrics = {
        "symbol": symbol,
        "start_date": str(oos.index.min().date()),
        "end_date": str(oos.index.max().date()),
        "initial_capital": INITIAL_CAPITAL,
        "strategy_final_value": round(
            float(oos["strategy_equity"].iloc[-1]), 2
        ),
        "buy_hold_final_value": round(
            float(oos["buy_hold_equity"].iloc[-1]), 2
        ),
        "strategy_cagr_pct": round(
            annualized_return(oos["strategy_equity"]) * 100, 2
        ),
        "buy_hold_cagr_pct": round(
            annualized_return(oos["buy_hold_equity"]) * 100, 2
        ),
        "strategy_sharpe": round(
            sharpe_ratio(oos["strategy_return"]), 3
        ),
        "buy_hold_sharpe": round(
            sharpe_ratio(oos["buy_hold_return"]), 3
        ),
        "strategy_max_drawdown_pct": round(
            max_drawdown(oos["strategy_equity"]) * 100, 2
        ),
        "buy_hold_max_drawdown_pct": round(
            max_drawdown(oos["buy_hold_equity"]) * 100, 2
        ),
        "average_exposure_pct": round(
            float(oos["applied_exposure"].mean()) * 100, 2
        ),
        "exposure_changes": int(
            (oos["applied_exposure"].diff().abs() > 0).sum()
        ),
    }

    return result, metrics


def build_current_state(
    features: pd.DataFrame,
    results: dict[str, pd.DataFrame],
) -> dict:
    latest_date = features.index[-1]

    symbol_states = {}

    for symbol in ["SPY", "QQQ"]:
        row = results[symbol].iloc[-1]

        symbol_states[symbol] = {
            "risk_score": int(row["risk_score"]),
            "risk_level": str(row["risk_level"]),
            "max_exposure_factor": float(row["exposure_factor"]),
            "reasons": (
                str(row["reasons"]).split(" | ")
                if row["reasons"]
                else []
            ),
        }

    market_score = max(
        symbol_states["SPY"]["risk_score"],
        symbol_states["QQQ"]["risk_score"],
    )

    state = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_date": str(latest_date.date()),
        "market_risk_score": market_score,
        "risk_level": risk_level(market_score),
        "block_new_entries": market_score >= 70,
        "SPY": symbol_states["SPY"],
        "QQQ": symbol_states["QQQ"],
    }

    return state


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = download_data()
    features = build_features(data)

    results = {}
    metrics = []

    for symbol in ["SPY", "QQQ"]:
        result, symbol_metrics = backtest_symbol(features, symbol)

        results[symbol] = result
        metrics.append(symbol_metrics)

        result.to_csv(
            OUTPUT_DIR / f"{symbol.lower()}_risk_backtest.csv"
        )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(
        OUTPUT_DIR / "backtest_metrics.csv",
        index=False,
    )

    state = build_current_state(features, results)

    (OUTPUT_DIR / "risk_state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== MÉTRICAS FUERA DE MUESTRA ===")
    print(metrics_df.to_string(index=False))

    print("\n=== ESTADO ACTUAL ===")
    print(json.dumps(state, indent=2, ensure_ascii=False))

    print("\nArchivos generados:")
    for path in sorted(OUTPUT_DIR.glob("*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
