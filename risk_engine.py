#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

SYMBOLS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "IWM": "IWM",
    "TLT": "TLT",
}

DOWNLOAD_START = "2006-01-01"

TRAIN_START = "2007-01-01"
TRAIN_END = "2014-12-31"

VALIDATION_START = "2015-01-01"
VALIDATION_END = "2017-12-31"

TEST_START = "2018-01-01"

INITIAL_CAPITAL = 1000.0
TRANSACTION_COST = 0.0005


@dataclass(frozen=True)
class RiskConfig:
    name: str

    watch_threshold: int
    high_threshold: int
    critical_threshold: int

    worsen_confirmation_days: int
    recovery_confirmation_days: int
    cooldown_days: int

    normal_exposure: float
    watch_exposure: float
    high_exposure: float
    critical_exposure: float


CANDIDATES = [
    RiskConfig(
        name="balanced",
        watch_threshold=35,
        high_threshold=55,
        critical_threshold=70,
        worsen_confirmation_days=2,
        recovery_confirmation_days=4,
        cooldown_days=5,
        normal_exposure=1.00,
        watch_exposure=0.75,
        high_exposure=0.40,
        critical_exposure=0.00,
    ),
    RiskConfig(
        name="selective",
        watch_threshold=40,
        high_threshold=60,
        critical_threshold=75,
        worsen_confirmation_days=2,
        recovery_confirmation_days=4,
        cooldown_days=5,
        normal_exposure=1.00,
        watch_exposure=0.80,
        high_exposure=0.50,
        critical_exposure=0.10,
    ),
    RiskConfig(
        name="slow_confirmation",
        watch_threshold=35,
        high_threshold=55,
        critical_threshold=70,
        worsen_confirmation_days=3,
        recovery_confirmation_days=5,
        cooldown_days=7,
        normal_exposure=1.00,
        watch_exposure=0.75,
        high_exposure=0.40,
        critical_exposure=0.00,
    ),
    RiskConfig(
        name="partial_defense",
        watch_threshold=35,
        high_threshold=55,
        critical_threshold=70,
        worsen_confirmation_days=2,
        recovery_confirmation_days=5,
        cooldown_days=7,
        normal_exposure=1.00,
        watch_exposure=0.80,
        high_exposure=0.50,
        critical_exposure=0.20,
    ),
    RiskConfig(
        name="strict",
        watch_threshold=45,
        high_threshold=65,
        critical_threshold=80,
        worsen_confirmation_days=2,
        recovery_confirmation_days=4,
        cooldown_days=5,
        normal_exposure=1.00,
        watch_exposure=0.85,
        high_exposure=0.55,
        critical_exposure=0.20,
    ),
]


def download_symbol(name: str, ticker: str) -> pd.DataFrame:
    print(f"Descargando {name} ({ticker})...")

    raw = yf.download(
        ticker,
        start=DOWNLOAD_START,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(f"No se obtuvieron datos para {name}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [column[0] for column in raw.columns]

    price_column = "Adj Close" if "Adj Close" in raw.columns else "Close"

    if price_column not in raw.columns:
        raise RuntimeError(f"No se encontró precio para {name}")

    frame = raw[[price_column]].copy()
    frame.columns = [name]

    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame[name] = pd.to_numeric(frame[name], errors="coerce")

    frame = (
        frame.dropna()
        .loc[lambda value: value[name] > 0]
        .sort_index()
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_DIR / f"{name}.csv")

    return frame


def download_data() -> pd.DataFrame:
    frames = [
        download_symbol(name, ticker)
        for name, ticker in SYMBOLS.items()
    ]

    data = pd.concat(frames, axis=1, join="inner")
    data = data.dropna().sort_index()

    if len(data) < 2000:
        raise RuntimeError(
            f"Historial común insuficiente: {len(data)} sesiones"
        )

    print(
        f"Periodo común: {data.index.min().date()} "
        f"a {data.index.max().date()} | "
        f"{len(data)} sesiones"
    )

    return data


def annualized_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * math.sqrt(252)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.copy()

    for symbol in ("SPY", "QQQ"):
        returns = features[symbol].pct_change()

        features[f"{symbol}_RET1"] = returns
        features[f"{symbol}_RET3"] = features[symbol].pct_change(3)
        features[f"{symbol}_RET5"] = features[symbol].pct_change(5)
        features[f"{symbol}_RET20"] = features[symbol].pct_change(20)

        features[f"{symbol}_SMA20"] = features[symbol].rolling(20).mean()
        features[f"{symbol}_SMA50"] = features[symbol].rolling(50).mean()
        features[f"{symbol}_SMA100"] = features[symbol].rolling(100).mean()
        features[f"{symbol}_SMA200"] = features[symbol].rolling(200).mean()

        features[f"{symbol}_VOL20"] = annualized_volatility(
            returns,
            20,
        )

        features[f"{symbol}_VOL60"] = annualized_volatility(
            returns,
            60,
        )

        rolling_peak = features[symbol].rolling(63).max()

        features[f"{symbol}_DD63"] = (
            features[symbol] / rolling_peak - 1.0
        )

    features["VIX_RET1"] = features["VIX"].pct_change()
    features["VIX_RET3"] = features["VIX"].pct_change(3)
    features["VIX_RET5"] = features["VIX"].pct_change(5)

    features["VIX_SMA10"] = features["VIX"].rolling(10).mean()
    features["VIX_SMA20"] = features["VIX"].rolling(20).mean()

    features["IWM_SPY"] = features["IWM"] / features["SPY"]
    features["IWM_SPY_SMA20"] = features["IWM_SPY"].rolling(20).mean()
    features["IWM_SPY_SMA50"] = features["IWM_SPY"].rolling(50).mean()
    features["IWM_SPY_RET20"] = features["IWM_SPY"].pct_change(20)

    features["TLT_RET5"] = features["TLT"].pct_change(5)
    features["TLT_RET20"] = features["TLT"].pct_change(20)

    return features.dropna()


def calculate_raw_score(
    row: pd.Series,
    symbol: str,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    price = float(row[symbol])

    sma20 = float(row[f"{symbol}_SMA20"])
    sma50 = float(row[f"{symbol}_SMA50"])
    sma100 = float(row[f"{symbol}_SMA100"])
    sma200 = float(row[f"{symbol}_SMA200"])

    ret1 = float(row[f"{symbol}_RET1"])
    ret3 = float(row[f"{symbol}_RET3"])
    ret5 = float(row[f"{symbol}_RET5"])
    ret20 = float(row[f"{symbol}_RET20"])

    vol20 = float(row[f"{symbol}_VOL20"])
    vol60 = float(row[f"{symbol}_VOL60"])
    drawdown63 = float(row[f"{symbol}_DD63"])

    vix = float(row["VIX"])
    vix_ret1 = float(row["VIX_RET1"])
    vix_ret3 = float(row["VIX_RET3"])
    vix_ret5 = float(row["VIX_RET5"])

    iwm_spy = float(row["IWM_SPY"])
    iwm_spy_sma20 = float(row["IWM_SPY_SMA20"])
    iwm_spy_sma50 = float(row["IWM_SPY_SMA50"])
    iwm_spy_ret20 = float(row["IWM_SPY_RET20"])

    tlt_ret5 = float(row["TLT_RET5"])

    if price < sma200:
        score += 22
        reasons.append("Precio bajo SMA200")

    if price < sma100:
        score += 13
        reasons.append("Precio bajo SMA100")

    if price < sma50:
        score += 8
        reasons.append("Precio bajo SMA50")

    if price < sma20:
        score += 4
        reasons.append("Precio bajo SMA20")

    if sma50 < sma200:
        score += 8
        reasons.append("SMA50 bajo SMA200")

    if ret1 <= -0.030:
        score += 14
        reasons.append("Caída diaria superior a 3%")
    elif ret1 <= -0.020:
        score += 8
        reasons.append("Caída diaria superior a 2%")

    if ret3 <= -0.050:
        score += 12
        reasons.append("Caída de tres sesiones superior a 5%")
    elif ret3 <= -0.035:
        score += 7
        reasons.append("Caída de tres sesiones superior a 3.5%")

    if ret5 <= -0.070:
        score += 12
        reasons.append("Caída de cinco sesiones superior a 7%")
    elif ret5 <= -0.045:
        score += 7
        reasons.append("Caída de cinco sesiones superior a 4.5%")

    if ret20 <= -0.120:
        score += 10
        reasons.append("Momentum mensual extremadamente negativo")
    elif ret20 <= -0.080:
        score += 5
        reasons.append("Momentum mensual negativo")

    if drawdown63 <= -0.150:
        score += 10
        reasons.append("Drawdown trimestral superior a 15%")
    elif drawdown63 <= -0.100:
        score += 5
        reasons.append("Drawdown trimestral superior a 10%")

    if vix >= 40:
        score += 22
        reasons.append("VIX superior a 40")
    elif vix >= 30:
        score += 15
        reasons.append("VIX superior a 30")
    elif vix >= 24:
        score += 9
        reasons.append("VIX superior a 24")
    elif vix >= 20:
        score += 4
        reasons.append("VIX superior a 20")

    if vix_ret1 >= 0.25:
        score += 13
        reasons.append("VIX sube más de 25% en una sesión")
    elif vix_ret1 >= 0.15:
        score += 7
        reasons.append("VIX sube más de 15% en una sesión")

    if vix_ret3 >= 0.35:
        score += 10
        reasons.append("VIX sube más de 35% en tres sesiones")

    if vix_ret5 >= 0.50:
        score += 8
        reasons.append("VIX sube más de 50% en cinco sesiones")

    if iwm_spy < iwm_spy_sma20:
        score += 3
        reasons.append("IWM débil frente a SPY")

    if iwm_spy < iwm_spy_sma50:
        score += 4
        reasons.append("IWM/SPY bajo su media de 50 sesiones")

    if iwm_spy_ret20 <= -0.050:
        score += 7
        reasons.append("Deterioro fuerte de amplitud")

    if vol20 >= 0.35:
        score += 9
        reasons.append("Volatilidad realizada superior a 35%")
    elif vol20 >= 0.27:
        score += 5
        reasons.append("Volatilidad realizada elevada")

    if vol20 >= vol60 * 1.45:
        score += 7
        reasons.append("Aceleración brusca de volatilidad")

    if ret3 < -0.03 and tlt_ret5 < -0.02:
        score += 5
        reasons.append("Acciones y bonos caen simultáneamente")

    return min(int(score), 100), reasons


def raw_level(score: int, config: RiskConfig) -> int:
    if score >= config.critical_threshold:
        return 3

    if score >= config.high_threshold:
        return 2

    if score >= config.watch_threshold:
        return 1

    return 0


def level_name(level: int) -> str:
    return {
        0: "NORMAL",
        1: "WATCH",
        2: "HIGH",
        3: "CRITICAL",
    }[level]


def level_exposure(level: int, config: RiskConfig) -> float:
    return {
        0: config.normal_exposure,
        1: config.watch_exposure,
        2: config.high_exposure,
        3: config.critical_exposure,
    }[level]


def apply_state_machine(
    scores: pd.Series,
    config: RiskConfig,
) -> pd.DataFrame:
    current_level = 0
    candidate_level = 0
    candidate_count = 0
    cooldown_remaining = 0

    records: list[dict[str, Any]] = []

    for date, score_value in scores.items():
        score = int(score_value)
        desired_level = raw_level(score, config)

        changed = False

        if desired_level > current_level:
            required = config.worsen_confirmation_days

            if desired_level != candidate_level:
                candidate_level = desired_level
                candidate_count = 1
            else:
                candidate_count += 1

            if candidate_count >= required:
                current_level = desired_level
                cooldown_remaining = config.cooldown_days
                candidate_level = current_level
                candidate_count = 0
                changed = True

        elif desired_level < current_level:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                candidate_level = current_level
                candidate_count = 0
            else:
                required = config.recovery_confirmation_days

                if desired_level != candidate_level:
                    candidate_level = desired_level
                    candidate_count = 1
                else:
                    candidate_count += 1

                if candidate_count >= required:
                    current_level = max(
                        desired_level,
                        current_level - 1,
                    )

                    cooldown_remaining = config.cooldown_days
                    candidate_level = current_level
                    candidate_count = 0
                    changed = True

        else:
            candidate_level = current_level
            candidate_count = 0

            if cooldown_remaining > 0:
                cooldown_remaining -= 1

        records.append({
            "Date": date,
            "confirmed_level": current_level,
            "confirmed_level_name": level_name(current_level),
            "exposure_factor": level_exposure(
                current_level,
                config,
            ),
            "level_changed": changed,
            "cooldown_remaining": cooldown_remaining,
        })

    return pd.DataFrame(records).set_index("Date")


def build_score_series(
    features: pd.DataFrame,
    symbol: str,
) -> tuple[pd.Series, dict[pd.Timestamp, list[str]]]:
    scores: dict[pd.Timestamp, int] = {}
    reasons: dict[pd.Timestamp, list[str]] = {}

    for date, row in features.iterrows():
        score, row_reasons = calculate_raw_score(row, symbol)
        scores[date] = score
        reasons[date] = row_reasons

    return pd.Series(scores, name="risk_score"), reasons


def max_drawdown(equity: pd.Series) -> float:
    equity = equity.dropna()

    if equity.empty:
        return 0.0

    peak = equity.cummax()
    drawdown = equity / peak - 1.0

    return float(drawdown.min())


def cagr(
    final_value: float,
    initial_value: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> float:
    years = (end_date - start_date).days / 365.25

    if years <= 0 or initial_value <= 0:
        return 0.0

    return float(
        (final_value / initial_value) ** (1.0 / years) - 1.0
    )


def sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()

    std = float(returns.std())

    if not np.isfinite(std) or std <= 0:
        return 0.0

    return float(
        returns.mean() / std * math.sqrt(252)
    )


def backtest_config(
    features: pd.DataFrame,
    symbol: str,
    scores: pd.Series,
    reasons: dict[pd.Timestamp, list[str]],
    config: RiskConfig,
) -> pd.DataFrame:
    state = apply_state_machine(scores, config)

    result = state.copy()
    result["risk_score"] = scores
    result["reasons"] = [
        " | ".join(reasons[index])
        for index in result.index
    ]

    result["asset_return"] = features[symbol].pct_change()

    # La exposición decidida al cierre se aplica al día siguiente.
    result["applied_exposure"] = (
        result["exposure_factor"]
        .shift(1)
        .fillna(config.normal_exposure)
    )

    result["turnover"] = (
        result["applied_exposure"]
        .diff()
        .abs()
        .fillna(0.0)
    )

    result["strategy_return"] = (
        result["asset_return"] * result["applied_exposure"]
        - result["turnover"] * TRANSACTION_COST
    )

    result["buy_hold_return"] = result["asset_return"]

    return result.dropna()


def period_metrics(
    result: pd.DataFrame,
    start: str,
    end: str | None,
    symbol: str,
    config_name: str,
    period_name: str,
) -> dict[str, Any]:
    period = result.loc[result.index >= pd.Timestamp(start)].copy()

    if end is not None:
        period = period.loc[
            period.index <= pd.Timestamp(end)
        ].copy()

    if period.empty:
        raise RuntimeError(
            f"No hay datos para {symbol}, periodo {period_name}"
        )

    strategy_equity = (
        INITIAL_CAPITAL
        * (1.0 + period["strategy_return"].fillna(0.0)).cumprod()
    )

    buy_hold_equity = (
        INITIAL_CAPITAL
        * (1.0 + period["buy_hold_return"].fillna(0.0)).cumprod()
    )

    start_date = period.index[0]
    end_date = period.index[-1]

    strategy_final = float(strategy_equity.iloc[-1])
    buy_hold_final = float(buy_hold_equity.iloc[-1])

    years = max(
        (end_date - start_date).days / 365.25,
        1 / 365.25,
    )

    changes = int(
        (period["applied_exposure"].diff().abs() > 1e-12).sum()
    )

    return {
        "symbol": symbol,
        "config": config_name,
        "period": period_name,
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "initial_capital": INITIAL_CAPITAL,
        "strategy_final_value": round(strategy_final, 2),
        "buy_hold_final_value": round(buy_hold_final, 2),
        "strategy_cagr_pct": round(
            cagr(
                strategy_final,
                INITIAL_CAPITAL,
                start_date,
                end_date,
            ) * 100,
            3,
        ),
        "buy_hold_cagr_pct": round(
            cagr(
                buy_hold_final,
                INITIAL_CAPITAL,
                start_date,
                end_date,
            ) * 100,
            3,
        ),
        "strategy_sharpe": round(
            sharpe(period["strategy_return"]),
            4,
        ),
        "buy_hold_sharpe": round(
            sharpe(period["buy_hold_return"]),
            4,
        ),
        "strategy_max_drawdown_pct": round(
            max_drawdown(strategy_equity) * 100,
            3,
        ),
        "buy_hold_max_drawdown_pct": round(
            max_drawdown(buy_hold_equity) * 100,
            3,
        ),
        "average_exposure_pct": round(
            float(period["applied_exposure"].mean()) * 100,
            3,
        ),
        "exposure_changes": changes,
        "changes_per_year": round(changes / years, 3),
        "total_turnover": round(
            float(period["turnover"].sum()),
            4,
        ),
    }


def validation_objective(metrics: dict[str, Any]) -> float:
    strategy_cagr = metrics["strategy_cagr_pct"] / 100.0
    buy_hold_cagr = metrics["buy_hold_cagr_pct"] / 100.0

    strategy_dd = metrics["strategy_max_drawdown_pct"] / 100.0
    buy_hold_dd = metrics["buy_hold_max_drawdown_pct"] / 100.0

    drawdown_improvement = strategy_dd - buy_hold_dd
    return_capture = (
        strategy_cagr / buy_hold_cagr
        if buy_hold_cagr > 0
        else 0.0
    )

    changes_penalty = metrics["changes_per_year"] * 0.012

    score = (
        metrics["strategy_sharpe"]
        + drawdown_improvement * 2.5
        + return_capture * 0.35
        - changes_penalty
    )

    # Se rechazan modelos que apenas reduzcan el drawdown.
    if drawdown_improvement < 0.04:
        score -= 1.0

    # Se rechazan modelos con rotación excesiva.
    if metrics["changes_per_year"] > 18:
        score -= 0.75

    return float(score)


def choose_config(
    features: pd.DataFrame,
    symbol: str,
) -> tuple[RiskConfig, pd.DataFrame, pd.DataFrame]:
    scores, reasons = build_score_series(features, symbol)

    candidates: list[dict[str, Any]] = []
    result_by_name: dict[str, pd.DataFrame] = {}

    for config in CANDIDATES:
        result = backtest_config(
            features,
            symbol,
            scores,
            reasons,
            config,
        )

        result_by_name[config.name] = result

        validation = period_metrics(
            result,
            VALIDATION_START,
            VALIDATION_END,
            symbol,
            config.name,
            "validation",
        )

        validation["selection_score"] = round(
            validation_objective(validation),
            6,
        )

        candidates.append(validation)

    comparison = pd.DataFrame(candidates).sort_values(
        "selection_score",
        ascending=False,
    )

    selected_name = str(comparison.iloc[0]["config"])

    selected_config = next(
        config
        for config in CANDIDATES
        if config.name == selected_name
    )

    return (
        selected_config,
        result_by_name[selected_name],
        comparison,
    )


def build_state(
    selected: dict[str, RiskConfig],
    results: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    symbol_states: dict[str, Any] = {}

    for symbol in ("SPY", "QQQ"):
        row = results[symbol].iloc[-1]
        config = selected[symbol]

        reasons = (
            str(row["reasons"]).split(" | ")
            if str(row["reasons"]).strip()
            else []
        )

        symbol_states[symbol] = {
            "selected_config": config.name,
            "risk_score": int(row["risk_score"]),
            "risk_level": str(row["confirmed_level_name"]),
            "raw_risk_level": level_name(
                raw_level(int(row["risk_score"]), config)
            ),
            "max_exposure_factor": float(
                row["exposure_factor"]
            ),
            "cooldown_remaining": int(
                row["cooldown_remaining"]
            ),
            "reasons": reasons,
        }

    market_score = max(
        symbol_states["SPY"]["risk_score"],
        symbol_states["QQQ"]["risk_score"],
    )

    levels = {
        symbol_states["SPY"]["risk_level"],
        symbol_states["QQQ"]["risk_level"],
    }

    if "CRITICAL" in levels:
        market_level = "CRITICAL"
    elif "HIGH" in levels:
        market_level = "HIGH"
    elif "WATCH" in levels:
        market_level = "WATCH"
    else:
        market_level = "NORMAL"

    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "market_date": str(results["SPY"].index[-1].date()),
        "market_risk_score": int(market_score),
        "risk_level": market_level,
        "block_new_entries": market_level == "CRITICAL",
        "SPY": symbol_states["SPY"],
        "QQQ": symbol_states["QQQ"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = download_data()
    features = build_features(data)

    selected: dict[str, RiskConfig] = {}
    results: dict[str, pd.DataFrame] = {}

    validation_tables: list[pd.DataFrame] = []
    final_metrics: list[dict[str, Any]] = []

    for symbol in ("SPY", "QQQ"):
        config, result, comparison = choose_config(
            features,
            symbol,
        )

        selected[symbol] = config
        results[symbol] = result

        comparison.insert(0, "selection_symbol", symbol)
        validation_tables.append(comparison)

        test_metrics = period_metrics(
            result,
            TEST_START,
            None,
            symbol,
            config.name,
            "out_of_sample_test",
        )

        final_metrics.append(test_metrics)

        result.to_csv(
            OUTPUT_DIR / f"{symbol.lower()}_risk_backtest_v2.csv"
        )

    validation_df = pd.concat(
        validation_tables,
        ignore_index=True,
    )

    validation_df.to_csv(
        OUTPUT_DIR / "model_selection_validation.csv",
        index=False,
    )

    metrics_df = pd.DataFrame(final_metrics)

    metrics_df.to_csv(
        OUTPUT_DIR / "backtest_metrics_final.csv",
        index=False,
    )

    selected_payload = {
        symbol: asdict(config)
        for symbol, config in selected.items()
    }

    (
        OUTPUT_DIR / "selected_config.json"
    ).write_text(
        json.dumps(
            selected_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    state = build_state(selected, results)

    (
        OUTPUT_DIR / "risk_state.json"
    ).write_text(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CONFIGURACIONES SELECCIONADAS ===")

    for symbol, config in selected.items():
        print(f"{symbol}: {config.name}")

    print("\n=== PRUEBA FUERA DE MUESTRA 2018-ACTUALIDAD ===")
    print(metrics_df.to_string(index=False))

    print("\n=== ESTADO ACTUAL ===")
    print(
        json.dumps(
            state,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nArchivos generados:")

    for path in sorted(OUTPUT_DIR.glob("*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
