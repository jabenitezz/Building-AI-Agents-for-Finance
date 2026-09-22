"""Phase B v2: compare committee-only and adversarially validated signals."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf

HERE = Path(__file__).resolve().parent
DEFAULT_SIGNALS = HERE / "output" / "signals.csv"
DEFAULT_OUTPUT_DIR = HERE / "output" / "backtest"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest v2: Committee Only vs Committee + Adversarial."
    )
    p.add_argument("--signals", default=str(DEFAULT_SIGNALS))
    p.add_argument("--initial-cash", type=float, default=100_000.0)
    p.add_argument("--fees-bps", type=float, default=0.0)
    p.add_argument("--slippage-bps", type=float, default=0.0)
    p.add_argument("--evaluation-end", default=None)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument(
        "--equal-buy-size-pct",
        type=float,
        default=None,
        help=(
            "Tamaño fijo para el diagnóstico Equal-Size BUY. "
            "Por defecto usa la media de SIZE_PCT de los BUY finales del periodo."
        ),
    )
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return p.parse_args()


def _naive_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def load_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el CSV de señales: {path}")

    df = pd.read_csv(path)
    required = {
        "decision_date",
        "execution_date",
        "execution_open_price",
        "ticker",
        "position_semantics",
        "pm_action",
        "pm_size_pct",
        "committee_action",
        "committee_size_pct",
        "action",
        "size_pct",
        "final_decision",
        "debate_status",
        "bull_conviction",
        "bear_conviction",
        "judge_action",
        "judge_confidence",
        "validation_status",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"signals.csv no contiene columnas requeridas de v2: {missing}")

    for col in ("decision_date", "execution_date"):
        df[col] = pd.to_datetime(df[col], errors="raise").dt.normalize()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for col in ("pm_action", "committee_action", "action"):
        df[col] = df[col].fillna("").astype(str).str.upper().str.strip()
    for col in (
        "pm_size_pct",
        "committee_size_pct",
        "size_pct",
        "execution_open_price",
    ):
        df[col] = pd.to_numeric(df[col], errors="raise")

    expected = "target_pct_total_nav_each_rebalance"
    bad = df["position_semantics"].astype(str).ne(expected)
    if bad.any():
        found = sorted(set(df.loc[bad, "position_semantics"].astype(str)))
        raise ValueError(
            f"Semántica incorrecta. Esperado={expected!r}; encontrado={found}."
        )

    def validate_pair(action_col: str, size_col: str) -> None:
        invalid = sorted(set(df[action_col]) - {"BUY", "HOLD", "SELL"})
        if invalid:
            raise ValueError(f"{action_col}: acciones no soportadas: {invalid}")
        if (df[size_col] < 0).any():
            raise ValueError(f"{size_col} no puede ser negativo.")
        bad_non_buy = df[action_col].isin(["HOLD", "SELL"]) & (df[size_col] != 0)
        if bad_non_buy.any():
            raise ValueError(f"{action_col}: HOLD/SELL deben tener {size_col}=0.")
        if (df.loc[df[action_col] == "BUY", size_col] <= 0).any():
            raise ValueError(f"{action_col}: BUY debe tener {size_col} > 0.")

    validate_pair("committee_action", "committee_size_pct")
    validate_pair("action", "size_pct")

    dup = df.duplicated(["execution_date", "ticker"], keep=False)
    if dup.any():
        bad_rows = df.loc[dup, ["execution_date", "ticker", "action"]]
        raise ValueError(
            "Hay señales duplicadas para execution_date/ticker:\n"
            + bad_rows.to_string(index=False)
        )

    return df.sort_values(["execution_date", "ticker"]).reset_index(drop=True)


def infer_evaluation_end(
    signals: pd.DataFrame,
    explicit: str | None,
) -> pd.Timestamp:
    if explicit:
        end = pd.Timestamp(explicit).normalize()
    else:
        last_decision = signals["decision_date"].max()
        end = last_decision + pd.offsets.MonthEnd(1)

    if end < signals["execution_date"].max():
        raise ValueError(
            "evaluation_end no puede ser anterior a la última execution_date."
        )
    return pd.Timestamp(end).normalize()


def fetch_ohlc(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}

    for ticker in tickers:
        df = yf.Ticker(ticker).history(
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=False,
        )
        if df.empty or "Open" not in df or "Close" not in df:
            raise RuntimeError(f"Sin OHLC histórico suficiente para {ticker}.")
        df = df.copy()
        df.index = _naive_index(df.index)
        df = df[(df.index >= start) & (df.index <= end)]
        opens[ticker] = df["Open"].astype(float)
        closes[ticker] = df["Close"].astype(float)

    open_df = pd.concat(opens, axis=1, join="inner").sort_index()
    close_df = pd.concat(closes, axis=1, join="inner").sort_index()
    common = open_df.index.intersection(close_df.index)
    open_df = open_df.loc[common, tickers]
    close_df = close_df.loc[common, tickers]

    if open_df.empty:
        raise RuntimeError("No hay sesiones comunes para los activos del backtest.")
    return open_df, close_df


def validate_execution_prices(
    signals: pd.DataFrame,
    open_df: pd.DataFrame,
    warn_pct: float = 0.005,
) -> None:
    for row in signals.itertuples(index=False):
        d = pd.Timestamp(row.execution_date)
        if d not in open_df.index:
            raise RuntimeError(
                f"{row.ticker}: execution_date {d.date()} no es sesión disponible."
            )
        fetched = float(open_df.at[d, row.ticker])
        recorded = float(row.execution_open_price)
        if recorded <= 0:
            raise ValueError(
                f"{row.ticker} {d.date()}: execution_open_price inválido {recorded}."
            )
        rel = abs(fetched / recorded - 1.0)
        if rel > warn_pct:
            print(
                f"[WARN] {row.ticker} {d.date()}: open CSV={recorded:.4f} "
                f"vs Yahoo={fetched:.4f} ({rel:.2%}). Se usa CSV."
            )


def build_target_frames(
    signals: pd.DataFrame,
    index: pd.DatetimeIndex,
    tickers: list[str],
    open_df: pd.DataFrame,
    *,
    action_col: str,
    size_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = pd.DataFrame(np.nan, index=index, columns=tickers, dtype=float)
    order_prices = open_df.copy().astype(float)

    for row in signals.itertuples(index=False):
        d = pd.Timestamp(row.execution_date)
        ticker = str(row.ticker)
        if d not in targets.index:
            raise RuntimeError(
                f"execution_date {d.date()} de {ticker} no está en precios."
            )
        action = str(getattr(row, action_col))
        size_pct = float(getattr(row, size_col))
        targets.at[d, ticker] = size_pct / 100.0 if action == "BUY" else 0.0
        order_prices.at[d, ticker] = float(row.execution_open_price)

    return targets, order_prices


def build_mas_portfolio(
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    signals: pd.DataFrame,
    initial_cash: float,
    fees: float,
    slippage: float,
    *,
    action_col: str,
    size_col: str,
) -> vbt.Portfolio:
    tickers = list(close_df.columns)
    targets, order_prices = build_target_frames(
        signals,
        close_df.index,
        tickers,
        open_df,
        action_col=action_col,
        size_col=size_col,
    )
    return vbt.Portfolio.from_orders(
        close_df,
        size=targets,
        size_type="targetpercent",
        direction="longonly",
        price=order_prices,
        init_cash=initial_cash,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        fees=fees,
        slippage=slippage,
        freq="1D",
    )



def infer_equal_buy_size_pct(
    signals: pd.DataFrame,
    *,
    action_col: str = "action",
    size_col: str = "size_pct",
) -> float:
    """Fixed BUY size for the sizing diagnostic.

    By default we use the arithmetic mean of the final BUY target sizes in the
    sample. This is a diagnostic, not a tradable rule: it keeps the average
    target size per BUY unchanged while removing the PM's time-varying sizing.
    """
    buys = signals.loc[
        signals[action_col].astype(str).str.upper().eq("BUY"),
        size_col,
    ]
    if buys.empty:
        return 0.0
    return float(pd.to_numeric(buys, errors="raise").mean())


def build_equal_size_buy_portfolio(
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    signals: pd.DataFrame,
    initial_cash: float,
    fees: float,
    slippage: float,
    *,
    fixed_size_pct: float,
    action_col: str = "action",
) -> vbt.Portfolio:
    """Replay the same final BUY/HOLD/SELL signals with one fixed BUY size."""
    if fixed_size_pct < 0:
        raise ValueError("fixed_size_pct no puede ser negativo.")

    diagnostic = signals.copy()
    diagnostic["diag_action"] = diagnostic[action_col]
    diagnostic["diag_size_pct"] = np.where(
        diagnostic[action_col].astype(str).str.upper().eq("BUY"),
        float(fixed_size_pct),
        0.0,
    )

    # If there are no BUY signals, keep the long-only portfolio entirely in cash.
    return build_mas_portfolio(
        close_df,
        open_df,
        diagnostic,
        initial_cash,
        fees,
        slippage,
        action_col="diag_action",
        size_col="diag_size_pct",
    )


def build_exposure_matched_benchmark(
    benchmark_open: pd.Series,
    benchmark_close: pd.Series,
    signals: pd.DataFrame,
    initial_cash: float,
    fees: float,
    slippage: float,
    *,
    action_col: str = "action",
    size_col: str = "size_pct",
) -> tuple[vbt.Portfolio, pd.DataFrame]:
    """Apply MAS's aggregate target exposure schedule to one benchmark.

    At each MAS rebalance date, the benchmark target equals the sum of the
    final long target weights across all tickers. The rest remains cash.
    This isolates asset-selection value from the decision to carry little or
    much market exposure. It matches TARGET exposure at rebalance dates, not
    realized daily exposure between rebalances.
    """
    schedule_rows: list[dict[str, Any]] = []
    target = pd.Series(np.nan, index=benchmark_close.index, dtype=float)

    for d, group in signals.groupby("execution_date", sort=True):
        d = pd.Timestamp(d)
        if d not in target.index:
            raise RuntimeError(
                f"Benchmark: execution_date {d.date()} no está en precios."
            )

        gross_pct = float(
            group.loc[
                group[action_col].astype(str).str.upper().eq("BUY"),
                size_col,
            ].sum()
        )
        if gross_pct < -1e-12:
            raise ValueError("La exposición agregada no puede ser negativa.")
        if gross_pct > 100.0 + 1e-9:
            raise ValueError(
                f"Exposición agregada {gross_pct:.2f}% > 100% en {d.date()}."
            )

        target.at[d] = gross_pct / 100.0
        schedule_rows.append(
            {
                "execution_date": d,
                "target_exposure_pct": gross_pct,
            }
        )

    pf = vbt.Portfolio.from_orders(
        benchmark_close,
        size=target,
        size_type="targetpercent",
        direction="longonly",
        price=benchmark_open,
        init_cash=initial_cash,
        fees=fees,
        slippage=slippage,
        freq="1D",
    )
    return pf, pd.DataFrame(schedule_rows)


def build_buy_hold_portfolio(
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    initial_cash: float,
    fees: float,
    slippage: float,
) -> vbt.Portfolio:
    n = close_df.shape[1]
    target = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    target.iloc[0, :] = 1.0 / n
    return vbt.Portfolio.from_orders(
        close_df,
        size=target,
        size_type="targetpercent",
        direction="longonly",
        price=open_df,
        init_cash=initial_cash,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        fees=fees,
        slippage=slippage,
        freq="1D",
    )


def build_equal_weight_rebalanced(
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    initial_cash: float,
    fees: float,
    slippage: float,
) -> vbt.Portfolio:
    n = close_df.shape[1]
    target = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    for d in rebalance_dates:
        if d in target.index:
            target.loc[d, :] = 1.0 / n
    return vbt.Portfolio.from_orders(
        close_df,
        size=target,
        size_type="targetpercent",
        direction="longonly",
        price=open_df,
        init_cash=initial_cash,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        fees=fees,
        slippage=slippage,
        freq="1D",
    )


def build_spy_buy_hold(
    spy_open: pd.Series,
    spy_close: pd.Series,
    initial_cash: float,
    fees: float,
    slippage: float,
) -> vbt.Portfolio:
    target = pd.Series(np.nan, index=spy_close.index, dtype=float)
    target.iloc[0] = 1.0
    return vbt.Portfolio.from_orders(
        spy_close,
        size=target,
        size_type="targetpercent",
        direction="longonly",
        price=spy_open,
        init_cash=initial_cash,
        fees=fees,
        slippage=slippage,
        freq="1D",
    )


def _value_series(pf: vbt.Portfolio) -> pd.Series:
    value = pf.value()
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 1:
            value = value.iloc[:, 0]
        else:
            value = value.sum(axis=1)
    return pd.Series(value, copy=False).astype(float)


def _asset_value_series(pf: vbt.Portfolio) -> pd.Series:
    av = pf.asset_value(group_by=False)
    if isinstance(av, pd.DataFrame):
        return av.abs().sum(axis=1).astype(float)
    return pd.Series(av, copy=False).abs().astype(float)


def _orders_frame(pf: vbt.Portfolio) -> pd.DataFrame:
    try:
        return pf.orders.records_readable.copy()
    except Exception:
        return pd.DataFrame()


def _trades_frame(pf: vbt.Portfolio) -> pd.DataFrame:
    try:
        return pf.trades.records_readable.copy()
    except Exception:
        return pd.DataFrame()


def compute_metrics(
    name: str,
    pf: vbt.Portfolio,
    initial_cash: float,
) -> dict[str, Any]:
    value = _value_series(pf)
    initial = float(initial_cash)
    final = float(value.iloc[-1])

    baseline_idx = value.index[0] - pd.Timedelta(days=1)
    value_with_baseline = pd.concat(
        [pd.Series([initial], index=pd.DatetimeIndex([baseline_idx])), value]
    )
    returns = (
        value_with_baseline.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    total_return = final / initial - 1.0
    periods = max(len(returns), 1)
    annualized_return = (
        (final / initial) ** (252.0 / periods) - 1.0
        if initial > 0 and final > 0
        else np.nan
    )
    vol = float(returns.std(ddof=1) * math.sqrt(252)) if len(returns) > 1 else np.nan
    sharpe = (
        float(returns.mean() / returns.std(ddof=1) * math.sqrt(252))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )
    dd = value_with_baseline / value_with_baseline.cummax() - 1.0
    max_drawdown = float(dd.min()) if not dd.empty else np.nan

    asset_value = _asset_value_series(pf).reindex(value.index).fillna(0.0)
    exposure = (asset_value / value.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    avg_exposure = float(exposure.mean()) if not exposure.empty else np.nan
    max_exposure = float(exposure.max()) if not exposure.empty else np.nan

    orders = _orders_frame(pf)
    if not orders.empty and {"Size", "Price"}.issubset(orders.columns):
        gross_traded = float((orders["Size"].abs() * orders["Price"].abs()).sum())
        turnover = gross_traded / float(value.mean()) if value.mean() else np.nan
    else:
        turnover = 0.0

    trades = _trades_frame(pf)
    closed = trades
    if not trades.empty and "Status" in trades.columns:
        closed = trades.loc[
            trades["Status"].astype(str).str.lower().eq("closed")
        ]

    hit_rate = np.nan
    if not closed.empty and "PnL" in closed.columns:
        hit_rate = float((closed["PnL"] > 0).mean())

    return {
        "strategy": name,
        "initial_value": initial,
        "final_value": final,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": vol,
        "sharpe_0rf": sharpe,
        "max_drawdown": max_drawdown,
        "avg_gross_exposure": avg_exposure,
        "max_gross_exposure": max_exposure,
        "turnover_total": turnover,
        "orders": int(len(orders)),
        "closed_trades": int(len(closed)),
        "hit_rate": hit_rate,
    }


def build_judge_effect(
    signals: pd.DataFrame,
    close_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for ticker, group in signals.groupby("ticker", sort=False):
        group = group.sort_values("execution_date").reset_index(drop=True)
        for i, row in group.iterrows():
            if str(row["pm_action"]) != "BUY":
                continue
            if str(row.get("debate_status", "")) != "RUN":
                continue

            start_date = pd.Timestamp(row["execution_date"])
            start_price = float(row["execution_open_price"])

            if i + 1 < len(group):
                nxt = group.iloc[i + 1]
                end_date = pd.Timestamp(nxt["execution_date"])
                end_price = float(nxt["execution_open_price"])
                end_rule = "next_rebalance_open"
            else:
                end_date = close_df.index.max()
                end_price = float(close_df.at[end_date, ticker])
                end_rule = "evaluation_end_close"

            rows.append(
                {
                    "decision_date": row["decision_date"],
                    "execution_date": start_date,
                    "ticker": ticker,
                    "pm_size_pct": float(row["pm_size_pct"]),
                    "bull_conviction": row.get("bull_conviction", np.nan),
                    "bear_conviction": row.get("bear_conviction", np.nan),
                    "conviction_gap": (
                        float(row.get("bull_conviction", 0))
                        - float(row.get("bear_conviction", 0))
                    ),
                    "judge_action": row.get("judge_action", ""),
                    "judge_confidence": row.get("judge_confidence", np.nan),
                    "validation_status": row.get("validation_status", ""),
                    "committee_action": row["committee_action"],
                    "adversarial_action": row["action"],
                    "period_end": end_date,
                    "period_end_rule": end_rule,
                    "start_price": start_price,
                    "end_price": end_price,
                    "forward_return": end_price / start_price - 1.0,
                }
            )

    return pd.DataFrame(rows)


def print_judge_effect(effect: pd.DataFrame) -> None:
    if effect.empty:
        print("\nJUDGE EFFECT: no hubo BUY debatidos en este periodo.")
        return

    print("\n" + "=" * 100)
    print("JUDGE EFFECT — retorno del subyacente durante el siguiente periodo")
    print("=" * 100)
    for action in ("BUY", "HOLD", "SELL"):
        part = effect[
            effect["judge_action"].astype(str).str.upper().eq(action)
        ]
        if part.empty:
            print(f"{action:5s}: n=0")
        else:
            print(
                f"{action:5s}: n={len(part)} | "
                f"media={part['forward_return'].mean():.2%} | "
                f"mediana={part['forward_return'].median():.2%} | "
                f"gap_conv_medio={part['conviction_gap'].mean():+.1f}"
            )
    print("=" * 100)


def print_summary(summary: pd.DataFrame) -> None:
    out = summary.copy()
    for c in (
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "avg_gross_exposure",
        "max_gross_exposure",
        "hit_rate",
    ):
        out[c] = out[c].map(
            lambda x: "N/D" if pd.isna(x) else f"{float(x):.2%}"
        )
    out["sharpe_0rf"] = out["sharpe_0rf"].map(
        lambda x: "N/D" if pd.isna(x) else f"{float(x):.2f}"
    )
    out["turnover_total"] = out["turnover_total"].map(
        lambda x: "N/D" if pd.isna(x) else f"{float(x):.2f}x"
    )
    out["initial_value"] = out["initial_value"].map(lambda x: f"{x:,.2f}")
    out["final_value"] = out["final_value"].map(lambda x: f"{x:,.2f}")

    cols = [
        "strategy",
        "final_value",
        "total_return",
        "annualized_volatility",
        "sharpe_0rf",
        "max_drawdown",
        "avg_gross_exposure",
        "turnover_total",
        "orders",
    ]
    print("\n" + "=" * 120)
    print("RESUMEN BACKTEST")
    print("=" * 120)
    print(out[cols].to_string(index=False))
    print("=" * 120)


def main() -> None:
    args = parse_args()
    signal_path = Path(args.signals)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signals = load_signals(signal_path)
    tickers = list(dict.fromkeys(signals["ticker"].tolist()))
    first_exec = signals["execution_date"].min()
    evaluation_end = infer_evaluation_end(signals, args.evaluation_end)

    print("=" * 100)
    print("FASE B — COMMITTEE ONLY vs ADVERSARIAL")
    print("=" * 100)
    print(f"Signals          : {signal_path}")
    print(f"Tickers          : {', '.join(tickers)}")
    print(f"Primera ejecución: {first_exec.date()}")
    print(f"Fin evaluación   : {evaluation_end.date()}")
    print(f"Capital inicial  : {args.initial_cash:,.2f}")
    print(f"Fees             : {args.fees_bps:.2f} bps")
    print(f"Slippage         : {args.slippage_bps:.2f} bps")
    print("SIZE_PCT         : target como % del NAV TOTAL")
    print("Judge            : BUY valida; HOLD/SELL vetan; nunca cambia SIZE_PCT")
    print("Diagnósticos     : Equal-Size BUY + benchmark con target exposure equivalente")
    print("=" * 100)

    fees = args.fees_bps / 10_000.0
    slippage = args.slippage_bps / 10_000.0

    open_df, close_df = fetch_ohlc(tickers, first_exec, evaluation_end)
    validate_execution_prices(signals, open_df)

    first_common = open_df.index.min()
    last_common = open_df.index.max()
    signals = signals[
        (signals["execution_date"] >= first_common)
        & (signals["execution_date"] <= last_common)
    ].copy()

    committee = build_mas_portfolio(
        close_df,
        open_df,
        signals,
        args.initial_cash,
        fees,
        slippage,
        action_col="committee_action",
        size_col="committee_size_pct",
    )
    adversarial = build_mas_portfolio(
        close_df,
        open_df,
        signals,
        args.initial_cash,
        fees,
        slippage,
        action_col="action",
        size_col="size_pct",
    )

    equal_buy_size_pct = (
        float(args.equal_buy_size_pct)
        if args.equal_buy_size_pct is not None
        else infer_equal_buy_size_pct(
            signals,
            action_col="action",
            size_col="size_pct",
        )
    )
    if equal_buy_size_pct < 0:
        raise ValueError("--equal-buy-size-pct no puede ser negativo.")

    equal_size_buy = build_equal_size_buy_portfolio(
        close_df,
        open_df,
        signals,
        args.initial_cash,
        fees,
        slippage,
        fixed_size_pct=equal_buy_size_pct,
        action_col="action",
    )

    universe_bh = build_buy_hold_portfolio(
        close_df, open_df, args.initial_cash, fees, slippage
    )
    rebalance_dates = sorted(pd.to_datetime(signals["execution_date"].unique()))
    equal_weight = build_equal_weight_rebalanced(
        close_df,
        open_df,
        rebalance_dates,
        args.initial_cash,
        fees,
        slippage,
    )

    spy_open_df, spy_close_df = fetch_ohlc(
        [args.benchmark], first_exec, evaluation_end
    )
    spy = build_spy_buy_hold(
        spy_open_df[args.benchmark],
        spy_close_df[args.benchmark],
        args.initial_cash,
        fees,
        slippage,
    )
    spy_exposure_matched, exposure_schedule = build_exposure_matched_benchmark(
        spy_open_df[args.benchmark],
        spy_close_df[args.benchmark],
        signals,
        args.initial_cash,
        fees,
        slippage,
        action_col="action",
        size_col="size_pct",
    )

    portfolios = {
        "MAS Committee Only": committee,
        "MAS + Adversarial": adversarial,
        f"Equal-Size BUY ({equal_buy_size_pct:.2f}%)": equal_size_buy,
        f"{args.benchmark} Target-Exposure Matched": spy_exposure_matched,
        f"{args.benchmark} Buy&Hold": spy,
        "Universe Buy&Hold": universe_bh,
        "EqualWeight Rebalanced": equal_weight,
    }

    summary = pd.DataFrame(
        [
            compute_metrics(name, pf, args.initial_cash)
            for name, pf in portfolios.items()
        ]
    )

    print("\n" + "=" * 100)
    print("DIAGNÓSTICOS DE SEÑAL / SIZING / EXPOSICIÓN")
    print("=" * 100)
    print(
        f"Equal-Size BUY   : mismas señales finales BUY, "
        f"tamaño fijo={equal_buy_size_pct:.2f}% por BUY."
    )
    if args.equal_buy_size_pct is None:
        print(
            "                    El tamaño fijo es la media de SIZE_PCT de los BUY "
            "finales del propio periodo."
        )
    else:
        print("                    Tamaño fijado explícitamente por CLI.")
    print(
        f"{args.benchmark} Exposure    : en cada rebalanceo usa la suma de los "
        "targets BUY finales de MAS; el resto queda en cash."
    )
    print(
        "Interpretación    : Equal-Size aísla el sizing temporal; "
        "Exposure-Matched ayuda a aislar selección frente al mercado."
    )
    print("=" * 100)

    print_summary(summary)

    judge_effect = build_judge_effect(signals, close_df)
    print_judge_effect(judge_effect)

    equity = pd.concat(
        {name: _value_series(pf) for name, pf in portfolios.items()},
        axis=1,
    ).sort_index().ffill()
    baseline_date = equity.index.min() - pd.Timedelta(days=1)
    baseline = pd.DataFrame(
        {col: [float(args.initial_cash)] for col in equity.columns},
        index=pd.DatetimeIndex([baseline_date]),
    )
    equity = pd.concat([baseline, equity]).sort_index()

    summary.to_csv(output_dir / "summary.csv", index=False)
    equity.to_csv(output_dir / "equity_curve.csv", index_label="date")
    _orders_frame(committee).to_csv(output_dir / "committee_orders.csv", index=False)
    _trades_frame(committee).to_csv(output_dir / "committee_trades.csv", index=False)
    _orders_frame(adversarial).to_csv(output_dir / "adversarial_orders.csv", index=False)
    _trades_frame(adversarial).to_csv(output_dir / "adversarial_trades.csv", index=False)
    _orders_frame(equal_size_buy).to_csv(
        output_dir / "equal_size_buy_orders.csv", index=False
    )
    _trades_frame(equal_size_buy).to_csv(
        output_dir / "equal_size_buy_trades.csv", index=False
    )
    _orders_frame(spy_exposure_matched).to_csv(
        output_dir / "benchmark_exposure_matched_orders.csv", index=False
    )
    _trades_frame(spy_exposure_matched).to_csv(
        output_dir / "benchmark_exposure_matched_trades.csv", index=False
    )
    exposure_schedule.to_csv(
        output_dir / "benchmark_exposure_schedule.csv", index=False
    )

    # Compatibility aliases: MAS is the final adversarial path in v2.
    _orders_frame(adversarial).to_csv(output_dir / "mas_orders.csv", index=False)
    _trades_frame(adversarial).to_csv(output_dir / "mas_trades.csv", index=False)
    judge_effect.to_csv(output_dir / "judge_effect.csv", index=False)

    signals[
        [
            "decision_date",
            "execution_date",
            "ticker",
            "position_semantics",
            "pm_action",
            "pm_size_pct",
            "hard_risk_status",
            "committee_action",
            "committee_size_pct",
            "debate_status",
            "bull_conviction",
            "bear_conviction",
            "conviction_gap",
            "judge_action",
            "judge_confidence",
            "validation_status",
            "qualitative_risk_status",
            "action",
            "size_pct",
            "execution_open_price",
        ]
    ].to_csv(output_dir / "signals_used.csv", index=False)

    print("\nArchivos generados:")
    for name in (
        "summary.csv",
        "equity_curve.csv",
        "committee_orders.csv",
        "committee_trades.csv",
        "adversarial_orders.csv",
        "adversarial_trades.csv",
        "equal_size_buy_orders.csv",
        "equal_size_buy_trades.csv",
        "benchmark_exposure_matched_orders.csv",
        "benchmark_exposure_matched_trades.csv",
        "benchmark_exposure_schedule.csv",
        "mas_orders.csv",
        "mas_trades.csv",
        "signals_used.csv",
        "judge_effect.csv",
    ):
        print(f"  {output_dir / name}")
    print("\nNo se ha realizado ninguna llamada a un LLM en esta fase.")


if __name__ == "__main__":
    main()
