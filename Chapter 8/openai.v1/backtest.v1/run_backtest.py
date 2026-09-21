"""Phase B: backtest signals.csv with VectorBT, without any LLM calls.

Strategy semantics
------------------
- BUY  -> add size_pct of CURRENT portfolio value at execution open.
- HOLD -> keep the current position unchanged.
- SELL -> fully close the long position.
- Long-only, shared cash across all tickers.
- Signals are generated EOD and executed at the recorded next-session open.

Benchmarks
----------
- SPY Buy & Hold: 100% invested at the first execution open.
- Universe Buy & Hold: equal-weight buy at the first execution open, no rebalance.
- Equal Weight Rebalanced: equal-weight the signal universe at each execution date.

The final signal needs time to work. If --evaluation-end is omitted, the
simulation ends at the month-end following the final decision date.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from numba import njit

import vectorbt as vbt
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import Direction, NoOrder, SizeType

HERE = Path(__file__).resolve().parent
DEFAULT_SIGNALS = HERE / "output" / "signals.csv"
DEFAULT_OUTPUT_DIR = HERE / "output" / "backtest"


ACTION_NONE = 0
ACTION_BUY = 1
ACTION_SELL = -1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest de signals.csv con VectorBT. No realiza llamadas LLM."
    )
    p.add_argument("--signals", default=str(DEFAULT_SIGNALS))
    p.add_argument("--initial-cash", type=float, default=100_000.0)
    p.add_argument(
        "--fees-bps",
        type=float,
        default=0.0,
        help="Comisión proporcional por operación en puntos básicos. 10 bps = 0.10%%.",
    )
    p.add_argument(
        "--slippage-bps",
        type=float,
        default=0.0,
        help="Slippage por operación en puntos básicos.",
    )
    p.add_argument(
        "--evaluation-end",
        default=None,
        help=(
            "Último día del backtest YYYY-MM-DD. Si se omite, usa fin del mes "
            "posterior a la última decision_date."
        ),
    )
    p.add_argument("--benchmark", default="SPY")
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
        "action",
        "size_pct",
        "final_decision",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"signals.csv no contiene columnas requeridas: {missing}")

    for col in ("decision_date", "execution_date"):
        df[col] = pd.to_datetime(df[col], errors="raise").dt.normalize()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].astype(str).str.upper().str.strip()
    df["size_pct"] = pd.to_numeric(df["size_pct"], errors="raise")
    df["execution_open_price"] = pd.to_numeric(
        df["execution_open_price"], errors="raise"
    )

    invalid_actions = sorted(set(df["action"]) - {"BUY", "HOLD", "SELL"})
    if invalid_actions:
        raise ValueError(f"Acciones no soportadas: {invalid_actions}")

    dup = df.duplicated(["execution_date", "ticker"], keep=False)
    if dup.any():
        bad = df.loc[dup, ["execution_date", "ticker", "action"]]
        raise ValueError(
            "Hay señales duplicadas para la misma execution_date/ticker:\n"
            + bad.to_string(index=False)
        )

    if (df["size_pct"] < 0).any():
        raise ValueError("size_pct no puede ser negativo.")

    bad_non_buy = df["action"].isin(["HOLD", "SELL"]) & (df["size_pct"] != 0)
    if bad_non_buy.any():
        raise ValueError("HOLD/SELL deben tener size_pct=0.")

    if (df.loc[df["action"] == "BUY", "size_pct"] <= 0).any():
        raise ValueError("BUY debe tener size_pct > 0.")

    return df.sort_values(["execution_date", "ticker"]).reset_index(drop=True)


def infer_evaluation_end(signals: pd.DataFrame, explicit: str | None) -> pd.Timestamp:
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
    """Fetch raw Open/Close and align every asset on common US sessions."""
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
    """Warn if a recorded execution price diverges materially from Yahoo."""
    for row in signals.itertuples(index=False):
        d = pd.Timestamp(row.execution_date)
        if d not in open_df.index:
            raise RuntimeError(
                f"{row.ticker}: execution_date {d.date()} no es una sesión disponible."
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
                f"vs Yahoo={fetched:.4f} ({rel:.2%}). Se usa el precio del CSV."
            )


def build_signal_arrays(
    signals: pd.DataFrame,
    index: pd.DatetimeIndex,
    tickers: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.zeros((len(index), len(tickers)), dtype=np.int8)
    pcts = np.zeros((len(index), len(tickers)), dtype=np.float64)
    order_prices = np.full((len(index), len(tickers)), np.nan, dtype=np.float64)

    row_lookup = {pd.Timestamp(d): i for i, d in enumerate(index)}
    col_lookup = {ticker: j for j, ticker in enumerate(tickers)}

    for row in signals.itertuples(index=False):
        d = pd.Timestamp(row.execution_date)
        if d not in row_lookup:
            raise RuntimeError(
                f"execution_date {d.date()} de {row.ticker} no está en los precios."
            )
        i = row_lookup[d]
        j = col_lookup[row.ticker]

        if row.action == "BUY":
            actions[i, j] = ACTION_BUY
            pcts[i, j] = float(row.size_pct) / 100.0
        elif row.action == "SELL":
            actions[i, j] = ACTION_SELL
        else:
            actions[i, j] = ACTION_NONE

        order_prices[i, j] = float(row.execution_open_price)

    return actions, pcts, order_prices


@njit
def _pre_segment_nb(c, actions, open_prices):
    """Revalue at today's open and process SELL before BUY within shared cash."""
    for col in range(c.from_col, c.to_col):
        px = open_prices[c.i, col]
        if not np.isnan(px):
            c.last_val_price[col] = px

    k = 0
    # SELL first
    for rel_col in range(c.group_len):
        col = c.from_col + rel_col
        if actions[c.i, col] == ACTION_SELL:
            c.call_seq_now[k] = rel_col
            k += 1
    # HOLD/no-order second
    for rel_col in range(c.group_len):
        col = c.from_col + rel_col
        if actions[c.i, col] == ACTION_NONE:
            c.call_seq_now[k] = rel_col
            k += 1
    # BUY last
    for rel_col in range(c.group_len):
        col = c.from_col + rel_col
        if actions[c.i, col] == ACTION_BUY:
            c.call_seq_now[k] = rel_col
            k += 1
    return ()


@njit
def _mas_order_nb(c, actions, pcts, order_prices, fees, slippage):
    action = actions[c.i, c.col]
    if action == ACTION_NONE:
        return NoOrder

    price = order_prices[c.i, c.col]
    if np.isnan(price) or price <= 0:
        return NoOrder

    if action == ACTION_SELL:
        if c.position_now <= 0:
            return NoOrder
        return nb.close_position_nb(
            price=price,
            fees=fees,
            slippage=slippage,
            allow_partial=True,
        )

    pct = pcts[c.i, c.col]
    if pct <= 0:
        return NoOrder

    # Incremental sizing: buy size_pct of current TOTAL portfolio value.
    order_value = c.value_now * pct
    if order_value <= 0:
        return NoOrder

    return nb.order_nb(
        size=order_value,
        price=price,
        size_type=SizeType.Value,
        direction=Direction.LongOnly,
        fees=fees,
        slippage=slippage,
        allow_partial=True,
    )


def build_mas_portfolio(
    close_df: pd.DataFrame,
    open_df: pd.DataFrame,
    signals: pd.DataFrame,
    initial_cash: float,
    fees: float,
    slippage: float,
) -> vbt.Portfolio:
    tickers = list(close_df.columns)
    actions, pcts, order_prices = build_signal_arrays(
        signals, close_df.index, tickers
    )

    return vbt.Portfolio.from_order_func(
        close_df,
        _mas_order_nb,
        actions,
        pcts,
        order_prices,
        np.asarray(fees),
        np.asarray(slippage),
        init_cash=initial_cash,
        cash_sharing=True,
        group_by=True,
        row_wise=True,
        update_value=True,
        pre_segment_func_nb=_pre_segment_nb,
        pre_segment_args=(actions, open_df.to_numpy(dtype=np.float64)),
        freq="1D",
    )


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


def compute_metrics(name: str, pf: vbt.Portfolio) -> dict[str, Any]:
    value = _value_series(pf)
    returns = value.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    initial = float(value.iloc[0])
    final = float(value.iloc[-1])
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

    dd = value / value.cummax() - 1.0
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
        mask = trades["Status"].astype(str).str.lower().eq("closed")
        closed = trades.loc[mask]

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


def print_summary(summary: pd.DataFrame) -> None:
    out = summary.copy()
    pct_cols = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "avg_gross_exposure",
        "max_gross_exposure",
        "hit_rate",
    ]
    for c in pct_cols:
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
    print("FASE B — BACKTEST VECTORBT")
    print("=" * 100)
    print(f"Signals          : {signal_path}")
    print(f"Tickers          : {', '.join(tickers)}")
    print(f"Primera ejecución: {first_exec.date()}")
    print(f"Fin evaluación   : {evaluation_end.date()}")
    print(f"Capital inicial  : {args.initial_cash:,.2f}")
    print(f"Fees             : {args.fees_bps:.2f} bps")
    print(f"Slippage         : {args.slippage_bps:.2f} bps")
    print("Semántica BUY    : añadir size_pct del valor TOTAL actual de cartera")
    print("Semántica HOLD   : mantener")
    print("Semántica SELL   : liquidar posición")
    print("=" * 100)

    fees = args.fees_bps / 10_000.0
    slippage = args.slippage_bps / 10_000.0

    open_df, close_df = fetch_ohlc(tickers, first_exec, evaluation_end)
    validate_execution_prices(signals, open_df)

    # Restrict to actual common sessions returned by Yahoo.
    first_common = open_df.index.min()
    last_common = open_df.index.max()
    signals = signals[
        (signals["execution_date"] >= first_common)
        & (signals["execution_date"] <= last_common)
    ].copy()

    mas = build_mas_portfolio(
        close_df,
        open_df,
        signals,
        args.initial_cash,
        fees,
        slippage,
    )

    universe_bh = build_buy_hold_portfolio(
        close_df,
        open_df,
        args.initial_cash,
        fees,
        slippage,
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
    spy_open = spy_open_df[args.benchmark]
    spy_close = spy_close_df[args.benchmark]
    spy = build_spy_buy_hold(
        spy_open,
        spy_close,
        args.initial_cash,
        fees,
        slippage,
    )

    portfolios = {
        "MAS": mas,
        f"{args.benchmark} Buy&Hold": spy,
        "Universe Buy&Hold": universe_bh,
        "EqualWeight Rebalanced": equal_weight,
    }

    summary = pd.DataFrame(
        [compute_metrics(name, pf) for name, pf in portfolios.items()]
    )
    print_summary(summary)

    equity = pd.concat(
        {
            name: _value_series(pf)
            for name, pf in portfolios.items()
        },
        axis=1,
    ).sort_index().ffill()

    summary.to_csv(output_dir / "summary.csv", index=False)
    equity.to_csv(output_dir / "equity_curve.csv", index_label="date")

    _orders_frame(mas).to_csv(output_dir / "mas_orders.csv", index=False)
    _trades_frame(mas).to_csv(output_dir / "mas_trades.csv", index=False)

    signals[
        [
            "decision_date",
            "execution_date",
            "ticker",
            "action",
            "confidence",
            "size_pct",
            "execution_open_price",
        ]
    ].to_csv(output_dir / "signals_used.csv", index=False)

    print("\nArchivos generados:")
    print(f"  {output_dir / 'summary.csv'}")
    print(f"  {output_dir / 'equity_curve.csv'}")
    print(f"  {output_dir / 'mas_orders.csv'}")
    print(f"  {output_dir / 'mas_trades.csv'}")
    print(f"  {output_dir / 'signals_used.csv'}")
    print("\nNo se ha realizado ninguna llamada a un LLM.")


if __name__ == "__main__":
    main()
