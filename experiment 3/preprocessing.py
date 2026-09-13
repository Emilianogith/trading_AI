"""
Forex OHLC -> MLP input transformation pipeline.

Implements, for a single prediction instant on a single currency pair:

  1. The 10 most recent weekly candles -> per-candle log-returns + shape
     features (body / upper wick / lower wick), each relative to the
     previous candle's close.
  2. Two "summary" blocks (candles 11-20 and 21-30) -> each block is
     collapsed into a single SYNTHETIC candle (Open=first Open,
     Close=last Close, High=max High, Low=min Low), then given the
     exact same return + shape treatment as the fine-grained candles,
     plus one extra "choppiness" scalar describing how directly price
     moved from the block's Open to its Close.
  3. Volatility normalization: every log-return-based feature is divided
     by a causal, rolling estimate of that pair's own volatility, so
     "the same trend strength" produces comparable numbers across pairs
     with very different typical volatility (e.g. EUR/USD vs EUR/JPY).

The final output is a single flat feature vector, scale-free and
volatility-free, ready to feed into an MLP. The same recipe is reused
for pooling many currency pairs into one training set.

Requires only numpy and pandas.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Low-level building blocks
# --------------------------------------------------------------------------

def log_return(price_to: float, price_from: float) -> float:
    """log(price_to / price_from). `price_from` is the earlier reference price."""
    return float(np.log(price_to / price_from))


def candle_shape_features(o: float, h: float, l: float, c: float) -> dict:
    """Scale-free intra-candle shape, relative to that candle's own open."""
    body = (c - o) / o
    upper_wick = (h - max(o, c)) / o
    lower_wick = (min(o, c) - l) / o
    return {"body": body, "upper_wick": upper_wick, "lower_wick": lower_wick}


def candle_returns(o: float, h: float, l: float, c: float, prev_close: float) -> dict:
    """Log-returns of a candle's OHLC relative to the previous candle's close."""
    return {
        "r_open": log_return(o, prev_close),
        "r_high": log_return(h, prev_close),
        "r_low": log_return(l, prev_close),
        "r_close": log_return(c, prev_close),
    }


def synthetic_candle(block: pd.DataFrame) -> dict:
    """
    Collapse a block of consecutive candles into one higher-timeframe
    synthetic candle: Open of the first, Close of the last, High/Low of
    the whole block. `block` must be ordered chronologically ascending.
    """
    return {
        "open": float(block["Open"].iloc[0]),
        "high": float(block["High"].max()),
        "low": float(block["Low"].min()),
        "close": float(block["Close"].iloc[-1]),
    }


def choppiness(block: pd.DataFrame, block_net_log_return: float, eps: float = 1e-8) -> float:
    """
    Sum of |log-returns| of the individual candles inside the block,
    divided by the block's net log-return. ~1 => clean direct trend,
    >>1 => a lot of back-and-forth to reach the same net destination.
    """
    closes = block["Close"].to_numpy()
    prev_closes = np.concatenate([[block["Open"].iloc[0]], closes[:-1]])
    step_returns = np.log(closes / prev_closes)
    total_abs_movement = np.sum(np.abs(step_returns))
    return float(total_abs_movement / (abs(block_net_log_return) + eps))


def rolling_volatility(history: pd.DataFrame, end_idx: int, lookback: int = 52,
                        eps: float = 1e-8) -> float:
    """
    Causal (no look-ahead) std of weekly log-returns for a pair, using the
    `lookback` closes strictly BEFORE `end_idx`.
    """
    start_idx = max(0, end_idx - lookback - 1)
    closes = history["Close"].iloc[start_idx:end_idx].to_numpy()
    if len(closes) < 3:
        return eps  # not enough history yet: fall back to a tiny epsilon
    log_rets = np.diff(np.log(closes))
    sigma = float(np.std(log_rets))
    return sigma if sigma > eps else eps


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

@dataclass
class InputVector:
    values: np.ndarray
    names: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(zip(self.names, self.values))


def build_input_vector(history: pd.DataFrame, t: int, vol_lookback: int = 52) -> InputVector:
    """
    Build the full MLP input vector for one pair at prediction instant `t`.

    Parameters
    ----------
    history : DataFrame with columns ['Open','High','Low','Close'], one row
        per weekly candle, ordered chronologically ascending, indexed by
        integer position. Must contain at least `t` rows (rows 0..t-1),
        plus ideally `vol_lookback` extra rows before that for the
        volatility estimate.
    t : the row index representing "now" (prediction instant). Candles used
        are history.iloc[t-31 : t] (31 candles: 1 reference + 30 used).
    vol_lookback : number of past weekly returns used for the causal
        volatility estimate (rolling std of log-returns).

    Returns
    -------
    InputVector with a flat numpy array and matching feature names.
    """
    needed_start = t - 31
    if needed_start < 0:
        raise ValueError("Not enough history before t to build the 31-candle window.")

    window = history.iloc[needed_start:t].reset_index(drop=True)
    # window row 0  = reference candle (close only is used, as base for block C)
    # window rows 1..10  = block C (oldest of the 3 groups, original "21-30")
    # window rows 11..20 = block B ("11-20")
    # window rows 21..30 = block A, the 10 most recent candles ("1-10")
    ref_close = float(window["Close"].iloc[0])
    block_C = window.iloc[1:11].reset_index(drop=True)
    block_B = window.iloc[11:21].reset_index(drop=True)
    block_A = window.iloc[21:31].reset_index(drop=True)

    sigma = rolling_volatility(history, end_idx=needed_start + 1, lookback=vol_lookback)

    feats = {}

    # --- Block A: 10 individual most-recent candles -----------------------
    prev_close_for_A_start = float(window["Close"].iloc[20])  # close just before block A
    running_prev_close = prev_close_for_A_start
    for i in range(10):
        o, h, l, c = block_A.loc[i, ["Open", "High", "Low", "Close"]]
        rets = candle_returns(o, h, l, c, running_prev_close)
        shape = candle_shape_features(o, h, l, c)
        # candle index 10 = oldest of the recent block, candle index 1 = most recent
        candle_num = 10 - i
        for k, v in rets.items():
            feats[f"candle{candle_num}_{k}_norm"] = v / sigma
        for k, v in shape.items():
            feats[f"candle{candle_num}_{k}"] = v  # shape features are already scale-free
        running_prev_close = float(c)

    # --- Block B: synthetic candle for candles 11-20 -----------------------
    synth_B = synthetic_candle(block_B)
    base_B = float(window["Close"].iloc[10])  # close right before block B (last of block C)
    rets_B = candle_returns(synth_B["open"], synth_B["high"], synth_B["low"], synth_B["close"], base_B)
    shape_B = candle_shape_features(synth_B["open"], synth_B["high"], synth_B["low"], synth_B["close"])
    net_ret_B = log_return(synth_B["close"], base_B)
    chop_B = choppiness(block_B, net_ret_B)
    for k, v in rets_B.items():
        feats[f"block11_20_{k}_norm"] = v / sigma
    for k, v in shape_B.items():
        feats[f"block11_20_{k}"] = v
    feats["block11_20_choppiness"] = chop_B

    # --- Block C: synthetic candle for candles 21-30 -----------------------
    synth_C = synthetic_candle(block_C)
    base_C = ref_close  # the single reference candle before the whole window
    rets_C = candle_returns(synth_C["open"], synth_C["high"], synth_C["low"], synth_C["close"], base_C)
    shape_C = candle_shape_features(synth_C["open"], synth_C["high"], synth_C["low"], synth_C["close"])
    net_ret_C = log_return(synth_C["close"], base_C)
    chop_C = choppiness(block_C, net_ret_C)
    for k, v in rets_C.items():
        feats[f"block21_30_{k}_norm"] = v / sigma
    for k, v in shape_C.items():
        feats[f"block21_30_{k}"] = v
    feats["block21_30_choppiness"] = chop_C

    names = list(feats.keys())
    values = np.array([feats[n] for n in names], dtype=np.float64)
    return InputVector(values=values, names=names)


def build_dataset(pair_histories: dict, prediction_points: dict, vol_lookback: int = 52) -> pd.DataFrame:
    """
    Build a pooled training dataset across multiple pairs.

    pair_histories : dict {pair_name: DataFrame[Open,High,Low,Close]}
    prediction_points : dict {pair_name: list of int row-indices t to use}

    Returns a DataFrame, one row per (pair, t), with all feature columns
    plus 'pair' and 't' identifier columns. Feature columns are identical
    and comparable across pairs thanks to the normalization above.
    """
    rows = []
    for pair, history in pair_histories.items():
        for t in prediction_points.get(pair, []):
            vec = build_input_vector(history, t, vol_lookback=vol_lookback)
            row = vec.as_dict()
            row["pair"] = pair
            row["t"] = t
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Demo / sanity check with synthetic data
# --------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def fake_pair(n_weeks: int, start_price: float, weekly_vol: float,
                  close_shocks: np.ndarray, wick_shocks: np.ndarray) -> pd.DataFrame:
        """Build a synthetic OHLC history from a given shock sequence, scaled
        by this pair's own weekly_vol/start_price (see previous discussion)."""
        closes = start_price * np.exp(np.cumsum(close_shocks * weekly_vol))
        opens = np.concatenate([[start_price], closes[:-1]])
        highs = np.maximum(opens, closes) * (1 + np.abs(wick_shocks[:, 0]) * weekly_vol)
        lows = np.minimum(opens, closes) * (1 - np.abs(wick_shocks[:, 1]) * weekly_vol)
        return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes})

    def raw_window_preview(history: pd.DataFrame, t: int, n: int = 31) -> pd.DataFrame:
        """The full raw candle window used by build_input_vector (1 reference
        candle + 30 candles) -- default n=31 shows everything, not a sample."""
        return history.iloc[t - n:t].reset_index(drop=True)

    n_weeks, t = 150, 140
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)

    # ---- Pattern 1, shared by pair A (EURUSD-like) and pair B (EURJPY-like) ----
    close_shocks_1 = rng.normal(0, 1, n_weeks)
    wick_shocks_1 = rng.normal(0, 1, size=(n_weeks, 2))
    pair_A = fake_pair(n_weeks, 1.08, 0.006, close_shocks_1, wick_shocks_1)
    pair_B = fake_pair(n_weeks, 162.0, 0.012, close_shocks_1, wick_shocks_1)

    # ---- Pattern 2: an independently-drawn, genuinely different trend ----
    close_shocks_2 = rng.normal(0, 1, n_weeks)
    wick_shocks_2 = rng.normal(0, 1, size=(n_weeks, 2))
    pair_C = fake_pair(n_weeks, 1.08, 0.006, close_shocks_2, wick_shocks_2)  # same level as A, different pattern

    vec_A = build_input_vector(pair_A, t)
    vec_B = build_input_vector(pair_B, t)
    vec_C = build_input_vector(pair_C, t)

    def print_feature_comparison(vec_1: InputVector, vec_2: InputVector, label_1: str, label_2: str) -> np.ndarray:
        df = pd.DataFrame({
            "feature": vec_1.names,
            label_1: vec_1.values,
            label_2: vec_2.values,
        })
        df["abs_diff"] = (df[label_1] - df[label_2]).abs()
        print(df.round(5).to_string(index=False))
        return df["abs_diff"].to_numpy()

    print("=" * 78)
    print("RAW INPUT, full 31-candle window -- SAME pattern, different pair (A vs B)")
    print("=" * 78)
    print("\nPair A (EURUSD-like, price~1.1):")
    print(raw_window_preview(pair_A, t).round(5).to_string())
    print("\nPair B (EURJPY-like, price~160+):")
    print(raw_window_preview(pair_B, t).round(5).to_string())

    print("\n" + "=" * 78)
    print("TRANSFORMED FEATURES (all 86) -- SAME pattern (A vs B): expect near-equal values")
    print("=" * 78)
    diff_AB = print_feature_comparison(vec_A, vec_B, "A", "B")
    print(f"\n  -> max diff = {diff_AB.max():.5f}, mean diff = {diff_AB.mean():.5f}")

    print("\n" + "=" * 78)
    print("RAW INPUT, full 31-candle window -- DIFFERENT pattern, same level (A vs C)")
    print("=" * 78)
    print("\nPair A (pattern 1, price~1.1):")
    print(raw_window_preview(pair_A, t).round(5).to_string())
    print("\nPair C (pattern 2, price~1.1, same level/vol as A):")
    print(raw_window_preview(pair_C, t).round(5).to_string())

    print("\n" + "=" * 78)
    print("TRANSFORMED FEATURES (all 86) -- DIFFERENT pattern (A vs C): expect clearly different values")
    print("=" * 78)
    diff_AC = print_feature_comparison(vec_A, vec_C, "A", "C")
    print(f"\n  -> max diff = {diff_AC.max():.5f}, mean diff = {diff_AC.mean():.5f}")

    print("\n" + "=" * 78)
    print(f"CONCLUSION: mean diff same-pattern (A vs B) = {diff_AB.mean():.5f}")
    print(f"            mean diff different-pattern (A vs C) = {diff_AC.mean():.5f}")
    print("The transform should collapse the first gap and preserve the second.")
