import os
import time

import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class ConfigData:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir

        # self.csv_path = f"{data_dir}/BTC_1sec_with_sentiment_risk_train.csv"
        self.csv_path = f"{data_dir}/BTC_1sec_with_sentiment_risk_train.csv"
        self.input_ary_path = f"{data_dir}/BTC_1sec_input.npy"
        self.label_ary_path = f"{data_dir}/BTC_1sec_label.npy"

        self.predict_ary_path = f"{data_dir}/BTC_1sec_predict.npy"
        self.predict_net_path = f"{data_dir}/BTC_1sec_predict.pth"

def convert_csv_to_level5_csv():
    args = ConfigData()
    load_csv_path = args.csv_path
    save_csv_path = args.csv_path

    df = pd.read_csv(load_csv_path)
    (df.drop(columns=["Unnamed: 0"], inplace=True) if "Unnamed: 0" in df.columns else None)

    df_columns = [
        "system_time",  # Trading system timestamp
        "midpoint",  # Market midpoint price
        "spread",  # Bid-ask spread
        "buys",  # Number of buy transactions
        "sells",  # Number of sell transactions
    ]
    num_levels = 5  # Only use levels 0–4, ignore levels 0–15 from the original data
    for i in range(0, num_levels):
        df_columns.append(f"bids_distance_{i}")  # Difference between the bid price at level {i} and the midpoint price
        df_columns.append(f"asks_distance_{i}")  # Difference between the ask price at level {i} and the midpoint price
        df_columns.append(f"bids_notional_{i}")  # Total value of unfilled bid orders at level {i}
        df_columns.append(f"asks_notional_{i}")  # Total value of unfilled ask orders at level {i}
        df_columns.append(f"bids_cancel_notional_{i}")  # Total value of canceled bid orders at level {i}
        df_columns.append(f"asks_cancel_notional_{i}")  # Total value of canceled ask orders at level {i}
        df_columns.append(f"bids_limit_notional_{i}")  # Total value of unfilled bid limit orders at level {i}
        df_columns.append(f"asks_limit_notional_{i}")  # Total value of unfilled ask limit orders at level {i}

    df = df[df_columns]
    df = df.sort_values(by="system_time")
    df.to_csv(save_csv_path, index=None)
    print(df)


def check_btc_1s_csv():
    args = ConfigData()

    df = pd.read_csv(args.csv_path)
    print(df)

    df["system_time"] = pd.to_datetime(df["system_time"])
    df["system_time"] = df["system_time"].dt.floor("s")

    # Calculate the time difference between adjacent timestamps
    time_diff = df["system_time"].diff()

    # Check if the time difference is within 2 seconds
    invalid_pairs = df["system_time"][time_diff > pd.Timedelta(seconds=2)]

    # import matplotlib.pyplot as plt

    if not invalid_pairs.empty:
        print("There are adjacent time pairs with a time difference greater than 2 seconds:")
        for prev_time, curr_time in zip(invalid_pairs.shift(), invalid_pairs):
            print(f"Previous time: {prev_time}, Current time: {curr_time}, Time difference: {curr_time - prev_time}")
    else:
        print("All adjacent time differences are within 2 seconds")


"""input: technical indicator from alpha101"""

WINDOW = 10
PERIOD = 10


# region Auxiliary functions
def ref(s, n=1):  # 对序列整体下移动N,返回序列(shift后会产生NAN)
    return pd.Series(s).shift(n).values


def ts_sum(df, window=WINDOW):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """

    return df.rolling(window).sum()


def sma(df, window=WINDOW):
    """
    Wrapper function to estimate simple moving average.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()


def ema(df, window, *, adjust=True, min_periods=1):
    return df.ewm(ignore_na=False, span=window, min_periods=min_periods, adjust=adjust).mean()


def stddev(df, window=WINDOW):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()


def correlation(x, y, window=WINDOW):
    """
    Wrapper function to estimate rolling corelations.
    :param x: a pandas DataFrame.
    :param y: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)


def covariance(x, y, window=WINDOW):
    """
    Wrapper function to estimate rolling covariance.
    :param x: a pandas DataFrame.
    :param y: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)


def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]


def ts_rank(df, window=WINDOW):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)


def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)


def product(df, window=WINDOW):
    """
    Wrapper function to estimate rolling product.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)


def ts_min(df, window=WINDOW):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()


def ts_max(df, window=WINDOW):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()


def df_delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)


def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)


def rank(df, window_size=WINDOW):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    # return df.rank(axis=1, pct=True)
    return df.rolling(window=window_size).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    # return df.rank(pct=True)


def scale(df, window_size=WINDOW, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    scaled = df.mul(k)
    rolling_sums = np.abs(df).rolling(window=window_size, min_periods=1).sum()
    normalized = scaled.div(rolling_sums)
    # scaled = df.mul(k)
    # cumulative_sums = np.abs(df).expanding().sum()
    # normalized = scaled.div(cumulative_sums)
    return normalized
    # return df.mul(k).div(np.abs(df).sum())


def ts_argmax(df, window=WINDOW):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df, window=WINDOW):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1


def decay_linear(df, period=PERIOD):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :].values
    # na_series = df.as_matrix()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        # x = na_series[row - period + 1: row + 1, :]
        x = df.iloc[row - period + 1 : row + 1, :].values
        na_lwma[row, :] = np.dot(x.T, y)
    return pd.DataFrame(na_lwma, index=df.index, columns=["LWMA"])


# endregion


class TechIndicator:
    def __init__(self, df):
        self.vwap = df["midpoint"]
        self.spread = df["spread"]
        self.num_asks = df["sells"]
        self.num_bids = df["buys"]

        self.best_bid = self.vwap *(1 + df["bids_distance_3"])  # The best/highest bid price
        self.best_ask = self.vwap * (1 + df["asks_distance_3"])  # The best/lowest ask price
        self.mid_price = (self.best_bid + self.best_ask) / 2
        self.bid_volume = df["bids_notional_3"]  # The best/highest bid volume
        self.ask_volume = df["asks_notional_3"]  # The best/lowest ask volume

        self.returns = np.log(self.mid_price / self.mid_price.shift(1))
        self.volume = (self.num_asks + self.num_bids) / 2

    def macd(self, fast_period=12, slow_period=30):
        df = pd.DataFrame()
        df["midprice"] = self.mid_price
        df["ema_fast"] = ema(df["midprice"], fast_period)
        df["ema_slow"] = ema(df["midprice"], slow_period)

        # Compute MACD
        df["macd"] = df["ema_fast"] - df["ema_slow"]
        return df["macd"]

    def rsi(self, window=14):
        delta = self.mid_price.diff(1)
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Alpha#1
    def alpha001(self):
        inner = self.mid_price
        inner[self.returns < 0] = stddev(self.returns, 20)
        return rank(ts_argmax(inner**2, 5))

    # Alpha#2
    def alpha002(self):
        df = -1 * correlation(
            rank(df_delta(log(self.bid_volume + 1), 2)),
            rank((self.mid_price - self.best_bid) / self.best_bid),
            6,
        )
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # Alpha#3
    def alpha003(self):
        df = -1 * correlation(rank(self.best_ask), rank(self.ask_volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # Alpha#4
    def alpha004(self):
        return -1 * ts_rank(rank(self.best_ask), 9)

    # Alpha#5
    def alpha005(self):
        return rank((self.best_bid - (ts_sum(self.vwap, 10) / 10))) * (-1 * abs(rank((self.mid_price - self.vwap))))

    # Alpha#6
    def alpha006(self):
        df = -1 * correlation(self.best_bid, self.bid_volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # Alpha#7
    def alpha007(self):
        adv20 = sma(self.bid_volume, 20)
        alpha = -1 * ts_rank(abs(df_delta(self.best_bid, 7)), 60) * sign(df_delta(self.best_bid, 7))
        alpha[adv20 >= self.bid_volume] = -1
        return alpha

    # Alpha#8
    def alpha008(self):
        return -1 * (
            rank(
                (
                    (ts_sum(self.best_bid, 5) * ts_sum(self.returns, 5))
                    - delay((ts_sum(self.best_bid, 5) * ts_sum(self.returns, 5)), 10)
                )
            )
        )

    # Alpha#9
    def alpha009(self):
        delta_midprice = df_delta(self.mid_price, 1)
        cond_1 = ts_min(delta_midprice, 5) > 0
        cond_2 = ts_max(delta_midprice, 5) < 0
        alpha = -1 * delta_midprice
        alpha[cond_1 | cond_2] = delta_midprice
        return alpha

    # Alpha#10
    def alpha010(self):
        delta_midprice = df_delta(self.mid_price, 1)
        cond_1 = ts_min(delta_midprice, 4) > 0
        cond_2 = ts_max(delta_midprice, 4) < 0
        alpha = -1 * delta_midprice
        alpha[cond_1 | cond_2] = delta_midprice
        return alpha

    # Alpha#11
    def alpha011(self):
        return (rank(ts_max((self.vwap - self.mid_price), 3)) + rank(ts_min((self.vwap - self.mid_price), 3))) * rank(
            df_delta(self.ask_volume, 3)
        )

    # Alpha#12
    def alpha012(self):
        return sign(df_delta(self.bid_volume, 1)) * (-1 * df_delta(self.mid_price, 1))

    # Alpha#13
    def alpha013(self):
        return -1 * rank(covariance(rank(self.mid_price), rank(self.ask_volume), 5))

    # Alpha#14
    def alpha014(self):
        df = correlation(self.best_bid, self.bid_volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * rank(df_delta(self.returns, 3)) * df

    # Alpha#15
    def alpha015(self):
        df = correlation(rank(self.best_ask), rank(self.ask_volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_sum(rank(df), 3)

    # Alpha#16
    def alpha016(self):
        return -1 * rank(covariance(rank(self.mid_price), rank(self.bid_volume), 5))

    # Alpha#17
    def alpha017(self):
        adv20 = sma(self.bid_volume, 20)
        return -1 * (
            rank(ts_rank(self.mid_price, 10))
            * rank(df_delta(df_delta(self.mid_price, 1), 1))
            * rank(ts_rank((self.bid_volume / adv20), 5))
        )

    # Alpha#18
    def alpha018(self):
        df = correlation(self.mid_price, self.best_bid, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank((stddev(abs((self.mid_price - self.best_bid)), 5) + (self.mid_price - self.best_bid)) + df))

    # Alpha#19
    def alpha019(self):
        return (-1 * sign((self.mid_price - delay(self.mid_price, 7)) + df_delta(self.mid_price, 7))) * (
            1 + rank(1 + ts_sum(self.returns, 250))
        )

    # Alpha#20
    def alpha020(self):
        return -1 * (
            rank(self.best_bid - delay(self.best_ask, 1))
            * rank(self.best_bid - delay(self.mid_price, 1))
            * rank(self.best_ask - delay(self.best_bid, 1))
        )

    # Alpha#21
    def alpha021(self):
        cond_1 = sma(self.mid_price, 8) + stddev(self.mid_price, 8) < sma(self.mid_price, 2)
        cond_2 = sma(self.ask_volume, 20) / self.ask_volume < 1
        alpha = pd.DataFrame(np.ones_like(self.mid_price), index=self.mid_price.index)
        alpha[cond_1 | cond_2] = -1
        return alpha.squeeze(1)

    # Alpha#22
    def alpha022(self):
        df = correlation(self.best_ask, self.ask_volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df_delta(df, 5) * rank(stddev(self.mid_price, 20))

    # Alpha#23
    def alpha023(self):
        cond = sma(self.best_ask, 20) < self.best_ask
        alpha = pd.DataFrame(
            np.zeros_like(self.mid_price),
            index=self.mid_price.index,
            columns=["midprice"],
        )
        alpha.loc[cond, "midprice"] = -1 * df_delta(self.best_ask, 2).fillna(value=0)
        return alpha.squeeze(1)

    # Alpha#24
    def alpha024(self):
        cond = df_delta(sma(self.mid_price, 100), 100) / delay(self.mid_price, 100) <= 0.05
        alpha = -1 * df_delta(self.mid_price, 3)
        alpha[cond] = -1 * (self.mid_price - ts_min(self.mid_price, 100))
        return alpha

    # Alpha#25
    def alpha025(self):
        adv20 = sma(self.ask_volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.best_ask - self.mid_price)))

    # Alpha#26
    def alpha026(self):
        df = correlation(ts_rank(self.bid_volume, 5), ts_rank(self.best_bid, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_max(df, 3)

    # Alpha#27
    def alpha027(self):
        alpha = rank((sma(correlation(rank(self.bid_volume), rank(self.best_bid), 6), 2) / 2.0))
        alpha[alpha > 0.5] = -1
        alpha[alpha <= 0.5] = 1
        return alpha

        # Alpha#28

    def alpha028(self):
        adv20 = sma(self.bid_volume, 20)
        df = correlation(adv20, self.best_bid, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return scale(df)

    def alpha029(self):
        delta_mid = df_delta((self.mid_price - 1), 5)
        rank_neg_delta = rank(-1 * delta_mid)
        sum_ranks = ts_sum(rank(rank_neg_delta), 2)
        scaled_log = scale(log(sum_ranks + 1))
        min_rank = ts_min(rank(rank(scaled_log)), 5)
        delayed_ret = delay(-1 * self.returns, 6)
        ts_rank_ret = ts_rank(delayed_ret, 5)
        return min_rank + ts_rank_ret

    # Alpha#30
    def alpha030(self):
        delta_midprice = df_delta(self.mid_price, 1)
        inner = sign(delta_midprice) + sign(delay(delta_midprice, 1)) + sign(delay(delta_midprice, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.ask_volume, 5)) / ts_sum(self.ask_volume, 20)

    # Alpha#31
    def alpha031(self):
        adv20 = sma(self.bid_volume, 20)
        df = correlation(adv20, self.best_bid, 12).replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = rank(rank(rank(decay_linear((-1 * rank(rank(df_delta(self.mid_price, 10)))).to_frame(), 10))))
        p2 = rank((-1 * df_delta(self.mid_price, 3)))
        p3 = sign(scale(df))

        return p1.LWMA + p2 + p3

    # Alpha#32
    def alpha032(self):
        return scale(((sma(self.mid_price, 7) / 7) - self.mid_price)) + (
            20 * scale(correlation(self.vwap, delay(self.mid_price, 5), 230))
        )

    # Alpha#33
    def alpha033(self):
        return rank(-1 + (self.best_ask / self.mid_price))

    # Alpha#34
    def alpha034(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return rank(2 - rank(inner) - rank(df_delta(self.mid_price, 1)))

    # Alpha#35
    def alpha035(self):
        return (ts_rank(self.bid_volume, 32) * (1 - ts_rank(self.mid_price + self.spread, 16))) * (
            1 - ts_rank(self.returns, 32)
        )

    def alpha036(self):
        adv20 = sma(self.bid_volume, 20)

        # Correlation and rank components
        corr_mid_bid = correlation(self.mid_price - self.best_bid, delay(self.bid_volume, 1), 15)
        rank_corr_mid_bid = rank(corr_mid_bid)
        rank_bid_mid = rank(self.best_bid - self.mid_price)

        # Time series rank of delayed negative returns
        delayed_neg_ret = delay(-1 * self.returns, 6)
        ts_rank_neg_ret = ts_rank(delayed_neg_ret, 5)
        rank_ts_rank_neg_ret = rank(ts_rank_neg_ret)

        # Absolute correlation and rank
        abs_corr_vwap_adv20 = abs(correlation(self.vwap, adv20, 6))
        rank_abs_corr_vwap_adv20 = rank(abs_corr_vwap_adv20)

        # SMA of midprice and rank
        sma_mid_200 = sma(self.mid_price, 200) / 200
        diff_sma_mid_bid = sma_mid_200 - self.best_bid
        prod_diff_mid_bid = diff_sma_mid_bid * (self.mid_price - self.best_bid)
        rank_prod_diff_mid_bid = rank(prod_diff_mid_bid)

        # Combine all components
        result = (
            (2.21 * rank_corr_mid_bid)
            + (0.7 * rank_bid_mid)
            + (0.73 * rank_ts_rank_neg_ret)
            + rank_abs_corr_vwap_adv20
            + (0.6 * rank_prod_diff_mid_bid)
        )
        return result

    # Alpha#37
    def alpha037(self):
        return rank(correlation(delay(self.best_ask - self.mid_price, 1), self.mid_price, 200)) + rank(
            self.best_ask - self.mid_price
        )

    # Alpha#38
    def alpha038(self):
        inner = self.mid_price / self.best_ask
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * rank(ts_rank(self.best_ask, 10)) * rank(inner)

    # Alpha#39
    def alpha039(self):
        adv20 = sma(self.ask_volume, 20)
        return (
            -1
            * rank(
                df_delta(self.mid_price, 7) * (1 - rank(decay_linear((self.ask_volume / adv20).to_frame(), 9).LWMA))
            )
        ) * (1 + rank(sma(self.returns, 250)))

    # Alpha#40
    def alpha040(self):
        return -1 * rank(stddev(self.best_bid, 10)) * correlation(self.best_bid, self.bid_volume, 10)

    # Alpha#41
    def alpha041(self):
        return pow((self.best_bid * self.best_ask), 0.5) - self.vwap

    # Alpha#42
    def alpha042(self):
        return rank((self.vwap - self.mid_price)) / rank((self.vwap + self.mid_price))

    # Alpha#43
    def alpha043(self):
        adv20 = sma(self.ask_volume, 20)
        return ts_rank(self.ask_volume / adv20, 20) * ts_rank((-1 * df_delta(self.mid_price, 7)), 8)

    # Alpha#44
    def alpha044(self):
        df = correlation(self.best_ask, rank(self.ask_volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

    # Alpha#45
    def alpha045(self):
        df = correlation(self.mid_price, self.ask_volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (
            rank(sma(delay(self.mid_price, 5), 20))
            * df
            * rank(correlation(ts_sum(self.mid_price, 5), ts_sum(self.mid_price, 20), 2))
        )

    # Alpha#46
    def alpha046(self):
        inner = ((delay(self.mid_price, 20) - delay(self.mid_price, 10)) / 10) - (
            (delay(self.mid_price, 10) - self.mid_price) / 10
        )
        alpha = -1 * df_delta(self.mid_price)
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    # Alpha#47
    def alpha047(self):
        adv20 = sma(self.bid_volume, 20)
        return (
            ((rank((1 / self.mid_price)) * self.bid_volume) / adv20)
            * ((self.best_bid * rank((self.best_bid - self.mid_price))) / (sma(self.best_bid, 5) / 5))
        ) - rank((self.vwap - delay(self.vwap, 5)))

    # Alpha#48 #self added
    def alpha048(self):
        adv20 = sma(self.bid_volume, 20)
        return (
            ((rank((1 / self.mid_price)) * self.ask_volume) / adv20)
            * ((self.best_ask * rank((self.best_ask - self.mid_price))) / (sma(self.best_ask, 5) / 5))
        ) - rank((self.vwap - delay(self.vwap, 5)))

    # Alpha#49
    def alpha049(self):
        inner = ((delay(self.mid_price, 20) - delay(self.mid_price, 10)) / 10) - (
            (delay(self.mid_price, 10) - self.mid_price) / 10
        )
        alpha = -1 * df_delta(self.mid_price)
        alpha[inner < -0.1] = 1
        return alpha

    # Alpha#50
    def alpha050(self):
        return -1 * ts_max(rank(correlation(rank(self.ask_volume), rank(self.vwap), 5)), 5)

    # Alpha#51
    def alpha051(self):
        inner = ((delay(self.mid_price, 20) - delay(self.mid_price, 10)) / 10) - (
            (delay(self.mid_price, 10) - self.mid_price) / 10
        )
        alpha = -1 * df_delta(self.mid_price)
        alpha[inner < -0.05] = 1
        return alpha

    # Alpha#52
    def alpha052(self):
        return (
            (-1 * df_delta(ts_min(self.best_bid, 5), 5))
            * rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))
        ) * ts_rank(self.bid_volume, 5)

    # Alpha#53
    def alpha053(self):
        inner = self.spread.replace(0, 0.0001)
        return -1 * df_delta(
            (((self.mid_price - self.best_bid) - (self.best_bid - self.mid_price)) / inner),
            9,
        )

    # Alpha#54
    def alpha054(self):
        inner = self.spread.replace(0, -0.0001)
        return -1 * (self.best_bid - self.mid_price) * (self.best_ask**5) / (inner * (self.mid_price**5))

    # Alpha#55
    def alpha055(self):
        divisor = (ts_max(self.best_ask, 12) - ts_min(self.best_bid, 12)).replace(0, 0.0001)
        inner = (self.mid_price - ts_min(self.best_bid, 12)) / divisor
        df = correlation(rank(inner), rank(self.bid_volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # Alpha#56  #self added
    def alpha056(self):
        inner = self.spread.replace(0, -0.0001)
        return -1 * (self.best_ask - self.mid_price) * (self.best_bid**5) / (inner * (self.mid_price**5))

    # Alpha#57
    def alpha057(self):
        return 0 - (
            1 * ((self.mid_price - self.vwap) / decay_linear(rank(ts_argmax(self.mid_price, 30)).to_frame(), 2).LWMA)
        )

    # Alpha#58
    def alpha058(self):
        return sign(df_delta(self.ask_volume, 1)) * (-1 * df_delta(self.mid_price, 1))

    # Alpha#59
    def alpha059(self):
        divisor = self.spread.replace(0, 0.0001)
        inner = ((self.mid_price - self.best_ask) - (self.best_ask - self.mid_price)) * self.ask_volume / divisor
        return -((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.mid_price, 10))))

    # Alpha#60
    def alpha060(self):
        divisor = self.spread.replace(0, 0.0001)
        inner = ((self.mid_price - self.best_bid) - (self.best_bid - self.mid_price)) * self.bid_volume / divisor
        return -((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.mid_price, 10))))

    # Alpha#61
    def alpha061(self):
        adv180 = sma(self.ask_volume, 180)
        return rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18))

    # Alpha#62
    def alpha062(self):
        adv20 = sma(self.ask_volume, 20)
        value1 = rank(correlation(self.vwap, sma(adv20, 22), 10))
        value2 = rank(self.best_ask) + rank(self.best_ask)
        value3 = rank(((self.best_ask + self.best_bid) / 2)) + rank(self.best_ask)
        return value1 - rank(value2 - value3)

    def alpha063(self):
        adv120 = sma(self.bid_volume, 120)
        sma_bid_ask = sma(((self.best_bid * 0.178404) + (self.best_ask * (1 - 0.178404))), 13)
        sma_adv120 = sma(adv120, 13)
        corr = correlation(sma_bid_ask, sma_adv120, 17)
        rank_corr = rank(corr)

        weighted_price = (self.mid_price * 0.178404) + (self.vwap * (1 - 0.178404))
        delta_weighted_price = df_delta(weighted_price, 3)
        rank_delta = rank(delta_weighted_price)

        return rank_corr - rank_delta

    # Alpha#64
    def alpha064(self):
        adv120 = sma(self.ask_volume, 120)
        return (
            rank(
                correlation(
                    sma(
                        ((self.best_ask * 0.178404) + (self.best_bid * (1 - 0.178404))),
                        13,
                    ),
                    sma(adv120, 13),
                    17,
                )
            )
            < rank(
                df_delta(
                    ((((self.best_ask + self.best_bid) / 2) * 0.178404) + (self.vwap * (1 - 0.178404))),
                    3,
                )
            )
        ) * -1

    # Alpha#65
    def alpha065(self):
        adv60 = sma(self.ask_volume, 60)
        return (
            rank(
                correlation(
                    ((self.best_ask * 0.00817205) + (self.vwap * (1 - 0.00817205))),
                    sma(adv60, 9),
                    6,
                )
            )
            < rank((self.best_ask - ts_min(self.best_ask, 14)))
        ) * -1

    def alpha066(self):
        # Calculate the delta of vwap and apply decay_linear
        vwap_delta = df_delta(self.vwap, 4).to_frame()
        decay_vwap_delta = decay_linear(vwap_delta, 7).LWMA
        rank_decay_vwap_delta = rank(decay_vwap_delta)

        # Calculate the expression involving best_bid, best_ask, and midprice, then apply decay_linear and ts_rank
        bid_ask_diff = ((self.best_bid * 0.96633) + (self.best_bid * (1 - 0.96633))) - self.vwap
        price_ratio = (bid_ask_diff / (self.best_ask - self.best_bid)).to_frame()
        decay_price_ratio = decay_linear(price_ratio, 11).LWMA
        ts_rank_decay_price_ratio = ts_rank(decay_price_ratio, 7)

        # Combine the results and return the final expression
        result = (rank_decay_vwap_delta + ts_rank_decay_price_ratio) * -1
        return result

    # Alpha#67
    def alpha067(self):
        adv15 = sma(self.bid_volume, 15)
        return (
            ts_rank(correlation(rank(self.best_bid), rank(adv15), 9), 14)
            < rank(df_delta(((self.mid_price * 0.518371) + (self.best_ask * (1 - 0.518371))), 3))
        ) * -1

    # Alpha#68
    def alpha068(self):
        adv15 = sma(self.ask_volume, 15)
        return (
            ts_rank(correlation(rank(self.best_ask), rank(adv15), 9), 14)
            < rank(df_delta(((self.mid_price * 0.518371) + (self.best_bid * (1 - 0.518371))), 3))
        ) * -1

    # Alpha#69
    def alpha069(self):
        adv180 = sma(self.bid_volume, 180)
        return rank((self.vwap - ts_min(self.vwap, 16))) < rank(correlation(self.vwap, adv180, 18))

    # Alpha#70
    def alpha070(self):
        adv60 = sma(self.bid_volume, 60)
        return (
            rank(
                correlation(
                    ((self.best_bid * 0.00817205) + (self.vwap * (1 - 0.00817205))),
                    sma(adv60, 9),
                    6,
                )
            )
            < rank((self.best_bid - ts_min(self.best_bid, 14)))
        ) * -1

    # Alpha#71
    def alpha071(self):
        adv180 = sma(self.bid_volume, 180)
        p1 = ts_rank(
            decay_linear(
                correlation(ts_rank(self.mid_price, 3), ts_rank(adv180, 12), 18).to_frame(),
                4,
            ).LWMA,
            16,
        )
        p2 = ts_rank(
            decay_linear(
                (rank(((self.best_bid + self.best_ask) - (self.vwap + self.vwap))).pow(2)).to_frame(),
                16,
            ).LWMA,
            4,
        )
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df.loc[df["p1"] >= df["p2"], "max"] = df["p1"]
        df.loc[df["p2"] >= df["p1"], "max"] = df["p2"]
        return df["max"]

    # Alpha#72
    def alpha072(self):
        adv40 = sma(self.bid_volume, 40)
        return rank(decay_linear(correlation(self.mid_price, adv40, 9).to_frame(), 10).LWMA) / rank(
            decay_linear(
                correlation(ts_rank(self.vwap, 4), ts_rank(self.bid_volume, 19), 7).to_frame(),
                3,
            ).LWMA
        )

    # Alpha#73
    def alpha073(self):
        p1 = rank(decay_linear(df_delta(self.vwap, 5).to_frame(), 3).LWMA)
        p2 = ts_rank(
            decay_linear(
                (
                    (
                        df_delta(
                            ((self.best_ask * 0.147155) + (self.best_bid * (1 - 0.147155))),
                            2,
                        )
                        / ((self.mid_price * 0.147155) + (self.best_bid * (1 - 0.147155)))
                    )
                    * -1
                ).to_frame(),
                3,
            ).LWMA,
            17,
        )
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df.loc[df["p1"] >= df["p2"], "max"] = df["p1"]
        df.loc[df["p2"] >= df["p1"], "max"] = df["p2"]
        return -1 * df["max"]

    # Alpha#74
    def alpha074(self):
        adv30 = sma(self.bid_volume, 30)
        return (
            rank(correlation(self.mid_price, sma(adv30, 37), 15))
            < rank(
                correlation(
                    rank(((self.best_bid * 0.0261661) + (self.vwap * (1 - 0.0261661)))),
                    rank(self.bid_volume),
                    11,
                )
            )
        ) * -1

    # Alpha#75
    def alpha075(self):
        adv50 = sma(self.bid_volume, 50)
        return rank(correlation(self.vwap, self.bid_volume, 4)) < rank(
            correlation(rank(self.best_bid), rank(adv50), 12)
        )

    # Alpha#76
    def alpha076(self):
        adv50 = sma(self.ask_volume, 50)
        return rank(correlation(self.vwap, self.ask_volume, 4)) < rank(
            correlation(rank(self.best_ask), rank(adv50), 12)
        )

    # Alpha#77
    def alpha077(self):
        adv40 = sma(self.ask_volume, 40)
        p1 = rank(
            decay_linear(
                ((self.mid_price + self.best_ask) - (self.vwap + self.best_ask)).to_frame(),
                20,
            ).LWMA
        )
        p2 = rank(decay_linear(correlation(self.mid_price, adv40, 3).to_frame(), 6).LWMA)
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df.loc[df["p1"] >= df["p2"], "min"] = df["p2"]
        df.loc[df["p2"] >= df["p1"], "min"] = df["p1"]
        return df["min"]

    # Alpha#78
    def alpha078(self):
        adv40 = sma(self.bid_volume, 40)
        return rank(
            correlation(
                ts_sum(((self.best_bid * 0.352233) + (self.vwap * (1 - 0.352233))), 20),
                ts_sum(adv40, 20),
                7,
            )
        ).pow(rank(correlation(rank(self.vwap), rank(self.bid_volume), 6)))

    # Alpha#79
    def alpha079(self):
        adv40 = sma(self.ask_volume, 40)
        return rank(
            correlation(
                ts_sum(((self.best_ask * 0.352233) + (self.vwap * (1 - 0.352233))), 20),
                ts_sum(adv40, 20),
                7,
            )
        ).pow(rank(correlation(rank(self.vwap), rank(self.ask_volume), 6)))

    # Alpha#80
    def alpha080(self):
        adv10 = sma(self.bid_volume, 10)
        return (
            rank(
                log(
                    product(
                        rank((rank(correlation(self.vwap, ts_sum(adv10, 50), 8)).pow(4))),
                        15,
                    )
                )
            )
            < rank(correlation(rank(self.vwap), rank(self.bid_volume), 5))
        ) * -1

    # Alpha#81
    def alpha081(self):
        adv10 = sma(self.ask_volume, 10)
        return (
            rank(
                log(
                    product(
                        rank((rank(correlation(self.vwap, ts_sum(adv10, 50), 8)).pow(4))),
                        15,
                    )
                )
            )
            < rank(correlation(rank(self.vwap), rank(self.ask_volume), 5))
        ) * -1

    # Alpha#82
    def alpha082(self):
        adv20 = sma(self.bid_volume, 20)
        return (
            ts_rank(correlation(self.mid_price, sma(adv20, 15), 6), 20)
            < rank(((self.best_bid + self.mid_price) - (self.vwap + self.best_bid)))
        ) * -1

    # Alpha#83
    def alpha083(self):
        return (rank(delay((self.spread / (ts_sum(self.mid_price, 5) / 5)), 2)) * rank(rank(self.bid_volume))) / (
            (self.spread / (ts_sum(self.mid_price, 5) / 5)) / (self.vwap - self.mid_price)
        )

    # Alpha#84
    def alpha084(self):
        return pow(
            ts_rank((self.vwap - ts_max(self.vwap, 15)), 21),
            df_delta(self.mid_price, 5),
        )

    # Alpha#85
    def alpha085(self):
        adv30 = sma(self.ask_volume, 30)
        return rank(
            correlation(
                ((self.best_ask * 0.876703) + (self.mid_price * (1 - 0.876703))),
                adv30,
                10,
            )
        ).pow(rank(correlation(ts_rank(self.mid_price, 4), ts_rank(self.ask_volume, 10), 7)))

    # Alpha#86
    def alpha086(self):
        adv20 = sma(self.ask_volume, 20)
        return (
            ts_rank(correlation(self.mid_price, sma(adv20, 15), 6), 20)
            < rank(((self.best_ask + self.mid_price) - (self.vwap + self.best_ask)))
        ) * -1

    # Alpha#87
    def alpha087(self):
        return -1 * ts_rank(rank(self.best_bid), 9)

    def alpha088(self):
        adv60 = sma(self.ask_volume, 60)

        # Calculate p1
        rank_ask = rank(self.best_ask)
        rank_bid = rank(self.best_bid)
        rank_mid = rank(self.mid_price)
        diff_rank = (rank_ask + rank_bid) - (rank_ask + rank_mid)
        decay_diff_rank = decay_linear(diff_rank.to_frame(), 8).LWMA
        p1 = rank(decay_diff_rank)

        # Calculate p2
        ts_rank_mid = ts_rank(self.mid_price, 8)
        ts_rank_adv60 = ts_rank(adv60, 21)
        corr_ts_rank = correlation(ts_rank_mid, ts_rank_adv60, 8)
        decay_corr_ts_rank = decay_linear(corr_ts_rank.to_frame(), 7).LWMA
        p2 = ts_rank(decay_corr_ts_rank, 3)

        # Create DataFrame and calculate 'min'
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df["min"] = df[["p1", "p2"]].min(axis=1)

        return df["min"]

    # Alpha#89
    def alpha089(self):
        df = -1 * correlation(
            rank(df_delta(log(self.ask_volume + 1), 2)),
            rank((self.mid_price - self.best_ask) / self.best_ask),
            6,
        )
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # Alpha#90
    def alpha090(self):
        return -1 * (
            rank(
                (
                    (ts_sum(self.best_ask, 5) * ts_sum(self.returns, 5))
                    - delay((ts_sum(self.best_ask, 5) * ts_sum(self.returns, 5)), 10)
                )
            )
        )

    # Alpha#91
    def alpha091(self):
        df = correlation(self.best_bid, self.bid_volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df_delta(df, 5) * rank(stddev(self.mid_price, 20))

    # Alpha#92
    def alpha092(self):
        adv30 = sma(self.bid_volume, 30)
        p1 = ts_rank(
            decay_linear(
                ((self.mid_price + self.mid_price) < (self.best_ask + self.best_bid)).to_frame(),
                15,
            ).LWMA,
            19,
        )
        p2 = ts_rank(
            decay_linear(correlation(rank(self.best_bid), rank(adv30), 8).to_frame(), 7).LWMA,
            7,
        )
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df.loc[df["p1"] >= df["p2"], "min"] = df["p2"]
        df.loc[df["p2"] >= df["p1"], "min"] = df["p1"]
        return df["min"]

    # Alpha#93
    def alpha093(self):
        adv60 = sma(self.bid_volume, 60)
        return (
            rank((self.vwap - ts_min(self.vwap, 12))).pow(
                ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)
            )
            * -1
        )

    # Alpha#94
    def alpha094(self):
        adv60 = sma(self.ask_volume, 60)
        return (
            rank((self.vwap - ts_min(self.vwap, 12))).pow(
                ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)
            )
            * -1
        )

    # Alpha#95
    def alpha095(self):
        adv40 = sma(self.bid_volume, 40)
        return rank((self.best_bid - ts_min(self.best_bid, 12))) < ts_rank(
            (rank(correlation(sma(self.mid_price, 19), sma(adv40, 19), 13)).pow(5)), 12
        )

    # Alpha#96
    def alpha096(self):
        adv60 = sma(self.ask_volume, 60)
        p1 = ts_rank(
            decay_linear(correlation(rank(self.vwap), rank(self.ask_volume).to_frame(), 4), 4).LWMA,
            8,
        )
        p2 = ts_rank(
            decay_linear(
                ts_argmax(correlation(ts_rank(self.mid_price, 7), ts_rank(adv60, 4), 4), 13).to_frame(),
                14,
            ).LWMA,
            13,
        )
        df = pd.DataFrame({"p1": p1, "p2": p2})
        df.loc[df["p1"] >= df["p2"], "max"] = df["p1"]
        df.loc[df["p2"] >= df["p1"], "max"] = df["p2"]
        return -1 * df["max"]

    # Alpha#97
    def alpha097(self):
        adv5 = sma(self.ask_volume, 5)
        adv15 = sma(self.ask_volume, 15)
        return rank(decay_linear(correlation(self.vwap, sma(adv5, 26), 5).to_frame(), 7).LWMA) - rank(
            decay_linear(
                ts_rank(ts_argmin(correlation(rank(self.best_ask), rank(adv15), 21), 9), 7).to_frame(),
                8,
            ).LWMA
        )

    # Alpha#98
    def alpha098(self):
        adv5 = sma(self.bid_volume, 5)
        adv15 = sma(self.bid_volume, 15)
        return rank(decay_linear(correlation(self.vwap, sma(adv5, 26), 5).to_frame(), 7).LWMA) - rank(
            decay_linear(
                ts_rank(ts_argmin(correlation(rank(self.best_bid), rank(adv15), 21), 9), 7).to_frame(),
                8,
            ).LWMA
        )

    # Alpha#99
    def alpha099(self):
        adv60 = sma(self.bid_volume, 60)
        return (
            rank(correlation(ts_sum(self.mid_price, 20), ts_sum(adv60, 20), 9))
            < rank(correlation(self.best_bid, self.bid_volume, 6))
        ) * -1

    # Alpha#100
    def alpha100(self):
        adv60 = sma(self.ask_volume, 60)
        return (
            rank(correlation(ts_sum(self.mid_price, 20), ts_sum(adv60, 20), 9))
            < rank(correlation(self.best_ask, self.ask_volume, 6))
        ) * -1

    # Alpha#101
    def alpha101(self):
        return (self.mid_price - self.best_bid) / (self.spread + 0.001)

class NewIndicator:
    def __init__(self, df, n=15):
        self.data = df
        self.vwap = df["midpoint"]
        self.spread = df["spread"]
        self.num_asks = df["sells"]
        self.num_bids = df["buys"]

        self.best_bid = self.vwap - df["bids_distance_3"]  # The best/highest bid price
        self.best_ask = self.vwap + df["asks_distance_3"]  # The best/lowest ask price
        self.mid_price = (self.best_bid + self.best_ask) / 2
        self.bid_volume = df["bids_notional_3"]  # The best/highest bid volume
        self.ask_volume = df["asks_notional_3"]  # The best/lowest ask volume

        self.n = n 

    def calculate_cumulative_notionals(self):
        for i in range(self.n):
            self.data[f'cumulative_bids_notional_0_to_{i+1}'] = self.data[[f'bids_notional_{j}' for j in range(i+1)]].sum(axis=1)
            self.data[f'cumulative_asks_notional_0_to_{i+1}'] = self.data[[f'asks_notional_{j}' for j in range(i+1)]].sum(axis=1)

    def calculate_bid_ask_volatility(self):
        for i in range(self.n):
            self.data[f'volatility_bids_distance_0_to_{i+1}'] = self.data[[f'bids_distance_{j}' for j in range(i+1)]].std(axis=1)
            self.data[f'volatility_asks_distance_0_to_{i+1}'] = self.data[[f'asks_distance_{j}' for j in range(i+1)]].std(axis=1)

    def calculate_price_impact(self):
        for i in range(1, self.n):
            self.data[f'price_impact_0_to_{i+1}'] = self.data[f'asks_distance_{i}'] - self.data[f'bids_distance_{i}']

    def calculate_bid_ask_imbalance(self):
        for i in range(self.n):
            self.data[f'imbalance_bids_0_to_{i+1}'] = self.data[[f'bids_notional_{j}' for j in range(i+1)]].sum(axis=1) - self.data[[f'asks_notional_{j}' for j in range(i+1)]].sum(axis=1)

    def calculate_market_depth_to_spread_ratio(self):
        for i in range(self.n):
            self.data[f'depth_to_spread_ratio_0_to_{i+1}'] = self.data[[f'bids_notional_{j}' for j in range(i+1)]].sum(axis=1) / (self.data[f'asks_distance_{i}'] - self.data[f'bids_distance_{i}'])

    def calculate_support_resistance(self):
        for i in range(self.n):
            self.data[f'support_level_0_to_{i+1}'] = self.data[[f'bids_notional_{j}' for j in range(i+1)]].idxmax(axis=1)
            self.data[f'resistance_level_0_to_{i+1}'] = self.data[[f'asks_notional_{j}' for j in range(i+1)]].idxmax(axis=1)

    def calculate_liquidity_pressure(self):
        for i in range(self.n):
            self.data[f'liquidity_pressure_{i}'] = self.data[f'bids_notional_{i}'] / (self.data[f'bids_distance_{i}'] + 1e-6)  # Avoid division by zero
            self.data[f'liquidity_pressure_asks_{i}'] = self.data[f'asks_notional_{i}'] / (self.data[f'asks_distance_{i}'] + 1e-6)  # Avoid division by zero

    def calculate_slippage(self):
        for i in range(self.n):  # Levels 0 to 15
            self.data[f'slippage_{i}'] = np.abs(self.data[f'asks_distance_{i}'] - self.data[f'bids_distance_{i}'])

    def order_flow_momentum(self):
        for i in range(self.n):  # Levels 0 to 15
            self.data[f'order_flow_momentum_{i}'] = self.data[f'bids_notional_{i}'].diff() + self.data[f'asks_notional_{i}'].diff()

    # Liquidity Imbalance Growth
    def liquidity_imbalance_growth(self):
        self.data['total_bid_notional'] = self.data[[f'bids_notional_{i}' for i in range(self.n)]].sum(axis=1)
        self.data['total_ask_notional'] = self.data[[f'asks_notional_{i}' for i in range(self.n)]].sum(axis=1)
        self.data['liquidity_imbalance'] = self.data['total_bid_notional'] - self.data['total_ask_notional']
        self.data['liquidity_imbalance_growth'] = self.data['liquidity_imbalance'].diff()

    # Order Book Depth Price Discovery: Measure depth of order book
    def order_book_depth(self):
        self.data['total_order_book_depth'] = self.data[[f'bids_notional_{i}' for i in range(self.n)]].sum(axis=1) + \
                                    self.data[[f'asks_notional_{i}' for i in range(self.n)]].sum(axis=1)

    # Midpoint Drift
    def midpoint_drift(self):
        self.data['midpoint_drift'] = self.data['midpoint'].diff()

    # Cancellation Pattern
    def cancellation_pattern(self):
        for i in range(self.n):  # Levels 0 to 15
            self.data[f'cancellation_ratio_{i}'] = (self.data[f'bids_cancel_notional_{i}'] + self.data[f'asks_cancel_notional_{i}']) / \
                                            (self.data[f'bids_notional_{i}'] + self.data[f'asks_notional_{i}'])

    # Cancellations and Replacements
    def cancellations_and_replacements(self):
        for i in range(self.n):  # Levels 0 to 15
            self.data[f'cancellations_and_replacements_{i}'] = self.data[f'bids_cancel_notional_{i}'] + self.data[f'asks_cancel_notional_{i}']

    # Market Depth Participation
    def market_depth_participation(self):
        self.data['total_market_depth'] = self.data[['bids_notional_0', 'bids_notional_1', 'bids_notional_2', 'bids_notional_3', 
                                        'bids_notional_4', 'bids_notional_5', 'bids_notional_6', 'bids_notional_7', 
                                        'bids_notional_8', 'bids_notional_9', 'bids_notional_10', 'bids_notional_11', 
                                        'bids_notional_12', 'bids_notional_13', 'bids_notional_14']].sum(axis=1) + \
                                    self.data[['asks_notional_0', 'asks_notional_1', 'asks_notional_2', 'asks_notional_3', 
                                        'asks_notional_4', 'asks_notional_5', 'asks_notional_6', 'asks_notional_7', 
                                        'asks_notional_8', 'asks_notional_9', 'asks_notional_10', 'asks_notional_11', 
                                        'asks_notional_12', 'asks_notional_13', 'asks_notional_14']].sum(axis=1)
        self.data['market_depth_participation'] = self.data['total_market_depth'] / (self.data['total_bid_notional'] + self.data['total_ask_notional'])

    # Liquidity-to-Volume Ratio
    def liquidity_to_volume_ratio(self):
        self.data['liquidity_to_volume_ratio'] = (self.data['total_bid_notional'] + self.data['total_bid_notional']) / self.data['spread']

    # Volatility of Spread
    def volatility_of_spread(self):
        for i in range(self.n):  # Levels 0 to 15
            self.data[f'spread_volatility_{i}'] = self.data[f'asks_distance_{i}'].rolling(window=10).std() + self.data[f'bids_distance_{i}'].rolling(window=10).std()

    # Implied Order Book Volatility
    def implied_order_book_volatility(self):
        self.data['order_book_volatility'] = (self.data[['bids_notional_0', 'bids_notional_1', 'bids_notional_2']].sum(axis=1) + 
                                    self.data[['asks_notional_0', 'asks_notional_1', 'asks_notional_2']].sum(axis=1)).rolling(window=10).std() * self.data['spread']

    # Liquidity Pressure-to-Price Ratio
    def liquidity_pressure_to_price(self):
        self.data['liquidity_pressure_to_price'] = (self.data['total_bid_notional'] + self.data['total_bid_notional']) / self.data['midpoint']

    # Liquidity Injection/Withdrawal
    def liquidity_injection_withdrawal(self):
        self.data['liquidity_injection'] = self.data[['bids_notional_0', 'bids_notional_1', 'bids_notional_2']].diff(axis=1).sum(axis=1) + \
                                    self.data[['asks_notional_0', 'asks_notional_1', 'asks_notional_2']].diff(axis=1).sum(axis=1)

    # Liquidity Shock
    def liquidity_shock(self):
        self.data['liquidity_shock'] = np.abs(self.data[['bids_notional_0', 'bids_notional_1', 'bids_notional_2']].diff().sum(axis=1)) + \
                                np.abs(self.data[['asks_notional_0', 'asks_notional_1', 'asks_notional_2']].diff().sum(axis=1))

    # Market Tightness Ratio
    def market_tightness_ratio(self):
        for i in range(0, self.n):  # Levels 1 to 15
            self.data[f'market_tightness_{i}'] = self.data['spread'] / (self.data[f'bids_notional_{i}'] + self.data[f'asks_notional_{i}'] +
                                                        self.data[f'bids_limit_notional_{i}'] + self.data[f'asks_limit_notional_{i}'] +
                                                        self.data[f'bids_market_notional_{i}'] + self.data[f'asks_market_notional_{i}'])

    # Depth-Tightness Index
    def depth_tightness_index(self):
        for i in range(0, self.n):  # Levels 1 to 15
            self.data[f'depth_tightness_{i}'] = (self.data[f'bids_notional_{i}'] + self.data[f'asks_notional_{i}'] +
                                        self.data[f'bids_limit_notional_{i}'] + self.data[f'asks_limit_notional_{i}'] +
                                        self.data[f'bids_market_notional_{i}'] + self.data[f'asks_market_notional_{i}']) / self.data['spread']

    # Bid vs Ask Imbalance Over Time
    def bid_ask_imbalance(self):
        for i in range(0, self.n):  # Levels 1 to 15
            self.data[f'bid_ask_imbalance_{i}'] = (self.data[f'bids_notional_{i}'] + self.data[f'bids_limit_notional_{i}'] +
                                            self.data[f'bids_market_notional_{i}']) - (self.data[f'asks_notional_{i}'] +
                                                                                self.data[f'asks_limit_notional_{i}'] +
                                                                                self.data[f'asks_market_notional_{i}'])

    # Relative Liquidity Pressure
    def relative_liquidity_pressure(self):
        for i in range(0, self.n):  # Levels 1 to 15
            avg_liquidity = (self.data[f'bids_notional_{i}'] + self.data[f'asks_notional_{i}'] +
                            self.data[f'bids_limit_notional_{i}'] + self.data[f'asks_limit_notional_{i}'] +
                            self.data[f'bids_market_notional_{i}'] + self.data[f'asks_market_notional_{i}']) / 2
            self.data[f'relative_liquidity_pressure_{i}'] = (self.data[f'bids_notional_{i}'] + self.data[f'bids_limit_notional_{i}'] +
                                                    self.data[f'bids_market_notional_{i}'] - 
                                                    (self.data[f'asks_notional_{i}'] + self.data[f'asks_limit_notional_{i}'] +
                                                    self.data[f'asks_market_notional_{i}'])) / avg_liquidity

    # Cumulative Depth Relative to Price Change
    def cumulative_depth_price_change(self):
        for i in range(0, self.n):  # Levels 1 to 15
            self.data[f'cumulative_depth_{i}'] = self.data[[f'bids_notional_{i}', f'asks_notional_{i}',
                                            f'bids_limit_notional_{i}', f'asks_limit_notional_{i}',
                                            f'bids_market_notional_{i}', f'asks_market_notional_{i}']].sum(axis=1)
            self.data[f'cumulative_depth_price_change_{i}'] = self.data[f'cumulative_depth_{i}'] / self.data['midpoint']

    # Cumulative Cancellations
    def cumulative_cancellations(self):
        for i in range(0, self.n):  # Levels 1 to 15
            self.data[f'cumulative_cancellations_{i}'] = self.data[[f'bids_cancel_notional_{i}', f'asks_cancel_notional_{i}']].sum(axis=1)

    # Cross-Level Imbalance
    def cross_level_imbalance(self):
        for i in range(0, self.n - 2):  # Levels 1 to 15
            self.data[f'cross_level_imbalance_{i}'] = (self.data[[f'bids_notional_{i}', f'bids_notional_{i+1}', f'bids_notional_{i+2}',
                                                    f'bids_limit_notional_{i}', f'bids_limit_notional_{i+1}', 
                                                    f'bids_limit_notional_{i+2}',
                                                    f'bids_market_notional_{i}', f'bids_market_notional_{i+1}',
                                                    f'bids_market_notional_{i+2}']].sum(axis=1) -
                                                self.data[[f'asks_notional_{i}', f'asks_notional_{i+1}', f'asks_notional_{i+2}',
                                                    f'asks_limit_notional_{i}', f'asks_limit_notional_{i+1}',
                                                    f'asks_limit_notional_{i+2}',
                                                    f'asks_market_notional_{i}', f'asks_market_notional_{i+1}',
                                                    f'asks_market_notional_{i+2}']].sum(axis=1))

    # Level Interactions and Momentum
    def level_interactions_momentum(self):
        for i in range(0, self.n):  # Levels 1 to 15
            self.data[f'level_interaction_momentum_{i}'] = self.data[[f'bids_notional_{i}', f'asks_notional_{i}',
                                                        f'bids_limit_notional_{i}', f'asks_limit_notional_{i}',
                                                        f'bids_market_notional_{i}', f'asks_market_notional_{i}']].diff(axis=1).sum(axis=1)

    # Liquidity Resilience Index
    def liquidity_resilience_index(self):
        for i in range(0, self.n):  # Levels 1 to 15
            self.data[f'liquidity_resilience_{i}'] = (self.data['spread'] / 
                                            (self.data[f'bids_notional_{i}'] + self.data[f'asks_notional_{i}'] +
                                                self.data[f'bids_limit_notional_{i}'] + self.data[f'asks_limit_notional_{i}'] +
                                                self.data[f'bids_market_notional_{i}'] + self.data[f'asks_market_notional_{i}']))
    # Liquidity Stress Indicator
    def liquidity_stress_indicator(self):
        for i in range(0, self.n):  # Levels 1 to 15
            self.data[f'liquidity_stress_{i}'] = (self.data[f'bids_notional_{i}'].diff() + self.data[f'asks_notional_{i}'].diff() +
                                        self.data[f'bids_limit_notional_{i}'].diff() + self.data[f'asks_limit_notional_{i}'].diff() +
                                        self.data[f'bids_market_notional_{i}'].diff() + self.data[f'asks_market_notional_{i}'].diff())

    # High-Frequency Activity Indicator
    def high_frequency_activity(self):
        for i in range(0, self.n):  # Levels 1 to 15
            self.data[f'high_frequency_activity_{i}'] = np.abs(self.data[f'bids_notional_{i}'].diff()) + np.abs(self.data[f'asks_notional_{i}'].diff())

    def market_liquidity(self):
        # Loop over all 15 levels (0 to 14) for the market liquidity factor
        liquidity_columns = [
            f'bids_notional_{i}' for i in range(self.n)
        ] + [
            f'asks_notional_{i}' for i in range(self.n)
        ] + [
            f'bids_market_notional_{i}' for i in range(self.n)
        ] + ['sells', 'buys']

        # Calculate the market liquidity factor
        self.data['market_liquidity_factor'] = sum(self.data[col] for col in liquidity_columns) / len(liquidity_columns)

    def price_pressure_factor(self):
        # Price Pressure Factor: Combining bid-ask distances and notional values
        pressure_columns = [
            (f'bids_distance_{i}', f'bids_limit_notional_{i}') for i in range(self.n)
        ] + [
            (f'asks_distance_{i}', f'asks_limit_notional_{i}') for i in range(self.n)
        ]

        # Calculate the price pressure factor
        self.data['price_pressure_factor'] = 0
        for bid_dist_col, bid_notional_col in pressure_columns[:15]:  # for bids
            self.data['price_pressure_factor'] += (self.data[bid_dist_col] * self.data[bid_notional_col])

        for ask_dist_col, ask_notional_col in pressure_columns[15:]:  # for asks
            self.data['price_pressure_factor'] -= (self.data[ask_dist_col] * self.data[ask_notional_col])

    def bid_ask_spread(self):
        #  **Bid-Ask Spread Factor**: Direct calculation from bid-ask distances
        self.data['bid_ask_spread'] = 0
        for i in range(self.n):
            self.data['bid_ask_spread'] += (self.data[f'bids_distance_{i}'] - self.data[f'asks_distance_{i}'])

    def supply_demand_pressure(self):
        # **Supply-Demand Pressure**: Difference between buys and sells
        self.data['supply_demand_pressure'] = self.data['buys'] - self.data['sells']

    def cancellation_factor(self):
        # **Cancellation Factor**: Assuming you have cancel notional columns like `bids_cancel_notional_0`, `asks_cancel_notional_0`
        # Example factor
        cancel_columns = [
            f'bids_cancel_notional_{i}' for i in range(self.n)
        ] + [
            f'asks_cancel_notional_{i}' for i in range(self.n)
        ]

        # Assuming we want to calculate the average cancellation factor
        self.data['cancellation_factor'] = sum(self.data[col] for col in cancel_columns) / len(cancel_columns)

    def run_all(self):
        self.calculate_cumulative_notionals()
        self.calculate_bid_ask_volatility()
        self.calculate_price_impact()
        self.calculate_bid_ask_imbalance()
        self.calculate_market_depth_to_spread_ratio()
        # self.calculate_support_resistance()
        self.calculate_liquidity_pressure()

        self.calculate_slippage()
        # self.implied_slippage()
        self.order_flow_momentum()
        self.liquidity_imbalance_growth()
        self.order_book_depth()
        self.midpoint_drift()
        self.cancellation_pattern()
        self.cancellations_and_replacements()
        self.market_depth_participation()
        self.liquidity_to_volume_ratio()
        self.volatility_of_spread()
        self.implied_order_book_volatility()

        self.liquidity_pressure_to_price()
        self.liquidity_injection_withdrawal()
        self.liquidity_shock()
        self.market_tightness_ratio()
        self.depth_tightness_index()

        self.bid_ask_imbalance()
        self.relative_liquidity_pressure()

        self.cumulative_depth_price_change()
        self.cumulative_cancellations()
        self.cross_level_imbalance()
        self.level_interactions_momentum()
        self.liquidity_resilience_index()
        self.liquidity_stress_indicator()
        self.high_frequency_activity()
        self.market_liquidity()
        self.price_pressure_factor()
        self.bid_ask_spread()
        self.supply_demand_pressure()
        self.cancellation_factor()
        # # Automatically run all methods (excluding __init__)
        # for method_name, method in self.__class__.__dict__.items():
        #     # Check if the method is callable (i.e., not an attribute) and doesn't start with '_'
        #     if callable(method) and not method_name.startswith('_') and method_name != 'get_data':
        #         print(f"Running {method_name}...")
        #         method(self)  # Call the method
        # return  

    def get_data(self):
        # To access the updated data
        return self.data



def normalize_with_quantiles(arys, q_low=0.01, q_high=0.99):
    # 计算每列的 0.01 和 0.99 分位数
    min_vals = np.quantile(arys, q_low, axis=0, keepdims=True)
    max_vals = np.quantile(arys, q_high, axis=0, keepdims=True)

    # 归一化到 ±1 之间

    arys = arys.clip(min_vals, max_vals)
    arys = 2 * (arys - min_vals) / (max_vals - min_vals) - 1
    return arys


"""label: predict the price in the future"""


def _normal_moving_average(ary, win_size=5):
    avg = ary.copy()
    avg[win_size - 1 :] = np.convolve(ary, np.ones(win_size) / win_size, mode="valid")
    return avg


def seq_to_label(ary, win_sizes=(10, 20, 40, 80, 160), if_print=False):
    labels = []
    win_sizes = np.array(win_sizes)
    offsets = np.cumsum(win_sizes)

    px_avg_0 = _normal_moving_average(ary, win_size=5)
    for win_size, offset in zip(win_sizes, offsets):
        px_avg_i = _normal_moving_average(ary, win_size=win_size)

        px_diff = px_avg_i[offset:] - px_avg_0[:-offset]

        # less_than
        lt_ary = np.quantile(px_diff, q=(0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20, 0.30, 0.40), axis=0)
        lt_ary = np.less(px_diff[:, None], lt_ary[None, :]).astype(np.float32).mean(axis=1)

        # greater_than
        gt_ary = np.quantile(px_diff, q=(0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.96, 0.98, 0.99), axis=0)
        gt_ary = np.greater(px_diff[:, None], gt_ary[None, :]).astype(np.float32).mean(axis=1)

        # merge
        label = gt_ary - lt_ary
        labels.append(label)

        """print"""
        if if_print:
            unique_values, counts = np.unique(label, return_counts=True)
            rates = np.round(counts / label.shape[0], 2)
            print(f"win_size {win_size:6} | unique_values {str(unique_values):32}  rate {rates}")

    min_length = min([label.shape[0] for label in labels])
    labels = np.concatenate([label[:min_length, None] for label in labels], axis=1)
    # assert labels.shape == (min_length, len(win_sizes))
    return labels

def run_pca_pipeline(dataframe, n_components=200):
    """
    Applies imputation, scaling, and PCA to the input DataFrame.
    
    Parameters:
        dataframe (pd.DataFrame): The input data.
        n_components (int): Number of PCA components to keep.
    
    Returns:
        X_pca (np.ndarray): Transformed PCA features.
        pipeline (Pipeline): The fitted sklearn pipeline.
        explained_variance (float): Total explained variance ratio.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components))
    ])
    
    X_pca = pipeline.fit_transform(dataframe)
    explained_variance = pipeline.named_steps['pca'].explained_variance_ratio_.sum()
    
    return X_pca, explained_variance

"""run"""


def convert_btc_csv_to_btc_npy(args=ConfigData()):
    csv_path = args.csv_path
    input_ary_path = args.input_ary_path
    label_ary_path = args.label_ary_path

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(df.columns)
    if not os.path.exists(label_ary_path):
        price_ary = df["midpoint"].values
        label_ary = seq_to_label(ary=price_ary, win_sizes=(10, 20, 30, 60, 80, 100, 200, 400), if_print=False)
        np.save(label_ary_path, label_ary)
        print(f"| save in {label_ary_path}")

    if not os.path.exists(input_ary_path):
        # indicator = TechIndicator(df=df)
        indicator = NewIndicator(df, n=15)
        alpha_arys = []
        timer0 = time.time()
        timer1 = time.time()
        indicator.run_all()
        updated_data = indicator.get_data()
        updated_data = updated_data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], errors='ignore')
        updated_data = updated_data.drop(columns=['system_time'], errors='ignore')
         # Using PCA to reduce diminish the factor to top 200 without 96 % information 
        pca_arys, explained_variance = run_pca_pipeline(updated_data)
#         alpha_arys = normalize_with_quantiles(pca_arys).astype(np.float16)
        np.save(args.input_ary_path, pca_arys)
        print(f"| save in {input_ary_path}")


if __name__ == "__main__":
    # convert_csv_to_level5_csv()
    # check_btc_1s_csv()
    convert_btc_csv_to_btc_npy()
