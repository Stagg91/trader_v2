import pandas as pd
import numpy as np

class TALib:
    """
    Lightweight Technical Analysis library using pure Pandas/Numpy.
    Replaces pandas_ta to avoid numba/Python 3.14 incompatibility.
    """

    @staticmethod
    def sma(series: pd.Series, length: int) -> pd.Series:
        return series.rolling(window=length).mean()

    @staticmethod
    def ema(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
        avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        fast_ema = TALib.ema(series, fast)
        slow_ema = TALib.ema(series, slow)
        macd_line = fast_ema - slow_ema
        signal_line = TALib.ema(macd_line, signal)
        hist = macd_line - signal_line

        return pd.DataFrame({
            f'MACD_{fast}_{slow}_{signal}': macd_line,
            f'MACDs_{fast}_{slow}_{signal}': signal_line,
            f'MACDh_{fast}_{slow}_{signal}': hist
        })

    @staticmethod
    def bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
        mid = TALib.sma(series, length)
        sigma = series.rolling(window=length).std()
        upper = mid + std * sigma
        lower = mid - std * sigma
        std_str = str(std)
        return pd.DataFrame({
            f'BBL_{length}_{std_str}': lower,
            f'BBM_{length}_{std_str}': mid,
            f'BBU_{length}_{std_str}': upper
        })

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up = high - high.shift(1)
        down = low.shift(1) - low

        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        alpha = 1/length
        trs = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / trs)
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / trs)

        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        return pd.DataFrame({
            f'ADX_{length}': adx,
            f'DMP_{length}': plus_di,
            f'DMN_{length}': minus_di
        })

    # --- New Indicators ---

    @staticmethod
    def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3, smooth_k: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k).min()
        highest_high = high.rolling(window=k).max()

        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        # Optional Smoothing on K? Standard is just K
        if smooth_k > 1:
            stoch_k = stoch_k.rolling(window=smooth_k).mean()

        stoch_d = stoch_k.rolling(window=d).mean()

        return pd.DataFrame({
            f'STOCHk_{k}_{d}_{smooth_k}': stoch_k,
            f'STOCHd_{k}_{d}_{smooth_k}': stoch_d
        })

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=length).max()
        lowest_low = low.rolling(window=length).min()
        r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return r

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=length).mean()
        mad = (tp - sma_tp).abs().rolling(window=length).mean()
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci

    @staticmethod
    def roc(series: pd.Series, length: int = 9) -> pd.Series:
        """Rate of Change"""
        return 100 * (series.diff(length) / series.shift(length))

    @staticmethod
    def psar(high: pd.Series, low: pd.Series, close: pd.Series, af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Parabolic SAR - Simplified Iterative Implementation (Slow in Python loop but correct logic)"""
        # Vectorized is extremely hard for PSAR due to state dependence.
        # Using a numba-free iterative approach.
        psar = close.copy()
        psar_vals = np.zeros(len(close))

        bullish = True
        af = af_start
        ep = high.iloc[0] # Extreme Point
        psar_val = low.iloc[0]

        psar_vals[0] = psar_val

        high_arr = high.values
        low_arr = low.values

        for i in range(1, len(close)):
            prev_psar = psar_vals[i-1]
            if bullish:
                psar_val = prev_psar + af * (ep - prev_psar)
                psar_val = min(psar_val, low_arr[i-1], low_arr[i-2] if i > 1 else low_arr[i-1])
            else:
                psar_val = prev_psar + af * (ep - prev_psar)
                psar_val = max(psar_val, high_arr[i-1], high_arr[i-2] if i > 1 else high_arr[i-1])

            # Switch Trend
            trend_reversed = False
            if bullish:
                if low_arr[i] < psar_val:
                    bullish = False
                    trend_reversed = True
                    psar_val = ep
                    ep = low_arr[i]
                    af = af_start
            else:
                if high_arr[i] > psar_val:
                    bullish = True
                    trend_reversed = True
                    psar_val = ep
                    ep = high_arr[i]
                    af = af_start

            if not trend_reversed:
                if bullish:
                    if high_arr[i] > ep:
                        ep = high_arr[i]
                        af = min(af + af_start, af_max)
                else:
                    if low_arr[i] < ep:
                        ep = low_arr[i]
                        af = min(af + af_start, af_max)

            psar_vals[i] = psar_val

        return pd.Series(psar_vals, index=close.index)

    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
        """Money Flow Index"""
        tp = (high + low + close) / 3
        rmf = tp * volume # Raw Money Flow

        # Positive/Negative Flow
        diff = tp.diff()
        pos_mf = np.where(diff > 0, rmf, 0.0)
        neg_mf = np.where(diff < 0, rmf, 0.0)

        pos_mf = pd.Series(pos_mf, index=close.index).rolling(window=length).sum()
        neg_mf = pd.Series(neg_mf, index=close.index).rolling(window=length).sum()

        mfi = 100 - (100 / (1 + (pos_mf / neg_mf)))
        return mfi

    @staticmethod
    def keltner(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
        """Keltner Channels"""
        mid = TALib.ema(close, length)
        atr_val = TALib.atr(high, low, close, length) # Using default ATR logic inside but passing same length
        upper = mid + mult * atr_val
        lower = mid - mult * atr_val

        return pd.DataFrame({
            f'KC_L_{length}_{mult}': lower,
            f'KC_M_{length}_{mult}': mid,
            f'KC_U_{length}_{mult}': upper
        })

    @staticmethod
    def donchian(high: pd.Series, low: pd.Series, length: int = 20) -> pd.DataFrame:
        """Donchian Channels"""
        upper = high.rolling(window=length).max()
        lower = low.rolling(window=length).min()
        mid = (upper + lower) / 2

        return pd.DataFrame({
            f'DC_L_{length}': lower,
            f'DC_M_{length}': mid,
            f'DC_U_{length}': upper
        })

    @staticmethod
    def ulcer(close: pd.Series, length: int = 14) -> pd.Series:
        """Ulcer Index"""
        max_close = close.rolling(window=length).max()
        pct_drawdown = 100 * (close - max_close) / max_close
        sq_dd = pct_drawdown ** 2
        return np.sqrt(sq_dd.rolling(window=length).mean())

    @staticmethod
    def force(close: pd.Series, volume: pd.Series, length: int = 13) -> pd.Series:
        """Force Index"""
        fi = close.diff(1) * volume
        return TALib.ema(fi, length)

    @staticmethod
    def eom(high: pd.Series, low: pd.Series, volume: pd.Series, length: int = 14, div: float = 100000000) -> pd.Series:
        """Ease of Movement"""
        dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
        br = (volume / div) / (high - low)
        eom = dm / br
        return TALib.sma(eom, length)

    @staticmethod
    def ao(high: pd.Series, low: pd.Series) -> pd.Series:
        """Awesome Oscillator"""
        mid = (high + low) / 2
        ao = TALib.sma(mid, 5) - TALib.sma(mid, 34)
        return ao

    @staticmethod
    def vortex(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
        """Vortex Indicator"""
        tr = TALib.atr(high, low, close, 1).rolling(window=length).sum() # Sum of 1-period TR

        vm_plus = (high - low.shift(1)).abs().rolling(window=length).sum()
        vm_minus = (low - high.shift(1)).abs().rolling(window=length).sum()

        vi_plus = vm_plus / tr
        vi_minus = vm_minus / tr

        return pd.DataFrame({
            f'VI_pos_{length}': vi_plus,
            f'VI_neg_{length}': vi_minus
        })

    @staticmethod
    def trix(close: pd.Series, length: int = 15) -> pd.Series:
        """TRIX"""
        ema1 = TALib.ema(close, length)
        ema2 = TALib.ema(ema1, length)
        ema3 = TALib.ema(ema2, length)
        return 100 * (ema3.diff() / ema3.shift(1))

    @staticmethod
    def cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
        """Chaikin Money Flow"""
        mf_mult = ((close - low) - (high - close)) / (high - low)
        mf_vol = mf_mult * volume
        cmf = mf_vol.rolling(window=length).sum() / volume.rolling(window=length).sum()
        return cmf
