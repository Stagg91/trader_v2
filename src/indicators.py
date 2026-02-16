import pandas as pd
from src.ta_lib import TALib
import inspect

class IndicatorEngine:
    @staticmethod
    def add_indicators(df: pd.DataFrame):
        """
        Adds default technical indicators to the DataFrame (used for Dashboards).
        """
        if df.empty:
            return df

        # MACD
        macd = TALib.macd(df['close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)

        # RSI
        df['RSI_14'] = TALib.rsi(df['close'], length=14)

        # Bollinger Bands
        bbands = TALib.bbands(df['close'], length=20, std=2)
        df = pd.concat([df, bbands], axis=1)

        return df

    @staticmethod
    def add_custom_indicator(df: pd.DataFrame, indicator_name: str, **kwargs):
        """
        Dynamically adds an indicator based on name and dispatches to appropriate TALib method signatures.
        """
        if not hasattr(TALib, indicator_name):
            print(f"Indicator {indicator_name} not found in TALib")
            return df

        method = getattr(TALib, indicator_name)
        col_name = kwargs.pop('col_name', None)

        # Check required columns
        required_cols = ['close']
        if indicator_name in ['atr', 'adx', 'stoch', 'williams_r', 'cci', 'psar', 'vortex', 'keltner', 'ao', 'donchian', 'mfi', 'cmf', 'eom']:
            required_cols.extend(['high', 'low'])
        if indicator_name in ['mfi', 'cmf', 'force', 'eom']:
            required_cols.append('volume')

        for col in required_cols:
            if col not in df.columns:
                # print(f"Missing column {col} for indicator {indicator_name}")
                return df

        try:
            # Filter kwargs to match method signature
            sig = inspect.signature(method)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

            # Dispatch based on known signatures
            if indicator_name in ['atr', 'adx', 'stoch', 'williams_r', 'cci', 'psar', 'vortex', 'keltner']:
                result = method(df['high'], df['low'], df['close'], **valid_kwargs)
            elif indicator_name in ['ao', 'donchian']:
                result = method(df['high'], df['low'], **valid_kwargs)
            elif indicator_name in ['mfi', 'cmf']:
                result = method(df['high'], df['low'], df['close'], df['volume'], **valid_kwargs)
            elif indicator_name in ['force']:
                result = method(df['close'], df['volume'], **valid_kwargs)
            elif indicator_name in ['eom']:
                result = method(df['high'], df['low'], df['volume'], **valid_kwargs)
            else:
                # sma, ema, rsi, macd, bbands, roc, ulcer, trix
                result = method(df['close'], **valid_kwargs)

            if isinstance(result, pd.Series):
                final_name = col_name if col_name else f"{indicator_name}"
                df[final_name] = result
            elif isinstance(result, pd.DataFrame):
                if col_name:
                    result = result.add_prefix(f"{col_name}_")
                df = pd.concat([df, result], axis=1)

        except Exception as e:
            print(f"Error adding custom indicator {indicator_name}: {e}")
            pass

        return df
