import pandas as pd
import numpy as np
import traceback
from src.strategies.schemas import StrategyRecipe
from src.logger import LabLogger
from src.ta_lib import TALib
from src.indicators import IndicatorEngine
import asyncio

class StrategyParser:
    def parse_and_execute(self, df: pd.DataFrame, strategy: StrategyRecipe) -> pd.DataFrame:
        """
        Applies indicators and logic to the DataFrame.
        Returns the DF with 'signal' column (-1, 0, 1).
        """
        # Async helper to log safely from sync context
        def log_sync(msg, details=None):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(LabLogger.log("BACKTEST", msg, details))
            except:
                pass

        # Work on a copy
        df = df.copy()

        # 1. Apply Indicators using robust IndicatorEngine
        for ind in strategy.indicators:
            try:
                # Prepare kwargs
                kwargs = ind.params.copy()
                if ind.col_name:
                    kwargs['col_name'] = ind.col_name

                # Use Engine to add indicator
                # It handles dispatch (HLC/HL/CV) and naming
                df = IndicatorEngine.add_custom_indicator(df, ind.name, **kwargs)

            except Exception as e:
                msg = f"Error calculating indicator '{ind.name}': {e}"
                print(msg)
                traceback.print_exc()
                log_sync(msg)

        # 2. Sanitize Column Names (Fix potential dots)
        # Pandas eval doesn't like dots in column names (interprets as attribute access)
        rename_map = {}
        for col in df.columns:
            if "." in col:
                new_col = col.replace(".", "_")
                rename_map[col] = new_col

        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # 3. Sanitize Logic Strings
        entry_logic = strategy.entry_logic
        exit_logic = strategy.exit_logic

        # Apply renames to logic strings
        for old, new in rename_map.items():
            # Basic replace (might need regex for safety, but usually sufficient)
            entry_logic = entry_logic.replace(old, new)
            exit_logic = exit_logic.replace(old, new)

        # 4. Evaluate Logic
        df['signal'] = 0

        if not entry_logic or not exit_logic:
             log_sync("Warning: Empty logic strings.", {"entry": entry_logic, "exit": exit_logic})
             return df

        try:
            # Debug: Print available columns
            # log_sync(f"DEBUG: Logic Evaluation", {"cols": df.columns.tolist(), "entry": entry_logic})

            entry_mask = df.eval(entry_logic)
            exit_mask = df.eval(exit_logic)

            # Apply signals
            df.loc[entry_mask, 'signal'] = 1
            df.loc[exit_mask, 'signal'] = -1

        except Exception as e:
            msg = f"Logic Evaluation Error: {e}"
            print(msg)
            log_sync(msg, {"entry": entry_logic, "exit": exit_logic, "cols": df.columns.tolist()})

        # Final Clean: Remove duplicate columns if any crept in
        df = df.loc[:, ~df.columns.duplicated()]
        return df
