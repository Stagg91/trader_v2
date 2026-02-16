import pandas as pd
import numpy as np
import asyncio
import time
import json
import itertools
import traceback
from sqlalchemy.orm import Session
from src.database import SessionLocal, BacktestJob, BacktestResult, Strategy, IndicatorDef, Settings
from src.backtester import Backtester
from src.strategies.schemas import StrategyRecipe, IndicatorConfig
from src.logger import LabLogger

class StandardStrategyLogic:
    @staticmethod
    def get_logic(indicator_name, params, col_prefix=None):
        """
        Returns (entry_logic, exit_logic) strings for a given indicator and params.
        col_prefix: If provided, use this prefix for columns (e.g. 'RSI_14').
        """
        name = indicator_name.lower()
        base = col_prefix if col_prefix else name.upper()

        entry = ""
        exit = ""

        if name == 'rsi':
            # Default Mean Reversion / Oversold
            entry = f"{base} < 30"
            exit = f"{base} > 70"

        elif name == 'stoch':
            # Returns k, d. STOCHk_..., STOCHd_...
            # If col_prefix is provided (e.g. 'ind_0'), StrategyParser will prefix columns:
            # ind_0_STOCHk_..., ind_0_STOCHd_...
            # This is complex because TALib returns fixed names.
            # StrategyParser logic:
            # if ind.col_name:
            #    if col_name in result.columns: ...
            #    else: result = result.add_prefix(f"{ind.col_name}_")

            # So if we use col_prefix='ind_0', columns become 'ind_0_STOCHk_14_3_3' etc.
            # We need to construct these names here.
            k = params.get('k', 14)
            d = params.get('d', 3)
            sm = params.get('smooth_k', 3)

            # We need to match TALib's naming exactly to build logic
            # TALib: f'STOCHk_{k}_{d}_{sm}'
            suffix = f"_{k}_{d}_{sm}"
            col_k = f"{base}_STOCHk{suffix}"
            col_d = f"{base}_STOCHd{suffix}"

            entry = f"{col_k} < 20"
            exit = f"{col_k} > 80"

        elif name == 'macd':
            fast = params.get('fast', 12)
            slow = params.get('slow', 26)
            sig = params.get('signal', 9)
            suffix = f"_{fast}_{slow}_{sig}"

            col_macd = f"{base}_MACD{suffix}"
            col_sig = f"{base}_MACDs{suffix}"

            entry = f"{col_macd} > {col_sig}"
            exit = f"{col_macd} < {col_sig}"

        elif name == 'bbands':
            l = params.get('length', 20)
            s = params.get('std', 2.0)
            suffix = f"_{l}_{s}"

            col_l = f"{base}_BBL{suffix}"
            col_u = f"{base}_BBU{suffix}"

            entry = f"close < {col_l}"
            exit = f"close > {col_u}"

        elif name == 'cci':
            entry = f"{base} < -100"
            exit = f"{base} > 100"

        elif name == 'williams_r':
            entry = f"{base} < -80"
            exit = f"{base} > -20"

        elif name == 'mfi':
            entry = f"{base} < 20"
            exit = f"{base} > 80"

        elif name in ['sma', 'ema', 'psar']:
            # Trend Following: Price vs Indicator
            entry = f"close > {base}"
            exit = f"close < {base}"

        elif name == 'adx':
            l = params.get('length', 14)
            col_adx = f"{base}_ADX_{l}"
            col_pos = f"{base}_DMP_{l}"
            col_neg = f"{base}_DMN_{l}"

            entry = f"({col_adx} > 25) & ({col_pos} > {col_neg})"
            exit = f"({col_adx} > 25) & ({col_neg} > {col_pos})"

        elif name == 'vortex':
            l = params.get('length', 14)
            col_p = f"{base}_VI_pos_{l}"
            col_n = f"{base}_VI_neg_{l}"

            entry = f"{col_p} > {col_n}"
            exit = f"{col_n} > {col_p}"

        elif name in ['keltner', 'donchian']:
            l = params.get('length', 20)
            if name == 'keltner':
                mult = params.get('mult', 2.0)
                col_u = f"{base}_KC_U_{l}_{mult}"
                col_l = f"{base}_KC_L_{l}_{mult}"
            else:
                col_u = f"{base}_DC_U_{l}"
                col_l = f"{base}_DC_L_{l}"

            entry = f"close > {col_u}"
            exit = f"close < {col_l}"

        elif name in ['ao', 'roc', 'force', 'eom', 'cmf', 'trix']:
            entry = f"{base} > 0"
            exit = f"{base} < 0"

        elif name == 'ulcer':
             # Risk indicator, not signal usually.
             entry = f"{base} < 5"
             exit = f"{base} > 10"

        else:
            # Fallback
            entry = "close > open"
            exit = "close < open"

        return entry, exit

class GridSearchRunner:
    def __init__(self, job_id: int):
        self.job_id = job_id
        self.should_stop = False

    async def run(self):
        """
        Main execution loop.
        """
        db = SessionLocal()
        job = db.query(BacktestJob).filter(BacktestJob.id == self.job_id).first()
        if not job:
            print(f"Job {self.job_id} not found.")
            db.close()
            return

        try:
            strategy = db.query(Strategy).filter(Strategy.id == job.strategy_id).first()
            if not strategy or not strategy.content_json:
                print("Invalid strategy.")
                return

            selected_names = strategy.content_json.get('indicators', [])

            # Fetch Indicator Defs for ranges
            indicators = []
            param_ranges = []

            for name in selected_names:
                ind_def = db.query(IndicatorDef).filter(IndicatorDef.name == name).first()
                if ind_def:
                    indicators.append(ind_def)
                    # Generate ranges
                    keys = []
                    lists = []
                    for p_name, p_cfg in ind_def.optimization_config.items():
                        start = p_cfg.get('start', 10)
                        stop = p_cfg.get('stop', 20)
                        step = p_cfg.get('step', 1)
                        # Create range (inclusive of stop)
                        r = list(np.arange(start, stop + step, step)) # Ensure float for arange

                        # Cast to int/float based on default param type
                        default_val = ind_def.default_params.get(p_name)
                        if isinstance(default_val, int):
                            r = [int(x) for x in r]
                        else:
                            r = [float(x) for x in r]

                        keys.append(p_name)
                        lists.append(r)

                    # Cartesian product of params for THIS indicator
                    ind_combinations = []
                    for values in itertools.product(*lists):
                        param_dict = dict(zip(keys, values))
                        ind_combinations.append(param_dict)

                    param_ranges.append(ind_combinations)

            # Load Data
            from src.data_engine import DataEngine
            de = DataEngine()
            # Fallback symbols
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]

            # Update Job Status
            job.status = "running"
            # Ensure int cast for SQLAlchemy
            # Fix OverflowError for large grid searches
            raw_total = len(symbols) * float(np.prod([len(x) for x in param_ranges]))
            max_int = 2**63 - 1
            if raw_total > max_int:
                total_combos = max_int # Clamp to max SQLite int
                asyncio.create_task(LabLogger.log("BACKTEST", f"Warning: Total combinations ({raw_total}) exceeds DB limit. Display clamped."))
            else:
                total_combos = int(raw_total)

            job.total_combinations = total_combos
            db.commit()

            current_count = 0

            # Loop Pairs
            for symbol in symbols:
                # Load Data Once per pair
                df = de.fetch_ohlcv(symbol, interval="60", limit=1000)
                if df.empty: continue

                # Loop Combinations
                for combo in itertools.product(*param_ranges):
                    # Check Pause/Stop
                    db.refresh(job)
                    if job.status == 'paused':
                        while job.status == 'paused':
                            await asyncio.sleep(5)
                            db.refresh(job)
                    if job.status == 'cancelled':
                        return

                    # Throttling
                    settings = db.query(Settings).first()
                    limit = settings.cpu_usage_limit if settings else 80
                    delay = (100 - limit) / 1000.0
                    if delay > 0:
                        await asyncio.sleep(delay)

                    # LOGGING: Emit progress log to LabLogger
                    if current_count % 5 == 0:
                        # Async log (fire and forget)
                        asyncio.create_task(LabLogger.log("BACKTEST", f"Job #{self.job_id} | {symbol} | Combo: {combo}"))

                    # Construct Recipe
                    recipe_inds = []
                    entry_conds = []
                    exit_conds = []

                    for i, params in enumerate(combo):
                        ind_def = indicators[i]

                        # Fix params types
                        safe_params = {k: (int(v) if isinstance(v, (np.int64, np.int32)) else v) for k,v in params.items()}

                        # Unique Column Name: ind_{i}
                        # This ensures safety against collision and predictable naming for logic
                        col_id = f"ind_{i}"

                        # Generate Logic
                        e, x = StandardStrategyLogic.get_logic(ind_def.name, safe_params, col_prefix=col_id)
                        entry_conds.append(f"({e})")
                        exit_conds.append(f"({x})")

                        recipe_inds.append(IndicatorConfig(
                            name=ind_def.name,
                            params=safe_params,
                            col_name=col_id # Explicit Name
                        ))

                    # Combine Logic
                    full_entry = " & ".join(entry_conds)
                    full_exit = " & ".join(exit_conds)

                    recipe = StrategyRecipe(
                        name="GridSearch",
                        description="Auto",
                        indicators=recipe_inds,
                        entry_logic=full_entry,
                        exit_logic=full_exit
                    )

                    # Run Backtest
                    bt = Backtester(df, initial_balance=10000)
                    res = bt.run_vectorized_backtest(recipe)

                    if res and res.get('total_trades', 0) > 0:
                        safe_metrics = {
                            "roi": res['roi_percent'],
                            "dd": res['max_drawdown'],
                            "trades": res['total_trades'],
                            "params": [str(c) for c in combo]
                        }

                        br = BacktestResult(
                            strategy_id=job.strategy_id,
                            job_id=job.id,
                            symbol=symbol,
                            start_date=str(df.iloc[0]['startTime']),
                            end_date=str(df.iloc[-1]['startTime']),
                            roi=res['roi_percent'],
                            sharpe=res['sharpe'],
                            max_drawdown=res['max_drawdown'],
                            win_rate=res['win_rate'],
                            trades_count=res['total_trades'],
                            metrics_json=json.dumps(safe_metrics),
                            timestamp=time.time()
                        )
                        db.add(br)
                        db.commit()

                        # LOGGING: Emit positive result
                        if res['roi_percent'] > 0:
                             asyncio.create_task(LabLogger.log("BACKTEST", f"Job #{self.job_id} HIT: {symbol} | ROI {res['roi_percent']:.2f}% | Params: {combo}"))

                    current_count += 1

                    # Update Progress
                    if current_count % 10 == 0:
                        job.progress = (current_count / total_combos) * 100
                        job.current_pair = symbol
                        db.commit()

            job.status = "completed"
            job.progress = 100.0
            job.completed_at = time.time()
            db.commit()

        except Exception as e:
            traceback.print_exc()
            job.status = "failed"
            db.commit()
        finally:
            db.close()
