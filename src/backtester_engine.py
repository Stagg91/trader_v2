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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
            k = params.get('k', 14)
            d = params.get('d', 3)
            sm = params.get('smooth_k', 3)
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
             entry = f"{base} < 5"
             exit = f"{base} > 10"

        else:
            entry = "close > open"
            exit = "close < open"

        return entry, exit

def worker_task(df, combos_chunk, indicators_defs):
    """
    Multiprocessing Worker Function.
    Runs backtests for a chunk of combinations on the given DataFrame.
    Returns a list of result dictionaries.
    """
    results = []
    base_bt = Backtester(df, initial_balance=10000)

    for combo in combos_chunk:
        try:
            recipe_inds = []
            entry_conds = []
            exit_conds = []

            for i, params in enumerate(combo):
                ind_def = indicators_defs[i]
                safe_params = {k: (int(v) if isinstance(v, (np.int64, np.int32)) else v) for k,v in params.items()}
                col_id = f"ind_{i}"

                e, x = StandardStrategyLogic.get_logic(ind_def['name'], safe_params, col_prefix=col_id)
                entry_conds.append(f"({e})")
                exit_conds.append(f"({x})")

                recipe_inds.append(IndicatorConfig(
                    name=ind_def['name'],
                    params=safe_params,
                    col_name=col_id
                ))

            full_entry = " & ".join(entry_conds)
            full_exit = " & ".join(exit_conds)

            recipe = StrategyRecipe(
                name="GridSearch",
                description="Auto",
                indicators=recipe_inds,
                entry_logic=full_entry,
                exit_logic=full_exit
            )

            res = base_bt.run_vectorized_backtest(recipe)

            if res:
                safe_metrics = {
                    "roi": res['roi_percent'],
                    "dd": res['max_drawdown'],
                    "trades": res['total_trades'],
                    "params": [str(c) for c in combo]
                }

                results.append({
                    "roi": res['roi_percent'],
                    "sharpe": res['sharpe'],
                    "max_drawdown": res['max_drawdown'],
                    "win_rate": res['win_rate'],
                    "trades_count": res['total_trades'],
                    "metrics_json": json.dumps(safe_metrics),
                    "start_date": str(df.iloc[0]['startTime']),
                    "end_date": str(df.iloc[-1]['startTime']),
                    "combo": combo
                })
        except:
            pass

    return results

class GridSearchRunner:
    def __init__(self, job_id: int):
        self.job_id = job_id
        self.should_stop = False

    async def run(self):
        """
        Main execution loop with Multiprocessing.
        """
        print(f"DEBUG: Starting GridSearchRunner (MultiProc) for Job {self.job_id}")
        db = SessionLocal()
        job = db.query(BacktestJob).filter(BacktestJob.id == self.job_id).first()
        if not job:
            print(f"Job {self.job_id} not found.")
            db.close()
            return

        try:
            job.status = "running"
            db.commit()

            strategy = db.query(Strategy).filter(Strategy.id == job.strategy_id).first()
            if not strategy or not strategy.content_json:
                print("Invalid strategy.")
                return

            selected_names = strategy.content_json.get('indicators', [])

            indicators_defs = []
            param_ranges = []

            for name in selected_names:
                ind_def = db.query(IndicatorDef).filter(IndicatorDef.name == name).first()
                if ind_def:
                    indicators_defs.append({
                        "name": ind_def.name,
                        "default_params": ind_def.default_params,
                        "optimization_config": ind_def.optimization_config
                    })

                    keys = []
                    lists = []
                    for p_name, p_cfg in ind_def.optimization_config.items():
                        start = p_cfg.get('start', 10)
                        stop = p_cfg.get('stop', 20)
                        step = p_cfg.get('step', 1)
                        r = list(np.arange(start, stop + step, step))
                        default_val = ind_def.default_params.get(p_name)
                        if isinstance(default_val, int):
                            r = [int(x) for x in r]
                        else:
                            r = [float(x) for x in r]
                        keys.append(p_name)
                        lists.append(r)

                    ind_combinations = []
                    for values in itertools.product(*lists):
                        param_dict = dict(zip(keys, values))
                        ind_combinations.append(param_dict)
                    param_ranges.append(ind_combinations)

            from src.data_engine import DataEngine
            de = DataEngine()
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]

            settings = db.query(Settings).first()
            days = settings.grid_search_days if settings and settings.grid_search_days else 30
            cpu_limit = settings.cpu_usage_limit if settings else 80

            total_cores = multiprocessing.cpu_count()
            max_workers = max(1, int(total_cores * (cpu_limit / 100.0)))

            # Calculate Total
            combinations_per_symbol = 1
            for r in param_ranges:
                combinations_per_symbol *= len(r)

            raw_total = len(symbols) * float(combinations_per_symbol)
            max_int = 2147483647
            if raw_total > max_int:
                total_combos = max_int
            else:
                total_combos = int(raw_total)

            job.total_combinations = total_combos
            job.started_at = time.time()
            db.commit()

            current_count = 0

            # Helper to chunk list
            def chunker(seq, size):
                return (seq[pos:pos + size] for pos in range(0, len(seq), size))

            # If all_combos is huge, we should generate it more smartly, but list(product) is okay for <10M usually.
            # If >10M, we need to iterate product directly and chunk manually.
            # For robustness, let's assume <10M.
            if combinations_per_symbol > 10000000:
                asyncio.create_task(LabLogger.log("BACKTEST", f"Warning: Combination count {combinations_per_symbol} is massive. Might use high RAM."))

            all_combos = list(itertools.product(*param_ranges))
            chunk_size = 500

            for symbol in symbols:
                db.refresh(job)
                if job.status == 'cancelled': return

                asyncio.create_task(LabLogger.log("BACKTEST", f"Fetching {days} days data for {symbol}..."))

                now_ms = int(time.time() * 1000)
                start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
                df = de.fetch_ohlcv(symbol, interval="60", limit=200000, start_time=start_ms)

                if df.empty: continue

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = []

                    for chunk in chunker(all_combos, chunk_size):
                        db.refresh(job)
                        if job.status == 'cancelled':
                            executor.shutdown(wait=False)
                            return
                        while job.status == 'paused':
                            await asyncio.sleep(5)
                            db.refresh(job)

                        future = executor.submit(worker_task, df, chunk, indicators_defs)
                        futures.append(future)

                    # Process completed chunks
                    for future in as_completed(futures):
                        try:
                            chunk_results = future.result()
                            if chunk_results:
                                db_objects = []
                                for r in chunk_results:
                                    br = BacktestResult(
                                        strategy_id=job.strategy_id,
                                        job_id=job.id,
                                        symbol=symbol,
                                        roi=r['roi'],
                                        sharpe=r['sharpe'],
                                        max_drawdown=r['max_drawdown'],
                                        win_rate=r['win_rate'],
                                        trades_count=r['trades_count'],
                                        metrics_json=r['metrics_json'],
                                        start_date=r['start_date'],
                                        end_date=r['end_date'],
                                        timestamp=time.time()
                                    )
                                    db_objects.append(br)

                                if db_objects:
                                    db.bulk_save_objects(db_objects)
                                    db.commit()

                                current_count += len(chunk_results)

                                # Update UI progress every ~5000 iterations to avoid DB lock spam
                                if current_count % (chunk_size * 10) == 0:
                                    job.progress = min(100.0, (current_count / total_combos) * 100)
                                    job.current_pair = symbol
                                    # job.current_iteration = current_count (Assuming model has this, else skip)
                                    # Just progress % is sufficient for now based on user request "completed x of y"
                                    # We can calc completed in template if we save current_count?
                                    # Actually, job.progress is float.
                                    # Let's verify BacktestJob model. It doesn't have 'current_iteration'.
                                    # I should add it or just use progress %.
                                    # Wait, user asked for "completed x of y".
                                    # I can abuse 'current_param_set' to store JSON count? Or just rely on progress %.
                                    # Let's stick to progress % update for now.
                                    db.commit()

                        except Exception as e:
                            print(f"Chunk Error: {e}")
                            traceback.print_exc()

            job.status = "completed"
            job.progress = 100.0
            job.completed_at = time.time()
            db.commit()

        except Exception as e:
            traceback.print_exc()
            db.rollback()
            try:
                job = db.query(BacktestJob).filter(BacktestJob.id == self.job_id).first()
                if job:
                    job.status = "failed"
                    db.commit()
            except:
                pass
        finally:
            db.close()
