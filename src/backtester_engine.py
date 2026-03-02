import pandas as pd
import numpy as np
import asyncio
import time
import json
import itertools
import traceback
import random
import os
from sqlalchemy.orm import Session
from src.database import SessionLocal, BacktestJob, BacktestResult, Strategy, IndicatorDef, Settings
from src.backtester import Backtester
from src.strategies.schemas import StrategyRecipe, IndicatorConfig
from src.logger import LabLogger
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# Global cache for worker process
_worker_df = None
_worker_df_path = None

class StandardStrategyLogic:
    @staticmethod
    def get_logic(indicator_name, params, col_prefix=None):
        name = indicator_name.lower()
        base = col_prefix if col_prefix else name.upper()

        entry = ""
        exit = ""

        if name == 'rsi':
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

def worker_task(data_source, combos_chunk, indicators_defs, is_file=True):
    """
    Worker Function.
    data_source: File path (str) OR DataFrame (if threading)
    """
    global _worker_df, _worker_df_path

    results = []

    try:
        # Load Data
        df = None
        if is_file:
            if _worker_df is None or _worker_df_path != data_source:
                try:
                    _worker_df = pd.read_parquet(data_source)
                    _worker_df_path = data_source
                except Exception as e:
                    return [{"error": f"Failed to load data from {data_source}: {e}"}]
            df = _worker_df
        else:
            df = data_source

        if df is None or df.empty:
             return [{"error": "Empty DataFrame passed to worker"}]

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

                if res and "error" not in res:
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
            except Exception as e:
                pass

        return results

    except Exception as e:
        return [{"error": f"Critical Worker Crash: {str(e)}"}]

class GridSearchRunner:
    def __init__(self, job_id: int):
        self.job_id = job_id
        self.should_stop = False

    def _generate_random_individual(self, indicators_defs):
        individual = []
        for ind_def in indicators_defs:
            params = {}
            for p_name, p_cfg in ind_def['optimization_config'].items():
                start = p_cfg.get('start', 10)
                stop = p_cfg.get('stop', 20)
                step = p_cfg.get('step', 1)

                default_val = ind_def['default_params'].get(p_name)
                is_int = isinstance(default_val, int)

                valid_values = list(np.arange(start, stop + step, step))
                if is_int:
                    valid_values = [int(x) for x in valid_values]
                else:
                    valid_values = [float(x) for x in valid_values]

                val = random.choice(valid_values)
                params[p_name] = val
            individual.append(params)
        return tuple(individual)

    def _mutate_individual(self, individual, indicators_defs, mutation_rate=0.2):
        new_ind = list(individual)
        for i, params in enumerate(new_ind):
            if not params: continue

            if random.random() < mutation_rate:
                ind_def = indicators_defs[i]
                new_params = params.copy()

                param_keys = list(params.keys())
                if not param_keys: continue

                p_name = random.choice(param_keys)
                p_cfg = ind_def['optimization_config'].get(p_name)
                if p_cfg:
                    start = p_cfg.get('start', 10)
                    stop = p_cfg.get('stop', 20)
                    step = p_cfg.get('step', 1)

                    current_val = params[p_name]
                    delta = random.choice([-step, step])
                    new_val = current_val + delta

                    new_val = max(start, min(stop, new_val))

                    default_val = ind_def['default_params'].get(p_name)
                    if isinstance(default_val, int):
                        new_val = int(new_val)
                    else:
                        new_val = float(new_val)

                    new_params[p_name] = new_val
                new_ind[i] = new_params
        return tuple(new_ind)

    async def run(self):
        print(f"DEBUG: Starting GridSearchRunner for Job {self.job_id}")
        db = SessionLocal()
        job = db.query(BacktestJob).filter(BacktestJob.id == self.job_id).first()
        if not job:
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

            combinations_per_symbol = 1
            for r in param_ranges:
                combinations_per_symbol *= len(r)

            raw_total = len(symbols) * float(combinations_per_symbol)
            LIMIT_FOR_GRID = 10000000

            use_genetic = False
            if raw_total > LIMIT_FOR_GRID:
                use_genetic = True
                asyncio.create_task(LabLogger.log("BACKTEST", f"Warning: Combination count {raw_total:.2e} exceeds limit. Switching to Genetic Optimization."))
                total_combos = 5000 * len(symbols)
            else:
                total_combos = int(raw_total)

            job.total_combinations = total_combos
            job.started_at = time.time()
            db.commit()

            def chunker(seq, size):
                return (seq[pos:pos + size] for pos in range(0, len(seq), size))

            # EXECUTION FLAGS
            fallback_to_threading = False

            current_count = 0

            for symbol in symbols:
                db.refresh(job)
                if job.status == 'cancelled': return

                asyncio.create_task(LabLogger.log("BACKTEST", f"Processing {symbol}..."))

                now_ms = int(time.time() * 1000)
                start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
                df = de.fetch_ohlcv(symbol, interval="60", limit=200000, start_time=start_ms)
                if df.empty: continue

                # Setup Data Passing
                temp_file = f"temp_data_{self.job_id}_{symbol}.parquet"
                df.to_parquet(temp_file)

                # Context
                ctx = multiprocessing.get_context('spawn')

                if use_genetic:
                    POP_SIZE = 100
                    GENERATIONS = 20
                    population = [self._generate_random_individual(indicators_defs) for _ in range(POP_SIZE)]

                    for gen in range(GENERATIONS):
                        db.refresh(job)
                        if job.status in ['cancelled', 'paused']:
                             if job.status == 'cancelled': return
                             while job.status == 'paused': await asyncio.sleep(5); db.refresh(job)

                        asyncio.create_task(LabLogger.log("BACKTEST", f"GA Gen {gen+1}/{GENERATIONS} | Pop: {len(population)}"))

                        chunk_size = 50
                        fitness_scores = []

                        # --- EXECUTION BLOCK ---
                        try:
                            # 1. Try Multiprocessing
                            if not fallback_to_threading:
                                with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                                    futures = []
                                    for chunk in chunker(population, chunk_size):
                                        # Pass PATH
                                        future = executor.submit(worker_task, temp_file, chunk, indicators_defs, True)
                                        futures.append(future)

                                    for future in as_completed(futures):
                                        res_list = future.result()
                                        if res_list:
                                            # Handle Logic Error
                                            if isinstance(res_list[0], dict) and "error" in res_list[0]:
                                                # If it's a logic error, logging it is enough.
                                                # If it's a crash, pool usually catches it.
                                                continue

                                            for r in res_list:
                                                ind = r['combo']
                                                roi = r['roi']
                                                dd = r['max_drawdown']
                                                fitness = roi - (dd * 0.5)
                                                fitness_scores.append((ind, fitness))

                                                # Save only high performers to DB to save space? Or all? All for matrix.
                                                br = BacktestResult(
                                                    strategy_id=job.strategy_id, job_id=job.id, symbol=symbol,
                                                    roi=roi, sharpe=r['sharpe'], max_drawdown=dd, win_rate=r['win_rate'],
                                                    trades_count=r['trades_count'], metrics_json=r['metrics_json'],
                                                    start_date=r['start_date'], end_date=r['end_date'], timestamp=time.time()
                                                )
                                                # Single save to avoid complexity in fallback logic refactor
                                                db.add(br)
                                            db.commit()
                                            current_count += len(res_list)

                            # 2. Try Threading if fallback is active
                            else:
                                raise RuntimeError("Force Threading Fallback")

                        except Exception as e:
                            # Catch Pool Crash or Forced Fallback
                            print(f"Pool Error: {e}. Falling back to Threading.")
                            fallback_to_threading = True
                            asyncio.create_task(LabLogger.log("BACKTEST", f"Switched to Threading Mode due to stability issues."))

                            # Rerun current chunks with Threading
                            # Pass DF directly
                            fitness_scores = [] # Reset for this gen
                            with ThreadPoolExecutor(max_workers=1) as executor: # Serial safety
                                futures = []
                                for chunk in chunker(population, chunk_size):
                                    future = executor.submit(worker_task, df, chunk, indicators_defs, False)
                                    futures.append(future)
                                for future in as_completed(futures):
                                    res_list = future.result()
                                    if res_list and not (isinstance(res_list[0], dict) and "error" in res_list[0]):
                                        for r in res_list:
                                            ind = r['combo']
                                            roi = r['roi']
                                            dd = r['max_drawdown']
                                            fitness = roi - (dd * 0.5)
                                            fitness_scores.append((ind, fitness))
                                            br = BacktestResult(
                                                strategy_id=job.strategy_id, job_id=job.id, symbol=symbol,
                                                roi=roi, sharpe=r['sharpe'], max_drawdown=dd, win_rate=r['win_rate'],
                                                trades_count=r['trades_count'], metrics_json=r['metrics_json'],
                                                start_date=r['start_date'], end_date=r['end_date'], timestamp=time.time()
                                            )
                                            db.add(br)
                                        db.commit()
                                        current_count += len(res_list)

                        # --- EVOLUTION LOGIC ---
                        fitness_scores.sort(key=lambda x: x[1], reverse=True)
                        top_performers = [x[0] for x in fitness_scores[:int(POP_SIZE * 0.2)]]

                        if not top_performers:
                            top_performers = [self._generate_random_individual(indicators_defs) for _ in range(10)]

                        new_pop = list(top_performers)
                        while len(new_pop) < POP_SIZE:
                            parent = random.choice(top_performers)
                            child = self._mutate_individual(parent, indicators_defs, mutation_rate=0.3)
                            new_pop.append(child)
                        population = new_pop

                        sym_idx = symbols.index(symbol)
                        progress_per_sym = 100.0 / len(symbols)
                        base_progress = sym_idx * progress_per_sym
                        gen_progress = ((gen + 1) / GENERATIONS) * progress_per_sym
                        job.progress = min(100.0, base_progress + gen_progress)
                        mode_str = "Thr" if fallback_to_threading else "Proc"
                        job.current_pair = f"{symbol} (Gen {gen+1}) [{mode_str}]"
                        db.commit()

                # Cleanup temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)

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
