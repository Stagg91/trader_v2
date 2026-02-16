import pandas as pd
import numpy as np
import asyncio
import time
import json
import itertools
import traceback
import random
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

    def _generate_random_individual(self, indicators_defs):
        """Generates a random parameter set for the given indicators."""
        individual = []
        for ind_def in indicators_defs:
            params = {}
            for p_name, p_cfg in ind_def['optimization_config'].items():
                start = p_cfg.get('start', 10)
                stop = p_cfg.get('stop', 20)
                step = p_cfg.get('step', 1)

                # Random choice within range
                # Use int/float based on default type
                default_val = ind_def['default_params'].get(p_name)
                is_int = isinstance(default_val, int)

                # Generate valid steps
                valid_values = list(np.arange(start, stop + step, step))
                if is_int:
                    valid_values = [int(x) for x in valid_values]
                else:
                    valid_values = [float(x) for x in valid_values]

                val = random.choice(valid_values)
                params[p_name] = val
            individual.append(params)
        return tuple(individual) # Tuple for hashability if needed

    def _mutate_individual(self, individual, indicators_defs, mutation_rate=0.2):
        """Mutates an individual slightly."""
        new_ind = list(individual)
        for i, params in enumerate(new_ind):
            if random.random() < mutation_rate:
                # Mutate params for this indicator
                ind_def = indicators_defs[i]
                new_params = params.copy()

                # Pick one param to change
                p_name = random.choice(list(params.keys()))
                p_cfg = ind_def['optimization_config'].get(p_name)
                if p_cfg:
                    start = p_cfg.get('start', 10)
                    stop = p_cfg.get('stop', 20)
                    step = p_cfg.get('step', 1)

                    current_val = params[p_name]
                    # +/- step
                    delta = random.choice([-step, step])
                    new_val = current_val + delta

                    # Clamp
                    new_val = max(start, min(stop, new_val))

                    # Type check
                    default_val = ind_def['default_params'].get(p_name)
                    if isinstance(default_val, int):
                        new_val = int(new_val)
                    else:
                        new_val = float(new_val)

                    new_params[p_name] = new_val
                new_ind[i] = new_params
        return tuple(new_ind)

    async def run(self):
        """
        Main execution loop. Switches between Grid Search and Genetic Algorithm based on size.
        """
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

            # 1. Fetch Definitions
            indicators_defs = []
            param_ranges = [] # For grid search

            for name in selected_names:
                ind_def = db.query(IndicatorDef).filter(IndicatorDef.name == name).first()
                if ind_def:
                    indicators_defs.append({
                        "name": ind_def.name,
                        "default_params": ind_def.default_params,
                        "optimization_config": ind_def.optimization_config
                    })

                    # For Grid Search Calculation
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

            # 2. Determine Scope
            from src.data_engine import DataEngine
            de = DataEngine()
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"]

            settings = db.query(Settings).first()
            days = settings.grid_search_days if settings and settings.grid_search_days else 30
            cpu_limit = settings.cpu_usage_limit if settings else 80

            total_cores = multiprocessing.cpu_count()
            max_workers = max(1, int(total_cores * (cpu_limit / 100.0)))

            # Calculate Total Size
            combinations_per_symbol = 1
            for r in param_ranges:
                combinations_per_symbol *= len(r)

            raw_total = len(symbols) * float(combinations_per_symbol)
            LIMIT_FOR_GRID = 10000000 # 10 Million

            use_genetic = False
            if raw_total > LIMIT_FOR_GRID:
                use_genetic = True
                asyncio.create_task(LabLogger.log("BACKTEST", f"Warning: Combination count {raw_total:.2e} exceeds limit. Switching to Genetic Optimization."))
                # Cap progress bar for GA
                total_combos = 5000 * len(symbols) # e.g. 50 gens * 100 pop
            else:
                total_combos = int(raw_total)

            job.total_combinations = total_combos
            job.started_at = time.time()
            db.commit()

            # 3. Execution Logic

            # Helper to chunk list
            def chunker(seq, size):
                return (seq[pos:pos + size] for pos in range(0, len(seq), size))

            # Loop Pairs
            current_count = 0

            for symbol in symbols:
                db.refresh(job)
                if job.status == 'cancelled': return

                asyncio.create_task(LabLogger.log("BACKTEST", f"Processing {symbol}..."))

                # Fetch Data
                now_ms = int(time.time() * 1000)
                start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
                df = de.fetch_ohlcv(symbol, interval="60", limit=200000, start_time=start_ms)
                if df.empty: continue

                # === GENETIC ALGORITHM PATH ===
                if use_genetic:
                    POP_SIZE = 100
                    GENERATIONS = 20 # 2000 evals per symbol

                    # Init Population (Random)
                    population = [self._generate_random_individual(indicators_defs) for _ in range(POP_SIZE)]

                    for gen in range(GENERATIONS):
                        db.refresh(job)
                        if job.status in ['cancelled', 'paused']:
                             if job.status == 'cancelled': return
                             while job.status == 'paused': await asyncio.sleep(5); db.refresh(job)

                        asyncio.create_task(LabLogger.log("BACKTEST", f"GA Gen {gen+1}/{GENERATIONS} | Pop: {len(population)}"))

                        # Evaluate Population (Parallel)
                        chunk_size = 50 # Half pop
                        fitness_scores = [] # (ind, roi)

                        with ProcessPoolExecutor(max_workers=max_workers) as executor:
                            futures = []
                            for chunk in chunker(population, chunk_size):
                                future = executor.submit(worker_task, df, chunk, indicators_defs)
                                futures.append(future)

                            for future in as_completed(futures):
                                try:
                                    res_list = future.result()
                                    if res_list:
                                        # Save Results
                                        db_objects = []
                                        for r in res_list:
                                            # Reconstruct individual from r['combo'] if needed, or just use r['roi']
                                            # We need to map back to original individual for breeding?
                                            # worker_task returns 'combo'. This is our individual.
                                            ind = r['combo']
                                            roi = r['roi']
                                            dd = r['max_drawdown']
                                            fitness = roi - (dd * 0.5) # Simple fitness function

                                            fitness_scores.append((ind, fitness))

                                            br = BacktestResult(
                                                strategy_id=job.strategy_id, job_id=job.id, symbol=symbol,
                                                roi=roi, sharpe=r['sharpe'], max_drawdown=dd, win_rate=r['win_rate'],
                                                trades_count=r['trades_count'], metrics_json=r['metrics_json'],
                                                start_date=r['start_date'], end_date=r['end_date'], timestamp=time.time()
                                            )
                                            db_objects.append(br)

                                        if db_objects:
                                            db.bulk_save_objects(db_objects)
                                            db.commit()

                                        current_count += len(res_list)

                                except Exception as e:
                                    traceback.print_exc()

                        # Selection (Elitism + Roulette?)
                        # Sort by fitness desc
                        fitness_scores.sort(key=lambda x: x[1], reverse=True)
                        top_performers = [x[0] for x in fitness_scores[:int(POP_SIZE * 0.2)]] # Top 20%

                        if not top_performers:
                            # Re-seed if all failed
                            top_performers = [self._generate_random_individual(indicators_defs) for _ in range(10)]

                        # Breeding (Crossover + Mutation)
                        new_pop = list(top_performers) # Elitism

                        while len(new_pop) < POP_SIZE:
                            parent = random.choice(top_performers)
                            child = self._mutate_individual(parent, indicators_defs, mutation_rate=0.3)
                            new_pop.append(child)

                        population = new_pop

                        # Update Progress
                        job.progress = min(100.0, (current_count / total_combos) * 100)
                        job.current_pair = f"{symbol} (Gen {gen})"
                        db.commit()

                # === GRID SEARCH PATH ===
                else:
                    # Use itertools.product directly with chunking to avoid huge list in memory
                    # Problem: We can't slice a product object easily.
                    # Solution: Helper generator that chunks the product iterator.

                    product_iter = itertools.product(*param_ranges)

                    # Custom chunker for iterator
                    def iter_chunker(iterable, size):
                        it = iter(iterable)
                        while True:
                            chunk = list(itertools.islice(it, size))
                            if not chunk:
                                break
                            yield chunk

                    chunk_size = 500

                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        # Limit pending futures to avoid memory bloat
                        MAX_PENDING_FUTURES = max_workers * 2

                        chunk_gen = iter_chunker(product_iter, chunk_size)

                        # Initial fill
                        for _ in range(MAX_PENDING_FUTURES):
                            try:
                                chunk = next(chunk_gen)
                                future = executor.submit(worker_task, df, chunk, indicators_defs)
                                futures.append(future)
                            except StopIteration:
                                break

                        while futures:
                            # Wait for at least one to complete
                            # Ideally we use as_completed but we want to refill continuously
                            # Simple approach: Wait for first completed, process, submit next
                            # Actually, as_completed yields futures as they finish.

                            # Let's use a set for active futures
                            active_futures = set(futures)
                            completed_futures = []

                            # Wait for ONE result
                            done, not_done = multiprocessing.connection.wait(active_futures, timeout=0.1) # Wait logic?
                            # concurrent.futures.wait is better
                            from concurrent.futures import wait, FIRST_COMPLETED
                            done, not_done = wait(active_futures, return_when=FIRST_COMPLETED)

                            for future in done:
                                active_futures.remove(future)
                                try:
                                    chunk_results = future.result()
                                    if chunk_results:
                                        db_objects = []
                                        for r in chunk_results:
                                            br = BacktestResult(
                                                strategy_id=job.strategy_id, job_id=job.id, symbol=symbol,
                                                roi=r['roi'], sharpe=r['sharpe'], max_drawdown=r['max_drawdown'],
                                                win_rate=r['win_rate'], trades_count=r['trades_count'],
                                                metrics_json=r['metrics_json'], start_date=r['start_date'],
                                                end_date=r['end_date'], timestamp=time.time()
                                            )
                                            db_objects.append(br)

                                        if db_objects:
                                            db.bulk_save_objects(db_objects)
                                            db.commit()

                                        current_count += len(chunk_results)
                                        if current_count % (chunk_size * 10) == 0:
                                            job.progress = min(100.0, (current_count / total_combos) * 100)
                                            job.current_pair = symbol
                                            db.commit()

                                    # Submit Next Chunk
                                    try:
                                        next_chunk = next(chunk_gen)

                                        # Check Pause/Cancel
                                        db.refresh(job)
                                        if job.status == 'cancelled':
                                            executor.shutdown(wait=False)
                                            return
                                        while job.status == 'paused':
                                            await asyncio.sleep(5)
                                            db.refresh(job)

                                        new_future = executor.submit(worker_task, df, next_chunk, indicators_defs)
                                        active_futures.add(new_future)
                                    except StopIteration:
                                        pass # No more chunks

                                except Exception as e:
                                    traceback.print_exc()

                            futures = list(active_futures)

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
