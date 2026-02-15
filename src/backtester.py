import pandas as pd
import numpy as np
import traceback
from src.strategy_parser import StrategyParser
from src.strategies.schemas import StrategyRecipe
from src.logger import LabLogger
import asyncio

class Backtester:
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0, fee_rate: float = 0.001, slippage: float = 0.0005):
        self.data = data.copy()
        # Ensure contiguous integer index for loc/iloc logic
        self.data.reset_index(drop=True, inplace=True)
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate # 0.1% default
        self.slippage = slippage # 0.05% default
        self.parser = StrategyParser()

    def _log(self, msg, details=None):
        try:
             loop = asyncio.get_event_loop()
             if loop.is_running():
                 loop.create_task(LabLogger.log("BACKTEST", msg, details))
        except: pass

    def run_vectorized_backtest(self, strategy: StrategyRecipe) -> dict:
        """
        Runs a backtest on the strategy.
        Now uses an iterative loop for accurate SL/TP/TS execution,
        but relies on vectorized signal generation.
        """
        try:
            # 1. Parse & Execute Strategy -> Get Signals
            df = self.parser.parse_and_execute(self.data, strategy)

            # Check if any signals exist
            if 'signal' not in df.columns:
                 self._log("Critical Error: 'signal' column missing after parsing.")
                 return {"error": "Signal generation failed", "roi_percent": 0, "max_drawdown": 0, "equity_curve": []}

            # 2. Iterative Execution (Event Loop) for SL/TP/TS

            # Config
            sl_pct = strategy.stop_loss / 100.0 if strategy.stop_loss else 0
            tp_pct = strategy.take_profit / 100.0 if strategy.take_profit else 0
            # Handle trailing_stop safely (schema update might lag if passed dict)
            ts_pct = 0.0
            if hasattr(strategy, 'trailing_stop') and strategy.trailing_stop:
                ts_pct = strategy.trailing_stop / 100.0

            capital = self.initial_balance
            equity_curve = []
            detailed_trades = []

            position = 0 # 0: Flat, 1: Long
            entry_price = 0.0
            entry_time = ""
            high_watermark = 0.0

            # Pre-compute numpy arrays for speed (20x faster than iloc)
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            times = df['startTime'].values
            signals = df['signal'].values

            for i in range(len(df)):
                current_price = closes[i]
                timestamp = str(times[i])

                # Update Equity Curve (Unrealized)
                if position == 1:
                    # Simple MTM
                    unrealized_val = capital * (current_price / entry_price)
                    equity_curve.append(unrealized_val)

                    # Manage Trade Exits

                    # Update High Watermark for Trailing Stop
                    if highs[i] > high_watermark:
                        high_watermark = highs[i]

                    # Calculate Trigger Prices
                    sl_price = entry_price * (1 - sl_pct)
                    tp_price = entry_price * (1 + tp_pct)
                    ts_price = high_watermark * (1 - ts_pct)

                    exit_reason = None
                    exec_price = current_price

                    # Priority: SL -> TS -> TP -> Signal
                    # Assumption: We assume worst case (SL) if both SL and TP could be hit in same candle,
                    # unless we have intra-candle data. Standard conservative backtesting.

                    if sl_pct > 0 and lows[i] <= sl_price:
                        exit_reason = "SL"
                        exec_price = sl_price
                    elif ts_pct > 0 and lows[i] <= ts_price:
                        exit_reason = "TS"
                        exec_price = ts_price
                    elif tp_pct > 0 and highs[i] >= tp_price:
                        exit_reason = "TP"
                        exec_price = tp_price
                    elif signals[i] == -1:
                        exit_reason = "Signal"
                        exec_price = current_price

                    if exit_reason:
                        # Calculate Net PnL
                        gross_pnl_pct = (exec_price - entry_price) / entry_price

                        # Fees: Entry (0.1%) + Exit (0.1%) + Slippage (0.05%)
                        # Simplified approximation subtraction
                        total_costs = self.fee_rate + self.fee_rate + self.slippage
                        net_pnl_pct = gross_pnl_pct - total_costs

                        pnl_abs = capital * net_pnl_pct
                        capital += pnl_abs

                        detailed_trades.append({
                            "timestamp": entry_time,
                            "entry": entry_price,
                            "exit": exec_price,
                            "reason": exit_reason,
                            "pnl": net_pnl_pct * 100,
                            "pnl_abs": pnl_abs
                        })

                        # Reset Position
                        position = 0
                        entry_price = 0.0
                        high_watermark = 0.0

                else:
                    # Flat - Check Entry
                    equity_curve.append(capital)

                    if signals[i] == 1:
                        position = 1
                        entry_price = current_price
                        entry_time = timestamp
                        high_watermark = current_price # Init TS

            # --- Metrics Calculation ---
            final_equity = equity_curve[-1]
            total_return = (final_equity - self.initial_balance) / self.initial_balance * 100

            # Max Drawdown
            eq_series = pd.Series(equity_curve)
            cum_max = eq_series.cummax()
            drawdown = (eq_series - cum_max) / cum_max
            max_drawdown = drawdown.min() * 100

            # Sharpe Ratio
            if len(equity_curve) > 1:
                returns = eq_series.pct_change().dropna()
                if returns.std() > 0:
                    sharpe = (returns.mean() / returns.std()) * np.sqrt(365*24) # Approx hourly
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0

            # Win Rate
            wins = sum(1 for t in detailed_trades if t['pnl'] > 0)
            total_trades = len(detailed_trades)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            # Fitness
            dd_abs = abs(max_drawdown)
            if dd_abs < 0.001: dd_abs = 0.001
            fitness = total_return / dd_abs

            result_summary = {
                "roi_percent": total_return,
                "max_drawdown": max_drawdown,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "total_trades": total_trades,
                "fitness": fitness,
                "equity_curve": equity_curve,
                "trades": detailed_trades
            }

            self._log(f"Backtest Complete. Trades: {total_trades}, ROI: {total_return:.2f}%")
            return result_summary

        except Exception as e:
            err_msg = f"Backtest Critical Error: {e}"
            print(err_msg)
            traceback.print_exc()
            self._log(err_msg)
            return {
                "roi_percent": -100,
                "max_drawdown": -100,
                "fitness": -100,
                "error": str(e),
                "trades": [],
                "equity_curve": []
            }

    def run_strategy_instance(self, strategy_instance):
         print("Deprecated: use run_vectorized_backtest")
         return {}

    def grid_search(self, strategy_func, param_grid):
        return []

def combined_strategy(df, params):
    return []
