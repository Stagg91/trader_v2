import unittest
import pandas as pd
import numpy as np
import traceback
from src.backtester import Backtester
from src.strategies.schemas import StrategyRecipe, IndicatorConfig

class TestBacktesterV2(unittest.TestCase):
    def setUp(self):
        # Create a simple trend scenario
        # Price goes up 10% each step
        prices = [100.0, 110.0, 121.0, 133.1, 146.41]
        dates = pd.date_range(start='2024-01-01', periods=5, freq='h')
        self.df = pd.DataFrame({
            'startTime': dates,
            'close': prices,
            'open': prices,
            'high': prices,
            'low': prices,
            'volume': [1000]*5
        })
        # Disable fees for pure logic test
        self.bt = Backtester(self.df, initial_balance=100, fee_rate=0.0, slippage=0.0)

    def test_vectorized_backtest_buy_and_hold(self):
        # Strategy: Buy always (Entry logic True)
        recipe = StrategyRecipe(
            name="BuyHold",
            description="Always Buy",
            indicators=[],
            entry_logic="close > 0", # Always True
            exit_logic="close < 0", # Always False
            stop_loss=0.0,
            take_profit=0.0,
            trailing_stop=0.0
        )

        metrics = self.bt.run_vectorized_backtest(recipe)

        # Logic:
        # idx 0: Buy Signal -> Position 1 (Entry @ 100)
        # idx 1: Price 110. Equity = 110.
        # idx 2: Price 121. Equity = 121.
        # ...
        # Final should be approx equal to price growth

        self.assertGreater(metrics['roi_percent'], 40) # 100 to 146 is 46%
        self.assertEqual(metrics['max_drawdown'], 0.0) # No drawdown

    def test_sl_execution(self):
        # Price drops: 100 -> 90 -> 80
        prices = [100.0, 90.0, 80.0]
        dates = pd.date_range(start='2024-01-01', periods=3, freq='h')
        df = pd.DataFrame({
            'startTime': dates,
            'close': prices,
            'open': prices,
            'high': prices,
            'low': prices, # Low equals Close here
            'volume': [1000]*3
        })
        bt = Backtester(df, initial_balance=100, fee_rate=0.0, slippage=0.0)

        # Strategy: Buy always, SL 5%
        recipe = StrategyRecipe(
            name="SLTest",
            description="SL 5%",
            indicators=[],
            entry_logic="close > 0",
            exit_logic="close < 0",
            stop_loss=5.0, # 5% SL
            take_profit=0.0,
            trailing_stop=0.0
        )

        metrics = bt.run_vectorized_backtest(recipe)

        # Entry @ 100 (idx 0)
        # Next candle (idx 1): Low is 90. SL trigger is 95 (100 * 0.95).
        # Since 90 <= 95, SL triggered.
        # Exit Price should be 95 (Stop Price) or 90 (Market)?
        # Backtester logic: exec_price = sl_price if triggered.
        # So Exit @ 95. PnL = -5%.
        # Remaining Equity = 95.

        # Check trades
        trades = metrics['trades']
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['reason'], "SL")
        self.assertAlmostEqual(trades[0]['exit'], 95.0)
        self.assertAlmostEqual(metrics['roi_percent'], -5.0)

    def test_tp_execution(self):
        # Price jumps: 100 -> 110 -> 120
        prices = [100.0, 110.0, 120.0]
        dates = pd.date_range(start='2024-01-01', periods=3, freq='h')
        df = pd.DataFrame({
            'startTime': dates,
            'close': prices,
            'open': prices,
            'high': prices,
            'low': prices,
            'volume': [1000]*3
        })
        bt = Backtester(df, initial_balance=100, fee_rate=0.0, slippage=0.0)

        # TP 5%
        recipe = StrategyRecipe(
            name="TPTest",
            description="TP 5%",
            indicators=[],
            entry_logic="close > 0",
            exit_logic="close < 0",
            stop_loss=0.0,
            take_profit=5.0, # 5% TP
            trailing_stop=0.0
        )

        metrics = bt.run_vectorized_backtest(recipe)

        # Entry @ 100.
        # Next High is 110. TP Trigger is 105.
        # Exit @ 105. PnL +5%.

        trades = metrics['trades']
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]['reason'], "TP")
        self.assertAlmostEqual(trades[0]['exit'], 105.0)
        self.assertAlmostEqual(metrics['roi_percent'], 5.0)

if __name__ == '__main__':
    unittest.main()
