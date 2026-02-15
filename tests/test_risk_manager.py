import unittest
import time
import datetime
from src.database import SessionLocal, Settings, init_db, engine
from src.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        # Ensure a settings record exists
        db = SessionLocal()
        if not db.query(Settings).first():
            db.add(Settings())
            db.commit()
        db.close()

    def test_persistence(self):
        # 1. Create RM, record loss
        rm1 = RiskManager(daily_loss_limit_pct=5.0)
        rm1.record_trade_close(-50.0)

        # 2. Create RM2, should see loss
        rm2 = RiskManager(daily_loss_limit_pct=5.0)
        # Note: rm2 loads from DB in init

        # Check if loaded (rm2._sync_state calls DB)
        # However, rm2.daily_pnl is set in init via _sync_state
        # Wait, _sync_state logic:
        # last_reset = settings.last_pnl_reset
        # if curr_date > last_date: reset
        # else: daily_pnl = settings.daily_pnl

        # Since we just ran it, dates should be same.
        self.assertEqual(rm2.daily_pnl, -50.0)

    def test_daily_reset(self):
        # Manually set last_pnl_reset to yesterday in DB
        db = SessionLocal()
        settings = db.query(Settings).first()
        settings.daily_pnl = -100.0
        yesterday = time.time() - 86400 * 2
        settings.last_pnl_reset = yesterday
        db.commit()
        db.close()

        # RM should detect old date and reset
        rm = RiskManager()
        self.assertEqual(rm.daily_pnl, 0.0)

        # DB should be updated
        db = SessionLocal()
        settings = db.query(Settings).first()
        self.assertEqual(settings.daily_pnl, 0.0)
        # self.assertAlmostEqual(settings.last_pnl_reset, time.time(), delta=10)
        db.close()

if __name__ == '__main__':
    unittest.main()
