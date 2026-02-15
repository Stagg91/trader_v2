import time
import datetime
from src.database import SessionLocal, Settings

class RiskManager:
    def __init__(self, daily_loss_limit_pct=5.0, max_position_size_usdt=100.0, max_open_positions=3):
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_position_size_usdt = max_position_size_usdt
        self.max_open_positions = max_open_positions
        self.daily_pnl = 0.0
        self.open_positions_count = 0

        try:
            self._sync_state()
        except Exception as e:
            print(f"RiskManager Init Error: {e}")

    def _sync_state(self):
        db = SessionLocal()
        try:
            settings = db.query(Settings).first()
            if settings:
                now = time.time()
                last_reset = settings.last_pnl_reset or 0.0

                # Check date (UTC)
                last_date = datetime.datetime.fromtimestamp(last_reset, datetime.timezone.utc).date()
                curr_date = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).date()

                if curr_date > last_date:
                    # New Day -> Reset
                    self.daily_pnl = 0.0
                    settings.daily_pnl = 0.0
                    settings.last_pnl_reset = now
                    db.commit()
                else:
                    self.daily_pnl = settings.daily_pnl or 0.0
        except Exception as e:
            print(f"RiskManager Sync Error: {e}")
        finally:
            db.close()

    def check_trade_allowed(self, symbol: str, size_usdt: float, current_balance: float) -> tuple[bool, str]:
        """
        Checks if a trade is safe to execute.
        Returns (Allowed: bool, Reason: str)
        """
        # Note: open_positions_count is strictly runtime here.
        # A robust system would query the exchange for open positions.
        if self.open_positions_count >= self.max_open_positions:
            return False, f"Max open positions ({self.max_open_positions}) reached."

        if size_usdt > self.max_position_size_usdt:
            return False, f"Position size {size_usdt} exceeds limit {self.max_position_size_usdt}."

        # Daily Loss Check
        max_loss_amount = current_balance * (self.daily_loss_limit_pct / 100.0)

        # If daily_pnl is negative and exceeds limit
        if self.daily_pnl < -max_loss_amount:
            return False, f"Daily loss limit reached ({self.daily_pnl:.2f} < -{max_loss_amount:.2f})."

        return True, "Trade Allowed"

    def record_trade_close(self, pnl: float):
        self.daily_pnl += pnl
        self.open_positions_count = max(0, self.open_positions_count - 1)

        # Persist PnL
        db = SessionLocal()
        try:
            settings = db.query(Settings).first()
            if settings:
                settings.daily_pnl = self.daily_pnl
                db.commit()
        except:
            pass
        finally:
            db.close()

    def record_trade_open(self):
        self.open_positions_count += 1
