from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

Base = declarative_base()

class Settings(Base):
    __tablename__ = 'settings'
    id = Column(Integer, primary_key=True)
    api_key = Column(String)
    api_secret = Column(String)
    testnet = Column(Boolean, default=True)
    gemini_api_key = Column(String, nullable=True)
    paper_trading = Column(Boolean, default=True)
    paper_balance = Column(Float, default=10000.0)
    is_active = Column(Boolean, default=False)
    auto_evolve = Column(Boolean, default=False) # Ensure default is False
    last_evolution_time = Column(Float, default=0.0)
    evolution_lookback_value = Column(Integer, default=3)
    evolution_lookback_unit = Column(String, default="Months")
    evolution_interval = Column(Integer, default=30) # Minutes
    cpu_usage_limit = Column(Integer, default=80) # Percentage max CPU
    grid_search_days = Column(Integer, default=30) # Days of data for Grid Search

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    hashed_password = Column(String)

class Notification(Base):
    __tablename__ = 'notifications'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    message = Column(String)
    timestamp = Column(Float)
    read = Column(Boolean, default=False)

class Strategy(Base):
    __tablename__ = 'strategies'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String, nullable=True)
    code = Column(String, nullable=True)  # Python source code (Legacy)
    content_json = Column(JSON, nullable=True) # Strategy Recipe (Selected Indicators)
    class_name = Column(String, nullable=True) # Class name to instantiate (Legacy)
    type = Column(String, default="manual") # manual, ai_gen, evolved, composite
    generation = Column(Integer, default=0)
    parent_id = Column(Integer, nullable=True) # ID of parent strategy
    is_active = Column(Boolean, default=False)
    created_at = Column(Float)

class IndicatorDef(Base):
    __tablename__ = 'indicator_defs'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    category = Column(String) # e.g. 'Trend', 'Momentum'
    default_params = Column(JSON) # e.g. {"length": 14}
    optimization_config = Column(JSON) # e.g. {"length": {"start": 5, "stop": 50, "step": 5}}

class BacktestJob(Base):
    __tablename__ = 'backtest_jobs'
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer)
    status = Column(String, default="pending") # pending, running, paused, completed, failed
    progress = Column(Float, default=0.0)
    current_pair = Column(String)
    current_param_set = Column(JSON) # Last checked params
    total_combinations = Column(Integer, default=0)
    started_at = Column(Float)
    completed_at = Column(Float)

class BacktestResult(Base):
    __tablename__ = 'backtest_results'
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer)
    job_id = Column(Integer, nullable=True) # Link to BacktestJob
    symbol = Column(String)
    start_date = Column(String)
    end_date = Column(String)
    roi = Column(Float)
    sharpe = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    trades_count = Column(Integer)
    metrics_json = Column(JSON) # Full metrics dump
    timestamp = Column(Float)

class TradeLog(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    side = Column(String)
    qty = Column(Float)
    price = Column(Float)
    timestamp = Column(String)
    profit = Column(Float, nullable=True)

class SentimentLog(Base):
    __tablename__ = 'sentiment_logs'
    id = Column(Integer, primary_key=True)
    timestamp = Column(Float)
    sentiment = Column(String) # BULLISH, BEARISH, NEUTRAL
    source = Column(String) # "RSS", "AI", etc
    raw_text = Column(String, nullable=True) # Summary of headlines

# Database Setup
import os
import sys

def get_db_path():
    """
    Returns the path to the database file.
    Uses AppData/Home directory to ensure write access in frozen mode.
    """
    app_name = "StaggsHecticTrader"
    if sys.platform == "win32":
        app_data = os.getenv("APPDATA")
        path = os.path.join(app_data, app_name)
    else:
        path = os.path.join(os.path.expanduser("~"), "." + app_name.lower())

    os.makedirs(path, exist_ok=True)
    return os.path.join(path, "trading_bot.db")

db_url = f"sqlite:///{get_db_path()}"
# print(f"DB URL: {db_url}")

# Use NullPool to disable connection pooling for SQLite to prevent QueuePool limit errors
# Increase timeout to 30s to reduce 'database is locked' errors
engine = create_engine(db_url, connect_args={"check_same_thread": False, "timeout": 30}, poolclass=NullPool)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def seed_indicators(db):
    """Populates the IndicatorDef table with default indicators if empty."""
    if db.query(IndicatorDef).count() > 0:
        return

    # Default Indicators
    # params: default value
    # optimization_config: start, stop, step for backtesting

    indicators = [
        {"name": "sma", "category": "Trend", "params": {"length": 20}, "opt": {"length": {"start": 5, "stop": 200, "step": 5}}},
        {"name": "ema", "category": "Trend", "params": {"length": 20}, "opt": {"length": {"start": 5, "stop": 200, "step": 5}}},
        {"name": "rsi", "category": "Momentum", "params": {"length": 14}, "opt": {"length": {"start": 5, "stop": 30, "step": 1}}},
        {"name": "macd", "category": "Momentum", "params": {"fast": 12, "slow": 26, "signal": 9},
         "opt": {"fast": {"start": 5, "stop": 20, "step": 1}, "slow": {"start": 20, "stop": 50, "step": 2}, "signal": {"start": 5, "stop": 20, "step": 1}}},
        {"name": "bbands", "category": "Volatility", "params": {"length": 20, "std": 2.0},
         "opt": {"length": {"start": 10, "stop": 50, "step": 5}, "std": {"start": 1.0, "stop": 3.0, "step": 0.5}}},
        {"name": "atr", "category": "Volatility", "params": {"length": 14}, "opt": {"length": {"start": 5, "stop": 30, "step": 1}}},
        {"name": "adx", "category": "Trend", "params": {"length": 14}, "opt": {"length": {"start": 5, "stop": 50, "step": 5}}},
        {"name": "stoch", "category": "Momentum", "params": {"k": 14, "d": 3, "smooth_k": 3},
         "opt": {"k": {"start": 5, "stop": 30, "step": 1}, "d": {"start": 3, "stop": 10, "step": 1}}},
        {"name": "williams_r", "category": "Momentum", "params": {"length": 14}, "opt": {"length": {"start": 5, "stop": 30, "step": 1}}},
        {"name": "cci", "category": "Momentum", "params": {"length": 20}, "opt": {"length": {"start": 10, "stop": 50, "step": 5}}},
        {"name": "roc", "category": "Momentum", "params": {"length": 9}, "opt": {"length": {"start": 5, "stop": 30, "step": 1}}},
        {"name": "psar", "category": "Trend", "params": {"af_start": 0.02, "af_max": 0.2},
         "opt": {"af_start": {"start": 0.01, "stop": 0.05, "step": 0.01}, "af_max": {"start": 0.1, "stop": 0.3, "step": 0.05}}},
        {"name": "mfi", "category": "Volume", "params": {"length": 14}, "opt": {"length": {"start": 5, "stop": 30, "step": 1}}},
        {"name": "keltner", "category": "Volatility", "params": {"length": 20, "mult": 2.0},
         "opt": {"length": {"start": 10, "stop": 50, "step": 5}, "mult": {"start": 1.0, "stop": 3.0, "step": 0.5}}},
        {"name": "donchian", "category": "Volatility", "params": {"length": 20}, "opt": {"length": {"start": 10, "stop": 50, "step": 5}}},
        {"name": "ulcer", "category": "Risk", "params": {"length": 14}, "opt": {"length": {"start": 5, "stop": 30, "step": 1}}},
        {"name": "force", "category": "Volume", "params": {"length": 13}, "opt": {"length": {"start": 5, "stop": 30, "step": 1}}},
        {"name": "eom", "category": "Volume", "params": {"length": 14, "div": 100000000},
         "opt": {"length": {"start": 5, "stop": 30, "step": 1}}},
        {"name": "ao", "category": "Momentum", "params": {}, "opt": {}}, # No params usually fixed 5/34
        {"name": "vortex", "category": "Trend", "params": {"length": 14}, "opt": {"length": {"start": 5, "stop": 30, "step": 1}}},
        {"name": "trix", "category": "Momentum", "params": {"length": 15}, "opt": {"length": {"start": 5, "stop": 30, "step": 1}}},
        {"name": "cmf", "category": "Volume", "params": {"length": 20}, "opt": {"length": {"start": 10, "stop": 50, "step": 5}}},
    ]

    for i in indicators:
        ind = IndicatorDef(name=i['name'], category=i['category'], default_params=i['params'], optimization_config=i['opt'])
        db.add(ind)

    db.commit()
    print(f"Seeded {len(indicators)} indicators.")

def init_db():
    from sqlalchemy import inspect, text

    # Enable Write-Ahead Logging (WAL) for better concurrency
    try:
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL;"))
            conn.commit()
    except Exception as e:
        print(f"Warning: Could not enable WAL mode: {e}")

    inspector = inspect(engine)

    # Basic migration hack: Check if 'strategies' table has 'code' column.
    # If not, drop it to recreate.
    if inspector.has_table("strategies"):
        columns = [c['name'] for c in inspector.get_columns("strategies")]
        if "code" not in columns:
            print("Detected old schema for 'strategies'. Dropping table to migrate...")
            with engine.connect() as conn:
                conn.execute(text("DROP TABLE strategies"))
                conn.commit()

        # Check for content_json
        if "content_json" not in columns:
             print("Migrating strategies table: adding content_json...")
             with engine.connect() as conn:
                try:
                    conn.execute(text("ALTER TABLE strategies ADD COLUMN content_json JSON"))
                    conn.commit()
                except Exception as e:
                    print(f"Migration Error: {e}")

        # Check for parent_id (New Migration)
        if "parent_id" not in columns:
             print("Migrating strategies table: adding parent_id...")
             with engine.connect() as conn:
                try:
                    conn.execute(text("ALTER TABLE strategies ADD COLUMN parent_id INTEGER DEFAULT NULL"))
                    conn.commit()
                except Exception as e:
                    print(f"Migration Error: {e}")

        # Check for description (New Migration)
        if "description" not in columns:
             print("Migrating strategies table: adding description...")
             with engine.connect() as conn:
                try:
                    conn.execute(text("ALTER TABLE strategies ADD COLUMN description VARCHAR DEFAULT NULL"))
                    conn.commit()
                except Exception as e:
                     print(f"Migration Error: {e}")

        # Check for grid_search_days
        if "grid_search_days" not in columns:
             print("Migrating settings table: adding grid_search_days...")
             with engine.connect() as conn:
                try:
                    conn.execute(text("ALTER TABLE settings ADD COLUMN grid_search_days INTEGER DEFAULT 30"))
                    conn.commit()
                except Exception as e:
                     print(f"Migration Error: {e}")

    # Check settings schema
    if inspector.has_table("settings"):
        columns = [c['name'] for c in inspector.get_columns("settings")]
        if "auto_evolve" not in columns:
             print("Migrating settings table...")
             with engine.connect() as conn:
                try:
                    conn.execute(text("ALTER TABLE settings ADD COLUMN auto_evolve BOOLEAN DEFAULT 0"))
                    conn.execute(text("ALTER TABLE settings ADD COLUMN last_evolution_time FLOAT DEFAULT 0.0"))
                    conn.commit()
                except Exception as e:
                    print(f"Migration Error: {e}")

        # Check for evolution_lookback (New Migration)
        if "evolution_lookback_value" not in columns:
             print("Migrating settings table: adding lookback config...")
             with engine.connect() as conn:
                try:
                    conn.execute(text("ALTER TABLE settings ADD COLUMN evolution_lookback_value INTEGER DEFAULT 3"))
                    conn.execute(text("ALTER TABLE settings ADD COLUMN evolution_lookback_unit VARCHAR DEFAULT 'Months'"))
                    conn.commit()
                except Exception as e:
                     print(f"Migration Error: {e}")

        # Check for evolution_interval
        if "evolution_interval" not in columns:
             print("Migrating settings table: adding evolution_interval...")
             with engine.connect() as conn:
                try:
                    conn.execute(text("ALTER TABLE settings ADD COLUMN evolution_interval INTEGER DEFAULT 30"))
                    conn.commit()
                except Exception as e:
                     print(f"Migration Error: {e}")

        # Check for cpu_usage_limit
        if "cpu_usage_limit" not in columns:
             print("Migrating settings table: adding cpu_usage_limit...")
             with engine.connect() as conn:
                try:
                    conn.execute(text("ALTER TABLE settings ADD COLUMN cpu_usage_limit INTEGER DEFAULT 80"))
                    conn.commit()
                except Exception as e:
                     print(f"Migration Error: {e}")

    # Check BacktestResult for job_id
    if inspector.has_table("backtest_results"):
        columns = [c['name'] for c in inspector.get_columns("backtest_results")]
        if "job_id" not in columns:
             print("Migrating backtest_results table: adding job_id...")
             with engine.connect() as conn:
                try:
                    conn.execute(text("ALTER TABLE backtest_results ADD COLUMN job_id INTEGER DEFAULT NULL"))
                    conn.commit()
                except Exception as e:
                     print(f"Migration Error: {e}")

    Base.metadata.create_all(bind=engine)

    # Seed Indicators
    try:
        db = SessionLocal()
        seed_indicators(db)
        db.close()
    except Exception as e:
        print(f"Error seeding indicators: {e}")
