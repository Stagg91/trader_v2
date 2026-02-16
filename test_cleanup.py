from src.database import init_db, SessionLocal, BacktestResult, BacktestJob
from src.web.app import clean_database
import asyncio
from unittest.mock import MagicMock

async def test_cleanup():
    print("Initializing DB...")
    init_db()
    db = SessionLocal()

    # Insert orphan data
    # Create result with strategy_id 9999 (non-existent)
    orphan_res = BacktestResult(strategy_id=9999, roi=10.0, timestamp=0)
    orphan_job = BacktestJob(strategy_id=9999, status='completed')
    db.add(orphan_res)
    db.add(orphan_job)
    db.commit()

    # Verify insertion
    count_res = db.query(BacktestResult).filter(BacktestResult.strategy_id == 9999).count()
    count_job = db.query(BacktestJob).filter(BacktestJob.strategy_id == 9999).count()
    print(f"Orphans inserted: Results={count_res}, Jobs={count_job}")

    if count_res == 0 or count_job == 0:
        print("FAIL: Insertion failed.")
        return

    # Run Cleanup
    print("Running Cleanup...")
    # Need to verify VACUUM runs.
    # The clean_database function commits and runs VACUUM.
    await clean_database(db=db)

    # Verify Deletion
    # Note: clean_database uses raw SQL execution, but we check via ORM.
    # We might need to close/reopen session if VACUUM affects it?
    # SQLAlchemy session should see changes after commit.

    db.expire_all() # Refresh data

    count_res_after = db.query(BacktestResult).filter(BacktestResult.strategy_id == 9999).count()
    count_job_after = db.query(BacktestJob).filter(BacktestJob.strategy_id == 9999).count()

    print(f"Orphans remaining: Results={count_res_after}, Jobs={count_job_after}")

    if count_res_after == 0 and count_job_after == 0:
        print("PASS: Cleanup successful.")
    else:
        print("FAIL: Cleanup failed.")

    db.close()

if __name__ == "__main__":
    asyncio.run(test_cleanup())
