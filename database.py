import shelve
from contextlib import contextmanager

DB_PATH = "retraining_jobs.db"

@contextmanager
def get_db():
    """
    Context manager to get a database connection.
    """
    db = shelve.open(DB_PATH)
    try:
        yield db
    finally:
        db.close()

def set_job_status(task_id: str, status: str):
    """
    Set the status of a retraining job.
    """
    with get_db() as db:
        db[task_id] = status

def get_job_status(task_id: str) -> str:
    """
    Get the status of a retraining job.
    """
    with get_db() as db:
        return db.get(task_id)