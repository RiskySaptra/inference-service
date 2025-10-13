import shelve
from contextlib import contextmanager
from config import settings

@contextmanager
def get_db():
    """
    Context manager to get a database connection.
    """
    db = shelve.open(settings.DATABASE_URL)
    try:
        yield db
    finally:
        db.close()

def set_job_status(task_id: str, status: str, progress=None):
    """
    Set the status and progress of a retraining job.
    """
    with get_db() as db:
        db[task_id] = {"status": status, "progress": progress}

def get_job_status(task_id: str):
    """
    Get the status and progress of a retraining job.
    """
    with get_db() as db:
        return db.get(task_id)