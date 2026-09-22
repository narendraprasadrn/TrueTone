from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field, create_engine, Session

def utc_now():
    return datetime.now(timezone.utc)

class AuditEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    call_id: str = Field(index=True)
    timestamp: datetime = Field(default_factory=utc_now, index=True)
    window_score: float
    call_level_score: float
    classification: str = Field(index=True)
    model_versions: str  # JSON string
    contributing_signals: str  # JSON string
    outcome: str = Field(index=True)  # "logged" | "alerted"
    tier: str

# SQLite engine
import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sqlite_file_name = os.path.join(base_dir, "database.db")
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
