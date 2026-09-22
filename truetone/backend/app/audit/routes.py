from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlmodel import Session, select

from app.audit.models import AuditEntry, get_session

router = APIRouter()

@router.get("/audit", response_model=List[AuditEntry])
def list_audit_logs(
    call_id: Optional[str] = None,
    classification: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    session: Session = Depends(get_session)
):
    query = select(AuditEntry)
    
    if call_id:
        query = query.where(AuditEntry.call_id == call_id)
    if classification:
        query = query.where(AuditEntry.classification == classification)
    if start_time:
        query = query.where(AuditEntry.timestamp >= start_time)
    if end_time:
        query = query.where(AuditEntry.timestamp <= end_time)
        
    query = query.order_by(AuditEntry.timestamp.desc())
    results = session.exec(query).all()
    return results

@router.get("/audit/{entry_id}", response_model=AuditEntry)
def get_audit_entry(entry_id: int, session: Session = Depends(get_session)):
    entry = session.get(AuditEntry, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry
