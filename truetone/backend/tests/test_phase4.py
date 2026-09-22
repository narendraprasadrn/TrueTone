import pytest
import pytest_asyncio
import asyncio
from fastapi.testclient import TestClient
from sqlmodel import Session, select
import json

from app.main import app
from app.audit.models import AuditEntry, engine, create_db_and_tables
from app.risk_engine.fusion import WindowRiskResult, CallRiskState, RiskEngine
from app.pipeline import CallPipeline
from app.alerts.dashboard_ws import manager as ws_manager
from app.alerts import email_sim, sms_sim, webhook

# Setup test DB
@pytest.fixture(autouse=True)
def setup_db():
    create_db_and_tables()
    with Session(engine) as session:
        # clear for tests
        session.exec(AuditEntry.__table__.delete())
        session.commit()
    yield
    with Session(engine) as session:
        session.exec(AuditEntry.__table__.delete())
        session.commit()
        
@pytest.fixture
def client():
    return TestClient(app)
    
@pytest_asyncio.fixture
async def mock_ws(monkeypatch):
    broadcasts = []
    async def mock_broadcast(msg):
        broadcasts.append(msg)
    monkeypatch.setattr(ws_manager, "broadcast", mock_broadcast)
    return broadcasts
    
@pytest.fixture
def mock_webhook(monkeypatch):
    webhooks = []
    def mock_dispatch(payload):
        webhooks.append(payload)
    monkeypatch.setattr("app.pipeline.dispatch_alert_webhook", mock_dispatch)
    return webhooks
    
@pytest.fixture
def mock_email(monkeypatch):
    emails = []
    def mock_dispatch(call_id, score, tier, sigs):
        emails.append((call_id, score, tier, sigs))
    monkeypatch.setattr("app.pipeline.dispatch_email_alert", mock_dispatch)
    return emails
    
@pytest.fixture
def mock_sms(monkeypatch):
    sms = []
    def mock_dispatch(call_id, score, tier):
        sms.append((call_id, score, tier))
    monkeypatch.setattr("app.pipeline.dispatch_sms_alert", mock_dispatch)
    return sms

@pytest_asyncio.fixture
async def pipeline():
    # Provide a dummy RiskEngine config path
    import os
    config_path = os.path.join(os.path.dirname(__file__), "../app/risk_engine/config.yaml")
    re = RiskEngine(config_path)
    
    return CallPipeline(re)

@pytest.mark.asyncio
async def test_alert_flow(pipeline, mock_ws, mock_webhook, mock_email, mock_sms):
    # Simulate crossing into alert
    # Need to manipulate RiskEngine EMA internally to force transition
    pipeline.risk_engine.call_states["call_p4"] = {"ema_score": 10.0, "is_alert": False}
    
    # Generate a dummy result that pushes EMA very high
    win_res = WindowRiskResult(
        raw_score=1.0, weighted_score=300.0, 
        multipliers_applied={}, contributing_signals={"aasist": 1.0}
    )
    
    await pipeline.process_window("call_p4", win_res, "tier1", {"aasist": "v1"})
    
    # Check outcomes
    # 1. 2 WS events: score_update and alert_triggered
    assert len(mock_ws) == 2
    assert mock_ws[0]["event"] == "score_update"
    assert mock_ws[1]["event"] == "alert_triggered"
    
    # 2. 1 Webhook
    assert len(mock_webhook) == 1
    
    # 3. 1 Email
    assert len(mock_email) == 1
    assert mock_email[0][0] == "call_p4"
    
    # 4. 1 SMS
    assert len(mock_sms) == 1
    
    # Check audit log
    with Session(engine) as session:
        entries = session.exec(select(AuditEntry)).all()
        # Should have 1 row for the window
        assert len(entries) == 1
        assert entries[0].outcome == "alerted"
        
@pytest.mark.asyncio
async def test_acknowledge_404s(client):
    # Hit acknowledge route, which has been removed
    response = client.post("/calls/call_ack/acknowledge")
    assert response.status_code == 404

def test_audit_routes(client):
    # Test HTTP methods on /audit
    get_res = client.get("/audit")
    assert get_res.status_code == 200
    
    post_res = client.post("/audit")
    assert post_res.status_code == 405
    
    put_res = client.put("/audit")
    assert put_res.status_code == 405
    
    delete_res = client.delete("/audit/1")
    assert delete_res.status_code == 405

@pytest.mark.asyncio
async def test_audit_has_no_audio(pipeline):
    win_res = WindowRiskResult(
        raw_score=0.1, weighted_score=10.0, 
        multipliers_applied={}, contributing_signals={"aasist": 0.1}
    )
    await pipeline.process_window("call_privacy", win_res, "tier1", {"aasist": "v1"})
    
    with Session(engine) as session:
        entry = session.exec(select(AuditEntry)).first()
        entry_dict = entry.model_dump()
        
        # Check that no audio/filepath fields exist
        assert "audio" not in entry_dict
        assert "filepath" not in entry_dict
        assert "raw_audio" not in entry_dict
        assert entry.contributing_signals == '{"aasist": 0.1}'
