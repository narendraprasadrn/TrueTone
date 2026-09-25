import json
import logging
from datetime import datetime, timezone
from sqlmodel import Session

from app.risk_engine.fusion import WindowRiskResult, RiskEngine
from app.audit.models import AuditEntry, engine
from app.alerts.dashboard_ws import manager as ws_manager
from app.alerts.webhook import dispatch_alert_webhook
from app.alerts.email_sim import dispatch_email_alert
from app.alerts.sms_sim import dispatch_sms_alert

logger = logging.getLogger(__name__)

class CallPipeline:
    def __init__(self, risk_engine: RiskEngine):
        self.risk_engine = risk_engine

    async def process_window(
        self,
        call_id: str,
        window_result: WindowRiskResult,
        tier: str,
        model_versions: dict
    ):
        """
        Main logic coordinating Phase 1-3 detector results with Phase 4 alerting/audit.
        """
        # 1. Update risk state
        call_state = self.risk_engine.call_states[call_id]
        
        outcome = "logged"
        
        # Determine if we need to alert
        if window_result.new_alert_transition:
            outcome = "alerted"
            
        # Write to audit log
        with Session(engine) as session:
            from sqlmodel import select
            existing = session.exec(select(AuditEntry).where(AuditEntry.call_id == call_id)).first()
            if existing:
                existing.window_score = window_result.r_window
                existing.call_level_score = window_result.r_final
                existing.classification = window_result.classification
                existing.contributing_signals = json.dumps(window_result.contributing_signals)
                if outcome == "alerted":
                    existing.outcome = "alerted"
                existing.timestamp = datetime.now(timezone.utc)
                session.add(existing)
            else:
                audit_entry = AuditEntry(
                    call_id=call_id,
                    window_score=window_result.r_window,
                    call_level_score=window_result.r_final,
                    classification=window_result.classification,
                    model_versions=json.dumps(model_versions),
                    contributing_signals=json.dumps(window_result.contributing_signals),
                    outcome=outcome,
                    tier=tier
                )
                session.add(audit_entry)
            session.commit()
            
        # 2. Broadcast standard window score
        msg = {
            "event": "score_update",
            "call_id": call_id,
            "window_ts": datetime.now(timezone.utc).isoformat(),
            "score": window_result.r_window,
            "call_score": window_result.r_final,
            "classification": window_result.classification,
            "contributing_signals": window_result.contributing_signals,
            "tier": tier
        }
        await ws_manager.broadcast(msg)
        
        # 3. Handle new alert transition
        if window_result.new_alert_transition:
            # Broadcast separate event
            alert_msg = msg.copy()
            alert_msg["event"] = "alert_triggered"
            await ws_manager.broadcast(alert_msg)
            
            # Webhook
            dispatch_alert_webhook(alert_msg)
            
            # Email & SMS
            dispatch_email_alert(call_id, window_result.r_final, tier, window_result.contributing_signals)
            dispatch_sms_alert(call_id, window_result.r_final, tier)

