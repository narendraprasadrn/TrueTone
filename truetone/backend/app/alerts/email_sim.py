import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Config flag: if True, we would actually make an API call (e.g. via Composio)
ACTUALLY_SEND_EMAIL = False

def dispatch_email_alert(call_id: str, score: float, tier: str, contributing_signals: Dict[str, Any]):
    """
    Builds the payload exactly as if going through Composio -> Gmail API.
    Defaults to log-only for demo purposes.
    """
    subject = f"[URGENT] High Risk Call Detected: {call_id}"
    
    body = f"""
    A high-risk call has been detected.
    
    Call ID: {call_id}
    Risk Score: {score:.2f}/100
    Tier: {tier}
    
    Contributing Signals:
    """
    for sig, val in contributing_signals.items():
        if val is not None:
            body += f"- {sig}: {val:.4f}\n"
            
    payload = {
        "to": "security-team@truetone.local",
        "subject": subject,
        "body": body,
        "is_html": False
    }
    
    if ACTUALLY_SEND_EMAIL:
        # TODO: integrate with Composio / actual Gmail API
        pass
    else:
        logger.info(f"EMAIL SIMULATION [Tier: {tier}]: Payload generated\n{payload}")
