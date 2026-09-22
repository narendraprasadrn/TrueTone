import logging

logger = logging.getLogger(__name__)

ACTUALLY_SEND_SMS = False

def dispatch_sms_alert(call_id: str, score: float, tier: str):
    """
    Builds an MSG91-style payload.
    Tier 1 -> standard SMS to admin.
    Tier 2 -> Flash SMS (Class 0).
    """
    
    # MSG91 payload structure
    payload = {
        "sender": "TRUTON",
        "route": "4",
        "country": "91",
        "sms": [
            {
                "message": f"TrueTone Alert: Call {call_id} flagged with risk score {score:.1f}. Immediate review required.",
                "to": ["9999999999"]
            }
        ]
    }
    
    if tier == "tier2":
        # Flash SMS flag (Class 0)
        payload["flash"] = 1
        
    if ACTUALLY_SEND_SMS:
        # TODO: Post to MSG91 API
        pass
    else:
        sms_type = "Flash SMS (Class 0)" if tier == "tier2" else "Standard SMS"
        logger.info(f"SMS SIMULATION [Tier: {tier} | {sms_type}]: Payload generated\n{payload}")
