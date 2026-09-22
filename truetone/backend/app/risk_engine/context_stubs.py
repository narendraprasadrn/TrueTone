from typing import Dict, Any

def evaluate_context_stubs(
    call_id: str, 
    first_time_caller: bool, 
    transcript_stub: str,
    ivr_allowlist: list = ["IVR_123", "IVR_999"]
) -> Dict[str, bool]:
    """
    STUB — real system would run ASR + NLP on a live transcript, 
    and query a real database for caller history.
    """
    
    # 1. unknown_caller
    unknown_caller = first_time_caller
    
    # 2. high_value_keywords
    high_value_keywords = False
    if transcript_stub:
        keywords = ["transfer", "otp", "password", "urgent", "wire", "verify your account"]
        text = transcript_stub.lower()
        if any(kw in text for kw in keywords):
            high_value_keywords = True
            
    # 3. ivr_allowlisted
    ivr_allowlisted = call_id in ivr_allowlist
    
    return {
        "unknown_caller": unknown_caller,
        "high_value_keywords": high_value_keywords,
        "ivr_allowlisted": ivr_allowlisted
    }
