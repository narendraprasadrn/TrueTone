import yaml
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class WindowRiskResult:
    r_window: float                     # Raw score for this window (0-1)
    r_final: float                      # Temporally aggregated score (0-1)
    contributing_signals: Dict[str, Optional[float]]
    classification: str                 # "LOW", "MEDIUM", or "HIGH"
    is_alert: bool                      # True if this window triggered/sustained an alert
    new_alert_transition: bool          # True if this window just crossed into alert

@dataclass
class CallState:
    history: List[float] = field(default_factory=list)
    classification: str = "LOW"
    consecutive_high_count: int = 0
    last_alert_time: float = 0.0
    is_alert: bool = False

class RiskEngine:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # call_id -> CallState
        self.call_states: Dict[str, CallState] = {}
        
    def _calculate_r_window(self, aasist_score: float, prosody_score: Optional[float], speaker_score: Optional[float]) -> float:
        """
        These weights are prototype starting values, not empirically optimized.
        They must be tuned against labeled validation data before any real deployment claim.
        """
        tier_w = self.config["weights"]["tier1"] if speaker_score is not None else self.config["weights"]["tier2"]
        
        w_a = tier_w.get("aasist", 0.60 if speaker_score is not None else 0.75)
        w_p = tier_w.get("prosody", 0.25)
        w_s = tier_w.get("speaker", 0.15) if speaker_score is not None else 0.0
        
        s_a = aasist_score
        s_p = prosody_score
        s_s = max(0.0, min(1.0, 1.0 - speaker_score)) if speaker_score is not None else 0.0
        
        total_w = w_a
        score = w_a * s_a
        
        if s_p is not None:
            total_w += w_p
            score += w_p * s_p
            
        if speaker_score is not None:
            total_w += w_s
            score += w_s * s_s
            
        r_window = score / total_w if total_w > 0 else 0.0
        return max(0.0, min(1.0, r_window))

    def score_window(
        self,
        call_id: str,
        aasist_score: float,
        prosody_score: Optional[float],
        speaker_score: Optional[float] = None,
        context_flags: dict = None
    ) -> WindowRiskResult:
        
        if call_id not in self.call_states:
            self.call_states[call_id] = CallState()
            
        state = self.call_states[call_id]
        
        # 1. Base window score
        r_current = self._calculate_r_window(aasist_score, prosody_score, speaker_score)
        
        # 2. Temporal Aggregation
        state.history.append(r_current)
        
        t_cfg = self.config.get("temporal_aggregation", {})
        recent_count = t_cfg.get("recent_window_count", 4)
        history_count = t_cfg.get("history_window_count", 12)
        
        # R_recent: mean of the last ~recent_count windows
        recent_windows = state.history[-recent_count:]
        r_recent = sum(recent_windows) / len(recent_windows) if recent_windows else r_current
        
        # R_history: mean of the last ~history_count windows
        history_windows = state.history[-history_count:]
        r_history = sum(history_windows) / len(history_windows) if history_windows else r_current
        
        r_final = (t_cfg.get("current_weight", 0.50) * r_current +
                   t_cfg.get("recent_weight", 0.30) * r_recent +
                   t_cfg.get("history_weight", 0.20) * r_history)
                   
        r_final = max(0.0, min(1.0, r_final))
        
        # 3. Thresholds + Hysteresis
        """
        Thresholds, hysteresis margins, and window counts here are prototype defaults 
        for demo purposes and require tuning against labeled validation data 
        (false-positive/false-negative rates) — they are not derived from any calibration study.
        """
        th_cfg = self.config.get("thresholds", {})
        h_cfg = self.config.get("hysteresis", {})
        
        low_max = th_cfg.get("low_max", 0.39)
        medium_max = th_cfg.get("medium_max", 0.69)
        
        esc_margin = h_cfg.get("escalate_margin", 0.03)
        desc_margin = h_cfg.get("deescalate_margin", 0.03)
        
        # Determine new classification with hysteresis
        new_class = state.classification
        
        if state.classification == "LOW":
            if r_final > low_max + esc_margin:
                if r_final > medium_max + esc_margin:
                    new_class = "HIGH"
                else:
                    new_class = "MEDIUM"
        elif state.classification == "MEDIUM":
            if r_final < low_max - desc_margin:
                new_class = "LOW"
            elif r_final > medium_max + esc_margin:
                new_class = "HIGH"
        elif state.classification == "HIGH":
            if r_final < medium_max - desc_margin:
                if r_final < low_max - desc_margin:
                    new_class = "LOW"
                else:
                    new_class = "MEDIUM"
                    
        state.classification = new_class
        
        # 4. Alert Triggering Logic
        now = time.time()
        new_alert_transition = False
        
        if state.classification == "HIGH":
            state.consecutive_high_count += 1
            min_high = h_cfg.get("min_consecutive_high_for_alert", 3)
            
            if state.consecutive_high_count >= min_high:
                # Check cooldown
                cooldown = h_cfg.get("cooldown_seconds", 20)
                if not state.is_alert and (now - state.last_alert_time > cooldown):
                    new_alert_transition = True
                    state.is_alert = True
                
                if state.is_alert:
                    state.last_alert_time = now # refresh alert time
        else:
            state.consecutive_high_count = 0
            if state.is_alert:
                state.is_alert = False # deactivate alert status if no longer HIGH
        
        return WindowRiskResult(
            r_window=r_current,
            r_final=r_final,
            contributing_signals={
                "aasist": aasist_score,
                "prosody": prosody_score,
                "speaker": speaker_score
            },
            classification=state.classification,
            is_alert=state.is_alert,
            new_alert_transition=new_alert_transition
        )
