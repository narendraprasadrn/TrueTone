import sys
from pathlib import Path
import asyncio
import soundfile as sf
from unittest.mock import patch

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.risk_engine.fusion import RiskEngine, WindowRiskResult
from app.risk_engine.context_stubs import evaluate_context_stubs
from app.simulator.simulator import CallSimulator
from app.detection.aasist import AasistDetector
from app.detection.prosody import ProsodyDetector
from app.detection.speaker_verification import SpeakerVerifier

def run_isolated_tests(engine: RiskEngine):
    print("=== ISOLATED FUSION MATH TESTS ===")
    print(f"{'Test Case':<35} | {'Raw Score':<10} | {'Weighted':<10} | {'Call Class':<10}")
    print("-" * 75)
    
    # 1. All-low inputs (0.1, 0.1, 0.9 speaker-match) -> stays "normal"
    res1 = engine.score_window(0.1, 0.1, 0.9, {})
    state1 = engine.update_call_score("call_1", res1)
    print(f"{'1. All-low':<35} | {res1.raw_score:<10.4f} | {res1.weighted_score:<10.4f} | {state1.classification:<10}")

    # 2. High aasist + no speaker enrollment (Tier 2 case) 
    # Tier 2 redistributes speaker weight (spoof: 0.65, behavior: 0.35)
    res2 = engine.score_window(0.9, 0.2, None, {})
    state2 = engine.update_call_score("call_2", res2)
    print(f"{'2. Tier 2 (high spoof)':<35} | {res2.raw_score:<10.4f} | {res2.weighted_score:<10.4f} | {state2.classification:<10}")

    # 3. Medium score + unknown_caller flag -> push to warning
    # unknown_caller = 1.15 multiplier
    # Try raw score around 0.5 * 1.15 = 0.575 (soft_warning is 55)
    res3 = engine.score_window(0.5, 0.5, 0.5, {"unknown_caller": True})
    state3 = engine.update_call_score("call_3", res3)
    print(f"{'3. Medium + unknown_caller':<35} | {res3.raw_score:<10.4f} | {res3.weighted_score:<10.4f} | {state3.classification:<10}")

    # 4. High score + ivr_allowlisted flag -> forced near-zero
    res4 = engine.score_window(0.9, 0.9, 0.1, {"ivr_allowlisted": True})
    state4 = engine.update_call_score("call_4", res4)
    print(f"{'4. High score + ivr_allowlisted':<35} | {res4.raw_score:<10.4f} | {res4.weighted_score:<10.4f} | {state4.classification:<10}")

    # 5. Simulate a call's EMA over 5 windows
    print("\n--- EMA Alert Transition Test (20 -> 40 -> 85 -> 90 -> 88) ---")
    simulated_window_scores = [20.0, 40.0, 85.0, 90.0, 88.0]
    # We'll just create dummy WindowRiskResult objects
    for i, w_score in enumerate(simulated_window_scores):
        res = WindowRiskResult(raw_score=w_score/100, weighted_score=w_score, multipliers_applied={}, contributing_signals={})
        # Mocking the engine's current EMA logic to trace just this progression exactly
        # But wait, the RiskEngine uses EMA formula: current = alpha * new + (1-alpha) * old
        # If we feed it exactly these window scores, the EMA will gradually adapt, not jump exactly to 20, 40, 85.
        # To strictly test the "transition fires exactly once when EMA crosses 80", let's force the EMA internal state to these values just for the sake of the transition test, or feed it values that yield exactly this EMA.
        # Let's just override the EMA internally to simulate the sequence of EMA scores:
        if "call_5" not in engine.call_states:
            engine.call_states["call_5"] = {"ema_score": w_score, "is_alert": False}
        else:
            engine.call_states["call_5"]["ema_score"] = w_score
            
        # We also need to manually trigger the classification logic on the forced EMA,
        # so let's just use update_call_score but patch the EMA update:
        with patch.dict(engine.call_states, {"call_5": {"ema_score": w_score, "is_alert": engine.call_states["call_5"]["is_alert"]}}):
            state = engine.update_call_score("call_5", res) # This will overwrite ema again.
            
        # Actually, let's just create a mock method or feed extremely high values to make EMA cross 80.
        # Easier: just feed the values and observe when it crosses 80.
        pass
    
    # Clean way to test transition:
    engine.call_states["call_5_proper"] = {"ema_score": 10.0, "is_alert": False}
    test_scores = [20.0, 40.0, 95.0, 95.0, 95.0] # Will push EMA across 80
    for idx, w_score in enumerate(test_scores):
        # We need the EMA to reach >= 80 at some point.
        # Alpha is 0.3.
        # Start at 10.
        # Window 1: 20 => 0.3*20 + 0.7*10 = 6 + 7 = 13 (normal)
        # Window 2: 40 => 0.3*40 + 0.7*13 = 12 + 9.1 = 21.1 (normal)
        # Window 3: 200 => 0.3*200 + 0.7*21.1 = 60 + 14.77 = 74.77 (warning)
        # Window 4: 200 => 0.3*200 + 0.7*74.77 = 60 + 52.33 = 112.33 (alert, transition = True)
        # Window 5: 200 => 0.3*200 + 0.7*112.33 = 60 + 78.63 = 138.63 (alert, transition = False)
        pass

    # Let's just use a clean instance of RiskEngine and feed values to it
    engine.call_states.pop("call_5", None)
    
    seq = [50.0, 50.0, 150.0, 150.0, 50.0]
    print(f"{'Window Input':<15} | {'EMA Score':<10} | {'Class':<10} | {'New Alert?':<10}")
    for w in seq:
        r = WindowRiskResult(raw_score=w/100, weighted_score=w, multipliers_applied={}, contributing_signals={})
        s = engine.update_call_score("call_5_ema", r)
        print(f"{w:<15} | {s.current_score:<10.4f} | {s.classification:<10} | {s.new_alert_transition}")


async def run_integration_test(engine: RiskEngine):
    print("\n=== END-TO-END INTEGRATION TEST ===")
    print("Loading ML models...")
    aasist = AasistDetector(str(backend_dir / "third_party" / "aasist" / "models" / "weights" / "AASIST-L.pth"))
    prosody = ProsodyDetector(str(backend_dir / "app" / "detection" / "models" / "prosody_lgbm.txt"))
    sv = SpeakerVerifier(savedir=str(backend_dir / "pretrained_models" / "ecapa-tdnn"))
    
    print("Enrolling speaker...")
    gen_path = str(backend_dir.parent / "sample_audio" / "genuine_sample.wav")
    gen_audio, sr = sf.read(gen_path)
    sv.enroll("user_123", gen_audio[:int(sr * 5.0)])
    
    sim = CallSimulator(window_duration_sec=4.0, step_duration_sec=1.0)
    
    print(f"\nProcessing cloned_sample.wav as a first-time caller with high value keywords...")
    cln_path = str(backend_dir.parent / "sample_audio" / "cloned_sample.wav")
    
    print(f"{'Window(s)':<10} | {'AASIST':<7} | {'Prosody':<7} | {'Speaker':<7} | {'Context':<25} | {'Win Score':<9} | {'EMA':<9} | {'Class':<8} | {'NewAlert'}")
    print("-" * 110)
    
    with patch("asyncio.sleep", return_value=None):
        async for frame in sim.stream_audio_file(
            filepath=cln_path,
            call_id="call_integration_1",
            direction="inbound",
            label="cloned",
            first_time_caller=True,
            transcript_stub="please transfer the funds immediately"
        ):
            audio = frame["audio_data"]
            win_str = f"{frame['window_start_sec']:.1f}-{frame['window_end_sec']:.1f}"
            
            a_score = aasist.score(audio, sr)
            p_score = prosody.score(audio, sr)
            s_score = sv.score(audio, "user_123")
            
            # Evaluate context stubs
            context_flags = evaluate_context_stubs(
                call_id=frame["call_id"],
                first_time_caller=frame["first_time_caller"],
                transcript_stub=frame["transcript_stub"]
            )
            
            win_res = engine.score_window(a_score, p_score, s_score, context_flags)
            call_res = engine.update_call_score(frame["call_id"], win_res)
            
            ctx_str = ",".join(k for k,v in win_res.multipliers_applied.items())
            
            print(f"{win_str:<10} | {a_score:<7.4f} | {p_score:<7.4f} | {s_score:<7.4f} | {ctx_str:<25} | {win_res.weighted_score:<9.2f} | {call_res.current_score:<9.2f} | {call_res.classification:<8} | {call_res.new_alert_transition}")


if __name__ == "__main__":
    config_path = str(backend_dir / "app" / "risk_engine" / "config.yaml")
    engine = RiskEngine(config_path)
    
    run_isolated_tests(engine)
    
    # Run integration test
    asyncio.run(run_integration_test(engine))
