import re
import os

with open('app/pipeline.py', 'r') as f:
    pipe_content = f.read()

pipe_content = pipe_content.replace('from app.risk_engine.fusion import WindowRiskResult, CallRiskState, RiskEngine', 'from app.risk_engine.fusion import WindowRiskResult, RiskEngine')
pipe_content = pipe_content.replace('call_state = self.risk_engine.update_call_score(call_id, window_result)', 'call_state = self.risk_engine.call_states[call_id]')
pipe_content = pipe_content.replace('window_result.weighted_score', 'window_result.r_window')
pipe_content = pipe_content.replace('call_state.current_score', 'window_result.r_final')

with open('app/pipeline.py', 'w') as f:
    f.write(pipe_content)

with open('app/demo.py', 'r') as f:
    demo_content = f.read()

demo_content = demo_content.replace('from app.simulator.simulator import CallSimulator', 'from app.simulator.simulator import CallSimulator\nfrom app.preprocessing.pipeline import preprocess_for_detection')
demo_content = demo_content.replace('window_duration_sec=3.0', 'window_duration_sec=2.0')

old_process = """                    a_score = aasist.score(audio, sr)
                    p_score = prosody.score(audio, sr)
                    s_score = sv.score(audio, "user_demo") if tier == "tier1" else None"""
new_process = """                    processed = preprocess_for_detection(audio, sr, debug=False)
                    a_score = aasist.score(processed, 16000)
                    p_score = prosody.score(processed, 16000)
                    s_score = sv.score(processed, "user_demo") if tier == "tier1" else None"""
demo_content = demo_content.replace(old_process, new_process)

demo_content = demo_content.replace('win_res = pipeline.risk_engine.score_window(a_score, p_score, s_score, ctx_flags)', 'win_res = pipeline.risk_engine.score_window(frame["call_id"], a_score, p_score, s_score, ctx_flags)')

with open('app/demo.py', 'w') as f:
    f.write(demo_content)
