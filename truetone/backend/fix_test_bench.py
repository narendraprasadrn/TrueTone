import re

with open('app/test_bench/routes.py', 'r') as f:
    content = f.read()

content = content.replace(
    'from app.preprocessing.audio import preprocess_audio',
    'from app.preprocessing.pipeline import preprocess_for_detection'
)
content = content.replace('window_duration_sec = 3.0', 'window_duration_sec = 2.0')
content = content.replace('re.call_states[call_id] = {"ema_score": 0.0, "is_alert": False, "is_warning": False}', '')
content = content.replace(
    'processed = preprocess_audio(window_chunk, sr, target_sr=16000)',
    'processed = preprocess_for_detection(window_chunk, sr, debug=False)'
)

fusion_old = """            win_res = re.score_window(a_score, p_score, s_score, ctx_flags)
            call_state = re.update_call_score(call_id, win_res)"""
fusion_new = """            win_res = re.score_window(call_id, a_score, p_score, s_score, ctx_flags)"""
content = content.replace(fusion_old, fusion_new)

content = content.replace('call_state.current_score', 'win_res.r_final')
content = content.replace('call_state.classification', 'win_res.classification')

final_status_old = """        # Final status
        final_state = re.call_states.get(call_id, {"ema_score": 0.0, "is_alert": False})
        overall_score = final_state.get("ema_score", 0.0)
        
        soft_warning = re.config["thresholds"]["soft_warning"]
        alert = re.config["thresholds"]["alert"]
        
        if overall_score >= alert:
            overall_class = "alert"
        elif overall_score >= soft_warning:
            overall_class = "warning"
        else:
            overall_class = "normal" """

final_status_new = """        # Final status
        final_state = re.call_states.get(call_id)
        if final_state and final_state.history:
            overall_score = final_state.history[-1]
            overall_class = final_state.classification
        else:
            overall_score = 0.0
            overall_class = "LOW" """

content = content.replace(final_status_old, final_status_new)

with open('app/test_bench/routes.py', 'w') as f:
    f.write(content)
