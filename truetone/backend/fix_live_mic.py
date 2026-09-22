import re

with open('app/live_mic/routes.py', 'r') as f:
    content = f.read()

content = content.replace(
    'from app.preprocessing.audio import preprocess_audio',
    'from app.preprocessing.pipeline import preprocess_for_detection'
)
content = content.replace('WINDOW_SEC = 3.0', 'WINDOW_SEC = 2.0')
content = content.replace('re.call_states[call_id] = {"ema_score": 0.0, "is_alert": False, "is_warning": False}', '')
content = content.replace(
    'processed = preprocess_audio(window_chunk, SR, target_sr=16000)',
    'processed = preprocess_for_detection(window_chunk, SR, debug=True)'
)

fusion_old = """                win_res = re.score_window(a_score, p_score, s_score, ctx_flags)
                call_state = re.update_call_score(call_id, win_res)"""
fusion_new = """                win_res = re.score_window(call_id, a_score, p_score, s_score, ctx_flags)
                call_state = re.call_states[call_id]"""
content = content.replace(fusion_old, fusion_new)

content = content.replace('call_state.current_score', 'win_res.r_final')

with open('app/live_mic/routes.py', 'w') as f:
    f.write(content)
