import re

with open('app/pipeline.py', 'r') as f:
    content = f.read()

content = content.replace('call_state.new_alert_transition', 'window_result.new_alert_transition')
content = content.replace('call_state.classification', 'window_result.classification')

with open('app/pipeline.py', 'w') as f:
    f.write(content)
