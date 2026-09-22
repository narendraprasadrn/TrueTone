import re

with open('app/test_bench/routes.py', 'r') as f:
    content = f.read()

final_status_pattern = re.compile(r'# Final status.*?overall_class = "normal"', re.DOTALL)
final_status_new = """# Final status
        final_state = re.call_states.get(call_id)
        if final_state and final_state.history:
            overall_score = final_state.history[-1]
            overall_class = final_state.classification
        else:
            overall_score = 0.0
            overall_class = "LOW" """
content = re.sub(final_status_pattern, final_status_new, content)

with open('app/test_bench/routes.py', 'w') as f:
    f.write(content)

