import re
import os

files_to_fix = [
    "src/app/(dashboard)/audit/page.tsx",
    "src/app/(dashboard)/dashboard/page.tsx",
    "src/app/(dashboard)/live-mic/page.tsx",
    "src/app/(dashboard)/test-bench/page.tsx",
    "src/lib/ws-client.ts"
]

for filepath in files_to_fix:
    if not os.path.exists(filepath): continue
    with open(filepath, 'r') as f:
        content = f.read()

    # Generic replacements
    content = content.replace("classification === 'alert'", "classification === 'HIGH'")
    content = content.replace("classification === 'warning'", "classification === 'MEDIUM'")
    content = content.replace("classification === 'normal'", "classification === 'LOW'")
    
    content = content.replace("overall_classification === 'alert'", "overall_classification === 'HIGH'")
    content = content.replace("overall_classification === 'warning'", "overall_classification === 'MEDIUM'")
    content = content.replace("overall_classification === 'normal'", "overall_classification === 'LOW'")
    
    # Text displays
    content = content.replace("'Alert'", "'HIGH'")
    content = content.replace("'Warning'", "'MEDIUM'")
    content = content.replace("'Normal'", "'LOW'")
    
    # ws-client.ts type
    content = content.replace("'normal' | 'warning' | 'alert'", "'LOW' | 'MEDIUM' | 'HIGH'")

    with open(filepath, 'w') as f:
        f.write(content)

