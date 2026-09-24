import os
import glob

replacements = {
    '"http://localhost:8000': 'process.env.NEXT_PUBLIC_API_URL + "',
    '`http://localhost:8000': '`${process.env.NEXT_PUBLIC_API_URL}',
    "'ws://localhost:8000": "process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:5000"
}

files_to_check = [
    "src/app/(dashboard)/admin/page.tsx",
    "src/app/(dashboard)/audit/page.tsx",
    "src/app/(dashboard)/dashboard/page.tsx",
    "src/app/(dashboard)/test-bench/page.tsx",
    "src/app/individual/page.tsx",
    "src/lib/ws-client.ts",
    "src/app/(dashboard)/live-mic/page.tsx"
]

for filepath in files_to_check:
    if not os.path.exists(filepath): continue
    with open(filepath, 'r') as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    with open(filepath, 'w') as f:
        f.write(content)

with open('.env.local', 'w') as f:
    f.write('NEXT_PUBLIC_API_URL=http://localhost:5000\nNEXT_PUBLIC_WS_URL=ws://localhost:5000\n')

