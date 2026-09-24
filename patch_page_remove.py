with open("truetone/frontend/src/app/(dashboard)/test-bench/page.tsx", "r") as f:
    content = f.read()

import re
content = re.sub(r'console\.log\("HEADLINE SOURCE".*?\}\);', '', content, flags=re.DOTALL)

with open("truetone/frontend/src/app/(dashboard)/test-bench/page.tsx", "w") as f:
    f.write(content)
