with open("truetone/frontend/src/app/(dashboard)/test-bench/page.tsx", "r") as f:
    content = f.read()

log_str = """
  console.log("HEADLINE SOURCE", {
    consolidated,
    aasistHeadlineValue: consolidated?.aasist,
    prosodyHeadlineValue: consolidated?.prosody,
    fusedHeadlineValue: consolidated?.fused,
    windows: result?.windows
  });
"""

if 'console.log("HEADLINE SOURCE"' not in content:
    content = content.replace(
        "const consolidated = result ? aggregateWindowScores(result.windows) : null;",
        "const consolidated = result ? aggregateWindowScores(result.windows) : null;\n" + log_str
    )

with open("truetone/frontend/src/app/(dashboard)/test-bench/page.tsx", "w") as f:
    f.write(content)
