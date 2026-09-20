---
description: Run the offline integration suite and report failures.
agent: fortis
---

Run the full offline integration suite for Fortis and summarize the result:

```
python -m pytest desktop/tests/ -q $ARGUMENTS
```

If it fails, diagnose the failing test(s) against the current source and fix
them (or hand off to the right subagent: `test-harness`, `report-engine`,
`rag-core`, or `api-orchestrator`). The suite must run with no GPU, no model
file, and no network access.