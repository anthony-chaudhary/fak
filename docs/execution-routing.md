# Execution routing: choose the execution envelope, not only the model

`fak route` answers one deliberately narrow question: which model plan should handle an
aspect of a request? Real agent execution has at least three independent choices:

1. **Harness route** — which declared harness profile can execute the work (wire,
   repoint mechanism, and account-rotation capability).
2. **Model/sub-model route** — which model or ensemble handles this request, tool call,
   query, state operation, step, or scout.
3. **Session route** — start fresh, resume for continuity, fork portable state, or compact
   and resume when retained context is full.

`fak execution-route` composes those choices into one inspectable JSON envelope. It does
not rename every choice “routing,” and it does not flatten them into one model string:
each decision keeps its own reason and vocabulary.

## Working spine

```powershell
fak execution-route `
  --harnesses openai-generic,codex `
  --rotatable `
  --aspect tool_call `
  --tool write_repository `
  --session session-7 `
  --continuity `
  --context-utilization .91
```

The ordered harness list is operator policy. In this example `openai-generic` cannot
rotate account homes, so the decision selects `codex`; the existing model router chooses
the tool-call model plan; and the session decision is `compact_resume` because continuity
is required above the default 80% threshold.

## Boundaries

- Harness profiles remain declared by `internal/harnessprofile` and the `dos.toml`
  `harnesses` table. Execution routing consumes those profiles; it does not add another
  harness registry.
- Model policy remains the `internal/modelroute` manifest. The composed route embeds its
  full decision, including rule and reduction details for sub-model/ensemble plans.
- Session routing is declarative. It chooses a lifecycle action but does not own session
  storage, compaction, or migration.
- Candidate ordering and requirements are explicit inputs. The oracle does not infer
  quality, price, or account health that the caller did not supply.

This is the minimal end-to-end spine. Follow-on work should add measured harness
capabilities/health, persistent session portability metadata, and gateway execution of the
composed envelope rather than hiding those concerns in heuristic branches.


