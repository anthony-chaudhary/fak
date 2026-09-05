---
parent_goal: goals/GOAL-audit-server-warm-on-bring-up-by-default.md
sub_step: 2_engine_model_warmup
witness: "code audit of internal/engine/ and internal/model/ warmup routines"
target_files:
  - internal/engine/
  - internal/model/
---
# Sub-Goal Objective
Audit the engine and model initialization and warm-up routines:
1. Does `internal/engine` or `internal/model` implement warmup functions (e.g. `Warmup()`, `Prewarm()`, dummy inference, kernel jit/compilation, graph capture)?
2. What does the warmup actually do (KV cache preallocation, dummy tokens, GEMM warmup, device synchronization)?
3. Is warmup invoked automatically during engine creation / model loading, or must it be explicitly called by the caller? What happens if not explicitly requested?
4. Find exact file names, function names, and line numbers.
