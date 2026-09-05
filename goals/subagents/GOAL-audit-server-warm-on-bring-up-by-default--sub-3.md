---
parent_goal: goals/GOAL-audit-server-warm-on-bring-up-by-default.md
sub_step: 3_gateway_memory_warmup
witness: "code audit of internal/gateway/, internal/ctxmmu/, internal/vdso/"
target_files:
  - internal/gateway/
  - internal/ctxmmu/
  - internal/vdso/
---
# Sub-Goal Objective
Audit gateway, context MMU, and vDSO memory/cache warmup on bringup:
1. Does the gateway initialize or warm its thread pools, memory arenas, routing tables, or upstream engine connections on bringup?
2. Does `ctxmmu` or `vdso` prewarm page tables, cache slots, or pre-allocate buffers on server startup?
3. Does the server have a "ready" / health check probe, and does "ready" wait for warmup to finish before serving traffic?
4. Find exact file names, function names, and line numbers.
