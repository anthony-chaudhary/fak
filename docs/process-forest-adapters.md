---
title: "Process-forest lifecycle adapters"
description: "Schema fak-lifecycle-adapter/1 is an additive adapter contract for prepare, pause, checkpoint, restore, resume, and readiness."
---
# Process-forest lifecycle adapters

Schema `fak-lifecycle-adapter/1` is an additive adapter contract for prepare, pause, checkpoint, restore, resume, and readiness. Negotiation records the transaction, forest/member, generation, complete capability document, requested operation, and typed supported/reason result so orchestration can store it beside the forest member and lifecycle transaction.

Built-ins:

- native fak harness: prepare/pause/checkpoint/restore/resume/readiness and an application-checkpoint claim;
- Codex and Claude: prepare/pause/resume/readiness only — neither claims OS suspension is an application checkpoint;
- custom: an explicit capability document plus injected implementation;
- unknown: no capabilities and fail-closed unsupported results.

Protocol version skew and unsupported operations fail closed. Every invocation requires a deadline and runs through a bounded context. Implementations are injected behind the interface for independent testing; platform-specific subprocess adapters must use the existing windowless process helper when added, rather than embedding unbounded `exec.Command` calls here.
