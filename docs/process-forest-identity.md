# Durable process-forest identity

Schema `fak-process-forest/1` makes logical ownership independent of observed PID ancestry. A forest has a stable `forest_id` and root authority; every member has a stable `member_id`, optional logical parent, adapter kind, monotonic generation, state, and optional host/PID/process-start observation.

`internal/processforest.Registry` supports idempotent registration, update, reparent, authority-checked adoption, terminalization, and deterministic snapshots. Parent edges may have arbitrary depth and fan-out. A wrapper process exiting does not remove its logical descendants: the member is terminalized and descendants can be adopted/reparented under a newer generation.

Safety rules:

- parent and child belong to the same forest and root authority;
- parent cycles and self-parenting are rejected;
- stale generations are rejected;
- exact host + PID + process-start identity may have only one live owner;
- a reused PID with a different process-start identity is distinct;
- adoption requires the forest's root authority;
- terminal transitions are typed and idempotent.

Snapshots are the shared read model for lifecycle orchestration and the future `fak ps` forest renderer. This leaf defines the durable contract and in-memory reference registry; persistent cross-host envelopes, adapters, and CLI rendering are follow-on leaves under #6432.
