---
title: "FrontierSWE environment adapter"
description: "How fak stands up a co-resident fak serve gateway inside a FrontierSWE task sandbox without claiming a benchmark result."
---

# FrontierSWE environment adapter

This is the C7 adapter for epic #1706: make the C6 `harbor_ext.fak_routed:FakRoutedAgent`
shim reachable inside a FrontierSWE task sandbox by starting `fak serve` co-resident with
the agent.

The adapter is plumbing, not a result claim. The emitted command enforces this route shape:

1. `fak serve` binds to loopback inside the task environment.
2. A `/healthz` gate passes before the agent's first turn.
3. One `/v1/chat/completions` smoke request goes through the same base URL the shim uses.
4. The FrontierSWE harness starts with the model name unchanged and only the base URL rerouted.

Generate the recipe:

```bash
fak frontierswe env-adapter --task git-to-zig
```

Machine-readable form:

```bash
fak frontierswe env-adapter --task git-to-zig --json > frontierswe-env-adapter.json
```

For `allow_internet=false` tasks, `--gateway-base-url` and `--upstream-base-url` must be
loopback or explicitly pinned:

```bash
fak frontierswe env-adapter \
  --task git-to-zig \
  --gateway-base-url http://fak-gateway.local:8080/v1 \
  --upstream-base-url http://model.local:11434/v1 \
  --pinned-hosts fak-gateway.local,model.local
```

Pure loopback mode emits `docker run --network none`. Pinned-host mode emits Docker bridge
networking with `--add-host <name>:host-gateway`; the FrontierSWE/Modal sandbox must still
enforce `pin_resolved_hosts` for the no-internet boundary.

If the current host lacks Docker/GHCR/Modal access, the command is honestly gated and prints
the exact `docker run ... /bin/sh -lc ...` command to run on a capable box. It does not
fabricate a live FrontierSWE turn or any time-to-solution number.

The emitted `job_yaml` block is the registration shape for FrontierSWE:

```yaml
agents:
  - name: fak-routed-claude-code
    import_path: harbor_ext.fak_routed:FakRoutedAgent
    model_name: ${FRONTIERSWE_MODEL}
    override_timeout_sec: 72000
    kwargs:
      wrapped: harbor_ext.claude_code:ClaudeCodeApiKeyNoSearch
      fak_base_url: http://127.0.0.1:8080/v1
      allow_internet: false
```

The next live steps remain separately gated:

- C9 drives a full task through the shim.
- C14 measures wall-clock and turns-to-correctness.
- C15 may add a `BENCHMARK-AUTHORITY.md` row only after the official grader and score-parity
  gates pass.
