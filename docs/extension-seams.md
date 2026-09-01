---
title: "Extension seams: where new behavior belongs"
description: "fak has no single plugin mechanism; the attachment point is part of the security contract. Pick the least-privileged seam from the runnable seam catalog."
---

# Extension seams

For how these seams fit the supported CLI, semantic protocol, `pkg/harnesskit`, sidecar, and internal layers, see [Builder contract ladder](builder-contract-ladder.md).: where new behavior belongs

fak is extensible, but it does not have one undifferentiated "plugin" mechanism. The
attachment point is part of the security and performance contract. Pick the
least-privileged seam that works; do not make user- or agent-authored code part of the
kernel just because an in-process registry is convenient.

The runnable catalog is the source of truth for the current choices:

```bash
go run ./cmd/extseamsdemo -selfcheck
go run ./cmd/extseamsdemo -json
```

The first command is the no-key/no-model spine witness. The second emits
`fak-extension-seams/1`, including each seam's attachment mode, trust class, failure
mode, source package, and selection rule.

## The three trust classes

| Class | Meaning | Examples |
|---|---|---|
| `data` | Parsed declarations, not executable extension code. They may only change what their contract permits. | Restrictive policy manifests. |
| `untrusted` | Code or artifacts may be authored by a user or agent. Run them across a process/artifact boundary and adjudicate the result. | Custom linters via agent hooks; generated skills, patches, or policy candidates entering the improvement loop. |
| `trusted-compiled` | Go code reviewed and compiled into the fak binary. It has the process's authority; a registry is composition, **not sandboxing**. | Kernel ABI registrants, middleware, quality oracles, trajectory scorers, console panes, compute backends. |

"In process" therefore never means "safe to install arbitrary code." Go's ordinary
package registries provide typed composition and low overhead, but no capability or
memory isolation. Keep replaceable, local, user-supplied, and agent-generated checks
out of process. Promote one into the binary only when its volume/latency requires it and
it can accept the review, compatibility, and release burden of trusted code.

## Custom and agent-authored linters

Use an **agent hook** by default. The concrete wire contract and conformance fixture are documented in [Custom linter subprocess ABI](integrations/custom-linters.md). The linter is a separate command, receives the host's
versioned hook payload, and returns a structured decision. The host owns deadlines,
output bounds, capability policy, logging, and fail-open/fail-closed behavior. A linter
that protects a security invariant must fail closed; advisory style or telemetry checks
may fail open but must record the failure.

A mature, deterministic, high-volume check may become a compiled **quality oracle** or
adjudicating **middleware**. That is a source contribution to fak, not a third-party
runtime plugin installation. It needs a stable name, duplicate/unknown-name refusal,
tests, and a release-compatible contract.

Do not let a linter mutate the intercepted operation. It returns findings or a verdict;
the kernel performs or refuses the operation. This preserves the checkpoint where fak
can witness what happened.

## In-process improvement

"In-process improvement" has two different meanings and they must not be conflated:

1. A trusted compiled scorer/oracle can *measure* a trajectory or candidate in process.
2. An agent can *propose* a linter, skill, policy, prompt, or code change, but that output
   remains an untrusted artifact.

The proposal crosses the `improvement-proposal` seam. The keep decision belongs to the
witnessed improvement loop: isolated application, independent test/truth/metric readback,
and keep-or-revert. No witness means no keep. Loading generated Go into the live fak
process would let the candidate rewrite its own judge and is therefore outside this
contract.

## Selection guide

1. **Can a restrictive manifest express it?** Use `policy-bundle`.
2. **Is it authored or changed by a user/agent?** Use `agent-hook` or an artifact
   proposal. Keep it outside the process.
3. **Is it a discoverable capability with a large body?** Use `capability-resolver` so
   only the card is resident and the body faults on demand.
4. **Must it surround every model/tool call?** Use observer/adjudicating `middleware`,
   accepting the trusted-code and hot-path burden.
5. **Is it task quality, trajectory ranking, UI, compute, or a low-level mechanism?** Use
   the corresponding typed compiled registry.
6. **Would a new model-visible tool be required?** Apply the
   [footprint ladder](footprint-ladder.md) first; a core tool is the last resort because
   its schema is charged on every request.

The discovery and distribution layer proposed by issue #3807 can index these seams, but
it must not erase their trust differences. The restrictive policy-bundle work in #2443
is similarly a data-plane extension, not permission to execute bundled code.

## Contract for a new seam

A new extension point is incomplete until it declares:

- stable, namespaced identity and compatibility/version behavior;
- attachment mode and trust class;
- input/output schema, deadlines, and resource/output bounds;
- duplicate, unknown, timeout, crash, and malformed-output behavior;
- whether failure is fail-open or fail-closed and why;
- what capabilities the extension receives (least authority by default);
- an independent witness for effects and improvement claims;
- discoverability without adding an always-sent model-visible schema;
- an end-to-end selfcheck or captured live witness.

If those fields cannot be stated, the code is not yet an extension contract; it is an ad
hoc callback.

## Common descriptor and local proof

The discovery envelope is `fak-extension-descriptor/1` in `internal/market`; see [`docs/integrations/extension-descriptors.md`](integrations/extension-descriptors.md). Its adapters cover ABI engines, compute backends, TUI panes, quality checks, and trajectory scorers. Enumeration is inert. Artifact and witness digests become evidence only after local re-verification; executable registrants remain `trusted-compiled`.
