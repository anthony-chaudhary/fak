---
title: "Cache reuse: choose the layer and verify the effect"
description: "Builder route for fak cache reuse: preserve a provider prompt prefix or use the kernel-owned KV layer, then run the proof appropriate to that mode."
---

# Cache reuse: choose the layer and verify the effect

**Audience:** the builder choosing how a fak-backed agent reuses shared setup across turns.

fak keeps repeated setup reusable at the checkpoint between an agent and its model calls. The
applicable cache depends on who runs the model: API-backed sessions preserve and steer the
provider's prompt-cache prefix; in-kernel serving owns the model's KV state directly.

Choose one row, follow its current guide, and take one checkable next action:

```sh
fak cachevalue report
```

Run that command after an API-backed guarded session to read back provider-reported cache
writes, reads, and the measured value attributed to reuse. For an in-kernel deployment, use the
Level 4 proof route in the table instead; `cachevalue report` does not prove kernel-owned KV
reuse.

## Choose the cache layer

| Workload | Current mode and default | What fak reuses | Next proof |
|---|---|---|---|
| Agent calls Anthropic through `fak manage` | Provider prompt cache. Fleet launchers select managed-cache `on`; a bare guard defaults to `auto`. | The byte-identical stable prompt prefix. On the Anthropic wire, active managed cache can request the longer cache tier so an idle return can read rather than rebuild that prefix. | Read [Managed cache in practice](caching/level-2-managed-cache-in-practice.md), run a guarded session, then run `fak cachevalue report`. |
| Agent calls OpenAI, Gemini, or another provider through an API wire | Provider-native cache behavior where that wire exposes it; fak remains passive when it cannot apply the Anthropic cache-control mechanism. | A stable prefix can remain eligible for provider reuse, but the provider decides whether it hits. | Use the [provider matrix and wire economics](caching/level-3-cache-economics-and-the-wire.md) and inspect that provider's reported cache fields. Do not infer a hit from prefix stability alone. |
| `fak serve --engine inkernel` or another local in-kernel model path | Kernel-owned KV cache. | Addressable token/KV spans and prefix state owned by fak's serving path, not a provider billing rebate. | Follow [The kernel-owned KV cache](caching/level-4-kernel-kv-cache.md) to its deterministic reuse and accounting witnesses. |

These modes are alternatives at the model boundary, not cumulative cache levels. An API-backed
session does not gain kernel-owned KV reuse, and an in-kernel serve does not produce provider
`cache_read` billing telemetry.

## What reuse changes

A model call contains stable setup—system instructions, tool schemas, and unchanged history—plus
the new turn. Reuse avoids recomputing or rebilling the stable prefix:

1. fak keeps the reusable prefix byte-identical while shedding only eligible old history;
2. the selected cache layer recognizes that prefix;
3. only the new suffix and any cold span need fresh work.

For API-backed sessions, fak can preserve eligibility and report what the provider returned; it
cannot force a provider cache hit. TTL expiry, eviction, or a moved cache breakpoint can still
produce a cold turn. For in-kernel serving, fak owns the reuse path and its deterministic proof
surface; that is a different claim from observed provider savings.

## API-backed managed-cache choices

`fak manage --managed-cache` accepts three values on the Anthropic wire:

| Choice | Behavior | Select it when |
|---|---|---|
| `on` | Requests active managed-cache steering regardless of whether fak can observe per-token billing. | The launcher default or an operator explicitly wants the longer Anthropic cache tier. |
| `auto` | Activates only when fak can identify API-key billing; otherwise it stays passive. | The bare-guard default, or when active steering should depend on observable billing. |
| `off` | Leaves provider cache markers unsteered. | A bounded comparison or a provider path where the operator deliberately wants passive behavior. |

Fleet launchers and bare `fak manage` have different defaults, so name the launch surface when
reporting a result. OpenAI-wire and other non-Anthropic sessions do not acquire Anthropic's
`cache_control` behavior merely because a launcher selected `on`.

## Verify the applicable effect

For an API-backed guarded session:

```sh
fak manage --managed-cache on -- claude
# Use the session, exit it, then inspect the captured provider telemetry.
fak cachevalue report
```

Treat provider `cache_read`/`cache_write` values as observations. A stable prefix makes a hit
possible; only the returned telemetry witnesses that the provider reused it. If the report is
empty, first confirm that the guarded run emitted usage telemetry and that the chosen provider
reports cache fields.

For an in-kernel deployment, do not substitute that report for a KV witness. Use the
[Level 4 guide](caching/level-4-kernel-kv-cache.md), which links deterministic prefix-reuse,
span-eviction, and accounting proofs. The [cache proofs index](../proofs/README.md) is the
maintained test authority; [cache value rollup](../cache-value-rollup.md) is the current
measurement ledger for economic observations.

## Generation, lifecycle, and support

- **Supported now:** provider-prefix preservation and cache telemetry on supported API wires;
  `--managed-cache on|off|auto` for guarded Anthropic-wire operation; and kernel-owned KV reuse
  on documented in-kernel serving paths.
- **Generation:** this route describes the current `gen/now` split between provider-managed
  prompt caching and kernel-owned KV caching. Research proposals and future cross-provider cache
  work stay in the [cache frontier](../cache-frontier/README.md), not in the current default.
- **Lifecycle:** the Level 2/3/4 guides are the detailed maintained authorities for their modes.
  A successful independent read-back promotes this route; a superseding guide or changed runtime
  default demotes it until reconciled.
- **Claim boundary:** provider reuse and savings are observed from returned telemetry; deterministic
  kernel reuse is witnessed by the linked proofs. Keep those provenance labels separate.

**Next check:** for an API-backed session, run `fak cachevalue report` and confirm it contains the
provider-reported cache fields from your guarded run; for in-kernel serving, complete the Level 4
reuse witness instead.
