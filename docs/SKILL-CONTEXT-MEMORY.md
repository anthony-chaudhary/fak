---
title: "Skill-context memory — the procedural twin of agent memory"
description: "How fak treats a skill invocation as procedural memory: a named, versioned, digest-keyed SkillContextRecord whose assembled context is a cacheable view, served as a HIT on an identical re-invocation instead of replayed."
---

# Skill-context memory — context capsules a skill earns once and reuses

*The declarative side of agent memory is a recorded page you page back in. The
procedural side is a skill invocation whose reusable artifact is the context it
assembled — keyed not by a page step, but by the invocation that produced it.*

## TL;DR

A `.claude/skills/` skill is the procedural twin of a `.claude/memory/` note: a
named, versioned, load-on-demand context capsule. fak makes that twin a *cacheable*
object. Running a skill at a version against a digested set of inputs assembles some
context; the reusable artifact is that assembled context, and its identity is the
**invocation digest** — not a source page. So a second run of the same skill, same
version, same digested inputs resolves to the same view and is served as a **HIT**,
re-rendering nothing. A changed version, producer, or digest is a distinct slot — a
cold **FAULT** that builds once and seeds the next HIT.

This is a thin binding over the view cache the rest of `internal/contextq` already
ships, not a second cache. The snippet/summary/kv views are projections of an
immutable recorded page, keyed by `(step, view type, producer)`. A procedural view
has no recorded page, so it carries a sentinel step, a constant `ViewProcedure` type,
and folds the invocation identity into the producer component of the key. Identity is
the **exact digest bytes** in the cache key — never a hash that could alias — so the
fail-closed rule (never serve one invocation's procedural memory as another's) holds
by construction.

## Why a skill is *procedural* memory

The views fak already materializes — `ViewSnippet`, `ViewSummary`, the KV view — are
**declarative**: each is a projection of a specific immutable page in a recorded core
image, and its identity *is* that page's step. You page it back in because the page
exists.

A skill invocation has no recorded source page. It is a **procedure**: a skill at a
version, run against a digested set of inputs, whose output is the context it
assembled. There is no page step to key on. What makes two runs "the same" is that
they are the same procedure over the same inputs — captured by an **invocation
digest**. That is the whole reason `ViewProcedure` is keyed differently from every
other view in the package.

## The record

`SkillContextRecord` (`internal/contextq/skillmemory.go`) is the procedural-memory
descriptor for one skill invocation:

| Field | Role |
|---|---|
| `SkillName` | which skill assembled the context |
| `Version` | the skill's version — a bump re-keys, so an updated skill never serves its old context |
| `InvocationDigest` | the identity of an identical re-invocation (same skill+version+inputs → same digest) |
| `Producer` | who assembled the context (human-facing; kept distinct from the composite cache key) |
| `Scope` | the share scope the assembled context may be reused across |
| `CacheEntry` | the lowered cache-metadata entry for the assembled context |

## How identity becomes a cache key

The shared `ViewCache` keys on `(step, view type, producer)`. A procedural-memory
view has no source page and a constant view type, so the invocation identity must
live in the **producer** component. `viewCacheProducer` folds the producer, skill
name, version, and the full invocation digest into one composite key, separated by
unit-separator (`\x1f`) bytes so a value that happens to contain a delimiter
character cannot forge a cross-field collision:

```
<producer>␟skill=<name>␟version=<ver>␟inv=<invocation-digest>
```

Two records that share all four collide on the same slot — a HIT. Any difference — a
new skill version, a different producer, a changed digest — is a distinct slot, a
cold build. The match is on the exact digest bytes carried in the key, not a
secondary hash, so the package never serves one invocation's procedural memory as
another's.

## The hot path

`SkillContextRecord.Resolve(cache, build)` is the procedural-memory hot path. It
consults the shared `ViewCache` for a procedural view keyed by this record's
invocation digest:

- **HIT (warm)** — a prior invocation with an identical `(producer, skill, version,
  digest)` already cached its view. It is served as a HIT, the `build` closure is
  **never** called, and nothing is re-rendered. This is the economic point: an
  identical re-invocation redoes no work. `SkillProcedureResult.Built` reports
  `false`, the proof the cold-build path did not run.
- **FAULT (cold)** — no view exists for this invocation digest yet. The `build`
  closure renders the procedural-memory body once, the view is stored under the
  digest key, and the next identical invocation becomes a HIT. `Built` reports `true`.

A nil cache forces the cold path on every call (build runs, nothing is stored); a nil
`build` closure yields an empty body.

The verdict speaks the package's closed `MaterializationKind` vocabulary
(HIT / FAULT / RECOMPUTE / REFUSE / ABSTAIN) — the same vocabulary the declarative
views use — so a procedural view is just another participant in the one
materialization protocol, not a parallel one.

## What is proven, and where

The behavior is witnessed by tests in `internal/contextq/skillmemory_test.go`:

- `TestSkillProcedureMemoryHitOnReInvocation` — an identical re-invocation is a HIT
  and the build closure does not run.
- `TestSkillProcedureVersionBumpReKeys` — a version bump re-keys to a cold build, so
  an updated skill never serves its predecessor's assembled context.
- `TestSkillProcedureSuppliedCacheEntry` — a supplied `CacheEntry` lowers into valid
  memory-view cache metadata.

## Where it sits

This file binds an existing cache to invocation identity; it does **not** reimplement
the `ViewCache` (shipped at `internal/contextq/contextq.go`) or the
`MemoryViewRecord` / `MaterializationVerdict` materialization protocol. Skill-context
memory is one more view type over that one cache — the procedural twin of the
declarative memory views — so byte-level recall and procedural reuse share a single
materialization decision rather than diverging into two caches with two trust models.
