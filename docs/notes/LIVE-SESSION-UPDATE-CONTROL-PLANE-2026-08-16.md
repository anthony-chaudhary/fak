# Live-session update control plane — research and ticket map (2026-08-16)

## Verdict

Long-running fak sessions need an **update control plane**, not an in-process binary replacement trick.
The safe unit is an immutable, signed **runtime generation**: the fak binary, harness/adapter
manifest, policy and route manifests, skills/hooks, and resolved dependency graph that a session is
currently executing. A running session keeps that generation until an explicit compatibility verdict
chooses one of three cutovers:

1. **Adopt live** — tighten-only or proven backward-compatible state can cross a steering point.
2. **Drain and resume** — checkpoint at a safe point, start the candidate generation, restore, and
   preserve the logical session lineage.
3. **Stay pinned** — incompatible, unproved, or unavailable updates never mutate the live session;
   they remain staged for a later boundary.

This separates four concerns that are currently easy to conflate:

- **distribution** answers whether bytes are authentic and how they arrived;
- **resolution** answers which exact transitive inputs form a candidate generation;
- **compatibility** answers whether current state can move to that generation;
- **steering** answers when and by whose authority the cutover occurs.

## Value frame

- **Centrality:** Core. It protects managed context and integrated operations during sessions that
  outlive the code and configuration that launched them.
- **For:** operators running attended multi-hour and unattended overnight agent sessions.
- **Problem:** improvements, policy changes, harness changes, and dependency revisions arrive while
  sessions are live, but today each mechanism has a different identity and cutover boundary.
- **Today:** binary self-update, manifest reload, live steering, checkpoint/restore, and harness
  dependency work are separate ticket clusters; a successful updater can still say nothing about the
  generation a live session uses.
- **Better because:** every session can report current/candidate generations and receive a deterministic
  `ADOPT_LIVE | DRAIN_RESUME | STAY_PINNED` verdict with rollback evidence.
- **Witness:** an overnight session stages a newer signed generation, rejects an incompatible live
  adoption, checkpoints and resumes into it, then reports the same logical session ID with a new
  generation epoch and a provenance-complete manifest.
- **Next-best alternative:** restart every session whenever trunk or a dependency moves. That is simple
  but discards hot state, interrupts unattended work, and cannot distinguish urgent policy tightening
  from optional feature churn.

## Problem checklist

| Check | Requirement |
|---|---|
| P1 managed context | Preserve logical session lineage and checkpointed state while generation identity changes explicitly. Never silently mix old prompt/tool state with new harness semantics. |
| P2 net-true efficiency | Coalesce candidate updates and cut over only at declared safe points; measure avoided restarts, drain latency, failed restores, and duplicate work. Poll-and-restart loops are not a gain. |
| P3 bounded adaptation | Candidate bytes and manifests are immutable; compatibility classes are fail-closed; policy changes may adopt live only when monotonic/tighten-only. |
| P4 integrated operations | One status surface must show desired, staged, current, and rollback generations for attended and headless sessions, with authority and audit rows. |

## The runtime-generation envelope

A generation is content-addressed and immutable. A practical envelope needs at least:

```text
RuntimeGeneration {
  generation_digest
  fak_binary:        {version, digest, source, signature/channel}
  harness:           {adapter, version, manifest_digest, protocol/API range}
  policy_manifest:   {digest, schema, monotonicity evidence}
  route_manifest:    {digest, schema}
  skills_hooks:      [{id, version/digest, permissions}]
  dependencies:      [{identity, version, digest, source, evidence, compatibility}]
  state_schema:      {session, journal, checkpoint, tool-call ABI}
  provenance:        {resolver version, resolved_at, witnesses}
}
```

A Git submodule is only one possible dependency source. Updating it safely means resolving the desired
ref to an immutable commit, verifying source/signature policy, computing the complete candidate
generation, testing its declared compatibility, and staging it without moving the live session's
working tree. A floating branch, mutable local path, or bare version string is not a generation.

The harness manifest belongs inside this envelope. It must declare not just command and environment,
but adapter/protocol compatibility, dependency pins, required capabilities, state/checkpoint schema,
and migration hooks. Dependency resolution should emit a lock/provenance result; it should never edit
a live generation in place.

## Lifecycle and authority

```text
DISCOVER -> RESOLVE -> VERIFY -> STAGE -> ASSESS -> ADOPT | DRAIN/RESUME | PIN
                                                     |             |
                                                     +-- OBSERVE --+
                                                           |
                                                        ROLLBACK
```

- **Discover:** notice a channel, trunk, manifest, policy, harness, skill, or dependency change.
- **Resolve:** produce one immutable transitive generation; coalesce newer candidates before cutover.
- **Verify:** authenticate artifacts and run generation-level preflight/self-checks.
- **Stage:** install beside the current generation. Never overwrite an executable or dependency that a
  live process is using.
- **Assess:** compare current state and candidate contracts, not only semver.
- **Adopt live:** allowed for compatibility-preserving route changes and monotonic policy tightening at
  a journaled steering point.
- **Drain/resume:** quiesce dispatch, checkpoint the process forest/session state, launch the staged
  generation, restore in dependency order, and advance the generation epoch.
- **Pin:** keep running the current generation with an explicit reason and retry/expiry policy.
- **Observe/rollback:** require readiness and invariant windows before retiring the prior generation;
  rollback is another epoch transition, not an in-place downgrade.

Authority is separate from mechanism. An operator, signed organization policy, or bounded unattended
policy may request a desired generation. The session owns its safe-point handshake. Security floor
changes may force drain/termination when a tighten-only live swap is impossible; optional feature
updates may not.

## Compatibility verdict

The assessment is the meet of independently checked dimensions:

| Dimension | Live-adopt requirement | Otherwise |
|---|---|---|
| binary/tool ABI | Existing calls and response schemas remain valid | drain/resume or pin |
| harness protocol | Candidate accepts the active adapter/API range and permissions | drain/resume or pin |
| policy | Proven tighten-only for active capabilities | drain/terminate; never widen silently |
| route/model | Active request semantics and required capabilities remain satisfiable | steering-point drain or pin |
| session/journal schema | Current state is readable without lossy migration | witnessed migration + resume |
| checkpoint/process forest | All required members are restorable in dependency order | pin |
| dependency graph | Every node is immutable, verified, and compatibility-evidenced | pin |

Unknown evidence yields `STAY_PINNED`, not optimistic adoption. A migration hook runs only in the staged
candidate against a copy/checkpoint and must be replayable or reversible.

## Operator surface

Extend the semantic-control spine rather than creating another transport-specific updater:

```text
status: current=gA desired=gC staged=gC candidate_age=7m verdict=DRAIN_RESUME reason=session_schema
request: stage gC
request: adopt-at next_safe_point       # CAS against session + current generation epoch
request: pin gA --until 2026-08-17T08:00Z --reason active benchmark
request: rollback gA
```

Every request and transition carries session ID, generation epoch, expected current digest, authority,
reason, deadline, and idempotency key. Stale requests fail CAS. `humanctl`, API, phone, and headless
policy adapters should normalize onto the same semantic event and journal.

## Existing ticket map

### A. Supply, authenticity, and installed binaries

- **#6628** is the signed OCI/TUF channel and revocation design. It should emit authenticated candidate
  artifact identities, not decide live-session cutover.
- Closed **#5873**, **#6508**, and **#4022** establish the self-update/binary-census baseline: updater
  success must refer to the executable roles actually launched and be skew-aware. Preserve these as
  prerequisites/evidence, not as the live-update epic.

### B. Harness and dependency resolution

- **#6886** owns evidence-aware native harness dependency resolution. Its output should become the
  immutable dependency/lock/provenance section of `RuntimeGeneration` and include submodule-like Git
  sources without treating mutable refs as resolved artifacts.
- **#6771** owns capability-ceiling manifests. Its projection contract is one compatibility input,
  especially when a candidate harness narrows capabilities.

### C. Live steering and identity

- **#6784** owns the transport-independent semantic control spine and should carry generation-control
  intents with CAS and idempotency.
- **#6343** owns persisted generation epochs and steering points; it is the authoritative session
  identity seam for current/candidate/adopted generations.
- **#4229** is a downstream phone adapter. It should not invent update semantics.

### D. Configuration hot reload

- **#3955** owns tighten-only capability-floor watching.
- **#3958** owns an explicit Unix SIGHUP trigger for floor/route reload.
- **#4003** owns the API route-reload trigger.

These are valid **live-adopt executors** for selected manifest classes. They must journal the generation
component swap and cannot stand in for binary/harness/dependency cutover.

### E. Drain, checkpoint, resume, and rollback

- **#6438** owns a standardized process-forest checkpoint manifest and restore recipe.
- **#6439** owns dependency-ordered restoration and readiness proof.
- **#3784**, **#3785**, and **#3788** own crash-survivable registration and recovery into the resume
  pipeline. Planned update cutover should reuse this identity/recovery substrate, not create a second
  restart journal.

## Proposed issue hierarchy and sequencing

One missing coordination epic should bind, not absorb, the existing leaves:

1. **Runtime-generation contract and status spine** — define the envelope, compatibility verdict, state
   machine, and current/desired/staged/rollback status. This is the minimal working spine.
2. **Resolver integration** — feed #6628 and #6886 outputs into an immutable staged generation.
3. **Live-adopt integration** — classify #3955/#3958/#4003 swaps and journal them through #6343.
4. **Drain/resume integration** — connect #6438/#6439 and #3784/#3788 to a requested generation change.
5. **Semantic steering adapters** — carry generation intents through #6784, then #4229 and headless
   policy adapters.
6. **Overnight witness and rollback drill** — coalesce multiple updates, reject one incompatible live
   adoption, resume once, and rollback on a failed readiness window.

Dependencies are a DAG, not a serial rewrite: (1) precedes integrations; (2), (3), and checkpoint work
can proceed in parallel; (4) needs the generation contract plus checkpoint/restore; (5) needs the
semantic contract but can initially target stage/pin; (6) witnesses the integrated path.

## Minimal end-to-end witness

A self-contained test/demo should:

1. launch a logical session on generation `gA` with a versioned harness manifest and dependency lock;
2. publish `gB` containing a tighten-only policy change and prove live adoption at a steering point;
3. publish `gC` containing a harness/state-schema change and prove live adoption is refused;
4. stage `gC`, quiesce, checkpoint, launch beside `gA`, restore in dependency order, and advance the
   persisted generation epoch while preserving the logical session ID;
5. fail a readiness invariant and roll back to `gA`/the last good generation through another witnessed
   epoch transition;
6. print a replayable journal and generation manifest proving exactly which binary, harness, policy,
   skills, and dependencies executed each turn.

## Gold-plating boundary

The first spine does **not** require fleet-wide simultaneous rollout, arbitrary in-memory code hot
patching, every package ecosystem, automatic migration synthesis, or phone UI polish. It requires one
local generation format, one staged side-by-side install, deterministic assessment, one live manifest
swap, one checkpoint/resume cutover, and one rollback witness. Fleet rollout and richer dependency
providers follow only after that path works end to end.
