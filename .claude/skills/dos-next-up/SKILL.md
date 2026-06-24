---
name: dos-next-up
description: "Snapshot the repo's phased-plan portfolio into a dispatch packet: audit candidates with `dos verify`, render who-does-what, and emit a `dos gate` verdict. Use when you need the current next-work view before dispatching agents."
---

# dos-next-up — the generic plan-and-ship snapshot

> **This is the baseline screenplay DOS ships, not a prerequisite.** It drives the
> kernel syscalls (`verify`, `gate`) against *any* repo whose layout lives in
> `dos.toml`. It names no host directory, no host lane, and no host commit
> convention — every host specific comes from `dos doctor --json` (paths/lanes,
> via WCR) or `dos.toml [stamp]` (the ship grammar, via SCV). Copy it into your
> own skills dir, point `dos` at your workspace, and it runs.

The shape is domain-free: **discover the layout → walk the plans → audit each
pick against the truth syscall → render a packet → gate the empty case.** The
*policy* (which lanes, which plan grammar, where output lands) is data the
screenplay reads, never literals it hardcodes.

## Inputs

- `--scope <name>` (repeatable, optional) — narrow candidates to one lane (a
  name from the active `[lanes]` taxonomy) or one plan id. Omitted = all plans.
- `--limit <N>` (optional, default 5) — how many top picks to render.

## Step 0 — Discover the workspace layout (one call)

Run the doctor verb and read the result. **This is the WCR on-ramp: every
path/lane below comes from here, never a literal.**

```bash
dos doctor --workspace . --json
```

Parse the JSON object. The fields you use:

- `paths.plans_glob` — the glob to walk for plan docs (whatever the workspace
  declared). **Use this value; never assume a fixed plans directory.**
- `paths.next_packets` — where the rendered packet is written (the configured
  output dir; `.dos/verdicts` under the generic default).
- `lanes.concurrent` / `lanes.exclusive` / `lanes.trees` — the active lane
  taxonomy, if you need to group picks by lane or honor a `--scope`.
- `stamp` — the active ship-subject grammar (informational; `dos verify` applies
  it for you — you never grep subjects yourself).

If `git` is `false`, warn the operator: `dos verify`'s git rung has no history to
read, so every pick will report `source="none"` unless a registry exists.

## Step 1 — Walk the plans glob → candidate picks

List the plan docs under `paths.plans_glob` (relative to `paths.root`). For each
plan doc, read its phase headings to extract `(plan_id, phase_id)` candidate
pairs — the next not-yet-shipped phases. Keep this generic: a "phase" is a
heading the doc marks as a unit of work; you do not need the job's exact
frontmatter. If a `--scope` was given, keep only plans/lanes that match it.

Collect a flat list of candidate `(plan, phase)` picks. Do not rank by any
host-specific signal — order by plan-doc order, then truncate later.

## Step 2 — Audit each candidate against the truth syscall

For **each** candidate pick, ask the kernel whether it already shipped — never
trust the plan doc's own stamp, and never grep commit subjects yourself:

```bash
dos verify --workspace . <PLAN> <PHASE> --json
```

Read the `ShipVerdict` JSON: `{shipped, source, sha?, plan, phase}`. `source`
tells you *how* the kernel knows: `registry` (a ship row), `grep` (a commit
subject under the active `[stamp]` grammar), or `none` (no positive evidence).
This is the SCV payoff — a foreign repo whose commits read `AUTH2: …` is
recognized iff its `dos.toml` declares the matching `[stamp]`.

Classify each pick into one of three dispositions — this is what the gate reads:

- **`shipped: false, source: "none"`** → **live**: a real next pick. Disposition
  `{phase, live: true}`.
- **`shipped: true, source: "registry"`** → **done, cleanly**: a ship row exists.
  Disposition `{phase, live: false, drop_reason: "shipped"}` — drop it.
- **`shipped: true, source: "grep"`** → a real **git ship**. Now check the plan
  doc: does its heading for this phase carry a SHIPPED stamp? If **yes**, drop it
  as a clean ship (as above). If **no**, this is a **stale stamp** — the work
  shipped in git but the plan doc lags — and you must encode it so the gate can
  catch the false-drain: `{phase, live: false, drop_reason: "shipped",
  "ship_via": "direct", "plan_doc_stamped": false}`.

The `ship_via: "direct"` + `plan_doc_stamped: false` pair is the *exact* shape
`dos gate` classifies as STALE-STAMP (a `grep` ship the plan doc doesn't reflect).
**Do not put verify's `source` value into `ship_via`** — `ship_via` is the gate's
own direct-ship marker, set to the literal `"direct"` only for an unstamped git
ship; a `registry` hit is a clean `shipped` drop, never STALE-STAMP.

Keep the first `--limit` live picks as the packet's dispatch list.

## Step 3 — Render the dispatch packet

Assemble a self-contained markdown packet **yourself** (DOS ships no packet
template in the kernel — see the friction log: the `[render]` packet-template
seam is a named open axis). Write it under `paths.next_packets`, named
`next-up-<UTC-date>-<N>.md`. The packet has, generically:

1. **Header** — the workspace root, the active lane taxonomy, the run timestamp.
2. **Portfolio snapshot** — one row per plan: plan id, how many phases, how many
   verified shipped (from Step 2), the next live phase.
3. **Dispatch list** — for each live pick, a self-contained prompt another agent
   could be launched with: the plan id, the phase id, the plan-doc path, and the
   one-line goal. Keep each prompt standalone (no shared context). **The dispatch
   list IS a proposed fan-out partition** — if these picks are about to be launched
   in parallel, price the partition first with
   [`dos-plan-price`](../dos-plan-price/SKILL.md) so a colliding set is caught
   before any agent launches, not when the colliding lease is refused mid-wave.
4. **Already shipped** — the picks Step 2 found `shipped: true`, with `source`/`sha`.

Alongside the packet, emit the gate sidecar so Step 4 can classify it — write
`<paths.next_packets>/.dispositions-<tag>.json` with the kernel's contract. One
entry per pick the packet considered, using the Step-2 classification:

```json
{
  "schema": "oc3-dispositions-v1",
  "tag": "<tag>",
  "dispositions": [
    {"phase": "AUTH2", "live": true},
    {"phase": "AUTH1", "live": false, "drop_reason": "shipped"},
    {"phase": "AUTH3", "live": false, "drop_reason": "shipped",
     "ship_via": "direct", "plan_doc_stamped": false}
  ]
}
```

In this example: `AUTH2` is live (a dispatch-list pick); `AUTH1` is a clean ship
(verify `source` was `registry`, or `grep` with the plan doc stamped) — a plain
`shipped` drop; `AUTH3` is a **stale stamp** (verify `source` was `grep` but the
plan doc is unstamped) — `ship_via: "direct"` + `plan_doc_stamped: false` is what
makes `dos gate` return STALE-STAMP for it. A clean ship omits both fields (it
must NOT carry `ship_via: "direct"`, or it would be misread as stale).

## Step 4 — Gate the packet (typed verdict)

Classify the packet through the kernel so the outcome is a typed verdict, not a
prose guess:

```bash
dos gate --workspace . <paths.next_packets>/.dispositions-<tag>.json
```

Read the exit code (the verdict IS the code):

- `0` **LIVE** — the packet has dispatchable work; report the packet path and the
  dispatch count. This is the success case.
- `3` **DRAIN** — no live picks: a genuine empty backlog. Report "nothing to
  dispatch — the portfolio is drained."
- `4` **STALE-STAMP** — phases shipped in git but the plan docs lag. Report the
  drift (the operator should reconcile the stamps); this is NOT an empty backlog.
- `5` **BLOCKED** — picks exist but are blocked (a sibling claim / quota). Surface.
- `6` **RACE** — a concurrent render lost a lock race; retry once.
- `2` — a contract error (the sidecar was malformed). Fix the sidecar; do not
  treat it as DRAIN.

## Step 5 — Return

Print the packet path and the gate verdict. **Return the packet path** as the
final line so a caller (e.g. `dos-dispatch`) can chain it:

```
Saved: <paths.next_packets>/next-up-<date>-<N>.md  (verdict=<VERDICT>)
```

## What this skill deliberately does NOT do (no silent gap)

- **No soft-claim leasing.** It does not register per-pick soft-claims (the heavy
  lease core stays host-side, `CLAUDE.md` heavy tier). `dos-dispatch` takes a
  *lane* lease via `dos arbitrate`; the per-pick soft-claim is out of scope.
- **No host evidence sources.** It does not read a host's curated ranking file or
  a postmortem stream — those are host gardening inputs (an evidence-source hook
  is a named open seam). It ranks by plan-doc order, audited by `dos verify`.
- **No host packet template.** It assembles a generic packet; the exact section
  grammar / commit subject a host wants is the `[render]` template seam (open).

**Log the gap, never silently skip it.** The first time the skill would have used
one of these (a soft-claim, a host evidence source), emit a one-line `log` naming
what it is not doing and why — so the capability gap is surfaced at runtime, not
just documented here.

## Worked example (live transcript)

> **The shape, run for real against a `dos` workspace.** Copy-pasteable verbs,
> actual captured output. The whole point is the last gloss on each block:
> **read the RUNG**, not the bare `shipped` boolean.

Step 0 — discover the layout (the WCR on-ramp):

```bash
$ dos doctor --workspace . --json
# dos_version 0.28.0; git true; stamp.style "grep" (generic, any/no dir prefix)
# paths.plans_glob "docs/**/*-plan.md"; paths.next_packets ".dos/verdicts"
# lanes.concurrent [benchmark, docs, examples, scripts, spikes, src, tests]
```
↳ every path/lane below is read from this object — never a literal.

Step 2 — audit a pick that DID ship. Read the `source` rung, not just `shipped`:

```bash
$ dos verify --workspace . docs/82_liveness-oracle-plan liveness --json
{"phase":"liveness","plan":"docs/82_liveness-oracle-plan","rung":"direct","sha":"80d4f30","shipped":true,"source":"grep-subject","summary":"80d4f30 liveness: exclude the BIRTH acquire from the ADVANCING event count"}
```
↳ RUNG `source="grep-subject"` (NOT bare `grep`): a commit SUBJECT carrying the
phase token flips this to SHIPPED even if little was built. Read the rung.

Step 2 — and a pick that has NOT shipped:

```bash
$ dos verify --workspace . docs/99_runtime-validation-and-the-actuation-boundary halt --json
{"phase":"halt","plan":"docs/99_runtime-validation-and-the-actuation-boundary","shipped":false,"source":"none"}
```
↳ RUNG `source="none"`: no positive git/registry evidence → a live pick.

Step 4 — gate the packet; the verdict IS the exit code:

```bash
$ dos gate --workspace . .dos/verdicts/.dispositions-<tag>.json ; echo "exit=$?"
exit=0     # LIVE — dispatchable work (3=DRAIN, 4=STALE-STAMP, 5=BLOCKED, 6=RACE, 2=contract-error)
```
↳ exit-code `0` carried the verdict — LIVE, not a prose guess.

## Anti-patterns

- ❌ Hardcoding a plans directory or an output directory — read both from
  `dos doctor --json` (the `paths` object).
- ❌ Greping commit subjects for a ship marker yourself — call `dos verify` (it
  applies the active `[stamp]` grammar for you).
- ❌ Treating a 0-pick packet as "drained" without `dos gate` — a STALE-STAMP is a
  false drain, and only the typed gate distinguishes it.
- ❌ Naming a specific lane as a literal — read the active `lanes` from doctor.
