# RATIONALE.md — the evidence behind every rule in `/fleet-wave`

> ⛔ **No worker loads this file, and it must stay that way.** The split is the point: an
> imperative in the fuel pointer is re-billed as `cache_read` on every turn of every worker,
> so the *rule* ships in [`refusals.md`](refusals.md) or the fuel and the *archaeology*
> stays here. Adding something? It goes in exactly one of the three, never all three.
>
> ⛔ **Read this before you weaken, delete, or "simplify" a rule.** Each one is paid for.

## Provenance — two classes, and do not confuse them

| class | meaning |
|---|---|
| **MEASURED-HERE** | re-derived against this repo/box on the stated date, with the probe shown. Trust it, and re-measure before quoting it as current. |
| **PORTED** | inherited from the tensorbuild `fleet-wave` skill (`C:\work\tb\tensor-build-main\.claude\skills\fleet-wave\`), measured on *that* fleet, not this one. The failure **shape** is real and transferable; the **numbers are not fak's.** ⛔ Never quote a PORTED number as a fak measurement. |

---

## MEASURED-HERE — 2026-08-08, fak 0.43.0 (build `85755142ab5c`, go1.26.5 windows/amd64)

### The default `-PointerFile` does not exist (SKILL.md § 1, trap 1)

```
$ ls .claude/goal-prompts/resolve-tickets-witnessed.md
ls: cannot access '...': No such file or directory
```
Both launchers default to it — `launch_wave_detached.ps1:49`, `launch_goal_detached.ps1:63`
— and `launch_goal_detached.ps1:163` is `if (-not (Test-Path $PointerFile)) { throw … }`.
The wave launcher dispatches **every** slot through that gate, so a bare
`-Count 30 -Launch` throws 30 times and spawns nobody. The 14 prompts that *do* exist are in
`.claude/goal-prompts/`; `resolve-top-issue-ultracode.md` is the one this skill's fuel is
derived from. **This is the single highest-value fact in the skill** — it converts a wave
that reads as "launched" into one that measurably never started.

### The 4000-character `/goal` cap, and why preambles cannot be concatenated (§ 2, trap 2)

`launch_goal_detached.ps1:165-166`:
```powershell
$cond = "/goal $body"
if ($cond.Length -gt 4000) { throw "goal condition is $($cond.Length) chars (>4000 cap) …" }
```
Measured bodies: `resolve-top-issue-ultracode.md` **2557**, `resolve-top-issue-witnessed.md`
**3988**, `resolve-top-issue-fable.md` **3966**. The witnessed spec is **6 characters** from
throwing. ⇒ Tensorbuild's model — `cat refusals.md worker-<mode>.md ADDENDUM.md M-<id>.md`
into one prompt — is structurally impossible here. fak's answer is the *pointer*: the fuel
names files the worker **reads** from the tree. That is why `refusals.md` exists as a file
rather than as prompt text, and why the rendered fuel is size-gated in Phase 2.
This skill's fuel measured **3361 bytes / cond 3367**, leaving 633 for substitution.

### The cap is a `min()`, and 30 does not fit today (§ 3, trap 3)

`python tools/dispatch_status.py --fast`:
```
workers : 0/24 live (headroom 24)  host=clean
limiter : configured_max (cap=24 live=0 max=24 target=0 host_cap=32 host_binding=cores seats=30 free=18)
switcher: account=aug5TWO-netra (t1) avail=True  preflight=SPAWN_OK
seats   : 34 seat(s) — available=5 busy=0 cooling=2 unavailable=2; slots free=20 leased=0;
          auth_failed=2 [jack-barker, zai2]
```
`tools/dispatch_preflight.py:127` sets `DEFAULT_MAX_WORKERS = _env_pos_int("FAK_MAX_WORKERS", 20)`
— the **code** default is 20, this box reads **24** via the env knob.

⛔ **The cap is NOT the wave size, and I got this wrong before running the plan.** The first
draft of this skill read the card's `cap=24` and wrote "a 30-ask under-fills to 24". The
actual dry-run of this skill's own Phase 3, same day, refutes it:

```
$ .\tools\launch_wave_detached.ps1 -Count 30 -PointerFile ".fak\wave\fw08081940.md" `
      -WorkKind engineering -Workspace "C:\work\fak"
WAVE PLAN  requested=30  allocation_requested=12  granted=12  shortfall=18  distinct_pools=3  target_tier=t1
  preflight: SPAWN_OK  live=0  cap=24  headroom=24  seat_free=12
  lane 1..12: aug5-netra / aug5TWO-netra / july20-netra, slots 1..4 of 6 each
  note: 18 lane(s) held by preflight headroom/seat limits before account allocation.
```

**Granted 12, not 24** — the wave must additionally buy *distinct-account session slots*
(`fak fleet-accounts wave`), and `seat_free=12` across 3 t1 pools binds well below the cap.
Two of 34 seats were `auth_failed` (`jack-barker`, `zai2`) at this reading, so the binding
term moves with seat health, which is why SKILL.md Phase 1 makes the plan the pricing
instrument instead of arithmetic over the card. ⭐ **This is the general lesson: a number
read from a status card is not the number the launcher will act on. Run the plan.**

### The default `-Workspace` fails preflight on this box (§ 3, trap 3b)

Same command, only `-Workspace C:\work\fleet` (the launcher's default at
`launch_wave_detached.ps1:51`):
```
WAVE PLAN  requested=30  allocation_requested=0  granted=0  shortfall=30  distinct_pools=0
  preflight: REFUSE_INSPECT  live=0  cap=4  headroom=4
             reason=guard not found: C:\work\fleet\tools\proc_resource_guard.py
```
`C:\work\fleet` is a real but stale sibling checkout that lacks
`tools/proc_resource_guard.py`; preflight is fail-safe, so a missing probe is
`REFUSE_INSPECT` rather than a pass. `C:\work\fak` has the file. ⇒ **two** of the launcher's
defaults are wrong for this repo (`-PointerFile` and `-Workspace`), which is most of the
answer to "why isn't this a one-liner".

⭐ The refill door is `fak dispatch auto --live`, whose whole design
is "compute Target, compute Refill = Target − live, drive the priced wave with that count" —
it reads live population instead of racing it, which the launcher's per-spawn gate cannot do
for still-starting siblings (`launch_wave_detached.ps1` .WHY, and the `super-loop` caveat).

### Every lane lease on this box is dead-holder — and reaping is the larger risk (Phase 0)

Same card: `lane lease: 48 held — live=0 dead-holder=48 unknown=0`, oldest 559 h
(`adjudicator`, `agentdojo`, `ggufload`, `ci`, +8). The card's own `next:` fold is
unambiguous and is quoted rather than paraphrased in SKILL.md because it inverts the obvious
move: `dos lease-lane release` runs **no liveness check** and its `--owner ""` matches any
holder (#5859), so a sweep evicts live work; there is no sanctioned per-lane reap verb
(`lane_lease.adopt()` and `OP_SCAVENGE` have no CLI surface); and none is needed, because
`live_leases(config, expire_dead=True)` already elides them at read time for `pretool_sensor`,
`decisions.py`, `dispatch_top` and `dos arbitrate`. A stale lease blocks exactly one
consumer — `lane_lease.acquire()` (`dos/lane_lease.py:453`) — so the correct response is to
watch for a repeated acquire REFUSE on **one** lane and escalate that lane.

### Guarded-by-default is real, and hand-wrapping is what breaks it (§ Phase 3)

`launch_goal_detached.ps1:300-321` wraps each worker in its **own** `fak guard` gateway
(`-Guarded` default) *after* stripping `ANTHROPIC_*` and the session-identity vars at
`:267-278`, then pins `CLAUDE_CONFIG_DIR` (`:282`) and `CLAUDE_CODE_OAUTH_TOKEN`. The
in-file comment records the failure that ordering prevents: a child inheriting a guarded
parent's loopback `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_KEY` bills the parent's seat and dies
when the parent gateway exits — the whole-wave same-instant crash, observed 2026-07-01,
child-stderr tell `claude.ai connectors are disabled because ANTHROPIC_API_KEY … is set`.
⇒ **The skill forbids hand-wrapping workers precisely because the launcher already does it
correctly.** Corroborating volume from the same card: `guard: 513 session(s), 50055
decision(s) [DENY=199 QUAR=293 CRASH=44]` — the guarded route is the well-trodden one here.

⚠️ Nuance worth keeping: nesting a guard *inside* an already-guarded session is possible but
needs `--base-url $ANTHROPIC_BASE_URL --api-key-env ANTHROPIC_API_KEY --managed-cache off
--vcache-anchor=false --compact-history-budget 0`, or two body-rewriting proxies conflict and
the outer gateway returns HTTP 400 malformed. That is the *interactive* nesting recipe. The
wave does not use it and must not: the launcher's strip-then-pin is the seat-hygiene-correct
path for detached workers.

### Attribution comes free from the pointer filename (§ Phase 2, trap 9)

`launch_goal_detached.ps1:252-256`: `$tag = [IO.Path]::GetFileNameWithoutExtension($PointerFile)`,
then `$LogDir/$tag-$stamp.{out.log,err.log,pid,in.txt}`. So a per-wave rendered pointer makes
every artifact self-identifying with no extra machinery. The hazard it defends against is
concrete: `C:\work\fleet\.goal-runs` currently holds **941** run artifacts, and ids recycle.
A probe that finds a `.pid` or a log at a reused tag is reading a *previous* wave's evidence
— the stale predecessor vouching for the corpse. `.fak/` is gitignored (`.gitignore:576`),
so rendering the fuel there never dirties the shared trunk.

### Workers self-select, so `fak intent` is not optional (§ 4, traps 7–8)

`fak intent claim --target T --holder H [--ttl SEC]` exists and is installed (`fak intent
--help`; refs under `refs/fak/locks/intent-*`, TTL default 3600, `rc=3` = `INTENT_COLLISION`
naming the incumbent). It is the **complement** to the file-tree lease, in the verb's own
words. The reason it is mandatory *in the fuel* rather than run by the orchestrator is
structural and is the sharpest difference from the ported skill: fak's fuel has each worker
pick the top-ranked ready leaf routed to **its own** lane
(`.claude/goal-prompts/resolve-top-issue-witnessed.md`, via
`python tools/issue_lane_router.py --view p0-p1 --json`). The orchestrator does not know
which ticket any worker will take, so it **cannot** pre-claim the roster. Tensorbuild's skill
claims at ranking time because its waves are orchestrator-assigned; porting that instruction
verbatim would have produced a rule no fak orchestrator could execute.

### Context for the closing target (§ Phase 1)

Same card, so the target is priced against something rather than asserted:
`watch: STALLED (LOOK; 6h) open 1361→1364 loop-closed 106→106`;
`supply: DRAINING arrival −0.167/h vs service 0.083/h over 23.998h`;
`backend: majority-stub recent logs [claude=1/1 stub]`. ⇒ A 30-closes-in-4-hours target is
**well above** the service rate this box has recently shown, and `loop-closed` did not move
over the window. That does not make the target wrong to *pursue*; it makes it dishonest to
*promise*. Hence the Phase 1 rule: price the target off rate, name the gap, and report
closes — never launches — at the end.

---

## PORTED — tensorbuild `fleet-wave`, not re-measured on fak

Kept because the failure shape transfers and each one cost that fleet a real wave. ⛔ Every
number below belongs to that fleet.

- **Depth buys nothing.** Over 1,385 workers / 53 waves: output-per-turn is flat with depth
  (1,203 → 1,161 tokens from the 21–40 band to 121+) while context cost per turn rises ~1.6×;
  workers past 80 turns were 17.0% of the roster and **34.7% of the bill**. ⇒ the fuel's
  landing deadline exists to convert depth into a land-or-park, and the share held constant
  across three days, so it is structural rather than noise. This is the whole justification
  for `LANDING DEADLINE` being the fuel's only schedule rule.
- **68.3% of a wave's bill is context, 31.3% is the model producing anything** — the reason
  this file is not loaded by workers and the fuel is a pointer. (fak's own per-worker cost is
  **UNMEASURED**; do not substitute these figures for it.)
- **A short report late in a run is the death signal; a `[fak]`-prefixed result is NOT.** The
  guard *prepends* per-tool feedback to complete results, so the prefix rule went 0-for-3 —
  it flagged the three longest reports as dead and missed the one real death. Acting on it
  relaunches your best workers and leaves the corpse running.
- **The third exit class: finished nothing, reported everything.** A worker exited at 77
  turns / $7.94 with a complete 25-file audit and seven findings, its own status line still
  `IN-PROGRESS`, one finding landed. ⇒ triage on what actually **landed**, not on report
  length — which is why the fuel mandates a `LANDED:` field and Phase 5 reconciles from
  `git log`.
- **A monitor that announces a wait dies after one pass**, returning a clean exit
  indistinguishable from a healthy watch. Two did.
- **`git tag -f` on a re-park cost half of one wave**, and three of six workers on another —
  workers exited `IN-PROGRESS` with a good tag already published and then destroyed it.
- **Restating a refusal rule in a second preamble is how it drifts.** Four workers were lost
  to a rule that existed in one file and not the other three; an instruction to "diff the
  other preamble when you edit one" is the thing that failed. One canonical file, read
  unconditionally, is the only version that held — hence `refusals.md`.
- **A count is not evidence; print the line.** `grep -c` self-matches the searching shell and
  returned a confident `1` for two dead processes; the true answer was 0.
- **`hold/*` refs strand.** 282 stranded refs, every one a **tag** (`refs/tags/hold/*`), while
  a health check over the `refs/hold/*` namespace alone read as *over*-published. Check both.

## Named, deliberately NOT ported

- **`tb lanes` acquire/release retry loops and their two `lock-busy` spellings.** fak's
  dispatch path prices lanes in-process (`dispatchorder`) rather than shelling out per lane,
  and workers take lanes via `dos arbitrate`. Porting the retry recipe would teach a verb
  surface fak does not use here.
- **Per-worker `BASELINE.md` unit economics.** fak has no equivalent measured corpus, and a
  fabricated baseline is worse than none. If someone measures one, it belongs in a new file
  beside this one, with its regeneration command.
- **`--assume-session-turns` / `--compact-history-budget` orchestrator tuning.** Those govern
  a *guarded interactive* session's compaction. Wave workers are guarded individually by the
  launcher, so the orchestrator's own flags do not propagate to them — the tensorbuild
  inheritance argument does not hold on this path. **UNMEASURED here; do not cargo-cult.**
