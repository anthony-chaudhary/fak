---
name: operator-heaviness-score
description: One repeatable pass that keeps fak light to DRIVE — the operator-facing counterpart of steerability-score. Runs the operator-heaviness scorecard (`fak operator heaviness`) over the live operator surface (the cmd/fak dispatch table, the front-door verb's flag set, the dos.toml refusal vocabulary, and whether the doc map makes the steering surfaces discoverable), reads the unbounded `heaviness_pressure` headline plus the HARD `heaviness_debt` gate, drives pressure DOWN and HARD debt to zero by REAL surface consolidation (group verbs, cut front-door flags, fold refusal reasons, wire a missing doc-map entry or appeal channel) — never by gaming a detector, re-measures to PROVE the pressure fell, regenerates the committed snapshot, and commits only the touched lane by explicit path. The operator-surface counterpart of steerability-score (package-graph shape) and the steering-effort companion of quality-score (code defects). Use after a verb, a front-door flag, or a refusal reason is added, when the dispatch table "feels" like a navigation wall, when `fak guard` grows another flag, or on a /loop cadence to keep the surface an operator must hold flat as the repo grows.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
argument-hint: "[--workspace PATH] [--compare baseline.json]  (no args = full measure + one consolidation iteration)"
---

# /operator-heaviness-score — keep the surface an operator drives flat as the repo grows

> **What this does.** Steerability-score grades the SHAPE of the package graph — how hard
> the CODE is to change. No sibling answers the question an OPERATOR asks: as fak doubles
> in verbs and flags, does the effort to **discover, drive, and recover** the surface stay
> roughly flat, and if it drifts do we *know* and can we *correct*? This is that pass. It is
> an instance of the shared [`scorecard`](../scorecard/SKILL.md) doctrine, pointed at the
> surface a person (or agent) actually DRIVES.

The shape: **run the scorecard → read the pressure + the worst magnitude KPI → drive
pressure DOWN with a REAL consolidation → re-measure and prove the pressure fell →
regenerate the snapshot → commit ONLY your lane by explicit path.**

The headline is **`heaviness_pressure`** — an *unbounded* integer that sums, per magnitude
KPI, the share of headroom consumed between the comfortable soft line and the blowout
ceiling (each term normalized to its own span so a verb and a flag are commensurable).
Lower is lighter. It is deliberately NOT a 0–100 grade: a grade would flatten the one fact
that matters — how far over the comfortable lines the surface sits today. `heaviness_debt`
is the HARD gate, ~0 on a disciplined tree.

---

## The measure (six KPIs, four groups, read from the live surface)

`fak operator heaviness` reads the operator-facing surface straight from the source — the
`cmd/fak` dispatch table, the front-door verb's flag set, the `dos.toml` refusal vocabulary,
and the `llms.txt` doc map — so it cannot be gamed by editing a JSON data file, only by
genuinely slimming the surface. Two KPIs emit HARD debt; four are SOFT magnitudes that score
the pressure but never emit debt (because their only cheap "fix" would be deleting a real
guard, which is not a fix).

| Group | KPI | HARD/SOFT | what it measures (the operator load) |
|---|---|---|---|
| discoverability | `docmap_covers_steering` | **HARD** | every steering surface is reachable from a *linked* doc-map entry (not a bare mention) |
| recovery | `appeal_channel_wired` | **HARD** | the dispatch table routes an in-product appeal verb (`fak complain`) for a wrong refusal |
| surface | `cli_verb_count` | SOFT, HARD past 200 | top-level subcommands an operator must hold (soft 90, blowout ceiling 200) |
| surface | `meta_verb_share` | SOFT | share of verbs that are meta-scorecards/RSI clutter (soft 8%, ref 25%) |
| config | `front_door_flag_burden` | SOFT, HARD past 80 | flags on the front-door verb `fak guard` (soft 20, blowout ceiling 80) |
| ceremony | `refusal_vocab_size` | SOFT | structured refusal reasons an operator must learn (soft 12, ref 30) |

**Why only two HARD KPIs.** Both have a genuine, non-gaming fix: a steering surface the doc
map hides is fixed by writing a real linked doc-map entry; a missing appeal channel is fixed
by wiring the verb. The four magnitude KPIs are SOFT because their cheapest "fix" — deleting
a verb, a flag, or a refusal reason — would delete a real capability or guard. They score
the pressure (so an operator sees the load climbing) and turn HARD only at the blowout
ceiling (verb count > 200, flag count > 80), where the fix is a genuine consolidation
program, not a tweak.

---

## Step 1 — Run the scorecard (it builds your work-list)

From the repo root:

```bash
fak operator heaviness                 # human scorecard (pressure + per-KPI + verdict)
fak operator heaviness --json          # machine payload (the loop uses this)
```

Read `corpus.heaviness_pressure`, `corpus.heaviness_debt`, and `corpus.pressure_by_term`
(the per-KPI headroom consumed — the **drift work-list**, where the load concentrates). If
`heaviness_debt > 0`, a HARD defect is live and that is the first thing to retire.

**Record the baseline before you change anything** so you can prove the delta:

```bash
fak operator heaviness --json > heaviness-baseline.json   # local, per-pass; not committed
```

(This is a per-pass local anchor like steerability-score's `baseline.json` — the pressure
moves as verbs/flags land, so a fixed committed anchor goes stale immediately; regenerate
it each pass and point `--compare` at it.)

## Step 2 — Drive pressure DOWN, HARD debt to zero, using REAL consolidation only

If `heaviness_debt > 0`, retire the HARD defect first — it is the only thing that reds the
control-pane ratchet. Then attack the largest `pressure_by_term`. The genuine moves, by KPI:

- **`docmap_covers_steering` (HARD, discoverability).** A steering surface (steerability,
  operator-heaviness) is not reachable from a *linked* doc-map entry. Fix it by adding a
  real `llms.txt` line that links to the surface's doc (e.g.
  `- [Operator-heaviness scorecard](docs/OPERATOR-HEAVINESS.md): ...`). A bare prose mention
  with no markdown link does NOT retire the gate — the oracle requires the link shape.
- **`appeal_channel_wired` (HARD, recovery).** The dispatch table has no in-product appeal
  verb for a wrong refusal. Fix it by wiring `fak complain` (the `internal/guardcomplaint`
  channel) into the dispatch table — a real recovery seam, never a stub `case` that exits 0.
- **`cli_verb_count` / `meta_verb_share` (surface, SOFT).** Every new top-level verb is one
  more thing an operator must discover; meta-scorecard/RSI verbs clutter without adding
  product capability. Drive pressure down by **grouping** verbs under a parent
  (`fak operator ...`, `fak guard-rsi ...`) so the top-level table shrinks — real
  consolidation, the same move that keeps a dispatch file steerable. NEVER delete a verb
  that has users to move the count.
- **`front_door_flag_burden` (config, SOFT).** `fak guard` carrying 60+ flags is its own
  steerability tax. Drive pressure down by **graduating** a stable cluster of flags into a
  subcommand or a policy-profile preset (`--policy examples/...`) so the front door shrinks.
  NEVER remove a flag that gates a real guard.
- **`refusal_vocab_size` (ceremony, SOFT).** Every `[reasons.*]` block is one more token an
  operator must learn. Drive pressure down only by **folding** two reasons that name the
  same recovery into one canonical token. NEVER delete a reason the kernel actually emits —
  that hides a real refusal from the closed vocabulary.

`heaviness_pressure` is advisory by design: a clean pass HOLDS it flat (no verbs/flags
added), a consolidation pass drives it DOWN. It is not a gate — do not chase a lower number
by deleting real surface.

## Step 3 — Validate every surface change (the honesty gate)

Any dispatch-table, flag, or doc-map edit must keep behavior identical and the build green.
Native `go test` is OS-blocked on the Windows dev box, so validate under WSL:

```bash
go build -o "$TEMP/fak-verify.exe" ./cmd/fak          # compiles (never write the in-tree binary)
go vet ./cmd/fak ./internal/heavinessscore/...
wsl -e bash -lc 'cd /mnt/c/work/fak && go test ./internal/heavinessscore/... -count=1'
```

A surface edit you have not built and tested is not done. Never add a stub verb/flag/reason
solely to move a number.

## Step 4 — Re-measure and PROVE the pressure fell

```bash
fak operator heaviness --json
fak operator heaviness --compare heaviness-baseline.json   # the pressure/debt delta + verdict
```

State the delta plainly: `pressure P → P' (−k)`, the KPI that moved, and whether `heaviness_debt`
reached 0. Regenerate the committed snapshot so the public doc tracks the surface:

```bash
fak operator heaviness --markdown > docs/OPERATOR-HEAVINESS.md   # use Bash '>'; PowerShell mangles glyphs
```

## Step 5 — Commit ONLY your lane, by explicit path

The scorecard reads five files across three lanes (`cmd`, `tools`, `docs`/`claude`). Commit
each lane's files in their OWN commit by explicit path — never `git add -A`:

```bash
git commit -s -F msg -- cmd/fak/<changed>.go                    # the surface edit (cmd lane)
git commit -s -F msg -- docs/OPERATOR-HEAVINESS.md llms.txt     # the snapshot + doc-map (docs lane)
dos commit-audit HEAD                                            # MUST print [diff-witnessed] / verdict OK
git push
```

Subject honesty: a real surface consolidation → `refactor(cmd):`; a doc-map/snapshot
regeneration → `docs(steer):` or `chore(steer):`. End every ship commit with a `(fak <leaf>)`
trailer matching the lane you touched so the `dos verify` referee binds it. Stay on `main`;
never force-push.

---

## The RSI loop

Each pass: measure → retire HARD debt, then drive the largest `pressure_by_term` down →
prove the pressure fell → commit witnessed. Next pass, the *new* largest term surfaces —
the loop walks the operator surface lighter because the pressure is re-derived from the live
dispatch table every time and can't be talked past. Because the soft lines sit below today's
surface on purpose, **a clean pass holds pressure flat (no new surface added)**, which is the
whole goal: the same driving effort at 2× the verbs.

## Anti-gaming laws (the pressure is only as honest as the pass)

1. **Never delete a verb, flag, or refusal reason to move a count.** The SOFT magnitude KPIs
   are SOFT for exactly this reason — deleting a real capability or guard is not consolidation.
   Group, graduate, or fold instead.
2. **Never satisfy `docmap_covers_steering` with a bare prose mention.** The oracle requires a
   *linked* doc-map entry (a markdown link on the same line as the token). Pasting the token
   into a sentence games the gate without aiding discovery.
3. **Never wire a stub `fak complain` case that exits 0.** `appeal_channel_wired` requires a
   real recovery seam (`internal/guardcomplaint`); a stub hides a wrong refusal with no appeal.
4. **Never re-derive a code-shape number here.** Steerability-score owns the package graph;
   this card owns the operator surface. Double-counting the same monolith is the orthogonality
   bug this split exists to prevent.
5. **Never commit a fixed pressure "baseline" data file.** The pressure moves as the surface
   lands; a committed anchor goes stale immediately and misleads the next `--compare`.
   Regenerate the local per-pass anchor each time (steerability-score's convention).
6. **Re-measure and `dos commit-audit` your commit** before claiming the pressure fell.

## When to run this

- To **baseline** operator-heaviness (first run records the pressure + the worst magnitude KPI).
- After the dispatch table **grows a verb**, `fak guard` **gains a flag**, or `dos.toml` **adds
  a refusal reason** — the three events that move the pressure.
- When the top-level verb list **feels like a navigation wall** — `cli_verb_count` names the
  cliff and grouping is the move.
- When a guard refusal has **no in-product appeal** — `appeal_channel_wired` is the HARD defect
  that reds the ratchet.
- On a `/loop` cadence to keep the driving surface flat as the repo grows between releases.

The scorecard is read-only; this skill's only writes are your genuine surface consolidations,
`docs/OPERATOR-HEAVINESS.md`, and `llms.txt`. It never deletes a real guard to move a number
and never edits the frozen ABI.
