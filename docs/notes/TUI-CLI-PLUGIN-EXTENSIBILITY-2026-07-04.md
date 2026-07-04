---
title: "Plugin extensibility for fak's operator surfaces: TUI panes first, then the verb switch (2026-07-04)"
description: "fak already has a real plugin model — internal/abi's Register* + reserved-range + atomic-snapshot seam — for compute backends and cross-worker policy. The operator-facing surfaces never got the same treatment: cmd/fak/tui.go dispatches panes through a 7-case switch and cmd/fak/main.go dispatches verbs through a ~200-case switch, both requiring a core-file edit to extend. This note designs a tuiplugin registry that ports the existing panes onto the ABI's own seam shape, a matching user-config layer for pane selection/ordering, and a path to keyboard-interactive panes without adopting a TUI framework — then generalizes the same recipe to the verb switch."
date: 2026-07-04
keywords:
  - plugin architecture
  - TUI extensibility
  - CLI verb registry
  - Register pattern
  - internal/abi
  - operator heaviness
  - user controls
  - pane registry
---

# Plugin extensibility for fak's operator surfaces

## 0. The one-paragraph version

fak's answer to "how do I extend the kernel without forking it" is already excellent —
`internal/abi/registry.go` gives every compute backend, adjudicator, engine, and
page-out codec a `Register*` call from its own `init()`, claims a disjoint numeric
range so two fleet workers can never collide, and reads back through one atomic
snapshot so the 1000th driver costs the 1st syscall nothing
([`EXTENDING.md`](../../EXTENDING.md), [`ARCHITECTURE.md`](../../ARCHITECTURE.md)).
**That discipline stops at the kernel boundary.** The two surfaces an operator
actually touches — `fak console <pane>` (`cmd/fak/tui.go:63-85`, a 7-case switch) and
`fak <verb>` (`cmd/fak/main.go:89` onward, a switch its own comment calls "the
200-case switch", `main.go:75`) — are both hand-written switches that require editing
a shared core file to add one more case. This is the same anti-pattern the ABI seam
was built to kill, just one layer up, and it is not hypothetical: the verb switch is
already at ~200 cases and already flagged as `operator-heaviness` pressure by the
`operator-heaviness-score` skill. This note proposes porting the *existing* built-in
panes onto a new `internal/tuiplugin` registry (proving the seam by dogfeeding it,
exactly like `ARCHITECTURE.md`'s "bake-in walkthrough"), a small user-config layer for
which panes show and in what order, and a route to interactive (keybinding-driven)
panes that reuses `golang.org/x/term` — already a dependency — instead of adopting a
TUI framework. Section 4 generalizes the same recipe to the verb switch.

## 1. What "plugin" already means in fak, and where it stops

Grounding first, so this note adds one new seam rather than a second philosophy:

| Seam | Where | What self-registers | Extended by |
|---|---|---|---|
| `internal/abi` `Register*` | frozen ABI core | adjudicators, fast-paths, engines, region backends, page-out codecs, verdict kinds | a new leaf package + one `Register*()` in `init()` ([`ARCHITECTURE.md:45`](../../ARCHITECTURE.md)) |
| `internal/compute` `Register` | device HAL | quantization/GPU/NPU backends | a new file behind a build tag, `init(){ Register(...) }` ([`EXTENDING.md:62-75`](../../EXTENDING.md)) |
| `internal/registrations` | wiring point | nothing itself — blank-imports the built-in drivers | adding/removing one `_ "import"` line |
| **`fak console <pane>`** | `cmd/fak/tui.go:63-85` | **nothing — a `switch argv[0]` naming 7 hardcoded functions** | editing `tui.go`'s switch + adding a `runTUI<Name>` function |
| **`fak <verb>`** | `cmd/fak/main.go:89` onward | **nothing — a `switch os.Args[1]` naming ~200 hardcoded functions** | editing `main.go`'s switch + adding a `cmd<Verb>` function |

The first three rows are the precedent: disjoint numeric ranges, panic-on-clash,
one atomic snapshot, writers-expensive/readers-O(1). The last two rows are exactly
the shape `ARCHITECTURE.md` describes as the thing to avoid — "the spine never
changes after freeze; everything else attaches as *a new package + one
`Register*()` call*, never an edit to the core" — except `tui.go` and `main.go`
**are** the thing being edited every time. No existing docs/notes memo proposes
fixing this; `internal/architest`'s five-tier layering gate doesn't reach `cmd/fak`
at all (it fences `internal/`), so there is no structural pressure stopping the two
switches from growing forever.

## 2. The TUI pane registry

### 2.1 Current shape

`cmd/fak/tui.go` is not a live TUI (no bubbletea, no event loop — `go.mod` pulls in
only `golang.org/x/term` and `golang.org/x/sys`). Each pane is: parse flags → build a
typed `tui*Report` struct (shared shapes in `tui_types.go`) → render once to a string
→ print. The seven panes today — `issues` (`tui_issues_garden.go`), `loops`
(`tui_loop_render.go`), `sessions`/`overview` (`tui_overview_sessions.go`), `garden`
(`tui_issues_garden.go`), `guard` (`tui_guard_report.go`, 1142 lines — the largest),
`agent` (`tui.go` itself) — each duplicate their own `build*Report`/`render*` pair and
their own schema constant (`tuiIssuesSchema`, `tuiGuardSchema`, ... `tui.go:32-38`).
The only live behavior is `--follow` on `tui guard` (`tui.go:461-484`): a polling
ticker that just re-runs the same build→render pair on an interval.

### 2.2 Proposed seam: `internal/tuiplugin`

Give every pane the exact shape `internal/abi` already proved out — a registry
package the panes import, never the other way around:

```go
// internal/tuiplugin/registry.go
package tuiplugin

type Spec struct {
    ID       string                                   // "guard", "loops", ...
    Summary  string                                   // one-line, for `fak console help`
    Flags    func(*flag.FlagSet)                       // pane-specific flags
    Build    func(ctx BuildCtx) (Report, error)         // gh/journal/gateway read → typed report
    Render   func(Report, RenderOpts) string            // report → terminal text
    Score    func(Report) int                           // attention weight, for `overview` roll-up
}

func Register(s Spec) { /* panics on duplicate ID — same discipline as abi.Register* */ }
func Lookup(id string) (Spec, bool)
func All() []Spec // stable order, for `overview` and `help`
```

Each existing pane moves into its own file with an `init(){ tuiplugin.Register(...) }`
call — a pure code-motion, not a behavior change, which is exactly the low-risk shape
`/modularize` looks for. `runTUI` in `tui.go:58-86` shrinks from a 7-case switch to:

```go
func runTUI(stdout, stderr io.Writer, argv []string) int {
    if len(argv) == 0 { tuiUsage(stderr); return 2 }
    spec, ok := tuiplugin.Lookup(argv[0])
    if !ok { /* ...unknown subcommand, tuiUsage... */ }
    return dispatchPane(spec, stdout, stderr, argv[1:])
}
```

A **new** pane — a plugin, in the sense this note means it — is a new file that calls
`tuiplugin.Register` from `init()`, plus one blank-import line in a
`cmd/fak/tuiplugins/` wiring file (mirroring `internal/registrations`). Zero edits to
`tui.go`. This is also how `overview` (`tui_overview_sessions.go:13-380`, already a
roll-up of the *other* panes) gets simpler: it stops hand-listing panes and instead
walks `tuiplugin.All()`, sorting by `Score`.

**Deliberately not proposed:** dynamic loading (Go's `plugin` package / `.so` files).
It has no Windows support (fak ships and is dogfooded on Windows — see this repo's own
host), and even on Linux it's brittle across toolchain versions. Every other extension
seam in fak is in-tree-Register-from-init, not dlopen; a TUI plugin should be an
ordinary Go package a contributor adds via a normal PR, identical in spirit to how a
new compute backend lands. "Plugin" here means *structurally separable*, not
*dynamically loaded* — consistent with `EXTENDING.md`'s whole thesis that you attach
through a named seam, the kernel never imports your code, and it still ships as one
binary.

### 2.3 User controls — pane selection, ordering, defaults

There is currently no persisted user preference for the TUI at all — only per-invocation
flags and scattered `FAK_*` env vars. fak already has the right shape for a user
override file: `~/.fak/targets.json` (or `$FAK_TARGETS_FILE`) is an **additive
override over the built-ins**, resolved by `envOrHomePath` (`cmd/fak/computetarget.go:261-275`).
Reuse that shape rather than inventing a new config format:

```jsonc
// ~/.fak/console.json (or $FAK_CONSOLE_FILE)
{
  "overview_panes": ["guard", "loops", "issues"],   // subset + order; omit = all, registry order
  "pane_defaults": {
    "issues": {"top": 40, "width": 160},              // pre-fills flags so `fak console issues` alone matches a saved view
    "guard":  {"min-attention": 200}
  }
}
```

`tuiplugin.BuildCtx` reads this file once (same additive-override resolution as
`computetarget.go`) and pre-seeds `Spec.Flags` defaults before `flag.Parse` runs, so a
CLI flag still wins over the file, which still wins over the built-in default — the
same three-tier precedence rule `EXTENDING.md`'s registries already use conceptually
(writes cheap at load, reads O(1) after).

**Interactivity, without a framework.** "User controls" for a render-once-and-print
tool are necessarily static (which panes, what order, what defaults). If fak wants
live keybindings — `n`/`p` to cycle panes, `r` to refresh, `/` to filter rows — the
cheapest path is to extend the *existing* `--follow` ticker (`tui.go:461-484`) into a
raw-mode keypress reader using `golang.org/x/term.MakeRaw`, which is already a
dependency, rather than pulling in bubbletea/tview. A keypress just re-invokes
`dispatchPane` with a different `Spec` from the same registry — the plugin contract
from §2.2 is what makes this cheap; the alternative (bubbletea's Model/Update/View)
would mean every one of the 7 panes gets rewritten twice. **This is worth flagging,
not building now** — see §6.

## 3. Why the verb switch is the same problem, further along

`cmd/fak/main.go:75`'s own comment calls it "the 200-case switch" and documents that
`fak dev <verb>` deliberately rewrites `os.Args` *before* the switch specifically so
the switch itself stays untouched (`main.go:72-88`) — i.e. the codebase already knows
editing that switch is a cost worth routing around, it just hasn't named the fix. The
`operator-heaviness-score` skill independently flags "the cmd/fak dispatch table" as a
`heaviness_pressure` contributor. A verb registry would look identical in shape to
§2.2: `Spec{ID, Summary, Run func([]string) int}`, `Register` panicking on a duplicate
ID, `main.go`'s switch replaced by a lookup. The `devindex` scanner mentioned in
`main.go:76` (keyed on the literal `switch os.Args[1]` header) would need to switch to
scanning the registry instead — the one real migration cost, and the reason this note
scopes the *proof* to the TUI pane switch (7 cases, no external scanner depending on
its shape) before touching the verb switch (200 cases, a scanner bound to its exact
syntax).

## 4. Orthogonality — what this does NOT propose

- **Does not touch `internal/abi` or `internal/compute`.** Those seams are correct,
  frozen, and the precedent this note follows — not a target for change.
- **Does not add a TUI framework dependency.** The render-once model is fine; §2.2's
  registry formalizes its existing shape, it doesn't replace it with bubbletea.
- **Does not open dynamic/out-of-tree code loading.** No `.so` plugins, no dlopen, no
  WASM sandboxing. "Plugin" = an in-tree Go package behind a `Register` call, same as
  every other fak extension point.
- **Does not change `architest`'s layering gate.** `cmd/fak` is already outside the
  `internal/` five-tier DAG; this note doesn't propose folding it in, just giving it
  its own much smaller registry discipline.

## 5. Promotion / demotion / assumption

- **Promotion evidence**: port 2 panes (`guard` — the largest, and `loops` — the
  smallest) onto `internal/tuiplugin` as a proof-of-concept; green
  `.\fak\test.ps1 ./cmd/fak/...`; `tui.go`'s switch shrinks by exactly those two cases.
  That diff, plus one dogfood readout (`fak console overview` unchanged output before
  vs. after the port) is the witness a promotion to "all 7 panes migrated" would need.
- **Demotion / retirement evidence**: if pane count never grows past ~7 and no
  external contributor has ever asked to add one, the switch isn't actually costing
  anyone anything and this stays a `docs/notes` idea, not a ticket — per
  [`CLAUDE.md`](../../CLAUDE.md)'s "don't design for hypothetical future
  requirements." The verb-switch side (§3) does **not** get this out: 200 cases and an
  explicit "we route around editing it" comment (`main.go:72-88`) is not a
  hypothetical, it's already-paid cost.
- **Invalidating assumption**: this assumes new panes/verbs keep being added by people
  other than whoever currently owns `tui.go`/`main.go`. If growth stays flat and
  single-owner, a registry adds indirection for no collision benefit — re-check pane
  count and unique-committer count on both files before promoting either half.

## 6. Continue here

For a future agent: the smallest next unit of work is the `internal/tuiplugin`
registry (§2.2) plus the two-pane port named in §5 — pure code motion, testable by
diffing `fak console guard` / `fak console loops` output before and after. **Do not**
start with the verb switch (§3) or the interactivity path (§2.3's keypress reader) —
both are named here as the generalization and the natural follow-on, not the
increment to ship. `fak issue fanout` (the repo's spine-first + fan-out convention,
[[spine-fanout-defaults-shipped]] in project memory) is the right tool to turn §2, §3,
and §2.3 into a scoped epic + follow-on issues once the two-pane proof lands — filing
that fan-out before the proof exists would be scope pre-committed to an unproven
seam.
