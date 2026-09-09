# CLAUDE.md

> **Not the human contributor guide.** This file is operating instructions for *automated*
> contributors working inside the maintainers' shared checkout. Humans want
> [`README.md`](README.md) for what fak is and [`CONTRIBUTING.md`](CONTRIBUTING.md) for how
> to contribute; nothing on this page applies to you.

## The canonical instructions

The canonical agent instructions for this repo are in **[`AGENTS.md`](AGENTS.md)** —
read it first for build/test/run, the repo map, and the rules.

## The six hard rules

The six that will bite you if you skip them:

- **Work directly on the trunk (`main`). Never open a feature branch or new worktree** —
  the trunk guard *refuses* off-trunk commits (`OFF_TRUNK`). The *one* sanctioned
  exception is a **detached** per-worker worktree that lands its diff back on `main` via
  `fak worktree worker prepare|land|reap` (build isolation, #1334 / epic #3165): because
  it is detached (never a branch) and commits only through the serialized land under the
  worker's lane lease, it is not off-trunk and never trips `OFF_TRUNK`. Feature branches
  and any other off-trunk commit stay forbidden. (Details in [`AGENTS.md`](AGENTS.md).)
- **Commit via `fak commit --path` or `fak sweep` by explicit path** — all commits must be made via
  `fak commit --path <p> -m "<subject> (fak <leaf>)"` or `fak sweep --apply --lane <lane> -m "<subject>"`.
  `fak commit` locks the lane, stages exactly your paths, provides automatic DCO sign-off (with `-s` accepted for compatibility),
  and verifies that no peer files were raced in (`PATHSPEC_RACE`). End every ship commit's Conventional-Commits
  subject with a bindable `(fak <leaf>)` trailer so the `dos verify` referee can bind it — e.g.
  `fix(gateway): treat same-tick ready as positive (fak gateway)`. A bare un-stamped subject stays NOT_SHIPPED.
  The [`/commit-clean`](.claude/skills/commit-clean/SKILL.md) skill mechanizes this rule end to end.
  WARNING: Raw git commits are strictly an emergency-only fallback, permitted ONLY when the `fak` binary is unbuilt
  (`git commit -s -m "<subject> (fak <leaf>)" -- <paths>`); never use `git add -A` or uncoordinated raw git commits
  on the shared multi-session tree. (Full convention in [`AGENTS.md`](AGENTS.md).)
- **Default is to ship green work and sync safely unprompted** — do more work by default:
  pre-flight sync with trunk (`fak sync check`, `fak sync reconcile --apply`, or `fak sync apply`),
  verify on-device (`fak validate --mine <paths>`, `go test ./internal/<pkg>/...`), commit by explicit path
  (`fak commit --path <p> -m "<subject> (fak <leaf>)"` or `fak sweep --apply --lane <lane> -m "<subject>"`), and push unprompted via `fak sync push` or `fak commit --push`.
  "Green" requires shift-left proof: for changes touching executable CLI verbs, gateway adapters, runtime logic, or hardware/compute paths,
  execute real paths in dogfood, integration tests, or live hardware sub-component runs rather than relying on mock-only or shallow tests.
  Bias heavily toward testing sub-components on live physical hardware (divide and conquer) with high volume and frequency (e.g. `fak validate --strix --subkernels=...`, `make mac-perf`, `make cuda-test`).
  Safe merge discipline: verify no in-flight `MERGE_HEAD` exists before staging (if active, unstage and wait; never clobber, abort, or finish a peer's merge); use `fak sync apply` (`--ff-only`) for clean trunk convergence; route divergence via `fak sync reconcile` (disjoint integration, superset merge, or dirty parking via `fak wip park`); never force-push, use `--autostash`, or perform raw 3-way merges.
  Dual-repo synchronization invariant: when working across both repos or touching shared interfaces (`pkg/*`), keep both `fak` and `fak-private` synchronized (fetch both remotes and `refs/fak/locks/*`, fast-forward both, sync `go.work` to prevent module skew, and audit with public leak scrub `tools/scrub_public_copy.py --audit-staged`). Cross-repo issue quoting rule: bare `#<num>` strictly denotes a public `fak` issue; quoting a `fak-private` issue in public `fak` MUST be explicitly qualified as `fak-private#<num>` (or `anthony-chaudhary/fak-private#<num>`) to ensure provenance is unambiguous. Full default + verify command in [`AGENTS.md`](AGENTS.md).
- **Divide and conquer: Delegate substantive work and keep this coordinator context clean; enforce capability-aware scoping and persistence** —
  structurally drive 10x subagent adoption by launching 4–8 (up to 16 on multi-core hosts) specialized subagents concurrently across pairwise tree-disjoint lanes.
  Decompose substantive or multi-part requests into atomic single-concern units and deploy multi-agent triads per leaf (researcher/explore -> worker -> cross-validator/issue-auditor).
  Divide and conquer applies equally to hardware verification: isolate compute workloads into discrete sub-components (sub-kernels, GEMV/GEMM microbenchmarks, Vulkan primitives) to test on live physical devices early and often, rather than bottlenecking on monolithic full-model runs.
  Top-level coordinator (depth 0) aggressively fans out parallel subagents; leaf workers (depth 1) execute directly within assigned package boundaries and must NOT invoke nested `task` calls (preventing recursion depth exhaustion, #12028).
  Use guarded headless agents or equivalent isolated workers for investigation,
  implementation, tests, and review. Constrain smaller models and workers to atomic S0/S1 leaf units (1–3 files,
  single package, exactly one witness). Scope abstention strictly to bounded high-difficulty aspects
  (concurrency, frozen ABI, kernel memory layout, security gates): emit a structured `ABSTAIN`
  for the escalated boundary while executing all independent, safe, solvable sub-components (such as
  reproduction tests or diagnostics). Treat guard refusals as actionable feedback rather than session stops:
  query `fak recover <TOKEN>`, adapt execution or wait out transient locks, and maintain momentum on the
  objective without repeating failing calls. Keep only decisions and compact witnessed
  evidence here; independently verify
  worker effects before landing or reporting them. Reserve
  direct work for lightweight coordination and truly trivial tasks. Full contract in [`AGENTS.md`](AGENTS.md).
- **The Go module is the repository root** — run `go` commands from the clone root;
  `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` resolves directly.
- **Strict Ban on PowerShell and Loose Scripts — Mandatory Modular Go Programs** —
  adding new PowerShell scripts (`.ps1`), Bash/POSIX shell scripts (`.sh`), batch files (`.bat`, `.cmd`),
  or loose scripts is **STRICTLY BANNED** across this repo and `fak-private`. An exception requires an
  extraordinary, super heavily justified reason (basically ~1 in the whole repo, such as initial host
  bootstrapping before Go is installed). All new automation, test runners, harnesses, session orchestrators,
  and background tooling **MUST be native Go programs** (`cmd/*`, Go leaves registered as `fak` CLI verbs,
  or quarantined sub-modules). Think modular, integrated, long-term value.

## Move forward over conclusions

Frame all diagnostics, benchmark reports, and investigations around **forward momentum rather than
terminal conclusions**. When an approach underperforms, a benchmark trails baseline, or an experiment
fails, focus on forward action rather than passive or defeatist editorializing (e.g. rather than
*"X didn't work ... therefore we suck..."*, state *"the next step to get better performance is X"*).
Treat unmet targets as empirical data that eliminates a variable and always formulate the concrete
next checkable step.

## Doc map

Doc map for humans and agents: [`llms.txt`](llms.txt). Full contributor contract:
[`CONTRIBUTING.md`](CONTRIBUTING.md).
