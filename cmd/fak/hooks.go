package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// cmd/fak/hooks.go — `fak hooks <pre-commit|commit-msg>`: run the repo's commit-boundary gates
// IN ONE PROCESS instead of spawning a Python interpreter per gate. The shell hooks call this
// once; it reads the staged diff once and runs every gate over it. Exit codes mirror the gate
// contract so the shell wrapper can fall back to Python: 0 = clean/pass, 1 = a block gate fired,
// 2 = could-not-run (the wrapper then runs the Python path — fail-open).
//
// Exit 0 does NOT by itself mean every gate ran. A single gate whose Check cannot reach its
// evidence is skipped and the run still exits 0 (fail-open, deliberate: a broken checker must
// never wedge every commit on a shared trunk). What tells the two apart is the skip ledger
// (#5299): each skipped gate is named on stderr, and the count plus the names ride in the --json
// payload as skipped_count / skipped_gates. Read that ledger — not the exit code — to know
// whether a green commit was checked by the full gate set or a degraded one.

func cmdHooks(argv []string) { os.Exit(runHooks(os.Stdout, os.Stderr, argv)) }

func runHooks(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak hooks: subcommand required (pre-commit | commit-msg <file> | pre-push | import-witness | popup-scan | claim-reclass | lane-audit | agent <pretool|posttool|stop>)")
		return 2
	}
	switch argv[0] {
	case "pre-commit":
		return runHooksPreCommit(stdout, stderr, argv[1:])
	case "commit-msg":
		return runHooksCommitMsg(stdout, stderr, argv[1:])
	case "pre-push":
		return runHooksPrePush(stdout, stderr, argv[1:])
	case "import-witness":
		return runHooksImportWitness(stdout, stderr, argv[1:])
	case "popup-scan":
		return runHooksPopupScan(stdout, stderr, argv[1:])
	case "claim-reclass":
		// Reads the claim-honesty review's output on stdin; the shell rung pipes it in.
		return runHooksClaimReclass(stdout, stderr, os.Stdin, argv[1:])
	case "lane-audit":
		return runHooksLaneAudit(stdout, stderr, argv[1:])
	case "agent":
		// Agent-LIFECYCLE hooks, not the commit boundary. Reads the harness envelope on stdin
		// and answers on a DIFFERENT exit-code contract — see cmd/fak/hooks_agent.go.
		return runHooksAgent(stdout, stderr, os.Stdin, argv[1:])
	default:
		fmt.Fprintf(stderr, "fak hooks: unknown subcommand %q (pre-commit | commit-msg | pre-push | import-witness | popup-scan | claim-reclass | lane-audit | agent)\n", argv[0])
		return 2
	}
}

// runHooksLaneAudit reports every internal/<leaf> Go package with no declared dos.toml lane —
// the standing, whole-tree form of the per-commit leaf check the commit-msg stamp lint runs. Each
// such leaf's `(fak <leaf>)` ship-stamp binds to a phantom unit and the arbiter cannot protect its
// edits. --gate N exits 1 when the count EXCEEDS N, so a ratchet can drive the drift to zero
// without reding the trunk on day one. Exit 0 report/at-or-under-gate, 1 over-gate, 2 could-not-run.
func runHooksLaneAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hooks lane-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	asJSON := fs.Bool("json", false, "emit the undeclared leaves as JSON")
	gate := fs.Int("gate", -1, "exit 1 if the count of undeclared-lane leaves exceeds this threshold (-1 = report only)")
	if !parseFlags(fs, argv) {
		return 2
	}
	r := resolveRoot(*root)
	if r == "" {
		return 2
	}
	gaps, err := hooks.UndeclaredLeaves(r)
	if err != nil {
		fmt.Fprintf(stderr, "fak hooks lane-audit: %v\n", err)
		return 2
	}
	if *asJSON {
		if encErr := writeIndentedJSON(stdout, map[string]any{"undeclared": gaps, "count": len(gaps)}); encErr != nil {
			fmt.Fprintf(stderr, "fak hooks lane-audit: %v\n", encErr)
			return 2
		}
	} else {
		fmt.Fprintf(stdout, "lane-audit: %d internal leaf(s) with a real Go package but NO declared dos.toml lane\n", len(gaps))
		for _, g := range gaps {
			fmt.Fprintf(stdout, "  - %s/%s — a `(fak %s)` stamp binds to a phantom unit; declare lane `%s` in dos.toml\n", g.Base, g.Leaf, g.Leaf, g.Leaf)
		}
		if len(gaps) == 0 {
			fmt.Fprintln(stdout, "  (every internal leaf has a declared lane)")
		}
	}
	if *gate >= 0 && len(gaps) > *gate {
		fmt.Fprintf(stderr, "lane-audit: %d undeclared leaves exceeds gate %d\n", len(gaps), *gate)
		return 1
	}
	return 0
}

// gateMode resolves a gate's FLEET_<NAME>_GUARD env to block (default) / warn / off, and its
// one-shot escape env. Identical semantics to the shell run_gate.
func gateMode(modeEnv, escapeEnv string) (mode string, escaped bool) {
	return gateModeDefault(modeEnv, escapeEnv, "block")
}

// gateModeDefault is gateMode with an explicit fallback for an UNSET ModeEnv. The historical
// default is "block"; an ADVISORY gate (Gate.DefaultMode = "warn", e.g. PRIOR_ART) passes
// "warn" so it only warns out of the box while its ModeEnv can still force "block". An empty
// def keeps the "block" default. Mirrors the commit-msg path's per-gate warn default for
// FLEET_MSG_GUARD.
func gateModeDefault(modeEnv, escapeEnv, def string) (mode string, escaped bool) {
	if def == "" {
		def = "block"
	}
	mode = strings.TrimSpace(os.Getenv(modeEnv))
	if mode == "" {
		mode = def
	}
	return mode, strings.TrimSpace(os.Getenv(escapeEnv)) == "1"
}

// errCheckBudgetExceeded marks a pre-commit gate that a per-gate wall-clock budget cut off
// before it returned. The pre-commit loop treats it exactly like any other could-not-run error
// — the gate is SKIPPED (fail-open) — but names it so a wedged or deliberately-slow gate is
// reported, never silently dropped.
var errCheckBudgetExceeded = errors.New("pre-commit gate exceeded its time budget")

// defaultCheckBudget bounds how long ONE pre-commit gate's Check may run before the hook treats
// it as could-not-run and skips it (fail-open). A gate over the staged diff finishes in well
// under a second; the cap exists only so a gate that HANGS on a lock or an O(refs) lease/fold —
// the #5335 wedge, observed blocking every commit in the clone at near-zero CPU — can never
// wedge the whole lane. Fail-open only ever SKIPS a slow gate; it can never add a block, so the
// safe direction is the only direction. Override with FAK_PRECOMMIT_CHECK_BUDGET_MS
// (milliseconds); a non-positive override is ignored, because disabling the bound reintroduces
// the wedge.
const defaultCheckBudget = 60 * time.Second

// resolveBudgetMS is the one override grammar both hook budgets share: the named variable is
// read as a millisecond count, and ONLY a value that parses to a positive count wins. Unset,
// unparseable, zero, and negative all fall back to the compiled-in default, because every one
// of those spellings would otherwise disable the bound and reintroduce the #5335 wedge.
func resolveBudgetMS(name string, fallback time.Duration) time.Duration {
	if raw := strings.TrimSpace(os.Getenv(name)); raw != "" {
		if ms, err := strconv.Atoi(raw); err == nil && ms > 0 {
			return time.Duration(ms) * time.Millisecond
		}
	}
	return fallback
}

// resolveCheckBudget returns the FAK_PRECOMMIT_CHECK_BUDGET_MS override (when it parses to a
// positive millisecond count) or defaultCheckBudget.
func resolveCheckBudget() time.Duration {
	return resolveBudgetMS("FAK_PRECOMMIT_CHECK_BUDGET_MS", defaultCheckBudget)
}

// defaultTotalBudget bounds the WHOLE pre-commit hook's wall clock, prologue included. The
// per-gate budget alone does not bound the hook: PreCommitGates() returns 17 gates, so a
// per-gate-only cap admits a ~17-minute worst case. That matters because #5335's failure is a
// FEEDBACK LOOP, not just a stall — a committer that outruns its ~120s tool timeout is SIGTERM'd
// mid-index-write and leaves a fresh stale `.git/index.lock`, which wedges the next committer.
// Bounding the hook under that timeout is what breaks the loop. 90s sits below the 120s cap and
// far above a healthy run (~0.3s for all gates), so it engages only when something is genuinely
// wedged. Override with FAK_PRECOMMIT_TOTAL_BUDGET_MS; a non-positive override is ignored,
// because disabling the bound reintroduces the wedge.
const defaultTotalBudget = 90 * time.Second

// resolveTotalBudget returns the FAK_PRECOMMIT_TOTAL_BUDGET_MS override (when it parses to a
// positive millisecond count) or defaultTotalBudget.
func resolveTotalBudget() time.Duration {
	return resolveBudgetMS("FAK_PRECOMMIT_TOTAL_BUDGET_MS", defaultTotalBudget)
}

// spentHookBudget reports whether a prologue step failed because the hook's shared wall clock
// ran out rather than for its own reasons, announcing the named step when it did. Both
// prologue steps need exactly this distinction, and they must not disagree about it: a step
// that ran out of budget means git is WEDGED, so falling through to the (unbounded) Python
// gates would rebuild the very wedge the budget exists to break — the caller returns
// exitBoundedFailOpen instead. A step that failed for any other reason is a plain
// could-not-run and does fall through.
func spentHookBudget(stderr io.Writer, asJSON bool, step string, deadline time.Time, budget time.Duration) bool {
	if !time.Now().After(deadline) {
		return false
	}
	if !asJSON {
		fmt.Fprintf(stderr, "pre-commit: %s exceeded the %s hook budget; gates skipped (fail-open, #5335)\n", step, budget)
	}
	return true
}

// gateBudgetFor returns the budget for the NEXT gate under the hook's shared wall clock: the
// per-gate budget clamped to whatever remains of the total. ok=false means the total is spent
// and the remaining gates must be skipped (fail-open) rather than each getting a fresh per-gate
// budget — the clamp is what stops N slow gates from summing past the total.
func gateBudgetFor(perGate, remaining time.Duration) (budget time.Duration, ok bool) {
	if remaining <= 0 {
		return 0, false
	}
	if remaining < perGate {
		return remaining, true
	}
	return perGate, true
}

// exitBoundedFailOpen is the exit code for "a wall-clock bound cut this hook short — allow the
// commit and do NOT fall through to the Python gates" (#5335).
//
// It is distinct from exit 2 on purpose. Exit 2 means could-not-run and tells the shell wrapper
// to run the Python checkers instead, which is right when `fak` is simply absent or the repo is
// unreadable. But when the bound fired, git itself is wedged — and every Python checker shells
// out to the same git, unbounded. Falling through would rebuild the exact wedge the bound just
// escaped, and would double the hook's wall clock while doing it. So this code says: the gates
// did not run, and re-running them by another route is not available. The issue's own DoD asks
// for exactly this ("exit 2 -> skip the Python fallthrough or the whole gate").
//
// Allowing the commit is the fail-open direction the hook already takes for every other
// could-not-run: a gate that cannot reach its evidence is skipped, never converted into a block.
const exitBoundedFailOpen = 3

// checkWithinBudget runs one gate's Check under a wall-clock budget. Returning within budget it
// yields the gate's own (findings, error) verbatim. On budget expiry it returns
// (nil, errCheckBudgetExceeded) and ABANDONS the still-running Check: the goroutine sends its
// result into a buffered channel and exits (at worst when the process does), so a hung gate can
// never block the commit. Gate.Check takes no context and so cannot be cancelled — abandonment
// is the only bound available, and it is safe here because a pre-commit gate is a read-only
// computation over the staged diff.
func checkWithinBudget(g hooks.Gate, d *hooks.StagedDiff, budget time.Duration) ([]hooks.Finding, error) {
	type outcome struct {
		findings []hooks.Finding
		err      error
	}
	done := make(chan outcome, 1) // buffered: an abandoned Check still sends and exits, never leaking
	go func() {
		f, e := g.Check(d)
		done <- outcome{f, e}
	}()
	timer := time.NewTimer(budget)
	defer timer.Stop()
	select {
	case o := <-done:
		return o.findings, o.err
	case <-timer.C:
		return nil, errCheckBudgetExceeded
	}
}

// preCommitGates is the gate set the pre-commit hook runs. It is indirected through a var —
// the same seam idiom as commitFn / execCommand elsewhere in this package — ONLY so a test can
// inject a gate with a known verdict (notably one that CANNOT run, which no real gate does on a
// healthy fixture repo). Production always reads hooks.PreCommitGates().
var preCommitGates = hooks.PreCommitGates

// couldNotRunClass names the CLASS of a gate's could-not-run error for the operator report, and
// deliberately never renders the error itself. Several gates scan for secret material
// (PUBLIC_LEAK, SECRET_SHAPE) over the staged diff, so an error value they return could carry a
// matched needle or a slice of that diff; printing it would leak through the very gate whose
// failure is being announced. Both return values are fixed literals, so a skip report can name a
// gate and a sentinel and nothing else.
func couldNotRunClass(err error) string {
	if errors.Is(err, hooks.ErrCouldNotRun) {
		return "ErrCouldNotRun"
	}
	return "unclassified error"
}

// enabledGateNames returns the names of the gates in gs that this run WOULD have checked — the
// ones an operator turned off (FLEET_<NAME>_GUARD=off) or escaped are deliberate intent, not a
// degraded run, so they never count as skipped. Used to name the tail of gates a spent hook
// budget drops (#5335) in the same skip ledger as a single could-not-run gate (#5299).
func enabledGateNames(gs []hooks.Gate) []string {
	var names []string
	for _, g := range gs {
		if mode, escaped := gateModeDefault(g.ModeEnv, g.EscapeEnv, g.DefaultMode); mode == "off" || escaped {
			continue
		}
		names = append(names, g.Name)
	}
	return names
}

func runHooksPreCommit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hooks pre-commit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	asJSON := fs.Bool("json", false, "emit findings as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	// The whole hook — prologue AND gates — runs inside one wall clock (#5335), so a wedged
	// git can never hold a commit past the tool timeout that manufactures the next stale lock.
	// The clock starts HERE, before the first git spawn: resolveRoot shells out too, so a
	// deadline taken after it would leave the hook's very first subprocess unbounded.
	totalBudget := resolveTotalBudget()
	hookDeadline := time.Now().Add(totalBudget)

	rootCtx, cancelRoot := context.WithDeadline(context.Background(), hookDeadline)
	r := resolveRootWithin(rootCtx, *root)
	cancelRoot()
	if r == "" {
		if spentHookBudget(stderr, *asJSON, "repo-root probe", hookDeadline, totalBudget) {
			return exitBoundedFailOpen
		}
		return 2 // not in a repo => could-not-run => fall through to python
	}

	// Bound the PROLOGUE. ReadStagedDiff spawns ~5 index-taking git reads and used to run with
	// no deadline at all, upstream of the per-gate budget below — which is exactly where #5335
	// observed the hook blocking at near-zero CPU while a peer held `.git/index.lock`.
	readCtx, cancelRead := context.WithDeadline(context.Background(), hookDeadline)
	d, err := hooks.ReadStagedDiffWithin(readCtx, r)
	cancelRead()
	if err != nil {
		// Git is wedged, not missing. Handing off to the (unbounded) Python gates would
		// rebuild the wedge, so stop here and let the commit through.
		if spentHookBudget(stderr, *asJSON, "staged-diff read", hookDeadline, totalBudget) {
			return exitBoundedFailOpen
		}
		// could-not-run: never block. The shell wrapper treats exit 2 as "fall back to python".
		return 2
	}

	budget := resolveCheckBudget()
	var allFindings []hooks.Finding
	// skipped names every enabled gate that produced NO verdict this run — one whose Check could
	// not reach its evidence (#5299) or one a wall-clock budget cut off (#5335). The fail-open
	// itself stands: a broken checker must never wedge every commit on a shared trunk. What
	// #5299 adds is that the skip is VISIBLE, so an operator can tell "every gate ran clean"
	// from "PUBLIC_LEAK never ran and the rest were clean" — two states this hook used to
	// report identically, letting a persistently-broken SECURITY gate go unnoticed.
	var skipped []string
	// reports is the per-gate CANDIDATE DENOMINATOR ledger (#5602): for each gate that was
	// ENABLED this run, how many staged items its own filter admitted. Gates turned off or
	// escaped are absent by the same rule that keeps them out of `skipped` — that is operator
	// intent, not a degraded run, and reporting a zero denominator for a gate nobody asked to
	// run would misdescribe it as having judged nothing.
	var reports []gateReport
	// scope is what this run quantified OVER (#5603): the staged set, plus the gates operator
	// intent left unable to refuse. `reports` says how much each gate judged; scope says which
	// population those numbers were drawn from and which gates are missing from them entirely.
	// Both narrowings are silent today — a run with half the gate set softened prints the same
	// "clean" as a full one, and a staged-clean commit reads like a clean tree.
	scope := runScope{Population: scopePopulationStaged}
	blocked := false
	gates := preCommitGates()
	for i, g := range gates {
		mode, escaped := gateModeDefault(g.ModeEnv, g.EscapeEnv, g.DefaultMode)
		if mode == "off" || escaped {
			// Operator intent, not a degraded run: still never counted as skipped (#5299 keeps
			// meaning "a checker broke"). But it IS a narrowing, so scope names it (#5603).
			why := "off"
			if escaped {
				why = "escaped"
			}
			scope.Narrowing.NotRun = append(scope.Narrowing.NotRun, g.Name+" ("+why+")")
			if escaped || gateModeIsEnvSet(g.ModeEnv) {
				scope.Narrowing.ByOperator = append(scope.Narrowing.ByOperator, g.Name)
			}
			continue
		}
		if mode != "block" {
			// Ran, judged real candidates, and cannot refuse. Distinguishing a gate that SHIPS
			// advisory from one an operator quietened this run is the whole point of ByEnv:
			// collapsing them would report a hollowed-out run as fak's designed posture.
			scope.Narrowing.Advisory = append(scope.Narrowing.Advisory, g.Name)
			if gateModeIsEnvSet(g.ModeEnv) {
				scope.Narrowing.ByOperator = append(scope.Narrowing.ByOperator, g.Name)
			}
		}
		// Charge every gate against the shared wall clock, so N slow gates cannot sum past the
		// total the way N per-gate budgets could.
		gateBudget, ok := gateBudgetFor(budget, time.Until(hookDeadline))
		if !ok {
			if !*asJSON {
				fmt.Fprintf(stderr, "pre-commit: %s hook budget spent; remaining gates from %s skipped (fail-open, #5335)\n", totalBudget, g.Name)
			}
			// The whole TAIL from here on is unchecked, this gate included — count all of it,
			// so the ledger never understates how degraded the run was.
			skipped = append(skipped, enabledGateNames(gates[i:])...)
			for _, name := range enabledGateNames(gates[i:]) {
				reports = append(reports, gateReport{Gate: name, Skipped: true})
			}
			break
		}
		started := time.Now()
		findings, gerr := checkWithinBudget(g, d, gateBudget)
		reports = append(reports, buildGateReport(d, g.Name, len(findings), gerr != nil, time.Since(started)))
		if gerr != nil {
			// A single gate that could-not-run — or ran past its wall-clock budget (#5335) — is
			// skipped (fail-open); the other gates still run. Either way the gate is NAMED and
			// counted, so a wedged, broken, or deliberately-slow gate is visible rather than a
			// silent bypass of a real check.
			skipped = append(skipped, g.Name)
			if !*asJSON {
				if errors.Is(gerr, errCheckBudgetExceeded) {
					fmt.Fprintf(stderr, "pre-commit: gate %s exceeded its %s budget; skipped (fail-open, #5335)\n", g.Name, gateBudget)
				} else {
					fmt.Fprintf(stderr, "pre-commit: gate %s could not run (%s); skipped (fail-open, #5299)\n", g.Name, couldNotRunClass(gerr))
				}
			}
			continue
		}
		if len(findings) == 0 {
			continue
		}
		// The gate mode is the disposition contract for this run: warn findings remain visible
		// but must trail every binding repair in the model-facing hand-off (#5972). Marking and
		// ordering live in internal/hooks so both the JSON payload and the printed block get the
		// same answer, and so the rule has a test that runs.
		allFindings = append(allFindings, hooks.MarkDisposition(findings, mode)...)
		if mode == "block" {
			blocked = true
			if !*asJSON {
				printGateFindings(stderr, g.Name, findings)
				printGateHint(stderr, g.Name, findings, true)
			}
		} else if !*asJSON { // warn
			printGateFindings(stderr, g.Name+" (advisory)", findings)
			printGateHint(stderr, g.Name, findings, false)
		}
	}

	buildcheckMode, buildcheckEscaped := gateModeDefault("FLEET_BUILDCHECK_GUARD", "ALLOW_COMMITTED_RED", "block")
	if buildcheckMode != "off" && !buildcheckEscaped {
		hasGoFiles := false
		for _, p := range d.StagedPaths {
			if strings.HasSuffix(p, ".go") {
				hasGoFiles = true
				break
			}
		}
		if hasGoFiles {
			outcome, detail := commitBuildCheckGate(stderr, r, d.StagedPaths)
			buildCheck, admitBuild, buildReason := safecommit.DecideBuildCheck(outcome, detail, os.Getenv("FAK_COMMIT_BUILD_CHECK") == "allow-timeout")
			if !admitBuild {
				fDetail := buildCheck.Detail
				if fDetail == "" {
					fDetail = buildReason
				}
				finding := hooks.Finding{
					Gate:     "COMMITTED_RED",
					Detail:   fDetail,
					Advisory: buildcheckMode != "block",
				}
				allFindings = append(allFindings, finding)
				if buildcheckMode == "block" {
					blocked = true
					if !*asJSON {
						printGateFindings(stderr, "COMMITTED_RED", []hooks.Finding{finding})
					}
				} else if !*asJSON {
					printGateFindings(stderr, "COMMITTED_RED (advisory)", []hooks.Finding{finding})
				}
			}
		}
	}

	// The count, once, at the end of the human report: the per-gate lines above scroll away in a
	// noisy commit, and "how much of the gate set actually ran" is the one number that separates
	// a clean commit from a degraded one.
	if len(skipped) > 0 && !*asJSON {
		fmt.Fprintf(stderr, "pre-commit: %d of %d gate(s) skipped (fail-open) — this commit ran a DEGRADED gate set: %s\n",
			len(skipped), len(gates), strings.Join(skipped, ", "))
	}

	// The clean path used to print NOTHING and return 0, and silence carries two meanings:
	// "every gate checked its candidates and found none" and "no gate had anything to check".
	// One line naming the domain separates them (#5602).
	if len(allFindings) == 0 && !*asJSON {
		fmt.Fprintln(stderr, cleanRunSummary(reports, len(d.StagedPaths), scope))
	}

	// One ordering for both consumers (#5972). The per-gate blocks above are printed in REGISTRY
	// order — parity order, not impact order — so the fix list an agent reads leads with an
	// advisory item the moment an early gate is softened or a late one hardened. The repair
	// summary restates the same findings binding-first with each disposition named; nothing is
	// re-judged and no gate's own block is withdrawn.
	orderFindingsForRepair(allFindings)
	if len(allFindings) > 0 && !*asJSON {
		fmt.Fprintln(stderr, hooks.RepairSummary(allFindings))
	}
	if *asJSON {
		emitFindingsJSON(stdout, stderr, allFindings, skipped, reports, scope)
	}
	if blocked {
		if !*asJSON {
			fmt.Fprintln(stderr, "")
			fmt.Fprintln(stderr, "commit refused by a fleet pre-commit gate (above).")
			fmt.Fprintln(stderr, "  soften one gate: FLEET_<NAME>_GUARD=warn (or =off), or use its one-shot escape.")
		}
		return 1
	}
	return 0
}

func runHooksCommitMsg(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hooks commit-msg", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	if !parseFlags(fs, argv) {
		return 2
	}
	rest := fs.Args()
	if len(rest) == 0 {
		fmt.Fprintln(stderr, "fak hooks commit-msg: message file required")
		return 2
	}
	msgBytes, err := os.ReadFile(rest[0])
	if err != nil {
		// could-not-read => fail open (exit 2), never wedge a commit over the message check.
		return 2
	}
	msg := string(msgBytes)
	r := resolveRoot(*root)

	// PUBLIC_LEAK over the message (block by default) — same needle list as pre-commit.
	// This is the fast PRE-block: a secret in the message is caught here without spawning
	// Python. A hit returns 1 (the shell short-circuits and blocks).
	leakMode, leakEscaped := gateMode("FLEET_SCRUB_GUARD", "FLEET_ALLOW_LEAK")
	if leakMode != "off" && !leakEscaped {
		if leaks := hooks.ScanMessageNeedles(msg, r); len(leaks) > 0 {
			for _, f := range leaks {
				fmt.Fprintf(stderr, "  %s\n", formatFinding(f))
			}
			if leakMode == "block" {
				fmt.Fprintln(stderr, "PUBLIC_LEAK: the commit MESSAGE carries a redact needle (above).")
				return 1
			}
			fmt.Fprintln(stderr, "PUBLIC_LEAK (advisory): redact needle in the commit message (above).")
		}
	}

	hwMode, hwEscaped := gateMode("FLEET_HW_GUARD", "FLEET_ALLOW_HW")
	if hwMode != "off" && !hwEscaped {
		if tells := hooks.ScanMessageHardwareTells(msg); len(tells) > 0 {
			fmt.Fprintf(stderr, "HARDWARE_TELL: the commit MESSAGE carries %d private hardware name(s):\n", len(tells))
			for _, f := range tells {
				fmt.Fprintf(stderr, "  %s\n", formatFinding(f))
			}
			fmt.Fprintln(stderr, "  fix: describe the box generically (GPU server / datacenter GPU), per PUBLIC-SCRUB-POLICY.md.")
			if hwMode == "block" {
				fmt.Fprintln(stderr, "  override once: FLEET_ALLOW_HW=1 <git cmd>  (a competitor citation / a commit about the scrubber).")
				return 1
			}
			fmt.Fprintln(stderr, "  advisory only because FLEET_HW_GUARD=warn.")
		}
	}

	freshMode, freshEscaped := gateMode("FLEET_FRESH_DELETE_GUARD", "FLEET_ALLOW_FRESH_DELETE")
	if freshMode != "off" && !freshEscaped && r != "" {
		if findings, ferr := hooks.FreshDeletionFindings(r, msg); ferr == nil && len(findings) > 0 {
			fmt.Fprintf(stderr, "FRESH_DELETION: %d fresh staged deletion(s) not named in the commit message:\n", len(findings))
			for _, f := range findings {
				fmt.Fprintf(stderr, "  %s\n", formatFinding(f))
			}
			if freshMode == "block" {
				fmt.Fprintln(stderr, "  (FLEET_FRESH_DELETE_GUARD=warn softens; =off disables; FLEET_ALLOW_FRESH_DELETE=1 overrides once)")
				return 1
			}
			fmt.Fprintln(stderr, "  advisory only because FLEET_FRESH_DELETE_GUARD=warn.")
		}
	}

	msgMode, msgEscaped := gateMode("FLEET_MSG_GUARD", "FLEET_ALLOW_MSG")
	if strings.TrimSpace(os.Getenv("FLEET_MSG_GUARD")) == "" {
		msgMode = "warn"
	}
	if msgMode == "off" || msgEscaped {
		return 0
	}
	if ok, why := hooks.CommitMsgVerdictWithGit(msg, r); !ok {
		fmt.Fprintf(stderr, "COMMIT_MSG: %s\n", why)
		if msgMode == "block" || strings.HasPrefix(why, "MERGE_WITNESS_FAIL") ||
			strings.HasPrefix(why, "MERGE_CONFLICT_") || strings.HasPrefix(why, "SILENT_DROP_MERGE_") {
			fmt.Fprintln(stderr, "  (FLEET_MSG_GUARD=warn softens; =off disables; FLEET_ALLOW_MSG=1 overrides once)")
			return 1
		}
		fmt.Fprintln(stderr, "  hard-enforce with FLEET_MSG_GUARD=block.")
	}
	return 0
}

func formatFinding(f hooks.Finding) string {
	loc := f.File
	if f.Line > 0 {
		if loc == "" {
			loc = fmt.Sprintf("msg:%d", f.Line)
		} else {
			loc = fmt.Sprintf("%s:%d", f.File, f.Line)
		}
	}
	if loc == "" {
		return f.Detail
	}
	return loc + ": " + f.Detail
}

// printGateFindings writes a gate's findings to w under "<label>: N finding(s):" —
// the human render the pre-commit and hygiene gate loops share. label carries the gate
// name plus any " (advisory)" suffix.
func printGateFindings(w io.Writer, label string, findings []hooks.Finding) {
	fmt.Fprintf(w, "%s: %d finding(s):\n", label, len(findings))
	for _, f := range findings {
		fmt.Fprintf(w, "  %s\n", formatFinding(f))
	}
}

// printGateHint writes a gate-specific recovery hint after a gate's findings. Today only
// HARDWARE_TELL carries one: a prose hardware tell in staged doc CONTENT is what reds the
// trunk later in `make ci` (#1455), and the scrubber can auto-fix it — so name the offending
// file(s) and the exact `tools/scrub_hardware_names.py --apply <file>` command (the same
// recovery the lint gives), plus the one-shot override when the gate actually blocked. Without
// this, the doc-content gate refused with only the generic "soften the gate" footer, leaving
// the author no pointer to the fix.
func printGateHint(w io.Writer, gate string, findings []hooks.Finding, blocked bool) {
	if gate != "HARDWARE_TELL" {
		return
	}
	files := distinctFindingFiles(findings)
	if len(files) == 0 {
		return
	}
	fmt.Fprintf(w, "  fix: tools/scrub_hardware_names.py --apply %s\n", strings.Join(files, " "))
	fmt.Fprintln(w, "       (auto-scrubs the prose tell before it reds the trunk in make ci; see PUBLIC-SCRUB-POLICY.md)")
	if blocked {
		fmt.Fprintln(w, "  override once: FLEET_ALLOW_HW=1 <git cmd>  (a competitor citation / a commit about the scrubber).")
	}
}

// distinctFindingFiles returns the unique non-empty File fields of findings, in first-seen
// order — the file set a recovery hint names.
func distinctFindingFiles(findings []hooks.Finding) []string {
	seen := map[string]bool{}
	var files []string
	for _, f := range findings {
		if f.File == "" || seen[f.File] {
			continue
		}
		seen[f.File] = true
		files = append(files, f.File)
	}
	return files
}

// gateReport is one enabled gate's line in the `gates` array (#5602): what it judged, how many
// items it judged, and how many findings that produced.
//
// Candidates is a POINTER so "the gate reports no denominator" serializes as JSON null and can
// never be mistaken for the number 0. Zero is a real answer here — "this gate ran and judged
// nothing" is exactly the state the issue exists to surface — so unlike Finding.Severity, whose
// zero means UNGRADED, this field cannot overload zero to mean absent.
type gateReport struct {
	Gate       string `json:"gate"`
	Candidates *int   `json:"candidates"`     // null = this gate reports no denominator
	Unit       string `json:"unit,omitempty"` // what Candidates counts, in that gate's own terms
	Findings   int    `json:"findings"`
	Skipped    bool   `json:"skipped,omitempty"` // ran no verdict this commit (#5299/#5335)
	ElapsedNS  int64  `json:"elapsed_ns"`
}

// buildGateReport folds one gate's outcome and the denominator that gate recorded for itself
// into a report line. A gate that recorded nothing leaves Candidates nil — UNREPORTED, which the
// caller must not render as 0.
func buildGateReport(d *hooks.StagedDiff, name string, findings int, skipped bool, elapsed ...time.Duration) gateReport {
	r := gateReport{Gate: name, Findings: findings, Skipped: skipped}
	if len(elapsed) > 0 {
		r.ElapsedNS = elapsed[0].Nanoseconds()
	}
	if n, unit, ok := d.Candidates(name); ok {
		r.Candidates = &n
		r.Unit = unit
	}
	return r
}

// cleanRunSummary is the one line the clean path prints so silence stops meaning two things.
// It splits the enabled gates into the ones that actually judged something, the ones that ran
// and admitted nothing, and the ones that report no denominator at all — three states the old
// empty payload rendered identically.
func cleanRunSummary(reports []gateReport, stagedFiles int, scope runScope) string {
	var judged, idle, unreported int
	var named []gateReport
	for _, r := range reports {
		switch {
		case r.Skipped:
			// Already named by the degraded-run line; not a clean-path claim.
		case r.Candidates == nil:
			unreported++
		case *r.Candidates == 0:
			idle++
		default:
			judged++
			named = append(named, r)
		}
	}
	sort.Slice(named, func(i, j int) bool {
		if *named[i].Candidates != *named[j].Candidates {
			return *named[i].Candidates > *named[j].Candidates
		}
		return named[i].Gate < named[j].Gate
	})
	const show = 3
	var parts []string
	for i, r := range named {
		if i == show {
			parts = append(parts, fmt.Sprintf("+%d more", len(named)-show))
			break
		}
		parts = append(parts, fmt.Sprintf("%s %d %s", r.Gate, *r.Candidates, r.Unit))
	}
	msg := fmt.Sprintf("pre-commit: clean — %d gate(s) over %d staged file(s): %d judged candidates",
		len(reports), stagedFiles, judged)
	if len(parts) > 0 {
		msg += " (" + strings.Join(parts, "; ") + ")"
	}
	msg += fmt.Sprintf(", %d judged nothing", idle)
	if unreported > 0 {
		// Say UNREPORTED out loud rather than folding it into the zero bucket: a gate that
		// reports no denominator has not told us it judged nothing, and printing it as if it
		// had would rebuild the exact ambiguity this line exists to remove.
		msg += fmt.Sprintf(", %d report no denominator", unreported)
	}
	// Every count above is a count OF something, and until now the line never said of what
	// (#5603). The scope clause closes that: which population the staged file count came from,
	// and which gates operator intent left out of the numbers entirely.
	return msg + " — " + scope.note()
}

// orderFindingsForRepair puts binding work before advisory work without disturbing the
// registry/file/line order inside either disposition. The rule itself lives in internal/hooks
// (repairorder.go, #5972) where it is testable; this stays as the call site's name.
func orderFindingsForRepair(findings []hooks.Finding) { hooks.OrderForRepair(findings) }

// emitFindingsJSON writes the --json report: the findings, and the ledger of gates that never
// delivered a verdict. skipped_gates / skipped_count are the machine-readable half of #5299 —
// with the fail-open deliberately kept, and stderr silenced under --json, they are the ONLY way
// a consumer can tell a run where every gate reached its evidence from one where a security
// gate errored out. Both keys are always present (empty list, zero count on a full run) so a
// reader never has to treat "absent" as "none". Only gate NAMES appear: a gate's error text can
// carry the secret material it was scanning for, and never reaches this payload.
//
// `gates` (#5602) is additive on the same principle: it is always present, and it carries each
// enabled gate's candidate DENOMINATOR so a clean payload states the domain it was clean over.
// `scope` / `scope_narrowing` (#5603) complete it — the denominators are meaningless to a
// consumer that cannot tell WHICH population they were drawn from, and `fak hygiene` emits
// overlapping gate names over a different one.
func emitFindingsJSON(stdout, stderr io.Writer, findings []hooks.Finding, skipped []string, gates []gateReport, scope runScope) {
	if findings == nil {
		findings = []hooks.Finding{}
	}
	if skipped == nil {
		skipped = []string{}
	}
	if gates == nil {
		gates = []gateReport{}
	}
	payload := map[string]any{
		"findings":      findings,
		"count":         len(findings),
		"skipped_gates": skipped,
		"skipped_count": len(skipped),
		// Additive only: every pre-existing key keeps its exact name and meaning, so a consumer
		// reading findings/count still parses this payload unchanged (#5602).
		"gates": gates,
		// scope is a bare string, not an object, so the cheapest possible check —
		// `.scope == "staged"` — is the one that works; the structured detail lives beside it
		// under its own key rather than forcing every consumer through a nested read (#5603).
		"scope":           scope.Population,
		"scope_narrowing": scope.narrowingPayload(),
	}
	if err := writeIndentedJSON(stdout, payload); err != nil {
		fmt.Fprintf(stderr, "fak hooks: %v\n", err)
	}
}

// resolveRoot returns the explicit --root, else the git toplevel from cwd, else "".
func resolveRoot(explicit string) string {
	return resolveRootWithin(context.Background(), explicit)
}

// resolveRootWithin is resolveRoot bounded by ctx. The `git rev-parse` it spawns is the FIRST
// subprocess the pre-commit hook runs, so leaving it unbounded would put a git call ahead of the
// hook's own wall clock (#5335). An expired ctx yields "" — the same not-in-a-repo answer a
// failed probe already gives — so the bound can only ever skip work, never add a refusal.
func resolveRootWithin(ctx context.Context, explicit string) string {
	if strings.TrimSpace(explicit) != "" {
		return explicit
	}
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}
