package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// cmd/fak/hygiene.go — `fak hygiene`: run the repo's whole-tree (`--audit-tree`) hygiene gates IN
// ONE PROCESS instead of spawning a Python interpreter per checker. `make hygiene` (part of the
// mandated pre-push `make ci`, mirrored in CI) historically spawned 11 Python interpreters, each
// doing sub-millisecond regex/path work over `git ls-files`; on Windows that is ~15-20s of pure
// process-create + Defender-scan tax before any checking happens. This reads `git ls-files` ONCE
// and runs every ported gate over it — the same collapse `fak hooks` made for the pre-commit hook.
//
// Exit codes mirror the gate contract so the Makefile / CI wrapper can fall back to Python:
// 0 = clean, 1 = REFUSED (a hygiene gate fired, OR --gates named no registered gate), 2 =
// could-not-run (the wrapper then runs the Python path).
//
// A bad `--gates` value is deliberately NOT exit 2 (#5604). Exit 2 is precisely the code that
// sends `make hygiene` to the Python sweep, which would then run, pass, and bury the typo — so
// routing a usage error through it would rebuild the silent green this refusal exists to remove.
// It is a refusal, so it shares the hard-fail code with a gate that fired.
//
// `--gates A,B,...` runs only the named gates (so `make index-sync` can call this for INDEX_SYNC
// while `make hygiene` runs the rest); the default is every gate HygieneGates() returns. Every
// entry must name a registered gate: an unknown name is REFUSED rather than silently selecting
// nothing, mirroring `fak test`'s `unknown check %q` (cmd/fak/test.go). All make-hygiene checkers
// now have native Go tree gates in internal/hooks/ (#928, #10940).
//
// A gate whose Check errors stays fail-open — one broken checker must never wedge the tree —
// but the skip is COUNTED and NAMED (#5299's pre-commit treatment, ported here by #5604), so
// "every gate ran clean" is distinguishable from "SECRET_SHAPE never ran and the rest were
// clean".

func cmdHygiene(argv []string) { os.Exit(runHygiene(os.Stdout, os.Stderr, argv)) }

func runHygiene(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("hygiene", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	asJSON := fs.Bool("json", false, "emit findings as JSON")
	gatesCSV := fs.String("gates", "", "comma-separated gate names to run (default: all)")
	if !parseFlags(fs, argv) {
		return 2
	}

	r := resolveRoot(*root)
	if r == "" {
		return 2 // not in a repo => could-not-run => fall through to python
	}

	d, err := hooks.ReadTrackedTree(r)
	if err != nil {
		// could-not-run: never wedge the gate. The wrapper treats exit 2 as "fall back to python".
		return 2
	}

	want, ferr := gateFilter(*gatesCSV)
	if ferr != nil {
		// A selector that names no real gate would otherwise run ZERO checks and exit 0 — a
		// report byte-identical to a genuine clean sweep. Refuse instead (#5604).
		fmt.Fprintf(stderr, "fak hygiene: %v\n", ferr)
		return 1
	}
	var allFindings []hooks.Finding
	var skipped []string
	// scope is what this sweep quantified OVER (#5603). It is the whole-tree twin of the staged
	// hook's scope, and the reason the two must both carry it: several gate names appear in BOTH
	// registries, so a reader handed one report in isolation cannot tell which population a
	// "clean" verdict covered. Advisory stays empty here on purpose — hygiene has no per-gate
	// warn mode; its advisory-ness is per FINDING (PushScoped / Finding.Advisory), which is a
	// property of the verdict rather than a narrowing of the gate set, and claiming otherwise
	// would put a state in this field that the operator never chose.
	scope := runScope{Population: scopePopulationTree}
	selected := 0
	blocked := false
	for _, g := range hooks.HygieneGates() {
		if want != nil && !want[g.Name] {
			// --gates is operator intent: real narrowing, never a degradation. Naming it keeps a
			// one-gate sweep from reading like the full one (#5603) — the same confusion #5604
			// refused for a selector that matched NOTHING, one step short of the empty case.
			scope.Narrowing.NotRun = append(scope.Narrowing.NotRun, g.Name+" (not selected)")
			scope.Narrowing.ByOperator = append(scope.Narrowing.ByOperator, g.Name)
			continue
		}
		// A DefaultOff gate (a migration-in-flight ratchet like BARE_DEV_SPELLING) runs only
		// when named explicitly via --gates, never in the default `make ci` sweep. That is a
		// compiled default, NOT this operator's choice, so it is named as not-run but never
		// counted against the operator.
		if g.DefaultOff && want == nil {
			scope.Narrowing.NotRun = append(scope.Narrowing.NotRun, g.Name+" (default-off)")
			continue
		}
		selected++
		findings, gerr := g.Check(d)
		if gerr != nil {
			// A single gate that could-not-run is skipped (fail-open); the others still run.
			// Fail-open stays — but the gate is NAMED and COUNTED, so a persistently-broken
			// checker is a visible degradation rather than a silent bypass (#5299 → #5604).
			skipped = append(skipped, g.Name)
			if !*asJSON {
				fmt.Fprintf(stderr, "hygiene: gate %s could not run (%s); skipped (fail-open, #5604)\n",
					g.Name, couldNotRunClass(gerr))
			}
			continue
		}
		if len(findings) == 0 {
			continue
		}
		if g.PushScoped {
			findings = scopeHygieneFindingsToPush(r, findings)
		}
		allFindings = append(allFindings, findings...)
		for _, finding := range findings {
			if !finding.Advisory {
				blocked = true
			}
		}
		if !*asJSON {
			var hard, advisory []hooks.Finding
			for _, finding := range findings {
				if finding.Advisory {
					advisory = append(advisory, finding)
				} else {
					hard = append(hard, finding)
				}
			}
			if len(hard) > 0 {
				printGateFindings(stderr, g.Name, hard)
			}
			if len(advisory) > 0 {
				printGateFindings(stderr, g.Name+" (advisory; outside this push)", advisory)
			}
		}
	}

	// Advisory core-lock fold (issue #1682): classify what this checkout has changed against the
	// shipped soft-contract taxonomy and surface coherence-bearing warnings with the witness that
	// would clear each. WARNING MODE — never blocks: it leaves `blocked` and the exit code alone.
	coreLockWarns := auditCoreLockPaths(changedTreePaths(r))

	// The count, once, at the end of the human report: the per-gate lines above scroll away in a
	// noisy run, and "how much of the selected gate set actually ran" is the one number that
	// separates a clean tree from a degraded sweep.
	if len(skipped) > 0 && !*asJSON {
		fmt.Fprintf(stderr, "hygiene: %d of %d gate(s) skipped (fail-open) — this run checked a DEGRADED gate set: %s\n",
			len(skipped), selected, strings.Join(skipped, ", "))
	}

	if *asJSON {
		emitHygieneJSON(stdout, stderr, allFindings, coreLockWarns, skipped, scope)
	} else {
		renderCoreLockWarnings(stderr, coreLockWarns)
	}
	if blocked {
		if !*asJSON {
			fmt.Fprintln(stderr, "")
			fmt.Fprintln(stderr, "hygiene refused by a tree gate (above) — the --audit-tree backstop make ci runs HARD.")
		}
		return 1
	}
	if !*asJSON {
		// Name the denominator on the clean path too: bare "hygiene OK" reads the same whether
		// nine gates ran or one did. The scope clause says what the denominator is a denominator
		// OF (#5603) — without it this line and the staged hook's clean line are the same shape
		// over two different populations.
		fmt.Fprintf(stdout, "hygiene OK — %d gate(s) over %d tracked file(s) — %s\n",
			selected-len(skipped), len(d.Paths), scope.note())
	}
	return 0
}

// gateFilter parses --gates into a set, or nil to mean "all". Names are upper-cased and trimmed so
// `--gates index_sync` and `--gates INDEX_SYNC` both resolve.
//
// Every entry must name a gate HygieneGates() actually registers. An unknown name — a typo, a
// lower-cased hyphenation, the singular of a plural — used to build a non-empty want set that
// matched nothing, ran zero checks and exited 0 (#5604). It is now an error, and so is a value
// that parses to an empty selection (`--gates ,`): an empty gate set is never a silent pass.
func gateFilter(csv string) (map[string]bool, error) {
	csv = strings.TrimSpace(csv)
	if csv == "" {
		return nil, nil
	}
	known := map[string]bool{}
	var valid []string
	for _, g := range hooks.HygieneGates() {
		known[g.Name] = true
		valid = append(valid, g.Name)
	}
	sort.Strings(valid)

	want := map[string]bool{}
	var unknown []string
	for _, n := range strings.Split(csv, ",") {
		n = strings.ToUpper(strings.TrimSpace(n))
		if n == "" {
			continue
		}
		if !known[n] {
			unknown = append(unknown, strconv.Quote(n))
			continue
		}
		want[n] = true
	}
	if len(unknown) > 0 {
		return nil, fmt.Errorf("unknown gate %s (valid: %s)", strings.Join(unknown, ", "), strings.Join(valid, ", "))
	}
	if len(want) == 0 {
		return nil, fmt.Errorf("--gates %q selected no gate (valid: %s)", csv, strings.Join(valid, ", "))
	}
	return want, nil
}
