package architest

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// `ps` dialect gate — no procps-only keyword outside internal/procguard.
//
// THE DEFECT THIS EXISTS FOR. `ps` is two tools wearing one name. procps-ng (Linux) accepts
// output keywords the BSD dialect (macOS) has never had — `nlwp`, `etimes`, `cputimes` — and
// BSD `ps` rejects the WHOLE invocation on an unknown keyword instead of dropping that one
// column. A call site that hard-codes the procps spelling therefore does not degrade on a
// Mac, it returns nothing; and because the callers all read "no rows" as "quiet host", the
// failure is silent and reads as good news.
//
// #5385 fixed this once, properly, in internal/procguard: psCensusSpec / psRelationSpec pick
// the argv per GOOS, psNoColumn types a column the dialect simply does not have as nil
// rather than a fabricated 0, and runTool keeps stdout when the tool exits non-zero so a
// printed-then-rejected table is still a census. Then #5537 found the fix had not travelled:
// three call sites OUTSIDE that package still shipped the pre-#5385 argv verbatim
// (cmd/fak/slack_fleet_process_unix.go, and two probes in
// cmd/fak/dispatch_tick_preflight_host.go). Fixing them by hand fixes today; this gate is
// what stops the fourth copy.
//
// WHY A GATE AND NOT A TEST OF THE FIXED CODE. The behaviour that broke is only observable
// on a BSD-dialect host, and nothing in CI is one — the repo's own POSIX census tests SKIP
// for want of a `ps`. So the property that CAN be checked from any host is the structural
// one the #5385 author actually argued for: one enumeration implementation, not a fork per
// call site. If a keyword only one dialect understands appears in a `ps` argv anywhere but
// the package that branches on GOOS, that is a fork, and it is the exact shape of the bug.
//
// WHY THIS LIVES IN architest AND NOT A NEW LEAF. Same reasoning as sbom_drift_test.go and
// zerodep_claim_test.go: a repo-wide source contract, stdlib-only, off the request path,
// never registered into the kernel. No new package leaf means no new architest tier row and
// the push gate's UNTIERED_LEAF check is untouched.

// procpsOnlyKeywords are `ps` output keywords that procps-ng defines and the BSD dialect
// does not. Each entry is a keyword whose PRESENCE in an argv makes that argv Linux-only.
//
// The list is deliberately short and holds only keywords with no BSD spelling at all, or
// whose BSD near-spelling means something different:
//
//   - nlwp     — thread count. BSD `ps` has no thread-count keyword whatsoever, which is
//     why internal/procguard leaves Threads nil on darwin instead of substituting.
//   - etimes   — elapsed seconds. BSD has `etime`, FORMATTED as [[dd-]hh:]mm:ss, so the
//     rename is only half a fix without a parser for that grammar.
//   - cputimes — cumulative CPU seconds. BSD has `time`, likewise formatted.
//
// Keywords BOTH dialects accept (`pid`, `ppid`, `rss`, `comm`, `args`, `time`, `etime`) are
// NOT listed: a gate that argues with a working invocation gets switched off, and a
// switched-off gate is worth less than no gate.
var procpsOnlyKeywords = map[string]string{
	"nlwp":     "thread count; BSD ps has no thread-count keyword at all",
	"etimes":   "elapsed seconds; BSD spells it etime and formats it [[dd-]hh:]mm:ss",
	"cputimes": "cumulative CPU seconds; BSD spells it time and formats it [[dd-]hh:]mm:ss",
}

// psDialectExemptDirs are the repo-relative directories allowed to name a procps-only
// keyword. There is exactly one, and it is the package that branches on GOOS: its whole job
// is to hold both dialects' argv side by side, and internal/procguard/collect_posix_test.go
// pins them keyword for keyword.
var psDialectExemptDirs = []string{"internal/procguard"}

// psKeywordSite is one `ps` invocation that hands the tool a keyword only procps-ng knows.
type psKeywordSite struct {
	Path    string // repo-relative, slash-separated
	Line    int
	Keyword string
	Arg     string // the offending argument, verbatim
}

func (s psKeywordSite) String() string {
	return fmt.Sprintf("%s:%d passes %q to `ps` (in %q) — %s", s.Path, s.Line, s.Keyword, s.Arg, procpsOnlyKeywords[s.Keyword])
}

// psKeywordTokens cuts one argv element into the keyword tokens `ps` would read out of it.
// `-eo` style packs several keywords into one comma-separated argument with `=` suffixes
// ("pid=,nlwp=,rss=,comm="), `-axo` style uses bare commas ("pid,rss,comm"), and a
// single-column argv is one bare word ("nlwp="). Splitting on everything that is not a
// letter or digit covers all three, and comparing whole TOKENS (never substrings) is what
// keeps `etime` from reading as `etimes`.
func psKeywordTokens(arg string) []string {
	return strings.FieldsFunc(arg, func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= 'A' && r <= 'Z') && !(r >= '0' && r <= '9')
	})
}

// scanPSCallsInSource reports every call in one Go source file that names the program "ps"
// as a string-literal argument, and — for those calls only — every OTHER string literal in
// the same call that carries a procps-only keyword. calls is the number of `ps`-naming calls
// seen, which is the gate's fail-closed signal: a scanner that finds no `ps` call anywhere in
// this repo is broken, not vindicated.
//
// Scoping the keyword search to the same call expression is what makes this precise enough
// to live with. The literal "ps" appears all over this repo as a verb name, a map key and a
// help string; none of those are calls that hand it an argv, so none of them are read here.
//
// Residual, stated rather than papered over: an argv built AWAY from the call site — a
// package-level []string, or a struct field like procguard's own psSpec.args — is invisible
// to this walk. That form is exactly the shared-seam shape this gate wants people to use, so
// the hole is on the safe side; a fresh hard-coded fork is what it catches.
func scanPSCallsInSource(rel string, src []byte) (sites []psKeywordSite, calls int, err error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, rel, src, 0)
	if err != nil {
		return nil, 0, err
	}
	ast.Inspect(file, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		lits := psStringLiterals(call.Args)
		namesPS := false
		for _, lit := range lits {
			if lit.value == "ps" {
				namesPS = true
				break
			}
		}
		if !namesPS {
			return true
		}
		calls++
		for _, lit := range lits {
			if lit.value == "ps" {
				continue
			}
			for _, tok := range psKeywordTokens(lit.value) {
				if _, bad := procpsOnlyKeywords[tok]; !bad {
					continue
				}
				sites = append(sites, psKeywordSite{
					Path:    rel,
					Line:    fset.Position(lit.pos).Line,
					Keyword: tok,
					Arg:     lit.value,
				})
			}
		}
		return true
	})
	return sites, calls, nil
}

type psLiteral struct {
	value string
	pos   token.Pos
}

// psStringLiterals returns the unquoted string literals directly among a call's arguments,
// descending one level into a []string{...} composite literal so the
// `f("ps", []string{...}...)` spelling is read too.
func psStringLiterals(args []ast.Expr) []psLiteral {
	var out []psLiteral
	add := func(e ast.Expr) {
		lit, ok := e.(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			return
		}
		v, err := strconv.Unquote(lit.Value)
		if err != nil {
			return
		}
		out = append(out, psLiteral{value: v, pos: lit.Pos()})
	}
	for _, a := range args {
		add(a)
		if comp, ok := a.(*ast.CompositeLit); ok {
			for _, e := range comp.Elts {
				add(e)
			}
		}
	}
	return out
}

// psDialectCorpus returns every .go file the module actually builds, repo-relative and
// slash-separated, minus the exempt package.
//
// Two classes of directory are skipped and the reason differs:
//
//   - `_`- and `.`-prefixed directories are what the GO TOOLCHAIN ITSELF ignores, so nothing
//     under them is compiled by `go build ./...`. In this repo that is where the scratch and
//     quarantine trees live (`.gitignore` maps `/_*`), and those hold FROZEN SNAPSHOTS of
//     source — including pre-fix copies of the very files #5537 repairs. Reporting a
//     quarantined snapshot as a live defect would send the next reader to edit a record, and
//     editing a record to green a checker is the falsification this repo forbids.
//   - testdata/, vendor/, node_modules/ hold third-party or fixture source this repo does
//     not own and must not gate.
func psDialectCorpus(root string) ([]string, error) {
	skipDirs := map[string]bool{"node_modules": true, "testdata": true, "vendor": true}
	var files []string
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, rerr := filepath.Rel(root, p)
		if rerr != nil {
			return rerr
		}
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			if rel == "." {
				return nil
			}
			if skipDirs[d.Name()] || strings.HasPrefix(d.Name(), ".") || strings.HasPrefix(d.Name(), "_") {
				return filepath.SkipDir
			}
			for _, ex := range psDialectExemptDirs {
				if rel == ex {
					return filepath.SkipDir
				}
			}
			return nil
		}
		// Same toolchain rule one level down: `go build` also ignores FILES whose name
		// starts with `_` or `.`, which is what this repo's per-session scratch
		// (`.st_*.go`) is named. Those are peer working files, not shipped source.
		if strings.HasPrefix(d.Name(), "_") || strings.HasPrefix(d.Name(), ".") {
			return nil
		}
		if strings.HasSuffix(rel, ".go") {
			files = append(files, rel)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk %s: %w", root, err)
	}
	sort.Strings(files)
	return files, nil
}

// scanPSDialectCorpus walks the corpus and returns every offending site, the number of
// `ps`-naming calls it saw, and the number of files it actually parsed.
func scanPSDialectCorpus(t *testing.T, root string) (sites []psKeywordSite, calls, parsed int) {
	t.Helper()
	files, err := psDialectCorpus(root)
	if err != nil {
		t.Fatalf("build the ps-dialect corpus: %v", err)
	}
	for _, rel := range files {
		src, rerr := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if rerr != nil {
			t.Fatalf("read %s: %v", rel, rerr)
		}
		found, n, perr := scanPSCallsInSource(rel, src)
		if perr != nil {
			// A file this scanner cannot parse is not evidence of cleanliness. Report it
			// rather than counting it as scanned, so a corpus that stops parsing shows up in
			// the fail-closed floor below instead of quietly shrinking.
			t.Logf("ps-dialect gate: %s did not parse (%v) — not scanned", rel, perr)
			continue
		}
		parsed++
		calls += n
		sites = append(sites, found...)
	}
	return sites, calls, parsed
}

// psDialectRepoRoot is the module root: the parent of internal/.
func psDialectRepoRoot(t *testing.T) string {
	t.Helper()
	return filepath.Dir(internalDir(t))
}

// TestNoProcpsOnlyPSKeywordOutsideProcguard is the gate. Any `ps` invocation outside
// internal/procguard that hard-codes a procps-only keyword reds here, named by file:line.
//
// The cure is never to change the keyword in place — that just moves the fork. It is to call
// the collector that already branches on GOOS: procguard.CollectProcesses for a resource
// census, procguard.CollectRelations for pid/ppid/cmdline/age.
func TestNoProcpsOnlyPSKeywordOutsideProcguard(t *testing.T) {
	root := psDialectRepoRoot(t)
	sites, calls, parsed := scanPSDialectCorpus(t, root)

	// Fail closed. A green from a scanner that read nothing, or that found no `ps`
	// invocation at all in a repo that demonstrably has several, is not a green.
	if parsed < 400 {
		t.Fatalf("parsed only %d Go files — the corpus walk is broken, not the tree clean", parsed)
	}
	if calls < 3 {
		t.Fatalf("found %d `ps`-naming calls in %d files; this repo has several outside internal/procguard, so the scanner is broken", calls, parsed)
	}
	t.Logf("ps-dialect gate: parsed %d Go files, read %d `ps`-naming call(s), %d offending site(s)", parsed, calls, len(sites))

	if len(sites) > 0 {
		var b strings.Builder
		for _, s := range sites {
			fmt.Fprintf(&b, "\n  %s", s)
		}
		t.Fatalf("%d `ps` invocation(s) outside internal/procguard hard-code a procps-only keyword (#5537):%s"+
			"\n\ncure: call internal/procguard (CollectProcesses / CollectRelations), which picks the argv per GOOS."+
			"\nBSD `ps` rejects the whole invocation on an unknown keyword, so these return NOTHING on macOS —"+
			"\nand every caller reads nothing as a quiet host.", len(sites), b.String())
	}
}

// TestPSDialectGateCatchesMutations is the witness that the gate can FAIL. Every case is
// real source: the three pre-#5537 call sites byte for byte in the "must fire" direction,
// and the invocations this repo legitimately ships in the "must not" direction. A gate that
// over-fires on a working argv is a gate someone deletes.
func TestPSDialectGateCatchesMutations(t *testing.T) {
	const head = "package p\n\nimport (\n\t\"context\"\n\t\"os/exec\"\n)\n\nvar _ = context.Background\n\nfunc f(ctx context.Context) {\n"
	const tail = "\n}\n"

	cases := []struct {
		name string
		body string
		want []string // the keywords that must be reported, in order
	}{
		// --- must fire: the three sites #5537 found, exactly as they read before the fix ---
		{
			"slack fleet-status background census",
			"\t_, _ = exec.Command(\"ps\", \"-eo\", \"pid=,ppid=,etimes=,args=\").Output()",
			[]string{"etimes"},
		},
		{
			"dispatch preflight process census",
			"\t_ = exec.CommandContext(ctx, \"ps\", \"-eo\", \"pid=,nlwp=,rss=,comm=\")",
			[]string{"nlwp"},
		},
		{
			"dispatch preflight host thread total",
			"\t_ = exec.CommandContext(ctx, \"ps\", \"-eo\", \"nlwp=\")",
			[]string{"nlwp"},
		},
		{
			"the procguard census argv, forked back out of the package",
			"\t_ = exec.CommandContext(ctx, \"ps\", \"-eo\", \"pid=,nlwp=,rss=,cputimes=,comm=\")",
			[]string{"nlwp", "cputimes"},
		},
		{
			"argv passed as a slice literal rather than variadic strings",
			"\t_ = exec.Command(\"ps\", []string{\"-eo\", \"pid=,nlwp=\"}...)",
			[]string{"nlwp"},
		},

		// --- must NOT fire: invocations this repo ships and that both dialects accept -----
		{
			"commitlane / dispatch worker rows — common vocabulary only",
			"\t_ = exec.CommandContext(ctx, \"ps\", \"-eo\", \"pid=,ppid=,comm=,args=\")",
			nil,
		},
		{
			"memgate big holders — BSD-style -axo, common keywords",
			"\t_ = exec.Command(\"ps\", \"-axo\", \"pid,rss,comm\")",
			nil,
		},
		{
			"the darwin relations argv — etime is BSD's, and must not read as etimes",
			"\t_ = exec.CommandContext(ctx, \"ps\", \"-eo\", \"pid=,ppid=,etime=,comm=,args=\")",
			nil,
		},
		{
			"the darwin census argv — time is BSD's, and must not read as cputimes",
			"\t_ = exec.CommandContext(ctx, \"ps\", \"-eo\", \"pid=,rss=,time=,comm=\")",
			nil,
		},
		{
			"a procps keyword in a call that does not name ps is not this gate's business",
			"\t_ = exec.CommandContext(ctx, \"top\", \"-eo\", \"nlwp=\")",
			nil,
		},
		{
			"the bare verb name, no argv",
			"\t_ = exec.Command(\"ps\")",
			nil,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sites, calls, err := scanPSCallsInSource("fixture.go", []byte(head+tc.body+tail))
			if err != nil {
				t.Fatalf("fixture did not parse: %v", err)
			}
			if calls != 1 && tc.name != "a procps keyword in a call that does not name ps is not this gate's business" {
				t.Fatalf("calls = %d, want exactly 1 `ps`-naming call in the fixture", calls)
			}
			var got []string
			for _, s := range sites {
				got = append(got, s.Keyword)
			}
			if strings.Join(got, ",") != strings.Join(tc.want, ",") {
				t.Fatalf("keywords = %v, want %v (fixture: %s)", got, tc.want, strings.TrimSpace(tc.body))
			}
		})
	}
}

// TestPSKeywordTokensAreWholeTokens pins the one parsing decision the gate turns on: BSD's
// `etime` and `time` must never be read as procps' `etimes` and `cputimes`. A substring
// match would fire on the CORRECT darwin argv and make the fix look like the bug.
func TestPSKeywordTokensAreWholeTokens(t *testing.T) {
	for _, tc := range []struct {
		arg  string
		want []string
	}{
		{"pid=,nlwp=,rss=,cputimes=,comm=", []string{"pid", "nlwp", "rss", "cputimes", "comm"}},
		{"pid=,rss=,time=,comm=", []string{"pid", "rss", "time", "comm"}},
		{"pid=,ppid=,etime=,comm=,args=", []string{"pid", "ppid", "etime", "comm", "args"}},
		{"pid,rss,comm", []string{"pid", "rss", "comm"}},
		{"nlwp=", []string{"nlwp"}},
		// A flag yields a token too ("eo"); harmless, because the gate compares tokens
		// against a closed keyword set rather than treating every token as a keyword.
		{"-eo", []string{"eo"}},
		{"-axo", []string{"axo"}},
	} {
		got := psKeywordTokens(tc.arg)
		if strings.Join(got, "|") != strings.Join(tc.want, "|") {
			t.Fatalf("psKeywordTokens(%q) = %v, want %v", tc.arg, got, tc.want)
		}
		for _, tok := range got {
			if _, bad := procpsOnlyKeywords[tok]; bad && (tok == "etime" || tok == "time") {
				t.Fatalf("%q read as procps-only; it is the BSD spelling", tok)
			}
		}
	}
}

// TestPSDialectCorpusExcludesOnlyProcguard pins the exemption in both directions: the one
// package allowed to hold both dialects must be out of the corpus, and the packages that
// actually call `ps` must be in it. An exemption that quietly widened is how a gate stops
// gating.
func TestPSDialectCorpusExcludesOnlyProcguard(t *testing.T) {
	root := psDialectRepoRoot(t)
	files, err := psDialectCorpus(root)
	if err != nil {
		t.Fatalf("build the ps-dialect corpus: %v", err)
	}
	present := make(map[string]bool, len(files))
	for _, rel := range files {
		present[rel] = true
		if strings.HasPrefix(rel, "internal/procguard/") {
			t.Fatalf("internal/procguard must be exempt, but %s is in the corpus", rel)
		}
		// Frozen snapshots under the toolchain-ignored trees are records, not live source.
		// One of them (_scratch/quarantine/...) is a verbatim PRE-#5537 copy of
		// cmd/fak/dispatch_tick_preflight_host.go, so this exclusion is load-bearing: without
		// it the gate reds forever and points the reader at a file they must not edit.
		for _, part := range strings.Split(rel, "/") {
			if strings.HasPrefix(part, "_") || strings.HasPrefix(part, ".") {
				t.Fatalf("%s is under a toolchain-ignored directory and must not be in the corpus", rel)
			}
		}
	}
	for _, must := range []string{
		"cmd/fak/dispatch_tick_preflight_host.go",
		"cmd/fak/slack_fleet_process_unix.go",
		"internal/commitlane/status.go",
		"internal/memgate/memgate.go",
	} {
		if !present[must] {
			t.Fatalf("%s must be in the corpus — it calls `ps` and is not the exempt package", must)
		}
	}
}
