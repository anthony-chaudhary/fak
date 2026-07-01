package claimcheck

// witnessfor.go mechanizes AGENTS.md's "pick the witness the bug actually has" table
// (#2153). Proof-by-default demands every fix ship a captured artifact matched to its
// failure class, but DESIGNING that witness each time is a tax that pushes agents
// toward "looks fixed" self-reports — exactly what the kernel exists to stop. Given a
// claim in plain words, WitnessFor classifies it into the table's closed failure
// classes and emits the cheapest command (or test skeleton) that would actually grade
// it, so the cost of proof drops toward the cost of running one line.
//
// HONEST BOUNDARY: this is a SCAFFOLD, not a verdict. It lowers the cost of authoring
// a witness; it never claims the witness passed. Placeholders in <angle brackets> are
// deliberately unfilled — the tool cannot know your SHA, package, or artifact, and
// filling them with guesses would manufacture fake evidence. The classification is a
// deterministic keyword scan with a fixed precedence, so the same claim always maps to
// the same plan (and the fixture corpus below is its own re-derivable witness, the
// same Q5 discipline `--self-test` applies to the grader).

import (
	"sort"
	"strings"
)

// The closed witness-class vocabulary (#2153) — the failure classes of the AGENTS.md
// proof-by-default table.
const (
	// WitnessShipped: a "shipped / done / landed" claim. The witness is a witnessed
	// commit — the diff and the registry, never the subject line.
	WitnessShipped = "shipped"
	// WitnessVisual: a rendering / TUI / terminal-corruption claim. The witness is a
	// captured render — a test that captures the bytes a surface emits.
	WitnessVisual = "visual"
	// WitnessPerf: a speed / throughput / efficiency claim. The witness is the
	// net-true-value grade over a real baseline with labeled provenance.
	WitnessPerf = "perf"
	// WitnessLogic: the default — a behavior/bug claim. The witness is a repro test
	// that fails before the fix and passes after, landed in the same commit.
	WitnessLogic = "logic"
)

// WitnessPlan is the scaffold WitnessFor emits: the class the claim fell into, the
// cheapest runnable proving command (placeholders in <angle brackets>), an optional
// test skeleton for the classes whose proof is a new test, the rationale naming the
// keywords that decided, and the repo reference the plan mechanizes.
type WitnessPlan struct {
	Claim     string `json:"claim"`
	Class     string `json:"class"`
	Command   string `json:"command"`
	Skeleton  string `json:"skeleton,omitempty"`
	Rationale string `json:"rationale"`
	Reference string `json:"reference"`
}

// witnessKeywords maps each non-default class to the lowercase tokens that vote for
// it. Precedence on multi-class hits is fixed and documented: shipped > visual > perf
// (a "shipped the faster render" claim is graded as a shipped claim first — the
// commit witness is the cheapest and the other witnesses still apply underneath);
// logic is the default when nothing votes.
var witnessKeywords = map[string][]string{
	WitnessShipped: {"shipped", "landed", "committed", "pushed", "merged", "released", "closed the issue", "done", "delivered"},
	WitnessVisual:  {"render", "tui", "pane", "screenshot", "layout", "flicker", "corrupt", "overlap", "ansi", "cursor", "garbage bytes", "screen", "redraw", "visual"},
	WitnessPerf:    {"faster", "slower", "speedup", "throughput", "tok/s", "latency", "p99", "p50", "ms ", " ms", "cache hit", "cheaper", "tokens", "×", "x faster", "perf", "efficien", "% of", "utilization"},
}

// witnessPrecedence is the deterministic tie-break order for multi-class hits.
var witnessPrecedence = []string{WitnessShipped, WitnessVisual, WitnessPerf}

// WitnessFor classifies claim into the proof-by-default failure classes and returns
// the cheapest witness scaffold. Pure and deterministic: a keyword scan with fixed
// precedence, no I/O — the same claim always yields the same plan.
func WitnessFor(claim string) WitnessPlan {
	low := strings.ToLower(claim)
	matched := map[string][]string{}
	for class, words := range witnessKeywords {
		for _, w := range words {
			if strings.Contains(low, w) {
				matched[class] = append(matched[class], strings.TrimSpace(w))
			}
		}
	}
	for _, hits := range matched {
		sort.Strings(hits)
	}

	class := WitnessLogic
	rationale := "no shipped/visual/perf keyword matched — a behavior claim's proof is the repro test (the default class)"
	for _, c := range witnessPrecedence {
		if len(matched[c]) > 0 {
			class = c
			rationale = "matched " + strings.Join(matched[c], ", ")
			if len(matched) > 1 {
				rationale += " (precedence: shipped > visual > perf > logic)"
			}
			break
		}
	}

	plan := WitnessPlan{Claim: claim, Class: class, Rationale: rationale}
	switch class {
	case WitnessShipped:
		plan.Command = `dos verify <plan> <leaf>   # or: dos commit-audit <sha> — the diff and the registry, never the subject line`
		plan.Reference = `AGENTS.md "Proof by default" — 'Shipped / done' claims: a witnessed commit ((fak <leaf>) trailer + dos verify)`
	case WitnessVisual:
		plan.Command = `go test ./<package> -run <YourRenderWitnessTest> -count=1   # after writing the skeleton below`
		plan.Skeleton = visualWitnessSkeleton
		plan.Reference = `AGENTS.md "Proof by default" — TUI/visual: a captured render; pattern: cmd/fak/watchdog_autoheal_test.go TestWatchdogAutohealKeepsAgentPaneClean`
	case WitnessPerf:
		plan.Command = `fak claim-check --statement "` + claim + `" --baseline real --baseline-desc "<the tuned alternative>" --net --scope "<holds under / vanishes under>" --provenance WITNESSED --witness "<artifact + reproduce command>" --json`
		plan.Reference = `docs/standards/net-true-value.md — a perf claim is graded against the real baseline with labeled provenance (Q1-Q6)`
	default:
		plan.Command = `go test ./<package> -run <YourReproTest> -count=1   # must FAIL before the fix, PASS after; land it in the same commit`
		plan.Skeleton = logicWitnessSkeleton
		plan.Reference = `AGENTS.md "Proof by default" — logic/behavior: a fail-before/pass-after repro test is the proof`
	}
	return plan
}

// visualWitnessSkeleton is the captured-render test shape the AGENTS.md table points
// at: capture the bytes the surface emits, assert the defect is gone. A green test
// that never renders the surface is not proof for a visual bug.
const visualWitnessSkeleton = `func Test<Surface>RenderWitness(t *testing.T) {
	// 1. Drive the surface exactly as the reporter did (same size, same input).
	var buf bytes.Buffer
	render<Surface>(&buf /* , the repro inputs */)

	// 2. The captured bytes ARE the artifact. Assert the defect is gone —
	//    e.g. zero stray bytes on the pane, no ANSI soup, the row present once.
	if got := buf.String(); strings.Contains(got, "<the defect signature>") {
		t.Fatalf("defect still renders:\n%s", got)
	}
}`

// logicWitnessSkeleton is the fail-before/pass-after repro shape: reproduce the defect
// as an assertion FIRST, watch it fail, then make it pass with the fix.
const logicWitnessSkeleton = `func Test<Defect>Repro(t *testing.T) {
	// 1. Arrange the exact conditions of the report.
	// 2. Assert the CORRECT behavior — this test must FAIL on the pre-fix tree.
	got := <the call the report names>
	if got != <the correct value> {
		t.Fatalf("got %v, want %v", got, <the correct value>)
	}
}`

// WitnessFixtureCase is one graded sample of the witness-scaffold corpus: a claim of a
// known class, the class WitnessFor derived, and whether the emitted plan is runnable
// (a non-empty command, plus a skeleton for the classes whose proof is a new test).
type WitnessFixtureCase struct {
	Name   string `json:"name"`
	Claim  string `json:"claim"`
	Expect string `json:"expect"`
	Got    string `json:"got"`
	OK     bool   `json:"ok"`
}

// RunWitnessFixture grades the built-in one-claim-per-class corpus (#2153's witness:
// for a sample claim of each class the tool emits a scaffold that actually grades that
// claim). Returns the cases and how many landed on their labeled class WITH a runnable
// plan — the same self-witness shape as RunFixture.
func RunWitnessFixture() (cases []WitnessFixtureCase, passed int) {
	fixture := []struct{ name, claim, expect string }{
		{"shipped-claim", "shipped the leaseref liveness classifier and pushed to main", WitnessShipped},
		{"visual-claim", "the agent pane renders garbage bytes after a resize", WitnessVisual},
		{"perf-claim", "the vcache warm path is 3.2x faster with a higher cache hit rate", WitnessPerf},
		{"logic-claim", "guard admission returns the wrong verdict when the tree is empty", WitnessLogic},
		{"precedence-shipped-over-perf", "shipped the 2x faster warm path", WitnessShipped},
	}
	for _, f := range fixture {
		plan := WitnessFor(f.claim)
		ok := plan.Class == f.expect && plan.Command != "" && plan.Reference != ""
		if plan.Class == WitnessVisual || plan.Class == WitnessLogic {
			ok = ok && plan.Skeleton != ""
		}
		if ok {
			passed++
		}
		cases = append(cases, WitnessFixtureCase{Name: f.name, Claim: f.claim, Expect: f.expect, Got: plan.Class, OK: ok})
	}
	return cases, passed
}
