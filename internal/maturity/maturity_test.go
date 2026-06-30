package maturity

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestAdjudicateLadder pins the monotonic ladder: each rung is reached only when
// every lower rung's evidence also holds. `benchmarked` is a badge, not a rung —
// a capability at the top of the ladder but unmeasured still has a "benchmark it"
// next step; only a measured default surface is fully matured (no next work).
func TestAdjudicateLadder(t *testing.T) {
	cases := []struct {
		name     string
		cap      Capability
		wantRung Rung
		wantNext Rung // the gap the next-work names; -1 means expect no next work
		wantSkip bool
	}{
		{"no code is proposed", Capability{Lane: "x", Dir: "internal/x"}, RungProposed, RungPrototyped, false},
		{"code only is prototyped", Capability{Lane: "x", HasCode: true}, RungPrototyped, RungTested, false},
		{"code+tests is tested", Capability{Lane: "x", HasCode: true, HasTests: true}, RungTested, RungDogfooded, false},
		{"through dogfooded", Capability{Lane: "x", HasCode: true, HasTests: true, Dogfooded: true}, RungDogfooded, RungDefault, false},
		{"default-but-unmeasured wants a benchmark", Capability{Lane: "x", HasCode: true, HasTests: true, Dogfooded: true, DefaultSurface: true}, RungDefault, rungBenchmark, false},
		{"measured default is fully matured", Capability{Lane: "x", HasCode: true, HasTests: true, Dogfooded: true, DefaultSurface: true, Benchmarked: true}, RungDefault, -1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := adjudicate(tc.cap)
			if got.Rung != tc.wantRung {
				t.Fatalf("rung = %v, want %v", got.Rung, tc.wantRung)
			}
			if got.Skip != tc.wantSkip {
				t.Fatalf("skip = %v, want %v", got.Skip, tc.wantSkip)
			}
			if tc.wantNext == -1 {
				if got.Next != nil {
					t.Fatalf("expected no next work, got %+v", got.Next)
				}
				return
			}
			if got.Next == nil {
				t.Fatalf("expected a next-work item")
			}
			if got.Next.Gap != tc.wantNext {
				t.Fatalf("next gap = %v, want %v", got.Next.Gap, tc.wantNext)
			}
		})
	}
}

// TestLadderSkipIsOverclaim is the heart of the subsystem: a capability fak runs
// (dogfooded) but never tested looks more mature than its evidence — a ladder-skip.
// Its current rung is capped at prototyped, but it carries higher evidence, and its
// next work item is the SKIPPED lower rung (tests), flagged as a skip.
func TestLadderSkipIsOverclaim(t *testing.T) {
	got := adjudicate(Capability{Lane: "skipper", Dir: "internal/skipper",
		HasCode: true, HasTests: false, Dogfooded: true, Benchmarked: true})
	if got.Rung != RungPrototyped {
		t.Fatalf("rung = %v, want prototyped (the unmet `tested` rung caps it)", got.Rung)
	}
	if got.TopEvidence != RungDogfooded {
		t.Fatalf("top evidence = %v, want dogfooded", got.TopEvidence)
	}
	if !got.Skip {
		t.Fatalf("expected Skip=true (fak relies on it but it has no tests)")
	}
	if got.Next == nil || got.Next.Gap != RungTested || !got.Next.Skip {
		t.Fatalf("next work should be the skipped `tested` rung flagged as a skip, got %+v", got.Next)
	}
	if !strings.Contains(got.Next.Title, "LADDER-SKIP") {
		t.Fatalf("skip title should name the inversion, got %q", got.Next.Title)
	}
}

// TestBuildFold checks the roll-up: distribution, debt == ladder-skips, OK gate,
// and that the backlog is ordered skips-first then least-mature-first.
func TestBuildFold(t *testing.T) {
	corpus := []Capability{
		{Lane: "alpha", HasCode: true, HasTests: true, Dogfooded: true, Benchmarked: true, DefaultSurface: true}, // default
		{Lane: "bravo", HasCode: true, HasTests: true},                                                           // tested
		{Lane: "charlie", HasCode: true},                                                                         // prototyped
		{Lane: "delta", HasCode: true, Dogfooded: true},                                                          // SKIP: dogfooded but untested -> capped at prototyped
	}
	p := Build(Options{Root: "/synthetic", facts: func(string) []Capability { return corpus }})

	if p.OK {
		t.Fatalf("expected OK=false: there is a ladder-skip")
	}
	if got := p.Corpus["maturity_debt"].(int); got != 1 {
		t.Fatalf("maturity_debt = %d, want 1 (one ladder-skip)", got)
	}
	dist := p.Corpus["distribution"].(map[string]int)
	if dist["default"] != 1 || dist["tested"] != 1 || dist["prototyped"] != 2 {
		t.Fatalf("distribution = %+v, want default:1 tested:1 prototyped:2", dist)
	}
	if len(p.Backlog) != 3 { // alpha is at the top, no next work
		t.Fatalf("backlog len = %d, want 3", len(p.Backlog))
	}
	// The skip (delta) must rank first.
	if p.Backlog[0].Lane != "delta" || !p.Backlog[0].Skip {
		t.Fatalf("backlog[0] = %+v, want the delta skip first", p.Backlog[0])
	}
}

// TestAllMaturePassesGate proves the green path: no skips -> OK, debt 0.
func TestAllMaturePassesGate(t *testing.T) {
	corpus := []Capability{
		{Lane: "a", HasCode: true, HasTests: true, Dogfooded: true},
		{Lane: "b", HasCode: true, HasTests: true},
	}
	p := Build(Options{Root: "/x", facts: func(string) []Capability { return corpus }})
	if !p.OK {
		t.Fatalf("expected OK=true with no skips, got reason %q", p.Reason)
	}
	if p.Corpus["maturity_debt"].(int) != 0 {
		t.Fatalf("expected zero debt")
	}
}

// TestGatherFactsFromTree exercises the impure shell against a synthetic tree: it
// must read only internal/<leaf> lanes from dos.toml, see code/tests in the leaf,
// detect a cmd import as dogfooding, and a Benchmark func as benchmarked.
func TestGatherFactsFromTree(t *testing.T) {
	root := t.TempDir()
	// dos.toml with a [lanes.trees] block: two internal leaves + one area lane.
	dosToml := `[lanes]
concurrent = ["alpha"]

[lanes.trees]
alpha = ["internal/alpha/**"]
bravo = ["internal/bravo/**"]
docs  = ["docs/**"]
`
	writeFile(t, root, "dos.toml", dosToml)
	// alpha: code + test + Benchmark; imported by cmd.
	writeFile(t, root, "internal/alpha/alpha.go", "package alpha\n\nfunc A() {}\n")
	writeFile(t, root, "internal/alpha/alpha_test.go", "package alpha\n\nimport \"testing\"\n\nfunc BenchmarkA(b *testing.B) {}\n")
	// bravo: code only.
	writeFile(t, root, "internal/bravo/bravo.go", "package bravo\n\nfunc B() {}\n")
	// a cmd that imports alpha (dogfoods it) and names alpha in cli-reference.
	writeFile(t, root, "cmd/fak/main.go", "package main\n\nimport _ \"github.com/anthony-chaudhary/fak/internal/alpha\"\n\nfunc main() {}\n")
	writeFile(t, root, "docs/cli-reference.md", "# verbs\n\nThe alpha verb does things.\n")

	caps := gatherFacts(root)
	if len(caps) != 2 {
		t.Fatalf("expected 2 internal leaf capabilities (area lane skipped), got %d: %+v", len(caps), caps)
	}
	byLane := map[string]Capability{}
	for _, c := range caps {
		byLane[c.Lane] = c
	}
	alpha, ok := byLane["alpha"]
	if !ok {
		t.Fatalf("alpha missing")
	}
	if !alpha.HasCode || !alpha.HasTests || !alpha.Benchmarked || !alpha.Dogfooded || !alpha.DefaultSurface {
		t.Fatalf("alpha facts wrong: %+v", alpha)
	}
	bravo := byLane["bravo"]
	if !bravo.HasCode || bravo.HasTests || bravo.Dogfooded {
		t.Fatalf("bravo facts wrong: %+v", bravo)
	}
}

func TestIssueItemsRenderRoutableStableMaturityIssues(t *testing.T) {
	p := Build(Options{Root: "/synthetic", facts: func(string) []Capability {
		return []Capability{
			{Lane: "alpha", Dir: "internal/alpha", HasCode: true},
			{Lane: "bravo", Dir: "internal/bravo", HasCode: true, HasTests: true, Dogfooded: true, DefaultSurface: true},
		}
	}})
	items := IssueItems(p, 1, []string{"maturity"})
	if len(items) != 1 {
		t.Fatalf("items len = %d, want 1", len(items))
	}
	it := items[0]
	if it.Key != "maturity/alpha/tested" {
		t.Fatalf("key = %q, want maturity/alpha/tested", it.Key)
	}
	if it.Title != "maturity(alpha): add tests for the capability" {
		t.Fatalf("title = %q", it.Title)
	}
	for _, want := range []string{
		"<!-- fak-maturity-work-key: maturity/alpha/tested -->",
		"Lane: `alpha`",
		"Gap: `tested`",
		"fak maturity route",
	} {
		if !strings.Contains(it.Body, want) {
			t.Fatalf("issue body missing %q:\n%s", want, it.Body)
		}
	}
	if got := MarkerKey(it.Body); got != it.Key {
		t.Fatalf("MarkerKey = %q, want %q", got, it.Key)
	}
}

func TestProjectIssueItemsSkipsPrivateBoundaryLanesBeforeLimit(t *testing.T) {
	p := Build(Options{Root: "/synthetic", facts: func(string) []Capability {
		return []Capability{
			{Lane: "dgxbridge", Dir: "internal/dgxbridge"},
			{Lane: "alpha", Dir: "internal/alpha", HasCode: true},
		}
	}})
	projection := ProjectIssueItems(p, 1, nil)
	if len(projection.Skipped) != 1 {
		t.Fatalf("skipped len = %d, want 1", len(projection.Skipped))
	}
	if projection.Skipped[0].Lane != "dgxbridge" || projection.Skipped[0].Key != "maturity/dgxbridge/prototyped" {
		t.Fatalf("skipped row = %+v", projection.Skipped[0])
	}
	if len(projection.Items) != 1 {
		t.Fatalf("items len = %d, want 1", len(projection.Items))
	}
	if projection.Items[0].Lane != "alpha" || projection.Items[0].Key != "maturity/alpha/tested" {
		t.Fatalf("routed item = %+v", projection.Items[0])
	}
}

func TestBuildIssuePlanUpdatesExistingMaturityIssue(t *testing.T) {
	item := IssueItem{
		Key:     "maturity/alpha/tested",
		Lane:    "alpha",
		Title:   "maturity(alpha): add tests for the capability",
		Body:    "<!-- fak-maturity-work-key: maturity/alpha/tested -->\nbody",
		Gap:     "tested",
		Witness: "a *_test.go in internal/alpha",
	}
	plan := BuildIssuePlan([]IssueItem{item}, []ExistingIssue{{
		Number: 42,
		State:  "OPEN",
		Body:   "<!-- fak-maturity-work-key: maturity/alpha/tested -->",
	}})
	if len(plan) != 1 {
		t.Fatalf("plan len = %d, want 1", len(plan))
	}
	row := plan[0]
	if row.Action != "update" || row.Number == nil || *row.Number != 42 {
		t.Fatalf("row = %+v, want update #42", row)
	}
}

func TestSyncIssuePlanUsesInjectedRunner(t *testing.T) {
	plan := []IssuePlanRow{
		{Action: "create", Key: "maturity/a/tested", Lane: "a", Title: "maturity(a): add tests", Body: "body"},
		{Action: "update", Key: "maturity/b/default", Lane: "b", Title: "maturity(b): default", Body: "body", Number: intPtr(7)},
	}
	var calls [][]string
	rows := SyncIssuePlan(plan, "owner/repo", []string{"maturity"}, func(args []string) (string, string, bool) {
		calls = append(calls, args)
		return "ok", "", true
	})
	if len(rows) != 2 || !rows[0].OK || !rows[1].OK {
		t.Fatalf("sync rows = %+v, want two OK rows", rows)
	}
	joined := strings.Join([]string{strings.Join(calls[0], " "), strings.Join(calls[1], " ")}, "\n")
	for _, want := range []string{
		"issue create",
		"--label maturity",
		"--repo owner/repo",
		"issue edit 7",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("sync argv missing %q:\n%s", want, joined)
		}
	}
}

func intPtr(n int) *int { return &n }

func writeFile(t *testing.T, root, rel, content string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
