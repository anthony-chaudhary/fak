package maturity

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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
	p := Build(Options{Root: "/synthetic", facts: func(string) []Capability { return corpus }, Witnesses: func(string) (map[string]RuntimeProof, error) {
		return map[string]RuntimeProof{"alpha": {Lane: "alpha", Command: "ok", OutputContains: "ok", DefaultOn: true, DefaultReason: "fixture default"}, "delta": {Lane: "delta", Command: "ok", OutputContains: "ok"}}, nil
	}})

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
	p := Build(Options{Root: "/x", facts: func(string) []Capability { return corpus }, Witnesses: func(string) (map[string]RuntimeProof, error) {
		return map[string]RuntimeProof{"a": {Lane: "a", Command: "ok"}}, nil
	}})
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
	if !alpha.HasCode || !alpha.HasTests || !alpha.Benchmarked || !alpha.Integrated || alpha.Dogfooded || alpha.DefaultSurface {
		t.Fatalf("alpha facts wrong: %+v", alpha)
	}
	bravo := byLane["bravo"]
	if !bravo.HasCode || bravo.HasTests || bravo.Integrated || bravo.Dogfooded {
		t.Fatalf("bravo facts wrong: %+v", bravo)
	}
}

func TestIssueItemsRenderRoutableStableMaturityIssues(t *testing.T) {
	p := Build(Options{Root: "/synthetic", facts: func(string) []Capability {
		return []Capability{
			{Lane: "alpha", Dir: "internal/alpha", HasCode: true},
			{Lane: "bravo", Dir: "internal/bravo", HasCode: true, HasTests: true},
		}
	}, Witnesses: func(string) (map[string]RuntimeProof, error) { return map[string]RuntimeProof{}, nil }})
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
		"dispatchability: `triage_only`",
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

func TestReconcileIssuesClosesDuplicateAndObsoleteOpenIssues(t *testing.T) {
	item := IssueItem{
		Key: "gateway:tested", Lane: "gateway", Title: "maturity(gateway): add tests",
		Body: "<!-- fak-maturity-work-key: gateway:tested -->\n",
	}
	existing := []ExistingIssue{
		{Number: 7, State: "OPEN", Title: item.Title, Body: item.Body},
		{Number: 8, State: "OPEN", Title: item.Title, Body: item.Body},
		{Number: 9, State: "OPEN", Title: "old gap", Body: "<!-- fak-maturity-work-key: gateway:prototyped -->\n"},
		{Number: 10, State: "CLOSED", Title: "old closed gap", Body: "<!-- fak-maturity-work-key: gateway:dogfooded -->\n"},
	}
	plan := ReconcileIssues([]IssueItem{item}, existing)
	var actions []string
	for _, row := range plan {
		n := 0
		if row.Number != nil {
			n = *row.Number
		}
		actions = append(actions, fmt.Sprintf("%s:%d", row.Action, n))
	}
	got := strings.Join(actions, ",")
	if got != "keep:8,close-duplicate:7,close-obsolete:9" {
		t.Fatalf("reconcile plan = %q", got)
	}
}

func TestBuildIssuePlanLimitedDoesNotCloseObsoleteKeys(t *testing.T) {
	item := IssueItem{Key: "gateway:tested", Lane: "gateway", Title: "next", Body: "<!-- fak-maturity-work-key: gateway:tested -->\n"}
	existing := []ExistingIssue{{Number: 9, State: "OPEN", Body: "<!-- fak-maturity-work-key: other:tested -->\n"}}
	plan := BuildIssuePlan([]IssueItem{item}, existing)
	if len(plan) != 1 || plan[0].Action != "create" {
		t.Fatalf("limited plan must not close unseen keys: %#v", plan)
	}
}

func TestSyncIssuePlanClosesManagedIssue(t *testing.T) {
	n := 9
	plan := []IssuePlanRow{{Action: "close-obsolete", Key: "old:tested", Number: &n}}
	var calls [][]string
	rows := SyncIssuePlan(plan, "owner/repo", nil, func(args []string) (string, string, bool) {
		calls = append(calls, append([]string(nil), args...))
		return "closed", "", true
	})
	if len(rows) != 1 || !rows[0].OK || len(calls) != 1 {
		t.Fatalf("unexpected close sync: rows=%#v calls=%#v", rows, calls)
	}
	if got := strings.Join(calls[0], " "); got != "issue close 9 --repo owner/repo" {
		t.Fatalf("close call = %q", got)
	}
}

func TestBuildIssuePlanKeepsUnchangedOpenSuccessor(t *testing.T) {
	item := IssueItem{
		Key: "gateway:tested", Lane: "gateway", Title: "maturity(gateway): add tests",
		Body: "<!-- fak-maturity-work-key: gateway:tested -->\n",
	}
	plan := BuildIssuePlan([]IssueItem{item}, []ExistingIssue{{
		Number: 7, State: "OPEN", Title: item.Title, Body: item.Body,
	}})
	if len(plan) != 1 || plan[0].Action != "keep" {
		t.Fatalf("unchanged open successor must be a no-op, got %#v", plan)
	}
	calls := 0
	rows := SyncIssuePlan(plan, "owner/repo", nil, func(args []string) (string, string, bool) {
		calls++
		return "", "", true
	})
	if calls != 0 || len(rows) != 1 || !rows[0].OK {
		t.Fatalf("keep must not call gh: calls=%d rows=%#v", calls, rows)
	}
}

func TestBuildIssuePlanReopensClosedMaturitySuccessor(t *testing.T) {
	item := IssueItem{
		Key: "gateway:tested", Lane: "gateway", Title: "maturity(gateway): add tests",
		Body: "<!-- fak-maturity-work-key: gateway:tested -->\n",
	}
	plan := BuildIssuePlan([]IssueItem{item}, []ExistingIssue{{
		Number: 7, State: "CLOSED", Body: item.Body,
	}})
	if len(plan) != 1 || plan[0].Action != "reopen" || plan[0].Number == nil || *plan[0].Number != 7 {
		t.Fatalf("closed successor must be reopened, got %#v", plan)
	}
}

func TestSyncIssuePlanReopensBeforeRefreshingClosedSuccessor(t *testing.T) {
	n := 7
	plan := []IssuePlanRow{{Action: "reopen", Key: "gateway:tested", Number: &n, Title: "next", Body: "body"}}
	var calls [][]string
	rows := SyncIssuePlan(plan, "owner/repo", nil, func(args []string) (string, string, bool) {
		calls = append(calls, append([]string(nil), args...))
		return "ok", "", true
	})
	if len(rows) != 1 || !rows[0].OK || len(calls) != 2 {
		t.Fatalf("unexpected reopen sync: rows=%#v calls=%#v", rows, calls)
	}
	if got := strings.Join(calls[0], " "); got != "issue reopen 7 --repo owner/repo" {
		t.Fatalf("first call = %q", got)
	}
	if got := strings.Join(calls[1], " "); !strings.HasPrefix(got, "issue edit 7 --title next --body body") {
		t.Fatalf("second call = %q", got)
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
		"--label needs-triage",
		"--label triage-only",
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

func TestBuildImportIsIntegratedButNotDogfoodedWithoutRuntimeWitness(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "dos.toml", "[lanes.trees]\nalpha = [\"internal/alpha/**\"]\n")
	writeFile(t, root, "internal/alpha/alpha.go", "package alpha\n")
	writeFile(t, root, "internal/alpha/alpha_test.go", "package alpha\n")
	writeFile(t, root, "cmd/fak/main.go", "package main\nimport _ \"github.com/anthony-chaudhary/fak/internal/alpha\"\nfunc main() {}\n")
	got := Build(Options{Root: root, Witnesses: func(string) (map[string]RuntimeProof, error) { return map[string]RuntimeProof{}, nil }})
	alpha := got.Caps[0]
	if !alpha.Integrated || alpha.Dogfooded || alpha.Rung != RungTested || alpha.Next == nil || alpha.Next.Gap != RungDogfooded {
		t.Fatalf("alpha=%+v", alpha)
	}
}

func TestBuildRuntimeWitnessPromotesDogfooding(t *testing.T) {
	facts := func(string) []Capability {
		return []Capability{{Lane: "alpha", Dir: "internal/alpha", HasCode: true, HasTests: true, Integrated: true}}
	}
	got := Build(Options{facts: facts, Witnesses: func(string) (map[string]RuntimeProof, error) {
		return map[string]RuntimeProof{"alpha": {Lane: "alpha", Command: "fak alpha --selfcheck"}}, nil
	}})
	alpha := got.Caps[0]
	if !alpha.Dogfooded || alpha.Rung != RungDogfooded || alpha.RuntimeProof != "fak alpha --selfcheck" {
		t.Fatalf("alpha=%+v", alpha)
	}
}

func TestLoadRuntimeProofsRejectsDuplicateAndVerifyRejectsFailure(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "internal", "maturity")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	data := `{"schema":"fak-maturity-runtime-proofs/2","witnesses":[{"lane":"x","command":"fak version","output_contains":"x version"},{"lane":"x","command":"fak env","output_contains":"x env"}]}`
	if err := os.WriteFile(filepath.Join(dir, "runtime-proofs.json"), []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadRuntimeProofs(root); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("err=%v", err)
	}
	data = `{"schema":"fak-maturity-runtime-proofs/2","witnesses":[{"lane":"x","command":"git definitely-not-a-command","output_contains":"x never"}]}`
	if err := os.WriteFile(filepath.Join(dir, "runtime-proofs.json"), []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := VerifyRuntimeProofs(root); err == nil || !strings.Contains(err.Error(), "x failed") {
		t.Fatalf("err=%v", err)
	}
}

func TestRealRuntimeWitnessRegistryEveryRowMeetsContract(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	data, err := os.ReadFile(filepath.Join(root, "internal", "maturity", "runtime-proofs.json"))
	if err != nil {
		t.Fatal(err)
	}
	var registry runtimeProofFile
	if err := json.Unmarshal(data, &registry); err != nil {
		t.Fatal(err)
	}
	if registry.Schema != runtimeProofSchema {
		t.Fatalf("schema = %q, want %q", registry.Schema, runtimeProofSchema)
	}
	if len(registry.Witnesses) == 0 {
		t.Fatal("runtime proof registry must retain at least one live witness")
	}

	seen := make(map[string]int, len(registry.Witnesses))
	for i, proof := range registry.Witnesses {
		if previous, ok := seen[proof.Lane]; ok {
			t.Errorf("row %d duplicates lane %q from row %d", i, proof.Lane, previous)
		} else {
			seen[proof.Lane] = i
		}
		t.Run(fmt.Sprintf("%03d_%s", i, proof.Lane), func(t *testing.T) {
			fixture := t.TempDir()
			writeRuntimeProofFixture(t, fixture, []RuntimeProof{proof})
			if _, err := loadRuntimeProofs(fixture); err != nil {
				t.Errorf("registry row %d (%q) violates the runtime proof contract: %v", i, proof.Lane, err)
			}
		})
	}
}

func TestRealRuntimeWitnessRegistryPasses(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	headOutput, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").CombinedOutput()
	if err != nil {
		t.Fatalf("resolve fixture HEAD: %v: %s", err, headOutput)
	}
	proofs, err := loadRuntimeProofs(root)
	if err != nil {
		t.Fatalf("load runtime proofs: %v", err)
	}
	var outputs []string
	for _, p := range proofs {
		outputs = append(outputs, p.OutputContains)
	}
	artifact := buildRuntimeFakFixture(t, strings.TrimSpace(string(headOutput)), strings.Join(outputs, "\n"))
	oldResolver := resolveRuntimeFak
	resolveRuntimeFak = func() (string, error) { return artifact, nil }
	t.Cleanup(func() { resolveRuntimeFak = oldResolver })
	if err := VerifyRuntimeProofs(root); err != nil {
		t.Fatal(err)
	}
}

func TestLoadRuntimeProofsRejectsTestAsDogfooding(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "internal", "maturity")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	data := `{"schema":"fak-maturity-runtime-proofs/2","witnesses":[{"lane":"x","command":"go test ./internal/x","output_contains":"x ok"}]}`
	if err := os.WriteFile(filepath.Join(dir, "runtime-proofs.json"), []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadRuntimeProofs(root); err == nil || !strings.Contains(err.Error(), "test evidence") {
		t.Fatalf("err=%v", err)
	}
}

func TestDocumentedDefaultWithoutRuntimeProofIsLadderSkip(t *testing.T) {
	got := adjudicate(Capability{Lane: "x", Dir: "internal/x", HasCode: true, HasTests: true, Integrated: true, DefaultSurface: true})
	if got.Rung != RungTested || got.TopEvidence != RungDefault || !got.Skip || got.Next == nil || got.Next.Gap != RungDogfooded {
		t.Fatalf("got=%+v", got)
	}
}

func configureRuntimeAlias(t *testing.T, root, lane string) {
	t.Helper()
	for _, args := range [][]string{{"init", root}, {"-C", root, "config", "alias." + lane, "!echo " + lane + " capability ran"}} {
		if output, err := exec.Command("git", args...).CombinedOutput(); err != nil {
			t.Fatalf("configure git runtime alias: %v: %s", err, output)
		}
	}
}

func writeRuntimeProofFixture(t *testing.T, root string, proofs []RuntimeProof) {
	t.Helper()
	path := filepath.Join(root, "internal", "maturity")
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(runtimeProofFile{Schema: runtimeProofSchema, Witnesses: proofs})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "runtime-proofs.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestBuildFailsHealthOnceWhenRuntimeProofsCannotVerify(t *testing.T) {
	caps := []Capability{{Lane: "alpha", Dir: "internal/alpha", HasCode: true, HasTests: true}}
	p := Build(Options{
		Root:  "/synthetic",
		facts: func(string) []Capability { return caps },
		Witnesses: func(string) (map[string]RuntimeProof, error) {
			return nil, fmt.Errorf("stale fak artifact")
		},
	})
	if p.OK || p.Verdict != "FAIL" || p.Finding != "runtime_proof_unverified" {
		t.Fatalf("proof failure did not fail scorecard health: %+v", p)
	}
	if p.RuntimeProofOK || p.RuntimeProofCount != 0 || p.RuntimeProofError != "stale fak artifact" {
		t.Fatalf("proof health = ok:%t count:%d err:%q", p.RuntimeProofOK, p.RuntimeProofCount, p.RuntimeProofError)
	}
	if !strings.Contains(p.NextAction, "repair or replace") {
		t.Fatalf("next action does not prioritize proof repair: %q", p.NextAction)
	}
	for _, capability := range p.Caps {
		if capability.Dogfooded || capability.RuntimeProof != "" {
			t.Fatalf("failed proof promoted capability: %+v", capability)
		}
	}
}

func TestBuildKeepsEmptyValidRuntimeRegistryHealthy(t *testing.T) {
	p := Build(Options{
		Root:  "/synthetic",
		facts: func(string) []Capability { return []Capability{{Lane: "alpha", HasCode: true, HasTests: true}} },
		Witnesses: func(string) (map[string]RuntimeProof, error) {
			return map[string]RuntimeProof{}, nil
		},
	})
	if !p.OK || !p.RuntimeProofOK || p.RuntimeProofCount != 0 || p.RuntimeProofError != "" {
		t.Fatalf("empty valid registry should remain healthy: %+v", p)
	}
}

func TestRuntimeProofRejectsDirtyMatchingArtifact(t *testing.T) {
	root := initRuntimeProofGitRepo(t)
	headOutput, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").CombinedOutput()
	if err != nil {
		t.Fatalf("resolve fixture HEAD: %v: %s", err, headOutput)
	}
	writeRuntimeProofFixture(t, root, []RuntimeProof{{
		Lane: "alpha", Command: "fak", OutputContains: "alpha capability ran",
	}})
	artifact := buildRuntimeFakFixture(t, strings.TrimSpace(string(headOutput))+" dirty", "alpha capability ran")
	oldResolver := resolveRuntimeFak
	resolveRuntimeFak = func() (string, error) { return artifact, nil }
	t.Cleanup(func() { resolveRuntimeFak = oldResolver })

	err = VerifyRuntimeProofs(root)
	if err == nil || !strings.Contains(err.Error(), "is dirty") {
		t.Fatalf("VerifyRuntimeProofs accepted dirty fak artifact: %v", err)
	}
}

func TestRuntimeProofRejectsStalePathArtifact(t *testing.T) {
	root := initRuntimeProofGitRepo(t)
	writeRuntimeProofFixture(t, root, []RuntimeProof{{
		Lane: "alpha", Command: "fak", OutputContains: "alpha capability ran",
	}})
	artifact := buildRuntimeFakFixture(t, "deadbeefdead", "alpha capability ran")
	oldResolver := resolveRuntimeFak
	resolveRuntimeFak = func() (string, error) { return artifact, nil }
	t.Cleanup(func() { resolveRuntimeFak = oldResolver })

	err := VerifyRuntimeProofs(root)
	if err == nil || !strings.Contains(err.Error(), "does not match scored source") {
		t.Fatalf("VerifyRuntimeProofs accepted stale fak artifact: %v", err)
	}
}

func buildRuntimeFakFixture(t *testing.T, revision, output string) string {
	t.Helper()
	dir := t.TempDir()
	source := fmt.Sprintf(`package main
import (
    "fmt"
    "os"
)
func main() {
    if len(os.Args) == 2 && os.Args[1] == "version" {
        fmt.Println("0.43.0")
        fmt.Println("build: %s")
        fmt.Println("go: fixture")
        return
    }
    fmt.Println(%q)
}
`, revision, output)
	sourcePath := filepath.Join(dir, "main.go")
	if err := os.WriteFile(sourcePath, []byte(source), 0o644); err != nil {
		t.Fatal(err)
	}
	artifact := filepath.Join(dir, "fak")
	if runtime.GOOS == "windows" {
		artifact += ".exe"
	}
	if buildOutput, err := exec.Command("go", "build", "-o", artifact, sourcePath).CombinedOutput(); err != nil {
		t.Fatalf("build runtime fak fixture: %v: %s", err, buildOutput)
	}
	return artifact
}

func initRuntimeProofGitRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	commands := [][]string{
		{"init", root},
		{"-C", root, "config", "user.email", "fixture@example.invalid"},
		{"-C", root, "config", "user.name", "fixture"},
		{"-C", root, "commit", "--allow-empty", "-m", "fixture"},
	}
	for _, args := range commands {
		if output, err := exec.Command("git", args...).CombinedOutput(); err != nil {
			t.Fatalf("initialize runtime proof repo: git %v: %v: %s", args, err, output)
		}
	}
	return root
}

func TestRuntimeProofRequiresMatchingCapabilityOutput(t *testing.T) {
	for _, tc := range []struct {
		name   string
		proof  RuntimeProof
		wantOK bool
	}{
		{name: "missing assertion", proof: RuntimeProof{Lane: "alpha", Command: "git --version"}},
		{name: "assertion omits lane", proof: RuntimeProof{Lane: "alpha", Command: "git --version", OutputContains: "git version"}},
		{name: "unrelated successful command", proof: RuntimeProof{Lane: "alpha", Command: "git --version", OutputContains: "alpha capability ran"}},
		{name: "matching assertion", proof: RuntimeProof{Lane: "alpha", Command: "git alpha", OutputContains: "alpha capability ran"}, wantOK: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			if tc.wantOK {
				configureRuntimeAlias(t, root, "alpha")
			}
			writeRuntimeProofFixture(t, root, []RuntimeProof{tc.proof})
			err := VerifyRuntimeProofs(root)
			if tc.wantOK && err != nil {
				t.Fatalf("VerifyRuntimeProofs rejected matching output: %v", err)
			}
			if !tc.wantOK && err == nil {
				t.Fatal("VerifyRuntimeProofs accepted an unbound runtime proof")
			}
		})
	}
}

func TestGatherFactsDefaultRequiresExplicitRuntimeProof(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "internal", "alpha"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "internal", "alpha", "alpha.go"), []byte("package alpha\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[lanes.trees]\nalpha = [\"internal/alpha/**\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "cli-reference.md"), []byte("# alpha documented command\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeRuntimeProofFixture(t, root, nil)
	facts := gatherFacts(root)
	if len(facts) != 1 || facts[0].DefaultSurface {
		t.Fatalf("documentation promoted default surface: %+v", facts)
	}

	configureRuntimeAlias(t, root, "alpha")
	writeRuntimeProofFixture(t, root, []RuntimeProof{{
		Lane: "alpha", Command: "git alpha", OutputContains: "alpha capability ran", DefaultOn: true,
		DefaultReason: "the command exercises alpha without an opt-in flag",
	}})
	payload := Build(Options{Root: root})
	if len(payload.Caps) != 1 || !payload.Caps[0].Dogfooded || !payload.Caps[0].DefaultSurface {
		t.Fatalf("explicit passing default proof did not promote alpha: %+v", payload.Caps)
	}
}

func TestRuntimeProofDefaultMetadataFailsClosed(t *testing.T) {
	for _, tc := range []struct {
		name  string
		proof RuntimeProof
	}{
		{name: "missing reason", proof: RuntimeProof{Lane: "alpha", Command: "git --version", OutputContains: "alpha capability ran", DefaultOn: true}},
		{name: "reason without declaration", proof: RuntimeProof{Lane: "alpha", Command: "git --version", OutputContains: "alpha capability ran", DefaultReason: "default somehow"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			writeRuntimeProofFixture(t, root, []RuntimeProof{tc.proof})
			if _, err := verifyRuntimeProofs(root); err == nil {
				t.Fatal("verifyRuntimeProofs accepted malformed default metadata")
			}
		})
	}
}

func TestMetalgemmRuntimeProofRecorded(t *testing.T) {
	root := filepath.Clean(filepath.Join("..", ".."))
	proofs, err := loadRuntimeProofs(root)
	if err != nil {
		t.Fatalf("loadRuntimeProofs: %v", err)
	}
	proof, ok := proofs["metalgemm"]
	if !ok {
		t.Fatalf("missing metalgemm runtime proof in runtime-proofs.json")
	}
	if proof.Command != "fak sota internal/metalgemm/decode.m" {
		t.Fatalf("unexpected command for metalgemm: %q", proof.Command)
	}
	if proof.OutputContains != "internal/metalgemm" {
		t.Fatalf("unexpected output_contains for metalgemm: %q", proof.OutputContains)
	}
}
