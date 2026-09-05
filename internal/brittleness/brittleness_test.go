package brittleness

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// --- Fold ------------------------------------------------------------------------------------

func TestFoldCleanTree(t *testing.T) {
	p := Fold(nil)
	if !p.OK {
		t.Fatalf("clean tree must be OK, got verdict %q", p.Verdict)
	}
	if got := p.Corpus[DebtKey]; got != 0 {
		t.Fatalf("clean debt = %v, want 0", got)
	}
	if got := p.Corpus[PressureKey]; got != 0 {
		t.Fatalf("clean pressure = %v, want 0", got)
	}
	// Every registered class still emits a KPI so the coverage surface is always visible.
	if len(p.KPIs) != len(registry) {
		t.Fatalf("clean KPIs = %d, want one per registered class (%d)", len(p.KPIs), len(registry))
	}
}

// The load-bearing invariant: a history-window finding is a SOFT signal, never debt, so
// it can never red a peer's gate. debt stays 0 while pressure rises.
func TestFindingsNeverGate(t *testing.T) {
	findings := []Finding{
		{Class: ClassFlakyRetryPass, Ref: "internal/foo", Detail: "flaked", Weight: 3},
		{Class: ClassRecurringFix, Ref: "a.go", Detail: "re-fixed", Weight: 4},
		{Class: ClassRevertedLanding, Ref: "deadbeef", Detail: "reverted", Weight: 1},
	}
	p := Fold(findings)
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("brittleness findings must NOT gate: OK=%v verdict=%q", p.OK, p.Verdict)
	}
	if got := p.Corpus[DebtKey]; got != 0 {
		t.Fatalf("debt = %v, want 0 (findings ride as Soft, never Defects)", got)
	}
	// No KPI may carry a Defect; every finding must land in Soft.
	softTotal := 0
	for _, k := range p.KPIs {
		if len(k.Defects) != 0 {
			t.Fatalf("KPI %s has %d defects; brittleness findings must be Soft-only", k.Key, len(k.Defects))
		}
		softTotal += len(k.Soft)
	}
	if softTotal != len(findings) {
		t.Fatalf("soft entries = %d, want %d (one per finding)", softTotal, len(findings))
	}
}

// pressure = Σ severity × weight, unbounded and severity-weighted per the registry.
func TestPressureMath(t *testing.T) {
	findings := []Finding{
		{Class: ClassFlakyRetryPass, Ref: "pkg", Weight: 3},  // sev 4 * 3 = 12
		{Class: ClassRecurringFix, Ref: "a.go", Weight: 2},   // sev 4 * 2 = 8
		{Class: ClassRevertedLanding, Ref: "sha", Weight: 5}, // sev 2 * 5 = 10
	}
	want := 12 + 8 + 10
	if got := Pressure(findings); got != want {
		t.Fatalf("Pressure = %d, want %d", got, want)
	}
	p := Fold(findings)
	if got := p.Corpus[PressureKey]; got != want {
		t.Fatalf("corpus[%s] = %v, want %d", PressureKey, got, want)
	}
}

// A zero/negative weight still counts as one occurrence (a seam that bit at least once).
func TestPressureWeightFloor(t *testing.T) {
	got := Pressure([]Finding{{Class: ClassRevertedLanding, Ref: "sha", Weight: 0}})
	if got != 2 { // sev 2 * floor(1)
		t.Fatalf("zero-weight pressure = %d, want 2 (weight floored to 1)", got)
	}
}

// An unregistered class contributes severity 1 (never panics), so a future producer bug
// degrades to a low-weight signal rather than crashing the fold.
func TestUnknownClassSeverityOne(t *testing.T) {
	got := Pressure([]Finding{{Class: Class("MYSTERY"), Ref: "x", Weight: 3}})
	if got != 3 {
		t.Fatalf("unknown-class pressure = %d, want 3 (severity defaults to 1)", got)
	}
}

func TestSortWorstFirst(t *testing.T) {
	fs := []Finding{
		{Class: ClassRevertedLanding, Ref: "z", Weight: 1},
		{Class: ClassRecurringFix, Ref: "a", Weight: 9},
		{Class: ClassFlakyRetryPass, Ref: "m", Weight: 9},
	}
	SortFindings(fs)
	// Highest weight first; ties broken by registry order (flaky before recurring-fix).
	if fs[0].Weight != 9 || fs[0].Class != ClassFlakyRetryPass {
		t.Fatalf("worst-first ordering wrong: %+v", fs[0])
	}
	if fs[1].Class != ClassRecurringFix {
		t.Fatalf("tie should break by registry order, got %s at [1]", fs[1].Class)
	}
	if fs[2].Weight != 1 {
		t.Fatalf("lightest finding should sort last, got weight %d", fs[2].Weight)
	}
}

// The captured-fresh evidence must survive into the rendered Soft worklist line.
func TestFreshEvidenceRendered(t *testing.T) {
	p := Fold([]Finding{{
		Class: ClassRecurringFix, Ref: "a.go", Detail: "re-fixed by 2",
		Weight: 2, Fresh: []string{"111aaa", "222bbb"},
	}})
	var line string
	for _, k := range p.KPIs {
		for _, s := range k.Soft {
			if strings.Contains(s, "a.go") {
				line = s
			}
		}
	}
	if line == "" {
		t.Fatal("recurring-fix finding not rendered in any KPI Soft list")
	}
	if !strings.Contains(line, "captured-fresh: 111aaa, 222bbb") {
		t.Fatalf("fresh evidence not captured in worklist line: %q", line)
	}
}

// The payload must round-trip through JSON with the control-pane keys a consumer reads.
func TestPayloadJSONContract(t *testing.T) {
	p := Fold([]Finding{{Class: ClassFlakyRetryPass, Ref: "pkg", Weight: 1}})
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round map[string]any
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round["schema"] != Schema {
		t.Fatalf("schema = %v, want %v", round["schema"], Schema)
	}
	corpus, _ := round["corpus"].(map[string]any)
	if corpus == nil {
		t.Fatal("corpus missing from payload")
	}
	for _, key := range []string{PressureKey, DebtKey, "seams_observed", "classes_registered"} {
		if _, ok := corpus[key]; !ok {
			t.Fatalf("corpus missing key %q", key)
		}
	}
}

// --- registry --------------------------------------------------------------------------------

// Every registered class must have a unique KPI key and a non-empty mitigation, and the
// registry must be closed (SpecOf resolves each canonical class).
func TestRegistryClosedAndUnique(t *testing.T) {
	keys := map[string]bool{}
	for _, s := range Registry() {
		if s.Mitigation == "" || s.Title == "" {
			t.Fatalf("class %s missing title/mitigation", s.Class)
		}
		if s.Severity <= 0 {
			t.Fatalf("class %s severity must be positive, got %d", s.Class, s.Severity)
		}
		k := kpiKey(s.Class)
		if keys[k] {
			t.Fatalf("duplicate KPI key %q", k)
		}
		keys[k] = true
		if _, ok := SpecOf(s.Class); !ok {
			t.Fatalf("registered class %s not resolvable via SpecOf", s.Class)
		}
	}
	if _, ok := SpecOf(Class("NOPE")); ok {
		t.Fatal("SpecOf must reject an unregistered class")
	}
}

// --- DetectRecurringFixes --------------------------------------------------------------------

func TestDetectRecurringFixes(t *testing.T) {
	commits := []Commit{
		{SHA: "aaa", Subject: "fix(cache): evict on ttl", Files: []string{"cache.go"}},
		{SHA: "bbb", Subject: "fix: cache eviction still leaks", Files: []string{"cache.go", "cache_test.go"}},
		{SHA: "ccc", Subject: "feat: add new pane", Files: []string{"pane.go"}}, // not a fix
		{SHA: "ddd", Subject: "fix(pane): typo", Files: []string{"pane.go"}},    // single fix, no recurrence
	}
	got := DetectRecurringFixes(commits)
	if len(got) != 1 {
		t.Fatalf("want exactly 1 recurring-fix seam (cache.go), got %d: %+v", len(got), got)
	}
	f := got[0]
	if f.Class != ClassRecurringFix || f.Ref != "cache.go" {
		t.Fatalf("wrong finding: %+v", f)
	}
	if f.Weight != 2 {
		t.Fatalf("weight = %d, want 2 (two fix commits touched cache.go)", f.Weight)
	}
	// Both fix SHAs captured fresh, sorted.
	if len(f.Fresh) != 2 || f.Fresh[0] != "aaa" || f.Fresh[1] != "bbb" {
		t.Fatalf("fresh SHAs not captured/sorted: %v", f.Fresh)
	}
}

// A file touched once by a feat and once by a fix is NOT a recurring fix (only one fix).
func TestRecurringFixIgnoresNonFixChurn(t *testing.T) {
	commits := []Commit{
		{SHA: "a", Subject: "feat: build widget", Files: []string{"w.go"}},
		{SHA: "b", Subject: "refactor: tidy widget", Files: []string{"w.go"}},
		{SHA: "c", Subject: "fix: widget nil deref", Files: []string{"w.go"}},
	}
	if got := DetectRecurringFixes(commits); len(got) != 0 {
		t.Fatalf("healthy iteration must not flag: %+v", got)
	}
}

// The same fix commit listing a file twice must not inflate the recurrence count.
func TestRecurringFixDedupesSameCommit(t *testing.T) {
	commits := []Commit{
		{SHA: "a", Subject: "fix: x", Files: []string{"f.go", "f.go"}},
	}
	if got := DetectRecurringFixes(commits); len(got) != 0 {
		t.Fatalf("one fix commit can never be a recurrence: %+v", got)
	}
}

// --- DetectReverts ---------------------------------------------------------------------------

func TestDetectReverts(t *testing.T) {
	commits := []Commit{
		{SHA: "a", Subject: `Revert "feat: risky landing"`, Files: []string{"x.go"}},
		{SHA: "b", Subject: "revert(core): back out cache change", Files: []string{"c.go"}},
		{SHA: "c", Subject: "revert: undo the thing", Files: []string{"d.go"}},
		{SHA: "d", Subject: "fix: something that mentions revert in prose", Files: []string{"e.go"}},
	}
	got := DetectReverts(commits)
	if len(got) != 3 {
		t.Fatalf("want 3 reverts (a,b,c), got %d: %+v", len(got), got)
	}
	for _, f := range got {
		if f.Class != ClassRevertedLanding {
			t.Fatalf("wrong class %s", f.Class)
		}
		if len(f.Fresh) != 1 || f.Fresh[0] != f.Ref {
			t.Fatalf("revert must capture its own sha fresh: %+v", f)
		}
	}
}

func TestDetectFromCommitsMerges(t *testing.T) {
	commits := []Commit{
		{SHA: "aaa", Subject: "fix: leak", Files: []string{"m.go"}},
		{SHA: "bbb", Subject: "fix: leak again", Files: []string{"m.go"}},
		{SHA: "ccc", Subject: `Revert "feat: x"`, Files: []string{"x.go"}},
	}
	got := DetectFromCommits(commits)
	var recurring, reverts int
	for _, f := range got {
		switch f.Class {
		case ClassRecurringFix:
			recurring++
		case ClassRevertedLanding:
			reverts++
		}
	}
	if recurring != 1 || reverts != 1 {
		t.Fatalf("merged detect wrong: recurring=%d reverts=%d (%+v)", recurring, reverts, got)
	}
}

func TestEmptyWindowNoFindings(t *testing.T) {
	if got := DetectFromCommits(nil); len(got) != 0 {
		t.Fatalf("empty window must yield no findings, got %+v", got)
	}
}

// --- FromFlakyPackages (capture-when-fresh bridge) -------------------------------------------

func TestFromFlakyPackages(t *testing.T) {
	got := FromFlakyPackages([]string{"internal/foo", "internal/bar", "internal/foo", ""}, "tree-sha-123")
	if len(got) != 2 {
		t.Fatalf("want 2 distinct flaky seams (blank skipped), got %d: %+v", len(got), got)
	}
	byRef := map[string]Finding{}
	for _, f := range got {
		if f.Class != ClassFlakyRetryPass {
			t.Fatalf("wrong class %s", f.Class)
		}
		if len(f.Fresh) != 1 || f.Fresh[0] != "tree-sha-123" {
			t.Fatalf("freshness stamp not captured: %+v", f)
		}
		byRef[f.Ref] = f
	}
	// foo flaked twice -> weight 2, a heavier seam than bar (weight 1).
	if byRef["internal/foo"].Weight != 2 {
		t.Fatalf("foo weight = %d, want 2", byRef["internal/foo"].Weight)
	}
	if byRef["internal/bar"].Weight != 1 {
		t.Fatalf("bar weight = %d, want 1", byRef["internal/bar"].Weight)
	}
}

// The bridge takes no clock: a blank freshness stamp yields no Fresh slice (never a
// fabricated timestamp), matching the rest of the clock-free scorecard family.
func TestFromFlakyPackagesNoClock(t *testing.T) {
	got := FromFlakyPackages([]string{"pkg"}, "")
	if len(got) != 1 || len(got[0].Fresh) != 0 {
		t.Fatalf("blank stamp must not fabricate freshness: %+v", got)
	}
}

// End-to-end: a fresh flaky verdict folds into a non-gating, pressure-bearing card.
func TestFlakyBridgeFoldsToCard(t *testing.T) {
	findings := FromFlakyPackages([]string{"internal/x", "internal/x", "internal/y"}, "sha")
	p := Fold(findings)
	if !p.OK {
		t.Fatal("flaky observations must not gate the card")
	}
	// x sev4*2 + y sev4*1 = 12.
	if got := p.Corpus[PressureKey]; got != 12 {
		t.Fatalf("pressure = %v, want 12", got)
	}
	if got := p.Corpus["flaky_retry_pass"]; got != 2 {
		t.Fatalf("flaky_retry_pass count = %v, want 2", got)
	}
}

// --- Benchmarks ------------------------------------------------------------------------------

var (
	benchFindingsSink []Finding
	benchPayloadSink  scorecard.Payload
	benchPressureSink int
)

func sampleCommits(n int) []Commit {
	commits := make([]Commit, n)
	for i := 0; i < n; i++ {
		sha := fmt.Sprintf("%07x", i+1)
		fileA := fmt.Sprintf("pkg%d/file%d.go", i%15, (i%15)+1)
		fileB := fmt.Sprintf("pkg%d/helper.go", i%15)
		var subject string
		switch i % 10 {
		case 0, 3, 6:
			subject = fmt.Sprintf("fix(pkg%d): resolve nil pointer in worker", i%15)
		case 1:
			subject = fmt.Sprintf("revert(pkg%d): back out cache change", i%15)
		case 2:
			subject = fmt.Sprintf("Revert \"feat: experimental feature %d\"", i)
		default:
			subject = fmt.Sprintf("feat(pkg%d): add telemetry tracking %d", i%15, i)
		}
		commits[i] = Commit{
			SHA:     sha,
			Subject: subject,
			Files:   []string{fileA, fileB},
		}
	}
	return commits
}

func sampleFindings(n int) []Finding {
	out := make([]Finding, n)
	classes := []Class{ClassFlakyRetryPass, ClassRecurringFix, ClassRevertedLanding}
	for i := 0; i < n; i++ {
		cls := classes[i%len(classes)]
		ref := fmt.Sprintf("internal/pkg%d/file%d.go", i%20, i)
		if cls == ClassRevertedLanding {
			ref = fmt.Sprintf("%07x", i)
		}
		out[i] = Finding{
			Class:      cls,
			Ref:        ref,
			Detail:     fmt.Sprintf("observation detail %d", i),
			Mitigation: "sample remediation action",
			Weight:     (i % 5) + 1,
			Fresh:      []string{fmt.Sprintf("%07x", i), fmt.Sprintf("%07x", i+100)},
		}
	}
	return out
}

func sampleFlakyPackages(n int) []string {
	pkgs := make([]string, n)
	for i := 0; i < n; i++ {
		pkgs[i] = fmt.Sprintf("internal/pkg%d", i%25)
	}
	return pkgs
}

func BenchmarkDetectFromCommits(b *testing.B) {
	for _, size := range []int{50, 200, 500} {
		b.Run(fmt.Sprintf("window_%d", size), func(b *testing.B) {
			commits := sampleCommits(size)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchFindingsSink = DetectFromCommits(commits)
			}
		})
	}
}

func BenchmarkDetectRecurringFixes(b *testing.B) {
	commits := sampleCommits(200)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchFindingsSink = DetectRecurringFixes(commits)
	}
}

func BenchmarkDetectReverts(b *testing.B) {
	commits := sampleCommits(200)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchFindingsSink = DetectReverts(commits)
	}
}

func BenchmarkFromFlakyPackages(b *testing.B) {
	for _, count := range []int{10, 50, 200} {
		b.Run(fmt.Sprintf("count_%d", count), func(b *testing.B) {
			pkgs := sampleFlakyPackages(count)
			stamp := "d3adb33f"
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchFindingsSink = FromFlakyPackages(pkgs, stamp)
			}
		})
	}
}

func BenchmarkFold(b *testing.B) {
	b.Run("clean", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchPayloadSink = Fold(nil)
		}
	})
	for _, n := range []int{10, 50, 200} {
		b.Run(fmt.Sprintf("findings_%d", n), func(b *testing.B) {
			base := sampleFindings(n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fs := append([]Finding(nil), base...)
				benchPayloadSink = Fold(fs)
			}
		})
	}
}

func BenchmarkPressure(b *testing.B) {
	for _, n := range []int{10, 50, 200} {
		b.Run(fmt.Sprintf("findings_%d", n), func(b *testing.B) {
			fs := sampleFindings(n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchPressureSink = Pressure(fs)
			}
		})
	}
}

func BenchmarkSortFindings(b *testing.B) {
	for _, n := range []int{10, 50, 200} {
		b.Run(fmt.Sprintf("findings_%d", n), func(b *testing.B) {
			base := sampleFindings(n)
			buf := make([]Finding, n)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(buf, base)
				SortFindings(buf)
			}
		})
	}
}

// TestBenchmarkSanity ensures benchmark routines execute without panic and perform iterations.
func TestBenchmarkSanity(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark sanity in short mode")
	}
	res := testing.Benchmark(BenchmarkDetectRecurringFixes)
	if res.N <= 0 {
		t.Fatalf("expected positive iterations, got %d", res.N)
	}
}
