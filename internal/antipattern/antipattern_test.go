package antipattern

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// --- registry integrity: the honesty invariant --------------------------------------------

func TestRegistryNoAspirationalRows(t *testing.T) {
	// The whole point of the registry is that a class is listed only once it has a wired
	// detector: every registered class must be at least R2 SCORED. An R0/R1 row would be the
	// overstated-coverage anti-pattern the registry exists to prevent.
	for _, s := range Registry() {
		if s.Rung < RungScored {
			t.Errorf("class %s registered at %s; a registered class must be >= R2-scored", s.Class, s.Rung.Label())
		}
		if s.Title == "" || s.Mitigation == "" {
			t.Errorf("class %s missing title or mitigation", s.Class)
		}
		if _, ok := SpecOf(s.Class); !ok {
			t.Errorf("class %s not indexed by SpecOf", s.Class)
		}
	}
}

// TestEveryClassHasACure is the mitigation-half invariant: detection without a paired cure
// is only half the loop. A registered class MUST name a routed remediation, so the meta-loop
// can turn every detected finding into a concrete action -- you cannot wire a detector here
// without also wiring how it gets mitigated.
func TestEveryClassHasACure(t *testing.T) {
	for _, s := range Registry() {
		if strings.TrimSpace(s.Cure) == "" {
			t.Errorf("class %s has no Cure: a detector must ship with its mitigation (concept-disambiguation, a dispatch verb, or an explicit manual action)", s.Class)
		}
	}
	// The folded payload must carry the cure manifest so a consumer reads the mitigation
	// straight off the control-pane payload instead of re-deriving it.
	p := Fold([]Finding{{Class: ClassRedundantRework, Ref: "abc", Detail: "redone", Weight: 2}}, nil)
	cures, ok := p.Corpus["cures"].(map[string]string)
	if !ok {
		t.Fatalf("payload corpus missing cures manifest, got %T", p.Corpus["cures"])
	}
	if len(cures) != len(Registry()) {
		t.Fatalf("cures manifest has %d entries, want one per registered class (%d)", len(cures), len(Registry()))
	}
	// With rework debt present, the NextAction must route to the concept-disambiguation cure
	// (the operator's named remedy for repetition), not a generic "fix it".
	if !strings.Contains(p.NextAction, "rename-concept") {
		t.Errorf("NextAction should route rework debt to concept-disambiguation, got %q", p.NextAction)
	}
}

// --- Fold: pure findings -> payload --------------------------------------------------------

func TestFoldCleanIsOK(t *testing.T) {
	p := Fold(nil, nil)
	if !p.OK || p.Verdict != "OK" {
		t.Fatalf("clean fold should be OK, got ok=%v verdict=%s", p.OK, p.Verdict)
	}
	if got := p.Corpus[DebtKey]; got != 0 {
		t.Fatalf("clean debt = %v, want 0", got)
	}
	// A clean card still shows every registered class as a KPI (full coverage surface).
	if len(p.KPIs) != len(Registry()) {
		t.Fatalf("clean card KPIs = %d, want %d (one per registered class)", len(p.KPIs), len(Registry()))
	}
}

func TestFoldCountsDebtAndOrdersWorstFirst(t *testing.T) {
	findings := []Finding{
		{Class: ClassOrphanFunc, Ref: "internal/a/a.go:10", Detail: "func lo", Weight: 1},
		{Class: ClassUnwiredPkg, Ref: "internal/big", Detail: "big pkg", Weight: 900},
		{Class: ClassRedundantRework, Ref: "abc123", Detail: "redone", Weight: 3},
	}
	p := Fold(findings, map[Class]int{ClassUnwiredPkg: 50})
	if p.OK {
		t.Fatal("fold with findings must not be OK")
	}
	if got := p.Corpus[DebtKey]; got != 3 {
		t.Fatalf("debt = %v, want 3", got)
	}
	// Per-class corpus counts are exposed for the header line.
	if p.Corpus["unwired_pkg"] != 1 || p.Corpus["redundant_rework"] != 1 || p.Corpus["orphan_func"] != 1 {
		t.Fatalf("per-class counts wrong: %+v", p.Corpus)
	}
	if p.Corpus["classes_with_debt"] != 3 {
		t.Fatalf("classes_with_debt = %v, want 3", p.Corpus["classes_with_debt"])
	}
}

func TestSortFindingsWorstFirst(t *testing.T) {
	fs := []Finding{
		{Class: ClassOrphanFunc, Ref: "z", Weight: 1},
		{Class: ClassUnwiredPkg, Ref: "y", Weight: 10},
		{Class: ClassOrphanFunc, Ref: "a", Weight: 1},
	}
	SortFindings(fs)
	if fs[0].Weight != 10 {
		t.Fatalf("heaviest finding should sort first, got weight %d", fs[0].Weight)
	}
	// Equal weight -> registry class order, then ref.
	if fs[1].Ref != "a" || fs[2].Ref != "z" {
		t.Fatalf("equal-weight tie-break wrong: %s then %s", fs[1].Ref, fs[2].Ref)
	}
}

// TestMitigationPlanRoutesWorstFirstAndFlagsAuto is the executable-loop invariant: the plan a
// `fak loop` tick runs must (1) order actions worst-first, (2) route each finding to its class
// cure, and (3) mark exactly the safe cures auto-dispatchable so the loop runs those unattended
// and hands the code-changing ones off as work orders.
func TestMitigationPlanRoutesWorstFirstAndFlagsAuto(t *testing.T) {
	findings := []Finding{
		{Class: ClassOrphanFunc, Ref: "internal/a/a.go:10", Detail: "func lo", Weight: 1},
		{Class: ClassUnwiredPkg, Ref: "internal/big", Detail: "big pkg", Weight: 900},
		{Class: ClassRedundantRework, Ref: "abc123", Detail: "redone", Weight: 3},
	}
	plan := MitigationPlan(findings)
	if len(plan) != 3 {
		t.Fatalf("plan has %d actions, want 3", len(plan))
	}
	// (1) worst-first: the 900-weight unwired package leads.
	if plan[0].Class != ClassUnwiredPkg || plan[0].Weight != 900 {
		t.Fatalf("heaviest action should lead, got %s weight %d", plan[0].Class, plan[0].Weight)
	}
	// (2) each action carries its class's routed cure.
	for _, a := range plan {
		s, ok := SpecOf(a.Class)
		if !ok {
			t.Fatalf("plan carries unregistered class %s", a.Class)
		}
		if a.Cure != s.Cure {
			t.Errorf("%s cure mismatch: plan %q vs spec %q", a.Class, a.Cure, s.Cure)
		}
	}
	// (3) only the issue-filing unwired cure is auto-dispatchable; code-changing cures are not.
	for _, a := range plan {
		wantAuto := a.Class == ClassUnwiredPkg
		if a.Auto != wantAuto {
			t.Errorf("%s Auto = %v, want %v (only the deduped-issue cure runs unattended)", a.Class, a.Auto, wantAuto)
		}
	}
	// MitigationPlan must not mutate the caller's slice order (it sorts a copy).
	if findings[0].Class != ClassOrphanFunc {
		t.Error("MitigationPlan mutated the caller's findings slice; it must sort a copy")
	}
}

// --- DetectRedundantRework: the post-hoc repetition detector -------------------------------

func TestRedundantReworkFlagsSameClaimOverlappingFiles(t *testing.T) {
	commits := []Commit{
		{SHA: "aaa", Subject: "feat(cache): add cache eviction policy", Files: []string{"internal/cache/evict.go"}},
		{SHA: "bbb", Subject: "feat: add cache eviction policy", Files: []string{"internal/cache/evict.go", "internal/cache/evict_test.go"}},
	}
	got := DetectRedundantRework(commits)
	if len(got) != 1 {
		t.Fatalf("want 1 redundant-rework finding, got %d: %+v", len(got), got)
	}
	if got[0].Class != ClassRedundantRework || got[0].Weight != 2 {
		t.Fatalf("finding shape wrong: %+v", got[0])
	}
	if !strings.Contains(got[0].Detail, "evict.go") {
		t.Fatalf("detail should name the shared file: %q", got[0].Detail)
	}
}

func TestRedundantReworkIgnoresNormalFeatureSequence(t *testing.T) {
	// A real multi-step feature: distinct claims, so distinct keys -> never clusters, even
	// though it touches the same file repeatedly. This is the precision guard.
	commits := []Commit{
		{SHA: "a1", Subject: "feat(auth): add token refresh", Files: []string{"internal/auth/token.go"}},
		{SHA: "a2", Subject: "test(auth): cover token refresh expiry", Files: []string{"internal/auth/token.go"}},
		{SHA: "a3", Subject: "docs(auth): document token refresh", Files: []string{"internal/auth/token.go"}},
	}
	if got := DetectRedundantRework(commits); len(got) != 0 {
		t.Fatalf("normal feature sequence must not flag, got %+v", got)
	}
}

func TestRedundantReworkRequiresFileOverlap(t *testing.T) {
	// Same claim words but disjoint files = two different jobs that share vocabulary.
	commits := []Commit{
		{SHA: "d1", Subject: "feat: add retry backoff", Files: []string{"internal/dispatch/retry.go"}},
		{SHA: "d2", Subject: "feat: add retry backoff", Files: []string{"internal/relay/retry.go"}},
	}
	if got := DetectRedundantRework(commits); len(got) != 0 {
		t.Fatalf("disjoint files must not flag, got %+v", got)
	}
}

func TestRedundantReworkThreeCommitCluster(t *testing.T) {
	commits := []Commit{
		{SHA: "c1", Subject: "feat: implement headroom governor", Files: []string{"internal/headroom/gov.go"}},
		{SHA: "c2", Subject: "feat: implement headroom governor", Files: []string{"internal/headroom/gov.go"}},
		{SHA: "c3", Subject: "fix: implement headroom governor", Files: []string{"internal/headroom/gov.go"}},
	}
	got := DetectRedundantRework(commits)
	if len(got) != 1 || got[0].Weight != 3 {
		t.Fatalf("want one weight-3 cluster, got %+v", got)
	}
}

func TestNormalizeClaimStripsPrefixAndStamp(t *testing.T) {
	cases := map[string]string{
		"feat(cache): add cache eviction policy":      "cache eviction policy",
		"feat: add cache eviction policy again":       "again cache eviction policy",
		"fix(headroom)!: implement headroom governor": "governor headroom implement",
		"chore: gofmt": "", // one salient token -> un-clusterable
		"docs(auth): document token refresh (#2105)": "document refresh token",
	}
	for subject, want := range cases {
		if got := normalizeClaim(subject); got != want {
			t.Errorf("normalizeClaim(%q) = %q, want %q", subject, got, want)
		}
	}
}

// --- adapter smoke test: real detectors over a temp tree -----------------------------------

func TestCollectOverTempTreeFindsOrphanFunc(t *testing.T) {
	root := t.TempDir()
	pkgDir := filepath.Join(root, "internal", "sample")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	// wiredHelper is called; strayHelper is defined but referenced nowhere -> ORPHAN_FUNC.
	src := `package sample

func Used() int { return wiredHelper() }

func wiredHelper() int { return 1 }

func strayHelper() int { return 2 }
`
	if err := os.WriteFile(filepath.Join(pkgDir, "sample.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	findings, _ := Collect(root, nil)
	var orphan *Finding
	for i := range findings {
		if findings[i].Class == ClassOrphanFunc && strings.Contains(findings[i].Detail, "strayHelper") {
			orphan = &findings[i]
			break
		}
	}
	if orphan == nil {
		t.Fatalf("expected an ORPHAN_FUNC finding for strayHelper, got %+v", findings)
	}
	if !strings.HasPrefix(orphan.Ref, "internal/sample/sample.go:") {
		t.Fatalf("orphan ref should be repo-relative file:line, got %q", orphan.Ref)
	}
}
