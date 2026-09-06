package issuefanout

import (
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

func spineInput() Input {
	return Input{
		Title:    "issue fanout planner",
		Leaf:     "issuefanout",
		SpineRef: "fak issue fanout --title ... --json (internal/issuefanout Build)",
	}
}

// The point of the leaf: every generated follow-on is dispatchable under the
// full issuecontract scope rules the moment it is planned.
func TestBuildEveryCandidateDispatchable(t *testing.T) {
	plan, err := Build(spineInput())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if len(plan.Candidates) < MinFanout {
		t.Fatalf("fan-out floor: got %d candidates, want >= %d", len(plan.Candidates), MinFanout)
	}
	for _, c := range plan.Candidates {
		r := issuepolicy.ReviewCandidate(c, issuepolicy.Options{})
		if r.Dispatchability != issuepolicy.Dispatchable {
			t.Errorf("candidate %s not dispatchable: verdict=%s reasons=%v missing=%v",
				c.Key, r.Verdict, r.Reasons, r.MissingFields)
		}
	}
}

// No fan-out without a spine witness: the refusal must name the recovery.
func TestBuildRequiresSpineWitness(t *testing.T) {
	in := spineInput()
	in.SpineRef = " "
	_, err := Build(in)
	if err == nil {
		t.Fatal("Build accepted an empty spine_ref")
	}
	if !strings.Contains(err.Error(), "working spine first") {
		t.Fatalf("refusal does not name the recovery: %v", err)
	}
}

func TestBuildAreaFilterAndCap(t *testing.T) {
	in := spineInput()
	in.Areas = []string{"qa"}
	plan, err := Build(in)
	if err != nil {
		t.Fatalf("Build(areas=qa): %v", err)
	}
	for _, c := range plan.Candidates {
		if len(c.Labels) < 2 || c.Labels[1] != "qa" {
			t.Fatalf("area filter leaked non-qa candidate %s (labels %v)", c.Key, c.Labels)
		}
	}
	if plan.AreaCounts["qa"] != len(plan.Candidates) {
		t.Fatalf("area counts disagree: %v vs %d candidates", plan.AreaCounts, len(plan.Candidates))
	}

	in = spineInput()
	in.Max = MinFanout
	plan, err = Build(in)
	if err != nil {
		t.Fatalf("Build(max=%d): %v", MinFanout, err)
	}
	if len(plan.Candidates) != MinFanout {
		t.Fatalf("cap ignored: got %d, want %d", len(plan.Candidates), MinFanout)
	}

	in.Max = MinFanout - 1
	if _, err := Build(in); err == nil {
		t.Fatal("Build accepted a cap below the fan-out floor")
	}

	in = spineInput()
	in.Areas = []string{"nonsense"}
	if _, err := Build(in); err == nil || !strings.Contains(err.Error(), "known:") {
		t.Fatalf("unknown area not refused with the known list: %v", err)
	}
}

func TestBuildMapsEveryTaxonomyAreaToWorkClass(t *testing.T) {
	plan, err := Build(spineInput())
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{
		"qa":            "class:infra",
		"dogfood":       "class:dev",
		"product":       "class:dev",
		"observability": "class:infra",
		"integration":   "class:infra",
		"docs":          "class:frontdoor",
		"release":       "class:frontdoor",
	}
	seen := map[string]bool{}
	for _, c := range plan.Candidates {
		if len(c.Labels) < 4 {
			t.Fatalf("%s labels = %v, want area/class/priority routing", c.Key, c.Labels)
		}
		area := c.Labels[1]
		seen[area] = true
		if got := c.Labels[2]; got != want[area] {
			t.Errorf("%s area %s class = %q, want %q", c.Key, area, got, want[area])
		}
	}
	for area := range want {
		if !seen[area] {
			t.Errorf("taxonomy area %q had no generated candidate", area)
		}
	}
}

func TestBuildMapsPriorityToModelTiers(t *testing.T) {
	plan, err := Build(spineInput())
	if err != nil {
		t.Fatal(err)
	}
	want := map[string][2]string{
		"priority/P1": {"T1", "T0"},
		"priority/P2": {"T2", "T1"},
	}
	seen := map[string]bool{}
	for _, c := range plan.Candidates {
		tiers, ok := want[c.Priority]
		if !ok {
			t.Fatalf("%s has unmapped priority %q", c.Key, c.Priority)
		}
		seen[c.Priority] = true
		if c.RequiredModelTier != tiers[0] || c.OptimalModelTier != tiers[1] {
			t.Errorf("%s priority %s tiers = %s/%s, want %s/%s", c.Key, c.Priority, c.RequiredModelTier, c.OptimalModelTier, tiers[0], tiers[1])
		}
	}
	for priority := range want {
		if !seen[priority] {
			t.Errorf("priority mapping %q had no generated candidate", priority)
		}
	}
}

func TestBuildDerivesQAPackageFromDeclaredPaths(t *testing.T) {
	in := spineInput()
	in.Leaf = "cmd"
	in.Paths = []string{
		"cmd/fak/guard_disable.go",
		"cmd/fak/guard_disable_test.go",
		"docs/integrations/openai-codex.md",
	}
	in.Areas = []string{"qa"}
	plan, err := Build(in)
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range plan.Candidates {
		for field, value := range map[string]string{
			"witness": candidate.Witness,
			"gate":    candidate.AcceptanceGate,
		} {
			if !strings.Contains(value, "./cmd/fak") || strings.Contains(value, "./internal/cmd") {
				t.Fatalf("%s %s = %q, want runnable ./cmd/fak command", candidate.Key, field, value)
			}
		}
	}

	internalPlan, err := Build(Input{Title: "internal spine", Leaf: "issuefanout", SpineRef: "abc", Areas: []string{"qa"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range internalPlan.Candidates {
		if !strings.Contains(candidate.Witness, "./internal/issuefanout") {
			t.Fatalf("%s witness = %q, want internal package mapping", candidate.Key, candidate.Witness)
		}
	}
}

func TestBuildAdmitsRunnableExamplePackage(t *testing.T) {
	in := spineInput()
	in.Leaf = "independent-server"
	in.Paths = []string{"examples/independent-server/**", "docs/integrations/independent-server.md"}
	in.Areas = []string{"qa"}
	in.Max = 3

	plan, err := Build(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Candidates) != 3 {
		t.Fatalf("got %d candidates, want 3", len(plan.Candidates))
	}
	for _, candidate := range plan.Candidates {
		for field, value := range map[string]string{"witness": candidate.Witness, "gate": candidate.AcceptanceGate} {
			if !strings.Contains(value, "./examples/independent-server") {
				t.Fatalf("%s %s = %q, want exact example package", candidate.Key, field, value)
			}
		}
	}
}

func TestBuildRefusesAmbiguousOrNonGoPackagePaths(t *testing.T) {
	for name, paths := range map[string][]string{
		"ambiguous":       {"cmd/fak/guard.go", "internal/issuefanout/issuefanout.go"},
		"non-go":          {"docs/integrations/openai-codex.md"},
		"docs-example":    {"examples/independent-server/README.md"},
		"missing-example": {"examples/not-a-package/**"},
	} {
		t.Run(name, func(t *testing.T) {
			in := spineInput()
			in.Paths = paths
			if _, err := Build(in); err == nil || !strings.Contains(err.Error(), "Go package") {
				t.Fatalf("Build(paths=%v) error = %v, want Go package refusal", paths, err)
			}
		})
	}
}

// The planner is pure: same input, byte-identical plan.
func TestBuildDeterministic(t *testing.T) {
	a, err := Build(spineInput())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	b, err := Build(spineInput())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatal("two Builds over the same input differ")
	}
}

// TestBuildConcurrentDeterminism is the #2513 race witness. The sequential
// TestBuildDeterministic proves same-input-same-output on one goroutine, but the
// planner's promise ("cohort/replay layers can trust it") is a CONCURRENT one:
// many callers fan the same spine out at once. This test runs a pool of
// goroutines over three distinct Build paths (default spine, area-filtered,
// capped) and asserts every concurrent plan deep-equals a sequential reference
// computed before any goroutine starts. Under `go test -race` it exercises the
// shared read of the package-level taxonomy and proves that path carries no data
// race — the meaningful `-race` witness the sequential test cannot give.
func TestBuildConcurrentDeterminism(t *testing.T) {
	qaDocs := spineInput()
	qaDocs.Areas = []string{"qa", "docs"}
	capped := spineInput()
	capped.Max = MinFanout
	inputs := []Input{spineInput(), qaDocs, capped}

	// Sequential references, fixed before the goroutines read them.
	refs := make([]Plan, len(inputs))
	for i, in := range inputs {
		p, err := Build(in)
		if err != nil {
			t.Fatalf("reference Build[%d]: %v", i, err)
		}
		refs[i] = p
	}

	const workers = 64
	got := make([]Plan, workers)
	errs := make([]error, workers)
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		// Each goroutine owns its own got[w]/errs[w] slot: the only shared reads
		// are the immutable inputs/refs and the package-level taxonomy.
		go func(w int) {
			defer wg.Done()
			got[w], errs[w] = Build(inputs[w%len(inputs)])
		}(w)
	}
	wg.Wait()

	for w := 0; w < workers; w++ {
		if errs[w] != nil {
			t.Fatalf("concurrent Build[%d]: %v", w, errs[w])
		}
		if ref := refs[w%len(inputs)]; !reflect.DeepEqual(got[w], ref) {
			t.Fatalf("concurrent Build[%d] diverged from its sequential reference", w)
		}
	}
}

func TestBuildGivesEveryChildCanonicalChildSpecificProblemFrame(t *testing.T) {
	plan, err := Build(Input{Title: "Cache spine", Leaf: "cache", SpineRef: "abc123", Max: 15})
	if err != nil {
		t.Fatal(err)
	}
	for _, candidate := range plan.Candidates {
		frame := candidate.ProblemFrame
		if frame.Schema != issuepolicy.ProblemFrameSchema || !frame.Enforced || !frame.Ready {
			t.Fatalf("%s frame = %+v", candidate.Key, frame)
		}
		if frame.Centrality == issuepolicy.CentralityEnabling && frame.CentralityTarget == "" {
			t.Fatalf("%s enabling frame lost Core target", candidate.Key)
		}
		if frame.Centrality == issuepolicy.CentralityStewardship && frame.CentralityTarget == "" {
			t.Fatalf("%s stewardship frame lost obligation", candidate.Key)
		}
		for _, id := range []string{"p1", "p2", "p3", "p4"} {
			if check := frame.Checks[id]; !check.Valid || check.Evidence == "" {
				t.Fatalf("%s %s = %+v", candidate.Key, id, check)
			}
		}
	}
}

func TestBuildDoesNotInheritProblemFrameFromParentMetadata(t *testing.T) {
	left, err := Build(Input{Title: "Core parent", Leaf: "core", SpineRef: "abc", ParentRef: "Core epic", Max: 3})
	if err != nil {
		t.Fatal(err)
	}
	right, err := Build(Input{Title: "Peripheral parent", Leaf: "peripheral", SpineRef: "abc", ParentRef: "Peripheral epic", Max: 3})
	if err != nil {
		t.Fatal(err)
	}
	for i := range left.Candidates {
		if !reflect.DeepEqual(left.Candidates[i].ProblemFrame, right.Candidates[i].ProblemFrame) {
			t.Fatalf("parent metadata changed child frame:\nleft=%+v\nright=%+v", left.Candidates[i].ProblemFrame, right.Candidates[i].ProblemFrame)
		}
	}
}

func TestBuildDerivesPerCandidateTreeFromScope(t *testing.T) {
	plan, err := Build(spineInput())
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	bySlug := map[string]issuepolicy.Candidate{}
	for _, c := range plan.Candidates {
		for _, tmpl := range taxonomy {
			if strings.HasSuffix(c.Key, "-"+tmpl.slug) {
				bySlug[tmpl.slug] = c
				break
			}
		}
	}

	// Done condition: docs-doctrine-linkage / release-claims / int-dos-wiring
	// do NOT carry internal/issuefanout/.
	for _, slug := range []string{"docs-doctrine-linkage", "release-claims", "int-dos-wiring"} {
		c, ok := bySlug[slug]
		if !ok {
			t.Fatalf("candidate %s missing from plan", slug)
		}
		for _, p := range c.Paths {
			if strings.Contains(p, "internal/issuefanout") {
				t.Errorf("candidate %s path %q carries internal/issuefanout", slug, p)
			}
		}
		for _, coord := range c.Coordination {
			if strings.Contains(coord, "internal/issuefanout") {
				t.Errorf("candidate %s coordination %q carries internal/issuefanout", slug, coord)
			}
		}
		if c.Lane == "issuefanout" {
			t.Errorf("candidate %s lane = %q, want non-leaf lane", slug, c.Lane)
		}
	}

	// All 8 templates whose scope lands outside the spine leaf declare their own tree and lane.
	wantScope := map[string]struct {
		lane  string
		paths []string
	}{
		"docs-doctrine-linkage": {lane: "docs", paths: []string{"docs/INDEX.md", "llms.txt", "AGENTS.md", "README.md"}},
		"release-claims":        {lane: "release", paths: []string{"CLAIMS.md", "docs/releases/"}},
		"int-dos-wiring":        {lane: "dos", paths: []string{"./dos.toml"}},
		"int-guard-gate":        {lane: "hooks", paths: []string{"internal/hooks/"}},
		"int-superloop":         {lane: "superloop", paths: []string{"internal/superloop/"}},
		"product-cli-reference": {lane: "docs", paths: []string{"docs/cli-reference.md"}},
		"product-lcd-demo":      {lane: "cmd", paths: []string{"cmd/*demo", "examples/"}},
		"dogfood-self-run":      {lane: "docs", paths: []string{"docs/notes/"}},
	}

	for slug, want := range wantScope {
		c, ok := bySlug[slug]
		if !ok {
			t.Fatalf("candidate %s missing from plan", slug)
		}
		if c.Lane != want.lane {
			t.Errorf("candidate %s lane = %q, want %q", slug, c.Lane, want.lane)
		}
		if !reflect.DeepEqual(c.Paths, want.paths) {
			t.Errorf("candidate %s paths = %v, want %v", slug, c.Paths, want.paths)
		}
		for _, coord := range c.Coordination {
			if strings.Contains(coord, "internal/issuefanout") {
				t.Errorf("candidate %s coordination %q carries internal/issuefanout", slug, coord)
			}
			if !strings.Contains(coord, strings.Join(want.paths, ", ")) {
				t.Errorf("candidate %s coordination %q does not carry its paths", slug, coord)
			}
		}
	}

	// Genuinely in-leaf QA/observability rows retain internal/<leaf>/.
	inLeafSlugs := []string{
		"qa-edge-sweep",
		"qa-failure-paths",
		"qa-determinism",
		"dogfood-usage-ledger",
		"product-error-ux",
		"obs-outcome-counters",
		"obs-scorecard",
	}
	for _, slug := range inLeafSlugs {
		c, ok := bySlug[slug]
		if !ok {
			t.Fatalf("candidate %s missing from plan", slug)
		}
		if c.Lane != "issuefanout" {
			t.Errorf("in-leaf candidate %s lane = %q, want issuefanout", slug, c.Lane)
		}
		if !reflect.DeepEqual(c.Paths, []string{"internal/issuefanout/"}) {
			t.Errorf("in-leaf candidate %s paths = %v, want [internal/issuefanout/]", slug, c.Paths)
		}
		if len(c.Coordination) == 0 || !strings.Contains(c.Coordination[0], "internal/issuefanout/") {
			t.Errorf("in-leaf candidate %s coordination = %v, want to contain internal/issuefanout/", slug, c.Coordination)
		}
	}
}
