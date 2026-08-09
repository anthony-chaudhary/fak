package devindex

import (
	"strings"
	"testing"
)

func TestCommandOwnershipIsTotalAndPinsProductBoundary(t *testing.T) {
	cat, err := Load(FindRoot("."))
	if err != nil {
		t.Fatal(err)
	}
	verbs := OwnershipVerbs(cat.Verbs())
	inventory := CommandOwnerships(verbs)
	if problems := ValidateCommandOwnership(verbs, inventory); len(problems) != 0 {
		t.Fatalf("ownership contract is not total:\n%s", strings.Join(problems, "\n"))
	}
	byName := make(map[string]CommandOwnership, len(inventory))
	for _, item := range inventory {
		byName[item.Name] = item
	}
	for _, name := range []string{"serve", "guard", "policy", "preflight", "agent"} {
		if got := byName[name]; got.Owner != OwnerRuntime || got.DispatchTarget != "fak" {
			t.Errorf("%s ownership = %+v, want runtime/fak", name, got)
		}
	}
	for _, name := range []string{"issue", "commit", "ci-preflight", "release", "fleet", "bench"} {
		if got := byName[name]; got.Owner != OwnerDev || got.DispatchTarget != "fak-dev" {
			t.Errorf("%s ownership = %+v, want dev/fak-dev", name, got)
		}
	}
}

func TestValidateCommandOwnershipRejectsMissingDuplicateUnknownAndIncomplete(t *testing.T) {
	verbs := []Verb{{Name: "serve", Tier: TierFrontdoor}, {Name: "issue", Tier: TierDev}}
	inventory := []CommandOwnership{
		{Name: "serve", Owner: OwnerRuntime, Rationale: "runtime", CompatibilityName: "serve", DispatchTarget: "fak"},
		{Name: "serve", Owner: OwnerRuntime, Rationale: "runtime", CompatibilityName: "serve", DispatchTarget: "fak"},
		{Name: "ghost", Owner: "elsewhere"},
	}
	got := strings.Join(ValidateCommandOwnership(verbs, inventory), "\n")
	for _, want := range []string{"duplicate command: serve", "missing command: issue", "unknown command: ghost", "invalid owner: ghost", "missing rationale: ghost"} {
		if !strings.Contains(got, want) {
			t.Errorf("problems missing %q:\n%s", want, got)
		}
	}
}

func TestBuildGraphReportFindsDirectAndTransitiveLeaks(t *testing.T) {
	nodes := []ImportNode{
		{ImportPath: "example/runtime", Imports: []string{"example/bridge", "example/devdirect"}},
		{ImportPath: "example/bridge", Imports: []string{"example/devtransitive"}},
		{ImportPath: "example/devdirect"},
		{ImportPath: "example/devtransitive"},
	}
	packages := []PackageOwnership{
		{Path: "example/devdirect", Owner: OwnerDev, Rationale: "negative control"},
		{Path: "example/devtransitive", Owner: OwnerDev, Rationale: "negative control"},
	}
	report := BuildGraphReport("example/runtime", nodes, packages)
	if len(report.Leaks) != 2 {
		t.Fatalf("leaks = %+v, want direct and transitive", report.Leaks)
	}
	want := map[string]string{
		"example/devdirect":     "example/runtime -> example/devdirect",
		"example/devtransitive": "example/runtime -> example/bridge -> example/devtransitive",
	}
	for _, leak := range report.Leaks {
		if got := strings.Join(leak.Path, " -> "); got != want[leak.Forbidden] {
			t.Errorf("path to %s = %s, want %s", leak.Forbidden, got, want[leak.Forbidden])
		}
	}
}

func TestRuntimeGraphWitnessReportsCurrentDevLeaks(t *testing.T) {
	root := FindRoot(".")
	nodes, err := LoadImportGraph(root, "./cmd/fak")
	if err != nil {
		t.Fatal(err)
	}
	report := BuildGraphReport("github.com/anthony-chaudhary/fak/cmd/fak", nodes, DevOnlyPackages)
	// #6020 is the baseline gate, not the migration: current leaks must remain
	// visible until #6022 drives this count to zero.
	if len(report.Leaks) == 0 {
		t.Fatal("runtime graph unexpectedly has zero dev-only leaks; update #6020 baseline and ratchet")
	}
	for _, leak := range report.Leaks {
		if len(leak.Path) < 2 || leak.Path[0] != report.Root || leak.Path[len(leak.Path)-1] != leak.Forbidden {
			t.Errorf("invalid witnessed path: %+v", leak)
		}
	}
}
