package devindex

import (
	"encoding/json"
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
		if got := byName[name]; got.Owner != OwnerRuntime || got.DispatchTarget != "fak" || got.DevReuse != DevReuseNA {
			t.Errorf("%s ownership = %+v, want runtime/fak/not-applicable", name, got)
		}
	}
	for _, name := range []string{"issue", "commit", "ci-preflight", "release", "fleet", "bench"} {
		if got := byName[name]; got.Owner != OwnerDev || got.DispatchTarget != "fak-dev" || got.DevReuse == DevReuseNA {
			t.Errorf("%s ownership = %+v, want classified dev/fak-dev", name, got)
		}
	}
}

func TestValidateCommandOwnershipRejectsMissingDuplicateUnknownAndIncomplete(t *testing.T) {
	verbs := []Verb{{Name: "serve", Tier: TierFrontdoor}, {Name: "issue", Tier: TierDev}}
	inventory := []CommandOwnership{
		{Name: "serve", Owner: OwnerRuntime, Rationale: "runtime", CompatibilityName: "serve", DispatchTarget: "fak", DevReuse: DevReuseNA, DevReuseRationale: "runtime"},
		{Name: "serve", Owner: OwnerRuntime, Rationale: "runtime", CompatibilityName: "serve", DispatchTarget: "fak", DevReuse: DevReuseNA, DevReuseRationale: "runtime"},
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
	// #6022 is the migration ratchet: cmd/fak must never regain a path to a
	// package explicitly declared development-only.
	if len(report.Leaks) != 0 {
		t.Fatalf("runtime graph has %d dev-only leak(s): %+v", len(report.Leaks), report.Leaks)
	}
}

func TestDevReuseSeparatesPortablePatternsFromFakInternals(t *testing.T) {
	cat, err := Load(FindRoot("."))
	if err != nil {
		t.Fatal(err)
	}
	verbs := OwnershipVerbs(cat.Verbs())
	byName := make(map[string]CommandOwnership, len(verbs))
	for _, item := range CommandOwnerships(verbs) {
		byName[item.Name] = item
	}
	for name, want := range map[string]DevReuse{
		"serve":          DevReuseNA,
		"commit":         DevReusePortable,
		"worktree":       DevReusePortable,
		"validate":       DevReusePortable,
		"release":        DevReuseMaintainer,
		"index":          DevReuseMaintainer,
		"lab":            DevReuseLab,
		"fleet-accounts": DevReuseLab,
	} {
		got, ok := byName[name]
		if !ok {
			t.Fatalf("missing command %q", name)
		}
		if got.DevReuse != want || got.DevReuseRationale == "" {
			t.Errorf("%s dev reuse = %q (%q), want %q with rationale", name, got.DevReuse, got.DevReuseRationale, want)
		}
	}
}

func TestCommandOwnershipJSONCarriesReuseAxis(t *testing.T) {
	item := CommandOwnerships([]Verb{{Name: "commit", Tier: TierDev}})[0]
	b, err := json.Marshal(item)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"dev_reuse":"portable-pattern"`, `"dev_reuse_rationale":`} {
		if !strings.Contains(string(b), want) {
			t.Errorf("ownership JSON missing %s: %s", want, b)
		}
	}
}

func TestValidateCommandOwnershipRejectsInvalidDevReuse(t *testing.T) {
	verbs := []Verb{{Name: "serve", Tier: TierFrontdoor}, {Name: "commit", Tier: TierDev}}
	inventory := []CommandOwnership{
		{Name: "serve", Owner: OwnerRuntime, Rationale: "runtime", CompatibilityName: "serve", DispatchTarget: "fak", DevReuse: DevReusePortable, DevReuseRationale: "wrong axis"},
		{Name: "commit", Owner: OwnerDev, Rationale: "dev", CompatibilityName: "commit", DispatchTarget: "fak-dev", DevReuse: DevReuseNA},
	}
	got := strings.Join(ValidateCommandOwnership(verbs, inventory), "\n")
	for _, want := range []string{"non-dev command \"serve\" has dev reuse", "empty dev reuse rationale", "dev command \"commit\" has not-applicable"} {
		if !strings.Contains(got, want) {
			t.Errorf("problems missing %q:\n%s", want, got)
		}
	}
}
