package devindex

import (
	"path/filepath"
	"strings"
	"testing"
)

// writeScopePlanRepo builds a synthetic checkout whose lane taxonomy exercises
// every branch the planner has: a single-tree lane, a multi-tree lane, a lane
// that also owns a bare file, and a lane that owns ONLY a bare file (so it has
// no package directory to plan against).
func writeScopePlanRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dosToml := `[lanes.trees]
gateway     = ["internal/gateway/**"]
multi       = ["internal/multi/**", "internal/other/**"]
cmd         = ["cmd/**"]
docs        = ["docs/**", "README.md"]
notes-only  = ["NOTES.md"]
`
	mustWrite(t, root, "dos.toml", dosToml)
	for _, dir := range []string{"internal/gateway", "internal/multi", "internal/other", "cmd/fak", "docs"} {
		mustMkdir(t, root, filepath.FromSlash(dir))
		mustWrite(t, filepath.Join(root, filepath.FromSlash(dir)), "x.go", "package x\n")
	}
	mustMkdir(t, root, "internal", "architest")
	mustWrite(t, filepath.Join(root, "internal", "architest"), "architest_test.go", `package architest

var tier = map[string]int{
	"gateway": 4,
	"multi":   2,
}
`)
	return root
}

func loadScopePlanCatalog(t *testing.T) *Catalog {
	t.Helper()
	c, err := Load(writeScopePlanRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return c
}

// TestScopePlanResolvesLanePathAndPackage is the acceptance case: the three
// scope spellings a human actually types all resolve to a bounded plan.
func TestScopePlanResolvesLanePathAndPackage(t *testing.T) {
	c := loadScopePlanCatalog(t)

	cases := []struct {
		name     string
		scope    string
		wantKind string
		wantLane string
		wantPkgs []string
	}{
		{"lane", "gateway", "lane", "gateway", []string{"./internal/gateway"}},
		{"path", "internal/gateway/server.go", "path", "gateway", []string{"./internal/gateway"}},
		{"package", "./internal/gateway", "path", "gateway", []string{"./internal/gateway"}},
		{"cmd lane", "cmd", "lane", "cmd", []string{"./cmd"}},
		{"lane owning a bare file too", "docs", "lane", "docs", []string{"./docs"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			plan := c.PlanScope(tc.scope)
			if !plan.Resolved() {
				t.Fatalf("PlanScope(%q) refused: %+v", tc.scope, plan.Refusal)
			}
			if plan.Schema != ScopePlanSchema {
				t.Fatalf("schema = %q, want %q", plan.Schema, ScopePlanSchema)
			}
			if plan.Kind != tc.wantKind {
				t.Fatalf("kind = %q, want %q", plan.Kind, tc.wantKind)
			}
			if plan.Lane != tc.wantLane {
				t.Fatalf("lane = %q, want %q", plan.Lane, tc.wantLane)
			}
			if strings.Join(plan.Packages, " ") != strings.Join(tc.wantPkgs, " ") {
				t.Fatalf("packages = %v, want %v", plan.Packages, tc.wantPkgs)
			}
			// Every emitted command must name exactly the planned packages, and
			// the honesty list must never be empty for a resolved plan.
			for _, pkg := range tc.wantPkgs {
				for _, want := range []string{"go build " + pkg, "go vet " + pkg, "go test " + pkg} {
					if !containsString(plan.Build, want) && !containsString(plan.Vet, want) && !containsString(plan.Test, want) {
						t.Fatalf("plan for %q is missing command %q (build=%v vet=%v test=%v)",
							tc.scope, want, plan.Build, plan.Vet, plan.Test)
					}
				}
			}
			if len(plan.DoesNotProve) == 0 {
				t.Fatalf("plan for %q carries an empty does_not_prove list", tc.scope)
			}
			if !strings.Contains(strings.Join(plan.DoesNotProve, "\n"), tc.wantPkgs[0]) {
				t.Fatalf("does_not_prove for %q does not name its own boundary %q: %v",
					tc.scope, tc.wantPkgs[0], plan.DoesNotProve)
			}
		})
	}
}

// TestScopePlanPathScopeDoesNotWidenToItsLane pins the property that makes a
// path scope useful: naming one file must not silently commit the caller to
// every package the lane owns.
func TestScopePlanPathScopeDoesNotWidenToItsLane(t *testing.T) {
	c := loadScopePlanCatalog(t)
	plan := c.PlanScope("internal/multi/x.go")
	if !plan.Resolved() {
		t.Fatalf("refused: %+v", plan.Refusal)
	}
	if plan.Lane != "multi" {
		t.Fatalf("lane = %q, want multi", plan.Lane)
	}
	if len(plan.Packages) != 1 || plan.Packages[0] != "./internal/multi" {
		t.Fatalf("path scope widened: packages = %v, want [./internal/multi]", plan.Packages)
	}
	if len(plan.LaneTree) != 2 {
		t.Fatalf("lane_tree = %v, want both declared globs", plan.LaneTree)
	}
	// The lane scope, by contrast, covers both trees.
	lanePlan := c.PlanScope("multi")
	if strings.Join(lanePlan.Packages, " ") != "./internal/multi ./internal/other" {
		t.Fatalf("lane scope packages = %v, want both trees", lanePlan.Packages)
	}
}

// TestScopePlanRefusesUnbounded is the safety case: "validate everything" is
// refused rather than expanded, so the whole-module commitment stays visible.
func TestScopePlanRefusesUnbounded(t *testing.T) {
	c := loadScopePlanCatalog(t)
	for _, scope := range []string{"./...", "...", "**", "**/*", "", ".", "all", "*"} {
		plan := c.PlanScope(scope)
		if plan.Refusal == nil {
			t.Fatalf("PlanScope(%q) resolved instead of refusing: %+v", scope, plan)
		}
		if plan.Refusal.Code != ScopeUnbounded {
			t.Fatalf("PlanScope(%q) code = %q, want %q", scope, plan.Refusal.Code, ScopeUnbounded)
		}
		if plan.Resolved() {
			t.Fatalf("PlanScope(%q) reports Resolved() on a refusal", scope)
		}
		if len(plan.Build)+len(plan.Vet)+len(plan.Test) != 0 {
			t.Fatalf("PlanScope(%q) emitted commands alongside a refusal", scope)
		}
	}
}

// TestScopePlanRefusesUndeclaredLane covers both ways a scope can fail to bind
// to a declared leaf: a name dos.toml never declares, and a declared lane that
// owns no package directory at all.
func TestScopePlanRefusesUndeclaredLane(t *testing.T) {
	c := loadScopePlanCatalog(t)
	for _, scope := range []string{"nosuchlane", "internal/nosuchpkg/x.go", "notes-only"} {
		plan := c.PlanScope(scope)
		if plan.Refusal == nil {
			t.Fatalf("PlanScope(%q) resolved instead of refusing: %+v", scope, plan)
		}
		if plan.Refusal.Code != ScopeUndeclaredLane {
			t.Fatalf("PlanScope(%q) code = %q, want %q", scope, plan.Refusal.Code, ScopeUndeclaredLane)
		}
		if plan.Refusal.Detail == "" {
			t.Fatalf("PlanScope(%q) refusal carries no detail", scope)
		}
	}
}

// TestScopePlanGlobScopeResolvesToItsPackage covers the glob spelling an agent
// actually types (the same one `fak-dev orient --paths` accepts): a glob must
// land on the package directory, not on a literal path containing `*`.
func TestScopePlanGlobScopeResolvesToItsPackage(t *testing.T) {
	c := loadScopePlanCatalog(t)
	for _, tc := range []struct{ scope, wantPkg string }{
		{"internal/gateway/**", "./internal/gateway"},
		{"cmd/fak/*.go", "./cmd/fak"},
		{"internal/multi/**", "./internal/multi"},
	} {
		plan := c.PlanScope(tc.scope)
		if !plan.Resolved() {
			t.Fatalf("PlanScope(%q) refused: %+v", tc.scope, plan.Refusal)
		}
		if len(plan.Packages) != 1 || plan.Packages[0] != tc.wantPkg {
			t.Fatalf("PlanScope(%q) packages = %v, want [%s]", tc.scope, plan.Packages, tc.wantPkg)
		}
	}
}

// TestOrientCarriesTheScopePlan is the reader-facing half: the plan must reach
// the surface a human already uses (`fak-dev orient --json`) without a new verb.
func TestOrientCarriesTheScopePlan(t *testing.T) {
	c := loadScopePlanCatalog(t)
	rows := c.Orient([]string{"internal/gateway/server.go", "nosuchlane"}, nil)
	if len(rows) != 2 {
		t.Fatalf("Orient rows = %d, want 2", len(rows))
	}
	if rows[0].Plan == nil {
		t.Fatal("orient row for a resolvable path carries no plan")
	}
	if !rows[0].Plan.Resolved() || rows[0].Plan.Packages[0] != "./internal/gateway" {
		t.Fatalf("orient plan = %+v, want a resolved ./internal/gateway plan", rows[0].Plan)
	}
	if len(rows[0].Plan.DoesNotProve) == 0 {
		t.Fatal("orient plan carries an empty does_not_prove list")
	}
	// A path that resolves to no package must carry no plan rather than a
	// half-filled one: the reader branches on presence, not on prose.
	if rows[1].Plan != nil {
		t.Fatalf("orient row for an unresolvable path carries a plan: %+v", rows[1].Plan)
	}
}

// TestScopePlanCarriesTierAndClosedVocabulary pins the two facts a caller
// branches on without reading prose: the architest tier, and the fact that the
// refusal vocabulary is closed (every code is one of the two declared values).
func TestScopePlanCarriesTierAndClosedVocabulary(t *testing.T) {
	c := loadScopePlanCatalog(t)
	plan := c.PlanScope("gateway")
	if plan.Tier == nil || *plan.Tier != 4 || plan.TierName != "integrator" {
		t.Fatalf("tier = %v %q, want 4 integrator", plan.Tier, plan.TierName)
	}
	closed := map[ScopeRefusalCode]bool{ScopeUnbounded: true, ScopeUndeclaredLane: true}
	for _, scope := range []string{"gateway", "./...", "nosuchlane", "notes-only", "cmd"} {
		if r := c.PlanScope(scope).Refusal; r != nil && !closed[r.Code] {
			t.Fatalf("PlanScope(%q) emitted an out-of-vocabulary code %q", scope, r.Code)
		}
	}
}
