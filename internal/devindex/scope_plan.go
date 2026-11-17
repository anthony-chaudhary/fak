package devindex

import (
	"path"
	"sort"
	"strings"
)

// ScopePlanSchema is the wire schema for a bounded sub-section validation plan.
const ScopePlanSchema = "fak.scope-plan.v1"

// ScopeRefusalCode is the closed vocabulary for refusing to plan a scope. A
// caller must be able to tell a safety refusal from a parse failure without
// reading prose, so every refusal carries one of these and nothing else.
type ScopeRefusalCode string

const (
	// ScopeUnbounded refuses a scope that names the whole module. "Validate
	// everything" is exactly the commitment this planner exists to avoid, so it
	// is refused rather than silently expanded into a full-tree plan.
	ScopeUnbounded ScopeRefusalCode = "SCOPE_UNBOUNDED"
	// ScopeUndeclaredLane refuses a scope whose lane has no declared leaf. An
	// undeclared leaf is not a cosmetic gap: laneadmit.Decide falls back to the
	// lane's declared tree when a request carries none, and an empty tree
	// conservatively overlaps EVERYTHING — so a plan built on an undeclared lane
	// would claim a boundary the arbiter does not honor.
	ScopeUndeclaredLane ScopeRefusalCode = "SCOPE_UNDECLARED_LANE"
)

// ScopeRefusal is one closed-vocabulary refusal, with the detail that produced it.
type ScopeRefusal struct {
	Code   ScopeRefusalCode `json:"code"`
	Detail string           `json:"detail"`
}

// ScopePlan is the read-only answer to "what is the smallest thing I can run to
// validate this sub-section, and what does that NOT prove?". It is a pure
// projection over the already-loaded Catalog: no I/O, no clock, no subprocess.
//
// The plan is deliberately a *plan*, not an execution: it names the commands a
// human or agent would run and the boundary those commands do not cross. Nothing
// in this surface runs anything.
type ScopePlan struct {
	Schema   string   `json:"schema"`
	Scope    string   `json:"scope"`
	Kind     string   `json:"kind"` // lane | path
	Lane     string   `json:"lane,omitempty"`
	LaneTree []string `json:"lane_tree,omitempty"`
	Tier     *int     `json:"tier,omitempty"`
	TierName string   `json:"tier_name,omitempty"`
	// Packages are the repo-relative package directories the scope covers. They
	// are derived from the lane's declared tree globs, never from a filesystem
	// walk, so the plan cannot silently widen past the declared boundary.
	Packages []string `json:"packages,omitempty"`
	// Build, Vet and Test are the minimal commands for exactly those packages.
	Build []string `json:"build,omitempty"`
	Vet   []string `json:"vet,omitempty"`
	Test  []string `json:"test,omitempty"`
	// DoesNotProve is the honesty list: what a green run of the above leaves
	// unproven. It is never empty for a resolved plan.
	DoesNotProve []string      `json:"does_not_prove"`
	Refusal      *ScopeRefusal `json:"refusal,omitempty"`
}

// Resolved reports whether the plan carries a runnable scope rather than a refusal.
func (p ScopePlan) Resolved() bool { return p.Refusal == nil && len(p.Packages) > 0 }

// unboundedScopes is the closed set of spellings that mean "the whole module".
// They are refused, not expanded: the point of the planner is to make the
// whole-module commitment explicit and avoidable.
var unboundedScopes = map[string]bool{
	"": true, ".": true, "./": true, "/": true, "**": true, "**/*": true,
	"./...": true, "...": true, "all": true, "*": true,
}

// PlanScope resolves one scope token — a lane name, a repo-relative path, or a
// package directory — to a bounded build+vet+test plan with an explicit
// does_not_prove list. It refuses SCOPE_UNBOUNDED for the whole module and
// SCOPE_UNDECLARED_LANE for a scope whose lane has no declared leaf.
func (c *Catalog) PlanScope(scope string) ScopePlan {
	raw := strings.TrimSpace(scope)
	plan := ScopePlan{Schema: ScopePlanSchema, Scope: raw}

	if unboundedScopes[strings.ToLower(raw)] {
		plan.Refusal = &ScopeRefusal{
			Code:   ScopeUnbounded,
			Detail: "scope names the whole module; name a lane, path, or package instead",
		}
		plan.DoesNotProve = []string{"nothing: an unbounded scope was refused, so no plan was produced"}
		return plan
	}

	norm := normPath(raw)
	lane := ""
	if strings.Contains(norm, "/") || strings.HasSuffix(norm, ".go") {
		plan.Kind = "path"
		lane = c.LaneForPath(norm)
	} else {
		plan.Kind = "lane"
		lane = strings.ToLower(norm)
	}

	leaf, ok := c.LeafByName(lane)
	if lane == "" || !ok {
		plan.Refusal = &ScopeRefusal{
			Code:   ScopeUndeclaredLane,
			Detail: "scope resolves to lane " + quoteLane(lane) + " which has no declared [lanes.trees] leaf",
		}
		plan.DoesNotProve = []string{"nothing: an undeclared lane was refused, so no plan was produced"}
		return plan
	}

	plan.Lane = leaf.Name
	plan.LaneTree = splitLaneTree(leaf.Tree)
	if tier, ok := c.TierForPath(leaf.Dir); ok {
		t := tier
		plan.Tier = &t
		plan.TierName = TierName(tier)
	}
	plan.Packages = scopePackages(plan.LaneTree, norm, plan.Kind)
	if len(plan.Packages) == 0 {
		plan.Refusal = &ScopeRefusal{
			Code:   ScopeUndeclaredLane,
			Detail: "lane " + leaf.Name + " declares no package directory to plan against",
		}
		plan.DoesNotProve = []string{"nothing: the lane declares no package directory, so no plan was produced"}
		return plan
	}
	for _, pkg := range plan.Packages {
		plan.Build = append(plan.Build, "go build "+pkg)
		plan.Vet = append(plan.Vet, "go vet "+pkg)
		plan.Test = append(plan.Test, "go test "+pkg)
	}
	plan.DoesNotProve = scopeDoesNotProve(plan)
	return plan
}

// scopePackages derives the package directories a scope covers. For a lane it is
// every subtree glob the lane declares; for a path it is the single package
// directory that path sits in, so a path scope can never widen to its whole lane.
func scopePackages(laneTree []string, norm, kind string) []string {
	if kind == "path" {
		dir := norm
		// A glob names a set of files, not a package directory: strip the glob
		// tail first so `internal/gateway/**` and `cmd/fak/*.go` both land on the
		// directory that actually holds the package.
		for _, tail := range []string{"/**", "/*", "*"} {
			if strings.HasSuffix(dir, tail) {
				dir = strings.TrimSuffix(dir, tail)
				break
			}
		}
		if strings.HasSuffix(dir, ".go") {
			dir = path.Dir(dir)
		}
		dir = strings.TrimSuffix(dir, "/")
		if dir == "" || dir == "." || strings.Contains(dir, "*") {
			return nil
		}
		return []string{"./" + dir}
	}
	seen := map[string]bool{}
	var out []string
	for _, glob := range laneTree {
		// Only a subtree glob names a package directory. A bare file entry
		// ("README.md") is a real lane member but not a buildable package, so
		// admitting it would emit `go build ./README.md` — a command that cannot
		// succeed. Skipping it keeps every emitted command runnable.
		if !strings.HasSuffix(glob, "**") {
			continue
		}
		g := strings.TrimSuffix(strings.TrimSuffix(glob, "**"), "/")
		if g == "" || strings.Contains(g, "*") {
			continue
		}
		pkg := "./" + g
		if seen[pkg] {
			continue
		}
		seen[pkg] = true
		out = append(out, pkg)
	}
	sort.Strings(out)
	return out
}

// scopeDoesNotProve is the honesty list for a resolved plan. It is built from the
// plan itself so the boundary it names is the boundary it actually has.
func scopeDoesNotProve(plan ScopePlan) []string {
	out := []string{
		"the whole-module build: a green run here does not prove `go build ./...` succeeds",
		"packages outside " + strings.Join(plan.Packages, ", ") + ": their tests are not run",
		"cross-lane integration: a peer lane's unlanded change to a shared dependency is not exercised",
		"hardware, accelerator, and serving paths: no device, model, or network is touched",
	}
	if plan.Tier != nil {
		out = append(out, "architest tier "+plan.TierName+" import rules beyond this package's own edges")
	}
	return out
}

func quoteLane(lane string) string {
	if lane == "" {
		return "(none)"
	}
	return lane
}
