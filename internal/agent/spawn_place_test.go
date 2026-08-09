package agent

// spawn_place_test.go — witnesses the per-spawn placement seam (#5420, epic #5416
// track E): a tool call that CREATES delegated work takes its OWN rung down the
// roster's ladder instead of inheriting the engine the parent turn was routed to.
//
// The defect this pins is not only a bill. Inheritance is a FLOOR BYPASS: the child
// never goes through Place, so the parent's floor — computed for the PARENT's class —
// is the only one that ever ran. So the tests assert BOTH directions, and the pair is
// the point:
//
//   - a routine child does NOT stay on the parent's vendor engine (the cost half), and
//   - a security/release child does NOT drop to the cheap device rung (the safety half).
//
// The contrast is asserted against the OLD path in the same test rather than as a bare
// expected value: resolveToolEngine is still present and still answers with the
// inherited, parent-shaped route, so each case pins the exact behaviour change and a
// revert of loop.go's call site cannot leave these green.

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// The three rungs of the ladder, as engine routes. These are the literal values
// Target.EngineRoute() produces, hardcoded rather than recomputed so the test pins the
// whole chain (class -> tier floor -> rung -> account -> wire id) instead of restating
// the production expression.
const (
	deviceRoute = "local:laptop/qwen3.6-4b"          // "tiny", the engineer's own machine
	fleetRoute  = "fleet:corp-glm/glm-5.2"           // "corp-mid", a machine the org operates
	vendorRoute = "anthropic:frontier/claude-opus-5" // "frontier", a third-party lab
)

// spawnPlaceRoster is the three-zone ladder an operator with capability evidence would
// declare, plus the spawn_classes block that is the only thing allowed to give a
// delegated turn a work class (spawnclass.go). "explore" does bounded lookup work;
// "release-cutter" can push and delete, so its floor never drops to T2.
func spawnPlaceRoster() *modelroute.Roster {
	return &modelroute.Roster{
		Accounts: []modelroute.Account{
			{ID: "laptop", Kind: modelroute.KindLocal, BaseURL: "http://127.0.0.1:11434/v1"},
			{ID: "corp-glm", Kind: modelroute.KindFleet, BaseURL: "http://glm.infer.corp.internal:8000/v1"},
			{ID: "frontier", Kind: modelroute.KindAnthropic, CredEnv: "ANTHROPIC_API_KEY"},
		},
		Bindings: []modelroute.Binding{
			{Model: "tiny", Account: "laptop", UpstreamModel: "qwen3.6-4b"},
			{Model: "corp-mid", Account: "corp-glm", UpstreamModel: "glm-5.2"},
			{Model: "frontier", Account: "frontier", UpstreamModel: "claude-opus-5"},
		},
		Default: "laptop",
		SpawnClasses: []modelroute.SpawnClass{
			{Type: "explore", Class: modelroute.ClassRoutine},
			{Type: "release-cutter", Class: modelroute.ClassSecurityRelease},
		},
	}
}

// spawnPlacePool is the candidate ladder, every rung MEASURED — an unmeasured candidate
// enters Admit at the most demanding tier, which would make these cases pass for the
// wrong reason.
func spawnPlacePool() []modelroute.Candidate {
	return []modelroute.Candidate{
		{Model: "tiny", Capability: modelroute.TierT2, Measured: true},
		{Model: "corp-mid", Capability: modelroute.TierT1, Measured: true},
		{Model: "frontier", Capability: modelroute.TierT0, Measured: true},
	}
}

// inheritManifest is the parent-shaped decision this issue exists to stop inheriting: a
// manifest that sends the spawn tools themselves to the frontier vendor, exactly as a
// spawn call inherits today by being just another routed tool call.
func inheritManifest() *modelroute.Manifest {
	return &modelroute.Manifest{
		Version: modelroute.Version,
		Default: modelroute.Plan{Members: []modelroute.Member{{Model: "frontier", Role: "primary"}}},
	}
}

// armedOnVendorParent is the loop as an operator arms it: the manifest above, the roster
// that resolves it, and spawn placement armed from a parent that landed on the vendor
// rung. The parent is the hard case — if a parent could move a child's rung at all,
// delegation would ratchet and one frontier turn near the root would pin its subtree.
func armedOnVendorParent() []RunOption {
	r := spawnPlaceRoster()
	return []RunOption{
		WithRouteManifest(inheritManifest()),
		WithRouteAccounts(r),
		WithSpawnPlacement(SpawnPlacementPolicy{
			Parent:     modelroute.Placement{Model: "frontier", Zone: modelroute.ZoneVendor, Measured: true},
			Candidates: spawnPlacePool(),
		}),
	}
}

// TestSpawnPlacementReplacesTheInheritedRoute is the repro and the fix in one assertion
// pair: the SAME spawn call routes to the vendor under the old per-tool-call path and to
// the engineer's own machine under the placement path. The first half is the defect
// (#5420: "sending them to a frontier vendor because the parent happened to be there"),
// and it is asserted rather than assumed so the second half cannot pass vacuously.
func TestSpawnPlacementReplacesTheInheritedRoute(t *testing.T) {
	cfg := resolveRunConfig(armedOnVendorParent())
	const args = `{"subagent_type":"explore","prompt":"sweep the repo for TODOs"}`

	// The DEFECT, still reachable through the old path: a spawn is just another routed
	// tool call, so it inherits the vendor engine the manifest names.
	inherited, err := cfg.resolveToolEngine("Agent")
	if err != nil {
		t.Fatalf("resolveToolEngine(Agent) = error %v, want the inherited route", err)
	}
	if inherited != vendorRoute {
		t.Fatalf("resolveToolEngine(Agent) = %q, want %q — the test's premise is that the OLD path inherits the vendor rung; if this changed, the contrast below is vacuous", inherited, vendorRoute)
	}

	// The FIX: the seam loop.go now calls gives the child its own walk down the ladder
	// for its OWN declared class (routine => floor T2 => the cheapest rung that serves).
	placed, err := cfg.resolveCallEngine("Agent", args)
	if err != nil {
		t.Fatalf("resolveCallEngine(Agent, explore) = error %v, want a placement", err)
	}
	if placed != deviceRoute {
		t.Fatalf("resolveCallEngine(Agent, explore) = %q, want %q — a routine sub-agent must get its own rung, not the parent's vendor engine (#5420)", placed, deviceRoute)
	}
	if placed == inherited {
		t.Fatal("spawn placement returned the inherited route: the child never got its own decision")
	}

	// The whole SpawnPlacement is available to a preflight witness, and it records the
	// event the epic counts rather than only the engine id.
	sp, ok, err := ResolveSpawnPlacement("Agent", args, armedOnVendorParent()...)
	if err != nil || !ok {
		t.Fatalf("ResolveSpawnPlacement = %v/%v, want a placement", ok, err)
	}
	if sp.Placement.Model != "tiny" || sp.Placement.Zone != modelroute.ZoneDevice {
		t.Fatalf("placed on %q/%s, want tiny/device", sp.Placement.Model, sp.Placement.Zone)
	}
	if !sp.Descended {
		t.Fatal("a vendor parent's routine child landed on the device rung but Descended is false — the epic counts this event per spawn")
	}
	if sp.ParentModel != "frontier" || sp.ParentZone != modelroute.ZoneVendor {
		t.Fatalf("parent recorded as %q/%s, want frontier/vendor (recorded as provenance, never obeyed)", sp.ParentModel, sp.ParentZone)
	}
}

// TestSpawnPlacementHoldsTheFloorForDestructiveWork is the safety half, and it is why
// this seam is worth wiring even where the delegated token share is small. TierPolicy
// fixes the floor by the WORK: a release-cutting child requires T1 however small the
// request looks, so it must NOT reach the device rung its sibling just got.
func TestSpawnPlacementHoldsTheFloorForDestructiveWork(t *testing.T) {
	cfg := resolveRunConfig(armedOnVendorParent())

	placed, err := cfg.resolveCallEngine("Agent", `{"subagent_type":"release-cutter"}`)
	if err != nil {
		t.Fatalf("resolveCallEngine(Agent, release-cutter) = error %v, want a placement", err)
	}
	if placed == deviceRoute {
		t.Fatal("a security/release child was placed on the device rung: its floor never drops to T2 (a cheaper rung must not be a weaker floor)")
	}
	if placed != fleetRoute {
		t.Fatalf("resolveCallEngine(Agent, release-cutter) = %q, want %q — the cheapest rung that still clears the T1 floor", placed, fleetRoute)
	}
}

// TestUndeclaredSpawnTypeKeepsTodaysRoute pins the ONE quiet path, which is a contract
// rather than a fallback: an operator who never classified an agent type gets today's
// behaviour, not a guess. Being wrong in this direction costs a spawn that stays where
// it already was; guessing would put unclassified work on a laptop.
func TestUndeclaredSpawnTypeKeepsTodaysRoute(t *testing.T) {
	cfg := resolveRunConfig(armedOnVendorParent())

	for _, typ := range []string{"code-reviewer", "expl", "explore-and-delete", ""} {
		args := `{"subagent_type":"` + typ + `"}`
		got, err := cfg.resolveCallEngine("Agent", args)
		if err != nil {
			t.Fatalf("undeclared type %q: resolveCallEngine = error %v, want today's route", typ, err)
		}
		if got != vendorRoute {
			t.Fatalf("undeclared type %q: resolveCallEngine = %q, want the unchanged route %q", typ, got, vendorRoute)
		}
		if _, placed, err := ResolveSpawnRoute("Agent", args, armedOnVendorParent()...); placed || err != nil {
			t.Fatalf("undeclared type %q: ResolveSpawnRoute placed=%v err=%v, want no placement", typ, placed, err)
		}
	}
}

// TestOnlySpawnToolsArePlaced guards the closed tool set. This harness ships TaskCreate,
// TaskUpdate and friends, which are todo-list bookkeeping and spawn nothing — a prefix
// match on "Task" would route them through a placement decision they have no business
// in, against a class an operator declared for delegated work.
func TestOnlySpawnToolsArePlaced(t *testing.T) {
	cfg := resolveRunConfig(armedOnVendorParent())
	const args = `{"subagent_type":"explore"}`

	for _, tool := range []string{"TaskCreate", "TaskUpdate", "TaskList", "TaskStop", "TaskOutput", "Read", "Bash"} {
		got, err := cfg.resolveCallEngine(tool, args)
		if err != nil {
			t.Fatalf("%s: resolveCallEngine = error %v, want the ordinary route", tool, err)
		}
		if got != vendorRoute {
			t.Fatalf("%s was placed (got %q, want the ordinary route %q) — only calls that CREATE delegated work take a placement", tool, got, vendorRoute)
		}
	}

	// And the spawners themselves ARE in the set, including the untyped background
	// workflow, which is half of what #5420 names.
	for _, tool := range []string{"Task", "Agent", "Workflow"} {
		if _, placed, err := ResolveSpawnRoute(tool, args, armedOnVendorParent()...); !placed || err != nil {
			t.Fatalf("%s: placed=%v err=%v, want a placement (it creates delegated work)", tool, placed, err)
		}
	}
}

// TestUnarmedLoopIsUnchanged: not arming the option leaves the loop byte-for-byte the
// historical one. A caller that never opts in must not observe a routing change.
func TestUnarmedLoopIsUnchanged(t *testing.T) {
	unarmed := []RunOption{WithRouteManifest(inheritManifest()), WithRouteAccounts(spawnPlaceRoster())}
	cfg := resolveRunConfig(unarmed)

	for _, tool := range []string{"Agent", "Task", "Workflow", "Read"} {
		viaCall, err := cfg.resolveCallEngine(tool, `{"subagent_type":"explore"}`)
		if err != nil {
			t.Fatalf("unarmed %s: resolveCallEngine = error %v", tool, err)
		}
		viaTool, err := cfg.resolveToolEngine(tool)
		if err != nil {
			t.Fatalf("unarmed %s: resolveToolEngine = error %v", tool, err)
		}
		if viaCall != viaTool {
			t.Fatalf("unarmed %s: resolveCallEngine = %q but resolveToolEngine = %q — an unarmed loop must route identically to the historical one", tool, viaCall, viaTool)
		}
	}
}

// TestArmedWithoutARosterFailsLoud: every failure here is loud except the undeclared
// type. A placement policy must never turn a misconfiguration into a silent fallback to
// the very inheritance it was wired to stop.
func TestArmedWithoutARosterFailsLoud(t *testing.T) {
	cfg := resolveRunConfig([]RunOption{
		WithRouteManifest(inheritManifest()),
		WithSpawnPlacement(SpawnPlacementPolicy{Candidates: spawnPlacePool()}), // no WithRouteAccounts
	})

	_, err := cfg.resolveCallEngine("Agent", `{"subagent_type":"explore"}`)
	if err == nil {
		t.Fatal("spawn placement armed with no roster resolved a route: a wiring error must fail loud, never fall back to inheriting the parent's engine")
	}
	if !strings.Contains(err.Error(), "no account roster is wired") {
		t.Fatalf("error %q does not name the missing roster — a refusal must say what to fix", err)
	}

	// An armed policy with a roster but NO candidates is also loud: the ladder can serve
	// nobody, and "no candidates" is a misconfiguration rather than a routine state.
	empty := resolveRunConfig([]RunOption{
		WithRouteManifest(inheritManifest()),
		WithRouteAccounts(spawnPlaceRoster()),
		WithSpawnPlacement(SpawnPlacementPolicy{}),
	})
	if _, err := empty.resolveCallEngine("Agent", `{"subagent_type":"explore"}`); err == nil {
		t.Fatal("an armed policy with an empty candidate pool resolved a route, want a loud refusal")
	}
}

// TestParentZoneIsAnInputNeverAFloor is the load-bearing invariant of track E, asserted
// at THIS seam rather than only in modelroute: the child's engine must be identical
// whatever rung the parent landed on. If a parent could move a child's placement,
// delegation would ratchet and the cheap rung would be unreachable under any real
// session — which is the whole behaviour #5420 removes.
func TestParentZoneIsAnInputNeverAFloor(t *testing.T) {
	const args = `{"subagent_type":"explore"}`

	parents := []modelroute.Placement{
		{}, // a root turn
		{Model: "tiny", Zone: modelroute.ZoneDevice, Measured: true},
		{Model: "corp-mid", Zone: modelroute.ZoneFleet, Measured: true},
		{Model: "frontier", Zone: modelroute.ZoneVendor, Measured: true},
	}
	for _, parent := range parents {
		cfg := resolveRunConfig([]RunOption{
			WithRouteManifest(inheritManifest()),
			WithRouteAccounts(spawnPlaceRoster()),
			WithSpawnPlacement(SpawnPlacementPolicy{Parent: parent, Candidates: spawnPlacePool()}),
		})
		got, err := cfg.resolveCallEngine("Agent", args)
		if err != nil {
			t.Fatalf("parent %s/%s: resolveCallEngine = error %v", parent.Zone, parent.Model, err)
		}
		if got != deviceRoute {
			t.Fatalf("parent %s/%s moved the child to %q, want %q — the parent is an input, never a floor", parent.Zone, parent.Model, got, deviceRoute)
		}
	}
}

// TestSpawnPlacementRefusesAnUnadmittedPrincipal: a cheaper rung is not a weaker tenancy
// boundary. A spawn must not be the one path that reaches an account this caller was
// never provisioned for (#5332), and the refusal names the principal and the account —
// never the credential.
func TestSpawnPlacementRefusesAnUnadmittedPrincipal(t *testing.T) {
	r := spawnPlaceRoster()
	r.Accounts[0].Principals = []string{"someone-else"} // scope the laptop to another tenant

	cfg := resolveRunConfig([]RunOption{
		WithRouteManifest(inheritManifest()),
		WithRouteAccounts(r),
		WithRoutePrincipal("this-tenant"),
		WithSpawnPlacement(SpawnPlacementPolicy{Candidates: spawnPlacePool()}),
	})

	_, err := cfg.resolveCallEngine("Agent", `{"subagent_type":"explore"}`)
	if err == nil {
		t.Fatal("a spawn placed onto an account the principal is not admitted to resolved a route, want a fail-closed refusal")
	}
	if !strings.Contains(err.Error(), "this-tenant") || !strings.Contains(err.Error(), "laptop") {
		t.Fatalf("refusal %q must name the principal and the account", err)
	}
	if strings.Contains(err.Error(), "ANTHROPIC_API_KEY") {
		t.Fatalf("refusal %q leaked a credential env name", err)
	}
}
