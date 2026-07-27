package dispatchtick

import (
	"reflect"
	"strings"
	"testing"
)

func TestIsSelfSourceTreeMatchesGoModuleRoots(t *testing.T) {
	selfSource := []string{
		"cmd/**",
		"cmd/fak/**",
		"internal/gateway/**",
		"internal/abi/**",
		"./cmd/fak/**",
		"fak/internal/agent/**",
		`internal\agent\**`, // a Windows-authored glob normalizes the same as POSIX
	}
	for _, g := range selfSource {
		if !IsSelfSourceTree(g) {
			t.Errorf("IsSelfSourceTree(%q) = false, want true (fak's own Go module source)", g)
		}
	}
	shippable := []string{"docs/**", "tools/**", "scripts/**", ".github/**", "examples/**", "visuals/**", ".claude/**", ""}
	for _, g := range shippable {
		if IsSelfSourceTree(g) {
			t.Errorf("IsSelfSourceTree(%q) = true, want false (not fak's Go module)", g)
		}
	}
}

func TestIsTrustCriticalTreeMatchesOnlyTheWitnessMachinery(t *testing.T) {
	// The trust-critical set: the referee's own trees + the witness gates + the taxonomy
	// files. These are what a guarded RSI worker must never SHIP an edit to.
	trustCritical := []string{
		"internal/abi/**",
		"internal/kernel/**",
		"internal/adjudicator/**",
		"internal/policy/**",
		"internal/registrations/**",
		"internal/architest/**",
		"internal/shipgate/**",
		"./internal/policy/decide.go",
		`fak\internal\adjudicator\**`,
		"dos.toml",
		".dos/state",
		"VERSION",
		"policy.json",
	}
	for _, g := range trustCritical {
		if !IsTrustCriticalTree(g) {
			t.Errorf("IsTrustCriticalTree(%q) = false, want true (trust-critical witness machinery)", g)
		}
	}
	// Merely-self-source trees the worker guard PERMITS shipping -- must NOT be trust-critical.
	shippable := []string{
		"cmd/**", "cmd/fak/**",
		"internal/gateway/**", "internal/agent/**", "internal/compute/**", "internal/metrics/**",
		"docs/**", "tools/**", "scripts/**", ".github/**", "examples/**", "",
	}
	for _, g := range shippable {
		if IsTrustCriticalTree(g) {
			t.Errorf("IsTrustCriticalTree(%q) = true, want false (guard-shippable, not the referee's own trees)", g)
		}
	}
}

func TestSelfModifyHoldOnlyHoldsGuardedTrustCriticalLanes(t *testing.T) {
	// Guarded worker + trust-critical lane tree -> held, naming the offending tree.
	if held, tree := SelfModifyHold(true, []string{"internal/adjudicator/**"}); !held || tree != "internal/adjudicator/**" {
		t.Fatalf("SelfModifyHold(true, [internal/adjudicator/**]) = (%v, %q), want held", held, tree)
	}
	if held, tree := SelfModifyHold(true, []string{"internal/policy/**"}); !held || tree != "internal/policy/**" {
		t.Fatalf("SelfModifyHold(true, [internal/policy/**]) = (%v, %q), want held", held, tree)
	}

	// Guarded worker + a merely-self-source lane the guard PERMITS -> NOT held. This is the
	// correction: cmd/** and internal/gateway are shippable under the real worker floor.
	for _, tree := range [][]string{{"cmd/**"}, {"internal/gateway/**"}, {"internal/agent/**"}} {
		if held, _ := SelfModifyHold(true, tree); held {
			t.Fatalf("SelfModifyHold(true, %v) held a guard-shippable self-source lane", tree)
		}
	}

	// Guarded worker + shippable non-source lane -> NOT held.
	if held, _ := SelfModifyHold(true, []string{"docs/**"}); held {
		t.Fatalf("SelfModifyHold(true, [docs/**]) held a shippable lane")
	}

	// Unguarded worker -> never held, even on trust-critical (the operator/worktree escape #1334).
	if held, _ := SelfModifyHold(false, []string{"internal/adjudicator/**"}); held {
		t.Fatalf("SelfModifyHold(false, [internal/adjudicator/**]) held an unguarded worker")
	}

	// A mixed tree holds on the first trust-critical member it finds.
	if held, tree := SelfModifyHold(true, []string{"docs/**", "internal/kernel/**"}); !held || tree != "internal/kernel/**" {
		t.Fatalf("SelfModifyHold(true, [docs/**, internal/kernel/**]) = (%v, %q), want held on internal/kernel/**", held, tree)
	}

	// No tree -> not held (nothing to protect).
	if held, _ := SelfModifyHold(true, nil); held {
		t.Fatalf("SelfModifyHold(true, nil) held with no tree")
	}
}

func TestIssueTextTargetsTrustCriticalCatchesBareAndPrefixedRefs(t *testing.T) {
	trustCritical := map[string]string{
		"the fix lives in `internal/adjudicator/**`":      "internal/adjudicator/**",
		"work in internal/policy/ where the loader lives": "internal/policy/",
		"see ./internal/kernel/decide.go":                 "./internal/kernel/decide.go",
		"touches fak/internal/shipgate/gate.go":           "fak/internal/shipgate/gate.go",
		"the internal/abi stream needs a fix":             "internal/abi",
	}
	for text, want := range trustCritical {
		held, tree := IssueTextTargetsTrustCritical(text)
		if !held || tree != want {
			t.Errorf("IssueTextTargetsTrustCritical(%q) = (%v, %q), want (true, %q)", text, held, tree, want)
		}
	}
	// A merely-self-source ref (cmd/**, internal/gateway) is guard-shippable and does NOT
	// match -- the dispatcher must not hold work the worker floor permits. Nor does a bare
	// mention without a trust-critical path.
	notHeld := []string{
		"most of the backlog lives in `cmd/**` + `internal/gateway/**`",
		"work in cmd/fak/ where the verb shell lives",
		"the internal/agent stream needs a fix",
		"first-class fak dispatch verb",
		"the subcommand/foo helper and internals/x are unrelated",
		"document the tools/ and docs/ lanes",
		"",
	}
	for _, text := range notHeld {
		if held, tree := IssueTextTargetsTrustCritical(text); held {
			t.Errorf("IssueTextTargetsTrustCritical(%q) = (true, %q), want not held", text, tree)
		}
	}
}

func TestLaneDispatchableUnderGuard(t *testing.T) {
	// Guarded: a trust-critical lane tree is NOT dispatchable; a merely-self-source or
	// shippable one IS.
	if LaneDispatchableUnderGuard(true, []string{"internal/adjudicator/**"}) {
		t.Fatalf("guarded internal/adjudicator lane reported dispatchable")
	}
	if LaneDispatchableUnderGuard(true, []string{"internal/policy/**"}) {
		t.Fatalf("guarded internal/policy lane reported dispatchable")
	}
	// The correction: gateway and cmd are guard-shippable, so they ARE dispatchable.
	if !LaneDispatchableUnderGuard(true, []string{"internal/gateway/**"}) {
		t.Fatalf("guarded internal/gateway lane reported NOT dispatchable (the guard permits it)")
	}
	if !LaneDispatchableUnderGuard(true, []string{"cmd/**"}) {
		t.Fatalf("guarded cmd lane reported NOT dispatchable (the guard permits it)")
	}
	if !LaneDispatchableUnderGuard(true, []string{"docs/**", "README.md"}) {
		t.Fatalf("guarded docs lane reported NOT dispatchable")
	}
	// A mixed tree with any trust-critical member is held.
	if LaneDispatchableUnderGuard(true, []string{"docs/**", "internal/kernel/**"}) {
		t.Fatalf("guarded mixed trust-critical lane reported dispatchable")
	}
	// Unguarded: every lane is dispatchable (the operator/worktree escape #1334).
	if !LaneDispatchableUnderGuard(false, []string{"internal/adjudicator/**"}) {
		t.Fatalf("unguarded trust-critical lane reported NOT dispatchable")
	}
	// No declared tree -> fail OPEN (no trust-critical witness to hold on).
	if !LaneDispatchableUnderGuard(true, nil) {
		t.Fatalf("guarded lane with no tree reported NOT dispatchable")
	}
}

func TestDispatchableLanesUnderGuardHoldsOnlyTheTrustCriticalReferee(t *testing.T) {
	// The corrected #1397 surface: the backlog routes mostly to internal/** lanes, but only
	// the REFEREE's own trees (adjudicator, policy, kernel, ...) are held -- gateway, compute,
	// metrics, model, and cmd are all guard-shippable and must surface. Before the narrowing
	// the whole module was held and the surface was starved of ~85% of the real work.
	trees := map[string][]string{
		"adjudicator": {"internal/adjudicator/**"},
		"policy":      {"internal/policy/**"},
		"kernel":      {"internal/kernel/**"},
		"compute":     {"internal/compute/**"},
		"gateway":     {"internal/gateway/**"},
		"metrics":     {"internal/metrics/**"},
		"model":       {"internal/model/**"},
		"cmd":         {"cmd/**"},
		"docs":        {"docs/**", "README.md"},
		"tools":       {"tools/**", "scripts/**"},
	}

	dispatchable, held := DispatchableLanesUnderGuard(true, trees)
	wantDispatchable := []string{"cmd", "compute", "docs", "gateway", "metrics", "model", "tools"}
	wantHeld := []string{"adjudicator", "kernel", "policy"}
	if !reflect.DeepEqual(dispatchable, wantDispatchable) {
		t.Fatalf("guarded dispatchable lanes = %v, want %v", dispatchable, wantDispatchable)
	}
	if !reflect.DeepEqual(held, wantHeld) {
		t.Fatalf("guarded held lanes = %v, want %v", held, wantHeld)
	}
	// The guarded plan surface is NON-EMPTY and carries the core backlog, not just docs/tools.
	if len(dispatchable) == 0 {
		t.Fatalf("guarded dispatch surface is EMPTY despite shippable core + docs/tools work")
	}

	// Unguarded: every lane is dispatchable and none is held (the operator escape #1334).
	allDispatchable, noneHeld := DispatchableLanesUnderGuard(false, trees)
	if len(allDispatchable) != len(trees) || len(noneHeld) != 0 {
		t.Fatalf("unguarded partition = %d dispatchable / %d held, want %d / 0", len(allDispatchable), len(noneHeld), len(trees))
	}
}

func TestSelfModifyHoldForPickCatchesMisroutedTrustCriticalIssue(t *testing.T) {
	// A guarded worker routed to a SAFE lane (tools) whose target issue's text targets the
	// trust-critical machinery is held -- the mis-route the lane tree alone hides.
	if held, tree := SelfModifyHoldForPick(true, []string{"tools/**", "scripts/**"}, "fix(policy): the work lives in `internal/policy/`"); !held || tree != "internal/policy/" {
		t.Fatalf("SelfModifyHoldForPick(tools lane, internal/policy issue text) = (%v, %q), want held on internal/policy/", held, tree)
	}
	// The lane-tree arm wins first and names the lane glob when the lane itself is trust-critical.
	if held, tree := SelfModifyHoldForPick(true, []string{"internal/adjudicator/**"}, "no path here"); !held || tree != "internal/adjudicator/**" {
		t.Fatalf("SelfModifyHoldForPick(trust-critical lane) = (%v, %q), want held on internal/adjudicator/**", held, tree)
	}
	// A safe lane + a merely-self-source issue (cmd/**) is NOT held -- the guard permits that ship.
	if held, _ := SelfModifyHoldForPick(true, []string{"tools/**"}, "the work lives in cmd/fak/main.go"); held {
		t.Fatalf("SelfModifyHoldForPick held a guard-shippable cmd pick")
	}
	// A safe lane + a shippable issue (no self-source ref) is NOT held.
	if held, _ := SelfModifyHoldForPick(true, []string{"docs/**"}, "update the README front door"); held {
		t.Fatalf("SelfModifyHoldForPick held a shippable docs pick")
	}
	// An UNGUARDED worker is never held, even when the issue text targets trust-critical trees.
	if held, _ := SelfModifyHoldForPick(false, []string{"tools/**"}, "edit internal/adjudicator/decide.go"); held {
		t.Fatalf("SelfModifyHoldForPick held an unguarded worker")
	}
}

// misrouteTaxonomy mirrors the live dos.toml shape the routing-time hold has to survive:
// the trust-critical trees that DO have their own concurrent lane (policy), the one whose
// lane is EXCLUSIVE and therefore absent from the concurrent set (abi -> internal/abi/**),
// and the shippable lanes a mis-route lands on.
var misrouteTaxonomy = LaneTaxonomy{
	Concurrent: []string{"tools", "docs", "policy", "gateway"},
	Trees: map[string][]string{
		"tools":   {"tools/**", "scripts/**"},
		"docs":    {"docs/**"},
		"policy":  {"internal/policy/**"},
		"gateway": {"internal/gateway/**"},
	},
}

func TestRouteIssueHoldsTrustCriticalTextMisroutedToShippableLane(t *testing.T) {
	// The root-cause case (#3122): internal/abi's lane is exclusive, so the path rung
	// yields NO concurrent lane and the `dispatch` scope alias drops the issue on the
	// shippable `tools` lane -- where a guarded worker can never ship it. Held at routing
	// time so it never reaches the front of tools' priority order.
	got := RouteIssue(Issue{
		Number: 2514,
		Title:  "fix(dispatch): tighten the tick's refusal",
		Body:   "the real work is in `internal/abi/kernel.go`",
	}, misrouteTaxonomy, RouteOptions{})
	if got.Lane != "" {
		t.Fatalf("RouteIssue(internal/abi text, dispatch->tools alias).Lane = %q, want \"\" (held)", got.Lane)
	}
	if !strings.HasPrefix(got.Signal, TrustCriticalMisrouteSignalPrefix) {
		t.Fatalf("held route Signal = %q, want the %q prefix", got.Signal, TrustCriticalMisrouteSignalPrefix)
	}
	if !strings.Contains(got.Signal, "internal/abi/kernel.go") || !strings.Contains(got.Signal, "tools") {
		t.Fatalf("held route Signal = %q, want it to name both the trust-critical tree and the lane it was held from", got.Signal)
	}
	if !strings.Contains(got.UnroutedReason, "internal/abi/kernel.go") {
		t.Fatalf("held route UnroutedReason = %q, want the trust-critical tree named", got.UnroutedReason)
	}

	// A CORRECTLY-routed trust-critical issue keeps its lane: the lane tree already
	// reveals the hazard, so the pick-time lane-tree arm holds it with a better witness
	// and the lane attribution operators read stays intact.
	correct := RouteIssue(Issue{
		Number: 2515,
		Title:  "fix(policy): reload the manifest",
		Body:   "in `internal/policy/loader.go`",
	}, misrouteTaxonomy, RouteOptions{})
	if correct.Lane != "policy" {
		t.Fatalf("RouteIssue(correctly-routed policy issue).Lane = %q, want policy (never held at routing time)", correct.Lane)
	}

	// A merely-SELF-SOURCE issue is NOT held: the guard floor permits shipping
	// internal/gateway, so holding it would starve the dispatch surface.
	shippable := RouteIssue(Issue{
		Number: 2516,
		Title:  "fix(gateway): drop the stale keyset",
		Body:   "in `internal/gateway/keyset.go`",
	}, misrouteTaxonomy, RouteOptions{})
	if shippable.Lane != "gateway" {
		t.Fatalf("RouteIssue(guard-shippable gateway issue).Lane = %q, want gateway (never held)", shippable.Lane)
	}

	// An ordinary shippable issue with no trust-critical reference routes untouched.
	plain := RouteIssue(Issue{
		Number: 2517,
		Title:  "docs(docs): refresh the front door",
		Body:   "update `docs/README.md`",
	}, misrouteTaxonomy, RouteOptions{})
	if plain.Lane != "docs" {
		t.Fatalf("RouteIssue(plain docs issue).Lane = %q, want docs", plain.Lane)
	}
}

func TestRouteIssuesKeepsTrustCriticalMisrouteOffTheShippableLaneFront(t *testing.T) {
	// The acceptance criterion stated in payload terms: the mis-routed issue must not
	// appear in ANY shippable lane's priority order in `fak dispatch route --json`, and
	// must be legible to an operator as its own hold class rather than a generic miss.
	// Both fixtures use the CONTRACT-COMPLETE body builder: RouteIssues drops any issue
	// failing the shared issue contract into the skipped set BEFORE routing, so a thin
	// body would make every assertion below pass vacuously. Distinct scopes + disjoint
	// path hints also keep the duplicate-risk skip from firing for the same reason.
	payload := RouteIssues(RouterInput{
		Workspace: "ws",
		Taxonomy:  misrouteTaxonomy,
		Issues: []Issue{
			{Number: 2514, Title: "fix(dispatch): tighten the tick's refusal",
				Body: scopedDispatchIssueBody("2", "fak/internal/abi/kernel.go")},
			{Number: 2600, Title: "chore(ops): drop the stale run dir",
				Body: scopedDispatchIssueBody("3", "tools/dispatch_status.py")},
		},
	})
	if got := len(payload.Issues); got != 2 {
		t.Fatalf("routed %d issue(s), want 2 -- both fixtures must survive the contract/duplicate filters or the assertions below are vacuous", got)
	}
	for lane, group := range payload.Lanes {
		for _, num := range group.Issues {
			if num == 2514 {
				t.Fatalf("mis-routed trust-critical issue #2514 is still in lane %q's priority order %v", lane, group.Issues)
			}
		}
	}
	if got := payload.Lanes["tools"].Issues; len(got) != 1 || got[0] != 2600 {
		t.Fatalf("tools lane issues = %v, want only the genuinely-shippable #2600", got)
	}
	var bucket string
	for _, row := range payload.UnroutableBacklog {
		if row.Number == 2514 {
			bucket = row.Bucket
		}
	}
	if bucket != "trust_critical" {
		t.Fatalf("unroutable bucket for #2514 = %q, want trust_critical (an operator-legible hold, not a generic no_lane miss)", bucket)
	}
}
