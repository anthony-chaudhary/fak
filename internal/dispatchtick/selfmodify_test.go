package dispatchtick

import (
	"reflect"
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
