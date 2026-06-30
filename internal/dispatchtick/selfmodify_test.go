package dispatchtick

import "testing"

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
			t.Errorf("IsSelfSourceTree(%q) = true, want false (a guard-shippable lane)", g)
		}
	}
}

func TestSelfModifyHoldOnlyHoldsGuardedSelfSourceLanes(t *testing.T) {
	// Guarded worker + self-source lane tree -> held, naming the offending tree.
	if held, tree := SelfModifyHold(true, []string{"cmd/**"}); !held || tree != "cmd/**" {
		t.Fatalf("SelfModifyHold(true, [cmd/**]) = (%v, %q), want (true, cmd/**)", held, tree)
	}
	if held, tree := SelfModifyHold(true, []string{"internal/gateway/**"}); !held || tree != "internal/gateway/**" {
		t.Fatalf("SelfModifyHold(true, [internal/gateway/**]) = (%v, %q), want held", held, tree)
	}

	// Guarded worker + shippable lane -> NOT held (a guarded worker CAN ship docs/tools).
	if held, _ := SelfModifyHold(true, []string{"docs/**"}); held {
		t.Fatalf("SelfModifyHold(true, [docs/**]) held a shippable lane")
	}
	if held, _ := SelfModifyHold(true, []string{"tools/**", "scripts/**"}); held {
		t.Fatalf("SelfModifyHold(true, [tools/**, scripts/**]) held a shippable lane")
	}

	// Unguarded worker -> never held, even on self-source (the operator/worktree escape #1334).
	if held, _ := SelfModifyHold(false, []string{"cmd/**"}); held {
		t.Fatalf("SelfModifyHold(false, [cmd/**]) held an unguarded worker")
	}

	// A mixed tree holds on the first self-source member it finds.
	if held, tree := SelfModifyHold(true, []string{"docs/**", "internal/agent/**"}); !held || tree != "internal/agent/**" {
		t.Fatalf("SelfModifyHold(true, [docs/**, internal/agent/**]) = (%v, %q), want held on internal/agent/**", held, tree)
	}

	// No tree -> not held (nothing to protect).
	if held, _ := SelfModifyHold(true, nil); held {
		t.Fatalf("SelfModifyHold(true, nil) held with no tree")
	}
}

func TestIssueTextTargetsSelfSourceCatchesBareAndPrefixedRefs(t *testing.T) {
	selfSource := map[string]string{
		"most of the backlog lives in `cmd/**` + `internal/**`": "cmd/**",
		"work in cmd/fak/ where the verb shell lives":           "cmd/fak/",
		"see ./cmd/fak/dispatch_tick.go":                        "./cmd/fak/dispatch_tick.go",
		"touches fak/internal/gateway/http.go":                  "fak/internal/gateway/http.go",
		"the internal/agent stream needs a fix":                 "internal/agent",
	}
	for text, want := range selfSource {
		held, tree := IssueTextTargetsSelfSource(text)
		if !held || tree != want {
			t.Errorf("IssueTextTargetsSelfSource(%q) = (%v, %q), want (true, %q)", text, held, tree, want)
		}
	}
	// A bare mention without a cmd/internal path, or a near-miss word, does NOT match --
	// the dispatcher must not hold a genuinely shippable issue.
	notSelfSource := []string{
		"Resolve the issue and keep literal braces like {\"ok\":true} intact.",
		"first-class fak dispatch verb",
		"the subcommand/foo helper and internals/x are unrelated",
		"document the tools/ and docs/ lanes",
		"",
	}
	for _, text := range notSelfSource {
		if held, tree := IssueTextTargetsSelfSource(text); held {
			t.Errorf("IssueTextTargetsSelfSource(%q) = (true, %q), want not held", text, tree)
		}
	}
}

func TestSelfModifyHoldForPickCatchesMisroutedSelfSourceIssue(t *testing.T) {
	// A guarded worker routed to a SAFE lane (tools) whose target issue's text targets
	// fak's own source is held -- the #1338/#1397 mis-route the lane tree alone hides.
	if held, tree := SelfModifyHoldForPick(true, []string{"tools/**", "scripts/**"}, "fix(dispatch): the work lives in `cmd/**`"); !held || tree != "cmd/**" {
		t.Fatalf("SelfModifyHoldForPick(tools lane, cmd/** issue text) = (%v, %q), want held on cmd/**", held, tree)
	}
	// The lane-tree arm wins first and names the lane glob when the lane itself is self-source.
	if held, tree := SelfModifyHoldForPick(true, []string{"internal/gateway/**"}, "no path here"); !held || tree != "internal/gateway/**" {
		t.Fatalf("SelfModifyHoldForPick(self-source lane) = (%v, %q), want held on internal/gateway/**", held, tree)
	}
	// A safe lane + a shippable issue (no self-source ref) is NOT held -- guarded docs work ships.
	if held, _ := SelfModifyHoldForPick(true, []string{"docs/**"}, "update the README front door"); held {
		t.Fatalf("SelfModifyHoldForPick held a shippable docs pick")
	}
	// An UNGUARDED worker is never held, even when the issue text targets self-source.
	if held, _ := SelfModifyHoldForPick(false, []string{"tools/**"}, "edit cmd/fak/main.go"); held {
		t.Fatalf("SelfModifyHoldForPick held an unguarded worker")
	}
}
