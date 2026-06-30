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
