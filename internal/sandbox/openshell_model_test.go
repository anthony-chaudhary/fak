package sandbox

import "testing"

// fak-test:runtime fast est=5ms lane=default
func TestOpenShellModelSafeDefaultsAndAdmission(t *testing.T) {
	b := DefaultOpenShellBoundary("/workspace/proj")

	// Safe in-workspace execution on L1 passes.
	ok := b.Admit(OpenShellExecRequest{
		Tier:       TierL1NativeOS,
		WritePaths: []string{"/workspace/proj/out.txt"},
	})
	if !ok.Allowed {
		t.Fatalf("expected workspace write to be allowed, got: %s", ok.Reason)
	}

	// Lower tier (L0) than required (L1) is refused.
	lowTier := b.Admit(OpenShellExecRequest{Tier: TierL0Wasm})
	if lowTier.Allowed || lowTier.RequiredTier != TierL1NativeOS {
		t.Fatalf("expected tier refusal, got %+v", lowTier)
	}

	// Network, host env, subprocess, and outside-workspace writes are denied by default.
	for _, req := range []OpenShellExecRequest{
		{Tier: TierL1NativeOS, NeedsNetwork: true, TargetHost: "example.com"},
		{Tier: TierL1NativeOS, InheritEnv: true},
		{Tier: TierL1NativeOS, SpawnsChild: true},
		{Tier: TierL1NativeOS, WritePaths: []string{"/etc/passwd"}},
	} {
		if dec := b.Admit(req); dec.Allowed {
			t.Fatalf("expected denial for %+v", req)
		}
	}
}
