package gateway

import (
	"strings"
	"testing"
	"time"
)

// #2182: each FAK_ABLATE_* wire-lever env token must map to the runtime lever it names at
// gateway construction, COMPOSING with (never replacing) the Config flag — the same
// `cfg.X || envEnabled(...)` pattern CacheTTL1H ships. These tests pin the mapping for the
// three tokens this issue wires (bp_plan, prefix_guard, uncached_trim); ttl_1h was already
// wired and keeps its own coverage.

func newAblateLeverServer(t *testing.T, cfg Config) *Server {
	t.Helper()
	if cfg.EngineID == "" {
		cfg.EngineID = "mock"
	}
	if cfg.Model == "" {
		cfg.Model = "test-model"
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func TestAblateBPPlanEnvArmsVCacheAnchor(t *testing.T) {
	if s := newAblateLeverServer(t, Config{}); s.vcacheAnchor {
		t.Fatalf("vcacheAnchor armed with no config flag and no ablation env")
	}
	t.Setenv("FAK_ABLATE_BP_PLAN", "1")
	if s := newAblateLeverServer(t, Config{}); !s.vcacheAnchor {
		t.Fatalf("FAK_ABLATE_BP_PLAN=1 did not arm the PlaceAnthropicCacheBreakpoint pre-flight (vcacheAnchor)")
	}
	// Compose, not replace: the guard flag alone still arms the lever with the env unset.
	t.Setenv("FAK_ABLATE_BP_PLAN", "")
	if s := newAblateLeverServer(t, Config{VCacheAnchor: true}); !s.vcacheAnchor {
		t.Fatalf("Config.VCacheAnchor no longer arms the lever once the ablation env is wired")
	}
}

func TestAblatePrefixGuardEnvArmsGuard(t *testing.T) {
	if s := newAblateLeverServer(t, Config{}); s.prefixGuard {
		t.Fatalf("prefixGuard armed with no config flag and no ablation env")
	}
	t.Setenv("FAK_ABLATE_PREFIX_GUARD", "1")
	if s := newAblateLeverServer(t, Config{}); !s.prefixGuard {
		t.Fatalf("FAK_ABLATE_PREFIX_GUARD=1 did not arm the prefix-determinism guard")
	}
	t.Setenv("FAK_ABLATE_PREFIX_GUARD", "")
	if s := newAblateLeverServer(t, Config{PrefixGuard: true}); !s.prefixGuard {
		t.Fatalf("Config.PrefixGuard no longer arms the guard once the ablation env is wired")
	}
}

func TestAblateUncachedTrimEnvArmsElideDefault(t *testing.T) {
	if s := newAblateLeverServer(t, Config{}); s.elideResultBytes != 0 {
		t.Fatalf("elideResultBytes = %d with no config and no env, want 0 (inert)", s.elideResultBytes)
	}
	t.Setenv("FAK_ABLATE_UNCACHED_TRIM", "1")
	if s := newAblateLeverServer(t, Config{}); s.elideResultBytes != DefaultElideResultBytes {
		t.Fatalf("FAK_ABLATE_UNCACHED_TRIM=1 armed elideResultBytes=%d, want the documented default %d",
			s.elideResultBytes, DefaultElideResultBytes)
	}
	// An explicit configured threshold wins verbatim over the arm's default.
	if s := newAblateLeverServer(t, Config{ElideResultBytes: 1234}); s.elideResultBytes != 1234 {
		t.Fatalf("explicit ElideResultBytes=1234 became %d under the ablation env", s.elideResultBytes)
	}
}

// TestPrefixGuardWitnessesDeterminism drives the armed guard through the same per-turn seam
// both live passthrough finalizers use and checks the verdict fold: an un-anchored turn is
// unknown, the first anchored digest primes the baseline (stable), a repeat is stable, and a
// diverging digest is one mutation that re-primes the baseline. The rendered
// fak_prefix_guard_* family must carry the same counts (the metric-family witness the
// ablation arm reads).
func TestPrefixGuardWitnessesDeterminism(t *testing.T) {
	t.Setenv("FAK_ABLATE_PREFIX_GUARD", "1")
	s := newAblateLeverServer(t, Config{})
	now := time.Now()
	for _, digest := range []string{"", "aaa", "aaa", "bbb"} {
		s.observeHarnessCoherenceAndArm("trace-1", now, digest, false, "", false, false, 0, 0, 0)
		now = now.Add(time.Second)
	}
	snap := s.metrics.harnessCoherence.snapshot()
	if snap.prefixGuardTurns != 4 || snap.prefixGuardUnknown != 1 || snap.prefixGuardStable != 2 || snap.prefixGuardMutated != 1 {
		t.Fatalf("guard fold = turns %d unknown %d stable %d mutated %d, want 4/1/2/1",
			snap.prefixGuardTurns, snap.prefixGuardUnknown, snap.prefixGuardStable, snap.prefixGuardMutated)
	}
	var b strings.Builder
	s.metrics.harnessCoherence.writeHarnessCoherenceMetrics(&b)
	out := b.String()
	for _, want := range []string{
		"fak_prefix_guard_turns_total 4",
		`fak_prefix_guard_verdicts_total{verdict="stable"} 2`,
		`fak_prefix_guard_verdicts_total{verdict="mutated"} 1`,
		`fak_prefix_guard_verdicts_total{verdict="unknown"} 1`,
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered metrics missing %q in:\n%s", want, out)
		}
	}
}

// TestPrefixGuardOffFoldsNothing pins the no-default-flip contract: with neither the config
// flag nor the ablation env set, the guard folds nothing and the family stays at its
// emit-at-0 zeros — the arm's off-baseline really is a no-op.
func TestPrefixGuardOffFoldsNothing(t *testing.T) {
	s := newAblateLeverServer(t, Config{})
	s.observeHarnessCoherenceAndArm("trace-1", time.Now(), "aaa", false, "", false, false, 0, 0, 0)
	if got := s.metrics.harnessCoherence.snapshot().prefixGuardTurns; got != 0 {
		t.Fatalf("unarmed guard folded %d turns, want 0", got)
	}
}
