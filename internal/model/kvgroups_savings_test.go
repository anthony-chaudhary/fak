package model

import (
	"strings"
	"testing"
)

// TestKVGroupSavingsBeatsUniformPastWindow witnesses the operator readout at a context well
// past the window: grouped is strictly below uniform, the saving is positive, and every field
// is internally consistent (SavedFloats == Uniform-Grouped, bytes = floats*4, fraction matches).
func TestKVGroupSavingsBeatsUniformPastWindow(t *testing.T) {
	cfg := hybridKVGroupCfg()
	const ctx = 1024
	s := cfg.KVGroupSavingsAt(ctx)

	if want := cfg.KVGroupBudgetAt(ctx).TotalFloats(); s.GroupedFloats != want {
		t.Fatalf("GroupedFloats = %d, want budget total %d", s.GroupedFloats, want)
	}
	if want := cfg.UniformKVFloats(ctx); s.UniformFloats != want {
		t.Fatalf("UniformFloats = %d, want %d", s.UniformFloats, want)
	}
	if s.SavedFloats != s.UniformFloats-s.GroupedFloats {
		t.Fatalf("SavedFloats = %d, want %d", s.SavedFloats, s.UniformFloats-s.GroupedFloats)
	}
	if s.SavedFloats <= 0 {
		t.Fatalf("expected a positive saving past the window, got %d", s.SavedFloats)
	}
	if s.SavedBytes() != s.SavedFloats*4 {
		t.Fatalf("SavedBytes = %d, want %d", s.SavedBytes(), s.SavedFloats*4)
	}
	if want := float64(s.SavedFloats) / float64(s.UniformFloats); s.SavedFraction != want {
		t.Fatalf("SavedFraction = %v, want %v", s.SavedFraction, want)
	}
	if r := s.Readout(); !strings.Contains(r, "ctx=1024") || !strings.Contains(r, "recurrent") {
		t.Fatalf("Readout missing expected fields: %q", r)
	}
}

// TestKVGroupSavingsSignedBelowBreakEven witnesses the honest edge named in the file's
// invalidating assumption: at a tiny context a recurrent layer's fixed O(1) state can exceed
// its uniform per-position share, so the saving is SIGNED and reports negative rather than a
// clamped zero. At ctx=1 the hybrid config's grouped total (48 full + 48 window + 80 recurrent
// = 176) exceeds uniform (3 layers * 1 * 16 stride * 3 planes = 144).
func TestKVGroupSavingsSignedBelowBreakEven(t *testing.T) {
	cfg := hybridKVGroupCfg()
	s := cfg.KVGroupSavingsAt(1)
	if s.SavedFloats != s.UniformFloats-s.GroupedFloats {
		t.Fatalf("SavedFloats = %d, want %d", s.SavedFloats, s.UniformFloats-s.GroupedFloats)
	}
	if s.SavedFloats >= 0 {
		t.Fatalf("expected a negative saving at ctx=1 (recurrent state exceeds uniform share), got %d", s.SavedFloats)
	}
	if s.SavedFraction >= 0 {
		t.Fatalf("expected a negative saved fraction at ctx=1, got %v", s.SavedFraction)
	}
}

// TestKVGroupSavingsUniformOnPlainModel witnesses the non-hybrid invariant: a model with no
// window and no recurrent layers has grouped == uniform, so the saving and fraction are zero.
func TestKVGroupSavingsUniformOnPlainModel(t *testing.T) {
	plain := Config{NumLayers: 4, NumHeads: 4, NumKVHeads: 2, HeadDim: 8}
	s := plain.KVGroupSavingsAt(256)
	if s.SavedFloats != 0 {
		t.Fatalf("plain-model saving = %d, want 0", s.SavedFloats)
	}
	if s.SavedFraction != 0 {
		t.Fatalf("plain-model fraction = %v, want 0", s.SavedFraction)
	}
}
