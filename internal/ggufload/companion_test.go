package ggufload

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestDiscoverCompanionSingleMTPAssociatedWithMainVariant verifies that a single MTP companion
// draft is identified and paired with the primary model variant.
func TestDiscoverCompanionSingleMTPAssociatedWithMainVariant(t *testing.T) {
	target := "models/unsloth/Qwen3.8-27B-Q4_K_M.gguf"
	candidates := []string{
		"models/unsloth/Qwen3.8-27B-Q4_K_M.gguf",
		"models/unsloth/mtp-Q4_K_M.gguf",
	}

	comp, err := DiscoverCompanion(target, candidates)
	if err != nil {
		t.Fatalf("DiscoverCompanion failed: %v", err)
	}
	if comp == nil {
		t.Fatal("expected companion candidate, got nil")
	}
	if comp.Path != "models/unsloth/mtp-Q4_K_M.gguf" {
		t.Errorf("Path = %q, want %q", comp.Path, "models/unsloth/mtp-Q4_K_M.gguf")
	}
	if comp.DraftKind != DraftKindMtp {
		t.Errorf("DraftKind = %q, want %q", comp.DraftKind, DraftKindMtp)
	}
	if !comp.Exact {
		t.Errorf("Exact = %v, want true", comp.Exact)
	}
	if comp.Distance != 0.0 {
		t.Errorf("Distance = %v, want 0.0", comp.Distance)
	}
}

// TestDiscoverCompanionNearestBitDistanceSelectsQ4_0ForQ4_K_M verifies that when no exact
// quantization match exists, the candidate with the smallest bit-distance is selected
// (e.g., Q4_0 draft with ~4.0 bpw is selected for Q4_K_M with ~4.5 bpw over Q8_0 with ~8.0 bpw).
func TestDiscoverCompanionNearestBitDistanceSelectsQ4_0ForQ4_K_M(t *testing.T) {
	target := "Qwen3.8-27B-Q4_K_M.gguf"
	candidates := []string{
		"mtp-Q8_0.gguf",
		"mtp-Q4_0.gguf",
		"mtp-Q1_0.gguf",
	}

	comp, err := DiscoverCompanion(target, candidates)
	if err != nil {
		t.Fatalf("DiscoverCompanion failed: %v", err)
	}
	if comp == nil {
		t.Fatal("expected companion candidate, got nil")
	}
	if comp.Path != "mtp-Q4_0.gguf" {
		t.Errorf("Path = %q, want %q (nearest bit-distance to Q4_K_M)", comp.Path, "mtp-Q4_0.gguf")
	}
	if comp.Exact {
		t.Errorf("Exact = %v, want false", comp.Exact)
	}
	if comp.Quant != "Q4_0" {
		t.Errorf("Quant = %q, want %q", comp.Quant, "Q4_0")
	}
	if math.Abs(comp.Distance-0.5) > 1e-6 {
		t.Errorf("Distance = %v, want ~0.5 (|4.0 - 4.5|)", comp.Distance)
	}
}

// TestDiscoverCompanionMixedMechanismsAmbiguous verifies that when multiple speculative mechanisms
// (e.g. MTP and DFlash) coexist in the candidate pool without explicit selection, pairing returns
// ErrAmbiguousDraftMechanism without silent guessing.
func TestDiscoverCompanionMixedMechanismsAmbiguous(t *testing.T) {
	target := "Qwen3.8-27B-Q4_K_M.gguf"
	candidates := []string{
		"mtp-Q4_0.gguf",
		"dflash-Q4_0.gguf",
	}

	comp, err := DiscoverCompanion(target, candidates)
	if !errors.Is(err, ErrAmbiguousDraftMechanism) {
		t.Fatalf("DiscoverCompanion error = %v, want %v", err, ErrAmbiguousDraftMechanism)
	}
	if comp != nil {
		t.Errorf("expected nil companion, got %+v", comp)
	}
}

// TestDiscoverCompanionPreferredKindResolvesAmbiguity verifies that specifying PreferredKind
// resolves mechanism ambiguity and selects the intended draft kind.
func TestDiscoverCompanionPreferredKindResolvesAmbiguity(t *testing.T) {
	target := "Qwen3.8-27B-Q4_K_M.gguf"
	candidates := []string{
		"mtp-Q4_0.gguf",
		"dflash-Q4_0.gguf",
	}

	comp, err := DiscoverCompanion(target, candidates, CompanionOptions{PreferredKind: DraftKindMtp})
	if err != nil {
		t.Fatalf("DiscoverCompanion with PreferredKind failed: %v", err)
	}
	if comp.Path != "mtp-Q4_0.gguf" || comp.DraftKind != DraftKindMtp {
		t.Errorf("got companion %+v, want mtp draft", comp)
	}

	compDflash, err := DiscoverCompanion(target, candidates, CompanionOptions{PreferredKind: DraftKindDflash})
	if err != nil {
		t.Fatalf("DiscoverCompanion with PreferredKind Dflash failed: %v", err)
	}
	if compDflash.Path != "dflash-Q4_0.gguf" || compDflash.DraftKind != DraftKindDflash {
		t.Errorf("got companion %+v, want dflash draft", compDflash)
	}
}

// TestDiscoverCompanionSharedDirectoryDepthPreference verifies that candidates with deeper
// shared directory ancestry with the target are ranked before distant candidates.
func TestDiscoverCompanionSharedDirectoryDepthPreference(t *testing.T) {
	target := "models/family/qwen/model-Q4_K_M.gguf"
	candidates := []string{
		"models/other/mtp-Q4_K_M.gguf",       // shared depth 1: "models"
		"models/family/qwen/mtp-Q4_K_M.gguf", // shared depth 3: "models", "family", "qwen"
		"models/family/mtp-Q4_K_M.gguf",      // shared depth 2: "models", "family"
	}

	ranked, err := RankCompanionDrafts(target, candidates)
	if err != nil {
		t.Fatalf("RankCompanionDrafts failed: %v", err)
	}
	if len(ranked) != 3 {
		t.Fatalf("len(ranked) = %d, want 3", len(ranked))
	}
	if ranked[0].Path != "models/family/qwen/mtp-Q4_K_M.gguf" || ranked[0].Depth != 3 {
		t.Errorf("rank 0 = %s (depth %d), want models/family/qwen/mtp-Q4_K_M.gguf (depth 3)", ranked[0].Path, ranked[0].Depth)
	}
	if ranked[1].Path != "models/family/mtp-Q4_K_M.gguf" || ranked[1].Depth != 2 {
		t.Errorf("rank 1 = %s (depth %d), want models/family/mtp-Q4_K_M.gguf (depth 2)", ranked[1].Path, ranked[1].Depth)
	}
	if ranked[2].Path != "models/other/mtp-Q4_K_M.gguf" || ranked[2].Depth != 1 {
		t.Errorf("rank 2 = %s (depth %d), want models/other/mtp-Q4_K_M.gguf (depth 1)", ranked[2].Path, ranked[2].Depth)
	}
}

// TestDiscoverCompanionExactQuantMatchPreference verifies that an exact quantization match
// ranks ahead of candidates requiring bit-distance approximation.
func TestDiscoverCompanionExactQuantMatchPreference(t *testing.T) {
	target := "model-Q4_K_M.gguf"
	candidates := []string{
		"mtp-Q4_0.gguf",   // distance 0.5, exact false
		"mtp-Q4_K_M.gguf", // distance 0.0, exact true
	}

	comp, err := DiscoverCompanion(target, candidates)
	if err != nil {
		t.Fatalf("DiscoverCompanion failed: %v", err)
	}
	if comp.Path != "mtp-Q4_K_M.gguf" {
		t.Errorf("Path = %q, want mtp-Q4_K_M.gguf (exact match preference)", comp.Path)
	}
	if !comp.Exact {
		t.Errorf("Exact = %v, want true", comp.Exact)
	}
}

// TestDiscoverCompanionLexicographicalTieBreak verifies that when directory depth, exactness,
// and bit-distance are equal, ranking breaks ties deterministically in lexicographical order.
func TestDiscoverCompanionLexicographicalTieBreak(t *testing.T) {
	target := "model-Q4_0.gguf"
	candidates := []string{
		"mtp-draft-beta-Q4_0.gguf",
		"mtp-draft-alpha-Q4_0.gguf",
	}

	comp, err := DiscoverCompanion(target, candidates)
	if err != nil {
		t.Fatalf("DiscoverCompanion failed: %v", err)
	}
	if comp.Path != "mtp-draft-alpha-Q4_0.gguf" {
		t.Errorf("Path = %q, want mtp-draft-alpha-Q4_0.gguf (lexicographical tie-break)", comp.Path)
	}
}

// TestDiscoverCompanionExcludesTargetAndNonDrafts verifies that the target model itself
// and unrelated non-draft files are filtered out.
func TestDiscoverCompanionExcludesTargetAndNonDrafts(t *testing.T) {
	target := "models/Qwen3.8-27B-Q4_K_M.gguf"
	candidates := []string{
		"models/Qwen3.8-27B-Q4_K_M.gguf", // target itself
		"models/README.md",               // not a gguf draft
		"models/vocab.json",              // not a gguf draft
		"models/other-model.gguf",        // non-draft model
	}

	comp, err := DiscoverCompanion(target, candidates)
	if !errors.Is(err, ErrNoCompanionFound) {
		t.Fatalf("DiscoverCompanion error = %v, want %v", err, ErrNoCompanionFound)
	}
	if comp != nil {
		t.Errorf("expected nil companion, got %+v", comp)
	}
}

// TestDetectDraftKind verifies classification of draft sidecars vs primary model files.
func TestDetectDraftKind(t *testing.T) {
	tests := []struct {
		path string
		want DraftKind
	}{
		{"mtp-Q4_0.gguf", DraftKindMtp},
		{"mtp-Q4_K_M.gguf", DraftKindMtp},
		{"mtp_q4_k_m.gguf", DraftKindMtp},
		{"model.mtp.gguf", DraftKindMtp},
		{"Qwen3.8-27B-mtp-Q4_0.gguf", DraftKindMtp},
		{"nextn-q4_0.gguf", DraftKindMtp},
		{"model-nextn.gguf", DraftKindMtp},
		{"dflash-Q4_0.gguf", DraftKindDflash},
		{"dflash_q8_0.gguf", DraftKindDflash},
		{"model-dflash-q4_k_m.gguf", DraftKindDflash},
		{"Qwen-dflash.gguf", DraftKindDflash},
		{"Qwen3.8-27B-Q4_K_M.gguf", DraftKindNone},
		{"model.gguf", DraftKindNone},
		{"mtp-dflash-Q4_0.gguf", DraftKindNone}, // mixed in filename -> rejected
	}

	for _, tc := range tests {
		got := DetectDraftKind(tc.path)
		if got != tc.want {
			t.Errorf("DetectDraftKind(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}

// TestExtractQuantTagAndQuantBits verifies quantization extraction and bpw mapping.
func TestExtractQuantTagAndQuantBits(t *testing.T) {
	tests := []struct {
		path     string
		wantTag  string
		wantBits float64
	}{
		{"Qwen3.8-27B-Q4_K_M.gguf", "Q4_K_M", 4.5},
		{"Qwen3.8-27B-UD-Q4_K_M.gguf", "Q4_K_M", 4.5},
		{"mtp-Q4_0.gguf", "Q4_0", 4.0},
		{"mtp-q8_0.gguf", "Q8_0", 8.0},
		{"model-Q3_K_XL.gguf", "Q3_K_XL", 3.7},
		{"model-iq2_xs.gguf", "IQ2_XS", 2.31},
		{"dflash-bf16.gguf", "BF16", 16.0},
		{"model.fp16.gguf", "F16", 16.0},
		{"model.f32.gguf", "F32", 32.0},
	}

	for _, tc := range tests {
		tag := ExtractQuantTag(tc.path)
		if tag != tc.wantTag {
			t.Errorf("ExtractQuantTag(%q) = %q, want %q", tc.path, tag, tc.wantTag)
		}
		bits, ok := QuantBits(tag)
		if !ok {
			t.Errorf("QuantBits(%q) not found", tag)
		}
		if math.Abs(bits-tc.wantBits) > 1e-6 {
			t.Errorf("QuantBits(%q) = %v, want %v", tag, bits, tc.wantBits)
		}
	}
}

// TestDiscoverDirectoryCompanions creates a temporary directory tree and proves
// real filesystem companion discovery.
func TestDiscoverDirectoryCompanions(t *testing.T) {
	tmpDir := t.TempDir()

	targetPath := filepath.Join(tmpDir, "Qwen3.8-27B-Q4_K_M.gguf")
	if err := os.WriteFile(targetPath, []byte("main-model"), 0o644); err != nil {
		t.Fatalf("WriteFile target failed: %v", err)
	}

	companionPath := filepath.Join(tmpDir, "mtp-Q4_0.gguf")
	if err := os.WriteFile(companionPath, []byte("mtp-draft"), 0o644); err != nil {
		t.Fatalf("WriteFile companion failed: %v", err)
	}

	comp, err := DiscoverDirectoryCompanions(targetPath)
	if err != nil {
		t.Fatalf("DiscoverDirectoryCompanions failed: %v", err)
	}
	if filepath.Clean(comp.Path) != filepath.Clean(companionPath) {
		t.Errorf("Path = %q, want %q", comp.Path, companionPath)
	}
	if comp.DraftKind != DraftKindMtp {
		t.Errorf("DraftKind = %q, want %q", comp.DraftKind, DraftKindMtp)
	}
	if comp.Quant != "Q4_0" {
		t.Errorf("Quant = %q, want %q", comp.Quant, "Q4_0")
	}
}
