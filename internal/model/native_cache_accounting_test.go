package model

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// TestNativeCacheAccountingConservesTokens pins the three valid shapes required
// by the leaf: a cold request computes the whole prompt, an exact hit restores
// the whole prompt, and a partial hit splits it. Every valid example must
// conserve restored+computed == total prompt.
func TestNativeCacheAccountingConservesTokens(t *testing.T) {
	cold := &NativeCacheAccounting{
		Operation:         NativeCacheCold,
		TotalPromptTokens: 100,
		StableTokens:      0,
		CachedTokens:      0,
		RestoredTokens:    0,
		ComputedTokens:    100,
		RestoreSource:     NativeCacheRestoreNone,
	}
	exact := &NativeCacheAccounting{
		Operation:         NativeCacheExactHit,
		TotalPromptTokens: 100,
		StableTokens:      100,
		CachedTokens:      100,
		RestoredTokens:    100,
		ComputedTokens:    0,
		RestoreSource:     NativeCacheRestoreHost,
	}
	partial := &NativeCacheAccounting{
		Operation:         NativeCachePartialHit,
		TotalPromptTokens: 100,
		StableTokens:      80,
		CachedTokens:      80,
		RestoredTokens:    80,
		ComputedTokens:    20,
		RestoreSource:     NativeCacheRestoreDevice,
	}
	for _, tc := range []struct {
		name string
		a    *NativeCacheAccounting
	}{
		{"cold", cold},
		{"exact_hit", exact},
		{"partial_hit", partial},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.a.Validate(); err != nil {
				t.Fatalf("valid %s accounting refused: %v", tc.name, err)
			}
			if got, want := tc.a.RestoredTokens+tc.a.ComputedTokens, tc.a.TotalPromptTokens; got != want {
				t.Fatalf("%s does not conserve: restored+computed=%d, total prompt=%d", tc.name, got, want)
			}
		})
	}
}

// TestNativeCacheAccountingRefusesInconsistentCounters pins the negative cases:
// a block that does not conserve, a negative counter, an over-restored block,
// and unknown closed-vocabulary values are all refused with the typed sentinel.
func TestNativeCacheAccountingRefusesInconsistentCounters(t *testing.T) {
	for _, tc := range []struct {
		name string
		a    *NativeCacheAccounting
	}{
		{
			name: "does_not_conserve",
			a: &NativeCacheAccounting{
				Operation: NativeCachePartialHit, TotalPromptTokens: 100,
				StableTokens: 80, CachedTokens: 80, RestoredTokens: 80, ComputedTokens: 30,
				RestoreSource: NativeCacheRestoreHost,
			},
		},
		{
			name: "negative_counter",
			a: &NativeCacheAccounting{
				Operation: NativeCacheCold, TotalPromptTokens: 10,
				StableTokens: -1, ComputedTokens: 10, RestoreSource: NativeCacheRestoreNone,
			},
		},
		{
			name: "negative_duration",
			a: &NativeCacheAccounting{
				Operation: NativeCacheCold, TotalPromptTokens: 10, ComputedTokens: 10,
				RestoreSource: NativeCacheRestoreNone, PrimeDurationSecs: -0.5,
			},
		},
		{
			name: "restored_exceeds_cached",
			a: &NativeCacheAccounting{
				Operation: NativeCachePartialHit, TotalPromptTokens: 100,
				StableTokens: 90, CachedTokens: 50, RestoredTokens: 80, ComputedTokens: 20,
				RestoreSource: NativeCacheRestoreHost,
			},
		},
		{
			name: "cached_exceeds_prompt",
			a: &NativeCacheAccounting{
				Operation: NativeCacheCold, TotalPromptTokens: 10, CachedTokens: 20,
				ComputedTokens: 10, RestoreSource: NativeCacheRestoreNone,
			},
		},
		{
			name: "unknown_operation",
			a: &NativeCacheAccounting{
				Operation: "mystery", TotalPromptTokens: 10, ComputedTokens: 10,
				RestoreSource: NativeCacheRestoreNone,
			},
		},
		{
			name: "unknown_restore_source",
			a: &NativeCacheAccounting{
				Operation: NativeCacheCold, TotalPromptTokens: 10, ComputedTokens: 10,
				RestoreSource: "telepathy",
			},
		},
		{
			name: "cold_claims_restore",
			a: &NativeCacheAccounting{
				Operation: NativeCacheCold, TotalPromptTokens: 10, CachedTokens: 10,
				RestoredTokens: 10, ComputedTokens: 0, RestoreSource: NativeCacheRestoreHost,
			},
		},
		{
			name: "exact_hit_computes",
			a: &NativeCacheAccounting{
				Operation: NativeCacheExactHit, TotalPromptTokens: 10, StableTokens: 10,
				CachedTokens: 10, RestoredTokens: 9, ComputedTokens: 1,
				RestoreSource: NativeCacheRestoreHost,
			},
		},
		{
			name: "exact_hit_without_source",
			a: &NativeCacheAccounting{
				Operation: NativeCacheExactHit, TotalPromptTokens: 10, StableTokens: 10,
				CachedTokens: 10, RestoredTokens: 10, ComputedTokens: 0,
				RestoreSource: NativeCacheRestoreNone,
			},
		},
		{
			name: "partial_hit_without_compute",
			a: &NativeCacheAccounting{
				Operation: NativeCachePartialHit, TotalPromptTokens: 10, StableTokens: 10,
				CachedTokens: 10, RestoredTokens: 10, ComputedTokens: 0,
				RestoreSource: NativeCacheRestoreHost,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.a.Validate(); err == nil {
				t.Fatalf("%s: inconsistent accounting was accepted", tc.name)
			} else if !errors.Is(err, ErrNativeCacheAccountingInvalid) {
				t.Fatalf("%s: refusal is not the typed sentinel: %v", tc.name, err)
			}
		})
	}
}

// TestNativeCacheAccountingStartupPrimeIsZeroWork pins that a startup-prime
// receipt generates no tokens, refuses a nonzero generated count, and can never
// be read as a demand throughput sample.
func TestNativeCacheAccountingStartupPrimeIsZeroWork(t *testing.T) {
	prime := &NativeCacheAccounting{
		Operation:         NativeCacheStartupPrime,
		TotalPromptTokens: 512,
		StableTokens:      512,
		CachedTokens:      512,
		RestoredTokens:    0,
		ComputedTokens:    512,
		RestoreSource:     NativeCacheRestoreNone,
		PrimeDurationSecs: 1.25,
		GeneratedTokens:   0,
	}
	if err := prime.Validate(); err != nil {
		t.Fatalf("valid startup prime refused: %v", err)
	}
	if prime.SatisfiesDemandThroughput() {
		t.Fatal("startup prime was accepted as a demand throughput sample")
	}

	bad := *prime
	bad.GeneratedTokens = 1
	if err := bad.Validate(); err == nil {
		t.Fatal("startup prime with generated tokens was accepted")
	} else if !errors.Is(err, ErrNativeCacheAccountingInvalid) {
		t.Fatalf("startup prime refusal is not the typed sentinel: %v", err)
	}

	demand := &NativeCacheAccounting{Operation: NativeCacheCold, TotalPromptTokens: 4, ComputedTokens: 4, GeneratedTokens: 4, RestoreSource: NativeCacheRestoreNone}
	if !demand.SatisfiesDemandThroughput() {
		t.Fatal("a demand operation was refused as a throughput sample")
	}
}

// TestNativeInferenceReceiptNativeCacheAccountingOptional pins serialization
// compatibility: an absent block leaves the JSON unchanged, and a present block
// round-trips through the receipt without losing any field.
func TestNativeInferenceReceiptNativeCacheAccountingOptional(t *testing.T) {
	base := NativeInferenceReceipt{TokenIDs: []int{7}, Model: "test", Engine: "fak-native"}
	without, err := json.Marshal(base)
	if err != nil {
		t.Fatalf("marshal receipt without accounting: %v", err)
	}
	if strings.Contains(string(without), "native_cache_accounting") {
		t.Fatalf("absent accounting emitted a JSON field: %s", without)
	}
	if err := base.NativeCacheAccounting.Validate(); err != nil {
		t.Fatalf("absent accounting must validate: %v", err)
	}

	base.NativeCacheAccounting = &NativeCacheAccounting{
		Operation: NativeCachePartialHit, TotalPromptTokens: 100,
		StableTokens: 80, CachedTokens: 80, RestoredTokens: 80, ComputedTokens: 20,
		RestoreSource: NativeCacheRestoreDevice, RestoreDurationSecs: 0.02,
		PrimeDurationSecs: 1.5,
	}
	with, err := json.Marshal(base)
	if err != nil {
		t.Fatalf("marshal receipt with accounting: %v", err)
	}
	if !strings.Contains(string(with), "\"native_cache_accounting\"") {
		t.Fatalf("present accounting did not serialize: %s", with)
	}
	var round NativeInferenceReceipt
	if err := json.Unmarshal(with, &round); err != nil {
		t.Fatalf("unmarshal receipt with accounting: %v", err)
	}
	if round.NativeCacheAccounting == nil {
		t.Fatal("round-tripped receipt dropped the accounting block")
	}
	if got := *round.NativeCacheAccounting; got != *base.NativeCacheAccounting {
		t.Fatalf("accounting round-trip mismatch:\n got %+v\nwant %+v", got, *base.NativeCacheAccounting)
	}
	if err := round.NativeCacheAccounting.Validate(); err != nil {
		t.Fatalf("round-tripped accounting does not validate: %v", err)
	}
}
