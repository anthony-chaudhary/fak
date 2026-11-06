package ctxmmu

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestSharedTokenPoolExplicitCapacity(t *testing.T) {
	t.Setenv(EnvHalogenKVPoolFit, "0.01")
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name            string
		tokens          int
		bytes, perToken int64
		factor          float64
		capacity        int
		wantErr         error
	}{
		{"token bound", 16, 256, 8, 0.5, 8, nil},
		{"byte bound", 32, 48, 4, 1, 12, nil},
		{"zero tokens", 0, 64, 4, 1, 0, ErrInvalidTokens},
		{"negative tokens", -1, 64, 4, 1, 0, ErrInvalidTokens},
		{"zero bytes", 16, 0, 4, 1, 0, ErrInvalidPoolCapacity},
		{"negative bytes", 16, -1, 4, 1, 0, ErrInvalidPoolCapacity},
		{"zero geometry", 16, 64, 0, 1, 0, ErrInvalidPoolCapacity},
		{"negative geometry", 16, 64, -1, 1, 0, ErrInvalidPoolCapacity},
		{"no complete token", 16, 3, 4, 1, 0, ErrInvalidPoolCapacity},
		{"zero factor", 16, 64, 4, 0, 0, ErrInvalidPoolCapacity},
		{"negative factor", 16, 64, 4, -1, 0, ErrInvalidPoolCapacity},
		{"nan factor", 16, 64, 4, math.NaN(), 0, ErrInvalidPoolCapacity},
		{"positive infinity", 16, 64, 4, math.Inf(1), 0, ErrInvalidPoolCapacity},
		{"negative infinity", 16, 64, 4, math.Inf(-1), 0, ErrInvalidPoolCapacity},
		{"empty effective capacity", 16, 64, 4, 0.01, 0, ErrInvalidPoolCapacity},
		{"factor product overflow", 16, 64, 4, math.MaxFloat64, 0, ErrInvalidPoolCapacity},
		{"effective int overflow", maxInt, math.MaxInt64, 1, 2, 0, ErrInvalidPoolCapacity},
		{"nominal byte overflow", 3, math.MaxInt64, math.MaxInt64 / 2, 0.5, 0, ErrInvalidPoolCapacity},
		{"effective byte overflow", 2, math.MaxInt64, math.MaxInt64 / 3, 2, 0, ErrInvalidPoolCapacity},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, err := NewSharedTokenPoolWithCapacity(tc.tokens, tc.bytes, tc.perToken, tc.factor)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("constructor error = %v, want %v", err, tc.wantErr)
			}
			if tc.wantErr != nil {
				if p != nil {
					t.Fatal("invalid capacity returned a pool")
				}
				return
			}
			if p.FreeTokens() != tc.capacity || p.DownscaleFactor() != tc.factor {
				t.Fatalf("profile changed: capacity=%d factor=%v", p.FreeTokens(), p.DownscaleFactor())
			}
			if err := p.Reserve("a", tc.capacity-1); err != nil {
				t.Fatal(err)
			}
			if err := p.Commit("a", 2); err != nil {
				t.Fatal(err)
			}
			if err := p.Reserve("b", 1); err != nil {
				t.Fatal(err)
			}
			if p.CommittedTokens() != 2 || p.ReservedTokens() != tc.capacity-2 || p.AllocatedBytes() != int64(tc.capacity)*tc.perToken || p.FreeTokens() != 0 {
				t.Fatal("reserve/commit conservation failed")
			}
			// Clone both maps as well as checking totals: a refusal must not create an empty stream.
			beforeReserved, beforeCommitted := map[string]int{}, map[string]int{}
			for k, v := range p.reservations {
				beforeReserved[k] = v
			}
			for k, v := range p.committed {
				beforeCommitted[k] = v
			}
			for _, request := range []struct {
				id     string
				tokens int
				want   error
			}{
				{"a", 1, ErrPoolExhausted}, {"new", maxInt, ErrPoolExhausted},
				{"a", 0, ErrInvalidTokens}, {"a", -1, ErrInvalidTokens}, {"", 1, ErrEmptyStreamID},
			} {
				if err := p.Reserve(request.id, request.tokens); !errors.Is(err, request.want) {
					t.Fatalf("Reserve = %v, want %v", err, request.want)
				}
			}
			if err := p.Commit("a", tc.capacity); !errors.Is(err, ErrExceedsReservation) {
				t.Fatalf("Commit = %v", err)
			}
			if !reflect.DeepEqual(beforeReserved, p.reservations) || !reflect.DeepEqual(beforeCommitted, p.committed) || p.CommittedTokens() != 2 || p.ReservedTokens() != tc.capacity-2 {
				t.Fatal("refusal mutated ledger")
			}
			p.ReleaseHeadroom("a")
			if p.CommittedTokens() != 2 || p.ReservedTokens() != 1 || p.FreeTokens() != tc.capacity-3 {
				t.Fatal("headroom conservation failed")
			}
			p.Release("a")
			p.Release("b")
			p.Release("b")
			if p.FreeTokens() != tc.capacity || p.AllocatedBytes() != 0 || p.ActiveStreams() != 0 {
				t.Fatal("release conservation failed")
			}
			p.FitPoolToAvailableMemory(tc.bytes)
			if p.DownscaleFactor() != tc.factor {
				t.Fatal("explicit profile read Halo environment during fit")
			}
		})
	}
}

// fak-test:runtime fast est=10ms
func TestSharedTokenPoolCapacityArithmetic(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	p, err := NewSharedTokenPoolWithCapacity(maxInt, math.MaxInt64, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if p.EffectiveMaxTokens() != maxInt {
		t.Fatal("unit factor rounded integer capacity")
	}
	if err := p.Reserve("a", maxInt-1); err != nil {
		t.Fatal(err)
	}
	if err := p.Commit("a", maxInt-2); err != nil {
		t.Fatal(err)
	}
	if err := p.Reserve("b", 2); !errors.Is(err, ErrPoolExhausted) {
		t.Fatalf("overflow admitted: %v", err)
	}
	if p.CommittedTokens() != maxInt-2 || p.ReservedTokens() != 1 || p.FreeTokens() != 1 {
		t.Fatal("overflow refusal mutated ledger")
	}
	if err := p.Reserve("b", 1); err != nil {
		t.Fatal(err)
	}
	p.Release("a")
	p.Release("b")
	if p.FreeTokens() != maxInt {
		t.Fatal("large release failed")
	}
	// Legacy constructors retain permissive defaults, but cannot wrap admission bytes.
	legacy := NewSharedTokenPoolWithConfig(3, math.MaxInt64, math.MaxInt64/2, 1)
	if err := legacy.Reserve("a", 3); !errors.Is(err, ErrPoolExhausted) {
		t.Fatalf("byte overflow admitted: %v", err)
	}
	if legacy.ActiveStreams() != 0 || legacy.AllocatedBytes() != 0 {
		t.Fatal("legacy overflow refusal mutated ledger")
	}
	// Byte budgets are int64 even on 32-bit platforms; fitting/free accounting must not narrow early.
	p, err = NewSharedTokenPoolWithCapacity(16, math.MaxInt64, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	if p.FreeTokens() != 16 || p.FitPoolToAvailableMemory(math.MaxInt64) != 16 {
		t.Fatal("large byte budget narrowed")
	}
	if err := p.Reserve("a", 1); err != nil {
		t.Fatal(err)
	}
	p.FitPoolToAvailableMemory(math.MinInt64)
	if p.FreeTokens() != 0 {
		t.Fatal("negative fit regained capacity")
	}
	if err := p.Reserve("b", 1); !errors.Is(err, ErrPoolExhausted) {
		t.Fatalf("negative fit admitted: %v", err)
	}
}

// fak-test:runtime fast est=10ms
func TestSharedTokenPoolLegacyCapacityDefaults(t *testing.T) {
	t.Setenv(EnvHalogenKVPoolFit, "0.5")
	for _, p := range []*SharedTokenPool{NewSharedTokenPool(), NewSharedTokenPoolWithConfig(0, 0, 0, 0)} {
		if p.MaxTokens() != DefaultMaxTokens || p.TotalGTTBytes() != DefaultTotalGTTBytes || p.BytesPerToken() != DefaultPoolBytesPerToken || p.DownscaleFactor() != 0.5 {
			t.Fatal("legacy defaults changed")
		}
	}
	p := NewSharedTokenPoolWithConfig(16, 64, 4, 1)
	if p.FitPoolToAvailableMemory(64) != 8 {
		t.Fatal("legacy environment refresh changed")
	}
}
