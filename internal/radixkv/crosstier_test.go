package radixkv

import (
	"reflect"
	"testing"
)

// (a) Hot-only: nil cold spans yields the single hot range and total == hotMatchLen.
func TestAssembleCrossTierPrefixHotOnly(t *testing.T) {
	ranges, total := AssembleCrossTierPrefix(7, nil)
	want := []PrefixRange{{Start: 0, End: 7, Tier: TierHot}}
	if !reflect.DeepEqual(ranges, want) {
		t.Fatalf("ranges = %+v, want %+v", ranges, want)
	}
	if total != 7 {
		t.Fatalf("total = %d, want 7", total)
	}
}

// (b) Hot + one contiguous cold span: two ranges with the correct tiers, summed total.
func TestAssembleCrossTierPrefixHotPlusCold(t *testing.T) {
	ranges, total := AssembleCrossTierPrefix(4, []ColdSpan{{Digest: "sha-a", Len: 3}})
	want := []PrefixRange{
		{Start: 0, End: 4, Tier: TierHot},
		{Start: 4, End: 7, Tier: TierCold},
	}
	if !reflect.DeepEqual(ranges, want) {
		t.Fatalf("ranges = %+v, want %+v", ranges, want)
	}
	if total != 7 {
		t.Fatalf("total = %d, want 7", total)
	}
}

// Multiple contiguous cold spans stay distinct ranges (one per backing digest), all credited.
func TestAssembleCrossTierPrefixMultipleColdSpans(t *testing.T) {
	ranges, total := AssembleCrossTierPrefix(2, []ColdSpan{
		{Digest: "sha-a", Len: 3},
		{Digest: "sha-b", Len: 5},
	})
	want := []PrefixRange{
		{Start: 0, End: 2, Tier: TierHot},
		{Start: 2, End: 5, Tier: TierCold},
		{Start: 5, End: 10, Tier: TierCold},
	}
	if !reflect.DeepEqual(ranges, want) {
		t.Fatalf("ranges = %+v, want %+v", ranges, want)
	}
	if total != 10 {
		t.Fatalf("total = %d, want 10", total)
	}
}

// (c) A gap in the middle truncates: trailing contiguous spans after the gap are dropped.
func TestAssembleCrossTierPrefixGapTruncates(t *testing.T) {
	cases := []struct {
		name string
		gap  ColdSpan
	}{
		{name: "zero-length span", gap: ColdSpan{Digest: "sha-hole", Len: 0}},
		{name: "negative-length span", gap: ColdSpan{Digest: "sha-hole", Len: -2}},
		{name: "missing span (empty digest)", gap: ColdSpan{Digest: "", Len: 6}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ranges, total := AssembleCrossTierPrefix(4, []ColdSpan{
				{Digest: "sha-a", Len: 3},
				tc.gap,
				{Digest: "sha-c", Len: 5}, // contiguous-looking, but past the gap: NOT credited
			})
			want := []PrefixRange{
				{Start: 0, End: 4, Tier: TierHot},
				{Start: 4, End: 7, Tier: TierCold},
			}
			if !reflect.DeepEqual(ranges, want) {
				t.Fatalf("ranges = %+v, want %+v", ranges, want)
			}
			if total != 7 {
				t.Fatalf("total = %d, want 7", total)
			}
		})
	}
}

// A gap as the FIRST cold span leaves only the hot range credited.
func TestAssembleCrossTierPrefixGapAtFirstColdSpan(t *testing.T) {
	ranges, total := AssembleCrossTierPrefix(4, []ColdSpan{
		{Digest: "", Len: 3},
		{Digest: "sha-b", Len: 5},
	})
	want := []PrefixRange{{Start: 0, End: 4, Tier: TierHot}}
	if !reflect.DeepEqual(ranges, want) {
		t.Fatalf("ranges = %+v, want %+v", ranges, want)
	}
	if total != 4 {
		t.Fatalf("total = %d, want 4", total)
	}
}

// (d) Empty/zero inputs: empty ranges, total 0.
func TestAssembleCrossTierPrefixEmptyInputs(t *testing.T) {
	cases := []struct {
		name string
		hot  int
		cold []ColdSpan
	}{
		{name: "all zero", hot: 0, cold: nil},
		{name: "zero hot, empty slice", hot: 0, cold: []ColdSpan{}},
		{name: "negative hot, no cold", hot: -3, cold: nil},
		{name: "zero hot, only holes", hot: 0, cold: []ColdSpan{{Digest: "", Len: 0}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ranges, total := AssembleCrossTierPrefix(tc.hot, tc.cold)
			if len(ranges) != 0 {
				t.Fatalf("ranges = %+v, want empty", ranges)
			}
			if total != 0 {
				t.Fatalf("total = %d, want 0", total)
			}
		})
	}
}

// Cold-only assembly: with hotMatchLen 0 the prefix may assemble entirely from cold spans.
func TestAssembleCrossTierPrefixColdOnly(t *testing.T) {
	ranges, total := AssembleCrossTierPrefix(0, []ColdSpan{{Digest: "sha-a", Len: 3}})
	want := []PrefixRange{{Start: 0, End: 3, Tier: TierCold}}
	if !reflect.DeepEqual(ranges, want) {
		t.Fatalf("ranges = %+v, want %+v", ranges, want)
	}
	if total != 3 {
		t.Fatalf("total = %d, want 3", total)
	}
}

// Tier names are stable for logs and failure messages.
func TestTierString(t *testing.T) {
	if got := TierHot.String(); got != "hot" {
		t.Fatalf("TierHot.String() = %q, want %q", got, "hot")
	}
	if got := TierCold.String(); got != "cold" {
		t.Fatalf("TierCold.String() = %q, want %q", got, "cold")
	}
	if got := Tier(99).String(); got != "unknown" {
		t.Fatalf("Tier(99).String() = %q, want %q", got, "unknown")
	}
}
