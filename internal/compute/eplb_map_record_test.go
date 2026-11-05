package compute

// CPU-oracle witnesses for the EPLB logical-to-physical map-and-record contract
// (eplb_map_record.go, device twin eplb_map_record.cu). The expected values below
// are hand-computed from the vLLM _eplb_map_and_record_i32_kernel semantics pinned
// by ticket oss-port-vllmtop-eplb-map-record-20261103: replica selection is the
// low 32 bits of token*2654435769 modulo the clamped replica count, hit counts
// accumulate per physical target, and disabled/unpadded/out-of-range entries
// never record.

import (
	"errors"
	"reflect"
	"testing"
)

func TestEplbMapRecordMatchesOracle(t *testing.T) {
	t.Run("identity map passes ids through with per-logical hit counts", func(t *testing.T) {
		// 4 logical experts, 2 live replicas each, 2 map slots, 4 physical experts;
		// every slot of expert e names e itself, so any replica selection still
		// yields the input id. 3 tokens x topk 2. Token hashes: h(0)=0 (even),
		// h(1)=2654435769 (odd), h(2)=1013904242 (even), so token 1 takes replica
		// slot 1 and tokens 0/2 take slot 0 — both identity.
		args := EplbMapRecordArgs{
			TopkIDs:             []int32{0, 1, 2, 0, 3, 1},
			NumActiveExperts:    2,
			NumLogicalExperts:   4,
			LogicalToPhysical:   []int32{0, 0, 1, 1, 2, 2, 3, 3},
			LogicalReplicaCount: []int32{2, 2, 2, 2},
			MapSlots:            2,
			NumPhysicalExperts:  4,
			RecordEnabled:       true,
			NumUnpaddedTokens:   -1,
		}
		got, err := EplbMapRecord(args)
		if err != nil {
			t.Fatalf("EplbMapRecord: %v", err)
		}
		want := &EplbMapRecordResult{
			PhysicalIDs: []int32{0, 1, 2, 0, 3, 1},
			HitCounts:   []int32{2, 2, 1, 1},
		}
		assertEplb(t, got, want)
	})

	t.Run("permutation with duplicate physical targets accumulates per-physical hits", func(t *testing.T) {
		// 5 logical experts over 4 physical experts, single replica each: 0<->1 are
		// swapped and logical 4 duplicates physical 1's target. 2 tokens x topk 3.
		args := EplbMapRecordArgs{
			TopkIDs:             []int32{0, 1, 4, 2, 3, 1},
			NumActiveExperts:    3,
			NumLogicalExperts:   5,
			LogicalToPhysical:   []int32{1, 0, 2, 3, 1},
			LogicalReplicaCount: []int32{1, 1, 1, 1, 1},
			MapSlots:            1,
			NumPhysicalExperts:  4,
			RecordEnabled:       true,
			NumUnpaddedTokens:   -1,
		}
		got, err := EplbMapRecord(args)
		if err != nil {
			t.Fatalf("EplbMapRecord: %v", err)
		}
		want := &EplbMapRecordResult{
			PhysicalIDs: []int32{1, 0, 1, 2, 3, 0},
			HitCounts:   []int32{2, 2, 1, 1},
		}
		assertEplb(t, got, want)
	})

	t.Run("record disabled maps but keeps hit counts at zero", func(t *testing.T) {
		// Same panel as the permutation case above; only RecordEnabled flips.
		args := EplbMapRecordArgs{
			TopkIDs:             []int32{0, 1, 4, 2, 3, 1},
			NumActiveExperts:    3,
			NumLogicalExperts:   5,
			LogicalToPhysical:   []int32{1, 0, 2, 3, 1},
			LogicalReplicaCount: []int32{1, 1, 1, 1, 1},
			MapSlots:            1,
			NumPhysicalExperts:  4,
			RecordEnabled:       false,
			NumUnpaddedTokens:   -1,
		}
		got, err := EplbMapRecord(args)
		if err != nil {
			t.Fatalf("EplbMapRecord: %v", err)
		}
		want := &EplbMapRecordResult{
			PhysicalIDs: []int32{1, 0, 1, 2, 3, 0},
			HitCounts:   []int32{0, 0, 0, 0},
		}
		assertEplb(t, got, want)
	})

	t.Run("unpadded window records only real tokens", func(t *testing.T) {
		// Permutation panel again, 2 tokens x topk 3. NumUnpaddedTokens=1 admits
		// entries 0..2 (token 0) into recording; =0 admits none.
		base := EplbMapRecordArgs{
			TopkIDs:             []int32{0, 1, 4, 2, 3, 1},
			NumActiveExperts:    3,
			NumLogicalExperts:   5,
			LogicalToPhysical:   []int32{1, 0, 2, 3, 1},
			LogicalReplicaCount: []int32{1, 1, 1, 1, 1},
			MapSlots:            1,
			NumPhysicalExperts:  4,
			RecordEnabled:       true,
		}
		for _, tc := range []struct {
			unpadded int
			wantHits []int32
		}{
			{1, []int32{1, 2, 0, 0}}, // token 0 maps to phys 1, 0, 1: p1 takes two
			{0, []int32{0, 0, 0, 0}},
		} {
			args := base
			args.NumUnpaddedTokens = tc.unpadded
			got, err := EplbMapRecord(args)
			if err != nil {
				t.Fatalf("EplbMapRecord(unpadded=%d): %v", tc.unpadded, err)
			}
			want := &EplbMapRecordResult{
				PhysicalIDs: []int32{1, 0, 1, 2, 3, 0},
				HitCounts:   tc.wantHits,
			}
			assertEplb(t, got, want)
		}
	})

	t.Run("out-of-range physical map value stores raw and skips recording", func(t *testing.T) {
		// Faithful to upstream: a map value outside [0, NumPhysicalExperts) is
		// stored in PhysicalIDs but never recorded.
		args := EplbMapRecordArgs{
			TopkIDs:             []int32{0, 1},
			NumActiveExperts:    2,
			NumLogicalExperts:   2,
			LogicalToPhysical:   []int32{99, 2},
			LogicalReplicaCount: []int32{1, 1},
			MapSlots:            1,
			NumPhysicalExperts:  4,
			RecordEnabled:       true,
			NumUnpaddedTokens:   -1,
		}
		got, err := EplbMapRecord(args)
		if err != nil {
			t.Fatalf("EplbMapRecord: %v", err)
		}
		want := &EplbMapRecordResult{
			PhysicalIDs: []int32{99, 2},
			HitCounts:   []int32{0, 0, 1, 0},
		}
		assertEplb(t, got, want)
	})

	t.Run("zero-replica logical expert clamps to slot 0", func(t *testing.T) {
		// A logical expert with replica count 0 clamps to 1 and maps through slot
		// 0, exactly like the kernel's tl.maximum(replica_count, 1).
		args := EplbMapRecordArgs{
			TopkIDs:             []int32{0, 1},
			NumActiveExperts:    2,
			NumLogicalExperts:   2,
			LogicalToPhysical:   []int32{7, 8, 3, 4},
			LogicalReplicaCount: []int32{0, 1},
			MapSlots:            2,
			NumPhysicalExperts:  8,
			RecordEnabled:       true,
			NumUnpaddedTokens:   -1,
		}
		got, err := EplbMapRecord(args)
		if err != nil {
			t.Fatalf("EplbMapRecord: %v", err)
		}
		want := &EplbMapRecordResult{
			PhysicalIDs: []int32{7, 3},
			HitCounts:   []int32{0, 0, 0, 1, 0, 0, 0, 1},
		}
		assertEplb(t, got, want)
	})

	t.Run("out-of-range logical id is a typed refusal", func(t *testing.T) {
		// Pin (c): unlike the device's -1 fallback, the host API refuses.
		base := EplbMapRecordArgs{
			TopkIDs:             []int32{0, 1},
			NumActiveExperts:    2,
			NumLogicalExperts:   2,
			LogicalToPhysical:   []int32{0, 1},
			LogicalReplicaCount: []int32{1, 1},
			MapSlots:            1,
			NumPhysicalExperts:  2,
			RecordEnabled:       true,
			NumUnpaddedTokens:   -1,
		}
		for _, tc := range []struct {
			name string
			id   int32
		}{
			{"id equals numLogical", 2},
			{"negative id", -1},
		} {
			args := base
			args.TopkIDs = []int32{tc.id, 1}
			_, err := EplbMapRecord(args)
			if !errors.Is(err, ErrEplbLogicalIDOutOfRange) {
				t.Errorf("%s: errors.Is(err, ErrEplbLogicalIDOutOfRange) = false, err=%v", tc.name, err)
			}
		}
	})

	t.Run("geometry refusals are closed sentinels", func(t *testing.T) {
		ok := EplbMapRecordArgs{
			TopkIDs:             []int32{0, 1},
			NumActiveExperts:    2,
			NumLogicalExperts:   2,
			LogicalToPhysical:   []int32{0, 1},
			LogicalReplicaCount: []int32{1, 1},
			MapSlots:            1,
			NumPhysicalExperts:  2,
			RecordEnabled:       true,
			NumUnpaddedTokens:   -1,
		}
		for _, tc := range []struct {
			name   string
			mutate func(*EplbMapRecordArgs)
		}{
			{"zero active experts", func(a *EplbMapRecordArgs) { a.NumActiveExperts = 0 }},
			{"zero logical experts", func(a *EplbMapRecordArgs) { a.NumLogicalExperts = 0 }},
			{"zero map slots", func(a *EplbMapRecordArgs) { a.MapSlots = 0 }},
			{"zero physical experts", func(a *EplbMapRecordArgs) { a.NumPhysicalExperts = 0 }},
			{"short replica count", func(a *EplbMapRecordArgs) { a.LogicalReplicaCount = a.LogicalReplicaCount[:1] }},
			{"short map", func(a *EplbMapRecordArgs) { a.LogicalToPhysical = a.LogicalToPhysical[:1] }},
			{"unpadded sentinel below -1", func(a *EplbMapRecordArgs) { a.NumUnpaddedTokens = -2 }},
			{"unpadded beyond int32", func(a *EplbMapRecordArgs) { a.NumUnpaddedTokens = 1 << 31 }},
		} {
			args := ok
			tc.mutate(&args)
			_, err := EplbMapRecord(args)
			if !errors.Is(err, ErrEplbGeometry) {
				t.Errorf("%s: errors.Is(err, ErrEplbGeometry) = false, err=%v", tc.name, err)
			}
		}
	})
}

func assertEplb(t *testing.T, got, want *EplbMapRecordResult) {
	t.Helper()
	if !reflect.DeepEqual(got.PhysicalIDs, want.PhysicalIDs) {
		t.Errorf("physicalIDs = %v want %v", got.PhysicalIDs, want.PhysicalIDs)
	}
	if !reflect.DeepEqual(got.HitCounts, want.HitCounts) {
		t.Errorf("hitCounts = %v want %v", got.HitCounts, want.HitCounts)
	}
}
