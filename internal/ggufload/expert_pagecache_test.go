package ggufload

// expert_pagecache_test.go — the internal/ggufload witnesses for the page-cache/mmap-aware expert
// source (#1302).
//
// Three properties carry the rung, in order of how badly their reversal would hurt:
//
//   - The gate is a REAL gate. WithExpertPageCache must default OFF and be honoured only when a
//     load request passes it explicitly, because an accidental "on" changes expert fault semantics
//     on every load. The retired FAK_EXPERT_PAGECACHE ambient read and its process-global
//     sync.Once cache are gone (#13605 leaf A-cfg-int4): the decision now resolves once per load
//     request from the caller's Q4K load options.
//   - Adoption is exact-or-refused. A region that is not exactly the shard size, or is empty, must
//     never be adopted; a wrong region would let the tier's offset arithmetic index outside the
//     real mapping.
//   - Carriage is faithful. When a shard's reader DOES carry a mapped region (the FAK_GGUF_MMAP
//     path), FusedExpertTensors must put the SAME bytes on the FusedExpertShard, so the model tier
//     can adopt them; when it does not (the default os.Open path, Windows included), Data stays nil
//     and the historical ReadAt path is preserved byte-for-byte.

import (
	"bytes"
	"testing"
)

// TestExpertPageCacheOptionResolves pins the default-OFF gate through the option seam that
// replaced the retired FAK_EXPERT_PAGECACHE ambient read: no option means off, and an explicit
// WithExpertPageCache(true) is the only way on.
func TestExpertPageCacheOptionResolves(t *testing.T) {
	if got := ApplyQ4KLoadOptions(nil).ExpertPageCache; got {
		t.Fatal("no options enabled the expert page cache; the default must stay off")
	}
	if got := probeQ4KLoadOptions(nil).expertPageCache; got {
		t.Fatal("zero-value options enabled the expert page cache; the default must stay off")
	}
	if got := ApplyQ4KLoadOptions([]Q4KLoadOption{WithExpertPageCache(true)}).ExpertPageCache; !got {
		t.Fatal("WithExpertPageCache(true) did not enable the expert page cache")
	}
	if got := ApplyQ4KLoadOptions([]Q4KLoadOption{WithExpertPageCache(false)}).ExpertPageCache; got {
		t.Fatal("WithExpertPageCache(false) left the expert page cache enabled")
	}
}

// TestExpertPageCacheAdoptionGates is the pure adoption core. Only enabled + non-empty + exact
// size may adopt; every other arm must refuse and thereby preserve the historical ReadAt path.
func TestExpertPageCacheAdoptionGates(t *testing.T) {
	for _, tc := range []struct {
		name    string
		data    []byte
		size    int64
		enabled bool
		wantOK  bool
	}{
		{"enabled exact nonempty", []byte("mapped-region"), 13, true, true},
		{"disabled", []byte("mapped-region"), 13, false, false},
		{"len mismatch short", []byte("short"), 99, true, false},
		{"len mismatch long", []byte("this-is-longer-than-size"), 4, true, false},
		{"empty", nil, 0, true, false},
		{"empty with size", []byte{}, 3, true, false},
		{"nil data enabled", nil, 13, true, false},
		{"zero size nonempty", []byte("x"), 0, true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := FusedExpertShard{Size: tc.size, Data: tc.data}
			got, ok := pageCacheAdoption(f, tc.enabled)
			if ok != tc.wantOK {
				t.Fatalf("pageCacheAdoption(ok=%v) = %v, want %v", tc.wantOK, ok, tc.wantOK)
			}
			if ok {
				if !bytes.Equal(got, tc.data) {
					t.Fatalf("adopted region %q is not the shard's mapped region %q", got, tc.data)
				}
			} else if got != nil {
				t.Fatalf("refused adoption returned non-nil data %q", got)
			}
		})
	}
}

// TestFusedExpertShardCarriesMappedData is the carriage witness. A WeightSource over a real
// glm_moe_dsa fixture is given a fake mapped region (dataFor == the region) exactly the size of the
// shard file; FusedExpertTensors with WithExpertPageCache(true) must then hand back that SAME slice
// on shard.Data, which is what licenses the model tier to adopt zero-copy. The region is garbage —
// it is never read here — which is the point: carriage is a pointer/slice move, decided from the
// directory, never a payload scan.
func TestFusedExpertShardCarriesMappedData(t *testing.T) {
	path, _ := glmMoeDsaCheckpointFixture(t, TensorQ6_K)

	f, gg, size, err := openAndRead(path)
	if err != nil {
		t.Fatalf("openAndRead: %v", err)
	}
	defer f.Close()
	ws, err := NewWeightSource(gg, f, size)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}

	// A stand-in for the FAK_GGUF_MMAP region: same length as the shard file, distinct identity.
	mapped := make([]byte, size)
	mapped[0] = 0xAB
	ws.dataFor = make([][]byte, len(gg.Tensors))
	for i := range ws.dataFor {
		ws.dataFor[i] = mapped
	}

	shards, err := ws.FusedExpertTensors(WithExpertPageCache(true))
	if err != nil {
		t.Fatalf("FusedExpertTensors: %v", err)
	}
	if len(shards) != 1 {
		t.Fatalf("a single-file checkpoint produced %d shard groups, want 1", len(shards))
	}
	if shards[0].Data == nil {
		t.Fatal("an mmap-backed shard carried no mapped region; the model tier could not adopt it")
	}
	if &shards[0].Data[0] != &mapped[0] {
		t.Fatal("shard.Data is a copy, not the mapped region; zero-copy adoption requires the same backing array")
	}
	if int64(len(shards[0].Data)) != shards[0].Size {
		t.Fatalf("carried region is %d bytes for a %d-byte shard", len(shards[0].Data), shards[0].Size)
	}

	// And the default (non-mapped) source stays nil even with the gate on: no dataFor, no
	// adoption licence.
	ws.dataFor = nil
	ws.data = nil
	plain, err := ws.FusedExpertTensors(WithExpertPageCache(true))
	if err != nil {
		t.Fatalf("FusedExpertTensors (unmapped source): %v", err)
	}
	if len(plain) != 1 || plain[0].Data != nil {
		t.Fatalf("the unmapped source carried Data=%v; it must stay nil so ReadAt is preserved", plain[0].Data)
	}
	if _, ok := pageCacheAdoption(plain[0], true); ok {
		t.Fatal("the unmapped source was adopted as zero-copy despite carrying no mapped region")
	}
}

// TestFusedExpertShardGateOffCarriesNoData is the default-off half of the carriage witness: with no
// WithExpertPageCache in the request, an mmap-backed source must still hand back Data=nil so no
// consumer can adopt a zero-copy path the operator did not ask for. This is the byte-for-byte
// default.
func TestFusedExpertShardGateOffCarriesNoData(t *testing.T) {
	path, _ := glmMoeDsaCheckpointFixture(t, TensorQ6_K)

	f, gg, size, err := openAndRead(path)
	if err != nil {
		t.Fatalf("openAndRead: %v", err)
	}
	defer f.Close()
	ws, err := NewWeightSource(gg, f, size)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	ws.dataFor = make([][]byte, len(gg.Tensors))
	for i := range ws.dataFor {
		ws.dataFor[i] = make([]byte, size)
	}

	shards, err := ws.FusedExpertTensors()
	if err != nil {
		t.Fatalf("FusedExpertTensors: %v", err)
	}
	if len(shards) != 1 {
		t.Fatalf("a single-file checkpoint produced %d shard groups, want 1", len(shards))
	}
	if shards[0].Data != nil {
		t.Fatal("gate off but the shard carried a mapped region; the expert path would change without an opt-in")
	}
	if _, ok := pageCacheAdoption(shards[0], false); ok {
		t.Fatal("gate off but adoption still reported ok")
	}
}
