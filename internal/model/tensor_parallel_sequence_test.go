package model

import (
	"math"
	"testing"
)

// tensor_parallel_sequence_test.go — the gates for the MEGATRON SEQUENCE-PARALLEL subsystem
// (tensor_parallel_sequence.go, the Megatron-LM mappings.py adaptation). The gate grammar
// matches tensor_parallel_forward_test.go / _vocab_test.go:
//
//   - the band dataflow primitives round-trip byte-for-byte (slice into bands, AllGather
//     back in rank order == the original full sequence);
//   - ForwardTP(SequenceRanks=P) == (max|Δ|=0) ForwardTP(SequenceRanks off) end to end —
//     full logits AND every hidden snapshot — at every P, including non-divisible tilings
//     (SP changes only WHERE data lives, never the arithmetic or the reduce order);
//   - SP composes with the sharded sub-layers: ForwardTP(SequenceRanks=P, AttnRanks=2,
//     FFNRanks=2, VocabRanks=2) == (max|Δ|=0) the same config with SP off (the TP
//     reassociation round-off is present identically on BOTH sides, so the difference is
//     pure data placement);
//   - the three supported topologies (PreNorm / PostNorm / SandwichNorm) each stay
//     bit-exact; any other topology refuses with a named error when SP is armed.

// spBitExact reports whether two matrices are Float32bits-identical element-wise, and the
// max abs diff (for the failure message).
func spBitExact(a, b [][]float32) (bool, float64) {
	exact := true
	mx := 0.0
	if len(a) != len(b) {
		return false, math.Inf(1)
	}
	for r := range a {
		if len(a[r]) != len(b[r]) {
			return false, math.Inf(1)
		}
		for i := range a[r] {
			if math.Float32bits(a[r][i]) != math.Float32bits(b[r][i]) {
				exact = false
			}
			if d := math.Abs(float64(a[r][i] - b[r][i])); d > mx {
				mx = d
			}
		}
	}
	return exact, mx
}

// TestSequenceParallelBandRoundTrip pins the band dataflow primitives: slicing a full
// sequence into per-rank bands (the ReduceScatter scatter half) and AllGather-ing them back
// in rank order (upstream gather_from_sequence_parallel_region) reproduces the original
// bytes — the witness-materialization identity — across every tiling shape, including the
// non-divisible remainder-on-first-shards bands upstream's padded divide never produces.
func TestSequenceParallelBandRoundTrip(t *testing.T) {
	const H, seq = 16, 7
	full := make([][]float32, seq)
	for t := range full {
		full[t] = tpRand(H, uint64(t)*13+1)
	}
	for _, p := range []int{1, 2, 3, 5, 7} {
		plan, err := NewTPPlan(seq, p)
		if err != nil {
			t.Fatalf("NewTPPlan(%d,%d): %v", seq, p, err)
		}
		bands := spSliceBands(full, plan)
		back, err := spGatherBands(bands, plan, H, LocalCollective{})
		if err != nil {
			t.Fatalf("spGatherBands(P=%d): %v", p, err)
		}
		if exact, mx := spBitExact(back, full); !exact {
			t.Fatalf("P=%d band round trip: max|Δ|=%.3e, want 0 (bit-exact)", p, mx)
		}
	}
}

// TestSequenceParallelMatchesNonSP is the end-to-end gate: ForwardTP with the sequence
// sharded into P bands reproduces the unsharded ForwardTP BIT-EXACTLY — full logits AND
// every hidden snapshot — at every P. P=1 is the degenerate (unsharded) leg; P=3 and P=5
// over seq=8 exercise the near-even remainder-on-first-shards bands.
func TestSequenceParallelMatchesNonSP(t *testing.T) {
	m := NewSynthetic(tpFwdBaseCfg())
	ids := []int{1, 2, 3, 4, 5, 6, 7, 8}
	ref, err := m.ForwardTP(ids, TPConfig{})
	if err != nil {
		t.Fatalf("ForwardTP(SP off): %v", err)
	}
	for _, p := range []int{1, 2, 3, 5} {
		got, err := m.ForwardTP(ids, TPConfig{SequenceRanks: p})
		if err != nil {
			t.Fatalf("ForwardTP(SequenceRanks=%d): %v", p, err)
		}
		if exact, mx := spBitExact(got.Logits, ref.Logits); !exact {
			t.Fatalf("P=%d logits: max|Δ|=%.3e, want 0 (bit-exact)", p, mx)
		}
		if exact, mx := spBitExact(got.Hidden, ref.Hidden); !exact {
			t.Fatalf("P=%d hidden snapshots: max|Δ|=%.3e, want 0 (bit-exact)", p, mx)
		}
	}
}

// TestSequenceParallelComposesWithShardedSublayers proves SP composes with the intra-layer
// sharding: ForwardTP(SequenceRanks=P, AttnRanks=2, FFNRanks=2, VocabRanks=2) is
// BIT-EXACT vs the same config with SP off. The row-parallel o_proj/down AllReduce
// reassociation (~1e-6) is present identically on BOTH sides — SP moves data between the
// collectives, it never reorders or reassociates a reduction — so the two forwards differ
// nowhere.
func TestSequenceParallelComposesWithShardedSublayers(t *testing.T) {
	m := NewSynthetic(tpFwdBaseCfg())
	ids := tpVocabIDs(m.Cfg.VocabSize)[:9] // 9 tokens (< V=40) crossing the vocab band seams
	cfg := func(sp int) TPConfig {
		return TPConfig{SequenceRanks: sp, AttnRanks: 2, FFNRanks: 2, VocabRanks: 2}
	}
	ref, err := m.ForwardTP(ids, cfg(0))
	if err != nil {
		t.Fatalf("ForwardTP(SP off, sharded sublayers): %v", err)
	}
	for _, p := range []int{2, 3} {
		got, err := m.ForwardTP(ids, cfg(p))
		if err != nil {
			t.Fatalf("ForwardTP(SequenceRanks=%d, sharded sublayers): %v", p, err)
		}
		if exact, mx := spBitExact(got.Logits, ref.Logits); !exact {
			t.Fatalf("P=%d logits: max|Δ|=%.3e, want 0 (bit-exact)", p, mx)
		}
		if exact, mx := spBitExact(got.Hidden, ref.Hidden); !exact {
			t.Fatalf("P=%d hidden snapshots: max|Δ|=%.3e, want 0 (bit-exact)", p, mx)
		}
	}
}

// TestSequenceParallelTopologyCoverage gates the three supported norm placements under SP:
// each stays BIT-EXACT vs its unsharded twin (the norms are per-position elementwise, so
// band-local pre/post norms equal the monolith's at exactly those positions). The one
// unsupported topology (ParallelResidual) refuses with a named error at the SP gate.
func TestSequenceParallelTopologyCoverage(t *testing.T) {
	ids := []int{1, 2, 3, 4, 5, 6, 7, 8}
	for _, topo := range []BlockTopology{PreNorm, PostNorm, SandwichNorm} {
		t.Run(topo.String(), func(t *testing.T) {
			cfg := tpFwdBaseCfg()
			cfg.BlockTopology = topo
			m := NewSynthetic(cfg)
			ref, err := m.ForwardTP(ids, TPConfig{})
			if err != nil {
				t.Fatalf("ForwardTP(SP off, %s): %v", topo, err)
			}
			for _, p := range []int{2, 3} {
				got, err := m.ForwardTP(ids, TPConfig{SequenceRanks: p})
				if err != nil {
					t.Fatalf("ForwardTP(SequenceRanks=%d, %s): %v", p, topo, err)
				}
				if exact, mx := spBitExact(got.Logits, ref.Logits); !exact {
					t.Fatalf("[%s] P=%d logits: max|Δ|=%.3e, want 0 (bit-exact)", topo, p, mx)
				}
				if exact, mx := spBitExact(got.Hidden, ref.Hidden); !exact {
					t.Fatalf("[%s] P=%d hidden snapshots: max|Δ|=%.3e, want 0 (bit-exact)", topo, p, mx)
				}
			}
		})
	}

	// The unsupported topology refuses at the SP gate with a named error.
	if err := spTopologySupported(ParallelResidual); err == nil {
		t.Fatal("ParallelResidual: want a named refusal from the SP gate, got nil")
	} else if !contains(err.Error(), "sequence-parallel") {
		t.Fatalf("SP gate refusal = %q, want it to name sequence-parallel", err.Error())
	}
}

// TestSequenceParallelFailClosed pins the fail-closed contract: armed SP on an unsupported
// topology returns a NAMED error — not a panic, not a silent mis-serve. ParallelResidual is
// refused by the architecture gate (which names the topology); an out-of-range topology
// value is refused by the SP gate itself (which names sequence-parallel) instead of being
// lowered as PreNorm.
func TestSequenceParallelFailClosed(t *testing.T) {
	ids := []int{1, 2, 3, 4}

	// ParallelResidual: the shared-norm two-branch graph is not lowered by the band loop.
	pr := tpFwdBaseCfg()
	pr.BlockTopology = ParallelResidual
	if _, err := (&Model{Cfg: pr}).ForwardTP(ids, TPConfig{SequenceRanks: 2}); err == nil {
		t.Fatal("armed SP on ParallelResidual: want a named refusal, got nil")
	} else if !contains(err.Error(), "ParallelResidual") {
		t.Fatalf("ParallelResidual refusal = %q, want it to name the topology", err.Error())
	}

	// Out-of-range topology value: the SP gate itself refuses (it would otherwise fall into
	// composeSeqSublayer's PreNorm default and silently mis-serve).
	odd := tpFwdBaseCfg()
	odd.BlockTopology = BlockTopology(99)
	if _, err := NewSynthetic(odd).ForwardTP(ids, TPConfig{SequenceRanks: 2}); err == nil {
		t.Fatal("armed SP on an unknown topology: want a named refusal, got nil")
	} else if !contains(err.Error(), "sequence-parallel") {
		t.Fatalf("unknown-topology refusal = %q, want the named sequence-parallel refusal", err.Error())
	}
}
