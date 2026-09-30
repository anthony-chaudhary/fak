//go:build darwin && arm64 && cgo

package model

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestQwen35MetalWholeTokenPoolDefault exercises the ordinary P32->two-P1
// selector against the explicit rollback. Native graph allocation receipts must
// show the default reduces retained buffers while both continuations remain exact.
// fak-test:runtime integration est=5s lane=default
func TestQwen35MetalWholeTokenPoolDefault(t *testing.T) {
	if !metalgemm.Available() || metalgemm.DeviceName() == "" {
		t.Skip("whole-token pool witness requires a physical Metal device")
	}
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	t.Setenv("FAK_QWEN35_PERSISTENT_DECODE_DKV", "0")
	cfg := qwen35HybridQ4KTestCfg()
	cfg.QKNorm = true
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*19 + 7) % cfg.VocabSize
	}
	baselineGDN := metalgemm.GDNLiveBufferCount()
	newSession := func() *Session {
		t.Helper()
		s := m.NewSession()
		s.Q4K, s.MetalQ4K, s.captureTargetHidden = true, true, true
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			s.Close()
			t.Fatal(err)
		}
		return s
	}
	// Absorb one-time GDN promotion before comparing the two independent owners.
	t.Setenv("FAK_QWEN35_WHOLE_TOKEN_POOL", "0")
	primer := newSession()
	primer.Prefill(prompt)
	if accepted, err := primer.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !accepted {
		primer.Close()
		t.Fatalf("primer finalize accepted=%v err=%v", accepted, err)
	}
	primer.Step(7)
	primer.Close()
	type witness struct {
		logits   [][]float32
		receipts []Qwen35MetalForwardSequenceReceipt
		cache    *KVCache
		hidden   [][]float32
		gdn      map[int][2][]float32
	}
	run := func(pool string) witness {
		t.Helper()
		t.Setenv("FAK_QWEN35_WHOLE_TOKEN_POOL", pool)
		s := newSession()
		defer s.Close()
		out := witness{gdn: make(map[int][2][]float32)}
		out.logits = append(out.logits, slices.Clone(s.Prefill(prompt)))
		if accepted, err := s.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !accepted {
			t.Fatalf("pool=%q finalize accepted=%v err=%v", pool, accepted, err)
		}
		for _, id := range []int{7, 11} {
			out.logits = append(out.logits, slices.Clone(s.Step(id)))
			r := s.Qwen35MetalForwardSequenceReceipt()
			if !r.Available || !r.Committed || !r.CompletedWait || r.Tokens != 1 ||
				r.SelectorState != Qwen35MetalSequenceSelectorOn || r.EvidenceState != Qwen35MetalSequenceEvidenceExecuted ||
				r.Path != Qwen35MetalGDNSequenceForwardPath || r.CommandBuffers != 1 || r.TerminalWaits != 1 ||
				r.IntermediateWaits != 0 || r.IntermediateReadbacks != 0 || r.FallbackCount != 0 ||
				r.AllocatedBuffers <= 0 || r.RetainedBufferBytes == 0 {
				t.Fatalf("pool=%q token=%d lacks executed whole-token allocation witness: %+v", pool, id, r)
			}
			out.receipts = append(out.receipts, r)
		}
		if s.Cache.Len() != 34 {
			t.Fatalf("pool=%q cache positions=%d, want 34", pool, s.Cache.Len())
		}
		out.cache = s.Cache.Clone()
		out.hidden = cloneTargetHidden(s.targetHidden)
		snapshotter, ok := s.qwen35HAL.sequenceBackend.(qwen35GDNSequenceSnapshotter)
		if !ok {
			t.Fatal("executed GDN sequence has no snapshot oracle")
		}
		for layer := 0; layer < cfg.NumLayers; layer++ {
			if cfg.isLinearAttnLayer(layer) {
				conv, recurrent, err := snapshotter.SnapshotQwen35GDNAuxState(s.qwen35HAL.sequenceLayers[layer])
				if err != nil {
					t.Fatal(err)
				}
				out.gdn[layer] = [2][]float32{conv, recurrent}
			}
		}
		return out
	}
	control := run("0")
	for _, arm := range []struct {
		name string
		got  witness
	}{{"explicit", run("1")}, {"default", run("")}} {
		candidate := arm.got
		for i := range control.logits {
			assertPoolFiniteBitExact(t, fmt.Sprintf("logits step %d", i), control.logits[i], candidate.logits[i])
		}
		if !slices.Equal(control.cache.pos, candidate.cache.pos) || !slices.Equal(control.cache.lineage.ids, candidate.cache.lineage.ids) ||
			control.cache.lineage.fault != candidate.cache.lineage.fault || len(control.hidden) != len(candidate.hidden) {
			t.Fatal("pool changed cache positions, lineage, or captured hidden rows")
		}
		for row := range control.hidden {
			assertPoolFiniteBitExact(t, fmt.Sprintf("raw hidden row %d", row), control.hidden[row], candidate.hidden[row])
		}
		for layer := 0; layer < cfg.NumLayers; layer++ {
			for _, plane := range []struct {
				name      string
				want, got []float32
			}{
				{"K", control.cache.K[layer], candidate.cache.K[layer]},
				{"Kraw", control.cache.Kraw[layer], candidate.cache.Kraw[layer]},
				{"V", control.cache.V[layer], candidate.cache.V[layer]},
			} {
				assertPoolFiniteBitExact(t, fmt.Sprintf("layer %d %s", layer, plane.name), plane.want, plane.got)
			}
			if cfg.isLinearAttnLayer(layer) {
				for side, name := range []string{"conv", "recurrent"} {
					assertPoolFiniteBitExact(t, fmt.Sprintf("layer %d %s", layer, name), control.gdn[layer][side], candidate.gdn[layer][side])
				}
			}
		}
		for i, want := range control.receipts {
			got := candidate.receipts[i]
			if got.Encoders != want.Encoders || got.HostUploadBytes != want.HostUploadBytes || got.HostReadbackBytes != want.HostReadbackBytes ||
				got.AllocatedBuffers >= want.AllocatedBuffers || got.RetainedBufferBytes >= want.RetainedBufferBytes {
				t.Fatalf("P1[%d] default did not preserve work and reduce tracked allocation: rollback=%+v default=%+v", i, want, got)
			}
			t.Logf("physical %s P1[%d] native tracked buffers %d->%d retained bytes %d->%d; logits/cache/GDN finite bit-exact",
				arm.name, i, want.AllocatedBuffers, got.AllocatedBuffers, want.RetainedBufferBytes, got.RetainedBufferBytes)
		}
	}
	if got := metalgemm.GDNLiveBufferCount(); got != baselineGDN {
		t.Fatalf("session cleanup leaked GDN buffers: baseline=%d after=%d", baselineGDN, got)
	}
}

func assertPoolFiniteBitExact(t *testing.T, name string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s lengths: rollback=%d default=%d", name, len(want), len(got))
	}
	for i, x := range want {
		y := got[i]
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.IsNaN(float64(y)) || math.IsInf(float64(y), 0) ||
			math.Float32bits(x) != math.Float32bits(y) {
			t.Fatalf("%s element %d differs or is non-finite: rollback=%g default=%g", name, i, x, y)
		}
	}
}
