//go:build darwin && arm64 && cgo

package model

import (
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func qwen35RestoredMetalModel(t *testing.T) (*Model, []int) {
	t.Helper()
	if !metalgemm.Available() {
		t.Skip("Metal device unavailable")
	}
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	prompt := make([]int, 96)
	for i := range prompt {
		prompt[i] = (i*31 + 9) % cfg.VocabSize
	}
	return m, prompt
}

func qwen35ResidentQ4KSession(m *Model) *Session {
	s := m.NewSession()
	s.Q4K, s.MetalQ4K = true, true
	return s
}

func assertQwen35RestoredParity(t *testing.T, label string, want, got []float32) {
	t.Helper()
	assertCosineAtLeast(t, label, want, got, Qwen35GDNParityCosineMin)
	if argmax(want) != argmax(got) {
		t.Fatalf("%s argmax=%d, want %d", label, argmax(got), argmax(want))
	}
}

// T1: a session restored from a host prefix admits the resident sequence at
// base>0 with owners seeded from the restored Cache.linear, matching an
// uncached reference through prefill and the first resident decode step.
func TestMetalQwen35RestoredPrefixSeedsResidentSequenceParity(t *testing.T) {
	m, prompt := qwen35RestoredMetalModel(t)
	runQwen35RestoredPrefixSeedParity(t, m, prompt, 32)
}

// T1 short-history variant: the restored prefix is shorter than K-1, so the host
// carries fewer convolution rows than the owner layout and the seed must supply
// the causal zero padding as the oldest rows.
func TestMetalQwen35RestoredShortPrefixSeedsResidentSequenceParity(t *testing.T) {
	m, prompt := qwen35RestoredMetalModel(t)
	short := m.Cfg.LinearConvKernelDim - 2
	if short < 1 {
		t.Skipf("kernel K=%d leaves no restored prefix shorter than K-1", m.Cfg.LinearConvKernelDim)
	}
	runQwen35RestoredPrefixSeedParity(t, m, prompt, short)
}

func runQwen35RestoredPrefixSeedParity(t *testing.T, m *Model, prompt []int, base int) {
	t.Helper()
	const panel = 32
	end := base + panel
	ref := qwen35ResidentQ4KSession(m)
	defer ref.Close()
	ref.PrefillNoLogits(prompt[:base])
	want := ref.Prefill(prompt[base:end])

	donor := qwen35ResidentQ4KSession(m)
	donor.PrefillNoLogits(prompt[:base])
	snap, err := donor.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	donor.Close()
	got := qwen35ResidentQ4KSession(m)
	defer got.Close()
	if err := snap.Restore(got); err != nil {
		t.Fatal(err)
	}
	snap.Close()
	if got.Cache.Len() != base {
		t.Fatalf("restored cache len=%d, want %d", got.Cache.Len(), base)
	}
	if keep := m.Cfg.LinearConvKernelDim - 1; base < keep {
		for layer := 0; layer < m.Cfg.NumLayers; layer++ {
			if m.Cfg.isLinearAttnLayer(layer) && len(got.Cache.linear.layers[layer].conv) != base {
				t.Fatalf("layer %d restored conv rows=%d, want short history %d", layer, len(got.Cache.linear.layers[layer].conv), base)
			}
		}
	}
	if err := got.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatalf("restored-prefix admission: %v", err)
	}
	owners := append([]Qwen35GDNAuxState(nil), got.qwen35HAL.sequenceLayers...)
	actual := got.Prefill(prompt[base:end])
	receipt := got.Qwen35MetalForwardSequenceReceipt()
	if !receipt.Available || receipt.Tokens != panel || !receipt.Committed {
		t.Fatalf("restored base=%d sequence receipt=%+v", base, receipt)
	}
	if got.Cache.Len() != end || got.q4kHybridPrefillLastBase != base {
		t.Fatalf("restored append cache=%d base=%d, want %d/%d", got.Cache.Len(), got.q4kHybridPrefillLastBase, end, base)
	}
	if !reflect.DeepEqual(got.qwen35HAL.sequenceLayers, owners) {
		t.Fatal("restored prefill replaced resident GDN owner identity")
	}
	assertQwen35RestoredParity(t, "restored base>0 sequence logits", want, actual)
	if executed, err := got.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !executed {
		t.Fatalf("finalize executed=%v err=%v", executed, err)
	}
	if path, selected := got.Qwen35GDNDecodePath(); !selected || path != Qwen35MetalGDNDecodeForwardPath {
		t.Fatalf("decode path=(%q,%v), want resident Metal", path, selected)
	}
	next := argmax(want)
	assertQwen35RestoredParity(t, "restored resident decode", ref.Step(next), got.Step(next))
}

// T2: a PrefixSnapshot (and a bare post-finalize KV clone) of a session whose
// resident owners hold live GDN state carries that state into a plain host
// session, and the owner session itself continues correctly afterwards.
func TestMetalQwen35OwnerBackedPrefixSnapshotCarriesGDNState(t *testing.T) {
	m, prompt := qwen35RestoredMetalModel(t)
	ref := qwen35ResidentQ4KSession(m)
	defer ref.Close()
	ref.PrefillNoLogits(prompt[:32])
	refPrefix := ref.Cache.Clone() // host-recurrence state at the checkpoint
	// Prefill/Step return the session-owned logits buffer (headLogitsBuf reuses
	// s.qDecode.Logits), so each reference is copied before the next ref call
	// overwrites it in place.
	want := append([]float32(nil), ref.Prefill(prompt[32:64])...)
	next := argmax(want)
	wantNext := append([]float32(nil), ref.Step(next)...)
	next2 := argmax(wantNext)
	wantNext2 := append([]float32(nil), ref.Step(next2)...)

	got := qwen35ResidentQ4KSession(m)
	defer got.Close()
	if err := got.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatal(err)
	}
	got.PrefillNoLogits(prompt[:32])

	// Sequence-phase checkpoint, as the agent takes mid-request.
	snap, err := got.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	// The logits below are only weakly sensitive to a stale prefix state (the
	// last panel row sees two conv rows and a decayed recurrence), so witness the
	// carried linear-attention state directly against host recurrence.
	assertQwen35LinearStateClose(t, m.Cfg, refPrefix, snap.Cache, 1e-4)
	plain := qwen35ResidentQ4KSession(m)
	defer plain.Close()
	if err := snap.Restore(plain); err != nil {
		t.Fatal(err)
	}
	snap.Close()
	assertQwen35RestoredParity(t, "sequence-checkpoint restored host prefill", want, plain.Prefill(prompt[32:64]))
	assertQwen35RestoredParity(t, "owner session continues after checkpoint", want, got.Prefill(prompt[32:64]))

	if executed, err := got.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !executed {
		t.Fatalf("finalize executed=%v err=%v", executed, err)
	}
	// Bare host KV clone at the full-prompt admission boundary.
	bare := m.NewSession()
	defer bare.Close()
	bare.Q4K, bare.MetalQ4K = true, true
	bare.Cache = got.Cache.Clone()
	assertQwen35RestoredParity(t, "post-finalize bare clone decode", wantNext, bare.Step(next))

	assertQwen35RestoredParity(t, "owner resident decode", wantNext, got.Step(next))
	// Decode-phase snapshot: owners advanced past the finalized host copy.
	snap, err = got.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	plain2 := qwen35ResidentQ4KSession(m)
	defer plain2.Close()
	if err := snap.Restore(plain2); err != nil {
		t.Fatal(err)
	}
	snap.Close()
	assertQwen35RestoredParity(t, "decode-checkpoint restored host step", wantNext2, plain2.Step(next2))
}

// T3: a batched prefill on a session already promoted to resident decode
// continues from the live owner state (demoted to the host), not from the
// stale prompt-boundary host copy.
func TestMetalQwen35PrefillAfterResidentDecodeUsesLiveState(t *testing.T) {
	m, prompt := qwen35RestoredMetalModel(t)
	ref := qwen35ResidentQ4KSession(m)
	defer ref.Close()
	want := ref.Prefill(prompt[:32])
	next := argmax(want)
	ref.Step(next)
	wantMore := ref.Prefill(prompt[32:64])

	got := qwen35ResidentQ4KSession(m)
	defer got.Close()
	if err := got.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatal(err)
	}
	got.Prefill(prompt[:32])
	if executed, err := got.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !executed {
		t.Fatalf("finalize executed=%v err=%v", executed, err)
	}
	got.Step(next)
	assertQwen35RestoredParity(t, "prefill after resident decode", wantMore, got.Prefill(prompt[32:64]))
	if _, selected := got.Qwen35GDNDecodePath(); selected {
		t.Fatal("non-auto-eligible session kept stale resident owners after host prefill")
	}
	nextMore := argmax(wantMore)
	assertQwen35RestoredParity(t, "decode after demoted prefill", ref.Step(nextMore), got.Step(nextMore))
}

// assertQwen35LinearStateClose compares every linear-attention layer's conv rows
// and recurrent heads of got against want within an absolute tolerance.
func assertQwen35LinearStateClose(t *testing.T, cfg Config, want, got *KVCache, tol float64) {
	t.Helper()
	for layer := 0; layer < cfg.NumLayers; layer++ {
		if !cfg.isLinearAttnLayer(layer) {
			continue
		}
		w, g := &want.linear.layers[layer], &got.linear.layers[layer]
		if len(w.conv) != len(g.conv) || len(w.recurrent) != len(g.recurrent) {
			t.Fatalf("layer %d linear state rows conv=%d/%d recurrent=%d/%d", layer, len(g.conv), len(w.conv), len(g.recurrent), len(w.recurrent))
		}
		for i := range w.conv {
			assertQwen35RowsClose(t, layer, "conv", i, w.conv[i], g.conv[i], tol)
		}
		for i := range w.recurrent {
			assertQwen35RowsClose(t, layer, "recurrent", i, w.recurrent[i], g.recurrent[i], tol)
		}
	}
}

func assertQwen35RowsClose(t *testing.T, layer int, kind string, row int, want, got []float32, tol float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("layer %d %s row %d elements=%d, want %d", layer, kind, row, len(got), len(want))
	}
	for i := range want {
		if d := math.Abs(float64(want[i] - got[i])); d > tol {
			t.Fatalf("layer %d %s row %d[%d]=%g, want %g (|diff| %g > %g)", layer, kind, row, i, got[i], want[i], d, tol)
		}
	}
}
