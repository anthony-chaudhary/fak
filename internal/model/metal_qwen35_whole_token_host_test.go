//go:build darwin && arm64 && cgo

package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestQwen35WholeTokenDecodeMirrorHeadAndFlushParity holds the whole-token decode's
// host-gap levers to the historical one-command-buffer, host-prefix, separate-LM-head
// decode on the same session shape: the segmented commit, the in-graph LM head and the
// decode KV mirror (seeded from the host prefill, caught up, regrown past its first
// capacity and reseeded after a cache mutation) must all leave every step's logits and
// the host KV cache unchanged.
//
// fak-test:runtime integration est=5s lane=default
func TestQwen35WholeTokenDecodeMirrorHeadAndFlushParity(t *testing.T) {
	if !metalgemm.Available() || metalgemm.DeviceName() == "" {
		t.Skip("whole-token decode requires a physical Metal device")
	}
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	cfg.QKNorm = true
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*19 + 7) % cfg.VocabSize
	}
	// Past the first mirror capacity (need+256 rows) so the regrow path runs.
	const steps = 300
	const mutateAt = 150
	ids := make([]int, steps)
	for i := range ids {
		ids[i] = (i*37 + 11) % cfg.VocabSize
	}
	attnLayers := 0
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			attnLayers++
		}
	}

	type stepRecord struct {
		logits      []float32
		receipt     Qwen35MetalForwardSequenceReceipt
		dispatchCBs int
	}
	run := func(levers bool) (*Session, []stepRecord) {
		t.Helper()
		on := map[bool]string{false: "0", true: "1"}[levers]
		t.Setenv("FAK_QWEN35_DECODE_DKV", on)
		t.Setenv("FAK_QWEN35_WHOLE_TOKEN_HEAD", on)
		if levers {
			t.Setenv("FAK_QWEN35_WHOLE_TOKEN_FLUSH", "")
		} else {
			t.Setenv("FAK_QWEN35_WHOLE_TOKEN_FLUSH", "0")
		}
		t.Setenv("FAK_QWEN35_PERSISTENT_DECODE_DKV", "0")
		s := m.NewSession()
		s.Q4K, s.MetalQ4K = true, true
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatal(err)
		}
		s.Prefill(prompt)
		if executed, err := s.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !executed {
			t.Fatalf("finalize levers=%v executed=%v err=%v", levers, executed, err)
		}
		out := make([]stepRecord, steps)
		for i, id := range ids {
			if levers && i == mutateAt {
				// A cache mutation the mirror cannot see row-by-row must force a reseed.
				s.Cache.mutGen++
			}
			s.ResetMetalCommandBuffers()
			out[i].logits = append([]float32(nil), s.Step(id)...)
			out[i].dispatchCBs = s.MetalDispatchCommandBuffers()
			out[i].receipt = s.Qwen35MetalForwardSequenceReceipt()
		}
		return s, out
	}

	control, want := run(false)
	defer control.Close()
	candidate, got := run(true)
	defer candidate.Close()

	kvRowBytes := uint64(2*attnLayers*cfg.NumKVHeads*cfg.HeadDim) * 4
	var steadyUpload uint64
	var fullReseeds []int
	for i := range want {
		if c := cosine(want[i].logits, got[i].logits); c < 0.9999 || argmaxF(want[i].logits) != argmaxF(got[i].logits) {
			t.Fatalf("step %d logits cosine=%g argmax=%d/%d", i, c, argmaxF(want[i].logits), argmaxF(got[i].logits))
		}
		w, g := want[i].receipt, got[i].receipt
		if w.CommandBuffers != 1 || w.DeviceKVAttentionLayers != 0 {
			t.Fatalf("control step %d receipt=%+v, want the one-buffer host-prefix decode", i, w)
		}
		// The in-graph head adds exactly the logits row to the one terminal readback and
		// leaves no separate head dispatch.
		if g.CommandBuffers <= 1 || g.DeviceKVAttentionLayers != attnLayers || got[i].dispatchCBs != 0 || !g.CompletedWait ||
			g.HostReadbackBytes != w.HostReadbackBytes+uint64(cfg.VocabSize)*4 {
			t.Fatalf("candidate step %d receipt=%+v dispatch=%d, want segmented device-KV decode with the head in-graph", i, g, got[i].dispatchCBs)
		}
		// The control re-uploads the whole prefix every token; the mirror uploads it only
		// on a (re)seed, which is the same bytes.
		if w.HostUploadBytes < g.HostUploadBytes {
			t.Fatalf("step %d host upload control=%d candidate=%d", i, w.HostUploadBytes, g.HostUploadBytes)
		}
		switch {
		case i == 0:
			if g.HostUploadBytes < uint64(len(prompt))*kvRowBytes {
				t.Fatalf("first step uploaded %d bytes, want the %d-row prompt seed", g.HostUploadBytes, len(prompt))
			}
		case i == 1:
			steadyUpload = g.HostUploadBytes
			if steadyUpload+uint64(len(prompt)+i)*kvRowBytes != w.HostUploadBytes {
				t.Fatalf("steady upload %d, control %d: want exactly the prefix saved", steadyUpload, w.HostUploadBytes)
			}
		case g.HostUploadBytes == steadyUpload:
		case g.HostUploadBytes == steadyUpload+uint64(len(prompt)+i)*kvRowBytes:
			fullReseeds = append(fullReseeds, i)
		default:
			t.Fatalf("step %d uploaded %d bytes, steady state %d", i, g.HostUploadBytes, steadyUpload)
		}
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			continue
		}
		if len(control.Cache.K[l]) != len(candidate.Cache.K[l]) ||
			cosine(control.Cache.Kraw[l], candidate.Cache.Kraw[l]) < 0.9999 ||
			cosine(control.Cache.K[l], candidate.Cache.K[l]) < 0.9999 ||
			cosine(control.Cache.V[l], candidate.Cache.V[l]) < 0.9999 {
			t.Fatalf("host KV cache parity failed at layer %d", l)
		}
	}
	// The mutation reseed and exactly one regrow re-upload the whole prefix; no other
	// step uploads any KV row.
	if len(fullReseeds) != 2 || fullReseeds[0] != mutateAt {
		t.Fatalf("full-prefix reseed steps=%v, want [%d <regrow>]", fullReseeds, mutateAt)
	}
	backend := candidate.qwen35HAL.sequenceBackend.(*metalQwen35GDNSequenceBackend)
	backend.mu.Lock()
	reseeds, rows, capacity := backend.decodeKVReseeds, backend.decodeKVRows, backend.decodeKVCapacity
	backend.mu.Unlock()
	// Initial seed, the mutation reseed, and one regrow past need+256.
	if reseeds != 3 || rows != candidate.Cache.Len() || capacity <= candidate.Cache.Len() {
		t.Fatalf("decode mirror reseeds=%d rows=%d capacity=%d len=%d", reseeds, rows, capacity, candidate.Cache.Len())
	}
}
