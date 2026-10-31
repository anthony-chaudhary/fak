//go:build cuda && cgo && fak_cuda_capture_once_qualification && fak_cuda_capture_once_device_witness

package model

import (
	"github.com/anthony-chaudhary/fak/internal/compute"
	"math"
	"os"
	"reflect"
	"testing"
)

// fak-test:runtime integration est=5s lane=cuda
// Real model + real CUDA only. Sequential sessions avoid claiming concurrency.
func TestCUDAGraphOnceRequiredDenseModel(t *testing.T) {
	for _, key := range []string{"FAK_CUDA_REQUIRE_DEVICE", "FAK_CUDA_GRAPH", "FAK_CUDA_CAPTURE_ONCE"} {
		if os.Getenv(key) != "1" {
			t.Fatalf("%s=1 required; no skip/fallback", key)
		}
	}
	be, ok := compute.Lookup("cuda")
	if !ok || be == nil || be.Name() != "cuda" || !be.Caps().DeviceMemory || !be.Caps().GraphCompile {
		t.Fatal("real graph-enabled CUDA backend required")
	}
	witness, ok := be.(interface {
		GraphOnceQualificationSnapshot() compute.CUDAGraphOnceQualificationSnapshot
	})
	if !ok {
		t.Fatal("qualified CUDA witness surface missing")
	}
	once, ok := be.(captureOnceBackend)
	if !ok {
		t.Fatal("capture-once model binding missing")
	}
	initial := witness.GraphOnceQualificationSnapshot()
	if initial.Owner != 0 {
		t.Fatal("another graph owner is live; require isolated witness process")
	}
	cfg := Config{HiddenSize: 64, NumLayers: 2, NumHeads: 2, NumKVHeads: 1, HeadDim: 32, IntermediateSize: 128, VocabSize: 32, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true, EOSTokenID: -1}
	m := NewSynthetic(cfg) // Existing fixed-seed, in-memory production fixture; no downloads.
	defer func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	}()
	ids := []int{3, 17, 5, 23, 11, 2, 19, 7}
	readKV := func(s *Session) compute.KVHostSnapshot {
		t.Helper()
		snap, ok := s.halKV.(compute.KVHostSnapshotter)
		if !ok {
			t.Fatal("real KV snapshot surface missing")
		}
		out, err := snap.SnapshotToHost()
		if err != nil {
			t.Fatal(err)
		}
		if out.Config.Precision != compute.KVPrecisionF32 {
			t.Fatal("witness requires contiguous F32 KV")
		}
		return out
	}
	checkKV := func(want, got compute.KVHostSnapshot) {
		t.Helper()
		if !reflect.DeepEqual(want.Pos, got.Pos) || len(want.K) != len(got.K) || len(want.KRaw) != len(got.KRaw) || len(want.V) != len(got.V) {
			t.Fatal("KV geometry/positions differ")
		}
		for _, pair := range [][2][][]float32{{want.K, got.K}, {want.KRaw, got.KRaw}, {want.V, got.V}} {
			for l := range pair[0] {
				if len(pair[0][l]) != len(pair[1][l]) {
					t.Fatal("KV row extent differs")
				}
				for i, w := range pair[0][l] {
					g := pair[1][l][i]
					if math.IsNaN(float64(w)) || math.IsInf(float64(w), 0) || math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) || w != g {
						t.Fatalf("F32 KV differs layer=%d element=%d want=%g got=%g", l, i, w, g)
					}
				}
			}
		}
	}
	checkLogits := func(want, got []float32) {
		t.Helper()
		if len(want) == 0 || len(want) != len(got) {
			t.Fatal("logit extent differs")
		}
		var dot, wn, gn float64
		wa, ga := 0, 0
		for i, w := range want {
			g := got[i]
			a, b := float64(w), float64(g)
			if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(b) || math.IsInf(b, 0) {
				t.Fatal("nonfinite logits")
			}
			dot += a * b
			wn += a * a
			gn += b * b
			if w > want[wa] {
				wa = i
			}
			if g > got[ga] {
				ga = i
			}
		}
		// Existing CUDA graph Approx contract (cuda_graph_test.go), never tuned to results.
		if wn == 0 || gn == 0 || wa != ga || dot/math.Sqrt(wn*gn) < 0.999 {
			t.Fatal("graph/eager Approx gate failed: argmax-exact and cosine >=0.999 required")
		}
	}
	t.Setenv("FAK_CUDA_CAPTURE_ONCE", "0")
	ref, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer ref.Close()
	referenceKV := ref.halKV
	expected := make([][]float32, len(ids))
	expectedKV := make([]compute.KVHostSnapshot, len(ids))
	legacyBefore := compute.CUDAGraphReplays()
	for i, id := range ids {
		ref.halLogitsWarm = false // Explicit ordinary forward; keep identical fixed-capacity KV.
		expected[i] = append([]float32(nil), ref.Step(id)...)
		expectedKV[i] = readKV(ref)
	}
	if !ref.captureOnceF32Weights() || m.Q2KEmbedding != nil {
		t.Fatal("fixture escaped dense F32 weight scope")
	}
	afterRef := witness.GraphOnceQualificationSnapshot()
	if afterRef.Captures != initial.Captures || afterRef.Replays != initial.Replays || afterRef.Owner != 0 || compute.CUDAGraphReplays() != legacyBefore {
		t.Fatal("ordinary reference captured a graph")
	}
	ref.Close()
	t.Setenv("FAK_CUDA_CAPTURE_ONCE", "1")
	selected, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer selected.Close()
	if selected.halKV == referenceKV {
		t.Fatal("reference and selected share KV storage")
	}
	var stable compute.CUDAGraphOnceQualificationSnapshot
	for i, id := range ids {
		got := selected.Step(id)
		checkLogits(expected[i], got)
		checkKV(expectedKV[i], readKV(selected))
		if !selected.captureOnceF32Weights() {
			t.Fatal("selected weights are not F32")
		}
		now := witness.GraphOnceQualificationSnapshot()
		if i == 0 {
			if now.Captures != initial.Captures || now.Replays != initial.Replays {
				t.Fatal("first token must be uncaptured warmup")
			}
			continue
		}
		if now.Captures-initial.Captures != 1 || now.Replays-initial.Replays != uint64(i-1) || now.Next != i+1 || now.Owner == 0 || now.Input == 0 || now.Output == 0 || !now.SharedOwner || !now.ArenaPinned || !now.DenseF32KV || now.Borrowed == 0 || now.Capturing || now.Invalidated {
			t.Fatalf("capture/replay/arena state invalid at token %d: %+v", i, now)
		}
		if i == 1 {
			stable = now
		} else if now.Owner != stable.Owner || now.Input != stable.Input || now.Output != stable.Output || !reflect.DeepEqual(now.Arena, stable.Arena) || !reflect.DeepEqual(now.KV, stable.KV) {
			t.Fatal("graph addresses changed during replay")
		}
	}
	fresh := be.Upload(compute.NewF32(be, []int{cfg.HiddenSize}, make([]float32, cfg.HiddenSize)), compute.F32)
	defer be.Free(fresh)
	beforeReject := witness.GraphOnceQualificationSnapshot()
	if _, used := once.GraphOnceReplay(fresh, selected.halKV, len(ids)+1); used {
		t.Fatal("out-of-order replay accepted")
	}
	afterReject := witness.GraphOnceQualificationSnapshot()
	if afterReject.Owner != 0 || len(afterReject.Arena) != 0 || afterReject.Captures != beforeReject.Captures || afterReject.Replays != beforeReject.Replays {
		t.Fatal("refusal failed to tear down without launch")
	}
	checkKV(expectedKV[len(ids)-1], readKV(selected))
	selected.Close()
	if witness.GraphOnceQualificationSnapshot().Owner != 0 {
		t.Fatal("session close retained graph owner")
	}
	t.Log("real dense-F32 model: 1 warmup, 1 capture, 6 replays; logits Approx parity, exact F32 KV/positions, stable arena, refusal and teardown; no throughput claim")
}
