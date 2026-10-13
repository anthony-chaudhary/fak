package model

import (
	"bytes"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

type v41ExpertReuseWitness struct {
	*v41ExpertTileWitness
	uploads   int
	failNext  bool
	declineQ3 bool
	fault     error
}

func (b *v41ExpertReuseWitness) Upload(src compute.Tensor, dt compute.Dtype) compute.Tensor {
	if dt == compute.Q2_K || dt == compute.Q3_K {
		b.uploads++
		if b.failNext {
			panic(b.fault)
		}
	}
	return b.v41ExpertTileWitness.Upload(src, dt)
}

func (b *v41ExpertReuseWitness) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return !(b.declineQ3 && dt == compute.Q3_K) && b.v41ExpertTileWitness.SupportsDeviceWeightDtype(dt)
}

// The actual cohort starts with its up/down resident and older than an unrelated
// gate in an exactly one-triple ring. A cold gate must evict only that unrelated
// weight. Without demand protection, acquisition evicts and rereads up/down too.
// This records real checkpoint faults, uploads and token-tile consumption on a
// CPU backend; it establishes no physical GPU performance or shader parity.
// fak-test:runtime medium est=5s lane=default
func TestV41ExpertCohortKeepsResidentDemand(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	m := v41IncrementalExpertFixture(t, true, false)
	tier := NewExpertCheckpointTier(0)
	names := func(expert int) [3]string {
		stem := "ffn.experts." + itoa(expert)
		return [3]string{layerName(0, stem+".w1.weight"), layerName(0, stem+".w3.weight"), layerName(0, stem+".w2.weight")}
	}
	selected := names(0)
	var tripleBytes, gateBytes int64
	for p, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
		name := names(0)[p]
		w := m.kqw[name]
		stride := len(w.raw)
		blob := append(append([]byte(nil), w.raw...), m.kqw[names(1)[p]].raw...)
		quant := ExpertCheckpointQ2K
		if p == 2 {
			quant = ExpertCheckpointQ3K
		}
		if err := tier.AddShard(bytes.NewReader(blob), int64(len(blob)), []FusedExpertTensor{{Name: proj, Layer: 0, Proj: proj, Arch: "deepseek41", Quant: quant, Experts: 2, Rows: w.out, Cols: w.in}}); err != nil {
			t.Fatal(err)
		}
		tripleBytes += int64(stride)
		if p == 0 {
			gateBytes = int64(stride)
		}
		delete(m.kqw, name)
		delete(m.kqw, names(1)[p])
	}
	if err := tier.SetDeviceRingBudget(tripleBytes); err != nil {
		t.Fatal(err)
	}
	m.SetExpertCheckpoint(tier)
	inputs, weights := make([][]float32, 8), make([]float32, 8)
	for row := range inputs {
		inputs[row] = make([]float32, m.Cfg.HiddenSize)
		for i := range inputs[row] {
			inputs[row][i] = float32((row+i)%13-6) / 16
		}
		weights[row] = float32(row+1) / 32
	}
	for _, shared := range []bool{false, true} {
		for _, mode := range []string{"reuse", "decline", "upload_failure"} {
			t.Run(map[bool]string{false: "private/", true: "shared/"}[shared]+mode, func(t *testing.T) {
				b := &v41ExpertReuseWitness{v41ExpertTileWitness: &v41ExpertTileWitness{
					v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true},
					tiles:                  true, ownedTiles: map[compute.Buffer]bool{},
				}}
				s, err := m.NewBackendSessionChecked(b)
				if err != nil {
					t.Fatal(err)
				}
				cleanup := s.v41HoldResidentExpertTriple(selected[0], selected[1], selected[2])
				cleanup()
				if s.expertRing != nil {
					t.Fatal("protection eagerly initialized a cold ring")
				}
				if shared {
					sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: b, BudgetBytes: tripleBytes})
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() {
						if err := sh.Close(); err != nil {
							t.Error(err)
						}
					})
					if err := sh.Attach(s, "cohort-demand-reuse"); err != nil {
						t.Fatal(err)
					}
				}
				t.Cleanup(s.Close)
				for _, name := range []string{names(0)[1], names(0)[2], names(1)[0]} {
					_, release, ok := s.expertInputDeviceWeight(name)
					if !ok {
						t.Fatalf("warm %s declined", name)
					}
					release()
				}
				r := s.expertRing
				resident := map[polymodel.ModelID]compute.Buffer{}
				for _, name := range selected[1:] {
					w, _ := s.resolveExpertWeight(name)
					key := polymodel.ModelID(w.halKey())
					resident[key] = r.resident[key].Buf()
				}
				var pinnedKey polymodel.ModelID
				if mode == "decline" {
					w, _ := s.resolveExpertWeight(names(0)[1])
					pinnedKey = polymodel.ModelID(w.halKey())
					pinned, _ := r.pool.Get(pinnedKey)
					pinned.Pinned = true
					r.repin(pinnedKey, pinned)
					b.declineQ3 = true
				}
				if mode == "upload_failure" {
					b.failNext = true
					b.fault = &compute.BackendError{Backend: "test", Site: "Upload", Err: compute.ErrVulkanExecutionFailed}
				}
				before, sourceBefore, uploads := s.ExpertRing(), tier.Stats(), b.uploads
				if before.ResidentBytes != tripleBytes || before.ResidentCount != 3 {
					t.Fatalf("ring fixture is not full: %+v", before)
				}
				out, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
				if len(r.holds) != 0 || len(r.heldPins) != 0 || s.ringDepth != 0 {
					t.Fatalf("leaked holds=%v temporary pins=%v depth=%d", r.holds, r.heldPins, s.ringDepth)
				}
				switch mode {
				case "decline":
					if out != nil || outcome != v41RoutedExpertDeclined || err != nil || len(b.tileRows) != 0 {
						t.Fatalf("dtype decline selected arithmetic: %v/%v", outcome, err)
					}
					if pinned, ok := r.pool.Get(pinnedKey); !ok || !pinned.Pinned {
						t.Fatal("cleanup removed the durable pin")
					}
				case "upload_failure":
					if out != nil || outcome != v41RoutedExpertError || !errors.Is(err, b.fault) || !s.BackendSessionClosed() || b.uploads != uploads+1 || len(b.tileRows) != 0 {
						t.Fatalf("selected failure replayed or changed: %v/%v", outcome, err)
					}
				case "reuse":
					if outcome != v41RoutedExpertHandled || err != nil || len(out) != len(inputs) || !reflect.DeepEqual(b.tileRows, []int{8, 8}) {
						t.Fatalf("token tile cohort did not run: %v/%v tiles=%v", outcome, err, b.tileRows)
					}
					after, sourceAfter := s.ExpertRing(), tier.Stats()
					if b.uploads-uploads != 1 || after.PageIns-before.PageIns != 1 || after.Hits-before.Hits != 2 || after.Evictions-before.Evictions != 1 || after.PageInBytes-before.PageInBytes != gateBytes {
						t.Fatalf("resident demand reuploaded: uploads=%d before=%+v after=%+v", b.uploads-uploads, before, after)
					}
					if sourceAfter.Reads-sourceBefore.Reads != 1 || sourceAfter.BytesRead-sourceBefore.BytesRead != gateBytes || sourceAfter.Prefetched != sourceBefore.Prefetched || sourceAfter.ResidentBytes != 0 || after.PeakBytes > after.BudgetBytes {
						t.Fatalf("checkpoint traffic or bounds changed: before=%+v after=%+v ring=%+v", sourceBefore, sourceAfter, after)
					}
					for key, handle := range resident {
						if r.resident[key].Buf() != handle {
							t.Fatalf("resident projection %s was replaced", key)
						}
					}
					// The recorder checks each call's panel lifetimes independently.
					b.panels = nil
					warm, outcome, err := s.v41RoutedExpertCohort(0, "ffn.experts.0", inputs, weights)
					if outcome != v41RoutedExpertHandled || err != nil || !reflect.DeepEqual(out, warm) || b.uploads != uploads+1 || tier.Stats().Reads != sourceAfter.Reads || len(r.holds) != 0 || len(r.heldPins) != 0 || s.ringDepth != 0 {
						t.Fatalf("warm replay lost arithmetic/reuse/cleanup: %v/%v", outcome, err)
					}
				}
			})
		}
	}
}
