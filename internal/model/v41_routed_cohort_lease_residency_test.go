package model

import (
	"bytes"
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

type v41ResidentLeaseWitness struct {
	compute.Backend
	owner         sync.Mutex
	weightUploads int
	declineDown   bool
	uploadFault   error
}

func (b *v41ResidentLeaseWitness) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *v41ResidentLeaseWitness) checkOwner() {
	if b.owner.TryLock() {
		b.owner.Unlock()
		panic("unowned routed cohort acquisition")
	}
}
func (b *v41ResidentLeaseWitness) Caps() compute.Caps {
	caps := b.Backend.Caps()
	caps.DeviceMemory, caps.UploadDtype = true, true
	return caps
}

func (b *v41ResidentLeaseWitness) Upload(src compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.checkOwner()
	if dt == compute.Q2_K || dt == compute.Q3_K {
		b.weightUploads++
		if b.uploadFault != nil {
			panic(b.uploadFault)
		}
	}
	return b.Backend.Upload(src, dt)
}

func (b *v41ResidentLeaseWitness) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.Q2_K || dt == compute.Q3_K && !b.declineDown
}
func (b *v41ResidentLeaseWitness) Free(x compute.Tensor) {
	b.checkOwner()
	b.Backend.Free(x)
}

// Advertise the real acquisition requirements, but reject all arithmetic. This
// fixture qualifies lease preparation and ownership only, never kernel parity.
func (*v41ResidentLeaseWitness) V41RoutedSwiGLUAvailable() bool       { return true }
func (*v41ResidentLeaseWitness) SupportsV41DeferredStatus(n int) bool { return n > 0 }
func (*v41ResidentLeaseWitness) SupportsV41BF16Publish(n int) bool    { return n > 0 }
func (*v41ResidentLeaseWitness) MatMul(compute.Tensor, compute.Tensor) compute.Tensor {
	panic("arithmetic during routed cohort acquisition")
}
func (*v41ResidentLeaseWitness) V41RoutedSwiGLU(compute.Tensor, compute.Tensor, float32, float32) (compute.Tensor, error) {
	panic("arithmetic during routed cohort acquisition")
}
func (*v41ResidentLeaseWitness) V41BF16Publish(compute.Tensor, uint32) (compute.Tensor, error) {
	panic("publication during routed cohort acquisition")
}
func (*v41ResidentLeaseWitness) V41RoutedSwiGLUDeferred(compute.Tensor, compute.Tensor, float32, float32) (compute.Tensor, compute.V41DeferredStatus, error) {
	panic("arithmetic during routed cohort acquisition")
}
func (*v41ResidentLeaseWitness) V41BF16PublishDeferred(compute.Tensor, uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	panic("publication during routed cohort acquisition")
}
func (*v41ResidentLeaseWitness) V41ReadDeferred([]compute.V41DeferredStatus, []compute.Tensor) ([]error, [][]float32, error) {
	panic("readback during routed cohort acquisition")
}

// A cold gate must not evict this lease's older resident up/down weights while
// acquiring its triple. The witness uses real checkpoint reads and a CPU owner
// recorder; it makes no claim about physical I/O or device execution time.
// fak-test:runtime fast est=10ms lane=default
func TestV41RoutedCohortLeaseKeepsResidentDemand(t *testing.T) {
	t.Setenv(v41DeferredBF16PublishEnv, "1")
	t.Setenv(v41RoutedIDEnv, "0")
	t.Setenv(v41RoutedFusedEnv, "0")
	t.Setenv(v41LayerExpertBatchEnv, "1")
	// Only lease admission runs: no constructor, forward, KV or router fixture
	// is involved. Packed checkpoint bytes still take the actual source/staging
	// path. The full arithmetic protocol has its own independent witness.
	m := &Model{Cfg: Config{NumLayers: 1, HeadDim: v41KVLoraRank, HiddenSize: qkK, MoEIntermediateSize: qkK, NumExperts: 2}}
	tier := NewExpertCheckpointTier(0)
	names := func(expert int) [3]string {
		stem := "ffn.experts." + itoa(expert)
		return [3]string{layerName(0, stem+".w1.weight"), layerName(0, stem+".w3.weight"), layerName(0, stem+".w2.weight")}
	}
	selected := names(0)
	var tripleBytes, gateBytes int64
	for p, proj := range []string{"gate_proj", "up_proj", "down_proj"} {
		quant := ExpertCheckpointQ2K
		if p == 2 {
			quant = ExpertCheckpointQ3K
		}
		blockWeights, blockBytes, ok := quant.blockGeometry()
		if !ok {
			t.Fatal("checkpoint fixture dtype has no block geometry")
		}
		stride := qkK * (qkK / blockWeights) * blockBytes
		blob := make([]byte, 2*stride)
		// Distinct payloads identify the selected and unrelated experts. These
		// raw packed values are staged, never submitted to arithmetic.
		for i := range blob {
			blob[i] = byte(i/stride + 1)
		}
		if err := tier.AddShard(bytes.NewReader(blob), int64(len(blob)), []FusedExpertTensor{{Name: proj, Layer: 0, Proj: proj, Arch: "deepseek41", Quant: quant, Experts: 2, Rows: qkK, Cols: qkK}}); err != nil {
			t.Fatal(err)
		}
		tripleBytes += int64(stride)
		if p == 0 {
			gateBytes = int64(stride)
		}
	}
	if err := tier.SetDeviceRingBudget(tripleBytes); err != nil {
		t.Fatal(err)
	}
	m.SetExpertCheckpoint(tier)
	for _, shared := range []bool{false, true} {
		for _, mode := range []string{"reuse", "nested", "decline", "upload_failure"} {
			t.Run(map[bool]string{false: "private/", true: "shared/"}[shared]+mode, func(t *testing.T) {
				b := &v41ResidentLeaseWitness{Backend: compute.Default()}
				sessions := make([]*Session, 2)
				for i := range sessions {
					sessions[i] = &Session{M: m, Backend: b, ExpertRingBytes: tripleBytes, halW: map[string]compute.Tensor{}}
					t.Cleanup(sessions[i].Close)
				}
				s := sessions[0]
				if shared {
					sh, err := NewSharedExpertRing(SharedExpertRingConfig{Model: m, Backend: b, BudgetBytes: tripleBytes})
					if err != nil {
						t.Fatal(err)
					}
					// Session cleanup must precede the shared ring's final close.
					defer func() {
						for _, session := range sessions {
							session.Close()
						}
						if err := sh.Close(); err != nil {
							t.Error(err)
						}
					}()
					for i, session := range sessions {
						if err := sh.Attach(session, "decode-lease-demand-reuse-"+itoa(i)); err != nil {
							t.Fatal(err)
						}
					}
				}
				scope := beginBackendExecution(b)
				defer scope.finish()
				for _, session := range sessions {
					scope.bind(session)
				}
				for _, name := range []string{selected[1], selected[2], names(1)[0]} {
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
				releaseOuter := func() {}
				wantHolds := 1
				if mode == "nested" {
					releaseOuter = s.v41HoldResidentExpertTriple(selected[0], selected[1], selected[2])
					defer releaseOuter()
					wantHolds++
				}
				b.declineDown = mode == "decline"
				if mode == "upload_failure" {
					b.uploadFault = &compute.BackendError{Backend: "test", Site: "Upload", Err: compute.ErrVulkanExecutionFailed}
				}
				before, sourceBefore, uploads := s.ExpertRing(), tier.Stats(), b.weightUploads
				if before.ResidentBytes != tripleBytes || before.ResidentCount != 3 {
					t.Fatalf("ring fixture is not full: %+v", before)
				}
				rows := []v41RoutedCohortRow{
					{Lane: 2, Session: sessions[0], Input: make([]float32, m.Cfg.HiddenSize), Weight: .5},
					{Lane: 7, Session: sessions[1], Input: make([]float32, m.Cfg.HiddenSize), Weight: .25},
				}
				lease, outcome, lane, err := prepareV41RoutedExpertCohortOwned(scope, 0, "ffn.experts.0", rows)
				if lease != nil {
					defer lease.Close()
				}
				switch mode {
				case "reuse", "nested":
					if outcome != v41RoutedExpertHandled || err != nil || lane != -1 || lease == nil {
						t.Fatalf("prepare failed: outcome=%v lane=%d err=%v", outcome, lane, err)
					}
					after, sourceAfter := s.ExpertRing(), tier.Stats()
					if b.weightUploads-uploads != 1 || after.PageIns-before.PageIns != 1 || after.Hits-before.Hits != 2 || after.Evictions-before.Evictions != 1 || after.PageInBytes-before.PageInBytes != gateBytes {
						t.Fatalf("resident demand reuploaded: uploads=%d before=%+v after=%+v", b.weightUploads-uploads, before, after)
					}
					if sourceAfter.Reads-sourceBefore.Reads != 1 || sourceAfter.BytesRead-sourceBefore.BytesRead != gateBytes || sourceAfter.Prefetched != sourceBefore.Prefetched || sourceAfter.ResidentBytes != 0 || after.PeakBytes > after.BudgetBytes {
						t.Fatalf("checkpoint traffic or bounds changed: before=%+v after=%+v ring=%+v", sourceBefore, sourceAfter, after)
					}
					for key, handle := range resident {
						if r.resident[key].Buf() != handle || r.holds[key] != wantHolds {
							t.Fatalf("resident projection %s was replaced or retains a temporary hold", key)
						}
					}
				case "decline":
					if lease != nil || outcome != v41RoutedExpertDeclined || err != nil || lane != -1 {
						t.Fatalf("down dtype decline changed: outcome=%v lane=%d err=%v", outcome, lane, err)
					}
				case "upload_failure":
					if lease != nil || outcome != v41RoutedExpertError || !errors.Is(err, b.uploadFault) || b.weightUploads != uploads+1 {
						t.Fatalf("selected failure replayed or changed: outcome=%v err=%v", outcome, err)
					}
					for _, session := range sessions {
						if !session.BackendSessionClosed() {
							t.Fatal("selected failure left a cohort participant reusable")
						}
					}
				}
				lease.Close()
				lease.Close()
				if mode == "nested" {
					for key := range resident {
						if pinned, ok := r.pool.Get(key); !ok || !pinned.Pinned || r.holds[key] != 1 {
							t.Fatalf("closing the lease consumed outer hold %s", key)
						}
					}
				}
				releaseOuter()
				if len(r.holds) != 0 || len(r.heldPins) != 0 || s.ringDepth != 0 {
					t.Fatalf("leaked holds=%v temporary pins=%v depth=%d", r.holds, r.heldPins, s.ringDepth)
				}
			})
		}
	}
}
