package model

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This provider witnesses model dispatch and ownership, not a device kernel or
// numerical parity. Its projection receipts and logits are deliberately synthetic.
type portableQwen35BatchBackend struct {
	*sequencePrefillBackend
	batchPath   string
	batchCalls  int
	batchErr    error
	badReceipt  bool
	shortLogits bool
	mutatedKV   []int
	request     compute.Qwen35IndependentDecodeBatchRequest
	logits      compute.Tensor
}

func (*portableQwen35BatchBackend) Name() string { return "portable-batch-test" }
func (*portableQwen35BatchBackend) Qwen35GDNPath() string {
	return Qwen35GDNCapabilityIdentity
}
func (b *portableQwen35BatchBackend) Qwen35IndependentDecodeBatchPath() string {
	return b.batchPath
}
func (b *portableQwen35BatchBackend) Read(t compute.Tensor) []float32 {
	flat := b.sequencePrefillBackend.Read(t)
	if b.shortLogits && t.Buf() == b.logits.Buf() && len(flat) > 0 {
		return flat[:len(flat)-1]
	}
	return flat
}

func (b *portableQwen35BatchBackend) Qwen35IndependentDecodeBatch(req compute.Qwen35IndependentDecodeBatchRequest) (compute.Qwen35IndependentDecodeBatchResult, error) {
	b.batchCalls++
	b.request = req
	if b.batchErr != nil {
		return compute.Qwen35IndependentDecodeBatchResult{}, b.batchErr
	}
	vocab := b.model.Cfg.VocabSize
	flat := make([]float32, len(req.Lanes)*vocab)
	for i, lane := range req.Lanes {
		for j := 0; j < vocab; j++ {
			flat[i*vocab+j] = float32(lane.TokenIDs[0])
		}
		attention := 0
		for _, layer := range lane.Layers {
			if layer.Linear {
				continue
			}
			width := lane.NumKVHeads * lane.HeadDim
			row := compute.NewF32(b.Backend, []int{width}, make([]float32, width))
			lane.KV.AppendKV(attention, row, row, row, lane.StartPos)
			b.Backend.Free(row)
			attention++
		}
		// Capture real fixture mutation before production retirement can clear KV.
		b.mutatedKV = append(b.mutatedKV, lane.KV.Len())
	}
	b.logits = compute.NewF32(b.Backend, []int{len(req.Lanes), vocab}, flat)
	result := compute.Qwen35IndependentDecodeBatchResult{
		Logits: b.logits, Lanes: len(req.Lanes), SharedProjectionPanels: 3,
		ProjectionMACs: int64(len(req.Lanes)) * 7,
	}
	if b.badReceipt {
		result.SharedProjectionPanels = 0
	}
	return result, nil
}

func newPortableQwen35Batch(t *testing.T, lanes int) (*BatchSession, *portableQwen35BatchBackend, []int) {
	t.Helper()
	m := NewSynthetic(qwen35HybridTestCfg())
	be := &portableQwen35BatchBackend{sequencePrefillBackend: newSequencePrefillBackend(m), batchPath: compute.Qwen35IndependentDecodeBatchPath}
	bs := &BatchSession{M: m, Seqs: make([]*Session, lanes)}
	ids := make([]int, lanes)
	for i := range bs.Seqs {
		var err error
		bs.Seqs[i], err = m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(bs.Seqs[i].Close)
		ids[i] = 3 + 4*i
	}
	return bs, be, ids
}

// Exercise the actual generic selector with no native override. The same test
// source is available without Vulkan or cgo; it makes no hardware support claim.
// fak-test:runtime fast est=100ms lane=default
func TestQwen35IndependentDecodeBatchCapabilityDispatch(t *testing.T) {
	old := qwen35HybridQ4KBatchStep
	t.Cleanup(func() { qwen35HybridQ4KBatchStep = old })
	qwen35HybridQ4KBatchStep = nil

	for _, lanes := range []int{2, 8} {
		t.Run(fmt.Sprintf("B%d", lanes), func(t *testing.T) {
			bs, be, ids := newPortableQwen35Batch(t, lanes)
			fallbacks := 0
			bs.ObserveSerialStep = func(string, time.Duration) { fallbacks++ }
			got := bs.StepBatch(ids)
			if be.batchCalls != 1 || be.gdnCalls != 0 || fallbacks != 0 || bs.LastStepSharedPanels() != 3 || bs.LastStepMACs() != int64(lanes)*7 {
				t.Fatalf("calls=%d scalar=%d fallbacks=%d panels=%d macs=%d", be.batchCalls, be.gdnCalls, fallbacks, bs.LastStepSharedPanels(), bs.LastStepMACs())
			}
			if be.request.Path != compute.Qwen35IndependentDecodeBatchPath || be.freeCalls[be.logits.Buf()] != 1 {
				t.Fatalf("path=%q logits frees=%d", be.request.Path, be.freeCalls[be.logits.Buf()])
			}
			owners := make(map[compute.KVStore]bool)
			states := make(map[compute.Buffer]bool)
			for i, lane := range be.request.Lanes {
				if owners[lane.KV] || lane.KV != bs.Seqs[i].halKV || lane.KV.Len() != 1 || len(got[i]) != bs.M.Cfg.VocabSize || got[i][0] != float32(ids[i]) {
					t.Fatalf("lane %d owner, position, or output mismatch", i)
				}
				owners[lane.KV] = true
				if _, err := bs.Seqs[i].VerifyTokenLineage([]int{ids[i]}); err != nil {
					t.Fatalf("lane %d lineage: %v", i, err)
				}
				for _, state := range lane.States {
					for _, tensor := range []compute.Tensor{state.Conv, state.Recurrent} {
						if buffer := tensor.Buf(); buffer != nil {
							if states[buffer] {
								t.Fatalf("lane %d aliases recurrent state", i)
							}
							states[buffer] = true
						}
					}
				}
			}
		})
	}

	for _, name := range []string{"singleton", "oversized", "wrong_marker", "mixed_backend", "mixed_model", "missing_state"} {
		t.Run(name, func(t *testing.T) {
			lanes := 2
			if name == "singleton" {
				lanes = 1
			} else if name == "oversized" {
				lanes = 9
			}
			bs, be, ids := newPortableQwen35Batch(t, lanes)
			var other *portableQwen35BatchBackend
			switch name {
			case "wrong_marker":
				be.batchPath = "wrong"
			case "mixed_backend":
				other = &portableQwen35BatchBackend{sequencePrefillBackend: newSequencePrefillBackend(bs.M), batchPath: compute.Qwen35IndependentDecodeBatchPath}
				s, err := bs.M.NewBackendSessionChecked(other)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(s.Close)
				bs.Seqs[1] = s
			case "mixed_model":
				bs.M = NewSynthetic(qwen35HybridTestCfg())
			case "missing_state":
				saved := bs.Seqs[1].qwen35HAL
				bs.Seqs[1].qwen35HAL = nil
				defer func() { bs.Seqs[1].qwen35HAL = saved }()
			}
			if _, accepted := stepBatchQwen35HybridQ4KIndependent(bs, ids); accepted || be.batchCalls != 0 {
				t.Fatalf("accepted=%t calls=%d", accepted, be.batchCalls)
			}
			if other != nil && other.batchCalls != 0 {
				t.Fatalf("second backend calls=%d", other.batchCalls)
			}
			for i, s := range bs.Seqs {
				if s.halKV.Len() != 0 || len(s.halLineage.ids) != 0 || s.BackendSessionClosed() {
					t.Fatalf("lane %d mutated on decline", i)
				}
			}
		})
	}

	for _, name := range []string{"missing_capability", "wrong_marker", "unmutated_error", "native_override"} {
		t.Run(name, func(t *testing.T) {
			bs, be, ids := newPortableQwen35Batch(t, 2)
			wantReason, wantCalls, nativeCalls := "backend_declined", 1, 0
			switch name {
			case "missing_capability":
				for i := range bs.Seqs {
					s, err := bs.M.NewBackendSessionChecked(be.sequencePrefillBackend)
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(s.Close)
					bs.Seqs[i] = s
				}
				wantReason, wantCalls = "unsupported_config", 0
			case "wrong_marker":
				be.batchPath = "wrong"
				wantReason, wantCalls = "unsupported_config", 0
			case "unmutated_error":
				be.batchErr = &compute.Qwen35IndependentDecodeBatchError{Stage: "preflight", Cause: errors.New("synthetic refusal")}
			case "native_override":
				qwen35HybridQ4KBatchStep = func(*BatchSession, []int) ([][]float32, bool) { nativeCalls++; return nil, false }
				defer func() { qwen35HybridQ4KBatchStep = nil }()
				wantCalls = 0
			}
			fallbacks := 0
			bs.ObserveSerialStep = func(reason string, _ time.Duration) {
				if reason != wantReason {
					t.Fatalf("reason=%q want=%q", reason, wantReason)
				}
				fallbacks++
			}
			if got := bs.StepBatch(ids); len(got) != len(ids) || be.batchCalls != wantCalls || fallbacks != len(ids) || bs.LastStepSharedPanels() != 0 || bs.LastStepMACs() != 0 {
				t.Fatalf("rows=%d calls=%d fallbacks=%d panels=%d macs=%d", len(got), be.batchCalls, fallbacks, bs.LastStepSharedPanels(), bs.LastStepMACs())
			}
			if name == "native_override" && nativeCalls != 1 {
				t.Fatalf("native override calls=%d", nativeCalls)
			}
			for i, s := range bs.Seqs {
				if s.BackendSessionClosed() || s.halFailure != nil || s.halKV.Len() != 1 {
					t.Fatalf("lane %d did not preserve scalar continuation", i)
				}
			}
		})
	}

	t.Run("mutated_error", func(t *testing.T) {
		bs, be, ids := newPortableQwen35Batch(t, 2)
		cause := errors.New("synthetic mutated batch failure")
		be.batchErr = &compute.Qwen35IndependentDecodeBatchError{Stage: "selected", Mutated: true, Cause: cause}
		be.freeHook = func() { panic("cleanup must not replace selected failure") }
		fallbacks := 0
		bs.ObserveSerialStep = func(string, time.Duration) { fallbacks++ }
		var recovered any
		func() { defer func() { recovered = recover() }(); bs.StepBatch(ids) }()
		if recovered != be.batchErr || !errors.Is(be.batchErr, cause) || be.batchCalls != 1 || be.gdnCalls != 0 || fallbacks != 0 {
			t.Fatalf("panic=%v calls=%d scalar=%d fallbacks=%d", recovered, be.batchCalls, be.gdnCalls, fallbacks)
		}
		for i, s := range bs.Seqs {
			if !s.BackendSessionClosed() || s.halFailure != be.batchErr {
				t.Fatalf("lane %d failed to retain terminal lifecycle", i)
			}
			// A cleanup panic can interrupt clearing the container pointer, but
			// every recurrent owner must already have been detached.
			if s.qwen35HAL != nil {
				for _, state := range s.qwen35HAL.layers {
					if state.owner != nil || state.conv.Buf() != nil || state.recurrent.Buf() != nil {
						t.Fatalf("lane %d retained a recurrent owner after cleanup", i)
					}
				}
			}
			var retry any
			func() { defer func() { retry = recover() }(); s.Step(41 + i) }()
			if retry != be.batchErr || be.batchCalls != 1 || be.gdnCalls != 0 {
				t.Fatalf("lane %d retry=%v calls=%d scalar=%d", i, retry, be.batchCalls, be.gdnCalls)
			}
		}
	})

	// fak-test:runtime fast est=100ms lane=default
	// Both failures happen after the actual provider has mutated every lane.
	// The estimate is inherited from this existing small synthetic host family.
	for _, output := range []string{"malformed_selected_receipt", "short_selected_logits"} {
		for _, cleanupPanic := range []bool{false, true} {
			// Short logits frees its output before validation; inject retained-state
			// cleanup failure only on the receipt branch, not that earlier Free.
			if output == "short_selected_logits" && cleanupPanic {
				continue
			}
			t.Run(fmt.Sprintf("%s/cleanup_panic_%t", output, cleanupPanic), func(t *testing.T) {
				bs, be, ids := newPortableQwen35Batch(t, 2)
				be.badReceipt = output == "malformed_selected_receipt"
				be.shortLogits = output == "short_selected_logits"
				cleanupCalls := 0
				if cleanupPanic {
					be.freeHook = func() { cleanupCalls++; panic("cleanup must not replace selected failure") }
				}
				fallbacks := 0
				bs.ObserveSerialStep = func(string, time.Duration) { fallbacks++ }
				var recovered any
				func() { defer func() { recovered = recover() }(); bs.StepBatch(ids) }()
				want := "model: malformed Qwen Vulkan independent decode logits"
				if be.badReceipt {
					want = fmt.Sprintf("model: malformed Qwen Vulkan independent decode result: lanes=2 panels=0 fallback=0 shape=[2 %d]", bs.M.Cfg.VocabSize)
				}
				if recovered != want {
					t.Errorf("initial panic = %#v, want unchanged string %#v", recovered, want)
				}
				if len(be.mutatedKV) != 2 || be.mutatedKV[0] != 1 || be.mutatedKV[1] != 1 {
					t.Fatalf("provider did not mutate both KV owners: %v", be.mutatedKV)
				}
				if be.batchCalls != 1 || be.gdnCalls != 0 || fallbacks != 0 || bs.LastStepSharedPanels() != 0 || bs.LastStepMACs() != 0 {
					t.Errorf("calls=%d scalar=%d fallbacks=%d panels=%d macs=%d", be.batchCalls, be.gdnCalls, fallbacks, bs.LastStepSharedPanels(), bs.LastStepMACs())
				}
				var failure error
				for i, s := range bs.Seqs {
					if !s.BackendSessionClosed() || s.halFailure == nil || s.halFailure.Error() != want {
						t.Errorf("lane %d survived mutated failure: closed=%t failure=%v", i, s.BackendSessionClosed(), s.halFailure)
					}
					if i == 0 {
						failure = s.halFailure
					} else if s.halFailure != failure {
						t.Error("participants did not retain one failure")
					}
				}
				if cleanupPanic && cleanupCalls == 0 {
					t.Error("retirement never attempted injected cleanup")
				}
				if cleanupPanic && (bs.Seqs[0].qwen35HAL == nil || bs.Seqs[0].halKV == nil) {
					t.Error("cleanup-panic fixture did not retain both structural pointers for batch retry")
				}
				// Restore a healthy provider: retry refusal must come from session state,
				// not another provider error. Keep output ownership cleanup fixture-local.
				be.freeHook = nil
				be.badReceipt, be.shortLogits = false, false
				if be.freeCalls[be.logits.Buf()] == 0 {
					be.Free(be.logits)
				}
				var retry any
				func() { defer func() { retry = recover() }(); bs.StepBatch(ids) }()
				if failure == nil || retry != failure || be.batchCalls != 1 || be.gdnCalls != 0 || fallbacks != 0 {
					t.Errorf("batch retry re-entered selected work: panic=%v failure=%v calls=%d scalar=%d fallbacks=%d", retry, failure, be.batchCalls, be.gdnCalls, fallbacks)
				}
				for i, s := range bs.Seqs {
					func() { defer func() { retry = recover() }(); s.Step(41 + i) }()
					if failure == nil || retry != failure || be.batchCalls != 1 || be.gdnCalls != 0 {
						t.Errorf("lane %d scalar retry resumed: panic=%v failure=%v calls=%d scalar=%d", i, retry, failure, be.batchCalls, be.gdnCalls)
					}
				}
			})
		}
	}
}
