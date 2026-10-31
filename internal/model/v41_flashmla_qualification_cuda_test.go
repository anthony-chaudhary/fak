//go:build fak_v41_flashmla_component_witness && cuda && cgo && cudaarchspecific && fak_cuda_flashmla_decode

package model

import (
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestV41FlashQualificationOwnedPhysical is a required-device component witness
// for the actual Session binding -> composer -> packed codec -> checked CUDA
// owner path. Its synthetic projected rows are not a checkpoint, full-model
// forward, public backend-admission, device-cache or performance witness.
// The external executor must pin source, binary, native archive and device.
// fak-test:runtime integration est=5s lane=optin
func TestV41FlashQualificationOwnedPhysical(t *testing.T) {
	if os.Getenv("FAK_FLASHMLA_DECODE_ACCEPT") != "1" {
		t.Fatal("required FlashMLA component witness needs FAK_FLASHMLA_DECODE_ACCEPT=1")
	}
	backend, ok := compute.Lookup("cuda")
	if !ok || backend == nil || backend.Name() != "cuda" || compute.RequireReference(backend) {
		t.Fatal("required actual CUDA backend unavailable; no reference fallback")
	}
	if owner, ok := backend.(compute.ForwardExecutionOwnerBackend); !ok || owner.ForwardExecutionOwner() == nil {
		t.Fatal("required actual CUDA execution owner unavailable")
	}
	// Deliberately a component owner, not NewBackendSession's full-model
	// admission. No model weights, logits or checkpoint inference are claimed.
	s := &Session{M: &Model{}, Backend: backend}
	s.v41Forward = &v41ForwardState{callbackOwner: s}
	scope := s.beginBackendExecution()
	if scope == nil {
		t.Fatal("CUDA did not establish an execution scope")
	}
	complete := false
	defer scope.finishCompleted(&complete)
	// LIFO: all session cleanup occurs while the actual recording lease lives.
	defer s.closeBackendOwned()
	binding := s.v41FlashMLAQualificationBinding()
	s.v41Forward.sharedAttentionBinding = binding

	// Match the existing TestFlashMLADecodeOwnedPhysical's explicit absolute
	// tolerance. The 0.25 duplicate-control separation is far above this bound.
	const tolerance = 0.02
	for _, allInvalid := range []bool{false, true} {
		request := v41FlashQualificationFixture()
		clear(request.q)
		if allInvalid {
			for i := range request.idx {
				request.idx[i] = -1
			}
		}
		rows := request.rowKV.window
		// Every compressed ordinal is causal at pos=1, ratio=1. No window rows
		// are added: IDs {1,-1,1,0} retain two contributions from the second row.
		compose := func(attend v41SharedAttentionFunc) ([]float32, error) {
			return v41CombinedAttention(0, 1, 1, 64, 512, request.q, request.sink,
				nil, rows, request.idx, request.plain.Softmax, attend)
		}
		want, err := compose(nil)
		if err != nil {
			t.Fatal("reference composer:", err)
		}
		got, err := compose(binding.attentionFunc())
		if err != nil {
			t.Fatal("selected actual model/FlashMLA binding:", err)
		}
		if len(got) != 32768 || len(want) != len(got) {
			t.Fatal("component output shape")
		}
		analytic := float32(1.25) // (2+2+1)/(three selected rows + sink).
		if allInvalid {
			analytic = 0
		}
		for i, value := range got {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || want[i] != analytic ||
				math.Abs(float64(value-want[i])) > tolerance || (allInvalid && value != 0) {
				t.Fatalf("all-invalid=%t output[%d]=%g reference=%g analytic=%g", allInvalid, i, value, want[i], analytic)
			}
		}
		// The ordinary model callback intentionally has no LSE result. Exercise
		// the same real exported owner independently to retain donor LSE evidence;
		// never compare it with a sink-included or all-invalid -Inf convention.
		input, err := v41PackFlashMLAQualification(request)
		if err != nil {
			t.Fatal(err)
		}
		output, err := compute.QualifyFlashMLADecode(backend, input)
		if err != nil {
			// Preserve the selected native primary through deferred teardown,
			// including the owner's uncertain-drain/quarantine refusal. No retry.
			s.halFailure = &BackendForwardOperationError{Backend: backend.Name(), Forward: ForwardPathKind("deepseek41"),
				Path: "v41-flashmla-qualification", Layer: 0, Stage: "direct-owner-witness", Cause: err}
			t.Fatal("selected direct owner:", s.halFailure)
		}
		if len(output.ValuesBF16) != 32768 || len(output.LSE) != 64 {
			t.Fatal("donor output shape")
		}
		for i, bits := range output.ValuesBF16 {
			value := math.Float32frombits(uint32(bits) << 16)
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value-analytic)) > tolerance || (allInvalid && value != 0) {
				t.Fatalf("donor output[%d]=%g", i, value)
			}
		}
		for i, lse := range output.LSE {
			if allInvalid {
				if !math.IsInf(float64(lse), 1) {
					t.Fatalf("all-invalid LSE[%d]=%g", i, lse)
				}
			} else if math.IsNaN(float64(lse)) || math.IsInf(float64(lse), 0) || math.Abs(float64(lse)-math.Log(3)) > tolerance {
				t.Fatalf("sink-excluded LSE[%d]=%g want log(3)", i, lse)
			}
		}
		if s.halClosed || s.halFailure != nil || s.v41Forward.sharedAttentionBinding != binding || !scope.live {
			t.Fatal("successful owner lost its binding/scope")
		}
	}
	t.Logf("synthetic projected-row component only: actual backend=%s tier=%s; duplicate/sink and all-invalid device cases completed", backend.Name(), backend.Tier())
	complete = true
}
