package model

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// refusingQwen35CUDA wraps the real cpu-ref Backend surface but identifies as
// CUDA and records the generic ops that would constitute the forbidden fallback.
// The production Model is real; this wrapper exists only to prove the checked
// constructor refuses before any generic QKV/attention operator can execute.
type refusingQwen35CUDA struct {
	compute.Backend
	matmuls   int
	attention int
}

func (b *refusingQwen35CUDA) Name() string                    { return "cuda" }
func (b *refusingQwen35CUDA) Tier() string                    { return "sm_80" }
func (b *refusingQwen35CUDA) Class() compute.CorrectnessClass { return compute.Approx }
func (b *refusingQwen35CUDA) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	return b.Backend.MatMul(w, x)
}
func (b *refusingQwen35CUDA) Attention(q compute.Tensor, kv compute.KVStore, layer int, causal bool, grp int, scale float32) compute.Tensor {
	b.attention++
	return b.Backend.Attention(q, kv, layer, causal, grp, scale)
}

// claimedQwen35CUDA proves that a capability marker alone cannot bypass the
// refusal while the model HAL has no Qwen35GDNDecode dispatch branch.
type claimedQwen35CUDA struct{ *refusingQwen35CUDA }

func (*claimedQwen35CUDA) Qwen35GDNPath() string { return Qwen35GDNCUDAPath }
func (*claimedQwen35CUDA) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState compute.Tensor, err error) {
	panic("must remain unreachable until the model HAL dispatch is wired")
}

func TestQwen35CUDAGDNFixtureFailsClosedBeforeFallback(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	prompt := []int{3, 7, 11, 5, 17, 19, 23}

	// Deterministic CPU/reference fixture: the real Qwen35 GDN/SSM object and
	// persistent-state path must reproduce exactly before it can serve as the
	// future CUDA parity oracle. State is exercised by a split prefill+decode.
	want := m.NewSession().Prefill(prompt)
	reference := m.NewSession()
	reference.Prefill(prompt[:4])
	var got []float32
	for _, id := range prompt[4:] {
		got = reference.Step(id)
	}
	if len(got) != len(want) {
		t.Fatalf("CPU/reference fixture logits=%d, want %d", len(got), len(want))
	}
	var dot, nw, ng float64
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("CPU/reference fixture is not deterministic at logit %d: got=%v want=%v", i, got[i], want[i])
		}
		dot += float64(got[i]) * float64(want[i])
		nw += float64(want[i]) * float64(want[i])
		ng += float64(got[i]) * float64(got[i])
	}
	if nw == 0 || ng == 0 {
		t.Fatalf("CPU/reference fixture has zero-norm logits: want=%g got=%g", nw, ng)
	}
	cosine := dot / (math.Sqrt(nw) * math.Sqrt(ng))
	if cosine < Qwen35GDNParityCosineMin {
		t.Fatalf("CPU/reference fixture cosine %.9f < explicit CUDA acceptance floor %.3f", cosine, Qwen35GDNParityCosineMin)
	}

	be := &refusingQwen35CUDA{Backend: compute.Default()}
	s, err := m.NewBackendSessionChecked(be)
	if s != nil {
		t.Fatalf("missing CUDA GDN path returned a session: %#v", s)
	}
	var unsupported *UnsupportedBackendForwardError
	if !errors.As(err, &unsupported) {
		t.Fatalf("checked constructor error=%T (%v), want *UnsupportedBackendForwardError", err, err)
	}
	if unsupported.Backend != "cuda" || unsupported.Forward != ForwardQwen35GDN ||
		unsupported.IntendedPath != Qwen35GDNCUDAPath || unsupported.ParityCosineMin != Qwen35GDNParityCosineMin {
		t.Fatalf("wrong refusal identity: %#v", unsupported)
	}
	for _, wantText := range []string{"qwen35-gdn", Qwen35GDNCUDAPath, "generic QKV/CPU fallback", "0.999", "#4714"} {
		if !strings.Contains(err.Error(), wantText) {
			t.Errorf("refusal missing %q:\n%s", wantText, err)
		}
	}
	if be.matmuls != 0 || be.attention != 0 {
		t.Fatalf("refused construction executed fallback ops: matmul=%d attention=%d", be.matmuls, be.attention)
	}

	// Even a backend test double that advertises the intended operation and path
	// stays refused until the production model dispatcher actually invokes it.
	claimed := &claimedQwen35CUDA{refusingQwen35CUDA: &refusingQwen35CUDA{Backend: compute.Default()}}
	if _, err := m.NewBackendSessionChecked(claimed); err == nil || !strings.Contains(err.Error(), "does not yet dispatch") {
		t.Fatalf("marker-only GDN backend must remain refused, got %v", err)
	}
	if claimed.matmuls != 0 || claimed.attention != 0 {
		t.Fatalf("marker-only backend executed fallback ops: matmul=%d attention=%d", claimed.matmuls, claimed.attention)
	}
}

func TestQwen35CUDAGDNUncheckedConstructorPanicsWithTypedRefusal(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	be := &refusingQwen35CUDA{Backend: compute.Default()}
	defer func() {
		r := recover()
		var unsupported *UnsupportedBackendForwardError
		if !errors.As(asError(r), &unsupported) {
			t.Fatalf("NewBackendSession panic=%T (%v), want *UnsupportedBackendForwardError", r, r)
		}
		if be.matmuls != 0 || be.attention != 0 {
			t.Fatalf("panic path executed fallback ops: matmul=%d attention=%d", be.matmuls, be.attention)
		}
	}()
	_ = m.NewBackendSession(be)
}

func asError(v any) error {
	if err, ok := v.(error); ok {
		return err
	}
	return nil
}
