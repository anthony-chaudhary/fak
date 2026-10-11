package model

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model/norm"
)

// v41ExactNormTestBackend is a cpu-ref recorder for the dedicated exact RMSNorm
// operation. It qualifies only the operation's own domain and computes
// norm.SerialRMSNorm; it never implements the per-seam qualification methods.
type v41ExactNormTestBackend struct {
	*v41DenseTestBackend
	supported bool
	calls     int
	fail      error
	// Wrapping recorders set this to the outer operation owner, as real
	// backends do. CPU storage does not change the output's declared owner.
	outputOwner compute.Backend
}

func newV41ExactNormTestBackend() *v41ExactNormTestBackend {
	b := &v41ExactNormTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), supported: true}
	b.outputOwner = b
	return b
}

func (b *v41ExactNormTestBackend) SupportsV41ExactRMSNorm(width int, eps float32) bool {
	return b.supported && compute.V41ExactRMSNormDomain(width, eps)
}

func (b *v41ExactNormTestBackend) V41ExactRMSNorm(x, weight compute.Tensor, eps float32) (compute.Tensor, error) {
	b.calls++
	if b.fail != nil {
		return compute.Tensor{}, b.fail
	}
	out := norm.SerialRMSNorm(b.Backend.Read(x), b.Backend.Read(weight), eps)
	return compute.NewF32(b.outputOwner, append([]int(nil), x.Shape...), out), nil
}

func v41ExactNormTestRow(width int, seed float32) []float32 {
	row := make([]float32, width)
	for i := range row {
		row[i] = v41RoundBF16(float32((i*37)%101-50)*seed + 1/float32(i+3))
	}
	return row
}

// Ref #13668. A backend with only the dedicated exact RMSNorm binds both the
// compressor and index-key norm seams, and each selected row is bit-identical
// to the host tail it replaces: the compressor's BF16 -> RMSNorm -> BF16 tail
// and the unrounded index-key RMSNorm. A width outside the exact domain or an
// unqualified backend leaves the host arm.
// fak-test:runtime fast est=50ms lane=default
func TestV41NormSeamsBindExactRMSNorm(t *testing.T) {
	const eps = float32(1e-6)
	for _, width := range []int{128, 512} {
		gain := v41ExactNormTestRow(width, 0.01)
		pooled := v41ExactNormTestRow(width, 3.5)

		m := v41CompressorNormTinyModel(t, gain, eps)
		b := newV41ExactNormTestBackend()
		s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
		t.Cleanup(s.Close)
		compress := s.v41CompressorNormFunc()
		if compress == nil {
			t.Fatalf("width %d: exact RMSNorm did not bind the compressor norm", width)
		}
		got, err := compress(0, pooled, gain, eps)
		if want := v41CompressorNormRefTail(pooled, gain, eps, ""); err != nil || !v41CompressorNormEqualBits(got, want) {
			t.Fatalf("width %d: compressor norm %v differs from the host tail", width, err)
		}

		km := v41IndexKeyNormTinyModel(t, gain, eps)
		kb := newV41ExactNormTestBackend()
		ks := &Session{M: km, Backend: kb, halW: map[string]compute.Tensor{}}
		t.Cleanup(ks.Close)
		key := ks.v41IndexKeyNormFunc()
		if key == nil {
			t.Fatalf("width %d: exact RMSNorm did not bind the index-key norm", width)
		}
		got, err = key(0, pooled, gain, eps)
		if want := v41IndexKeyNormReference(pooled, gain, eps); err != nil || !v41CompressorNormEqualBits(got, want) {
			t.Fatalf("width %d: index-key norm %v differs from the host RMSNorm", width, err)
		}
		if b.calls != 1 || kb.calls != 1 {
			t.Fatalf("exact operation calls = %d/%d, want one each", b.calls, kb.calls)
		}
	}

	gain := v41ExactNormTestRow(96, 0.01)
	odd := &Session{M: v41CompressorNormTinyModel(t, gain, eps), Backend: newV41ExactNormTestBackend(), halW: map[string]compute.Tensor{}}
	t.Cleanup(odd.Close)
	if odd.v41CompressorNormFunc() != nil {
		t.Fatal("width 96 is outside the exact domain but bound the compressor norm")
	}
	declined := newV41ExactNormTestBackend()
	declined.supported = false
	off := &Session{M: v41IndexKeyNormTinyModel(t, v41ExactNormTestRow(128, 0.01), eps), Backend: declined, halW: map[string]compute.Tensor{}}
	t.Cleanup(off.Close)
	if off.v41IndexKeyNormFunc() != nil {
		t.Fatal("an unqualified exact operation bound the index-key norm")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41ExactRMSNormFailureClosesWithoutReplay(t *testing.T) {
	const eps = float32(1e-6)
	gain := v41ExactNormTestRow(128, 0.01)
	b := newV41ExactNormTestBackend()
	cause := &compute.BackendError{Backend: "vulkan", Site: "V41ExactRMSNorm", Err: compute.ErrVulkanExecutionFailed}
	b.fail = cause
	s := &Session{M: v41IndexKeyNormTinyModel(t, gain, eps), Backend: b, halW: map[string]compute.Tensor{}}
	t.Cleanup(s.Close)
	key := s.v41IndexKeyNormFunc()
	if key == nil {
		t.Fatal("exact RMSNorm did not bind")
	}
	got, err := key(0, v41ExactNormTestRow(128, 2), gain, eps)
	var op *V41ProjectionOperationError
	var closed *BackendForwardOperationError
	if got != nil || !errors.As(err, &op) || !errors.As(err, &closed) || closed.Path != "v41-index-key-norm" ||
		closed.Stage != "rmsnorm" || !errors.Is(err, compute.ErrVulkanExecutionFailed) || !s.BackendSessionClosed() {
		t.Fatalf("selected failure = %v (closed=%v), want typed close without host replay", err, s.BackendSessionClosed())
	}
	if b.calls != 1 {
		t.Fatalf("exact operation calls = %d, want 1", b.calls)
	}
}

// fak-test:runtime medium est=2s lane=default
func TestV41DeviceOnlyBindsNormsThroughExactRMSNorm(t *testing.T) {
	m := v41IndexerTestFixture(t)
	if !v41UsesIndexer(m.Cfg) || !v41UsesCompressedKV(m.Cfg) {
		t.Fatal("fixture must run the compressed indexer path")
	}
	unbound := func(b compute.Backend) []string {
		return v41HostOpNames(v41DenseTestSession(t, m, b).v41DeviceOnlyUnboundOps())
	}
	without := unbound(newV41DenseTestBackend())
	with := unbound(newV41ExactNormTestBackend())
	for _, op := range []string{"compressor-norm", "index-key-norm"} {
		if !slices.Contains(without, op) {
			t.Fatalf("plain backend leaves %s bound: %v", op, without)
		}
		if slices.Contains(with, op) {
			t.Fatalf("exact RMSNorm backend still leaves %s unbound: %v", op, with)
		}
	}
	if eps := float32(m.Cfg.RMSNormEps); !compute.V41ExactRMSNormDomain(v41CompressorWidth(m.Cfg), eps) ||
		!compute.V41ExactRMSNormDomain(m.Cfg.IndexHeadDim, eps) || math.IsNaN(float64(eps)) {
		t.Fatal("fixture widths are outside the exact domain")
	}
}
