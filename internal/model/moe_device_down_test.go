package model

import (
	"encoding/binary"
	"math"
	"math/rand"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expertDownQ6KTensorBounded builds a Q6_K tensor like expertHALQ6KTensor but with a
// PINNED, moderate f16 block scale d = 1.0, mirroring how expertHALQ5KTensor pins d for
// the same reason (see its comment "keeps three-projection fixture finite").
//
// The shared expertHALQ6KTensor fills EVERY byte with a plain PRNG, including the f16 d
// field that lives in the LAST 2 bytes of each 210-byte super-block (q6kBlockBytes-2,
// read by q6kDequantSuperBlock at quant_kquant.go:238). A random d bit pattern decodes to
// an arbitrary half-precision value — often ~1e3..1e5 — which multiplies through the
// three-projection MLP into a 1e7..1e10 activation and a tolerance larger than any
// plausible error, making a parity assertion nearly vacuous. Pinning d keeps the fixture
// in an order-1..1e3 range where an absolute tolerance is a real bound.
func expertDownQ6KTensorBounded(out, in int, seed int64) *kQuantTensor {
	nblk := in / qkK
	raw := make([]byte, out*nblk*q6kBlockBytes)
	rng := rand.New(rand.NewSource(seed))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := (o*nblk+b)*q6kBlockBytes + q6kBlockBytes - 2
			binary.LittleEndian.PutUint16(raw[base:], f16One) // d = 1.0, keeps the fixture finite
		}
	}
	return &kQuantTensor{out: out, in: in, nblk: nblk, kind: kindQ6K, raw: raw}
}

// expertDownQ4KTensorBounded builds a Q4_K weight like buildRawQ4K but with PINNED,
// moderate f16 d and min scales, mirroring the Q5_K/Q6_K down helpers' pinned scale.
//
// buildRawQ4K delegates to randQ4KBlock, which only constrains the f16 EXPONENT into
// [1,30]; exponent 30 decodes to ~2^15, so a random sign/frac yields d up to ~6e4 and a
// gate/up activation whose SwiGLU output reaches 1e14..1e18 before the down projection is
// ever applied. That is the dominant magnitude in this fixture — NOT the down scale — and
// it makes the 1e-5 parity tolerance vacuous. Pinning d = min = 2^-11 keeps the whole
// three-projection MLP in an order-1..1e3 range where an absolute tolerance is a real bound.
func expertDownQ4KTensorBounded(t *testing.T, out, in int, seed int64) []byte {
	t.Helper()
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	rng := rand.New(rand.NewSource(seed))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	for b := 0; b < out*nblk; b++ {
		blk := raw[b*q4kBlockBytes : (b+1)*q4kBlockBytes]
		binary.LittleEndian.PutUint16(blk[0:], 0x1000) // d = 2^-11
		binary.LittleEndian.PutUint16(blk[2:], 0x1000) // min = 2^-11
	}
	return raw
}

// q4kExpertDownModel builds a single-expert MoE whose gate/up are resident Q4_K and
// whose down projection is the requested k-quant kind — the live q4_k_m mixture. It
// returns the model plus the six operation counters a device-route witness reads.
//
// downKind == kindQ6K (the live checkpoint) exercises the seam this ticket closes;
// downKind == kindIQ3XXS is the negative-test encoding: a resident k-quant with no
// device kernel, which MUST keep the host path byte-for-byte.
//
// Every projection uses a PINNED moderate f16 scale (gate/up via expertDownQ4KTensorBounded,
// down via expertDownQ6KTensorBounded / expertHALQ5KTensor). An unprefixed random f16 scale
// explodes the fixture into 1e7..1e18 activations where the parity tolerance stops bounding
// anything — see the helper comments.
func q4kExpertDownModel(t *testing.T, H int, downKind kQuantKind) *Model {
	t.Helper()
	m := NewSyntheticMoE(expertHALTestConfig(H))
	m.q4kw = map[string]*q4kTensor{}
	m.kqw = map[string]*kQuantTensor{}
	for i, proj := range []string{"gate_proj.weight", "up_proj.weight"} {
		name := expertName(0, 0, proj)
		m.q4kw[name] = &q4kTensor{out: H, in: H, raw: expertDownQ4KTensorBounded(t, H, H, int64(401+i)), nblk: 1}
	}
	down := expertName(0, 0, "down_proj.weight")
	switch downKind {
	case kindQ6K:
		m.kqw[down] = expertDownQ6KTensorBounded(H, H, 907)
	case kindQ5K:
		m.kqw[down] = expertHALQ5KTensor(H, H, 907)
	case kindIQ3XXS:
		// A registered k-quant with no device HAL support: SupportsHALKQuant is false, so
		// resolveExpertWeight declines it and the host path must run unchanged.
		m.kqw[down] = &kQuantTensor{out: H, in: H, nblk: H / qkK, kind: kindIQ3XXS, raw: make([]byte, H*(H/qkK)*kindIQ3XXS.blockBytes())}
	default:
		t.Fatalf("q4kExpertDownModel: unsupported kind %v", downKind)
	}
	return m
}

func expertDownTestInput(H int) []float32 {
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%23)-11) / 37
	}
	return x
}

// TestExpertDownDeviceKernelStagesQ5KQ6KOnce is the acceptance witness for fak#13129:
// a routed expert whose gate/up are resident Q4_K and whose down projection is a
// resident Q5_K/Q6_K k-quant completes ENTIRELY on the device backend (three MatMuls
// plus one SwiGLU), staging each k-quant weight exactly once, with the down matmul
// taking the resident Q6_K/Q5_K weight rather than falling back to host.
func TestExpertDownDeviceKernelStagesQ5KQ6KOnce(t *testing.T) {
	const H = 256
	for _, kind := range []kQuantKind{kindQ5K, kindQ6K} {
		name := "Q5_K"
		wantUp := compute.Q5_K
		if kind == kindQ6K {
			name = "Q6_K"
			wantUp = compute.Q6_K
		}
		t.Run(name, func(t *testing.T) {
			m := q4kExpertDownModel(t, H, kind)
			be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
			s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
			x := expertDownTestInput(H)

			got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
			if len(got) != H {
				t.Fatalf("expert output len=%d want %d", len(got), H)
			}
			// The whole expert MLP — gate, up, SwiGLU AND the down projection — runs on
			// the backend. Before this ticket the down projection returned to the host,
			// so matmuls was 2 and the result reached the caller through a host fallback.
			if be.matmuls != 3 || be.swiglu != 1 {
				t.Fatalf("device expert ops matmul=%d swiglu=%d, want 3/1 (down projection must stay on the backend)", be.matmuls, be.swiglu)
			}
			if be.uploads[compute.Q4_K] != 2 {
				t.Fatalf("Q4_K uploads=%d, want one resident upload for gate and one for up", be.uploads[compute.Q4_K])
			}
			if be.uploads[wantUp] != 1 {
				t.Fatalf("%s down uploads=%d, want exactly one resident staging", name, be.uploads[wantUp])
			}
			if be.uploads[compute.F16] != 0 {
				t.Fatalf("expanded F16 uploads=%d, want 0 (raw k-quant residency)", be.uploads[compute.F16])
			}
			for i, v := range got {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("output[%d]=%v", i, v)
				}
			}
			if s.Q4KExpertStats().RoutedExpertsDeviceHAL != 1 {
				t.Fatalf("device-HAL experts=%d want 1", s.Q4KExpertStats().RoutedExpertsDeviceHAL)
			}

			// A second token reuses every resident weight: only the activation upload repeats.
			expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
			if be.uploads[compute.Q4_K] != 2 || be.uploads[wantUp] != 1 {
				t.Fatalf("resident weights reuploaded: q4=%d kq=%d", be.uploads[compute.Q4_K], be.uploads[wantUp])
			}
		})
	}
}

// TestExpertDownDeviceKernelHostParity is the resident-warm parity witness: the
// device-executed expert (all three projections + SwiGLU on compute.Backend) must
// match the host k-quant path within the existing tolerance, on the same fixture.
//
// The host arm runs the SAME weights through the SAME backend primitives but with the
// device routed-expert capability off, so the comparison isolates the seam under test
// from the quantized-arithmetic agreement both arms share.
func TestExpertDownDeviceKernelHostParity(t *testing.T) {
	const H = 256
	for _, kind := range []kQuantKind{kindQ5K, kindQ6K} {
		name := "Q5_K"
		if kind == kindQ6K {
			name = "Q6_K"
		}
		t.Run(name, func(t *testing.T) {
			m := q4kExpertDownModel(t, H, kind)
			x := expertDownTestInput(H)

			// Host arm: capability declined, so expertSwiGLU takes the host k-quant path.
			hostBe := &routedIncapableBackend{&expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}}
			hostSession := &Session{M: m, Backend: hostBe, Q4K: true, halW: map[string]compute.Tensor{}}
			want := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: hostSession})

			// Device arm: the same model, routed-capable backend.
			devBe := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
			devSession := &Session{M: m, Backend: devBe, Q4K: true, halW: map[string]compute.Tensor{}}
			got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: devSession})

			if devBe.matmuls != 3 || devBe.swiglu != 1 {
				t.Fatalf("device route matmul=%d swiglu=%d, want 3/1", devBe.matmuls, devBe.swiglu)
			}
			var maxAbs, scale float64
			for i := range want {
				d := math.Abs(float64(got[i] - want[i]))
				if d > maxAbs {
					maxAbs = d
				}
				if a := math.Abs(float64(want[i])); a > scale {
					scale = a
				}
			}
			tol := 1e-5 * math.Max(1, scale)
			if maxAbs > tol {
				t.Fatalf("%s down-projection device/host parity maxAbs=%g > tol=%g", name, maxAbs, tol)
			}
			t.Logf("%s routed expert full-MLP parity maxAbs=%g tol=%g", name, maxAbs, tol)
		})
	}
}

// TestExpertDownDeviceKernelDeclinesWithoutKernel is the negative test the issue
// requires: an encoding with NO device down kernel keeps the host path byte-for-byte.
//
// The fixture carries a resident IQ3_XXS down projection. resolveExpertWeight declines
// it (SupportsHALKQuant is false), so expertSwiGLUHAL returns ok=false before any
// backend op — the full device expert MLP seam must DECLINE, not partially execute a
// down projection it has no kernel for. The witness is the pair of admission facts plus
// an exact host-result comparison, not a count of backend ops: a later host-side fast
// path (the Metal fused q4kFusedMLP) legitimately issues its own backend ops, and
// asserting those to zero would pin that unrelated route instead of this seam.
func TestExpertDownDeviceKernelDeclinesWithoutKernel(t *testing.T) {
	const H = 256
	m := q4kExpertDownModel(t, H, kindIQ3XXS)
	x := expertDownTestInput(H)

	// A capable backend must NOT be offered this expert: the down projection has no
	// device kernel, so the whole three-weight seam declines together.
	be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
	if _, ok := s.expertSwiGLUHAL(
		expertName(0, 0, "gate_proj.weight"),
		expertName(0, 0, "up_proj.weight"),
		expertName(0, 0, "down_proj.weight"), x); ok {
		t.Fatal("device expert seam admitted an encoding with no down-projection device kernel")
	}
	if be.matmuls != 0 || be.swiglu != 0 {
		t.Fatalf("declined seam still issued backend ops: matmul=%d swiglu=%d", be.matmuls, be.swiglu)
	}

	// The expert as a whole then runs somewhere else, and the counters say so: an
	// off-HAL booking, never a device-HAL one.
	got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	if len(got) != H {
		t.Fatalf("expert output len=%d want %d", len(got), H)
	}
	stats := s.Q4KExpertStats()
	if stats.RoutedExpertsDeviceHAL != 0 {
		t.Fatalf("device-HAL experts=%d want 0 (encoding has no device kernel)", stats.RoutedExpertsDeviceHAL)
	}
	if stats.RoutedExpertsOffHAL != 1 {
		t.Fatalf("off-HAL experts=%d want 1", stats.RoutedExpertsOffHAL)
	}
	// The declined k-quant was never staged on the device under a k-quant key — a
	// partial staging would be the half-executed expert this negative test forbids.
	for key := range s.halW {
		if strings.HasPrefix(key, "kquant-raw:") {
			t.Fatalf("declined encoding staged a device k-quant weight: %s", key)
		}
	}

	// Byte-for-byte host preservation, against an INDEPENDENT oracle. A second
	// expertSwiGLU on a non-capable backend would share this arm's gate/up/down
	// arithmetic by construction, so it could only show the decline is
	// observable-transparent. Instead rebuild the expert from the host primitives
	// directly: host k-quant GEMV for all three projections plus the scalar SwiGLU.
	// That shares no code with the device seam, so agreement is real corroboration
	// that the declined encoding ran the unchanged host path.
	want := expertDownHostReference(m, x)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("declined expert diverged from the independent host oracle at %d: %v != %v", i, got[i], want[i])
		}
	}
}

// expertDownHostReference computes one expert's SwiGLU MLP entirely from the host
// k-quant GEMV primitives — no Backend, no session, no device seam. It is the
// independent oracle the decline test compares against. Each projection is served
// from whichever resident store the fixture populated: Q4_K in m.q4kw, raw k-quant
// in m.kqw.
func expertDownHostReference(m *Model, x []float32) []float32 {
	cfg := m.Cfg
	I := cfg.expertIntermediate()
	hostRows := func(name string, in []float32) []float32 {
		if qt := m.q4kw[name]; qt != nil {
			return q4kMatRows(qt, in)
		}
		return kQuantMatRows(m.kqw[name], in)
	}
	g := hostRows(expertName(0, 0, "gate_proj.weight"), x)
	u := hostRows(expertName(0, 0, "up_proj.weight"), x)
	h := make([]float32, I)
	for i := 0; i < I; i++ {
		h[i] = act(g[i], cfg) * u[i]
	}
	return hostRows(expertName(0, 0, "down_proj.weight"), h)
}
