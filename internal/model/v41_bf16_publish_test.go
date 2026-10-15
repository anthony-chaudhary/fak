package model

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A cpu-ref-backed recorder for the ordinary BF16 publication (and the
// attention pair it shares a module with). It witnesses the Session wiring,
// ownership and site policy, not a physical Vulkan command.
type v41BF16PublishRecorder struct {
	*v41DenseTestBackend
	sites   []uint32
	calls   int
	faultAt int
	fault   error
	// self names the outermost recorder when another recorder embeds this
	// one, so fresh tensors carry the session backend's identity.
	self compute.Backend
	// A bound ledger adds the deferred-status publication.
	*v41DeferredLedger
}

func (b *v41BF16PublishRecorder) owner() compute.Backend {
	if b.self != nil {
		return b.self
	}
	return b
}

func newV41BF16PublishRecorder() *v41BF16PublishRecorder {
	return &v41BF16PublishRecorder{v41DenseTestBackend: newV41DenseTestBackend()}
}

func (b *v41BF16PublishRecorder) SupportsV41BF16Publish(n int) bool   { return n > 0 }
func (b *v41BF16PublishRecorder) SupportsV41AttentionBF16(n int) bool { return n > 0 }
func (b *v41BF16PublishRecorder) V41AttentionBF16(x compute.Tensor, site uint32) (compute.Tensor, error) {
	if site != 3 && site != 4 {
		panic("attention publication site")
	}
	return b.publish(x, site)
}
func (b *v41BF16PublishRecorder) V41BF16Publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	if !compute.V41BF16PublishSite(site) {
		panic("ordinary publication site")
	}
	return b.publish(x, site)
}
func (b *v41BF16PublishRecorder) publish(x compute.Tensor, site uint32) (compute.Tensor, error) {
	b.calls++
	b.sites = append(b.sites, site)
	if len(x.Shape) != 1 || x.Dtype != compute.F32 {
		panic("publication shape")
	}
	if b.calls == b.faultAt {
		return compute.Tensor{}, b.fault
	}
	values := append([]float32(nil), b.Backend.Read(x)...)
	for i, v := range values {
		bits := math.Float32bits(v)
		rounded := v41BF16BoundaryOracle(bits)
		if bits&0x7f800000 == 0x7f800000 || rounded&0x7f800000 == 0x7f800000 {
			return compute.Tensor{}, &compute.V41BF16CheckedArithmeticError{Site: site, Index: i}
		}
		values[i] = math.Float32frombits(rounded)
	}
	return compute.NewF32(b.owner(), []int{len(values)}, values), nil
}
func (b *v41BF16PublishRecorder) UploadClass(x compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if site != "V4.1 BF16 publication input" && site != "V4.1 attention BF16 input" {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	return compute.NewF32(b.owner(), append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
}

// v41BF16PublishRows are finite rows that are not widened BF16, so a bound
// publication must round them: ties both ways, subnormals and signed zero.
func v41BF16PublishRow(n int) []float32 {
	words := []uint32{0x3f808000, 0x3f818000, 0xbf818000, 0x00018000, 0x80000000, 0x7e7f7fff, 0x3f807fff}
	row := make([]float32, n)
	for i := range row {
		row[i] = math.Float32frombits(words[i%len(words)] + uint32(i/len(words)%64)<<16)
	}
	return row
}

func v41BF16PublishHostRounds(t *testing.T, run func()) int64 {
	t.Helper()
	stop := enableV41BF16Witness()
	defer stop()
	run()
	return v41HostBF16Rounds.Load()
}

// Every helper routes through the bound publication without a host round, is
// bit-exact against its host arm, and copies an already widened row unchanged.
// fak-test:runtime fast est=200ms lane=default
func TestV41BF16PublishHelpersRoundOnDevice(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	b := newV41BF16PublishRecorder()
	s := v41DenseTestSession(t, m, b)
	publish := s.v41BF16PublishFunc()
	if publish == nil || s.v41State().gatherOps.bf16Publish == nil || s.v41State().bf16Boundary == nil {
		t.Fatal("ordinary publication not bound for a backend that supports it")
	}
	H := m.Cfg.HiddenSize
	row := v41BF16PublishRow(H)
	type helper struct {
		name string
		site uint32
		run  func(v41BF16BoundaryFunc) ([]float32, error)
	}
	helpers := []helper{
		{"query-norm", compute.V41BF16PublishQueryNormInput, func(p v41BF16BoundaryFunc) ([]float32, error) {
			return v41LatentNormBF16Publish(0, "attn.wq_a_norm.weight", "projection", row, p)
		}},
		{"ffn-norm", compute.V41BF16PublishFFNNormOutput, func(p v41BF16BoundaryFunc) ([]float32, error) {
			return v41LatentNormBF16Publish(0, "ffn_norm.weight", "normalization", row, p)
		}},
		{"routed", compute.V41BF16PublishRoutedDown, func(p v41BF16BoundaryFunc) ([]float32, error) {
			return v41RoutedBF16Publish(0, "down", row, H, p)
		}},
		{"inverse-output", compute.V41BF16PublishInverseOutputInput, func(p v41BF16BoundaryFunc) ([]float32, error) {
			return v41RoPEBF16Publish(0, v41InverseOutputInputSite, row, p)
		}},
		{"tail-rope-pair", compute.V41BF16PublishTailRoPEOutput, func(p v41BF16BoundaryFunc) ([]float32, error) {
			q, kv, err := v41RoPEBF16Pair(0, v41TailRoPEOutputSite, row[:H-3], row[H-3:], p)
			return append(q, kv...), err
		}},
		{"indexer", compute.V41BF16PublishIndexerProjection, func(p v41BF16BoundaryFunc) ([]float32, error) {
			return v41IndexBF16Publish(0, "indexer.wq_b.weight", "projection", row, p)
		}},
		{"indexer-tails", compute.V41BF16PublishIndexerRotary, func(p v41BF16BoundaryFunc) ([]float32, error) {
			out := append([]float32(nil), row[:16]...)
			return out, v41IndexTailsBF16(0, out, 4, 4, 2, p)
		}},
		{"final-norm", compute.V41BF16PublishFinalNormOutput, func(p v41BF16BoundaryFunc) ([]float32, error) {
			return v41BF16BoundaryCopy(-1, "model.norm.weight output", row, p)
		}},
	}
	for _, h := range helpers {
		t.Run(h.name, func(t *testing.T) {
			want, err := h.run(nil)
			if err != nil {
				t.Fatal(err)
			}
			calls := b.calls
			var got []float32
			if rounds := v41BF16PublishHostRounds(t, func() { got, err = h.run(publish) }); rounds != 0 || err != nil {
				t.Fatalf("bound publication rounded %d values on the host (err=%v)", rounds, err)
			}
			if b.calls != calls+1 || b.sites[len(b.sites)-1] != h.site {
				t.Fatalf("publication calls=%d site=%v, want one call at site %d", b.calls-calls, b.sites[calls:], h.site)
			}
			if len(got) != len(want) {
				t.Fatalf("width %d want %d", len(got), len(want))
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("lane %d device %08x host %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
			widened := make([]float32, len(row))
			for i, v := range row {
				widened[i] = v41RoundBF16(v)
			}
			calls = b.calls
			var again []float32
			if rounds := v41BF16PublishHostRounds(t, func() {
				again, err = v41PublishBF16(0, "attn.kv_norm.weight projection", widened, publish, nil)
			}); rounds != 0 || err != nil || b.calls != calls || !slices.Equal(again, widened) {
				t.Fatalf("widened row was not copied unchanged without a publication: calls=%d rounds=%d err=%v", b.calls-calls, rounds, err)
			}
			again[0] = 99
			if widened[0] == 99 {
				t.Fatal("widened copy aliases its input")
			}
		})
	}
	if s.BackendSessionClosed() {
		t.Fatal("successful publications retired the session")
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestV41BF16PublishSiteTableIsClosed(t *testing.T) {
	seen := map[uint32]bool{}
	for site, id := range v41BF16PublishSites {
		if !compute.V41BF16PublishSite(id) || id == 3 || id == 4 {
			t.Fatalf("model site %q maps to %d outside the ordinary policy", site, id)
		}
		seen[id] = true
	}
	for id := uint32(0); id <= 64; id++ {
		if compute.V41BF16PublishSite(id) && id != compute.V41BF16PublishFinalNormInput && id != compute.V41BF16PublishFinalNormOutput && !seen[id] {
			t.Fatalf("ordinary site %d has no model boundary", id)
		}
	}
	for _, tc := range []struct {
		layer int
		site  string
		want  uint32
	}{
		{-1, "model.norm.weight input", 1}, {-1, "model.norm.weight output", 2},
		{0, "model.norm.weight input", 0}, {-1, "attn.kv_norm.weight projection", 0},
		{2, "attn.kv_norm.weight projection", 0}, {1, "attn_norm.weight input", 0},
		{1, "attn.kv_norm.weight projection", compute.V41BF16PublishKVNormInput},
	} {
		if got := v41BF16PublishSiteID(tc.layer, 2, tc.site); got != tc.want {
			t.Fatalf("site(%d,%q)=%d want %d", tc.layer, tc.site, got, tc.want)
		}
	}
}

// A real full-profile session publishes through the bound seam and keeps the
// host reference's logits; device-only admission binds the bf16-boundary row
// only when both the ordinary and attention publications are available.
// fak-test:runtime medium est=5s lane=default
func TestV41BF16PublishSession(t *testing.T) {
	v41AttentionBF16RequireCPU(t)
	m := v41FullStepPatchedNorms(t)
	b := newV41BF16PublishRecorder()
	s := v41DenseTestSession(t, m, b)
	st := s.v41State()
	normalize := st.attentionNormFunc()
	if normalize == nil || st.attentionBF16Func() == nil {
		t.Fatal("attention normalization/publication not bound")
	}
	attentionInputs := 0
	st.setAttentionNorm(func(layer int, row []float32) ([]float32, error) {
		attentionInputs++
		for i, value := range row {
			if !finite32(value) || math.Float32bits(value)&0xffff != 0 {
				t.Fatalf("attention norm received unpublished input at layer %d lane %d", layer, i)
			}
		}
		return normalize(layer, row)
	})
	v41AttentionBF16Parity(t, s.Prefill([]int{1, 2}), lastLogits(m.Forward([]int{1, 2})))
	v41AttentionBF16Parity(t, s.Step(3), lastLogits(m.Forward([]int{1, 2, 3})))
	if b.calls == 0 || s.BackendSessionClosed() {
		t.Fatalf("session issued %d publications (closed=%t)", b.calls, s.BackendSessionClosed())
	}
	attentionOutputs := 0
	for _, site := range b.sites {
		if !compute.V41BF16PublishSite(site) && site != 3 && site != 4 {
			t.Fatalf("publication outside the closed policy: %d", site)
		}
		if site == compute.V41BF16CheckedAttentionNormInput {
			t.Fatal("session redundantly published its already widened BF16 attention input")
		}
		if site == compute.V41BF16CheckedAttentionNormOutput {
			attentionOutputs++
		}
	}
	// This fixture's collapse is already BF16, so input publication is an
	// owned identity copy. Every token still normalizes and publishes its output.
	// TestV41AttentionBF16InputIdentity covers non-BF16 and invalid input dispatch.
	if want := 3 * m.Cfg.NumLayers; attentionInputs != want || attentionOutputs != want {
		t.Fatalf("attention inputs=%d outputs=%d, want %d each", attentionInputs, attentionOutputs, want)
	}
	for _, want := range []uint32{compute.V41BF16PublishFinalNormInput, compute.V41BF16PublishFinalNormOutput} {
		if !slices.Contains(b.sites, want) {
			t.Fatalf("session never published site %d: %v", want, b.sites)
		}
	}
	names := func(s *Session) []string { return v41HostOpNames(s.v41DeviceOnlyUnboundOps()) }
	if slices.Contains(names(s), "bf16-boundary") {
		t.Fatalf("publication backend leaves bf16-boundary unbound: %v", names(s))
	}
	if plain := v41DenseTestSession(t, m, newV41DenseTestBackend()); !slices.Contains(names(plain), "bf16-boundary") {
		t.Fatalf("backend without the publication binds bf16-boundary: %v", names(plain))
	}
}

// Under device-only, an unbound publication refuses typed and retires the
// session; a selected publication failure retires it without host replay.
// fak-test:runtime fast est=200ms lane=default
func TestV41BF16PublishFailsClosed(t *testing.T) {
	m := v41FullStepPatchedNorms(t)
	row := v41BF16PublishRow(m.Cfg.HiddenSize)
	t.Run("device-only-unbound", func(t *testing.T) {
		s := v41DenseTestSession(t, m, newV41DenseTestBackend())
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		ops := s.v41DeviceGatherOpsFunc()
		publish := ops.bf16PublishFunc()
		if publish == nil {
			t.Fatal("device-only accessor returned no refusal for an unbound publication")
		}
		widened := []float32{1, 2, 0.5}
		if out, err := v41LatentNormBF16Publish(0, "ffn_norm.weight", "collapse", widened, publish); err != nil || !slices.Equal(out, widened) || s.BackendSessionClosed() {
			t.Fatalf("widened row needs no publication: %v", err)
		}
		var rounds int64
		var err error
		rounds = v41BF16PublishHostRounds(t, func() {
			_, err = v41LatentNormBF16Publish(0, "ffn_norm.weight", "collapse", row, publish)
		})
		if rounds != 0 || !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.Is(err, ErrV41BF16BoundaryOperation) || !s.BackendSessionClosed() {
			t.Fatalf("unbound device-only publication = %v (rounds=%d closed=%t)", err, rounds, s.BackendSessionClosed())
		}
	})
	t.Run("selected-failure", func(t *testing.T) {
		b := newV41BF16PublishRecorder()
		s := v41DenseTestSession(t, m, b)
		publish := s.v41BF16PublishFunc()
		primary := &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: errors.New("device fault"), Recovered: &compute.V41BF16CheckedArithmeticError{Site: 9, Index: 2}}
		b.faultAt, b.fault = b.calls+1, primary
		var err error
		rounds := v41BF16PublishHostRounds(t, func() {
			_, err = v41LatentNormBF16Publish(0, "ffn_norm.weight", "collapse", row, publish)
		})
		var numeric *compute.V41BF16CheckedArithmeticError
		if rounds != 0 || !errors.Is(err, ErrV41BF16BoundaryOperation) || !errors.As(err, &numeric) || !s.BackendSessionClosed() {
			t.Fatalf("selected failure = %v (rounds=%d closed=%t)", err, rounds, s.BackendSessionClosed())
		}
		calls := b.calls
		var retry error
		if panicked := recoverError(func() { _, retry = publish(0, "ffn_norm.weight collapse", row) }); (retry == nil && panicked == nil) || b.calls != calls {
			t.Fatal("retired publication replayed")
		}
	})
	t.Run("unknown-site", func(t *testing.T) {
		b := newV41BF16PublishRecorder()
		s := v41DenseTestSession(t, m, b)
		if out, err := s.v41BF16PublishFunc()(0, "attn_norm.weight input", row); out != nil || !errors.Is(err, ErrV41BF16BoundaryOperation) || b.calls != 0 || !s.BackendSessionClosed() {
			t.Fatalf("attention site leaked into the ordinary publication: %v calls=%d", err, b.calls)
		}
	})
}

// fak-test:runtime fast est=20ms lane=default
func TestV41BF16PublishFaultPredicates(t *testing.T) {
	for _, tc := range []struct {
		bits     uint32
		overflow bool
	}{
		{0x7f7f7fff, false}, {0x7f7f8000, true}, {0xff7f8000, true}, {0x7f7fffff, true},
		{0x7f800000, false}, {0x3f800000, false}, {0x80000000, false}, {0x7f7f0000, false},
	} {
		v := math.Float32frombits(tc.bits)
		if v41BF16Overflows(v) != tc.overflow {
			t.Fatalf("%08x overflow=%t", tc.bits, !tc.overflow)
		}
		if finite32(v) && tc.overflow != !finite32(v41RoundBF16(v)) {
			t.Fatalf("%08x predicate disagrees with RNE", tc.bits)
		}
	}
	values := []float32{1, 2, math.Float32frombits(0x7f7f8000), float32(math.NaN())}
	if i, bad := v41BF16PublishFault(values); !bad || i != 2 {
		t.Fatalf("fault index=%d bad=%t want 2", i, bad)
	}
	if _, bad := v41BF16PublishFault([]float32{1, -0, 0x1p-130}); bad {
		t.Fatal("finite representable row reported a fault")
	}
	// The mHC refusal matches the host publication's typed error and index.
	for _, row := range [][]float32{{1, float32(math.Inf(1))}, {math.Float32frombits(0x7f7f8000), float32(math.NaN())}} {
		_, hostErr := v41MHCBF16(3, row)
		if got := v41MHCBF16Refusal(3, row); hostErr == nil || got == nil || got.Error() != hostErr.Error() || !errors.Is(got, ErrV41ForwardStage) {
			t.Fatalf("refusal %v differs from host %v", got, hostErr)
		}
	}
}
