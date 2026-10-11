//go:build vulkan && (windows || linux) && cgo

// Run serially from the repository root with the coupled shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41(AttentionInputNorm|IndexRoPE)Vulkan$' -count=1
// Synthetic fixtures; no checkpoint, coherence or performance claim.

package model

import (
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41SeamVulkanBackend(t *testing.T) compute.Backend {
	t.Helper()
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical seam witness requires explicit device opt-in")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	snap, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available {
		t.Fatalf("execution snapshot unavailable: available=%t err=%v", available, err)
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && snap.Identity.Device != expected {
		t.Fatalf("physical device=%q want=%q", snap.Identity.Device, expected)
	}
	t.Logf("Vulkan device: %s", snap.Identity.Device)
	return be
}

func v41SeamDispatches(t *testing.T, be compute.Backend, run func()) uint64 {
	t.Helper()
	before, ok, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !ok {
		t.Fatalf("execution snapshot before: ok=%t err=%v", ok, err)
	}
	run()
	after, ok, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !ok {
		t.Fatalf("execution snapshot after: ok=%t err=%v", ok, err)
	}
	delta, err := compute.BackendExecutionDelta(before, after)
	if err != nil {
		t.Fatal(err)
	}
	return delta.Counters.ComputeDispatches
}

// v41SeamBF16Ulps is the BF16 ulp distance between two finite BF16 values.
func v41SeamBF16Ulps(t *testing.T, label string, got, want float32) int {
	t.Helper()
	g, w := math.Float32bits(got), math.Float32bits(want)
	if !finite32(got) || g&0xffff != 0 || w&0xffff != 0 {
		t.Fatalf("%s got=%08x want=%08x is not a finite BF16 publication", label, g, w)
	}
	ordered := func(bits uint32) int {
		if bits&0x80000000 != 0 {
			return -int(bits >> 16 & 0x7fff)
		}
		return int(bits >> 16)
	}
	d := ordered(g) - ordered(w)
	if d < 0 {
		d = -d
	}
	return d
}

// v41SeamGreedy decodes one prompt in two default Vulkan sessions that differ
// only in the named seam: arm A keeps it on the device and counts its calls,
// arm B unbinds it so the forward runs that op's host arm. A must match B within
// the seam parity bound and keep the host reference's greedy tokens; logit drift
// that A shares with B belongs to the other bound seams, so it is only logged.
func v41SeamGreedy(t *testing.T, label string, fixture func() *Model, be compute.Backend, wrap func(*v41ForwardState) *int, unbind func(*v41ForwardState), wantCalls func(*Model) int) {
	t.Helper()
	prompt := []int{1, 2, 3}
	ref, devModel, hostArmModel := fixture(), fixture(), fixture()
	defer func() {
		for _, m := range []*Model{ref, devModel, hostArmModel} {
			if err := m.CloseWeights(); err != nil {
				t.Error(err)
			}
		}
	}()
	want := v41DeviceParityReference(t, ref, prompt, v41DeviceParitySteps)
	if v41DeviceParityMargin(want.logits[0]) == 0 {
		t.Fatal("reference greedy choice is a tie; the token witness would be vacuous")
	}
	dev, err := devModel.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	calls := wrap(dev.v41State())
	var got v41DeviceParityRun
	dispatches := v41SeamDispatches(t, be, func() { got = v41DeviceParitySession(t, dev, prompt, v41DeviceParitySteps) })
	if dev.HostFallbackObserved() {
		t.Fatal("default device session observed a host fallback")
	}
	hostArm, err := hostArmModel.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer hostArm.Close()
	unbind(hostArm.v41State())
	alt := v41DeviceParitySession(t, hostArm, prompt, v41DeviceParitySteps)
	abWorst := v41DeviceParityCompare(t, label+" device-vs-host-arm", got, alt)
	identical := true
	for i := range alt.logits {
		for j := range alt.logits[i] {
			identical = identical && math.Float32bits(got.logits[i][j]) == math.Float32bits(alt.logits[i][j])
		}
	}
	refWorst := 0.0
	for i := range want.tokens {
		if got.tokens[i] != want.tokens[i] {
			t.Errorf("%s greedy token[%d]=%d want reference %d", label, i, got.tokens[i], want.tokens[i])
		}
		var scale, delta float64
		for j, w := range want.logits[i] {
			scale = max(scale, math.Abs(float64(w)))
			delta = max(delta, math.Abs(float64(got.logits[i][j]-w)))
		}
		refWorst = max(refWorst, delta/max(1, scale))
	}
	t.Logf("%s default session seam_calls=%d compute_dispatches=%d tokens device=%v host_arm=%v reference=%v ab_worst_rel=%.4g ab_bit_identical=%t reference_worst_rel=%.4g", label, *calls, dispatches, got.tokens, alt.tokens, want.tokens, abWorst, identical, refWorst)
	if expected := wantCalls(devModel); *calls != expected {
		t.Errorf("%s default session forward called the device seam %d times, want %d", label, *calls, expected)
	}
}

// The attention-input RMSNorm runs on the device seam by default in a
// full-geometry session (#13810). The seam matches the host arm within one BF16
// ulp per lane, and the default session forward routes every layer and token
// through it, matching the host-arm session and the host greedy tokens.
// fak-test:justify why=integration when=changed:internal/model/v41_attention_input_norm*
// fak-test:runtime slow est=20s lane=optin
func TestV41AttentionInputNormVulkan(t *testing.T) {
	be := v41SeamVulkanBackend(t)
	fixture := func() *Model {
		m, _ := v41IncrementalEngramFixture(t, false)
		return m
	}
	t.Run("seam", func(t *testing.T) {
		m := fixture()
		t.Cleanup(func() {
			if err := m.CloseWeights(); err != nil {
				t.Error(err)
			}
		})
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		normalize, publish := s.v41State().attentionNorm, s.v41State().attentionBF16
		if normalize == nil {
			t.Fatal("attention-input norm seam was not bound for the full-geometry fixture")
		}
		eps, width := float32(m.Cfg.RMSNormEps), m.Cfg.HiddenSize
		lanes, exact, worst := 0, 0, 0
		for l := 0; l < m.Cfg.NumLayers; l++ {
			for row := 0; row < 4; row++ {
				x := make([]float32, width)
				for i := range x {
					x[i] = float32(((i*7+row*13+l*5)%29)-14)*0.0625 + float32(row+1)*0.0009765625
				}
				want, err := m.v41AttentionInputNorm(l, append([]float32(nil), x...), eps)
				if err != nil {
					t.Fatal(err)
				}
				var got []float32
				n := v41SeamDispatches(t, be, func() {
					got, err = m.v41AttentionInputNormWithPublications(l, append([]float32(nil), x...), eps, normalize, publish)
				})
				if err != nil {
					t.Fatal(err)
				}
				if n == 0 {
					t.Fatalf("layer %d row %d device norm issued no native dispatch", l, row)
				}
				if len(got) != len(want) {
					t.Fatalf("layer %d row %d width=%d want=%d", l, row, len(got), len(want))
				}
				for i := range want {
					d := v41SeamBF16Ulps(t, "attention-input norm", got[i], want[i])
					if d > 1 {
						t.Fatalf("layer %d row %d lane %d got %g want %g (%d BF16 ulps)", l, row, i, got[i], want[i], d)
					}
					lanes++
					worst = max(worst, d)
					if d == 0 {
						exact++
					}
				}
			}
		}
		t.Logf("V41_ATTENTION_INPUT_NORM_SEAM lanes=%d bit_exact=%d max_bf16_ulps=%d bf16_publication_bound=%t", lanes, exact, worst, publish != nil)
	})
	t.Run("default-session", func(t *testing.T) {
		v41SeamGreedy(t, "attention-input-norm", fixture, be, func(st *v41ForwardState) *int {
			calls, inner := new(int), st.attentionNorm
			if inner == nil {
				t.Fatal("attention-input norm seam was not bound")
			}
			st.attentionNorm = func(l int, x []float32) ([]float32, error) {
				*calls++
				return inner(l, x)
			}
			return calls
		}, func(st *v41ForwardState) {
			st.attentionNorm, st.attentionBF16 = nil, nil
		}, func(m *Model) int {
			return m.Cfg.NumLayers * (3 + v41DeviceParitySteps - 1)
		})
	})
}

// The indexer query RoPE runs on the tail-RoPE rows seam by default (#13817).
// Prefix lanes stay bit-exact and the BF16 tails match the host arm within one
// BF16 ulp; the default session forward routes every index-source layer through
// the seam, matching the host-arm session and the host greedy tokens.
// fak-test:justify why=integration when=changed:internal/model/v41_index*
// fak-test:runtime slow est=20s lane=optin
func TestV41IndexRoPEVulkan(t *testing.T) {
	be := v41SeamVulkanBackend(t)
	t.Run("seam", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		s := v41DenseTestSession(t, m, be)
		rotate := s.v41State().gatherOps.indexRoPE
		if rotate == nil {
			t.Fatal("indexer query RoPE seam was not bound")
		}
		cfg := m.Cfg
		heads, dim, rd := cfg.IndexNHeads, cfg.IndexHeadDim, cfg.QKRopeHeadDim
		if heads <= 0 || dim <= 0 || rd <= 0 || rd > dim {
			t.Fatalf("fixture index geometry heads=%d dim=%d rotary=%d", heads, dim, rd)
		}
		tails, exact, worst := 0, 0, 0
		for l := 0; l < cfg.NumLayers; l++ {
			if !indexSourceAt(cfg.DeepSeekV41, l) {
				continue
			}
			for _, pos := range []int{0, 1, 7, 130, 4097} {
				q := make([]float32, heads*dim)
				for i := range q {
					q[i] = v41RoundBF16(float32(((i*11+pos*3+l)%37)-18)*0.03125 + 0.00048828125)
				}
				want := append([]float32(nil), q...)
				if err := m.v41IndexRoPE(l, pos, want, heads); err != nil {
					t.Fatal(err)
				}
				got := append([]float32(nil), q...)
				var err error
				n := v41SeamDispatches(t, be, func() { err = m.v41IndexRoPEWithDevice(l, pos, got, heads, rotate) })
				if err != nil {
					t.Fatal(err)
				}
				if n == 0 {
					t.Fatalf("layer %d pos %d device rotation issued no native dispatch", l, pos)
				}
				for i := range want {
					if i%dim < dim-rd {
						if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
							t.Fatalf("layer %d pos %d prefix lane %d changed", l, pos, i)
						}
						continue
					}
					d := v41SeamBF16Ulps(t, "indexer query tail", got[i], want[i])
					if d > 1 {
						t.Fatalf("layer %d pos %d tail lane %d got %g want %g (%d BF16 ulps)", l, pos, i, got[i], want[i], d)
					}
					tails++
					worst = max(worst, d)
					if d == 0 {
						exact++
					}
				}
			}
		}
		if tails == 0 {
			t.Fatal("fixture has no index-source layer; the seam witness would be vacuous")
		}
		t.Logf("V41_INDEXER_QUERY_ROPE_SEAM tail_lanes=%d bit_exact=%d max_bf16_ulps=%d", tails, exact, worst)
	})
	t.Run("default-session", func(t *testing.T) {
		v41SeamGreedy(t, "indexer-query-rope", func() *Model { return v41IndexerTestFixture(t) }, be, func(st *v41ForwardState) *int {
			calls := new(int)
			if st.gatherOps == nil || st.gatherOps.indexRoPE == nil {
				t.Fatal("indexer query RoPE seam was not bound")
			}
			inner := st.gatherOps.indexRoPE
			st.gatherOps.indexRoPE = func(l int, x, cos, sin []float32, heads, hd, rd int) ([]float32, error) {
				*calls++
				return inner(l, x, cos, sin, heads, hd, rd)
			}
			return calls
		}, func(st *v41ForwardState) {
			st.gatherOps.indexRoPE = nil
		}, func(m *Model) int {
			sources := 0
			for l := 0; l < m.Cfg.NumLayers; l++ {
				if indexSourceAt(m.Cfg.DeepSeekV41, l) {
					sources++
				}
			}
			return sources * (3 + v41DeviceParitySteps - 1)
		})
	})
}
