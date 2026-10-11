//go:build vulkan && (windows || linux) && cgo

// Run serially from the repository root with the coupled shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41DeviceForwardGreedyParityVulkan$' -count=1 -timeout=10m
// Synthetic three-layer full-geometry fixtures; no checkpoint, coherence or performance claim.

package model

import (
	"errors"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	v41DeviceParitySteps = 6
	// Same relative bound as the V4.1 session seam witnesses (v41GroupedParity
	// in TestV41NormRoPECallbacksHalo). Device reductions reorder F32 sums, and
	// BF16 stream publications can turn a few F32 ulps into one BF16 ulp.
	v41DeviceParityLogitTol = 2e-3
)

type v41DeviceParityRun struct {
	tokens []int
	logits [][]float32
}

func v41DeviceParityReference(t *testing.T, m *Model, prompt []int, steps int) v41DeviceParityRun {
	t.Helper()
	out := append([]int(nil), prompt...)
	var run v41DeviceParityRun
	for i := 0; i < steps; i++ {
		act, err := m.forwardV41(out, nil)
		if err != nil {
			t.Fatalf("host reference forward at step %d: %v", i, err)
		}
		last := append([]float32(nil), act.Logits[len(out)-1]...)
		run.logits = append(run.logits, last)
		next := argmaxF32(last)
		run.tokens = append(run.tokens, next)
		out = append(out, next)
	}
	return run
}

func v41DeviceParitySession(t *testing.T, s *Session, prompt []int, steps int) v41DeviceParityRun {
	t.Helper()
	var run v41DeviceParityRun
	last := s.Prefill(prompt)
	for i := 0; i < steps; i++ {
		if len(last) == 0 {
			t.Fatalf("session returned no logits at step %d", i)
		}
		last = append([]float32(nil), last...)
		run.logits = append(run.logits, last)
		next := argmaxF32(last)
		run.tokens = append(run.tokens, next)
		if i+1 < steps {
			last = s.Step(next)
		}
	}
	return run
}

// v41DeviceParitySeams names the session seams the backend bound, so a parity
// pass states which device ops it actually covered.
func v41DeviceParitySeams(st *v41ForwardState) (bound, host []string) {
	for _, seam := range []struct {
		name string
		on   bool
	}{
		{"expert_gate_up", st.expertGateUp != nil}, {"expert_down", st.expertDown != nil},
		{"dense_projection", st.denseProjection != nil}, {"grouped_output", st.groupedOutput != nil},
		{"engram_projection", st.engramProjection != nil}, {"mhc_projection", st.mhcProjection != nil},
		{"mhc_ffn_projection", st.mhcFFNProjection != nil}, {"final_norm", st.finalNorm != nil},
		{"attention_norm", st.attentionNorm != nil}, {"attention_bf16", st.attentionBF16 != nil},
		{"query_norm", st.queryNorm != nil}, {"kv_norm", st.kvNorm != nil}, {"ffn_norm", st.ffnNorm != nil},
		{"compressor_norm", st.compressorNorm != nil}, {"index_key_norm", st.indexKeyNorm != nil},
		{"indexer_score", st.indexerScore != nil}, {"shared_activation", st.sharedActivation != nil},
		{"tail_rope", st.tailRoPE != nil}, {"shared_attention", st.sharedAttention != nil},
		{"router_topk", st.routerTopK != nil}, {"routed_sum", st.routedSum != nil},
		{"shared_add", st.sharedAdd != nil}, {"gather_ops", st.gatherOps != nil},
		{"mhc_device", st.mhcDevice != nil},
	} {
		if seam.on {
			bound = append(bound, seam.name)
		} else {
			host = append(host, seam.name)
		}
	}
	return bound, host
}

// v41DeviceParityCompare requires identical greedy tokens and per-step logits
// within the relative bound, and returns the worst relative logit delta.
func v41DeviceParityCompare(t *testing.T, label string, got, want v41DeviceParityRun) float64 {
	t.Helper()
	worst := 0.0
	for i := range want.tokens {
		if len(got.logits[i]) != len(want.logits[i]) {
			t.Fatalf("%s step %d logits width=%d want=%d", label, i, len(got.logits[i]), len(want.logits[i]))
		}
		var scale, delta, dot, a2, b2 float64
		for j, w := range want.logits[i] {
			g := got.logits[i][j]
			if !finite32(g) {
				t.Fatalf("%s step %d logit %d nonfinite", label, i, j)
			}
			scale = max(scale, math.Abs(float64(w)))
			delta = max(delta, math.Abs(float64(g-w)))
			dot += float64(g) * float64(w)
			a2 += float64(g) * float64(g)
			b2 += float64(w) * float64(w)
		}
		if a2 == 0 || b2 == 0 {
			t.Fatalf("%s step %d logits vacuous", label, i)
		}
		rel := delta / max(1, scale)
		worst = max(worst, rel)
		t.Logf("%s step=%d token=%d/%d margin=%.4g max_abs=%.4g rel=%.4g cosine=%.9f", label, i, got.tokens[i], want.tokens[i], v41DeviceParityMargin(want.logits[i]), delta, rel, dot/math.Sqrt(a2*b2))
		if got.tokens[i] != want.tokens[i] {
			t.Errorf("%s greedy token[%d]=%d want %d (got %v want %v)", label, i, got.tokens[i], want.tokens[i], got.tokens, want.tokens)
		}
		if rel > v41DeviceParityLogitTol {
			t.Errorf("%s step %d logits max_abs=%g scale=%g rel=%g > %g", label, i, delta, scale, rel, v41DeviceParityLogitTol)
		}
	}
	return worst
}

func v41DeviceParityMargin(logits []float32) float64 {
	top, second := math.Inf(-1), math.Inf(-1)
	for _, v := range logits {
		switch f := float64(v); {
		case f > top:
			top, second = f, top
		case f > second:
			second = f
		}
	}
	return top - second
}

// v41DeviceParityDeviceOnly runs the same greedy decode under the device-only
// policy. While the host-op inventory is non-empty the admission must refuse
// typed; once it drains, the device-only forward must meet the same parity.
func v41DeviceParityDeviceOnly(t *testing.T, m *Model, be compute.Backend, prompt []int, want v41DeviceParityRun) {
	t.Helper()
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
	var got v41DeviceParityRun
	refusal := func() (refused error) {
		defer func() {
			if r := recover(); r != nil {
				var op *BackendForwardOperationError
				if e, ok := r.(error); ok && errors.As(e, &op) && op.Path == "device-only" {
					refused = e
					return
				}
				panic(r)
			}
		}()
		got = v41DeviceParitySession(t, s, prompt, len(want.tokens))
		return nil
	}()
	inventory := v41DeviceOnlyRefusal()
	switch {
	case refusal != nil && inventory != nil:
		t.Logf("device-only admission refused while the host-op inventory is non-empty: %v", refusal)
	case refusal != nil:
		t.Errorf("device-only admission refused with an empty host-op inventory: %v", refusal)
	case inventory != nil:
		t.Errorf("device-only forward admitted despite host-op inventory %v", inventory)
	default:
		if s.HostFallbackObserved() {
			t.Error("device-only forward observed a host fallback")
		}
		v41DeviceParityCompare(t, "device-only", got, want)
	}
}

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=60s lane=optin
func TestV41DeviceForwardGreedyParityVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical whole-forward parity witness requires explicit device opt-in")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	t.Logf("Vulkan device: %s", be.Tier())
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" {
		snap, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available || snap.Identity.Device != expected {
			t.Fatalf("physical device=%q want=%q (available=%t err=%v)", snap.Identity.Device, expected, available, err)
		}
	}
	prompt := []int{1, 2, 3}
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprintf("layers3_engram_compressed_%t", compressed), func(t *testing.T) {
			fixture := func() *Model {
				m, _ := v41IncrementalEngramFixture(t, compressed)
				return m
			}
			ref, hostModel, devModel, onlyModel := fixture(), fixture(), fixture(), fixture()
			defer func() {
				for _, m := range []*Model{ref, hostModel, devModel, onlyModel} {
					if err := m.CloseWeights(); err != nil {
						t.Error(err)
					}
				}
			}()
			want := v41DeviceParityReference(t, ref, prompt, v41DeviceParitySteps)
			if v41DeviceParityMargin(want.logits[0]) == 0 {
				t.Fatal("reference greedy choice is a tie; the token witness would be vacuous")
			}

			host := &Session{M: hostModel}
			hostRun := v41DeviceParitySession(t, host, prompt, v41DeviceParitySteps)
			host.Close()
			v41DeviceParityCompare(t, "host-session", hostRun, want)

			dev, err := devModel.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			defer dev.Close()
			bound, unbound := v41DeviceParitySeams(dev.v41State())
			if len(bound) == 0 {
				t.Fatal("Vulkan session bound no V4.1 device seam; the parity would compare host with host")
			}
			t.Logf("device seams bound=%v host=%v", bound, unbound)
			before, observed, _ := compute.CaptureBackendExecutionSnapshot(be)
			devRun := v41DeviceParitySession(t, dev, prompt, v41DeviceParitySteps)
			if after, ok, _ := compute.CaptureBackendExecutionSnapshot(be); observed && ok {
				delta, err := compute.BackendExecutionDelta(before, after)
				if err != nil {
					t.Fatal(err)
				}
				if delta.Counters.ComputeDispatches == 0 {
					t.Fatal("device forward issued no compute dispatches")
				}
				t.Logf("device forward compute_dispatches=%d h2d_bytes=%d d2h_bytes=%d", delta.Counters.ComputeDispatches, delta.Counters.H2DBytes, delta.Counters.D2HBytes)
			}
			worst := v41DeviceParityCompare(t, "device", devRun, want)
			t.Logf("greedy tokens device=%v reference=%v worst_rel=%.4g", devRun.tokens, want.tokens, worst)

			v41DeviceParityDeviceOnly(t, onlyModel, be, prompt, want)
		})
	}
}
