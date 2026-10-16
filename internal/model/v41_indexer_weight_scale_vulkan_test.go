//go:build vulkan && (windows || linux) && cgo

// Run serially from the repository root with the coupled V13/73 shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41IndexWeightScaleVulkan$' -count=1
// Synthetic fixtures; no checkpoint, coherence or performance claim.

package model

import (
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The indexer weight scale runs on the strict scalar module and checked BF16
// publication, or their admitted fused operation. Scaled weights are bit-identical
// to the host arm across the V4.1 Flash geometry (dim 128, 64 heads) and the
// fixture geometries; a default session routes every index weight scale
// through the seam with logits bit-identical to the host-arm session; and with
// the static inventory drained, device-only admits the whole Q2_K/Q3_K
// routed-expert role fixture (compressed KV and the indexer) with an empty
// refusal list and decodes the reference's greedy tokens with no host fallback
// and no host BF16 round. The compressor fixture's F32-resident routed experts
// have no device staging path, so it serves only the default-policy arm.
// fak-test:justify why=integration when=changed:internal/model/v41_indexer_weight*
// fak-test:runtime slow est=30s lane=optin
func TestV41IndexWeightScaleVulkan(t *testing.T) {
	be := v41SeamVulkanBackend(t)
	exact, ok := be.(compute.V41ScaleExactBackend)
	if !ok || !exact.V41ScaleExactAvailable() {
		t.Fatal("ordinary Vulkan build did not admit the strict scalar twin on this device")
	}
	t.Run("seam", func(t *testing.T) {
		lanes, calls := 0, 0
		fusedCalls, scalarCalls := 0, 0
		for _, tc := range []struct{ dim, heads int }{{128, 64}, {128, 1}, {3, 4}, {64, 2}} {
			m := v41IndexerTestFixture(t)
			m.Cfg.IndexNHeads = tc.heads
			s, err := m.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			scaled, publish := s.v41IndexWeightScaleFunc(), s.v41BF16PublishFunc()
			if scaled == nil || publish == nil {
				t.Fatalf("dim=%d heads=%d: seam not bound on the device", tc.dim, tc.heads)
			}
			for r, row := range v41IndexWeightScaleRows(tc.heads) {
				want, err := v41IndexWeightsBF16(0, row, tc.dim, tc.heads)
				if err != nil {
					t.Fatal(err)
				}
				var got []float32
				var rounds int64
				var n uint64
				fusedSelected := false
				func() {
					scope := s.beginBackendExecution()
					complete := false
					defer scope.finishCompleted(&complete)
					if scope == nil || !scope.live || scope.owner != backendExecutionOwner(be) || s.backendExecutionScope != scope {
						t.Fatal("index weight scale witness requires the actual live execution owner")
					}
					// Probe the exact BF16 projection and scalar before measuring.
					// Admission alone is insufficient: the dispatch count below
					// must also distinguish fused execution from scalar fallback.
					if fused, ok := be.(compute.V41IndexScaleBF16Backend); ok {
						if _, admitted := v41DeferredBF16Publisher(be, tc.heads); admitted {
							projected, projectionErr := v41IndexBF16Copyback(0, "indexer.weights_proj.weight", "projection", row)
							if projectionErr != nil {
								t.Fatal(projectionErr)
							}
							func() {
								input := uploadHostF32Class(be, []int{tc.heads}, projected, compute.MemoryActivation, "index weight scale witness admission")
								defer be.Free(input)
								fusedSelected = fused.SupportsV41IndexScaleBF16(input, v41IndexWeightScale(tc.dim, tc.heads))
							}()
						}
					}
					n = v41SeamDispatches(t, be, func() {
						rounds = v41BF16PublishHostRounds(t, func() {
							got, err = v41IndexWeightsBF16Device(0, row, tc.dim, tc.heads, publish, scaled)
						})
					})
					complete = true
				}()
				if err != nil || rounds != 0 {
					t.Fatalf("dim=%d heads=%d row=%d: err=%v host_rounds=%d", tc.dim, tc.heads, r, err, rounds)
				}
				wantDispatches := uint64(2) // strict multiply plus scaled publication
				if fusedSelected {
					wantDispatches = 1
					fusedCalls++
				} else {
					scalarCalls++
				}
				if !v41BF16Widened(row) {
					wantDispatches++ // the initial projection also needs publication
				}
				if n != wantDispatches {
					t.Fatalf("dim=%d heads=%d row=%d fused=%t issued %d native dispatches, want %d", tc.dim, tc.heads, r, fusedSelected, n, wantDispatches)
				}
				if len(got) != len(want) {
					t.Fatalf("dim=%d heads=%d row=%d width=%d want=%d", tc.dim, tc.heads, r, len(got), len(want))
				}
				for i := range want {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("dim=%d heads=%d row=%d lane=%d device %08x host %08x", tc.dim, tc.heads, r, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
					}
				}
				lanes += len(want)
				calls++
			}
			if s.BackendSessionClosed() {
				t.Fatal("successful seams retired the session")
			}
			s.Close()
			if err := m.CloseWeights(); err != nil {
				t.Error(err)
			}
		}
		t.Logf("V41_INDEX_WEIGHT_SCALE_SEAM calls=%d lanes=%d bit_exact=%d fused=%d scalar=%d", calls, lanes, lanes, fusedCalls, scalarCalls)
	})
	t.Run("default-session", func(t *testing.T) {
		v41IndexWeightScaleVulkanForward(t, be, "compressor-fixture", func() *Model { return v41IndexerTestFixture(t) }, false)
	})
	t.Run("device-only-q2k-expert-role-fixture", func(t *testing.T) {
		v41IndexWeightScaleVulkanForward(t, be, "q2k-expert-role-fixture", func() *Model { return v41IncrementalExpertFixture(t, true, false) }, true)
	})
}

// v41IndexWeightScaleVulkanForward decodes the reference prompt on the device
// (under device-only when deviceOnly) and requires: every index weight scale on
// the seam (the cpu-ref recorder's count for the same fixture), zero host BF16
// rounds under device-only, logits bit-identical to a default-policy session
// whose weight scale runs on the host, and the reference's greedy tokens.
func v41IndexWeightScaleVulkanForward(t *testing.T, be compute.Backend, label string, fixture func() *Model, deviceOnly bool) {
	t.Helper()
	prompt := []int{1, 2, 3}
	ref, cpuModel, devModel, hostArmModel := fixture(), fixture(), fixture(), fixture()
	if !v41UsesIndexer(devModel.Cfg) {
		t.Fatalf("%s does not run the indexer; the witness would be vacuous", label)
	}
	want := v41DeviceParityReference(t, ref, prompt, v41DeviceParitySteps)
	if v41DeviceParityMargin(want.logits[0]) == 0 {
		t.Fatal("reference greedy choice is a tie; the token witness would be vacuous")
	}

	cpu := v41DenseTestSession(t, cpuModel, newV41IndexWeightScaleRecorder())
	stop := enableV41BF16Witness()
	v41DeviceParitySession(t, cpu, prompt, v41DeviceParitySteps)
	wantCalls := v41IndexWeightScaleCalls.Load()
	stop()
	if wantCalls == 0 {
		t.Fatalf("%s reaches no index weight scale; the session witness would be vacuous", label)
	}

	dev, err := devModel.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	if deviceOnly {
		dev.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		if len(v41DeviceOnlyHostOps) != 0 {
			t.Fatalf("static host inventory is not drained: %v", v41HostOpNames(v41DeviceOnlyHostOps))
		}
		if unbound := dev.v41DeviceOnlyUnboundOps(); len(unbound) != 0 {
			t.Fatalf("%s leaves device-only seams unbound: %v", label, v41HostOpNames(unbound))
		}
		if refusal := dev.v41DeviceOnlyAdmissionRefusal(); refusal != nil {
			t.Fatalf("%s device-only admission refusal = %v, want the empty list", label, refusal)
		}
	}
	if dev.v41State().gatherOps == nil || dev.v41State().gatherOps.indexWeightScale == nil {
		t.Fatal("index weight scale seam was not bound")
	}
	stop = enableV41BF16Witness()
	var got v41DeviceParityRun
	refusal := recoverError(func() {
		_ = v41SeamDispatches(t, be, func() { got = v41DeviceParitySession(t, dev, prompt, v41DeviceParitySteps) })
	})
	calls, rounds := v41IndexWeightScaleCalls.Load(), v41HostBF16Rounds.Load()
	stop()
	if refusal != nil {
		t.Fatalf("%s device session refused (device-only=%t): %v", label, deviceOnly, refusal)
	}
	if dev.HostFallbackObserved() || dev.BackendSessionClosed() {
		t.Fatalf("%s device session observed a host fallback or closed", label)
	}
	if calls != wantCalls {
		t.Fatalf("%s device session called the weight-scale seam %d times, want %d", label, calls, wantCalls)
	}
	if deviceOnly && rounds != 0 {
		t.Fatalf("%s device-only session rounded %d values to BF16 on the host", label, rounds)
	}

	hostArm, err := hostArmModel.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	hostArm.v41State().gatherOps.indexWeightScale = nil
	alt := v41DeviceParitySession(t, hostArm, prompt, v41DeviceParitySteps)
	for i := range alt.logits {
		for j := range alt.logits[i] {
			if math.Float32bits(got.logits[i][j]) != math.Float32bits(alt.logits[i][j]) {
				t.Fatalf("%s step %d logit %d device %08x host-arm %08x", label, i, j, math.Float32bits(got.logits[i][j]), math.Float32bits(alt.logits[i][j]))
			}
		}
	}
	refWorst := 0.0
	for i := range want.logits {
		var scale, delta float64
		for j, w := range want.logits[i] {
			scale = max(scale, math.Abs(float64(w)))
			delta = max(delta, math.Abs(float64(got.logits[i][j]-w)))
		}
		refWorst = max(refWorst, delta/max(1, scale))
	}
	if !slices.Equal(got.tokens, want.tokens) {
		t.Errorf("%s greedy tokens device=%v reference=%v", label, got.tokens, want.tokens)
	}
	t.Logf("V41_INDEX_WEIGHT_SCALE_FORWARD fixture=%s device_only=%t admission_refusal=none seam_calls=%d host_bf16_rounds=%d tokens device=%v reference=%v logits_bit_identical_to_host_arm=true reference_worst_rel=%.4g", label, deviceOnly, calls, rounds, got.tokens, want.tokens, refWorst)

	dev.Close()
	hostArm.Close()
	cpu.Close()
	for _, m := range []*Model{ref, cpuModel, devModel, hostArmModel} {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	}
}
