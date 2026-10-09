//go:build vulkan && linux && cgo

// Run serially from the repository root: FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41MHCProjectionVulkan$' -count=1 -timeout=5m
// The synthetic full H5120/HC4 leaf has 491,520 cells and 1,966,080 host F32 bytes; this does not witness a full checkpoint or its dtype/layout.
package model

import (
	"encoding/json"
	"math"
	"os"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41MHCProjVulkanFixture(t *testing.T, dtype string) (*Model, *Model) {
	t.Helper()
	cfg := v41FullGeometryConfig(t)
	cfg.HiddenSize, cfg.OGroups = 5120, 1
	H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
	qHeadDim, oDim := cfg.NumHeads*cfg.HeadDim, cfg.OLoraRank*cfg.OGroups
	tensors := []synthTensor{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, H}},
		{"lm_head.weight", []int{cfg.VocabSize, H}},
		{"model.norm.weight", []int{H}},
		{layerName(0, "attn_norm.weight"), []int{H}},
		{layerName(0, "ffn_norm.weight"), []int{H}},
		{layerName(0, "mhc.mixes.weight"), []int{24, 4 * H}},
		{layerName(0, "mhc.base"), []int{24}},
		{layerName(0, "mhc.scale"), []int{3}},
		{layerName(0, "attn.wq_a.weight"), []int{cfg.QLoraRank, H}},
		{layerName(0, "attn.wq_a_norm.weight"), []int{cfg.QLoraRank}},
		{layerName(0, "attn.wq_b.weight"), []int{qHeadDim, cfg.QLoraRank}},
		{layerName(0, "attn.wkv.weight"), []int{v41KVLoraRank, H}},
		{layerName(0, "attn.kv_norm.weight"), []int{v41KVLoraRank}},
		{layerName(0, "attn.wo_a.weight"), []int{cfg.OLoraRank, qHeadDim}},
		{layerName(0, "attn.wo_b.weight"), []int{H, oDim}},
		{layerName(0, "attn.sink"), []int{cfg.NumHeads}},
		{layerName(0, "ffn.gate.weight"), []int{cfg.NumExperts, H}},
		{layerName(0, "ffn.gate.e_score_correction_bias"), []int{cfg.NumExperts}},
		{layerName(0, "ffn.shared_experts.w1.weight"), []int{I, H}},
		{layerName(0, "ffn.shared_experts.w3.weight"), []int{I, H}},
		{layerName(0, "ffn.shared_experts.w2.weight"), []int{H, I}},
	}
	for expert := 0; expert < cfg.NumExperts; expert++ {
		stem := "ffn.experts." + itoa(expert)
		tensors = append(tensors, synthTensor{layerName(0, stem+".w1.weight"), []int{I, H}}, synthTensor{layerName(0, stem+".w3.weight"), []int{I, H}}, synthTensor{layerName(0, stem+".w2.weight"), []int{H, I}})
	}
	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case name == "model.norm.weight" || hasSuffix(name, "attn_norm.weight") || hasSuffix(name, "ffn_norm.weight") || hasSuffix(name, "attn.wq_a_norm.weight") || hasSuffix(name, "attn.kv_norm.weight"):
			return 1
		case hasSuffix(name, "mhc.scale"):
			return 1
		case hasSuffix(name, "mhc.base"):
			return 0
		case hasSuffix(name, "attn.sink"):
			return .25 * next()
		default:
			return synthMatmulFill(name, next)
		}
	})
	m := &Model{Cfg: cfg, manifest: man, raw: raw}
	name := layerName(0, "mhc.mixes.weight")
	if dtype == "Q2_K" {
		m.kqw = map[string]*kQuantTensor{name: q2kFixtureTensor(24, 4*H, 13668105120)}
		delete(m.manifest, name)
	} else {
		logical := cpuOracleTensor(t, m, name)
		transposed := v41TransposeMixBlock(logical, 4*H)
		meta := m.manifest[name]
		meta.Offset = len(m.raw)
		meta.Shape = []int{4 * H, 24}
		for _, value := range transposed {
			var bytes [4]byte
			putMHCProjectionFloat(bytes[:], value)
			m.raw = append(m.raw, bytes[:]...)
		}
		// Both model owners share immutable ordinary source bytes; only the control's mHC metadata selects the appended true transpose.
		control := v41RawFullFlattenedMHC(t)
		control.Cfg, control.raw = cfg, m.raw
		for key := range control.manifest {
			delete(control.manifest, key)
		}
		for key, value := range m.manifest {
			control.manifest[key] = value
		}
		control.manifest[name] = meta
		return m, control
	}
	control := v41RawFullFlattenedMHC(t)
	control.Cfg, control.raw, control.kqw = cfg, m.raw, m.kqw
	for key := range control.manifest {
		delete(control.manifest, key)
	}
	for key, value := range m.manifest {
		control.manifest[key] = value
	}
	return m, control
}

func putMHCProjectionFloat(dst []byte, value float32) {
	bits := math.Float32bits(value)
	dst[0], dst[1], dst[2], dst[3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
}

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=60s lane=optin
func TestV41MHCProjectionVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical mHC witness requires explicit device opt-in")
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("physical witness requires dispatch profiling before process start")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required physical Vulkan device unavailable")
	}
	profile, ok := be.(interface {
		VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	})
	if !ok {
		t.Fatal("physical dispatch profile unavailable")
	}
	identity, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available {
		t.Fatal("physical identity unavailable")
	}
	device := strings.ToLower(identity.Identity.Device)
	if !strings.Contains(device, "radeon") || !strings.Contains(device, "8060s") || !strings.Contains(device, "radv") {
		t.Fatalf("requires Radeon 8060S RADV, got %q", identity.Identity.Device)
	}
	if !identity.TransferCountersObserved || !compute.BackendSupportsDeviceWeightDtype(be, compute.F32) || !compute.BackendSupportsDeviceWeightDtype(be, compute.Q2_K) {
		t.Fatal("physical transfers/F32/Q2 support unavailable")
	}
	for _, dtype := range []string{"F32", "Q2_K"} {
		t.Run(dtype, func(t *testing.T) {
			m, control := v41MHCProjVulkanFixture(t, dtype)
			if m.Cfg.HiddenSize != 5120 || m.Cfg.NumExperts != 384 || m.Cfg.DeepSeekV41.HCMult != 4 {
				t.Fatal("full synthetic mHC H/HC/expert geometry changed")
			}
			hc := m.Cfg.DeepSeekV41.HCMult
			inputWidth := hc * m.Cfg.HiddenSize
			// Each stream has a pre/post coefficient and a residual coefficient for every stream pair.
			coefficientWidth := 2*hc + hc*hc
			b, hostBackend := newV41MHCProjBackend(20480, false), newV41MHCProjBackend(20480, dtype == "Q2_K")
			b.Backend, hostBackend.Backend = be, be
			s, host := v41EngProjSession(t, m, b), v41EngProjSession(t, control, hostBackend)
			defer func() {
				s.Close()
				host.Close()
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
				if err := control.CloseWeights(); err != nil {
					t.Error(err)
				}
			}()
			weights, ok := m.residentF32Mat(layerName(0, "mhc.mixes.weight"))
			if !ok || len(weights) != 24*20480 {
				t.Fatal("independent raw coefficient oracle unavailable")
			}
			for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
				phase := "prefill"
				if index == 1 {
					phase = "decode"
				}
				before, hostBefore := v41MHCProjPhase(t, m, phase), v41MHCProjPhase(t, control, phase)
				denseBefore, groupBefore := v41DenseTestPhase(t, m, phase), v41GroupedPhase(t, m, phase)
				hostDenseBefore, hostGroupBefore := v41DenseTestPhase(t, control, phase), v41GroupedPhase(t, control, phase)
				opening, available, err := compute.CaptureBackendExecutionSnapshot(be)
				if err != nil || !available || opening.Identity != identity.Identity {
					t.Fatal("physical opening identity changed")
				}
				profileBefore := profile.VulkanDebugDispatchProfileSnapshot()
				from, attempts, totalFrom, hostFrom := len(b.operations), b.attempts, len(b.ops), len(hostBackend.ops)
				var got, want []float32
				if index == 1 {
					got, want = s.Step(ids[0]), host.Step(ids[0])
				} else {
					got, want = s.Prefill(ids), host.Prefill(ids)
				}
				closing, available, err := compute.CaptureBackendExecutionSnapshot(be)
				if err != nil || !available {
					t.Fatal("physical closing observation unavailable")
				}
				observation, err := compute.BackendExecutionDelta(opening, closing)
				if err != nil {
					t.Fatal(err)
				}
				delta, hostDelta := v41DenseTestDelta(v41MHCProjPhase(t, m, phase), before), v41DenseTestDelta(v41MHCProjPhase(t, control, phase), hostBefore)
				rows, upload, read := 0, 0, 0
				for _, op := range b.operations[from:] {
					rows += op.rows
					upload += op.upload
					read += op.read
					if len(op.activation) != inputWidth || len(op.result) != coefficientWidth {
						t.Fatal("full physical raw activation/result geometry changed")
					}
					var signal float64
					for out := 0; out < 24; out++ {
						var dot float64
						for cell, value := range op.activation {
							dot += float64(weights[out*20480+cell]) * float64(value)
						}
						signal = math.Max(signal, math.Abs(dot))
						if math.Abs(float64(op.result[out])-dot) > 1e-4*math.Max(1, math.Abs(dot)) {
							t.Fatalf("physical raw-before-RMS projection output %d got=%g want=%g", out, op.result[out], dot)
						}
					}
					if signal == 0 {
						t.Fatal("physical raw oracle vacuous")
					}
				}
				if rows != len(ids) || delta["mhc_projection_device_rows"] != float64(len(ids)) || delta["mhc_projection_device_calls"] != float64(len(ids)) || delta["mhc_projection_host_calls"] != 0 || delta["mhc_projection_host_rows"] != 0 || delta["mhc_projection_host_weight_f32_bytes"] != 0 {
					t.Errorf("physical default mHC route=%v actual rows=%d", delta, rows)
				}
				if delta["mhc_projection_matmul_calls"] != float64(b.attempts-attempts) || delta["mhc_projection_activation_upload_bytes"] != float64(upload) || delta["mhc_projection_readback_bytes"] != float64(read) || upload != 4*20480*len(ids) || read != 4*24*len(ids) || delta["mhc_projection_nanos"] <= 0 {
					t.Errorf("physical actual API/transfer/time=%v actual=%d/%d/%d", delta, b.attempts-attempts, upload, read)
				}
				materializations := 1
				if index == 2 {
					materializations = len(ids)
				}
				if hostDelta["mhc_projection_device_calls"] != 0 || hostDelta["mhc_projection_device_rows"] != 0 || hostDelta["mhc_projection_matmul_calls"] != 0 || hostDelta["mhc_projection_host_calls"] != float64(len(ids)) || hostDelta["mhc_projection_host_rows"] != float64(len(ids)) || hostDelta["mhc_projection_host_weight_f32_bytes"] != float64(1966080*materializations) || hostDelta["mhc_projection_nanos"] <= 0 {
					t.Errorf("actual whole-weight mHC host control=%v", hostDelta)
				}
				dense, group := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), denseBefore), v41DenseTestDelta(v41GroupedPhase(t, m, phase), groupBefore)
				hostDense, hostGroup := v41DenseTestDelta(v41DenseTestPhase(t, control, phase), hostDenseBefore), v41DenseTestDelta(v41GroupedPhase(t, control, phase), hostGroupBefore)
				if dense["dense_projection_device_rows"] != float64(7*len(ids)) || group["grouped_output_device_rows"] != float64(len(ids)) || hostDense["dense_projection_device_rows"] != float64(7*len(ids)) || hostGroup["grouped_output_device_rows"] != float64(len(ids)) {
					t.Errorf("ordinary default composition target=%v/%v control=%v/%v", dense, group, hostDense, hostGroup)
				}
				q2 := profile.VulkanDebugDispatchProfileSnapshot().Q2KMatmulDispatches - profileBefore.Q2KMatmulDispatches
				wantQ2 := 0
				if dtype == "Q2_K" {
					wantQ2 = b.attempts - attempts
				}
				if q2 != uint64(wantQ2) {
					t.Errorf("actual Q2 mHC dispatches=%d want %d", q2, wantQ2)
				}
				if observation.Counters.H2DBytes < uint64(upload) || observation.Counters.D2HBytes < uint64(read) || observation.Counters.ComputeDispatches < uint64(len(b.ops)-totalFrom+len(hostBackend.ops)-hostFrom) {
					t.Error("physical profile does not cover actual recorded APIs/transfers")
				}
				if b.stages != 1 || hostBackend.stages != 0 {
					t.Errorf("immutable full mHC staging selected=%d control=%d", b.stages, hostBackend.stages)
				}
				for weight := range b.weights {
					if b.reads[weight] != 0 {
						t.Error("physical witness read cached immutable mHC weight")
					}
				}
				v41GroupedParity(t, got, want, 1e-4)
				var dot, a2, c2, maxDiff float64
				greedyGot, greedyWant := 0, 0
				for cell, value := range got {
					a, c := float64(value), float64(want[cell])
					dot += a * c
					a2 += a * a
					c2 += c * c
					maxDiff = math.Max(maxDiff, math.Abs(a-c))
					if value > got[greedyGot] {
						greedyGot = cell
					}
					if want[cell] > want[greedyWant] {
						greedyWant = cell
					}
				}
				if a2 == 0 || c2 == 0 {
					t.Fatal("physical logit control vacuous")
				}
				cosine := dot / math.Sqrt(a2*c2)
				if cosine < .999 || greedyGot != greedyWant {
					t.Errorf("physical parity cosine=%g max_diff=%g greedy=%d/%d", cosine, maxDiff, greedyGot, greedyWant)
				}
				data, _ := json.Marshal(delta)
				controlData, _ := json.Marshal(hostDelta)
				t.Logf("backend=%s device=%q driver=%q runtime=%q dtype=%s synthetic_H5120_HC4_out24_in20480 phase=%s cold=%t tokens=%d actual_mhc_matmuls=%d actual_rows=%d q2_dispatches=%d h2d_bytes=%d d2h_bytes=%d compute_dispatches=%d cosine=%g max_diff=%g greedy=%d/%d default_mhc_json=%s actual_host_control_json=%s", observation.Identity.Backend, observation.Identity.Device, observation.Identity.Driver, observation.Identity.Runtime, dtype, phase, index == 0, len(ids), b.attempts-attempts, rows, q2, observation.Counters.H2DBytes, observation.Counters.D2HBytes, observation.Counters.ComputeDispatches, cosine, maxDiff, greedyGot, greedyWant, data, controlData)
			}
		})
		// Sequential subtest cleanup releases sessions and model caches before the next ~755 MB immutable synthetic source is constructed.
		runtime.GC()
	}
}
