//go:build vulkan && linux && cgo

// Run serially from repository root: FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41EngramProjectionVulkan$' -count=1 -timeout=5m
// The synthetic HC4 projection has 7,864,320 cells (31,457,280 host F32 bytes); it is not a full checkpoint witness.
package model

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41EngProjVulkanFixture(t *testing.T) *Model {
	t.Helper()
	m := v41GroupedVulkanFixture(t)
	cfg := m.Cfg.DeepSeekV41
	cfg.EngramLayerIDs = []int{0}
	cfg.EngramNumEmbeddings = []int{48}
	cfg.EngramMaxNgramSize, cfg.EngramNHeads, cfg.EngramHeadDim = 4, 8, 256
	H := m.Cfg.HiddenSize
	man, raw := synthBuildRaw([]synthTensor{
		{layerName(0, "engram_q_norm.weight"), []int{4 * H}},
		{layerName(0, "engram_k_norm.weight"), []int{4 * H}},
	}, func(_ string, _ func() float32) float32 { return 1 })
	for name, meta := range man {
		meta.Offset += len(m.raw)
		m.manifest[name] = meta
	}
	m.raw = append(m.raw, raw...)
	m.kqw[layerName(0, "engram_kv.weight")] = q2kFixtureTensor(5*H, 24*256, 136680808)
	layout := v41EngramTestLayout(m.Cfg)
	src := &v41EngramMemorySource{packed: v41EngramTestPackedRows(48), rows: 48}
	if err := m.wireV41Engram(layout, []V41EngramRowSource{src}, int64(V41EngramPackedRowBytes)*4); err != nil {
		t.Fatal(err)
	}
	return m
}

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=60s lane=optin
func TestV41EngramProjectionVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical Engram witness requires explicit device opt-in")
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("physical witness requires dispatch profile before process start")
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
		t.Fatalf("physical identity unavailable: available=%t error=%v", available, err)
	}
	device := strings.ToLower(identity.Identity.Device)
	if !strings.Contains(device, "radeon") || !strings.Contains(device, "8060s") || !strings.Contains(device, "radv") {
		t.Fatalf("requires Radeon 8060S RADV, got %q", identity.Identity.Device)
	}
	if !identity.TransferCountersObserved || !compute.BackendSupportsDeviceWeightDtype(be, compute.Q2_K) {
		t.Fatal("physical transfer observations/packed Q2 support unavailable")
	}
	m, oracle := v41EngProjVulkanFixture(t), v41EngProjVulkanFixture(t)
	b := newV41EngProjBackend(m)
	b.Backend = be
	s := v41EngProjSession(t, m, b)
	host := oracle.NewSession()
	defer host.Close()
	defer func() {
		s.Close()
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	}()
	if b.out != 1280 || b.in != 6144 || m.Cfg.HiddenSize != 256 || m.Cfg.OGroups != 8 || m.Cfg.NumHeads != 64 {
		t.Fatal("bounded physical payload geometry changed")
	}
	cache := map[compute.Buffer]bool{}
	for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
		phase := "prefill"
		if index == 1 {
			phase = "decode"
		}
		engBefore, hostBefore := v41EngProjPhase(t, m, phase), v41EngProjPhase(t, oracle, phase)
		denseBefore, groupBefore := v41DenseTestPhase(t, m, phase), v41GroupedPhase(t, m, phase)
		opening, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available || opening.Identity != identity.Identity {
			t.Fatal("physical opening identity changed")
		}
		profileBefore := profile.VulkanDebugDispatchProfileSnapshot()
		from, engFrom, attempts := len(b.ops), len(b.operations), b.attempts
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
		eng := v41DenseTestDelta(v41EngProjPhase(t, m, phase), engBefore)
		control := v41DenseTestDelta(v41EngProjPhase(t, oracle, phase), hostBefore)
		rows, upload, read := 0, 0, 0
		for _, op := range b.operations[engFrom:] {
			rows += op.rows
			upload += op.upload
			read += op.read
		}
		if rows != len(ids) || eng["engram_projection_device_rows"] != float64(len(ids)) || eng["engram_projection_device_calls"] <= 0 || eng["engram_projection_host_calls"] != 0 || eng["engram_projection_host_rows"] != 0 || eng["engram_projection_host_weight_f32_bytes"] != 0 {
			t.Errorf("physical Engram route=%v recorded_rows=%d", eng, rows)
		}
		if eng["engram_projection_matmul_calls"] != float64(b.attempts-attempts) || eng["engram_projection_activation_upload_bytes"] != float64(upload) || eng["engram_projection_readback_bytes"] != float64(read) || eng["engram_projection_nanos"] <= 0 {
			t.Errorf("physical Engram API/transfer/time=%v observed=%d/%d/%d", eng, b.attempts-attempts, upload, read)
		}
		materializations := 1
		if index == 2 {
			materializations = len(ids)
		}
		if control["engram_projection_host_rows"] != float64(len(ids)) || control["engram_projection_host_calls"] <= 0 || control["engram_projection_device_calls"] != 0 || control["engram_projection_host_weight_f32_bytes"] != float64(4*b.out*b.in*materializations) || control["engram_projection_nanos"] <= 0 {
			t.Errorf("actual whole-weight host control=%v want_f32_bytes=%d", control, 4*b.out*b.in*materializations)
		}
		dense := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), denseBefore)
		group := v41DenseTestDelta(v41GroupedPhase(t, m, phase), groupBefore)
		if dense["dense_projection_device_rows"] != float64(7*len(ids)) || dense["dense_projection_host_rows"] != 0 || group["grouped_output_device_rows"] != float64(len(ids)) || group["grouped_output_host_rows"] != 0 {
			t.Errorf("default dense+grouped+Engram composition inactive dense=%v grouped=%v", dense, group)
		}
		groupCalls, groupRows, groupUpload, groupRead := v41GroupedVulkanOps(s, b.v41DenseTestBackend, from)
		if groupRows != len(ids) || group["grouped_output_matmul_calls"] != float64(groupCalls) || group["grouped_output_activation_upload_bytes"] != float64(groupUpload) || group["grouped_output_readback_bytes"] != float64(groupRead) {
			t.Error("physical grouped ledger differs from actual API records")
		}
		named, _, _ := v41DenseTestOps(s, b.v41DenseTestBackend, from)
		denseCalls := 0
		for _, ops := range named {
			denseCalls += len(ops)
		}
		routed := 2 * len(ids) * m.Cfg.NumExpertsPerTok
		q2 := profile.VulkanDebugDispatchProfileSnapshot().Q2KMatmulDispatches - profileBefore.Q2KMatmulDispatches
		if q2 != uint64(denseCalls+groupCalls+routed+b.attempts-attempts) {
			t.Errorf("actual Q2 dispatches=%d want dense%d+grouped%d+routed%d+Engram%d", q2, denseCalls, groupCalls, routed, b.attempts-attempts)
		}
		if observation.Counters.H2DBytes < uint64(upload+groupUpload) || observation.Counters.D2HBytes < uint64(read+groupRead) || observation.Counters.ComputeDispatches == 0 {
			t.Error("physical transfer profile does not cover observed activations/results")
		}
		if b.stages != 1 {
			t.Errorf("immutable Engram weight staged %d times want once", b.stages)
		}
		for weight := range b.weightBuffers {
			if index > 0 && !cache[weight] {
				t.Error("continuation changed Engram cached weight allocation")
			}
			cache[weight] = true
		}
		for key, weight := range s.halW {
			if strings.Contains(key, "engram_kv.weight") && weight.Dtype != compute.Q2_K {
				t.Error("Engram device staging expanded packed weight")
			}
		}
		if len(got) == 0 || len(got) != len(want) {
			t.Fatal("physical logits width invalid")
		}
		var dot, a2, c2, maxDiff, maxAbs float64
		greedyGot, greedyWant := 0, 0
		for i, value := range got {
			a, c := float64(value), float64(want[i])
			if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(c) || math.IsInf(c, 0) {
				t.Fatal("physical logits nonfinite")
			}
			dot += a * c
			a2 += a * a
			c2 += c * c
			maxDiff = max(maxDiff, math.Abs(a-c))
			maxAbs = max(maxAbs, math.Abs(c))
			if value > got[greedyGot] {
				greedyGot = i
			}
			if want[i] > want[greedyWant] {
				greedyWant = i
			}
		}
		if a2 == 0 || c2 == 0 {
			t.Fatal("physical numeric control vacuous")
		}
		cosine := dot / math.Sqrt(a2*c2)
		if cosine < .999 || maxDiff > .01*max(1, maxAbs) || greedyGot != greedyWant {
			t.Errorf("physical parity cosine=%g max_diff=%g scale=%g greedy=%d/%d", cosine, maxDiff, maxAbs, greedyGot, greedyWant)
		}
		engJSON, _ := json.Marshal(eng)
		controlJSON, _ := json.Marshal(control)
		t.Logf("backend=%s device=%q driver=%q runtime=%q phase=%s cold=%t tokens=%d synthetic_H256_HC4_KV_out1280_in6144 eng_api_calls=%d eng_rows=%d dense_calls=%d grouped_calls=%d routed_q2_calls=%d q2_dispatches=%d h2d_bytes=%d d2h_bytes=%d h2d_count=%d d2h_count=%d cosine=%g max_diff=%g greedy=%d/%d default_engram_json=%s actual_host_control_json=%s", observation.Identity.Backend, observation.Identity.Device, observation.Identity.Driver, observation.Identity.Runtime, phase, index == 0, len(ids), b.attempts-attempts, rows, denseCalls, groupCalls, routed, q2, observation.Counters.H2DBytes, observation.Counters.D2HBytes, observation.Counters.H2DCount, observation.Counters.D2HCount, cosine, maxDiff, greedyGot, greedyWant, engJSON, controlJSON)
	}
}
