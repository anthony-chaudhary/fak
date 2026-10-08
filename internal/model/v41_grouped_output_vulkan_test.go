//go:build vulkan && linux && cgo

// Run serially from repository root: FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41GroupedOutputVulkan$' -count=1 -timeout=5m
// This bounded synthetic payload witnesses actual Radeon 8060S execution, not a full checkpoint.

package model

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41GroupedVulkanFixture(t *testing.T) *Model {
	t.Helper()
	m := v41DenseHaloQ2Fixture(t)
	m.Cfg.NumHeads, m.Cfg.OGroups, m.Cfg.OLoraRank = 64, 8, 32
	c := m.Cfg
	for index, leaf := range []string{"attn.wq_b.weight", "attn.wo_a.weight", "attn.wo_b.weight"} {
		out, in := c.NumHeads*c.HeadDim, c.QLoraRank
		if leaf == "attn.wo_a.weight" {
			out, in = c.OGroups*c.OLoraRank, c.NumHeads/c.OGroups*c.HeadDim
		}
		if leaf == "attn.wo_b.weight" {
			out, in = c.HiddenSize, c.OGroups*c.OLoraRank
		}
		name := layerName(0, leaf)
		m.kqw[name] = q2kFixtureTensor(out, in, uint64(136680+index))
		delete(m.manifest, name)
	}
	manifest, raw := synthBuildRaw([]synthTensor{{layerName(0, "attn.sink"), []int{c.NumHeads}}}, synthMatmulFill)
	for name, meta := range manifest {
		meta.Offset += len(m.raw)
		m.manifest[name] = meta
	}
	m.raw = append(m.raw, raw...)
	return m
}

func v41GroupedVulkanOps(s *Session, b *v41DenseTestBackend, from int) (calls, rows, upload, read int) {
	for _, op := range b.ops[from:] {
		leaf := ""
		for key, weight := range s.halW {
			if weight.Buf() != op.weight {
				continue
			}
			if strings.Contains(key, "attn.wo_a.weight") {
				leaf = "a"
			}
			if strings.Contains(key, "attn.wo_b.weight") {
				leaf = "b"
			}
		}
		if leaf == "" {
			continue
		}
		calls++
		if leaf == "b" {
			rows += op.rows
		}
		upload += b.uploads[op.input]
		read += b.reads[op.output]
	}
	return
}

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=45s lane=optin
func TestV41GroupedOutputVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical grouped-output witness requires explicit device opt-in")
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("physical grouped witness requires dispatch profile before process start")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	profile, ok := be.(interface {
		VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	})
	if !ok {
		t.Fatal("physical Vulkan dispatch profile unavailable")
	}
	identity, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available {
		t.Fatalf("physical identity unavailable available=%t error=%v", available, err)
	}
	device := strings.ToLower(identity.Identity.Device)
	if !strings.Contains(device, "radeon") || !strings.Contains(device, "8060s") || !strings.Contains(device, "radv") {
		t.Fatalf("physical grouped witness requires Radeon 8060S RADV, got %q", identity.Identity.Device)
	}
	if !identity.TransferCountersObserved || !compute.BackendSupportsDeviceWeightDtype(be, compute.Q2_K) {
		t.Fatal("required physical transfer observations/packed Q2_K support unavailable")
	}
	m, controlModel := v41GroupedVulkanFixture(t), v41GroupedVulkanFixture(t)
	b := newV41DenseTestBackend()
	b.Backend = be
	s := v41DenseTestSession(t, m, b)
	control := controlModel.NewSession()
	defer control.Close()
	defer func() {
		s.Close()
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	}()
	if m.Cfg.HiddenSize != 256 || m.Cfg.NumHeads != 64 || m.Cfg.HeadDim != 32 || m.Cfg.OGroups != 8 || m.Cfg.OLoraRank != 32 {
		t.Fatal("physical payload geometry changed")
	}
	cache := map[string]compute.Buffer{}
	for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
		phase := "prefill"
		if index == 1 {
			phase = "decode"
		}
		before := v41GroupedPhase(t, m, phase)
		controlBefore := v41GroupedPhase(t, controlModel, phase)
		denseBefore := v41DenseTestPhase(t, m, phase)
		opening, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available || opening.Identity != identity.Identity {
			t.Fatal("physical opening identity changed or became unavailable")
		}
		profileBefore := profile.VulkanDebugDispatchProfileSnapshot()
		from := len(b.ops)
		var got, want []float32
		if index == 1 {
			got = s.Step(ids[0])
			want = control.Step(ids[0])
		} else {
			got = s.Prefill(ids)
			want = control.Prefill(ids)
		}
		closing, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available {
			t.Fatal("physical closing identity unavailable")
		}
		observation, err := compute.BackendExecutionDelta(opening, closing)
		if err != nil {
			t.Fatal(err)
		}
		profileAfter := profile.VulkanDebugDispatchProfileSnapshot()
		calls, rows, upload, read := v41GroupedVulkanOps(s, b, from)
		delta := v41DenseTestDelta(v41GroupedPhase(t, m, phase), before)
		controlDelta := v41DenseTestDelta(v41GroupedPhase(t, controlModel, phase), controlBefore)
		if rows != len(ids) || calls != 9*len(ids) {
			t.Errorf("actual grouped API calls=%d rows=%d want %d/%d", calls, rows, 9*len(ids), len(ids))
		}
		if delta["grouped_output_device_calls"] != float64(len(ids)) || delta["grouped_output_device_rows"] != float64(len(ids)) || delta["grouped_output_host_calls"] != 0 || delta["grouped_output_host_rows"] != 0 || delta["grouped_output_host_weight_f32_bytes"] != 0 {
			t.Errorf("physical grouped attribution=%v", delta)
		}
		if delta["grouped_output_matmul_calls"] != float64(calls) || delta["grouped_output_activation_upload_bytes"] != float64(upload) || delta["grouped_output_readback_bytes"] != float64(read) || delta["grouped_output_nanos"] <= 0 {
			t.Errorf("physical grouped API/transfer/time attribution=%v recorded calls=%d upload=%d read=%d", delta, calls, upload, read)
		}
		materializations := 1
		if index == 2 {
			materializations = len(ids)
		}
		if controlDelta["grouped_output_host_calls"] != float64(len(ids)) || controlDelta["grouped_output_host_rows"] != float64(len(ids)) || controlDelta["grouped_output_device_calls"] != 0 || controlDelta["grouped_output_host_weight_f32_bytes"] != float64(524288*materializations) || controlDelta["grouped_output_nanos"] <= 0 {
			t.Errorf("actual host control grouped materialization=%v want bytes=%d", controlDelta, 524288*materializations)
		}
		dense := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), denseBefore)
		if dense["dense_projection_device_rows"] != float64(7*len(ids)) || dense["dense_projection_host_rows"] != 0 {
			t.Errorf("physical dense+grouped default path not all on: dense=%v", dense)
		}
		named, _, _ := v41DenseTestOps(s, b, from)
		denseCalls := 0
		for _, ops := range named {
			denseCalls += len(ops)
		}
		routedQ2Calls := 2 * len(ids) * m.Cfg.NumExpertsPerTok
		q2Dispatches := profileAfter.Q2KMatmulDispatches - profileBefore.Q2KMatmulDispatches
		if q2Dispatches != uint64(calls+denseCalls+routedQ2Calls) {
			t.Errorf("physical Q2_K dispatches=%d want grouped%d+dense%d+routed%d", q2Dispatches, calls, denseCalls, routedQ2Calls)
		}
		if observation.Counters.H2DBytes < uint64(upload) || observation.Counters.D2HBytes < uint64(read) || observation.Counters.ComputeDispatches == 0 {
			t.Error("physical counters do not cover observed grouped operation transfers")
		}
		groupedWeights := 0
		for key, weight := range s.halW {
			if !strings.Contains(key, "attn.wo_a.weight") && !strings.Contains(key, "attn.wo_b.weight") {
				continue
			}
			groupedWeights++
			if weight.Dtype != compute.Q2_K {
				t.Error("physical grouped staging expanded Q2_K weight")
			}
			if index > 0 && cache[key] != weight.Buf() {
				t.Error("steady grouped continuation restaged an immutable weight")
			}
			cache[key] = weight.Buf()
		}
		if groupedWeights != 9 {
			t.Errorf("physical grouped immutable weights=%d want 9", groupedWeights)
		}
		if len(got) != len(want) || len(got) == 0 {
			t.Fatal("physical grouped logits width invalid")
		}
		var dot, a2, b2, maxDiff, maxAbs float64
		greedyGot, greedyWant := 0, 0
		for i, value := range got {
			a, c := float64(value), float64(want[i])
			if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(c) || math.IsInf(c, 0) {
				t.Fatal("physical grouped logits nonfinite")
			}
			dot += a * c
			a2 += a * a
			b2 += c * c
			maxDiff = max(maxDiff, math.Abs(a-c))
			maxAbs = max(maxAbs, math.Abs(c))
			if value > got[greedyGot] {
				greedyGot = i
			}
			if want[i] > want[greedyWant] {
				greedyWant = i
			}
		}
		if a2 == 0 || b2 == 0 {
			t.Fatal("physical grouped logits oracle vacuous")
		}
		cosine := dot / math.Sqrt(a2*b2)
		if cosine < 0.999 || maxDiff > 0.01*max(1, maxAbs) || greedyGot != greedyWant {
			t.Errorf("physical grouped parity cosine=%g maximum_diff=%g scale=%g greedy=%d/%d", cosine, maxDiff, maxAbs, greedyGot, greedyWant)
		}
		raw, err := json.Marshal(delta)
		if err != nil {
			t.Fatal(err)
		}
		controlRaw, err := json.Marshal(controlDelta)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("backend=%s device=%q driver=%q runtime=%q phase=%s cold=%t tokens=%d synthetic_geometry=G8-head64-D32-R32-H256 grouped_api_calls=%d grouped_rows=%d q2_dispatches=%d dense_calls=%d routed_q2_calls=%d compute_dispatches=%d dispatch_submits=%d h2d_bytes=%d d2h_bytes=%d h2d_count=%d d2h_count=%d cosine=%g maximum_diff=%g greedy=%d/%d default_grouped_json=%s actual_host_control_json=%s", observation.Identity.Backend, observation.Identity.Device, observation.Identity.Driver, observation.Identity.Runtime, phase, index == 0, len(ids), calls, rows, q2Dispatches, denseCalls, routedQ2Calls, observation.Counters.ComputeDispatches, observation.Counters.DispatchSubmits, observation.Counters.H2DBytes, observation.Counters.D2HBytes, observation.Counters.H2DCount, observation.Counters.D2HCount, cosine, maxDiff, greedyGot, greedyWant, raw, controlRaw)
	}
}
