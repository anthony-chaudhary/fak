//go:build vulkan && (windows || linux) && cgo

// Run with: FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41DenseProjectionHalo$' -count=1 -timeout=5m
// The Vulkan backend is process-shared; run this physical witness serially.

package model

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41DenseHaloQ2Fixture(t *testing.T) *Model {
	t.Helper()
	m := v41IncrementalExpertFixture(t, false, false)
	// Widen only the query latent axis so all seven reductions contain Q2_K blocks.
	m.Cfg.QLoraRank = 256
	for index, leaf := range v41DenseTestLeaves {
		shape := v41ProjectionShape(t, m, leaf)
		name := layerName(0, leaf)
		m.kqw[name] = q2kFixtureTensor(shape[0], shape[1], uint64(0x61380+index))
		delete(m.manifest, name)
	}
	return m
}

// fak-test:runtime slow est=30s lane=optin
func TestV41DenseProjectionHalo(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical dense projection witness requires explicit opt-in")
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("required physical dispatch profile must be enabled before process start")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	profile, ok := be.(interface {
		VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	})
	if !ok {
		t.Fatal("actual Vulkan backend lacks dispatch profile witness")
	}
	identity, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available {
		t.Fatalf("physical identity unavailable: available=%t error=%v", available, err)
	}
	device := strings.ToLower(identity.Identity.Device)
	if !strings.Contains(device, "radeon") || !strings.Contains(device, "8060s") || !strings.Contains(device, "radv") {
		t.Fatalf("physical witness requires Radeon8060S RADV, got %q", identity.Identity.Device)
	}
	if !identity.TransferCountersObserved || !compute.BackendSupportsDeviceWeightDtype(be, compute.Q2_K) {
		t.Fatal("actual backend lacks transfer observations or packed Q2_K promise")
	}
	m, oracle := v41DenseHaloQ2Fixture(t), v41DenseHaloQ2Fixture(t)
	if len(m.Cfg.DeepSeekV41.EngramLayerIDs) != 0 {
		t.Fatal("reduced dense witness unexpectedly enables Engram")
	}
	b := newV41DenseTestBackend()
	b.Backend = be
	s := v41DenseTestSession(t, m, b)
	s.v41State().groupedOutput = nil
	history := []int{1, 2, 3}
	for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
		phase := "prefill"
		if index == 1 {
			phase = "decode"
		}
		opening, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available || opening.Identity != identity.Identity {
			t.Fatalf("physical opening identity invalid: available=%t error=%v", available, err)
		}
		profileBefore := profile.VulkanDebugDispatchProfileSnapshot()
		phaseBefore := v41DenseTestPhase(t, m, phase)
		from := len(b.ops)
		var got []float32
		if index == 1 {
			got = s.Step(ids[0])
		} else {
			got = s.Prefill(ids)
		}
		if index > 0 {
			history = append(history, ids...)
		}
		closing, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available {
			t.Fatalf("physical closing identity unavailable: available=%t error=%v", available, err)
		}
		observation, err := compute.BackendExecutionDelta(opening, closing)
		if err != nil {
			t.Fatal(err)
		}
		profileAfter := profile.VulkanDebugDispatchProfileSnapshot()
		named, upload, read := v41DenseTestOps(s, b, from)
		calls, rows := 0, 0
		for _, leaf := range v41DenseTestLeaves {
			leafRows := 0
			for _, op := range named[leaf] {
				calls++
				leafRows += op.rows
			}
			if leafRows != len(ids) {
				t.Errorf("physical leaf %s device rows=%d want %d", leaf, leafRows, len(ids))
			}
			rows += leafRows
			if index == 0 && (leaf == "attn.wq_a.weight" || leaf == "attn.wq_b.weight") {
				if len(named[leaf]) != 1 || !named[leaf][0].batch {
					t.Errorf("physical cold query %s did not invoke bounded device panel", leaf)
				}
			}
			name := layerName(0, leaf)
			if m.has(name) || s.halW["kquant-raw:"+name].Dtype != compute.Q2_K {
				t.Errorf("physical projection %s did not preserve compressed Q2_K staging", leaf)
			}
		}
		delta := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), phaseBefore)
		if delta["dense_projection_device_calls"] != float64(calls) || delta["dense_projection_device_rows"] != float64(rows) || delta["dense_projection_host_calls"] != 0 || delta["dense_projection_host_rows"] != 0 {
			t.Errorf("physical dense attempt/row ledger=%v calls=%d rows=%d", delta, calls, rows)
		}
		if delta["dense_projection_activation_upload_bytes"] != float64(upload) || delta["dense_projection_readback_bytes"] != float64(read) || delta["dense_projection_nanos"] <= 0 {
			t.Errorf("physical dense transfer/time ledger=%v recorded upload=%d read=%d", delta, upload, read)
		}
		routedQ2Calls := 2 * len(ids) * m.Cfg.NumLayers * m.Cfg.NumExpertsPerTok
		if q2 := profileAfter.Q2KMatmulDispatches - profileBefore.Q2KMatmulDispatches; q2 != uint64(calls+routedQ2Calls) {
			t.Errorf("physical Q2_K dispatches=%d want dense%d+routed%d", q2, calls, routedQ2Calls)
		}
		if observation.Counters.D2HBytes < uint64(read) || observation.Counters.H2DBytes < uint64(upload) || observation.Counters.ComputeDispatches == 0 {
			t.Error("physical transfer/dispatch counters do not cover actual dense operations")
		}
		want := lastLogits(oracle.Forward(history))
		if len(got) == 0 || len(got) != len(want) {
			t.Fatal("physical dense logits width invalid")
		}
		var dot, a2, b2, maximumDiff, maximumAbs float64
		for i, value := range got {
			a, c := float64(value), float64(want[i])
			if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(c) || math.IsInf(c, 0) {
				t.Fatal("physical dense logits must be finite")
			}
			dot += a * c
			a2 += a * a
			b2 += c * c
			maximumDiff = max(maximumDiff, math.Abs(a-c))
			maximumAbs = max(maximumAbs, math.Abs(a))
		}
		if a2 == 0 || b2 == 0 {
			t.Fatal("physical dense cosine oracle vacuous")
		}
		cosine := dot / math.Sqrt(a2*b2)
		if cosine < 0.999 {
			t.Errorf("physical dense logits cosine=%g want >=0.999", cosine)
		}
		raw, err := json.Marshal(m.V41ExpertFaultAttribution())
		if err != nil {
			t.Fatal(err)
		}
		var phases map[string]json.RawMessage
		if err := json.Unmarshal(raw, &phases); err != nil {
			t.Fatal(err)
		}
		t.Logf("backend=%s device=%q driver=%q runtime=%q phase=%s cold=%t tokens=%d dense_calls=%d dense_rows=%d q2_dispatches=%d routed_q2_calls=%d compute_dispatches=%d dispatch_submits=%d h2d_bytes=%d d2h_bytes=%d h2d_count=%d d2h_count=%d cosine=%g maximum_diff=%g maximum_abs=%g default_phase_json=%s", observation.Identity.Backend, observation.Identity.Device, observation.Identity.Driver, observation.Identity.Runtime, phase, index == 0, len(ids), calls, rows, profileAfter.Q2KMatmulDispatches-profileBefore.Q2KMatmulDispatches, routedQ2Calls, observation.Counters.ComputeDispatches, observation.Counters.DispatchSubmits, observation.Counters.H2DBytes, observation.Counters.D2HBytes, observation.Counters.H2DCount, observation.Counters.D2HCount, cosine, maximumDiff, maximumAbs, phases[phase])
	}
}
