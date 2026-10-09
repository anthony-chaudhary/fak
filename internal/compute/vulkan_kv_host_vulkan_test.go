//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"context"
	"errors"
	"math"
	"os"
	"os/exec"
	"reflect"
	"strings"
	"testing"
	"time"
)

const vulkanKVHostRestoreFaultChildEnv = "FAK_TEST_VULKAN_KV_HOST_RESTORE_FAULT_CHILD"

// fak-test:runtime slow est=5s lane=optin
func TestVulkanKVHostSnapshotContract(t *testing.T) {
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required-device witness needs FAK_VULKAN_DISPATCH_PROFILE=1 before process startup")
		}
		t.Skip("requires FAK_VULKAN_DISPATCH_PROFILE=1 to witness native submits")
	}
	if os.Getenv(vulkanKVHostRestoreFaultChildEnv) == "1" {
		vulkanKVHostRestoreFailureContract(t)
		return
	}
	v := vk(t)
	v.Trim()

	t.Run("exact round trip remains independently usable", func(t *testing.T) {
		cfg := KVConfig{
			NumLayers:      2,
			NumKVHeads:     1,
			HeadDim:        4,
			RopeTheta:      10000,
			Precision:      KVPrecisionF32,
			WindowPerLayer: []int{0, 3},
		}
		source := v.NewKV(cfg)
		defer source.Free()
		positions := []int{7, 8, 9}
		for i, pos := range positions {
			vulkanKVHostAppend(t, v, source, cfg, pos, 100*i)
		}
		want := vulkanKVHostExpected(cfg, positions, []int{0, 100, 200})

		snapshotWindow, err := v.BeginBackendExecutionWindow()
		if err != nil {
			t.Fatalf("begin snapshot observation: %v", err)
		}
		host, err := SnapshotKVToHost(source)
		if err != nil {
			_, _ = snapshotWindow.End()
			t.Fatalf("snapshot Vulkan KV: %v", err)
		}
		snapshotObservation, err := snapshotWindow.End()
		if err != nil {
			t.Fatalf("end snapshot observation: %v", err)
		}
		if got := snapshotObservation.Counters; got.D2HBytes != uint64(host.TransferBytes()) || got.D2HCount == 0 || got.DispatchSubmits == 0 {
			t.Fatalf("snapshot device evidence: d2h_bytes=%d want=%d d2h_count=%d submits=%d", got.D2HBytes, host.TransferBytes(), got.D2HCount, got.DispatchSubmits)
		}
		vulkanKVHostEqual(t, host, want)

		beforeRestore, err := v.BackendExecutionSnapshot()
		if err != nil {
			t.Fatalf("restore allocation baseline: %v", err)
		}
		restoreWindow, err := v.BeginBackendExecutionWindow()
		if err != nil {
			t.Fatalf("begin restore observation: %v", err)
		}
		restored, err := RestoreKVFromHost(v, host)
		if err != nil {
			_, _ = restoreWindow.End()
			t.Fatalf("restore Vulkan KV: %v", err)
		}
		defer restored.Free()
		restoreObservation, err := restoreWindow.End()
		if err != nil {
			t.Fatalf("end restore observation: %v", err)
		}
		if got := restoreObservation.Counters; got.H2DBytes != uint64(host.TransferBytes()) || got.H2DCount == 0 || got.DispatchSubmits == 0 {
			t.Fatalf("restore device evidence: h2d_bytes=%d want=%d h2d_count=%d submits=%d", got.H2DBytes, host.TransferBytes(), got.H2DCount, got.DispatchSubmits)
		}
		if !restoreObservation.DeviceAllocationObserved || restoreObservation.DeviceAllocationPeakBytes <= beforeRestore.DeviceAllocationLiveBytes {
			t.Fatalf("restore allocation evidence: observed=%t opening_live=%d peak=%d", restoreObservation.DeviceAllocationObserved, beforeRestore.DeviceAllocationLiveBytes, restoreObservation.DeviceAllocationPeakBytes)
		}

		host.Pos[0] = -1
		host.K[0][0] = math.Float32frombits(0x3f000001)
		host.KRaw[0][0] = math.Float32frombits(0xbf000001)
		host.V[0][0] = math.Float32frombits(0x3e800001)
		gotRestored, err := SnapshotKVToHost(restored)
		if err != nil {
			t.Fatalf("snapshot restored Vulkan KV: %v", err)
		}
		vulkanKVHostEqual(t, gotRestored, want)

		vulkanKVHostAppend(t, v, restored, cfg, 10, 900)
		gotSource, err := SnapshotKVToHost(source)
		if err != nil {
			t.Fatalf("snapshot source after restored mutation: %v", err)
		}
		vulkanKVHostEqual(t, gotSource, want)

		vulkanKVHostAppend(t, v, source, cfg, 10, 900)
		if removed := source.Evict(1, 1); removed != 1 {
			t.Fatalf("source evict removed %d, want 1", removed)
		}
		if removed := restored.Evict(1, 1); removed != 1 {
			t.Fatalf("restored evict removed %d, want 1", removed)
		}
		continuedSource, err := SnapshotKVToHost(source)
		if err != nil {
			t.Fatalf("snapshot continued source: %v", err)
		}
		continuedRestored, err := SnapshotKVToHost(restored)
		if err != nil {
			t.Fatalf("snapshot continued restore: %v", err)
		}
		vulkanKVHostEqual(t, continuedRestored, continuedSource)
	})

	t.Run("hybrid empty attention layers include layer zero", func(t *testing.T) {
		cfg := KVConfig{NumLayers: 3, NumKVHeads: 1, HeadDim: 2, RopeTheta: 10000, Precision: KVPrecisionF32}
		state := KVHostSnapshot{
			Config: cfg,
			Pos:    []int{4, 5},
			K:      [][]float32{nil, {1, -2, 3, -4}, nil},
			KRaw:   [][]float32{nil, {5, -6, 7, -8}, nil},
			V:      [][]float32{nil, {9, -10, 11, -12}, nil},
		}
		window, err := v.BeginBackendExecutionWindow()
		if err != nil {
			t.Fatalf("begin hybrid restore observation: %v", err)
		}
		restored, err := RestoreKVFromHost(v, state)
		if err != nil {
			_, _ = window.End()
			t.Fatalf("restore hybrid Vulkan KV: %v", err)
		}
		defer restored.Free()
		observation, err := window.End()
		if err != nil {
			t.Fatalf("end hybrid restore observation: %v", err)
		}
		if got := observation.Counters; got.H2DBytes != uint64(state.TransferBytes()) || got.H2DCount == 0 || got.DispatchSubmits == 0 {
			t.Fatalf("hybrid restore device evidence: h2d_bytes=%d want=%d h2d_count=%d submits=%d", got.H2DBytes, state.TransferBytes(), got.H2DCount, got.DispatchSubmits)
		}
		roundTrip, err := SnapshotKVToHost(restored)
		if err != nil {
			t.Fatalf("snapshot restored hybrid Vulkan KV: %v", err)
		}
		vulkanKVHostEqual(t, roundTrip, state)
	})

	t.Run("snapshot staging failure is typed and leaves source intact", func(t *testing.T) {
		cfg := KVConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2, RopeTheta: 10000, Precision: KVPrecisionF32}
		source := v.NewKV(cfg)
		defer source.Free()
		vulkanKVHostAppend(t, v, source, cfg, 3, 400)
		want := vulkanKVHostExpected(cfg, []int{3}, []int{400})

		baseline, err := SnapshotKVToHost(source)
		if err != nil {
			t.Fatalf("snapshot fault baseline: %v", err)
		}
		vulkanKVHostEqual(t, baseline, want)

		v.VulkanDebugSetD2HStagingFailureOnce(true)
		defer v.VulkanDebugSetD2HStagingFailureOnce(false)
		failed, err := SnapshotKVToHost(source)
		if !errors.Is(err, ErrVulkanAllocationFailed) {
			t.Fatalf("snapshot staging error type=%T, want ErrVulkanAllocationFailed", err)
		}
		if failed.ResidentBytes() != 0 || len(failed.Pos) != 0 || len(failed.K) != 0 || len(failed.KRaw) != 0 || len(failed.V) != 0 {
			t.Fatalf("failed snapshot published partial state: resident=%d pos=%d K/KRaw/V=%d/%d/%d", failed.ResidentBytes(), len(failed.Pos), len(failed.K), len(failed.KRaw), len(failed.V))
		}

		recovered, err := SnapshotKVToHost(source)
		if err != nil {
			t.Fatalf("snapshot after one-shot staging failure: %v", err)
		}
		vulkanKVHostEqual(t, recovered, want)
	})

	t.Run("invalid and packed states refuse before allocation", func(t *testing.T) {
		invalid := KVHostSnapshot{
			Config: KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 2, Precision: KVPrecisionF32},
			Pos:    []int{0},
			K:      [][]float32{{1}},
			KRaw:   [][]float32{{2, 3}},
			V:      [][]float32{{4, 5}},
		}
		window, err := v.BeginBackendExecutionWindow()
		if err != nil {
			t.Fatalf("begin invalid restore observation: %v", err)
		}
		if restored, err := RestoreKVFromHost(v, invalid); err == nil {
			if restored != nil {
				restored.Free()
			}
			_, _ = window.End()
			t.Fatal("invalid snapshot geometry restored")
		}
		observation, err := window.End()
		if err != nil {
			t.Fatalf("end invalid restore observation: %v", err)
		}
		if got := observation.Counters; got.H2DCount != 0 || got.D2HCount != 0 || got.DispatchSubmits != 0 || observation.DeviceAllocationPeakBytes != observation.DeviceAllocationLiveBytes {
			t.Fatalf("invalid restore touched device: h2d=%d d2h=%d submits=%d live=%d peak=%d", got.H2DCount, got.D2HCount, got.DispatchSubmits, observation.DeviceAllocationLiveBytes, observation.DeviceAllocationPeakBytes)
		}

		packedCfg := KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 2, Precision: KVPrecisionQ8}
		packed := v.NewKV(packedCfg)
		defer packed.Free()
		if _, err := SnapshotKVToHost(packed); !errors.Is(err, ErrKVHostSnapshotUnsupported) {
			t.Fatalf("packed snapshot error matches unsupported sentinel = %t", errors.Is(err, ErrKVHostSnapshotUnsupported))
		}
		packedState := KVHostSnapshot{Config: packedCfg, K: [][]float32{nil}, KRaw: [][]float32{nil}, V: [][]float32{nil}}
		if restored, err := RestoreKVFromHost(v, packedState); !errors.Is(err, ErrKVHostSnapshotUnsupported) {
			if restored != nil {
				restored.Free()
			}
			t.Fatalf("packed restore error matches unsupported sentinel = %t", errors.Is(err, ErrKVHostSnapshotUnsupported))
		}
	})

	t.Run("pre-submit restore failure publishes nothing and poisons observation", func(t *testing.T) {
		exe, err := os.Executable()
		if err != nil {
			t.Fatalf("resolve restore-fault test executable: %v", err)
		}
		ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, exe, "-test.v", "-test.run=^TestVulkanKVHostSnapshotContract$")
		cmd.Env = append(os.Environ(), vulkanKVHostRestoreFaultChildEnv+"=1")
		out, err := cmd.CombinedOutput()
		if ctx.Err() != nil {
			t.Fatalf("restore-fault subprocess timed out: %v", ctx.Err())
		}
		if err != nil {
			t.Fatalf("restore-fault subprocess failed: %v\n%s", err, out)
		}
		if !strings.Contains(string(out), "--- PASS: TestVulkanKVHostSnapshotContract") {
			t.Fatalf("restore-fault subprocess did not report a passing contract test:\n%s", out)
		}
	})
}

func vulkanKVHostRestoreFailureContract(t *testing.T) {
	v := vk(t)
	cfg := KVConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2, RopeTheta: 10000, Precision: KVPrecisionF32}
	state := vulkanKVHostExpected(cfg, []int{2, 3}, []int{500, 600})
	before, available, err := CaptureBackendExecutionSnapshot(v)
	if err != nil || !available || !before.TransferCountersObserved || !before.DeviceAllocationObserved {
		t.Fatalf("restore failure observation baseline: snapshot=%+v available=%t err=%v", before, available, err)
	}

	// This seam fails before queue submission, so abort can retire the fresh
	// allocations safely. An uncertain submit/wait must retain its resources.
	v.VulkanDebugSetRestoreFailureAfterSubmits(0)
	defer v.VulkanDebugSetRestoreFailureAfterSubmits(-1)
	restored, restoreErr := RestoreKVFromHost(v, state)
	if !errors.Is(restoreErr, ErrVulkanDeviceLost) {
		if restored != nil {
			restored.Free()
		}
		t.Fatalf("restore submission error type=%T, want ErrVulkanDeviceLost", restoreErr)
	}
	var backendErr *BackendError
	if !errors.As(restoreErr, &backendErr) || backendErr.Class != VulkanClassDeviceLost {
		t.Fatalf("restore submission backend error type=%T class=%v, want *BackendError class=%s", restoreErr, backendErrClass(backendErr), VulkanClassDeviceLost)
	}
	if restored != nil {
		restored.Free()
		t.Fatal("failed restore published a KV store")
	}
	if v.VulkanDebugRestoreActive() {
		t.Fatal("failed restore left the native restore transaction active")
	}
	// Sticky device loss refuses physical observations even after a safe abort.
	// An unavailable snapshot is not evidence of zero retained device bytes.
	after, available, err := CaptureBackendExecutionSnapshot(v)
	if err == nil || !strings.Contains(err.Error(), "Vulkan transfer counters are unavailable") || available || after != (BackendExecutionSnapshot{}) {
		t.Fatalf("poisoned restore observation: snapshot=%+v available=%t err=%v, want unavailable with no partial snapshot", after, available, err)
	}
}

func backendErrClass(err *BackendError) VulkanErrorClass {
	if err == nil {
		return ""
	}
	return err.Class
}

func vulkanKVHostAppend(t *testing.T, v *vulkanBackend, kv KVStore, cfg KVConfig, pos, seed int) {
	t.Helper()
	c := cpu()
	width := cfg.NumKVHeads * cfg.HeadDim
	for layer := 0; layer < cfg.NumLayers; layer++ {
		raw := v.Upload(NewF32(c, []int{width}, vulkanKVHostRow(width, layer, seed, 1)), F32)
		rope := v.Upload(NewF32(c, []int{width}, vulkanKVHostRow(width, layer, seed, 2)), F32)
		value := v.Upload(NewF32(c, []int{width}, vulkanKVHostRow(width, layer, seed, 3)), F32)
		kv.AppendKV(layer, raw, rope, value, pos)
		v.Free(raw)
		v.Free(rope)
		v.Free(value)
	}
}

func vulkanKVHostExpected(cfg KVConfig, positions, seeds []int) KVHostSnapshot {
	out := KVHostSnapshot{
		Config: cloneKVConfig(cfg),
		Pos:    append([]int(nil), positions...),
		K:      make([][]float32, cfg.NumLayers),
		KRaw:   make([][]float32, cfg.NumLayers),
		V:      make([][]float32, cfg.NumLayers),
	}
	width := cfg.NumKVHeads * cfg.HeadDim
	for layer := 0; layer < cfg.NumLayers; layer++ {
		for _, seed := range seeds {
			out.KRaw[layer] = append(out.KRaw[layer], vulkanKVHostRow(width, layer, seed, 1)...)
			out.K[layer] = append(out.K[layer], vulkanKVHostRow(width, layer, seed, 2)...)
			out.V[layer] = append(out.V[layer], vulkanKVHostRow(width, layer, seed, 3)...)
		}
	}
	return out
}

func vulkanKVHostRow(width, layer, seed, offset int) []float32 {
	out := make([]float32, width)
	for i := range out {
		bits := uint32(0x3e000001 + seed + 97*layer + 11*offset + i)
		if (layer+offset+i)&1 != 0 {
			bits |= 1 << 31
		}
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func vulkanKVHostEqual(t *testing.T, got, want KVHostSnapshot) {
	t.Helper()
	if !reflect.DeepEqual(got.Config, want.Config) {
		t.Fatal("snapshot config differs")
	}
	if !reflect.DeepEqual(got.Pos, want.Pos) {
		t.Fatalf("snapshot positions differ: got=%v want=%v", got.Pos, want.Pos)
	}
	for _, plane := range []struct {
		name      string
		got, want [][]float32
	}{
		{name: "K", got: got.K, want: want.K},
		{name: "KRaw", got: got.KRaw, want: want.KRaw},
		{name: "V", got: got.V, want: want.V},
	} {
		if len(plane.got) != len(plane.want) {
			t.Fatalf("snapshot %s layer count=%d, want %d", plane.name, len(plane.got), len(plane.want))
		}
		for layer := range plane.want {
			if len(plane.got[layer]) != len(plane.want[layer]) {
				t.Fatalf("snapshot %s layer %d length=%d, want %d", plane.name, layer, len(plane.got[layer]), len(plane.want[layer]))
			}
			for i := range plane.want[layer] {
				if math.Float32bits(plane.got[layer][i]) != math.Float32bits(plane.want[layer][i]) {
					t.Fatalf("snapshot %s layer %d element %d bits=%08x, want %08x", plane.name, layer, i, math.Float32bits(plane.got[layer][i]), math.Float32bits(plane.want[layer][i]))
				}
			}
		}
	}
}
