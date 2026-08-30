package nativeperfbackend

import (
	"strings"
	"testing"
)

func ptr[T any](v T) *T { return &v }

func populatedSnapshot() Snapshot {
	return Snapshot{
		Backend:            BackendMetal,
		Engine:             Engine,
		ModelFamily:        "Qwen3.8",
		Available:          true,
		UnavailableReason:  ReasonNone,
		DeviceUtilization:  ptr(0.81),
		MemoryBytes:        map[MemoryKind]float64{MemoryAllocated: 12, MemoryResident: 10},
		MemoryPressure:     ptr(0.73),
		DelaySeconds:       map[DelayKind]float64{DelayQueue: 0.003, DelayStream: 0.002, DelayCommandBuffer: 0.001},
		TransferBytesTotal: map[Direction]float64{DirectionUpload: 1024, DirectionDownload: 512},
		KernelCallsTotal: map[KernelFamily]float64{
			KernelMatmul:        20,
			KernelAttention:     8,
			KernelNormalization: 6,
			KernelEmbedding:     4,
			KernelSampling:      3,
			KernelTransfer:      2,
			KernelOther:         1,
		},
		SyncEventsTotal: map[SyncKind]float64{SyncFence: 2},
		GraphState:      ptr(GraphReady),
	}
}

func TestPopulatedSnapshotValidates(t *testing.T) {
	if err := Validate(populatedSnapshot()); err != nil {
		t.Fatal(err)
	}
}

func TestDimensionsRemainBounded(t *testing.T) {
	s := populatedSnapshot()
	s.KernelCallsTotal[KernelFamily("prompt-sha-123")] = 1
	err := Validate(s)
	if err == nil || !strings.Contains(err.Error(), "unbounded dimension") {
		t.Fatalf("Validate error = %v, want bounded-dimension refusal", err)
	}

	for _, metric := range Metrics() {
		for _, label := range metric.Labels {
			switch label {
			case "backend", "engine", "model_family", "schema", "reason", "kind", "direction", "family", "state":
			default:
				t.Fatalf("metric %s exposes unbounded label %q", metric.Name, label)
			}
		}
	}
}

func TestUnavailableStateRequiresReasonAndNoMeasurements(t *testing.T) {
	s := Snapshot{Backend: BackendCUDA, Engine: Engine, ModelFamily: "Qwen3.8"}
	if err := Validate(s); err == nil || !strings.Contains(err.Error(), "explicit bounded reason") {
		t.Fatalf("missing reason error = %v", err)
	}

	s.UnavailableReason = ReasonDeviceNotFound
	if err := Validate(s); err != nil {
		t.Fatalf("honest unavailable state rejected: %v", err)
	}

	s.DeviceUtilization = ptr(0.0)
	if err := Validate(s); err == nil || !strings.Contains(err.Error(), "cannot publish measurement zeros") {
		t.Fatalf("unavailable measurement error = %v", err)
	}
}

func TestRejectsFallbackEngineAndArbitraryModelLabel(t *testing.T) {
	s := populatedSnapshot()
	s.Engine = "llama.cpp"
	s.ModelFamily = "Qwen3.8-4B-private-checkpoint-path"
	err := Validate(s)
	if err == nil || !strings.Contains(err.Error(), `engine must be "fak-native"`) || !strings.Contains(err.Error(), "model_family") {
		t.Fatalf("Validate error = %v", err)
	}
}
