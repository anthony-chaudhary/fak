package modelperfobs

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestCollectNVIDIADeviceSnapshotPreservesCounterMeaning(t *testing.T) {
	old := runNvidiaSMI
	t.Cleanup(func() { runNvidiaSMI = old })
	runNvidiaSMI = func(_ context.Context, args ...string) ([]byte, error) {
		joined := strings.Join(args, " ")
		if !strings.Contains(joined, "utilization.memory") || !strings.Contains(joined, "--id=GPU-1") {
			t.Fatalf("args=%q", joined)
		}
		return []byte("GPU-uuid, NVIDIA L4, 23034, 1024, 72, 88, 6251, 71.5, 62\n"), nil
	}
	got, err := collectNVIDIADeviceSnapshot(context.Background(), "GPU-1")
	if err != nil {
		t.Fatal(err)
	}
	if !got.available || got.collector != "nvidia-smi" || got.device.MemoryControllerUtilization == nil || *got.device.MemoryControllerUtilization != .88 {
		t.Fatalf("%+v", got)
	}
	if got.capacity.TotalBytes == nil || *got.capacity.TotalBytes != 23034*1024*1024 {
		t.Fatalf("%+v", got.capacity)
	}
}

func TestCollectNVIDIADeviceSnapshotUnavailableIsNotZero(t *testing.T) {
	old := runNvidiaSMI
	t.Cleanup(func() { runNvidiaSMI = old })
	runNvidiaSMI = func(context.Context, ...string) ([]byte, error) { return nil, errors.New("not installed") }
	got, err := collectNVIDIADeviceSnapshot(context.Background(), "")
	if err != nil || got.available || got.device.MemoryControllerUtilization != nil {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}

func TestCollectNVIDIADeviceSnapshotPreservesUnsupportedFields(t *testing.T) {
	old := runNvidiaSMI
	t.Cleanup(func() { runNvidiaSMI = old })
	runNvidiaSMI = func(context.Context, ...string) ([]byte, error) {
		return []byte("GPU-uuid, NVIDIA L4, 23034, 1024, 72, N/A, 6251, [Not Supported], 62\n"), nil
	}
	got, err := collectNVIDIADeviceSnapshot(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if !got.available || got.device.MemoryControllerUtilization != nil || got.device.PowerWatts != nil {
		t.Fatalf("%+v", got)
	}
}
