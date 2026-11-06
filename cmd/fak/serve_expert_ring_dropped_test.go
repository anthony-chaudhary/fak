package main

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serve_expert_ring_dropped_test.go — an operator-named --expert-ring-bytes / FAK_EXPERT_RING_BYTES
// value must never be silently dropped: a load arm that never evaluates the activated-expert ring
// placement warns loudly by default (naming the dropping arm and the env var that carries the
// value to the ring arm) and refuses with ErrServeExpertRingDropped under FAK_EXPERT_RING_STRICT.
// The ring arm itself stays silent: a named size it takes is honored, not warned away.

// captureServeStderr runs fn with os.Stderr pointed at a concurrently drained pipe and returns
// everything written, so a multi-minute load can never block on the pipe buffer.
func captureServeStderr(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stderr
	os.Stderr = w
	var out []byte
	done := make(chan struct{})
	go func() {
		out, _ = io.ReadAll(r)
		close(done)
	}()
	fn()
	_ = w.Close()
	<-done
	os.Stderr = old
	return string(out)
}

// fak-test:runtime fast est=1ms lane=default
func TestReportDroppedExpertRingContract(t *testing.T) {
	env := func(ring, strict string) func(string) string {
		return func(k string) string {
			switch k {
			case FakExpertRingBytesEnv:
				return ring
			case FakExpertRingStrictEnv:
				return strict
			}
			return ""
		}
	}
	const wantLine = `fak serve: WARNING: --expert-ring-bytes="2g" is dropped by the --cpu-offload-experts load arm: it never evaluates the activated-expert device ring placement, so no ring allocation was made; only the resident-Q4_K device arm takes it, and only when the full plan does not fit. The value reaches that arm through FAK_EXPERT_RING_BYTES.`

	t.Run("unset_names_nothing", func(t *testing.T) {
		var err error
		got := captureServeStderr(t, func() { err = reportDroppedExpertRing("lean CPU", env("", "")) })
		if err != nil || got != "" {
			t.Fatalf("no named size: err=%v stderr=%q, want silence", err, got)
		}
	})
	t.Run("default_warns_and_continues", func(t *testing.T) {
		var err error
		got := captureServeStderr(t, func() { err = reportDroppedExpertRing("--cpu-offload-experts", env("2g", "")) })
		if err != nil {
			t.Fatalf("default mode must warn, not refuse: %v", err)
		}
		if !strings.Contains(got, wantLine+"\n") {
			t.Fatalf("warning line missing: got %q, want %q", got, wantLine)
		}
	})
	t.Run("strict_refuses_typed_without_warning_line", func(t *testing.T) {
		var err error
		got := captureServeStderr(t, func() { err = reportDroppedExpertRing("--cpu-offload-experts", env("2g", "1")) })
		if !errors.Is(err, ErrServeExpertRingDropped) {
			t.Fatalf("strict mode err=%v, want ErrServeExpertRingDropped", err)
		}
		want := `fak serve: --expert-ring-bytes names a ring the chosen load arm does not evaluate: --expert-ring-bytes="2g" on the --cpu-offload-experts load arm`
		if err.Error() != want {
			t.Fatalf("refusal text = %q, want %q", err.Error(), want)
		}
		if strings.Contains(got, "is dropped by") {
			t.Fatalf("strict mode must refuse without the default warning line: %q", got)
		}
	})
	t.Run("strict_off_keeps_the_default_warning", func(t *testing.T) {
		var err error
		got := captureServeStderr(t, func() { err = reportDroppedExpertRing("--cpu-offload-experts", env("2g", "0")) })
		if err != nil || !strings.Contains(got, wantLine+"\n") {
			t.Fatalf("FAK_EXPERT_RING_STRICT=0: err=%v stderr=%q, want the default warning", err, got)
		}
	})
}

// fak-test:runtime fast est=50ms lane=default
func TestDroppedExpertRingWarnsOnLeanCPUArm(t *testing.T) {
	orig := serveMetalAvailable
	t.Cleanup(func() { serveMetalAvailable = orig })
	serveMetalAvailable = func() bool { return false }
	t.Setenv("FAK_Q4K", "")
	t.Setenv(FakExpertRingBytesEnv, "2g")
	path := createTestQ8GGUF(t)

	var m *fakmodel.Model
	var err error
	got := captureServeStderr(t, func() {
		m, _, _, _, _, err = loadServeInKernelModelPlacedErr(path, nil, false, false, 2048, nil, 1, nil)
	})
	if err != nil {
		t.Fatalf("default mode must still load the dropped arm: %v", err)
	}
	if m == nil {
		t.Fatal("no model loaded")
	}
	defer m.CloseWeights()
	const wantLine = `fak serve: WARNING: --expert-ring-bytes="2g" is dropped by the lean CPU load arm: it never evaluates the activated-expert device ring placement, so no ring allocation was made; only the resident-Q4_K device arm takes it, and only when the full plan does not fit. The value reaches that arm through FAK_EXPERT_RING_BYTES.`
	if !strings.Contains(got, wantLine+"\n") {
		t.Fatalf("lean-CPU arm did not print the exact dropped-ring warning:\ngot:\n%s\nwant line:\n%s", got, wantLine)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestDroppedExpertRingStrictRefusesBeforeLoad(t *testing.T) {
	orig := serveMetalAvailable
	t.Cleanup(func() { serveMetalAvailable = orig })
	serveMetalAvailable = func() bool { return false }
	t.Setenv("FAK_Q4K", "")
	t.Setenv(FakExpertRingBytesEnv, "2g")
	t.Setenv(FakExpertRingStrictEnv, "1")
	path := createTestQ8GGUF(t)

	var m *fakmodel.Model
	var err error
	got := captureServeStderr(t, func() {
		m, _, _, _, _, err = loadServeInKernelModelPlacedErr(path, nil, false, false, 2048, nil, 1, nil)
	})
	if m != nil {
		t.Fatal("strict mode must not load a serve that drops the named ring")
	}
	if !errors.Is(err, ErrServeExpertRingDropped) {
		t.Fatalf("strict refusal err=%v (%T), want ErrServeExpertRingDropped", err, err)
	}
	if strings.Contains(got, "is dropped by") {
		t.Fatalf("strict mode must refuse without the default warning line: %q", got)
	}
}

// fak-test:runtime fast est=2s lane=default
func TestDroppedExpertRingSilentOnRingArm(t *testing.T) {
	const E, H, I = 2, 256, 256
	path := filepath.Join(t.TempDir(), "mixed-moe.gguf")
	if err := os.WriteFile(path, nonHaloRingMixedMoEGGUF(t, E, H, I), 0o600); err != nil {
		t.Fatal(err)
	}
	v41PackedTestEnv(t)
	t.Setenv(FakExpertRingBytesEnv, "token")
	be := newNonHaloCUDARingBackend(nonHaloRingLoadBytes, nonHaloRingLoadBytes)
	fit := serveFitBudget{Base: nonHaloRingLoadBytes, Headroom: serveGGUFDeviceHeadroom}

	var m *fakmodel.Model
	var profile *gateway.ModelLoadProfile
	var err error
	got := captureServeStderr(t, func() {
		m, _, profile, _, _, err = loadServeInKernelModelPlacedErr(path, be, false, false, 64, nil, 1, &fit)
	})
	if err != nil {
		t.Fatalf("ring arm must load with a named ring: %v", err)
	}
	defer m.CloseWeights()
	residency := ""
	for _, msg := range profile.Messages {
		if msg.Kind == "serving-expert-residency" {
			residency = msg.Text
		}
	}
	if !strings.Contains(residency, "planned device expert-ring capacity=") {
		t.Fatalf("ring placement not taken: %q", residency)
	}
	if !strings.Contains(got, "fak serve: activated-expert device ring:") {
		t.Fatalf("the admitted ring log line is missing:\n%s", got)
	}
	if strings.Contains(got, "is dropped by") {
		t.Fatalf("the ring arm emitted the dropped-arm warning:\n%s", got)
	}
}
