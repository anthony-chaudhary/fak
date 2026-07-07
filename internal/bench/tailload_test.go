package bench

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// #2223 R6 smoke: both arms measure the same decide fold in one process, the
// loaded arm actually ran under synthetic load, and no gate is asserted — the
// numbers are OBSERVED research output, sized down here so CI stays fast.
func TestRunTailUnderLoad_Smoke(t *testing.T) {
	rep, err := RunTailUnderLoad(context.Background(), TailLoadConfig{
		Samples: 2_000, Warmup: 200, StreamWorkers: 1, ChurnWorkers: 1, SessionIDs: 64,
	})
	if err != nil {
		t.Fatalf("RunTailUnderLoad: %v", err)
	}
	if rep.Schema != TailLoadSchema {
		t.Errorf("schema = %q, want %q", rep.Schema, TailLoadSchema)
	}
	fineClock := !coarseClock()
	for _, arm := range []TailArm{rep.Quiet, rep.Loaded} {
		if arm.Samples != 2_000 {
			t.Errorf("%s samples = %d, want 2000", arm.Label, arm.Samples)
		}
		if fineClock && (arm.P50NS <= 0 || arm.P99NS < arm.P50NS || arm.MaxNS < arm.P99NS) {
			t.Errorf("%s distribution not ordered: p50=%d p99=%d max=%d", arm.Label, arm.P50NS, arm.P99NS, arm.MaxNS)
		}
	}
	if !fineClock {
		t.Log("coarse monotonic clock: distribution ordering not asserted on this host")
	}
	if !strings.Contains(rep.Fence, "no gate asserted") {
		t.Errorf("fence must travel with the artifact; got %q", rep.Fence)
	}
	if rep.Config.StreamWorkers != 1 || rep.Config.ChurnWorkers != 1 {
		t.Errorf("config not echoed: %+v", rep.Config)
	}
}

// The percentile fold is exact over raw samples (no bucket quantization).
func TestFoldTailArm_ExactPercentiles(t *testing.T) {
	samples := make([]int64, 1000)
	for i := range samples {
		samples[i] = int64(i + 1) // 1..1000 ns
	}
	arm := foldTailArm("synthetic", samples)
	if arm.P50NS != 501 || arm.P99NS != 991 || arm.P999NS != 1000 || arm.MaxNS != 1000 {
		t.Errorf("percentiles = p50 %d p99 %d p999 %d max %d, want 501/991/1000/1000",
			arm.P50NS, arm.P99NS, arm.P999NS, arm.MaxNS)
	}
	if arm.Over100us != 0 {
		t.Errorf("over-100µs count = %d, want 0", arm.Over100us)
	}
}

// Artifact writer, env-gated: FAK_TAILLOAD_OUT=<path> runs the full-size arm
// and writes the report JSON (the committed artifacts under
// docs/benchmarks/fullspan/ come from exactly this test).
func TestWriteTailLoadArtifact(t *testing.T) {
	out := os.Getenv("FAK_TAILLOAD_OUT")
	if out == "" {
		t.Skip("set FAK_TAILLOAD_OUT=<path> to run the full tail-under-load arm and write the artifact")
	}
	if coarseClock() {
		t.Skip("coarse monotonic clock (~0.5 ms tick): refusing to write a quantized artifact; run under Linux (e.g. WSL)")
	}
	rep, err := RunTailUnderLoad(context.Background(), TailLoadConfig{})
	if err != nil {
		t.Fatalf("RunTailUnderLoad: %v", err)
	}
	raw, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := os.WriteFile(out, append(raw, '\n'), 0o644); err != nil {
		t.Fatalf("write %s: %v", out, err)
	}
	t.Logf("tail-under-load report written to %s (quiet p99=%dns loaded p99=%dns)", out, rep.Quiet.P99NS, rep.Loaded.P99NS)
}
