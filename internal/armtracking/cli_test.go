package armtracking

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunCLIUsageAndErrors(t *testing.T) {
	var stdout, stderr bytes.Buffer

	// No args -> usage
	code := RunCLI(&stdout, &stderr, []string{})
	if code != 2 {
		t.Errorf("expected exit 2 on empty args, got %d", code)
	}

	// Unknown subcommand -> usage
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{"unknown-sub"})
	if code != 2 {
		t.Errorf("expected exit 2 on unknown subcommand, got %d", code)
	}

	// Record with missing args -> exit 2
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{"record", "--workload", "w1"})
	if code != 2 {
		t.Errorf("expected exit 2 on missing record flags, got %d", code)
	}

	// Compare with missing args -> exit 2
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{"compare", "--workload", "w1"})
	if code != 2 {
		t.Errorf("expected exit 2 on missing compare flags, got %d", code)
	}

	// Audit with missing workload -> exit 2
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{"audit"})
	if code != 2 {
		t.Errorf("expected exit 2 on missing audit workload, got %d", code)
	}
}

func TestRunCLILifecycle(t *testing.T) {
	storePath := filepath.Join(t.TempDir(), "cli-shifting-baselines.json")
	var stdout, stderr bytes.Buffer

	// Step 1: Record initial baseline arm
	code := RunCLI(&stdout, &stderr, []string{
		"record",
		"--workload", "strix-q4k",
		"--arm", "cpu_reference",
		"--kind", "baseline",
		"--metric", "throughput_tok_s",
		"--value", "50.0",
		"--unit", "tok/s",
		"--direction", "higher_is_better",
		"--dimension", "target",
		"--notes", "initial reference",
		"--store", storePath,
	})
	if code != 0 {
		t.Fatalf("record baseline failed (code %d): %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Recorded baseline arm \"cpu_reference\"") {
		t.Errorf("unexpected stdout: %s", stdout.String())
	}

	// Step 2: Baseline shifts! Record improved baseline
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{
		"record",
		"--workload", "strix-q4k",
		"--arm", "cpu_reference",
		"--kind", "baseline",
		"--metric", "throughput_tok_s",
		"--value", "75.0",
		"--unit", "tok/s",
		"--direction", "higher_is_better",
		"--dimension", "target",
		"--notes", "AVX-512 GEMV optimization",
		"--store", storePath,
	})
	if code != 0 {
		t.Fatalf("record baseline shift failed (code %d): %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Prior best: 50 -> New: 75 (delta: +25)") {
		t.Errorf("expected shift output with delta, got: %s", stdout.String())
	}

	// Step 3: Record an ablation arm (discrete Vulkan: 200 tok/s)
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{
		"record",
		"--workload", "strix-q4k",
		"--arm", "vulkan_discrete",
		"--kind", "ablation",
		"--metric", "throughput_tok_s",
		"--value", "200.0",
		"--unit", "tok/s",
		"--direction", "higher_is_better",
		"--dimension", "topology",
		"--notes", "discrete Vulkan kernel",
		"--store", storePath,
	})
	if code != 0 {
		t.Fatalf("record ablation arm failed: %s", stderr.String())
	}

	// Step 4: Record a candidate arm (fused device-local: 450 tok/s)
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{
		"record",
		"--workload", "strix-q4k",
		"--arm", "vulkan_fused_device_local",
		"--kind", "candidate",
		"--metric", "throughput_tok_s",
		"--value", "450.0",
		"--unit", "tok/s",
		"--direction", "higher_is_better",
		"--dimension", "residency",
		"--notes", "fused device-local kernel",
		"--store", storePath,
	})
	if code != 0 {
		t.Fatalf("record candidate arm failed: %s", stderr.String())
	}
	if !strings.Contains(stdout.String(), "Next-best comparison:") {
		t.Errorf("expected next-best comparison in record output, got: %s", stdout.String())
	}

	// Step 5: Compare candidate arm against next-best latest result
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{
		"compare",
		"--workload", "strix-q4k",
		"--arm", "vulkan_fused_device_local",
		"--store", storePath,
	})
	if code != 0 {
		t.Fatalf("compare champion failed: %s", stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Rank 1/3") {
		t.Errorf("expected champion Rank 1/3, got: %s", out)
	}
	if !strings.Contains(out, "Next Best Arm:   vulkan_discrete (Rank 2/3)") {
		t.Errorf("expected next-best arm vulkan_discrete (Rank 2/3), got: %s", out)
	}
	if !strings.Contains(out, "Speedup / Lift:  2.25× (+125.0%)") {
		t.Errorf("expected 2.25x speedup (+125.0%%), got: %s", out)
	}

	// Step 6: Compare runner-up against next-best latest result (the superior arm)
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{
		"compare",
		"--workload", "strix-q4k",
		"--arm", "vulkan_discrete",
		"--store", storePath,
	})
	if code != 0 {
		t.Fatalf("compare runner-up failed: %s", stderr.String())
	}
	out = stdout.String()
	if !strings.Contains(out, "Rank 2/3") {
		t.Errorf("expected runner-up Rank 2/3, got: %s", out)
	}
	if !strings.Contains(out, "Next Best Arm:   vulkan_fused_device_local (Rank 1/3)") {
		t.Errorf("expected next best arm vulkan_fused_device_local, got: %s", out)
	}

	// Step 7: Leaderboard view (tabular)
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{
		"leaderboard",
		"--workload", "strix-q4k",
		"--store", storePath,
	})
	if code != 0 {
		t.Fatalf("leaderboard failed: %s", stderr.String())
	}
	out = stdout.String()
	if !strings.Contains(out, "vulkan_fused_device_local") || !strings.Contains(out, "vulkan_discrete") || !strings.Contains(out, "cpu_reference") {
		t.Errorf("leaderboard missing arms: %s", out)
	}

	// Step 8: Audit history view
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{
		"audit",
		"--workload", "strix-q4k",
		"--store", storePath,
	})
	if code != 0 {
		t.Fatalf("audit failed: %s", stderr.String())
	}
	out = stdout.String()
	if !strings.Contains(out, "BASELINE_SHIFT") {
		t.Errorf("audit history missing BASELINE_SHIFT event: %s", out)
	}
	if !strings.Contains(out, "AVX-512 GEMV optimization") {
		t.Errorf("audit history missing shift note: %s", out)
	}

	// Step 9: Compare as JSON
	stdout.Reset()
	stderr.Reset()
	code = RunCLI(&stdout, &stderr, []string{
		"compare",
		"--workload", "strix-q4k",
		"--arm", "vulkan_fused_device_local",
		"--store", storePath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("compare JSON failed: %s", stderr.String())
	}
	if !strings.Contains(stdout.String(), `"is_champion": true`) {
		t.Errorf("expected JSON is_champion true, got: %s", stdout.String())
	}
}
