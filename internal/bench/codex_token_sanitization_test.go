package bench

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcacheextract"
)

func TestCodexTokenSanitizationComparisonWitness(t *testing.T) {
	result := vcacheextract.CompareLocal()
	want := []string{
		"fak native Codex token sanitizer",
		"raw JSONL pass-through",
		"fak + OpenTelemetry",
		"fak + Prometheus",
		"jq streaming projection",
		"Vector VRL remap",
		"Fluent Bit filter pipeline",
	}
	if len(result.Arms) != len(want) {
		t.Fatalf("arms=%d, want %d", len(result.Arms), len(want))
	}
	for i, arm := range result.Arms {
		if arm.Name != want[i] {
			t.Fatalf("arm[%d]=%q, want %q", i, arm.Name, want[i])
		}
		if i < 2 {
			if !arm.Available || arm.InputRows != 4 || arm.EligibleRows != 2 || arm.InputBytes == 0 || arm.OutputBytes == 0 {
				t.Fatalf("local arm %q: %+v", arm.Name, arm)
			}
			t.Logf("observed local arm: name=%q latency=%s input_rows=%d output_rows=%d input_bytes=%d output_bytes=%d", arm.Name, arm.Latency, arm.InputRows, arm.OutputRows, arm.InputBytes, arm.OutputBytes)
			continue
		}
		if arm.Available || arm.Correct || arm.CounterCorrect || arm.Latency != 0 || arm.InputRows != 0 || arm.EligibleRows != 0 || arm.OutputRows != 0 || arm.MissedRows != 0 || arm.ExtraRows != 0 || arm.ForbiddenFields != 0 || arm.ForbiddenBytes != 0 || arm.ParseFailures != 0 || arm.InputBytes != 0 || arm.OutputBytes != 0 || arm.CPUSeconds != 0 || arm.PeakRSSBytes != 0 || arm.NetworkBytes != 0 || arm.CostUSD != 0 {
			t.Fatalf("unwitnessed arm must remain an unavailable zero row: %+v", arm)
		}
	}
	native, raw := result.Arms[0], result.Arms[1]
	if !native.Correct || !native.CounterCorrect || native.OutputRows != 2 || native.MissedRows != 0 || native.ExtraRows != 0 || native.ForbiddenFields != 0 || native.ForbiddenBytes != 0 || native.ParseFailures != 0 {
		t.Fatalf("native sanitizer: %+v", native)
	}
	if raw.Correct || !raw.CounterCorrect || raw.OutputRows != 4 || raw.ExtraRows != 2 || raw.ForbiddenFields == 0 || raw.ForbiddenBytes == 0 {
		t.Fatalf("raw pass-through: %+v", raw)
	}
}
