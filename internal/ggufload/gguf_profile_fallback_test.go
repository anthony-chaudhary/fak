package ggufload

import (
	"bytes"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=100ms
func TestLoadQuantFallbackAlerts(t *testing.T) {
	pinHostMemStatus(t, compute.HostMemStatus{})
	const dim, vocab = 256, 256
	path := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ4_K, TensorQ6_K, false, false)
	floatPath := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorF32, TensorQ6_K, false, false)
	for _, tc := range []struct {
		name        string
		resident    bool
		writer      bool
		floatSource bool
	}{
		{name: "fallback-alert-writer", writer: true},
		{name: "resident-control", resident: true, writer: true},
		{name: "fallback-durable-without-writer"},
		{name: "float-source-control", resident: true, writer: true, floatSource: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var screen bytes.Buffer
			prof := NewLoadProfiler()
			if tc.writer {
				prof.AlertWriter = &screen
			}
			fixture := path
			if tc.floatSource {
				fixture = floatPath
			}
			m, err := LoadModelQ4KProfileOptions(fixture, prof,
				WithQ4KEmbeddingResident(tc.resident && !tc.floatSource), WithDenseKQuantResident(tc.resident))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = m.CloseWeights() })
			profile := prof.Snapshot("test", "fixture.gguf", 1)
			converted := map[string]LoadPathStat{}
			floatConversions := 0
			for _, row := range profile.LoadPaths {
				if row.DequantTensors > 0 {
					if row.QuantType == "F32" {
						floatConversions += row.DequantTensors
						continue
					}
					converted[row.QuantType] = row
				}
			}
			if tc.floatSource && floatConversions != 1 {
				t.Fatalf("float-source fixture conversions = %d, want one F32 source row: %+v", floatConversions, profile.LoadPaths)
			}
			wantConverted := 2
			if tc.resident {
				wantConverted = 0
			}
			if len(converted) != wantConverted {
				t.Fatalf("fixture converted quant classes = %d, want %d: %+v", len(converted), wantConverted, profile.LoadPaths)
			}
			if !tc.resident {
				for quant, sourceBytes := range map[string]int64{
					"Q4_K": int64(vocab * blockQ4KBytes),
					"Q6_K": int64(vocab * blockQ6KBytes),
				} {
					row, ok := converted[quant]
					if !ok || row.Expert || row.DequantTensors != 1 || row.DequantBytes != sourceBytes {
						t.Fatalf("%s fixture conversion = %+v, want one dense tensor from %d source bytes", quant, row, sourceBytes)
					}
				}
			}
			alerts := map[string]LoadAlert{}
			for _, alert := range profile.Alerts {
				if alert.Kind != "quant-dequant-fallback" {
					continue
				}
				matches := 0
				for quant, row := range converted {
					if !strings.Contains(alert.Text, quant) {
						continue
					}
					matches++
					if _, duplicate := alerts[quant]; duplicate {
						t.Fatalf("duplicate fallback alert for %s: %+v", quant, profile.Alerts)
					}
					alerts[quant] = alert
					if alert.Level != "warning" {
						t.Errorf("%s fallback level = %q, want warning", quant, alert.Level)
					}
					for _, fact := range []string{"dense", "F32", "dequant_tensors=1", fmt.Sprintf("source_bytes=%d", row.DequantBytes)} {
						if !strings.Contains(alert.Text, fact) {
							t.Errorf("%s fallback alert omits %q: %q", quant, fact, alert.Text)
						}
					}
					if tc.writer && !strings.Contains(screen.String(), alert.Text) {
						t.Errorf("%s durable fallback alert absent from startup output", quant)
					}
				}
				if matches != 1 {
					t.Errorf("fallback alert does not describe exactly one converted class: %+v", alert)
				}
			}
			if len(alerts) != wantConverted {
				t.Errorf("startup fallback visibility: converted_classes=%d alerted_classes=%d silent_classes=%d", len(converted), len(alerts), len(converted)-len(alerts))
			}
			t.Logf("startup fallback visibility: converted_classes=%d alerted_classes=%d silent_classes=%d", len(converted), len(alerts), len(converted)-len(alerts))
			if out := screen.String(); strings.Contains(out, "loading model") || strings.Contains(out, "% (") || strings.Contains(out, "load-path breakdown") {
				t.Errorf("alerts-only startup emitted ordinary progress:\n%s", out)
			}
			if tc.resident && screen.Len() != 0 {
				t.Errorf("fully resident load emitted a warning: %s", screen.String())
			}
			data, err := json.Marshal(profile)
			if err != nil {
				t.Fatal(err)
			}
			var durable LoadProfile
			if err := json.Unmarshal(data, &durable); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(durable.Alerts, profile.Alerts) || !reflect.DeepEqual(durable.LoadPaths, profile.LoadPaths) {
				t.Fatal("profile JSON lost fallback evidence")
			}
		})
	}
}
