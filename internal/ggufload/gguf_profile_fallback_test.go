package ggufload

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
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
					encoded, err := json.Marshal(row)
					if err != nil {
						t.Fatal(err)
					}
					var measured struct {
						Bytes   *int64 `json:"conversion_f32_bytes"`
						Tensors int    `json:"conversion_f32_tensors"`
					}
					if err := json.Unmarshal(encoded, &measured); err != nil {
						t.Fatal(err)
					}
					if measured.Bytes == nil || *measured.Bytes != int64(dim*vocab*4) || measured.Tensors != 1 {
						t.Errorf("%s emitted F32 payload: bytes=%v observed_tensors=%d want bytes=%d tensors=1", quant, measured.Bytes, measured.Tensors, dim*vocab*4)
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
					for _, fact := range []string{"dense", "F32", "dequant_tensors=1", fmt.Sprintf("source_bytes=%d", row.DequantBytes), fmt.Sprintf("emitted_f32_bytes=%d", dim*vocab*4), "observed_conversion_tensors=1/1"} {
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

// fak-test:runtime fast est=1ms
func TestLoadPathConversionCoverage(t *testing.T) {
	p := NewLoadProfiler()
	p.recordLoadPath("Q4_K", false, true, 144, 1, nil)
	resident := p.loadPathRows()[0]
	if resident.ConversionF32Bytes == nil || *resident.ConversionF32Bytes != 0 {
		t.Fatal("resident-only path must observe zero conversion")
	}
	p.recordLoadPath("Q4_K", false, false, 144, 1, []pendingTensor{{}})
	unknown := p.loadPathRows()[0]
	if unknown.ConversionF32Bytes != nil || unknown.ConversionF32Tensors != 0 {
		t.Fatal("missing payload must remain unobserved")
	}
	p.recordLoadPath("Q4_K", false, false, 144, 1, []pendingTensor{{f32: make([]float32, 256)}})
	partial := p.Snapshot("test", "fixture.gguf", 1).LoadPaths[0]
	if partial.ConversionF32Bytes == nil || *partial.ConversionF32Bytes != 1024 || partial.ConversionF32Tensors != 1 || partial.DequantTensors != 2 {
		t.Fatalf("partial observation lost its coverage: %+v", partial)
	}
	p.recordLoadPath("Q4_K", false, false, 144, 1, []pendingTensor{{f32: make([]float32, 256)}})
	if *partial.ConversionF32Bytes != 1024 || *resident.ConversionF32Bytes != 0 || unknown.ConversionF32Bytes != nil {
		t.Fatal("later collection changed earlier snapshots")
	}
	*partial.ConversionF32Bytes = 99
	if current := p.loadPathRows()[0]; current.ConversionF32Bytes == nil || *current.ConversionF32Bytes != 2048 {
		t.Fatal("snapshot mutation changed profiler accounting")
	}
	maxBytes := int64(math.MaxInt64)
	p.loadPaths[loadPathKey{quantType: "Q4_K"}].ConversionF32Bytes = &maxBytes
	p.recordLoadPath("Q4_K", false, false, 144, 1, []pendingTensor{{f32: []float32{1}}})
	overflow := p.loadPathRows()[0]
	if *overflow.ConversionF32Bytes != math.MaxInt64 || overflow.ConversionF32Tensors != 2 || overflow.DequantTensors != 4 {
		t.Fatalf("overflow must stay outside observation coverage: %+v", overflow)
	}
}
