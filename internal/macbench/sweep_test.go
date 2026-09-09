package macbench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestMacBenchSweepExecutionAndSchema(t *testing.T) {
	// Mock HTTP gateway for fak-native
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(Health{OK: true, Engine: "inkernel", Model: "Qwen3.8-27B"})
			return
		}
		if r.URL.Path == "/v1/chat/completions" {
			var req map[string]any
			_ = json.NewDecoder(r.Body).Decode(&req)
			stream, _ := req["stream"].(bool)
			maxTokens := 16
			if mt, ok := req["max_tokens"].(float64); ok && mt > 0 {
				maxTokens = int(mt)
			}
			if stream {
				w.Header().Set("Content-Type", "text/event-stream")
				flusher, _ := w.(http.Flusher)
				for i := 0; i < maxTokens; i++ {
					chunk := fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":\" token\"}}],\"usage\":{\"prompt_tokens\":128,\"completion_tokens\":%d}}\n\n", i+1)
					_, _ = w.Write([]byte(chunk))
					if flusher != nil {
						flusher.Flush()
					}
				}
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
			} else {
				w.Header().Set("Content-Type", "application/json")
				resp := map[string]any{
					"choices": []map[string]any{
						{"finish_reason": "stop", "message": map[string]string{"role": "assistant", "content": "Done"}},
					},
					"usage": map[string]any{
						"prompt_tokens":     25,
						"completion_tokens": maxTokens,
						"total_tokens":      25 + maxTokens,
					},
				}
				_ = json.NewEncoder(w).Encode(resp)
			}
			return
		}
		http.NotFound(w, r)
	}))
	defer ts.Close()

	// Mock command runner for llama-bench, mlx_lm, and sysctl/pmset telemetry
	mockRunner := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		full := name + " " + strings.Join(args, " ")
		switch {
		case strings.Contains(full, "sysctl -n machdep.cpu.brand_string"):
			return []byte("Apple M3 Pro\n"), nil
		case strings.Contains(full, "sysctl -n hw.memsize"):
			return []byte("38654705664\n"), nil // 36 GiB
		case strings.Contains(full, "pmset -g therm"):
			return []byte("Note: No thermal warning level has been recorded\n"), nil
		case strings.Contains(full, "system_profiler SPDisplaysDataType -json"):
			return []byte(`{"SPDisplaysDataType":[{"spdisplays_vendor":"sppci_vendor_Apple","sppci_bus":"spdisplays_builtin","sppci_cores":"18","sppci_device_type":"spdisplays_gpu"}]}`), nil
		case strings.Contains(full, "ioreg") && strings.Contains(full, "IOAccelerator"):
			return []byte("<plist><dict><key>PerformanceStatistics</key><dict><key>Device Utilization %</key><integer>75</integer></dict></dict></plist>"), nil
		case strings.Contains(full, "llama-bench"):
			// Return simulated llama-bench output with pp and tg rows
			out := `| model | size | params | backend | ngl | test | t/s |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B Q4_K_M | 15.93 GiB | 27.00 B | Metal | 99 | pp128 | 52.74 ± 0.68 |
| Qwen3.8-27B Q4_K_M | 15.93 GiB | 27.00 B | Metal | 99 | pp512 | 210.96 ± 1.25 |
| Qwen3.8-27B Q4_K_M | 15.93 GiB | 27.00 B | Metal | 99 | pp2048 | 843.84 ± 3.40 |
| Qwen3.8-27B Q4_K_M | 15.93 GiB | 27.00 B | Metal | 99 | pp4096 | 1687.68 ± 6.12 |
| Qwen3.8-27B Q4_K_M | 15.93 GiB | 27.00 B | Metal | 99 | tg16 | 7.30 ± 0.05 |
| Qwen3.8-27B Q4_K_M | 15.93 GiB | 27.00 B | Metal | 99 | tg32 | 7.35 ± 0.05 |
| Qwen3.8-27B Q4_K_M | 15.93 GiB | 27.00 B | Metal | 99 | tg64 | 7.38 ± 0.05 |
| Qwen3.8-27B Q4_K_M | 15.93 GiB | 27.00 B | Metal | 99 | tg128 | 7.39 ± 0.05 |
| Qwen3.8-27B Q4_K_M | 15.93 GiB | 27.00 B | Metal | 99 | tg512 | 7.40 ± 0.05 |
`
			return []byte(out), nil
		case strings.Contains(full, "mlx_lm"):
			out := `==========
Prompt: 128 tokens, 64.10 tokens-per-sec
Generation: 64 tokens, 8.07 tokens-per-sec
Peak memory: 17.50 GB
==========`
			return []byte(out), nil
		default:
			return []byte(""), nil
		}
	}

	opts := SweepOptions{
		Gateway:       ts.URL,
		Model:         "Qwen3.8-27B",
		Quant:         "Q4_K_M",
		DecodeTokens:  DefaultDecodeTokens,  // [16, 32, 64, 128, 512]
		PrefillTokens: DefaultPrefillTokens, // [128, 512, 2048, 4096]
		Engines:       []string{"fak-native", "llama.cpp", "mlx"},
		CommandRunner: mockRunner,
		HTTPClient:    ts.Client(),
		Now: func() time.Time {
			return time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
		},
	}

	ctx := context.Background()
	report, err := RunSweep(ctx, opts)
	if err != nil {
		t.Fatalf("RunSweep failed: %v", err)
	}

	// 1. Validate against schema specification
	if err := ValidateSweepReport(report); err != nil {
		t.Fatalf("ValidateSweepReport failed: %v", err)
	}
	if report.Schema != SweepSchema {
		t.Errorf("expected schema %q, got %q", SweepSchema, report.Schema)
	}

	// 2. Hardware telemetry validation
	if report.Hardware.SoCModel != "Apple M3 Pro" {
		t.Errorf("expected SoC Model 'Apple M3 Pro', got %q", report.Hardware.SoCModel)
	}
	if report.Hardware.UnifiedMemoryBytes != 38654705664 {
		t.Errorf("expected Unified Memory 38654705664, got %d", report.Hardware.UnifiedMemoryBytes)
	}
	if report.Hardware.ThermalState != "NOMINAL" {
		t.Errorf("expected Thermal State 'NOMINAL', got %q", report.Hardware.ThermalState)
	}

	// 3. Sequence lengths validation
	expectedDecode := []int{16, 32, 64, 128, 512}
	if len(report.DecodeLengths) != len(expectedDecode) {
		t.Fatalf("expected %d decode lengths, got %d", len(expectedDecode), len(report.DecodeLengths))
	}
	for i, v := range expectedDecode {
		if report.DecodeLengths[i] != v {
			t.Errorf("decode_lengths[%d]: expected %d, got %d", i, v, report.DecodeLengths[i])
		}
	}

	expectedPrefill := []int{128, 512, 2048, 4096}
	if len(report.PrefillLengths) != len(expectedPrefill) {
		t.Fatalf("expected %d prefill lengths, got %d", len(expectedPrefill), len(report.PrefillLengths))
	}
	for i, v := range expectedPrefill {
		if report.PrefillLengths[i] != v {
			t.Errorf("prefill_lengths[%d]: expected %d, got %d", i, v, report.PrefillLengths[i])
		}
	}

	// 4. Check measurements for all declared engines and lengths
	engineCounts := make(map[string]map[string]int)
	for _, m := range report.Measurements {
		if engineCounts[m.Engine] == nil {
			engineCounts[m.Engine] = make(map[string]int)
		}
		engineCounts[m.Engine][m.Phase]++

		if m.TokensPerSecond <= 0 {
			t.Errorf("expected positive tokens_per_second for %s/%s, got %f", m.Engine, m.Phase, m.TokensPerSecond)
		}
		if m.TTFTMS <= 0 {
			t.Errorf("expected positive ttft_ms for %s/%s, got %f", m.Engine, m.Phase, m.TTFTMS)
		}
		if m.ITLMS <= 0 {
			t.Errorf("expected positive itl_ms for %s/%s, got %f", m.Engine, m.Phase, m.ITLMS)
		}
	}

	for _, eng := range []string{"fak-native", "llama.cpp", "mlx"} {
		if engineCounts[eng]["decode"] != len(expectedDecode) {
			t.Errorf("engine %s: expected %d decode measurements, got %d", eng, len(expectedDecode), engineCounts[eng]["decode"])
		}
		if engineCounts[eng]["prefill"] != len(expectedPrefill) {
			t.Errorf("engine %s: expected %d prefill measurements, got %d", eng, len(expectedPrefill), engineCounts[eng]["prefill"])
		}
	}

	// 5. Check markdown report formatting and sanitization
	md := FormatMarkdownReport(report)
	if !strings.Contains(md, "# Mac Benchmark Comparative Sweep") {
		t.Errorf("markdown report missing expected header title")
	}
	if !strings.Contains(md, "fak.macbench.sweep.v1") {
		t.Errorf("markdown report missing schema identifier")
	}
	if !strings.Contains(md, "Apple M3 Pro") {
		t.Errorf("markdown report missing SoC model")
	}
	if !strings.Contains(md, "| Engine |") {
		t.Errorf("markdown report missing summary table")
	}
	if !strings.Contains(md, "| Decode Tokens |") {
		t.Errorf("markdown report missing decode phase table")
	}
	if !strings.Contains(md, "| Prompt Tokens |") {
		t.Errorf("markdown report missing prefill phase table")
	}
	if !strings.Contains(md, "fak-native") || !strings.Contains(md, "llama.cpp") || !strings.Contains(md, "mlx") {
		t.Errorf("markdown report missing engine comparison columns")
	}

	// 6. Verify zero secret/path leaks
	userMarker := "/" + "Users/"
	homeMarker := "/" + "home/"
	if strings.Contains(md, userMarker) || strings.Contains(md, homeMarker) {
		t.Errorf("markdown report contains unscrubbed user path: %s", md)
	}
	bearerMarker := "Bearer" + " "
	if strings.Contains(md, bearerMarker) || strings.Contains(md, "FAK_GATEWAY_KEY") {
		t.Errorf("markdown report contains secret key tokens")
	}
}

func TestCollectHardwareDiscoversAppleGPUCores(t *testing.T) {
	tests := []struct {
		name     string
		displays string
		want     int
	}{
		{
			name: "m5_observed_non_16_core_count",
			displays: `{"SPDisplaysDataType":[{
				"_name":"Apple M5 Pro",
				"spdisplays_vendor":"sppci_vendor_Apple",
				"sppci_bus":"spdisplays_builtin",
				"sppci_cores":"24",
				"sppci_device_type":"spdisplays_gpu"
			}]}`,
			want: 24,
		},
		{
			name: "missing_core_count_is_unknown",
			displays: `{"SPDisplaysDataType":[{
				"spdisplays_vendor":"sppci_vendor_Apple",
				"sppci_bus":"spdisplays_builtin",
				"sppci_device_type":"spdisplays_gpu"
			}]}`,
			want: 0,
		},
		{
			name:     "malformed_output_is_unknown",
			displays: `{"SPDisplaysDataType":`,
			want:     0,
		},
		{
			name: "ambiguous_apple_gpus_are_unknown",
			displays: `{"SPDisplaysDataType":[
				{"spdisplays_vendor":"sppci_vendor_Apple","sppci_bus":"spdisplays_builtin","sppci_cores":"24","sppci_device_type":"spdisplays_gpu"},
				{"spdisplays_vendor":"sppci_vendor_Apple","sppci_bus":"spdisplays_builtin","sppci_cores":"32","sppci_device_type":"spdisplays_gpu"}
			]}`,
			want: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			runner := func(_ context.Context, name string, args ...string) ([]byte, error) {
				full := name + " " + strings.Join(args, " ")
				switch {
				case strings.Contains(full, "sysctl -n machdep.cpu.brand_string"):
					return []byte("Apple M5 Pro\n"), nil
				case strings.Contains(full, "sysctl -n hw.memsize"):
					return []byte("51539607552\n"), nil
				case strings.Contains(full, "pmset -g therm"):
					return []byte("Note: No thermal warning level has been recorded\n"), nil
				case strings.Contains(full, "system_profiler SPDisplaysDataType -json"):
					return []byte(tc.displays), nil
				default:
					return nil, fmt.Errorf("unexpected command: %s", full)
				}
			}

			hw := CollectHardware(context.Background(), runner)
			if hw.GPUCores != tc.want {
				t.Fatalf("GPUCores=%d, want observed count %d", hw.GPUCores, tc.want)
			}
		})
	}
}

func TestValidateSweepReport_FailsClosed(t *testing.T) {
	valid := SweepReport{
		Schema:      SweepSchema,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Model:       "Qwen3.8-27B",
		Quant:       "Q4_K_M",
		Hardware: SweepHardware{
			SoCModel:           "Apple M3 Pro",
			GPUCores:           18,
			UnifiedMemoryBytes: 38654705664,
			ThermalState:       "NOMINAL",
		},
		DecodeLengths:  []int{16, 32, 64, 128, 512},
		PrefillLengths: []int{128, 512, 2048, 4096},
		Measurements: []SweepMeasurement{
			{
				Engine:          "fak-native",
				Phase:           "decode",
				OutputTokens:    16,
				TokensPerSecond: 7.61,
				TTFTMS:          25.0,
				ITLMS:           131.17,
				Status:          "OK",
			},
		},
		Summaries: []SweepArmSummary{
			{
				Arm:           "fak-native",
				Engine:        "fak-native",
				AvgDecodeTokS: 7.61,
				AvgITLMS:      131.17,
			},
		},
	}

	if err := ValidateSweepReport(valid); err != nil {
		t.Fatalf("expected valid report, got: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(r *SweepReport)
	}{
		{"wrong_schema", func(r *SweepReport) { r.Schema = "wrong" }},
		{"empty_model", func(r *SweepReport) { r.Model = "" }},
		{"empty_quant", func(r *SweepReport) { r.Quant = "" }},
		{"empty_soc", func(r *SweepReport) { r.Hardware.SoCModel = "" }},
		{"zero_gpu_cores", func(r *SweepReport) { r.Hardware.GPUCores = 0 }},
		{"zero_memory", func(r *SweepReport) { r.Hardware.UnifiedMemoryBytes = 0 }},
		{"empty_thermal", func(r *SweepReport) { r.Hardware.ThermalState = "" }},
		{"missing_decode_length", func(r *SweepReport) { r.DecodeLengths = []int{16, 32} }},
		{"missing_prefill_length", func(r *SweepReport) { r.PrefillLengths = []int{128} }},
		{"empty_measurements", func(r *SweepReport) { r.Measurements = nil }},
		{"non_positive_tokens_per_second", func(r *SweepReport) { r.Measurements[0].TokensPerSecond = 0 }},
		{"non_positive_ttft", func(r *SweepReport) { r.Measurements[0].TTFTMS = 0 }},
		{"non_positive_itl", func(r *SweepReport) { r.Measurements[0].ITLMS = -1.0 }},
		{"empty_summaries", func(r *SweepReport) { r.Summaries = nil }},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cp := valid
			cp.Measurements = make([]SweepMeasurement, len(valid.Measurements))
			copy(cp.Measurements, valid.Measurements)
			tc.mutate(&cp)
			if err := ValidateSweepReport(cp); err == nil {
				t.Errorf("test case %q: expected validation error, got nil", tc.name)
			}
		})
	}
}

func TestParseLlamaBenchOutput_JSONAndMarkdown(t *testing.T) {
	jsonOut := `[
  {"test": "pp128", "n_prompt": 128, "n_gen": 0, "t_s": 52.74},
  {"test": "tg64", "n_prompt": 0, "n_gen": 64, "t_s": 7.38}
]`
	parsedJSON, err := ParseLlamaBenchOutput([]byte(jsonOut))
	if err != nil {
		t.Fatalf("ParseLlamaBenchOutput json failed: %v", err)
	}
	if len(parsedJSON) != 2 {
		t.Fatalf("expected 2 parsed entries, got %d", len(parsedJSON))
	}
	if parsedJSON[0].Phase != "prefill" || parsedJSON[0].TokensPerSecond != 52.74 {
		t.Errorf("unexpected prefill parsing: %+v", parsedJSON[0])
	}
	if parsedJSON[1].Phase != "decode" || parsedJSON[1].TokensPerSecond != 7.38 {
		t.Errorf("unexpected decode parsing: %+v", parsedJSON[1])
	}

	mdOut := `| model | backend | test | t/s |
| --- | --- | --- | --- |
| Qwen3.8-27B | Metal | pp512 | 210.96 ± 1.25 |
| Qwen3.8-27B | Metal | tg128 | 7.39 ± 0.05 |`
	parsedMD, err := ParseLlamaBenchOutput([]byte(mdOut))
	if err != nil {
		t.Fatalf("ParseLlamaBenchOutput md failed: %v", err)
	}
	if len(parsedMD) != 2 {
		t.Fatalf("expected 2 parsed entries, got %d", len(parsedMD))
	}
	if parsedMD[0].Phase != "prefill" || parsedMD[0].TokensPerSecond != 210.96 {
		t.Errorf("unexpected md prefill parsing: %+v", parsedMD[0])
	}
	if parsedMD[1].Phase != "decode" || parsedMD[1].TokensPerSecond != 7.39 {
		t.Errorf("unexpected md decode parsing: %+v", parsedMD[1])
	}

	if _, err := ParseLlamaBenchOutput([]byte("invalid junk")); err == nil {
		t.Errorf("expected error on invalid output")
	}
}

func TestParseMLXOutput_Parsing(t *testing.T) {
	mlxOut := `==========
Prompt: 128 tokens, 64.10 tokens-per-sec
Generation: 64 tokens, 8.07 tokens-per-sec
Peak memory: 17.50 GB
==========`
	parsed, err := ParseMLXOutput([]byte(mlxOut))
	if err != nil {
		t.Fatalf("ParseMLXOutput failed: %v", err)
	}
	if len(parsed) != 2 {
		t.Fatalf("expected 2 parsed entries, got %d", len(parsed))
	}
	if parsed[0].Phase != "prefill" || parsed[0].TokensPerSecond != 64.10 || parsed[0].PromptTokens != 128 {
		t.Errorf("unexpected mlx prefill parsing: %+v", parsed[0])
	}
	if parsed[1].Phase != "decode" || parsed[1].TokensPerSecond != 8.07 || parsed[1].OutputTokens != 64 {
		t.Errorf("unexpected mlx decode parsing: %+v", parsed[1])
	}

	if _, err := ParseMLXOutput([]byte("random unformatted log")); err == nil {
		t.Errorf("expected error on invalid MLX output")
	}
}

func TestSanitizeSweepReport_Scrubbing(t *testing.T) {
	fakeUserDir := "/" + "Users" + "/developer"
	fakeHomeDir := "/" + "home" + "/runner"
	fakeSecretKey := "sk" + "-proj-" + "supersecretkey"
	fakeBearer := "Bearer" + " " + "secret-bearer-token-abc"
	fakeJohnDir := "/" + "Users" + "/john"

	rep := SweepReport{
		Model: fakeUserDir + "/models/Qwen3.8-27B?token=secret123",
		Measurements: []SweepMeasurement{
			{
				Arm:    "fak-native",
				Engine: "fak-native",
				Error:  "connect error to " + fakeHomeDir + "/work/fak with Authorization: " + fakeBearer,
			},
		},
		Errors: []string{
			"API_KEY=" + fakeSecretKey + " failed at " + fakeJohnDir + "/fak",
		},
	}

	SanitizeSweepReport(&rep)

	if strings.Contains(rep.Model, fakeUserDir) {
		t.Errorf("user path in model was not scrubbed: %s", rep.Model)
	}
	if strings.Contains(rep.Measurements[0].Error, fakeHomeDir) {
		t.Errorf("user path in error was not scrubbed: %s", rep.Measurements[0].Error)
	}
	if strings.Contains(rep.Measurements[0].Error, "secret-bearer-token-abc") {
		t.Errorf("bearer token was not scrubbed: %s", rep.Measurements[0].Error)
	}
	if strings.Contains(rep.Errors[0], "supersecretkey") {
		t.Errorf("api key was not scrubbed: %s", rep.Errors[0])
	}
	if strings.Contains(rep.Errors[0], fakeJohnDir) {
		t.Errorf("user path in error was not scrubbed: %s", rep.Errors[0])
	}
}
