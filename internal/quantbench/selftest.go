package quantbench

import "time"

type SelfTestCase struct {
	Name       string   `json:"name"`
	Covers     []string `json:"covers"`
	Input      Input    `json:"input"`
	Want       Outcome  `json:"want_outcome"`
	WantReason string   `json:"want_reason_code"`
	Result     Result   `json:"result"`
	Pass       bool     `json:"pass"`
}

type SelfTestReport struct {
	Schema             string         `json:"schema"`
	Kind               string         `json:"kind"`
	Cases              []SelfTestCase `json:"cases"`
	RequiredDimensions []string       `json:"required_dimensions"`
	Pass               bool           `json:"pass"`
}

func fixture(format, version, runtime string) Input {
	return Input{
		ID:         format + "-" + runtime,
		Artifact:   Artifact{Format: format, Version: version, Recipe: "reviewed public fixture; deterministic calibration recipe"},
		Runtime:    Runtime{Name: runtime, Version: "fixture-1"},
		Baseline:   TunedBaseline{Name: "tuned-fp16-native", Artifact: Artifact{Format: "safetensors", Version: "1", Recipe: "fp16"}, Runtime: Runtime{Name: runtime, Version: "fixture-1"}, Tuning: []string{"batch=1", "warmup=2", "same prompt and decode length"}},
		Provenance: Provenance{Label: "modeled", Source: "go test ./internal/quantbench -run TestSelfTestMatrix", CapturedAt: time.Date(2026, 8, 10, 0, 0, 0, 0, time.UTC).Format(time.RFC3339)},
		Hardware:   HardwareEnvelope{OS: "linux", Architecture: "amd64", Accelerator: "fixture-gpu", AcceleratorMemoryBytes: 24 << 30, HostMemoryBytes: 64 << 30, Driver: "fixture"},
		Native:     NativeMeasurements{LoadSuccess: true, PeakMemoryBytes: 5 << 30, FirstTokenMilliseconds: 40, OutputTokensPerSecond: 80, Quality: Quality{Metric: "perplexity", Value: 6.2, Unit: "score", Dataset: "fixture-corpus"}},
		FAK:        FAKOverhead{PeakMemoryBytes: 1 << 20, FirstTokenMilliseconds: .3, OutputTokensPerSecondDelta: -.1, QualityDelta: 0},
	}
}

func SelfTest() SelfTestReport {
	goodGGUF := fixture("gguf", "v3", "llama.cpp")
	goodAWQ := fixture("awq", "1", "vllm")
	missing := fixture("gptq", "1", "transformers")
	missing.Baseline.Name = ""
	missing.Hardware.Accelerator = ""
	missing.Provenance.Source = ""
	missing.Native.Quality.Metric = ""
	unknownFormat := fixture("futureq", "9", "vllm")
	unknownRuntime := fixture("gguf", "v3", "future-runtime")
	unsupported := fixture("awq", "1", "llama.cpp")
	rows := []SelfTestCase{
		{Name: "gguf-native-matrix", Covers: []string{"load_success", "memory", "first_token", "throughput", "quality", "native_vs_fak", "tuned_baseline", "provenance", "hardware_envelope"}, Input: goodGGUF, Want: OutcomeBenchmark, WantReason: ReasonSupported},
		{Name: "awq-native-matrix", Covers: []string{"load_success", "memory", "first_token", "throughput", "quality", "native_vs_fak"}, Input: goodAWQ, Want: OutcomeBenchmark, WantReason: ReasonSupported},
		{Name: "missing-claim-envelope", Covers: []string{"tuned_baseline", "provenance", "hardware_envelope", "quality"}, Input: missing, Want: OutcomeRefuse, WantReason: ReasonInvalidEvidence},
		{Name: "unknown-format", Covers: []string{"typed_abstain"}, Input: unknownFormat, Want: OutcomeAbstain, WantReason: ReasonUnknownFormat},
		{Name: "unknown-runtime", Covers: []string{"typed_handoff"}, Input: unknownRuntime, Want: OutcomeDelegate, WantReason: ReasonUnknownRuntime},
		{Name: "unsupported-combination", Covers: []string{"typed_refuse"}, Input: unsupported, Want: OutcomeRefuse, WantReason: ReasonPairRejected},
	}
	pass := true
	for i := range rows {
		rows[i].Result = Evaluate(rows[i].Input)
		rows[i].Pass = rows[i].Result.Outcome == rows[i].Want && rows[i].Result.ReasonCode == rows[i].WantReason
		pass = pass && rows[i].Pass
	}
	return SelfTestReport{Schema: Schema, Kind: "self-test-matrix", Cases: rows, RequiredDimensions: []string{"load_success", "memory", "first_token", "throughput", "quality", "native_vs_fak", "typed_abstain", "typed_refuse", "typed_handoff", "tuned_baseline", "provenance", "hardware_envelope"}, Pass: pass}
}
