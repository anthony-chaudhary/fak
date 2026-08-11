package quantbench

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/claimcheck"
)

const Schema = "fak-quantbench/1"

type Outcome string

const (
	OutcomeBenchmark Outcome = "benchmark"
	OutcomeAbstain   Outcome = "abstain"
	OutcomeRefuse    Outcome = "refuse"
	OutcomeDelegate  Outcome = "delegate"
)

const (
	ReasonSupported         = "COMBINATION_REVIEWED"
	ReasonUnknownFormat     = "FORMAT_UNRECOGNIZED"
	ReasonUnknownRuntime    = "RUNTIME_UNRECOGNIZED"
	ReasonPairRejected      = "COMBINATION_REJECTED"
	ReasonRuntimeOwnsFormat = "NATIVE_PROBE_NEEDED"
	ReasonInvalidEvidence   = "EVIDENCE_INCOMPLETE"
)

type Artifact struct {
	Format  string `json:"format"`
	Version string `json:"version"`
	Recipe  string `json:"recipe"`
}

type Runtime struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type TunedBaseline struct {
	Name     string   `json:"name"`
	Artifact Artifact `json:"artifact"`
	Runtime  Runtime  `json:"runtime"`
	Tuning   []string `json:"tuning"`
}

type Provenance struct {
	Label      string `json:"label"` // witnessed, observed, or modeled
	Source     string `json:"source"`
	CapturedAt string `json:"captured_at"`
}

type HardwareEnvelope struct {
	OS                     string `json:"os"`
	Architecture           string `json:"architecture"`
	Accelerator            string `json:"accelerator"`
	AcceleratorMemoryBytes uint64 `json:"accelerator_memory_bytes"`
	HostMemoryBytes        uint64 `json:"host_memory_bytes"`
	Driver                 string `json:"driver"`
}

type Quality struct {
	Metric  string  `json:"metric"`
	Value   float64 `json:"value"`
	Unit    string  `json:"unit"`
	Dataset string  `json:"dataset"`
}

type NativeMeasurements struct {
	LoadSuccess            bool    `json:"load_success"`
	PeakMemoryBytes        uint64  `json:"peak_memory_bytes"`
	FirstTokenMilliseconds float64 `json:"first_token_ms"`
	OutputTokensPerSecond  float64 `json:"output_tokens_per_second"`
	Quality                Quality `json:"quality"`
}

type FAKOverhead struct {
	PeakMemoryBytes            uint64  `json:"peak_memory_bytes"`
	FirstTokenMilliseconds     float64 `json:"first_token_ms"`
	OutputTokensPerSecondDelta float64 `json:"output_tokens_per_second_delta"`
	QualityDelta               float64 `json:"quality_delta"`
}

type Input struct {
	ID         string             `json:"id"`
	Artifact   Artifact           `json:"artifact"`
	Runtime    Runtime            `json:"runtime"`
	Baseline   TunedBaseline      `json:"tuned_baseline"`
	Provenance Provenance         `json:"provenance"`
	Hardware   HardwareEnvelope   `json:"hardware_envelope"`
	Native     NativeMeasurements `json:"native_quantizer_runtime"`
	FAK        FAKOverhead        `json:"fak_overhead"`
}

type ClaimGrade struct {
	Verdict string   `json:"verdict"`
	Reasons []string `json:"reasons,omitempty"`
}

type Result struct {
	Schema     string              `json:"schema"`
	ID         string              `json:"id,omitempty"`
	Outcome    Outcome             `json:"decision"`
	ReasonCode string              `json:"reason_code"`
	Detail     string              `json:"detail"`
	Artifact   Artifact            `json:"artifact"`
	Runtime    Runtime             `json:"runtime"`
	Baseline   *TunedBaseline      `json:"tuned_baseline,omitempty"`
	Provenance *Provenance         `json:"provenance,omitempty"`
	Hardware   *HardwareEnvelope   `json:"hardware_envelope,omitempty"`
	Native     *NativeMeasurements `json:"native_quantizer_runtime,omitempty"`
	FAK        *FAKOverhead        `json:"fak_overhead,omitempty"`
	ClaimGrade *ClaimGrade         `json:"claim_check,omitempty"`
}

type support struct{ versions, runtimes map[string]bool }

var formats = map[string]support{
	"gguf": {set("v2", "v3"), set("llama.cpp")},
	"awq":  {set("1"), set("vllm", "transformers")},
	"gptq": {set("1"), set("vllm", "transformers")},
}
var knownRuntimes = set("llama.cpp", "vllm", "transformers")

func set(v ...string) map[string]bool {
	m := map[string]bool{}
	for _, x := range v {
		m[x] = true
	}
	return m
}
func norm(s string) string { return strings.ToLower(strings.TrimSpace(s)) }

// Evaluate returns a typed decision before accepting any performance claim.
func Evaluate(in Input) Result {
	in.Artifact.Format, in.Artifact.Version = norm(in.Artifact.Format), norm(in.Artifact.Version)
	in.Runtime.Name = norm(in.Runtime.Name)
	base := Result{Schema: Schema, ID: in.ID, Artifact: in.Artifact, Runtime: in.Runtime}
	f, ok := formats[in.Artifact.Format]
	if !ok {
		base.Outcome, base.ReasonCode = OutcomeAbstain, ReasonUnknownFormat
		base.Detail = "format is not in the reviewed quantbench registry"
		return base
	}
	if !f.versions[in.Artifact.Version] {
		base.Outcome, base.ReasonCode = OutcomeAbstain, ReasonUnknownFormat
		base.Detail = "format version is not in the reviewed quantbench registry"
		return base
	}
	if !knownRuntimes[in.Runtime.Name] {
		base.Outcome, base.ReasonCode = OutcomeDelegate, ReasonUnknownRuntime
		base.Detail = "delegate capability discovery and measurement to the named native runtime"
		return base
	}
	if !f.runtimes[in.Runtime.Name] {
		base.Outcome, base.ReasonCode = OutcomeRefuse, ReasonPairRejected
		base.Detail = "reviewed registry marks this format/runtime pair unsupported"
		return base
	}
	if strings.TrimSpace(in.Runtime.Version) == "" {
		base.Outcome, base.ReasonCode = OutcomeDelegate, ReasonRuntimeOwnsFormat
		base.Detail = "native runtime version and probe are required"
		return base
	}
	if err := validateEvidence(in); err != nil {
		base.Outcome, base.ReasonCode = OutcomeRefuse, ReasonInvalidEvidence
		base.Detail = err.Error()
		return base
	}
	base.Outcome, base.ReasonCode = OutcomeBenchmark, ReasonSupported
	base.Detail = "native measurements and fak overhead are reported in separate fields"
	base.Baseline, base.Provenance, base.Hardware = &in.Baseline, &in.Provenance, &in.Hardware
	base.Native, base.FAK = &in.Native, &in.FAK
	grade := claimcheck.Grade(claimcheck.Claim{Statement: "quantization benchmark result versus " + in.Baseline.Name, Baseline: claimcheck.Baseline{Kind: claimcheck.BaselineReal, Description: in.Baseline.Name}, Net: true, Scope: hardwareScope(in.Hardware), Provenance: claimcheck.Provenance(strings.ToUpper(in.Provenance.Label)), Witness: in.Provenance.Source, Realized: claimcheck.Realized{OnByDefault: true}})
	base.ClaimGrade = &ClaimGrade{Verdict: string(grade.Verdict), Reasons: grade.Missing}
	if grade.Verdict != claimcheck.NetTrue {
		base.Outcome, base.ReasonCode = OutcomeRefuse, ReasonInvalidEvidence
		base.Detail = "claim-check grading did not return net-true"
	}
	return base
}

func validateEvidence(in Input) error {
	missing := []string{}
	add := func(ok bool, name string) {
		if !ok {
			missing = append(missing, name)
		}
	}
	add(strings.TrimSpace(in.Artifact.Recipe) != "", "artifact.recipe")
	add(strings.TrimSpace(in.Baseline.Name) != "", "tuned_baseline.name")
	add(strings.TrimSpace(in.Baseline.Artifact.Format) != "", "tuned_baseline.artifact")
	add(strings.TrimSpace(in.Baseline.Runtime.Name) != "", "tuned_baseline.runtime")
	add(len(in.Baseline.Tuning) > 0, "tuned_baseline.tuning")
	label := norm(in.Provenance.Label)
	add(label == "witnessed" || label == "observed" || label == "modeled", "provenance.label")
	add(strings.TrimSpace(in.Provenance.Source) != "", "provenance.source")
	add(strings.TrimSpace(in.Provenance.CapturedAt) != "", "provenance.captured_at")
	add(strings.TrimSpace(in.Hardware.OS) != "", "hardware_envelope.os")
	add(strings.TrimSpace(in.Hardware.Architecture) != "", "hardware_envelope.architecture")
	add(strings.TrimSpace(in.Hardware.Accelerator) != "", "hardware_envelope.accelerator")
	add(in.Hardware.AcceleratorMemoryBytes > 0 || in.Hardware.HostMemoryBytes > 0, "hardware_envelope.memory")
	add(in.Native.LoadSuccess, "native_quantizer_runtime.load_success")
	add(in.Native.PeakMemoryBytes > 0, "native_quantizer_runtime.peak_memory_bytes")
	add(in.Native.FirstTokenMilliseconds > 0, "native_quantizer_runtime.first_token_ms")
	add(in.Native.OutputTokensPerSecond > 0, "native_quantizer_runtime.output_tokens_per_second")
	add(strings.TrimSpace(in.Native.Quality.Metric) != "", "native_quantizer_runtime.quality.metric")
	add(strings.TrimSpace(in.Native.Quality.Dataset) != "", "native_quantizer_runtime.quality.dataset")
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("missing required benchmark evidence: %s", strings.Join(missing, ", "))
	}
	return nil
}

func Validate(in Input) error {
	r := Evaluate(in)
	if r.Outcome != OutcomeBenchmark {
		return errors.New(r.ReasonCode + ": " + r.Detail)
	}
	return nil
}

func hardwareScope(h HardwareEnvelope) string {
	return fmt.Sprintf("%s/%s; accelerator=%s; accelerator_memory=%d; host_memory=%d; driver=%s; no claim outside this envelope", h.OS, h.Architecture, h.Accelerator, h.AcceleratorMemoryBytes, h.HostMemoryBytes, h.Driver)
}
