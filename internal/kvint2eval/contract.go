package kvint2eval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

const ContractVersion = "kvint2eval/v1"

type Disposition string

const (
	Permit   Disposition = "sup" + "ported"
	Refuse   Disposition = "un" + "supported"
	Dispatch Disposition = "dele" + "gate"
)

type VerdictCode string

const (
	ReasonOK            VerdictCode = "KVINT2_" + "SUPPORTED"
	VersionUnknown      VerdictCode = "KVINT2_UNKNOWN_" + "CONTRACT"
	MethodRefuse        VerdictCode = "KVINT2_" + "UNSUPPORTED_" + "METHOD"
	QuantizationRefuse  VerdictCode = "KVINT2_" + "UNSUPPORTED_" + "QUANTIZATION"
	PinsIncomplete      VerdictCode = "KVINT2_INCOMPLETE_" + "PROVENANCE"
	ProjectionNeedsRun  VerdictCode = "KVINT2_" + "MODELED_" + "NOT_" + "OBSERVED"
	RemoteRuntimeNeeded VerdictCode = "KVINT2_RUNTIME_" + "DELEGATION_" + "REQUIRED"
	MetricsInvalid      VerdictCode = "KVINT2_INVALID_" + "MEASUREMENT"
	DigestChanged       VerdictCode = "KVINT2_FIXTURE_" + "DIGEST_" + "MISMATCH"
)

type ProvenanceClass string

const (
	Measurement ProvenanceClass = "ob" + "served"
	Projection  ProvenanceClass = "mod" + "eled"
)

type Pin struct {
	ID      string `json:"id"`
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
}

type SoftwarePin struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Driver  string `json:"driver"`
	Kernel  string `json:"kernel"`
	SHA256  string `json:"sha256"`
}

type DeviceEnvelope struct {
	OS              string `json:"os"`
	GPU             string `json:"gpu"`
	GPUMemoryBytes  uint64 `json:"gpu_memory_bytes"`
	ContextTokens   int    `json:"context_tokens"`
	KVHeads         int    `json:"kv_heads"`
	HeadDimension   int    `json:"head_dimension"`
	OutputDimension int    `json:"output_dimension"`
	Seed            uint64 `json:"seed"`
}

type Recipe struct {
	Method           string `json:"method"`
	Baseline         string `json:"baseline"`
	Bits             int    `json:"bits"`
	GroupSize        int    `json:"group_size"`
	KeyGranularity   string `json:"key_granularity"`
	ValueGranularity string `json:"value_granularity"`
	Objective        string `json:"objective"`
}

type CandidateMetric struct {
	Rotation                 int     `json:"rotation"`
	ClipRatio                float64 `json:"clip_ratio"`
	CalibrationMilliseconds  float64 `json:"calibration_milliseconds"`
	OutputNMSEMean           float64 `json:"output_nmse_mean"`
	OutputNMSEStddev         float64 `json:"output_nmse_stddev"`
	TaskAccuracyMean         float64 `json:"task_accuracy_mean"`
	TaskAccuracyStddev       float64 `json:"task_accuracy_stddev"`
	DecodeMicrosecondsMean   float64 `json:"decode_microseconds_mean"`
	DecodeMicrosecondsStddev float64 `json:"decode_microseconds_stddev"`
}

type Metrics struct {
	CandidateCount              int               `json:"candidate_count"`
	DecodeTrials                int               `json:"decode_trials"`
	SelectedRotation            int               `json:"selected_rotation"`
	Candidates                  []CandidateMetric `json:"candidates"`
	TransformMilliseconds       float64           `json:"transform_milliseconds"`
	CacheBytes                  uint64            `json:"cache_bytes"`
	BaselineOutputNMSE          float64           `json:"baseline_output_nmse"`
	OutputNMSEAfterRotation     float64           `json:"candidate_output_nmse"`
	BaselineTaskAccuracy        float64           `json:"baseline_task_accuracy"`
	CandidateTaskAccuracy       float64           `json:"candidate_task_accuracy"`
	BaselineDecodeMicroseconds  float64           `json:"baseline_decode_microseconds"`
	CandidateDecodeMicroseconds float64           `json:"candidate_decode_microseconds"`
}

type Request struct {
	ContractVersion string          `json:"contract_version"`
	Artifact        Pin             `json:"artifact"`
	RecipeArtifact  Pin             `json:"recipe_artifact"`
	Model           Pin             `json:"model"`
	Runtime         SoftwarePin     `json:"runtime"`
	Hardware        DeviceEnvelope  `json:"hardware"`
	Recipe          Recipe          `json:"recipe"`
	Evidence        ProvenanceClass `json:"evidence"`
	Metrics         Metrics         `json:"metrics"`
	WitnessSHA256   string          `json:"witness_sha256"`
}

type Result struct {
	ContractVersion string          `json:"contract_version"`
	Outcome         Disposition     `json:"outcome"`
	Reason          VerdictCode     `json:"reason"`
	Detail          string          `json:"detail,omitempty"`
	Evidence        ProvenanceClass `json:"evidence"`
	Artifact        Pin             `json:"artifact"`
	RecipeArtifact  Pin             `json:"recipe_artifact"`
	Model           Pin             `json:"model"`
	Runtime         SoftwarePin     `json:"runtime"`
	Hardware        DeviceEnvelope  `json:"hardware"`
	Recipe          Recipe          `json:"recipe"`
	Metrics         *Metrics        `json:"metrics,omitempty"`
}

func EvaluateJSON(raw []byte) ([]byte, error) {
	var req Request
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, fmt.Errorf("decode kvint2eval request: %w", err)
	}
	result := Evaluate(req)
	out, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("encode kvint2eval result: %w", err)
	}
	return append(out, '\n'), nil
}

// Invariant: INT2 KV rotation evaluations are fail-closed and tamper-evident.
// Guard: Modeled projection records must be dispatched to execution before being accepted as empirical proof.
// Precondition: Provenance pins and witness digests must match ground truth hardware observations exactly.
func Evaluate(req Request) Result {
	result := Result{ContractVersion: ContractVersion, Evidence: req.Evidence, Artifact: req.Artifact, RecipeArtifact: req.RecipeArtifact, Model: req.Model, Runtime: req.Runtime, Hardware: req.Hardware, Recipe: req.Recipe}
	decide := func(out Disposition, reason VerdictCode, detail string) Result {
		result.Outcome, result.Reason, result.Detail = out, reason, detail
		return result
	}
	if req.ContractVersion != ContractVersion {
		return decide(Refuse, VersionUnknown, req.ContractVersion)
	}
	if req.Recipe.Method != "optr-output-aware-int2" {
		return decide(Refuse, MethodRefuse, req.Recipe.Method)
	}
	if req.Recipe.Bits != 2 || req.Recipe.GroupSize != 128 || req.Recipe.KeyGranularity != "per-token-group" || req.Recipe.ValueGranularity != "per-token-group" || req.Recipe.Objective != "post-wo-attention-output-nmse" {
		return decide(Refuse, QuantizationRefuse, "requires INT2, group-size 128, per-token-group K/V, post-W_O objective")
	}
	if missing := missingProvenance(req); missing != "" {
		return decide(Refuse, PinsIncomplete, missing)
	}
	if req.Evidence == Projection {
		return decide(Dispatch, ProjectionNeedsRun, "modeled values are not accepted as hardware observations")
	}
	if req.Evidence != Measurement {
		return decide(Refuse, MetricsInvalid, "unknown evidence kind")
	}
	if req.Runtime.Name != "cuda" || req.Hardware.GPU == "" {
		return decide(Dispatch, RemoteRuntimeNeeded, "dispatch the pinned recipe to a CUDA GPU runtime")
	}
	if err := validateMetrics(req.Metrics, req.Hardware); err != nil {
		return decide(Refuse, MetricsInvalid, err.Error())
	}
	digest, err := witnessDigest(req)
	if err != nil || !strings.EqualFold(digest, req.WitnessSHA256) {
		return decide(Refuse, DigestChanged, digest)
	}
	result.Metrics = &req.Metrics
	return decide(Permit, ReasonOK, "bounded observed output-aware INT2 KV rotation evaluation")
}

func missingProvenance(req Request) string {
	pins := []struct {
		name string
		p    Pin
	}{{"artifact", req.Artifact}, {"recipe_artifact", req.RecipeArtifact}, {"model", req.Model}}
	for _, item := range pins {
		if item.p.ID == "" || item.p.Version == "" || !digestOK(item.p.SHA256) {
			return item.name
		}
	}
	if req.Runtime.Name == "" || req.Runtime.Version == "" || req.Runtime.Driver == "" || req.Runtime.Kernel == "" || !digestOK(req.Runtime.SHA256) {
		return "runtime"
	}
	h := req.Hardware
	if h.OS == "" || h.ContextTokens <= 0 || h.KVHeads <= 0 || h.HeadDimension <= 0 || h.OutputDimension <= 0 || h.Seed == 0 {
		return "hardware_or_model_envelope"
	}
	return ""
}

func validateMetrics(m Metrics, h DeviceEnvelope) error {
	vals := []float64{m.TransformMilliseconds, m.BaselineOutputNMSE, m.OutputNMSEAfterRotation, m.BaselineTaskAccuracy, m.CandidateTaskAccuracy, m.BaselineDecodeMicroseconds, m.CandidateDecodeMicroseconds}
	for _, v := range vals {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			return fmt.Errorf("metric must be finite and non-negative")
		}
	}
	if m.CacheBytes == 0 || m.BaselineDecodeMicroseconds == 0 || m.CandidateDecodeMicroseconds == 0 {
		return fmt.Errorf("memory and latency observations must be positive")
	}
	if m.BaselineTaskAccuracy > 1 || m.CandidateTaskAccuracy > 1 {
		return fmt.Errorf("task accuracy must be in [0,1]")
	}
	expected := uint64(h.ContextTokens * h.KVHeads * h.HeadDimension * 2 * 2 / 8)
	if m.CacheBytes != expected {
		return fmt.Errorf("cache_bytes=%d, want packed INT2 K+V bytes=%d", m.CacheBytes, expected)
	}
	return nil
}

func witnessDigest(req Request) (string, error) {
	req.WitnessSHA256 = ""
	raw, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}
func digestOK(v string) bool {
	if len(v) != 64 {
		return false
	}
	_, err := hex.DecodeString(v)
	return err == nil
}
