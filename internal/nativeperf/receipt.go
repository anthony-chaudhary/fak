package nativeperf

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/systembaseline"
)

const (
	ReceiptSchemaV1 = "fak-native-performance-receipt/v1"
	ReceiptSchemaV2 = "fak-native-performance-receipt/v2"
	ReceiptSchema   = ReceiptSchemaV2
)

const (
	RoleBaseline  = "baseline"
	RoleCandidate = "candidate"
)

// ExperimentReceipt is the portable, scrubbed A/B evidence contract shared by
// Metal and CUDA native-performance experiments.
type ExperimentReceipt struct {
	Schema            string                  `json:"schema"`
	Role              string                  `json:"role"`
	EnvelopeID        string                  `json:"envelope_id"`
	ChangedLeverID    string                  `json:"changed_lever_id"`
	Revision          string                  `json:"revision"`
	ArtifactSHA256    string                  `json:"artifact_sha256"`
	Machine           MachineIdentity         `json:"machine"`
	Controls          ExperimentControls      `json:"controls"`
	UnchangedControls []string                `json:"unchanged_controls"`
	ChangedAxes       []string                `json:"changed_axes"`
	Repetitions       []Repetition            `json:"repetitions"`
	AmbientEvidence   []AmbientEvidence       `json:"ambient_evidence,omitempty"`
	Memory            MemoryMetrics           `json:"memory"`
	Execution         ExecutionIdentity       `json:"execution"`
	Quality           QualityMetric           `json:"quality"`
	Comparison        ComparisonIdentity      `json:"comparison"`
	ModuleVersions    []ModuleRevision        `json:"module_versions"`
	Commands          []string                `json:"commands"`
	ProfilerArtifacts []ArtifactRef           `json:"profiler_artifacts"`
	SystemBaselines   []systembaseline.Report `json:"system_baselines,omitempty"`
}

type MachineIdentity struct {
	ScrubbedID string `json:"scrubbed_id"`
	Platform   string `json:"platform"`
	Backend    string `json:"backend"`
}

type ExperimentControls struct {
	PromptTokens  int     `json:"prompt_tokens"`
	DecodeTokens  int     `json:"decode_tokens"`
	Batch         int     `json:"batch"`
	ContextTokens int     `json:"context_tokens"`
	Temperature   float64 `json:"temperature"`
	Sampling      string  `json:"sampling"`
	CacheState    string  `json:"cache_state"`
	Warmups       int     `json:"warmups"`
	Repetitions   int     `json:"repetitions"`
}

type Repetition struct {
	EndToEndMilliseconds float64 `json:"end_to_end_milliseconds"`
	TokensPerSecond      float64 `json:"tokens_per_second"`
	TTFTMilliseconds     float64 `json:"ttft_milliseconds,omitempty"`
	PrefillMilliseconds  float64 `json:"prefill_milliseconds,omitempty"`
	DecodeMilliseconds   float64 `json:"decode_milliseconds,omitempty"`
	SystemBaselineDigest string  `json:"system_baseline_digest,omitempty"`
}

type MemoryMetrics struct {
	PeakBytes     uint64 `json:"peak_bytes"`
	ResidentBytes uint64 `json:"resident_bytes"`
}

type ExecutionIdentity struct {
	Engine        string `json:"engine"`
	ForwardPath   string `json:"forward_path"`
	FallbackCount int    `json:"fallback_count"`
}

type ArtifactRef struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

// PortableReceipt binds a structurally valid fak-native receipt to the exact
// bytes that were decoded. ReceiptSHA256 identifies the receipt record; the
// model artifact identity remains in Receipt.ArtifactSHA256.
type PortableReceipt struct {
	Receipt             ExperimentReceipt
	ReceiptSHA256       string
	ModelArtifactSHA256 string
}

type Comparison struct {
	Schema                  string  `json:"schema"`
	EnvelopeID              string  `json:"envelope_id"`
	ChangedLeverID          string  `json:"changed_lever_id"`
	CriterionDigest         string  `json:"criterion_digest"`
	BaselineRevision        string  `json:"baseline_revision"`
	CandidateRevision       string  `json:"candidate_revision"`
	BaselineMeanTokensPerS  float64 `json:"baseline_mean_tokens_per_second"`
	CandidateMeanTokensPerS float64 `json:"candidate_mean_tokens_per_second"`
	DeltaTokensPerS         float64 `json:"delta_tokens_per_second"`
	DeltaPercent            float64 `json:"delta_percent"`
}

var requiredControlNames = []string{"artifact_sha256", "machine", "prompt_tokens", "decode_tokens", "batch", "context_tokens", "temperature", "sampling", "cache_state", "warmups", "repetitions", "execution_identity"}

// BaselineTemplate returns a deterministic pre-change capture skeleton. Values
// marked FILL must be replaced by the capture command before comparison.
func BaselineTemplate(graph Graph, leverID string) (ExperimentReceipt, error) {
	lever, envelope, err := findLeverEnvelope(graph, leverID)
	if err != nil {
		return ExperimentReceipt{}, err
	}
	reps := make([]Repetition, envelope.Repetitions)
	return ExperimentReceipt{
		Schema: ReceiptSchema, Role: RoleBaseline, EnvelopeID: envelope.ID, ChangedLeverID: lever.ID,
		Revision: "FILL_COMMITTED_REVISION", ArtifactSHA256: envelope.ArtifactSHA256,
		Machine:           MachineIdentity{ScrubbedID: "FILL_SCRUBBED_MACHINE_ID", Platform: lever.Applicability.Platform, Backend: envelope.Backend},
		Controls:          ExperimentControls{PromptTokens: envelope.PromptTokens, DecodeTokens: envelope.DecodeTokens, Batch: 1, ContextTokens: envelope.PromptTokens + envelope.DecodeTokens, Temperature: float64(envelope.Temperature), Sampling: "greedy", CacheState: "cold", Warmups: 1, Repetitions: envelope.Repetitions},
		UnchangedControls: append([]string(nil), requiredControlNames...), ChangedAxes: []string{}, Repetitions: reps,
		Execution:      ExecutionIdentity{Engine: "fak-native", ForwardPath: envelope.ForwardPath},
		Quality:        QualityMetric{Name: "FILL_QUALITY_METRIC", Score: 0, HigherIsBetter: true},
		ModuleVersions: []ModuleRevision{{Module: "FILL_MODULE", Revision: "FILL_MODULE_REVISION"}},
		Commands:       []string{"FILL_SCRUBBED_CAPTURE_COMMAND"}, ProfilerArtifacts: []ArtifactRef{{Path: "FILL_RELATIVE_PROFILE_PATH", SHA256: strings.Repeat("0", 64)}},
	}, nil
}

func DecodeReceipt(data []byte) (ExperimentReceipt, error) {
	var r ExperimentReceipt
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&r); err != nil {
		return r, fmt.Errorf("decode receipt: %w", err)
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return r, errors.New("decode receipt: multiple JSON values")
		}
		return r, fmt.Errorf("decode receipt trailing data: %w", err)
	}
	return r, nil
}

// AttachSystemBaseline appends the next aggregate attestation without allowing
// callers to skip or overfill the receipt's one-per-repetition sequence.
// DecodePortableReceipt validates a receipt without consulting ActiveGraph.
// It applies every graph-independent receipt invariant and additionally
// requires a fak-native execution with no fallback runtime.
func DecodePortableReceipt(data []byte) (PortableReceipt, error) {
	r, err := DecodeReceipt(data)
	if err != nil {
		return PortableReceipt{}, err
	}
	problems := receiptValidationProblems(r)
	if r.Execution.Engine != "fak-native" {
		problems = append(problems, `execution engine must be exactly "fak-native"`)
	}
	if r.Execution.FallbackCount != 0 {
		problems = append(problems, "fallback count must be zero")
	}
	if err := receiptValidationError(problems); err != nil {
		return PortableReceipt{}, err
	}
	digest := sha256.Sum256(data)
	return PortableReceipt{
		Receipt:             r,
		ReceiptSHA256:       hex.EncodeToString(digest[:]),
		ModelArtifactSHA256: r.ArtifactSHA256,
	}, nil
}

func AttachSystemBaseline(graph Graph, receipt ExperimentReceipt, attestation systembaseline.Report) (ExperimentReceipt, error) {
	if len(receipt.AmbientEvidence) > 0 {
		return ExperimentReceipt{}, errors.New("receipt already carries legacy ambient evidence")
	}
	if receipt.Schema == ReceiptSchemaV1 {
		receipt.Schema = ReceiptSchemaV2
	}
	if err := ValidateReceipt(graph, receipt); err != nil {
		return ExperimentReceipt{}, err
	}
	if err := attestation.Validate(); err != nil {
		return ExperimentReceipt{}, fmt.Errorf("system baseline: %w", err)
	}
	if len(attestation.TopNonSUT) > 0 {
		return ExperimentReceipt{}, errors.New("system baseline contains high-cardinality process identities")
	}
	if len(receipt.SystemBaselines) >= len(receipt.Repetitions) {
		return ExperimentReceipt{}, errors.New("system baselines already cover every repetition")
	}
	receipt.SystemBaselines = append(receipt.SystemBaselines, attestation)
	receipt.Repetitions[len(receipt.SystemBaselines)-1].SystemBaselineDigest = attestation.Digest
	if err := ValidateReceipt(graph, receipt); err != nil {
		return ExperimentReceipt{}, err
	}
	return receipt, nil
}

func ValidateReceipt(graph Graph, r ExperimentReceipt) error {
	f := receiptValidationProblems(r)
	lever, env, err := findLeverEnvelope(graph, r.ChangedLeverID)
	if err != nil {
		f = append(f, err.Error())
	} else {
		if r.EnvelopeID != env.ID {
			f = append(f, "changed lever does not belong to receipt envelope")
		}
		if r.Machine.Platform != lever.Applicability.Platform || r.Machine.Backend != env.Backend {
			f = append(f, "machine platform/backend does not match envelope")
		}
		if r.ArtifactSHA256 != env.ArtifactSHA256 {
			f = append(f, "artifact hash does not match envelope")
		}
	}
	return receiptValidationError(f)
}

func receiptValidationProblems(r ExperimentReceipt) []string {
	var f []string
	if r.Schema != ReceiptSchemaV1 && r.Schema != ReceiptSchemaV2 {
		f = append(f, fmt.Sprintf("schema must be %q or %q", ReceiptSchemaV1, ReceiptSchemaV2))
	}
	if r.Schema == ReceiptSchemaV1 && len(r.SystemBaselines) > 0 {
		f = append(f, "v1 receipts cannot carry system baseline attestations")
	}
	if r.Schema == ReceiptSchemaV1 {
		for i, repetition := range r.Repetitions {
			if repetition.SystemBaselineDigest != "" {
				f = append(f, fmt.Sprintf("v1 repetition %d cannot bind system baseline evidence", i))
			}
		}
	}
	if r.Role != RoleBaseline && r.Role != RoleCandidate {
		f = append(f, "role must be baseline or candidate")
	}
	if strings.TrimSpace(r.Revision) == "" || strings.HasPrefix(r.Revision, "FILL_") {
		f = append(f, "revision is missing")
	}
	if !scrubbed(r.Machine.ScrubbedID) {
		f = append(f, "machine identity is empty or contains private path/host syntax")
	}
	if strings.TrimSpace(r.Execution.Engine) == "" || strings.TrimSpace(r.Execution.ForwardPath) == "" {
		f = append(f, "execution identity must name an engine and a forward path")
	} else if r.Execution.Engine != "fak-native" {
		f = append(f, "execution engine must be fak-native")
	}
	if r.Execution.FallbackCount != 0 {
		f = append(f, "fallback count must be zero")
	}
	if r.Schema != ReceiptSchemaV1 || r.Comparison != (ComparisonIdentity{}) {
		if err := validateComparisonIdentity(r); err != nil {
			f = append(f, err.Error())
		}
	}
	if strings.TrimSpace(r.Quality.Name) == "" || strings.HasPrefix(r.Quality.Name, "FILL_") || math.IsNaN(r.Quality.Score) || math.IsInf(r.Quality.Score, 0) {
		f = append(f, "quality metric must be captured and finite")
	}
	if len(r.ModuleVersions) == 0 {
		f = append(f, "at least one module revision is required")
	}
	seenModules := map[string]bool{}
	for _, m := range r.ModuleVersions {
		if strings.TrimSpace(m.Module) == "" || strings.HasPrefix(m.Module, "FILL_") || strings.TrimSpace(m.Revision) == "" || strings.HasPrefix(m.Revision, "FILL_") || seenModules[m.Module] {
			f = append(f, "module revisions must be unique captured module@rev identities")
		}
		seenModules[m.Module] = true
	}
	if r.Controls.PromptTokens <= 0 || r.Controls.DecodeTokens <= 0 || r.Controls.Batch <= 0 || r.Controls.ContextTokens <= 0 || r.Controls.Warmups < 0 || r.Controls.Repetitions <= 0 {
		f = append(f, "controls contain invalid dimensions")
	}
	if len(r.Repetitions) != r.Controls.Repetitions {
		f = append(f, "repetition count does not match controls")
	}
	for i, rep := range r.Repetitions {
		if !positive(rep.EndToEndMilliseconds) || !positive(rep.TokensPerSecond) {
			f = append(f, fmt.Sprintf("repetition %d lacks positive end-to-end latency/tok/s", i))
		}
	}
	if len(r.AmbientEvidence) > 0 && len(r.SystemBaselines) > 0 {
		f = append(f, "receipt cannot carry both legacy ambient evidence and system baseline attestations")
	}
	if len(r.AmbientEvidence) > 0 {
		if len(r.AmbientEvidence) != len(r.Repetitions) {
			f = append(f, fmt.Sprintf("ambient evidence count %d must align 1:1 with %d repetitions", len(r.AmbientEvidence), len(r.Repetitions)))
		} else {
			for i, evidence := range r.AmbientEvidence {
				if err := ValidateAmbientEvidence(evidence); err != nil {
					f = append(f, fmt.Sprintf("ambient evidence %d: %v", i, err))
				}
			}
		}
	}
	seenBaselineDigests := map[string]bool{}
	for i := range r.SystemBaselines {
		if err := r.SystemBaselines[i].Validate(); err != nil {
			f = append(f, fmt.Sprintf("system baseline %d is invalid: %v", i, err))
		}
		if len(r.SystemBaselines[i].TopNonSUT) > 0 {
			f = append(f, fmt.Sprintf("system baseline %d contains high-cardinality process identities", i))
		}
		if seenBaselineDigests[r.SystemBaselines[i].Digest] {
			f = append(f, fmt.Sprintf("system baseline %d reuses an attestation digest", i))
		}
		seenBaselineDigests[r.SystemBaselines[i].Digest] = true
	}
	if len(r.SystemBaselines) > len(r.Repetitions) {
		f = append(f, "system baselines exceed repetition count")
	}
	for i, repetition := range r.Repetitions {
		if i < len(r.SystemBaselines) {
			if repetition.SystemBaselineDigest != r.SystemBaselines[i].Digest {
				f = append(f, fmt.Sprintf("repetition %d system baseline binding does not match embedded attestation", i))
			}
		} else if repetition.SystemBaselineDigest != "" {
			f = append(f, fmt.Sprintf("repetition %d has a system baseline binding hole", i))
		}
	}
	if r.Memory.PeakBytes == 0 || r.Memory.ResidentBytes == 0 {
		f = append(f, "peak and resident memory are required")
	}
	if !sameStrings(r.UnchangedControls, requiredControlNames) {
		f = append(f, "unchanged_controls must contain the complete canonical control list")
	}
	wantAxes := []string{}
	if r.Role == RoleCandidate {
		wantAxes = []string{"lever:" + r.ChangedLeverID}
	}
	if !sameStrings(r.ChangedAxes, wantAxes) {
		f = append(f, "changed_axes must declare exactly the candidate lever and no other axis")
	}
	if len(r.Commands) == 0 {
		f = append(f, "at least one scrubbed command is required")
	}
	for _, command := range r.Commands {
		if private(command) || strings.HasPrefix(command, "FILL_") {
			f = append(f, "command contains private or placeholder details")
		}
	}
	if len(r.ProfilerArtifacts) == 0 {
		f = append(f, "at least one profiler artifact is required")
	}
	for _, a := range r.ProfilerArtifacts {
		if private(a.Path) || strings.HasPrefix(a.Path, "FILL_") || !validSHA256(a.SHA256) {
			f = append(f, "profiler artifact must use a relative scrubbed path and SHA-256")
		}
	}
	return f
}

func receiptValidationError(problems []string) error {
	if len(problems) == 0 {
		return nil
	}
	sort.Strings(problems)
	return fmt.Errorf("invalid native-performance receipt: %s", strings.Join(problems, "; "))
}

func CompareReceipts(graph Graph, baseline, candidate ExperimentReceipt) (Comparison, error) {
	if err := ValidateReceipt(graph, baseline); err != nil {
		return Comparison{}, fmt.Errorf("baseline: %w", err)
	}
	if err := ValidateReceipt(graph, candidate); err != nil {
		return Comparison{}, fmt.Errorf("candidate: %w", err)
	}
	if baseline.Role != RoleBaseline || candidate.Role != RoleCandidate {
		return Comparison{}, fmt.Errorf("comparison requires baseline then candidate roles")
	}
	if baseline.EnvelopeID != candidate.EnvelopeID || baseline.ChangedLeverID != candidate.ChangedLeverID {
		return Comparison{}, fmt.Errorf("receipts target different envelope or lever")
	}
	if baseline.ArtifactSHA256 != candidate.ArtifactSHA256 || baseline.Machine != candidate.Machine || baseline.Controls != candidate.Controls || baseline.Execution != candidate.Execution || baseline.Comparison != candidate.Comparison || !sameStrings(baseline.UnchangedControls, candidate.UnchangedControls) {
		return Comparison{}, fmt.Errorf("receipts are incomparable: an undeclared control axis drifted")
	}
	b, c := meanTPS(baseline.Repetitions), meanTPS(candidate.Repetitions)
	return Comparison{Schema: "fak-native-performance-comparison/v1", EnvelopeID: baseline.EnvelopeID, ChangedLeverID: baseline.ChangedLeverID, CriterionDigest: baseline.Comparison.CriterionDigest, BaselineRevision: baseline.Revision, CandidateRevision: candidate.Revision, BaselineMeanTokensPerS: b, CandidateMeanTokensPerS: c, DeltaTokensPerS: c - b, DeltaPercent: (c - b) / b * 100}, nil
}

func findLeverEnvelope(graph Graph, leverID string) (Lever, Envelope, error) {
	for _, l := range graph.Levers {
		if l.ID == leverID {
			for _, e := range graph.Envelopes {
				if e.ID == l.Applicability.EnvelopeID {
					return l, e, nil
				}
			}
			return Lever{}, Envelope{}, fmt.Errorf("lever %q has no envelope", leverID)
		}
	}
	return Lever{}, Envelope{}, fmt.Errorf("unknown lever %q", leverID)
}
func positive(v float64) bool { return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0) }
func meanTPS(rs []Repetition) float64 {
	var n float64
	for _, r := range rs {
		n += r.TokensPerSecond
	}
	return n / float64(len(rs))
}
func validSHA256(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, r := range s {
		if !strings.ContainsRune("0123456789abcdef", r) {
			return false
		}
	}
	return true
}
func private(s string) bool {
	l := strings.ToLower(s)
	return strings.Contains(s, "\\") || strings.HasPrefix(s, "/") || strings.Contains(l, "/users/") || strings.Contains(l, "/home/") || strings.Contains(s, "@") || strings.Contains(l, "ssh ")
}
func scrubbed(s string) bool {
	return strings.TrimSpace(s) != "" && !private(s) && !strings.Contains(s, ":")
}
func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	aa, bb := append([]string(nil), a...), append([]string(nil), b...)
	sort.Strings(aa)
	sort.Strings(bb)
	for i := range aa {
		if aa[i] != bb[i] {
			return false
		}
	}
	return true
}
