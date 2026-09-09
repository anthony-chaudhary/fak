// Package rawdecode owns the fak-native raw greedy-decode execution path.
//
// It deliberately returns observations, not physical provenance. Source,
// running-binary, device, memory, and dispatch-counter identity belong to the
// runner that can actually observe them and are not accepted as inputs here.
package rawdecode

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/qwen38quantrun"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// Request is the complete, explicit input to one real raw-decode execution.
// ExpectedArtifactSHA256 is mandatory and is checked from opened bytes before
// model loading or backend/session construction.
type Request struct {
	ArtifactPath           string
	ExpectedArtifactSHA256 string
	ModelName              string
	BackendName            string
	PromptTokenIDs         []int
	ContextLimit           int
	GeneratedTokenLimit    int
	Repetitions            int
	IgnoreEOS              bool
	VerifyCPU              bool
	Quant                  bool
	Lean                   bool
	Q4K                    bool
	StreamQ4K              bool
	Metal                  bool
	Q4KGateUpOutputSlab    bool
	VulkanQ4KProfile       bool
	VulkanStageQ4K         bool
	RequireNonReference    bool
	LoadProfiler           *ggufload.LoadProfiler
}

// Step is one observed greedy-selection result.
type Step struct {
	Step    int     `json:"step"`
	TokenID int     `json:"token_id"`
	Top1    float32 `json:"top1"`
	Top2    float32 `json:"top2"`
	Margin  float32 `json:"margin"`
}

// VerificationStep compares one actually observed candidate output with a
// fresh CPU replay of the same token history.
type VerificationStep struct {
	Step        int     `json:"step"`
	DeviceToken int     `json:"device_token"`
	CPUToken    int     `json:"cpu_token"`
	Agree       bool    `json:"agree"`
	Cosine      float64 `json:"cosine"`
	MaxDelta    float64 `json:"max_delta"`
}

// CPUVerification contains only values observed from the CPU replay.
type CPUVerification struct {
	Passed         bool               `json:"passed"`
	AllAgree       bool               `json:"all_agree"`
	AllArgmaxAgree bool               `json:"all_argmax_agree"`
	MinCosine      float64            `json:"min_cosine"`
	MaxDelta       float64            `json:"max_delta"`
	Prefill        VerificationStep   `json:"prefill"`
	Steps          []VerificationStep `json:"steps,omitempty"`
}

// GenerationObservation is an opaque account of the token-generation boundary
// observed by the real decode loop. Its state cannot be populated by callers;
// zero values are unavailable. It authenticates generation policy and accepted
// token IDs only; timing/resource authority has its own opaque observation.
type GenerationObservation struct {
	observed       bool
	executionSeal  *generationExecutionSeal
	repetition     int
	ignoreEOS      bool
	eosStopped     bool
	outputTokenIDs []int
}

// TimingObservation is an opaque account of the five timed phases and backend
// resource window observed around one real repetition. It is software evidence
// only: it grants no receipt, hardware, performance, comparison, or win credit.
type TimingObservation struct {
	observed         bool
	executionSeal    *generationExecutionSeal
	repetition       int
	boundaries       [6]time.Time
	backendExecution compute.BackendExecutionObservation
}

// OutputTextObservation is the artifact-owned decoding of the exact token IDs
// accepted by one real repetition. It is software evidence only and grants no
// receipt, hardware, quality, throughput, performance, comparison, or win credit.
type OutputTextObservation struct {
	observed       bool
	executionSeal  *generationExecutionSeal
	repetition     int
	outputTokenIDs []int
	text           string
}

// Text returns the exact non-empty UTF-8 text decoded from the artifact tokenizer.
func (o OutputTextObservation) Text() (string, bool) {
	if !o.observed {
		return "", false
	}
	return o.text, true
}

// SessionSetupDuration reports the runner-observed candidate-session setup.
func (o TimingObservation) SessionSetupDuration() time.Duration {
	return o.boundaries[1].Sub(o.boundaries[0])
}

// PrefillDuration reports the runner-observed prompt prefill.
func (o TimingObservation) PrefillDuration() time.Duration {
	return o.boundaries[2].Sub(o.boundaries[1])
}

// FirstSampleDuration reports the runner-observed first greedy selection.
func (o TimingObservation) FirstSampleDuration() time.Duration {
	return o.boundaries[3].Sub(o.boundaries[2])
}

// DecodeDuration reports the runner-observed remaining decode loop.
func (o TimingObservation) DecodeDuration() time.Duration {
	return o.boundaries[4].Sub(o.boundaries[3])
}

// TeardownDuration reports the runner-observed candidate-session teardown.
func (o TimingObservation) TeardownDuration() time.Duration {
	return o.boundaries[5].Sub(o.boundaries[4])
}

// BackendExecution returns the complete backend-owned resource window.
func (o TimingObservation) BackendExecution() (compute.BackendExecutionObservation, bool) {
	if !o.observed {
		return compute.BackendExecutionObservation{}, false
	}
	return o.backendExecution, true
}

type generationExecutionSeal struct {
	repetitions int
	bound       bool
	binding     generationExecutionBinding
}

type generationExecutionBinding struct {
	promptTokenIDs        []int
	contextLimit          int
	generatedLimit        int
	ignoreEOS             bool
	modelName             string
	modelConfigSHA256     [sha256.Size]byte
	modelConfigValid      bool
	artifactPath          string
	artifactSHA256        string
	tensorInventorySHA256 string
	tokenizerSHA256       string
	templateSHA256        string
	quantization          string
}

// IgnoreEOS reports the EOS policy observed by the real decode loop.
func (o GenerationObservation) IgnoreEOS() (bool, bool) { return o.ignoreEOS, o.observed }

// EOSStopped reports whether the real decode loop stopped on EOS.
func (o GenerationObservation) EOSStopped() (bool, bool) { return o.eosStopped, o.observed }

// ActualGeneratedTokens reports the number of token IDs accepted by the real
// decode loop.
func (o GenerationObservation) ActualGeneratedTokens() (int, bool) {
	return len(o.outputTokenIDs), o.observed
}

// OutputTokenIDs returns a clone of the IDs accepted by the real decode loop.
func (o GenerationObservation) OutputTokenIDs() ([]int, bool) {
	if !o.observed {
		return nil, false
	}
	return slices.Clone(o.outputTokenIDs), true
}

// Run is one observed candidate repetition.
type Run struct {
	SessionSetupDuration time.Duration
	PrefillDuration      time.Duration
	FirstSampleDuration  time.Duration
	DecodeDuration       time.Duration
	TeardownDuration     time.Duration
	CPUVerifyDuration    time.Duration
	PrefillOutputID      int
	StepTokens           []int
	GeneratedTokens      []int
	EOSStopped           bool
	generation           GenerationObservation
	timing               TimingObservation
	outputText           OutputTextObservation
	Steps                []Step
	CPUVerification      *CPUVerification
	BackendExecution     *compute.BackendExecutionObservation
	HostEnvironment      *compute.VulkanHostEnvironment
}

type selectedTokenLogprobRunEvidence struct {
	generatedTokenIDs []int
	steps             []Step
	logprobs          []float64
}

type selectedTokenLogprobEvidence struct {
	promptTokenIDs []int
	generatedLimit int
	repetitions    int
	runs           []selectedTokenLogprobRunEvidence
}

// BackendObservation describes the backend that was actually resolved. It is
// execution metadata, not device provenance.
type BackendObservation struct {
	Selected           string
	RegisteredBackends []string
	Tier               string
	Class              string
	Caps               compute.Caps
}

// Execution is the device-free observation produced by the real orchestration
// path. It intentionally has no source, binary, device, memory, or counter fields.
type Execution struct {
	ArtifactPath          string
	ArtifactSHA256        string
	TensorInventorySHA256 string
	TokenizerSHA256       string
	TemplateSHA256        string
	Quantization          string
	ModelName             string
	ModelConfig           model.Config
	Engine                string
	Precision             string
	Backend               BackendObservation
	LoadDuration          time.Duration
	QuantDuration         time.Duration
	PromptTokenIDs        []int
	ContextLimit          int
	GeneratedLimit        int
	IgnoreEOS             bool
	FiniteLogits          bool
	CPUModelParity        *bool
	Runs                  []Run

	selectedTokenLogprobs *selectedTokenLogprobEvidence
	generationSeal        *generationExecutionSeal
}

func generationBinding(execution Execution) generationExecutionBinding {
	modelConfigSHA256, modelConfigValid := modelConfigDigest(execution.ModelConfig)
	return generationExecutionBinding{
		promptTokenIDs: slices.Clone(execution.PromptTokenIDs), contextLimit: execution.ContextLimit,
		generatedLimit: execution.GeneratedLimit, ignoreEOS: execution.IgnoreEOS,
		modelName: execution.ModelName, modelConfigSHA256: modelConfigSHA256, modelConfigValid: modelConfigValid, artifactPath: execution.ArtifactPath,
		artifactSHA256: execution.ArtifactSHA256, tensorInventorySHA256: execution.TensorInventorySHA256,
		tokenizerSHA256: execution.TokenizerSHA256, templateSHA256: execution.TemplateSHA256,
		quantization: execution.Quantization,
	}
}

func (binding generationExecutionBinding) matches(execution Execution) bool {
	modelConfigSHA256, modelConfigValid := modelConfigDigest(execution.ModelConfig)
	return slices.Equal(binding.promptTokenIDs, execution.PromptTokenIDs) &&
		binding.contextLimit == execution.ContextLimit && binding.generatedLimit == execution.GeneratedLimit &&
		binding.ignoreEOS == execution.IgnoreEOS && binding.modelName == execution.ModelName &&
		binding.modelConfigValid && modelConfigValid && binding.modelConfigSHA256 == modelConfigSHA256 &&
		binding.artifactPath == execution.ArtifactPath && binding.artifactSHA256 == execution.ArtifactSHA256 &&
		binding.tensorInventorySHA256 == execution.TensorInventorySHA256 && binding.tokenizerSHA256 == execution.TokenizerSHA256 &&
		binding.templateSHA256 == execution.TemplateSHA256 && binding.quantization == execution.Quantization
}

func modelConfigDigest(config model.Config) ([sha256.Size]byte, bool) {
	// residualHook is intentionally unexported and callback identity has no
	// stable content representation. An enabled hook can change execution, so
	// refuse authority instead of binding a process-local function address.
	if config.EnableResidualHook {
		return [sha256.Size]byte{}, false
	}
	// Config's JSON form is content-canonical for its string-keyed maps and
	// dereferences nested pointers. Include the five intentionally JSON-hidden
	// model-identity fields explicitly so the seal covers the complete config.
	canonical, err := json.Marshal(struct {
		Config        model.Config        `json:"config"`
		GLM5Next      bool                `json:"glm5_next"`
		Name          string              `json:"name"`
		EOSTokenID    int                 `json:"eos_token_id"`
		EOSTokenIDs   []int               `json:"eos_token_ids"`
		BlockTopology model.BlockTopology `json:"block_topology"`
	}{
		Config: config, GLM5Next: config.GLM5Next, Name: config.Name,
		EOSTokenID: config.EOSTokenID, EOSTokenIDs: config.EOSTokenIDs,
		BlockTopology: config.BlockTopology,
	})
	if err != nil {
		return [sha256.Size]byte{}, false
	}
	return sha256.Sum256(canonical), true
}

func (execution *Execution) sealGenerationBinding() {
	if execution.generationSeal == nil || execution.generationSeal.bound {
		return
	}
	execution.generationSeal.binding = generationBinding(*execution)
	execution.generationSeal.bound = true
}

// GenerationObservation returns the runner-owned observation for one exact
// repetition after validating its private execution seal, ordinal, and binding.
// The seal authenticates generation policy and accepted token IDs, not timing.
func (execution Execution) GenerationObservation(repetition int) (GenerationObservation, bool) {
	seal := execution.generationSeal
	if seal == nil || !seal.bound || seal.repetitions != len(execution.Runs) || !seal.binding.matches(execution) || repetition < 0 || repetition >= len(execution.Runs) {
		return GenerationObservation{}, false
	}
	observed := execution.Runs[repetition].generation
	if !observed.observed || observed.executionSeal != seal || observed.repetition != repetition {
		return GenerationObservation{}, false
	}
	publicRun := execution.Runs[repetition]
	if len(observed.outputTokenIDs) == 0 || publicRun.EOSStopped != observed.eosStopped ||
		!slices.Equal(publicRun.GeneratedTokens, observed.outputTokenIDs) ||
		publicRun.PrefillOutputID != observed.outputTokenIDs[0] ||
		!slices.Equal(publicRun.StepTokens, observed.outputTokenIDs[1:]) ||
		len(publicRun.Steps) != len(observed.outputTokenIDs) {
		return GenerationObservation{}, false
	}
	for stepIndex, step := range publicRun.Steps {
		if step.Step != stepIndex || step.TokenID != observed.outputTokenIDs[stepIndex] {
			return GenerationObservation{}, false
		}
	}
	return observed, true
}

// TimingObservation returns the runner-owned timing/resource observation for
// one exact repetition after validating its execution seal, ordinal, public
// aliases, monotonic boundaries, and complete backend resource window.
func (execution Execution) TimingObservation(repetition int) (TimingObservation, bool) {
	if _, ok := execution.GenerationObservation(repetition); !ok {
		return TimingObservation{}, false
	}
	seal := execution.generationSeal
	run := execution.Runs[repetition]
	observed := run.timing
	if !observed.observed || observed.executionSeal != seal || observed.repetition != repetition ||
		execution.Backend.Selected != observed.backendExecution.Identity.Backend ||
		!monotonicTimingBoundaries(observed.boundaries) || !completeBackendExecutionObservation(observed.backendExecution) {
		return TimingObservation{}, false
	}
	if run.SessionSetupDuration != observed.SessionSetupDuration() || run.PrefillDuration != observed.PrefillDuration() ||
		run.FirstSampleDuration != observed.FirstSampleDuration() || run.DecodeDuration != observed.DecodeDuration() ||
		run.TeardownDuration != observed.TeardownDuration() || run.BackendExecution == nil || *run.BackendExecution != observed.backendExecution {
		return TimingObservation{}, false
	}
	return observed, true
}

// OutputTextObservation returns the runner-owned artifact decoding for one
// exact repetition after revalidating generation, execution binding, and ordinal.
func (execution Execution) OutputTextObservation(repetition int) (OutputTextObservation, bool) {
	generation, ok := execution.GenerationObservation(repetition)
	if !ok {
		return OutputTextObservation{}, false
	}
	outputTokenIDs, ok := generation.OutputTokenIDs()
	if !ok {
		return OutputTextObservation{}, false
	}
	seal := execution.generationSeal
	observed := execution.Runs[repetition].outputText
	if !observed.observed || observed.executionSeal != seal || observed.repetition != repetition ||
		len(observed.outputTokenIDs) == 0 || !slices.Equal(observed.outputTokenIDs, outputTokenIDs) ||
		observed.text == "" || !utf8.ValidString(observed.text) {
		return OutputTextObservation{}, false
	}
	return observed, true
}

func (execution *Execution) sealOutputText(tokenizer *tokenizer.Tokenizer) {
	if execution == nil || tokenizer == nil {
		return
	}
	for repetition := range execution.Runs {
		generation, ok := execution.GenerationObservation(repetition)
		if !ok {
			continue
		}
		outputTokenIDs, ok := generation.OutputTokenIDs()
		if !ok {
			continue
		}
		text, ok := decodeArtifactOutputText(tokenizer, outputTokenIDs)
		if !ok {
			continue
		}
		execution.Runs[repetition].outputText = OutputTextObservation{
			observed: true, executionSeal: execution.generationSeal, repetition: repetition,
			outputTokenIDs: slices.Clone(outputTokenIDs), text: text,
		}
	}
}

func decodeArtifactOutputText(tokenizer *tokenizer.Tokenizer, outputTokenIDs []int) (string, bool) {
	if tokenizer == nil || len(outputTokenIDs) == 0 {
		return "", false
	}
	for _, id := range outputTokenIDs {
		if id < 0 || id >= tokenizer.Vocab() {
			return "", false
		}
	}
	text, err := tokenizer.Decode(outputTokenIDs)
	if err != nil || text == "" || !utf8.ValidString(text) {
		return "", false
	}
	return text, true
}

func monotonicTimingBoundaries(boundaries [6]time.Time) bool {
	if boundaries[0].IsZero() {
		return false
	}
	for i := 1; i < len(boundaries); i++ {
		if boundaries[i].IsZero() || boundaries[i].Before(boundaries[i-1]) {
			return false
		}
	}
	return true
}

func completeBackendExecutionObservation(observed compute.BackendExecutionObservation) bool {
	identity := observed.Identity
	if strings.TrimSpace(identity.Backend) == "" || strings.TrimSpace(identity.Device) == "" ||
		strings.TrimSpace(identity.Driver) == "" || strings.TrimSpace(identity.Runtime) == "" ||
		!observed.TransferCountersObserved || !observed.DeviceAllocationObserved ||
		observed.DeviceAllocationPeakBytes < observed.DeviceAllocationLiveBytes {
		return false
	}
	counters := observed.Counters
	if (counters.H2DCount == 0) != (counters.H2DBytes == 0) ||
		(counters.D2HCount == 0) != (counters.D2HBytes == 0) ||
		(counters.D2DCopies == 0) != (counters.D2DBytes == 0) {
		return false
	}
	return !observed.DeviceMemoryObserved ||
		observed.DeviceMemoryTotalBytes > 0 && observed.DeviceMemoryFreeBytes <= observed.DeviceMemoryTotalBytes
}

// BindQwen38VulkanSelectedTokenLogprobsV3 binds runner-owned selected-token
// logprobs to an otherwise caller-populated raw Vulkan result. The binding is
// only an integrity-preserving data transfer: it grants no physical,
// comparator, candidate/reference-pair, or performance authority.
//
// Every error returns a zero raw result. The compute receipt builder owns the
// serialized digest and must derive it from the bound observations.
func BindQwen38VulkanSelectedTokenLogprobsV3(execution Execution, raw compute.Qwen38VulkanRawDecodeResult) (compute.Qwen38VulkanRawDecodeResult, error) {
	fail := func(format string, args ...any) (compute.Qwen38VulkanRawDecodeResult, error) {
		return compute.Qwen38VulkanRawDecodeResult{}, fmt.Errorf("raw decode selected-token logprobs: "+format, args...)
	}
	evidence := execution.selectedTokenLogprobs
	if evidence == nil {
		return fail("runner evidence is unavailable")
	}
	if !slices.Equal(execution.PromptTokenIDs, evidence.promptTokenIDs) ||
		execution.GeneratedLimit != evidence.generatedLimit {
		return fail("execution request fields do not match runner evidence")
	}
	if !equalInt32TokenIDs(raw.PromptTokenIDs, evidence.promptTokenIDs) ||
		raw.GeneratedTokenLimit != evidence.generatedLimit {
		return fail("raw request fields do not match runner evidence")
	}
	if evidence.repetitions < 1 || len(evidence.runs) != evidence.repetitions ||
		len(execution.Runs) != evidence.repetitions || len(raw.Runs) != evidence.repetitions {
		return fail("repetition count does not match runner evidence")
	}
	if raw.ReportedRuns != 0 && raw.ReportedRuns != evidence.repetitions {
		return fail("reported repetition count does not match runner evidence")
	}

	boundRuns := slices.Clone(raw.Runs)
	for i, sealedRun := range evidence.runs {
		publicRun := execution.Runs[i]
		rawRun := raw.Runs[i]
		if len(sealedRun.generatedTokenIDs) == 0 || len(sealedRun.logprobs) != len(sealedRun.generatedTokenIDs) {
			return fail("sealed repetition %d is incomplete", i+1)
		}
		if !slices.Equal(publicRun.GeneratedTokens, sealedRun.generatedTokenIDs) ||
			!slices.Equal(publicRun.Steps, sealedRun.steps) ||
			len(publicRun.Steps) != len(sealedRun.generatedTokenIDs) ||
			publicRun.PrefillOutputID != sealedRun.generatedTokenIDs[0] ||
			!slices.Equal(publicRun.StepTokens, sealedRun.generatedTokenIDs[1:]) {
			return fail("public repetition %d does not match runner evidence", i+1)
		}
		for stepIndex, step := range publicRun.Steps {
			if step.Step != stepIndex || step.TokenID != sealedRun.generatedTokenIDs[stepIndex] {
				return fail("public repetition %d step %d does not match runner evidence", i+1, stepIndex)
			}
		}
		if rawRun.Repetition != i+1 || rawRun.GeneratedTokenLimit != evidence.generatedLimit ||
			rawRun.ActualGeneratedTokens != len(sealedRun.generatedTokenIDs) ||
			!equalInt32TokenIDs(rawRun.OutputTokenIDs, sealedRun.generatedTokenIDs) ||
			!equalInt32TokenIDs(raw.OutputTokenIDs, sealedRun.generatedTokenIDs) {
			return fail("raw repetition %d does not match runner evidence", i+1)
		}
		if len(rawRun.SelectedTokenLogprobs) != 0 || rawRun.SelectedTokenLogprobsSHA256 != "" {
			return fail("raw repetition %d already contains selected-token evidence", i+1)
		}
		for stepIndex, logprob := range sealedRun.logprobs {
			if math.IsNaN(logprob) || math.IsInf(logprob, 0) || logprob > 0 {
				return fail("sealed repetition %d logprob %d is invalid", i+1, stepIndex)
			}
		}
		boundRuns[i].SelectedTokenLogprobs = slices.Clone(sealedRun.logprobs)
		boundRuns[i].SelectedTokenLogprobsSHA256 = ""
	}
	raw.Runs = boundRuns
	return raw, nil
}

func equalInt32TokenIDs(got []int32, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if int(got[i]) != want[i] {
			return false
		}
	}
	return true
}

type session interface {
	Prefill([]int) []float32
	Step(int) []float32
	Close()
}

type loadedModel interface {
	Config() model.Config
	IsEOS(int) bool
	NewCandidateSession(compute.Backend, Request) (session, error)
	NewCPUSession(Request) session
	CloseWeights() error
}

type dependencies struct {
	openArtifact    func(string) (io.ReadCloser, error)
	inspectArtifact func(string, func(string) (io.ReadCloser, error)) (artifactObservation, error)
	loadModel       func(context.Context, Request) (loadedModel, string, error)
	resolveBackend  func(Request) (compute.Backend, BackendObservation, error)
	observeHost     func(context.Context, compute.Backend) (compute.VulkanHostEnvironment, error)
	now             func() time.Time
	since           func(time.Time) time.Duration
}

// Execute hashes and loads the selected artifact, resolves the registered
// fak-native backend, and runs the single raw-decode orchestration path.
func Execute(ctx context.Context, req Request) (Execution, error) {
	return defaultDependencies().execute(ctx, req)
}

func defaultDependencies() dependencies {
	return dependencies{
		openArtifact:    func(path string) (io.ReadCloser, error) { return os.Open(path) },
		inspectArtifact: inspectGGUFArtifact,
		loadModel:       loadProductionModel,
		resolveBackend: func(req Request) (compute.Backend, BackendObservation, error) {
			registered := compute.Registered()
			observed := BackendObservation{Selected: "legacy", RegisteredBackends: slices.Clone(registered)}
			if req.BackendName == "" || req.BackendName == "legacy" {
				if req.RequireNonReference {
					return nil, observed, errors.New("raw decode: require-non-reference needs a named compute backend")
				}
				if req.VulkanQ4KProfile || req.VulkanStageQ4K {
					return nil, observed, errors.New("raw decode: Vulkan Q4_K controls require backend vulkan")
				}
				return nil, observed, nil
			}
			be, ok := compute.Lookup(req.BackendName)
			if !ok {
				return nil, observed, fmt.Errorf("raw decode: unknown backend %q (registered: %v)", req.BackendName, registered)
			}
			if (req.Quant || req.Q4K) && !be.Caps().UploadDtype {
				return nil, observed, fmt.Errorf("raw decode: backend %q is f32-only", be.Name())
			}
			if req.RequireNonReference && be.Class() == compute.Reference {
				return nil, observed, fmt.Errorf("raw decode: backend %q is reference-class", be.Name())
			}
			if (req.VulkanQ4KProfile || req.VulkanStageQ4K) && !compute.ConfigureVulkanQ4K(be, req.VulkanQ4KProfile, req.VulkanStageQ4K) {
				return nil, observed, errors.New("raw decode: Vulkan Q4_K controls require backend vulkan")
			}
			observed.Selected = be.Name()
			observed.Tier = be.Tier()
			observed.Class = be.Class().String()
			observed.Caps = be.Caps()
			return be, observed, nil
		},
		observeHost: compute.ObserveVulkanHostEnvironment,
		now:         time.Now,
		since:       time.Since,
	}
}

func (d dependencies) execute(ctx context.Context, req Request) (Execution, error) {
	if err := validateRequest(req); err != nil {
		return Execution{}, err
	}
	artifact := artifactObservation{Path: req.ArtifactPath}
	var inspectErr error
	if d.inspectArtifact != nil {
		artifact, inspectErr = d.inspectArtifact(req.ArtifactPath, d.openArtifact)
	} else {
		artifact.SHA256, inspectErr = hashArtifact(req.ArtifactPath, d.openArtifact)
	}
	if artifact.SHA256 == "" {
		return Execution{}, inspectErr
	}
	if !strings.EqualFold(artifact.SHA256, req.ExpectedArtifactSHA256) {
		return Execution{}, fmt.Errorf("raw decode: artifact SHA-256 mismatch: expected %s, observed %s", strings.ToLower(req.ExpectedArtifactSHA256), artifact.SHA256)
	}
	if inspectErr != nil {
		return Execution{}, inspectErr
	}

	be, backendObserved, err := d.resolveBackend(req)
	if err != nil {
		return Execution{}, err
	}
	if be != nil && backendObserved.Selected != be.Name() {
		return Execution{}, fmt.Errorf("raw decode: resolved backend label %q does not match actual backend %q", backendObserved.Selected, be.Name())
	}
	loadStart := d.now()
	lm, derivedName, err := d.loadModel(ctx, req)
	loadDuration := d.now().Sub(loadStart)
	if err != nil {
		return Execution{}, fmt.Errorf("raw decode: load artifact: %w", err)
	}
	defer lm.CloseWeights() // best-effort process-local resource release after execution

	quantDuration := time.Duration(0)
	if req.Quant && !req.Lean && !req.Q4K {
		qm, ok := lm.(*productionModel)
		if !ok {
			return Execution{}, errors.New("raw decode: loaded model does not support requested Q8 quantization")
		}
		quantStart := d.now()
		qm.model.Quantize()
		quantDuration = d.now().Sub(quantStart)
	}
	requestedModel := strings.TrimSpace(req.ModelName)
	if requestedModel != "" && requestedModel != derivedName && requestedModel != lm.Config().ModelType {
		return Execution{}, fmt.Errorf("raw decode: model selector %q does not match loaded model %q (type %q)", requestedModel, derivedName, lm.Config().ModelType)
	}
	since := d.since
	if since == nil {
		since = func(start time.Time) time.Duration { return d.now().Sub(start) }
	}
	exec, runErr := executeLoaded(ctx, req, lm, be, d.observeHost, d.now, since)
	if runErr != nil && len(exec.Runs) == 0 {
		return Execution{}, runErr
	}
	exec.ArtifactPath = artifact.Path
	exec.ArtifactSHA256 = artifact.SHA256
	exec.TensorInventorySHA256 = artifact.TensorInventorySHA256
	exec.TokenizerSHA256 = artifact.TokenizerSHA256
	exec.TemplateSHA256 = artifact.TemplateSHA256
	exec.Quantization = artifact.Quantization
	exec.ModelName = derivedName
	exec.ModelConfig = lm.Config()
	exec.Backend = backendObserved
	exec.LoadDuration = loadDuration
	exec.QuantDuration = quantDuration
	exec.Engine, exec.Precision = describeEngine(req, be)
	exec.sealGenerationBinding()
	exec.sealOutputText(artifact.outputTokenizer)
	return exec, runErr
}

func validateRequest(req Request) error {
	if strings.TrimSpace(req.ArtifactPath) == "" {
		return errors.New("raw decode: artifact path is required")
	}
	digest := strings.TrimSpace(req.ExpectedArtifactSHA256)
	if len(digest) != sha256.Size*2 {
		return errors.New("raw decode: expected artifact SHA-256 must be 64 hexadecimal characters")
	}
	for _, c := range digest {
		if !strings.ContainsRune("0123456789abcdefABCDEF", c) {
			return errors.New("raw decode: expected artifact SHA-256 must be hexadecimal")
		}
	}
	if len(req.PromptTokenIDs) == 0 {
		return errors.New("raw decode: prompt token IDs must be non-empty")
	}
	for _, id := range req.PromptTokenIDs {
		if id < 0 {
			return fmt.Errorf("raw decode: prompt token ID %d must be non-negative", id)
		}
	}
	if req.ContextLimit < len(req.PromptTokenIDs)+req.GeneratedTokenLimit {
		return fmt.Errorf("raw decode: context limit %d is smaller than prompt plus generated-token limit %d", req.ContextLimit, len(req.PromptTokenIDs)+req.GeneratedTokenLimit)
	}
	if req.GeneratedTokenLimit < 1 || req.Repetitions < 1 {
		return errors.New("raw decode: generated-token limit and repetitions must be positive")
	}
	if req.Q4K && req.Lean {
		return errors.New("raw decode: Q4_K and lean loading are mutually exclusive")
	}
	if req.StreamQ4K && !req.Q4K {
		return errors.New("raw decode: streamed Q4_K requires Q4_K loading")
	}
	return nil
}

func hashArtifact(path string, open func(string) (io.ReadCloser, error)) (string, error) {
	f, err := open(path)
	if err != nil {
		return "", fmt.Errorf("raw decode: open artifact: %w", err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", fmt.Errorf("raw decode: hash artifact: %w", err)
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

type artifactObservation struct {
	Path                  string
	SHA256                string
	TensorInventorySHA256 string
	TokenizerSHA256       string
	TemplateSHA256        string
	Quantization          string
	outputTokenizer       *tokenizer.Tokenizer
}

// inspectGGUFArtifact derives model identity and hashes the exact byte stream
// that supplied the parsed header. The tee includes any bytes Read prefetched,
// then io.Copy hashes the remainder without loading the artifact into memory.
func inspectGGUFArtifact(path string, open func(string) (io.ReadCloser, error)) (artifactObservation, error) {
	f, err := open(path)
	if err != nil {
		return artifactObservation{}, fmt.Errorf("raw decode: open artifact: %w", err)
	}
	defer f.Close()

	h := sha256.New()
	gg, parseErr := ggufload.Read(io.TeeReader(f, h))
	if _, err := io.Copy(h, f); err != nil {
		return artifactObservation{}, fmt.Errorf("raw decode: hash artifact: %w", err)
	}
	observed := artifactObservation{
		Path:   path,
		SHA256: fmt.Sprintf("%x", h.Sum(nil)),
	}
	if parseErr != nil {
		return observed, fmt.Errorf("raw decode: parse artifact header: %w", parseErr)
	}
	quant := ggufload.ClassifyTensorQuant(gg.Tensors)
	observed.TensorInventorySHA256 = gg.CanonicalManifestDigest()
	if identity, err := qwen38quantrun.DerivePromptPacketGGUFIdentityFromFile(gg); err == nil {
		observed.TokenizerSHA256 = identity.TokenizerDigest
		observed.TemplateSHA256 = identity.TemplateDigest
	}
	observed.outputTokenizer = buildArtifactOutputTokenizerFromGGUF(gg)
	observed.Quantization = quant.Recipe
	if observed.Quantization == "" {
		observed.Quantization = quant.Name
	}
	if observed.Quantization == "unknown" {
		return observed, errors.New("raw decode: parsed artifact has no quantization inventory")
	}
	return observed, nil
}

func buildArtifactOutputTokenizer(metadata *ggufload.GGMLTokenizer) *tokenizer.Tokenizer {
	if metadata == nil {
		return nil
	}
	for _, token := range metadata.Tokens {
		if token == "" {
			return nil
		}
	}
	artifactTokenizer, err := tokenizer.FromGGML(metadata.Tokens, metadata.Merges, metadata.TokenTypes, metadata.Pre)
	if err != nil {
		return nil
	}
	return artifactTokenizer
}

func buildArtifactOutputTokenizerFromGGUF(gg *ggufload.File) *tokenizer.Tokenizer {
	if gg == nil {
		return nil
	}
	if value, present := gg.Metadata["tokenizer.ggml.pre"]; present {
		pre, ok := value.Value.(string)
		if value.Type != ggufload.TypeString || !ok || pre == "" {
			return nil
		}
	}
	if value, present := gg.Metadata["tokenizer.ggml.token_type"]; present {
		if value.Type != ggufload.TypeArray {
			return nil
		}
		if _, ok := gg.Int32Array("tokenizer.ggml.token_type"); !ok {
			return nil
		}
	}
	metadata, ok := gg.GGMLTokenizer()
	if !ok {
		return nil
	}
	return buildArtifactOutputTokenizer(metadata)
}

type productionModel struct{ model *model.Model }

func (m *productionModel) Config() model.Config { return m.model.Cfg }
func (m *productionModel) IsEOS(id int) bool    { return m.model.Cfg.IsEOS(id) }
func (m *productionModel) CloseWeights() error  { return m.model.CloseWeights() }
func (m *productionModel) NewCPUSession(req Request) session {
	s := m.model.NewSession()
	applySessionFlags(s, req, false)
	return s
}
func (m *productionModel) NewCandidateSession(be compute.Backend, req Request) (session, error) {
	if be == nil {
		s := m.model.NewSession()
		applySessionFlags(s, req, true)
		return s, nil
	}
	s, err := m.model.NewBackendSessionChecked(be)
	if err != nil {
		return nil, err
	}
	applySessionFlags(s, req, true)
	return s, nil
}

func applySessionFlags(s *model.Session, req Request, candidate bool) {
	s.Quant = req.Quant || req.Lean || req.Q4K
	s.Q4K = req.Q4K
	s.Q4KGateUpOutputSlab = req.Q4KGateUpOutputSlab
	if candidate {
		s.Metal = req.Metal && !req.Q4K
		s.MetalQ4K = req.Metal && req.Q4K
	}
}

func loadProductionModel(ctx context.Context, req Request) (loadedModel, string, error) {
	ws, closer, err := openVerifiedProductionWeights(ctx, req)
	if err != nil {
		return nil, "", err
	}
	retained := false
	defer func() {
		if !retained {
			_ = closer.Close()
		}
	}()
	var (
		m     *model.Model
		label string
	)
	switch {
	case req.Q4K && req.BackendName != "" && req.BackendName != "legacy":
		opts := []ggufload.Q4KLoadOption{ggufload.WithDenseKQuantResident(false)}
		if req.BackendName == "vulkan" {
			opts = append(opts, ggufload.WithDenseQ2KResident(true))
			if q2kEmbeddingEligible(ws.File) {
				opts = append(opts, ggufload.WithQ2KEmbeddingResident(true))
			}
		}
		label = " [gguf-q4k]"
		if req.StreamQ4K {
			opts = append(opts, ggufload.WithStreamedDenseQ4K(true))
			label = " [gguf-q4k-streamed-dense]"
		}
		m, err = ws.QuantModelQ4KProfileOptionsContext(ctx, req.LoadProfiler, opts...)
	case req.Q4K:
		label = " [gguf-q4k]"
		if req.StreamQ4K {
			label = " [gguf-q4k-streamed-dense]"
			m, err = ws.QuantModelQ4KProfileOptionsContext(ctx, req.LoadProfiler, ggufload.WithStreamedDenseQ4K(true))
		} else {
			m, err = ws.QuantModelQ4KContext(ctx)
		}
	case req.Lean:
		label = " [gguf-lean]"
		m, err = ws.QuantModelProfile(req.LoadProfiler)
	default:
		label = " [gguf]"
		m, err = ws.Model()
	}
	if err != nil {
		return nil, "", err
	}
	// Streamed dense weights retain this reader. Model ownership also delays
	// unmapping until the final session releases any borrowed checkpoint data.
	m.SetWeightCloser(closer)
	retained = true
	return &productionModel{model: m}, filepath.Base(req.ArtifactPath) + label, nil
}

// openVerifiedProductionWeights binds parsing, hashing, and every later tensor
// read to one retained reader. Rechecking the pathname after loading would miss
// A -> B -> A swaps; matching this reader's digest to the inspected artifact
// instead makes a pathname replacement unable to substitute different weights.
func openVerifiedProductionWeights(ctx context.Context, req Request) (*ggufload.WeightSource, io.Closer, error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	var reader io.ReaderAt
	var size int64
	var closer io.Closer
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_GGUF_MMAP"))) {
	case "1", "on", "true":
		data, mappedCloser, available, err := model.MmapOpen(req.ArtifactPath)
		if err != nil {
			return nil, nil, err
		}
		if available {
			reader, size, closer = bytes.NewReader(data), int64(len(data)), mappedCloser
		}
	}
	if reader == nil {
		file, err := os.Open(req.ArtifactPath)
		if err != nil {
			return nil, nil, err
		}
		stat, err := file.Stat()
		if err != nil {
			_ = file.Close()
			return nil, nil, err
		}
		reader, size, closer = file, stat.Size(), file
	}
	accepted := false
	defer func() {
		if !accepted {
			_ = closer.Close()
		}
	}()
	// mmap is verified from its actual bytes, not from a separately reopened
	// file. Both mmap and portable ReadAt retain the same reader through teardown.
	stream := io.NewSectionReader(reader, 0, size)
	hash := sha256.New()
	parsed, err := ggufload.Read(io.TeeReader(stream, hash))
	if err != nil {
		return nil, nil, err
	}
	if _, err := io.Copy(hash, stream); err != nil {
		return nil, nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	digest := fmt.Sprintf("%x", hash.Sum(nil))
	if !strings.EqualFold(digest, req.ExpectedArtifactSHA256) {
		return nil, nil, fmt.Errorf("loaded artifact SHA-256 mismatch: expected %s, observed %s", strings.ToLower(req.ExpectedArtifactSHA256), digest)
	}
	// A single expected artifact digest cannot authenticate sibling shards.
	// Fail before model construction instead of reopening unverified siblings.
	if _, present := parsed.Metadata["split.count"]; present {
		count, ok := parsed.Uint64("split.count")
		if !ok || count != 1 {
			return nil, nil, errors.New("raw decode: split artifacts require a digest-bound shard manifest")
		}
	}
	ws, err := ggufload.NewWeightSource(parsed, reader, size)
	if err != nil {
		return nil, nil, err
	}
	accepted = true
	return ws, closer, nil
}

func q2kEmbeddingEligible(parsed *ggufload.File) bool {
	cfg, err := parsed.Config()
	if err != nil || !cfg.IsQwen35Hybrid() || cfg.IsMoE() || cfg.TieWordEmbeddings {
		return false
	}
	var embedding, output bool
	for _, tensor := range parsed.Tensors {
		embedding = embedding || tensor.Name == "token_embd.weight" && tensor.Type == ggufload.TensorQ2_K
		output = output || tensor.Name == "output.weight"
	}
	return embedding && output
}

func describeEngine(req Request, be compute.Backend) (string, string) {
	engine, precision := "fak-in-kernel (pure-Go, parallel matmul + batched prefill GEMM + fdot ILP)", "f32"
	if req.Quant || req.Lean {
		engine, precision = "fak-in-kernel Q8_0 (pure-Go, quantized weights+activations, int8×int8→int32 dot)", "Q8_0"
	}
	if req.Q4K {
		engine, precision = "fak-in-kernel GGUF resident mixed quantization", "GGUF resident mixed quantization"
		if req.Metal {
			engine, precision = "fak-in-kernel Metal Q4_K/Q8 hybrid (raw GGUF Q4_K majority through MetalQ4K; Q8 minority on CPU)", "Q4_K/Q8 resident hybrid + MetalQ4K"
		}
	}
	if be != nil {
		engine = fmt.Sprintf("fak-in-kernel via compute HAL backend %q", be.Name())
		if req.Q4K && req.BackendName == "vulkan" {
			precision = "resident Q4_K/Q2_K + unsupported dense formats converted to Q8"
		}
	}
	return engine, precision
}

// ExecuteModel is the compatibility seam used by modelbench tests and legacy
// non-GGUF command inputs. It executes the same orchestration loop but cannot
// produce artifact identity because it did not open the artifact.
func ExecuteModel(req Request, m *model.Model, be compute.Backend) (Execution, error) {
	return ExecuteModelWithHostObserver(context.Background(), req, m, be, compute.ObserveVulkanHostEnvironment)
}

// ExecuteModelWithHostObserver executes an already-loaded model while allowing
// callers to inject the host observer for deterministic device-free tests.
// Production callers should use ExecuteModel, which installs the strict
// backend-bound Linux/RADV observer.
func ExecuteModelWithHostObserver(ctx context.Context, req Request, m *model.Model, be compute.Backend, observeHost func(context.Context, compute.Backend) (compute.VulkanHostEnvironment, error)) (Execution, error) {
	if m == nil {
		return Execution{}, errors.New("raw decode: nil model")
	}
	pm := &productionModel{model: m}
	exec, err := executeLoaded(ctx, req, pm, be, observeHost, time.Now, time.Since)
	exec.ModelName = m.Cfg.ModelType
	exec.ModelConfig = m.Cfg
	exec.Engine, exec.Precision = describeEngine(req, be)
	exec.Backend.Selected = "legacy"
	if be != nil {
		exec.Backend.Selected = be.Name()
	}
	exec.sealGenerationBinding()
	return exec, err
}

func executeLoaded(ctx context.Context, req Request, m loadedModel, be compute.Backend, observeHost func(context.Context, compute.Backend) (compute.VulkanHostEnvironment, error), now func() time.Time, since func(time.Time) time.Duration) (Execution, error) {
	if len(req.PromptTokenIDs) == 0 || req.GeneratedTokenLimit < 1 || req.Repetitions < 1 || req.ContextLimit < len(req.PromptTokenIDs)+req.GeneratedTokenLimit {
		return Execution{}, errors.New("raw decode: invalid execution request")
	}
	for _, id := range req.PromptTokenIDs {
		if id < 0 || m.Config().VocabSize > 0 && id >= m.Config().VocabSize {
			return Execution{}, fmt.Errorf("raw decode: prompt token ID %d exceeds model vocabulary size %d", id, m.Config().VocabSize)
		}
	}
	generationSeal := &generationExecutionSeal{repetitions: req.Repetitions}
	exec := Execution{
		PromptTokenIDs: slices.Clone(req.PromptTokenIDs), ContextLimit: req.ContextLimit,
		GeneratedLimit: req.GeneratedTokenLimit, IgnoreEOS: req.IgnoreEOS, FiniteLogits: true,
		generationSeal: generationSeal,
	}
	evidence := &selectedTokenLogprobEvidence{
		promptTokenIDs: slices.Clone(req.PromptTokenIDs),
		generatedLimit: req.GeneratedTokenLimit,
		repetitions:    req.Repetitions,
	}
	var verifyErr error
	for rep := 0; rep < req.Repetitions; rep++ {
		window, windowAvailable, err := compute.BeginBackendExecutionObservation(be)
		if err != nil {
			return Execution{}, fmt.Errorf("raw decode: begin backend observation for run %d: %w", rep+1, err)
		}
		run, selectedTokenLogprobs, runErr := executeRun(req, m, be, generationSeal, rep, now, since)
		if windowAvailable {
			backendExecution, endErr := window.End()
			if runErr != nil {
				if endErr != nil {
					return Execution{}, errors.Join(runErr, fmt.Errorf("raw decode: end backend observation for run %d: %w", rep+1, endErr))
				}
				return Execution{}, runErr
			}
			if endErr != nil {
				return Execution{}, fmt.Errorf("raw decode: end backend observation for run %d: %w", rep+1, endErr)
			}
			if !completeBackendExecutionObservation(backendExecution) || backendExecution.Identity.Backend != be.Name() {
				return Execution{}, fmt.Errorf("raw decode: backend observation for run %d is incomplete or malformed", rep+1)
			}
			run.BackendExecution = &backendExecution
			run.timing.backendExecution = backendExecution
			run.timing.observed = true
		} else if runErr != nil {
			return Execution{}, runErr
		}
		if run.BackendExecution != nil && observeHost != nil && run.BackendExecution.Identity.Backend == "vulkan" {
			host, hostErr := observeHost(ctx, be)
			if hostErr == nil && rawDecodeHostMatchesBackend(host, run.BackendExecution.Identity) {
				run.HostEnvironment = &host
			}
		}
		exec.Runs = append(exec.Runs, run)
		evidence.runs = append(evidence.runs, selectedTokenLogprobRunEvidence{
			generatedTokenIDs: slices.Clone(run.GeneratedTokens),
			steps:             slices.Clone(run.Steps),
			logprobs:          slices.Clone(selectedTokenLogprobs),
		})
		if run.CPUVerification != nil && !run.CPUVerification.Passed {
			verifyErr = errors.New("raw decode CPU verification divergence: argmax divergence between device and CPU reference")
			break
		}
	}
	exec.selectedTokenLogprobs = evidence
	if req.VerifyCPU {
		parity := true
		for _, run := range exec.Runs {
			parity = parity && run.CPUVerification != nil && run.CPUVerification.Passed
		}
		exec.CPUModelParity = &parity
	}
	return exec, verifyErr
}

// rawDecodeHostMatchesBackend accepts only the strict #12281 tuple bound to
// the exact backend-owned identity for this repetition. The driver field is
// parsed only for its sealed PCI keys; Mesa identity comes from the observer.
func rawDecodeHostMatchesBackend(host compute.VulkanHostEnvironment, identity compute.BackendRuntimeIdentity) bool {
	if host.OS != "linux" || strings.TrimSpace(host.Arch) == "" || strings.TrimSpace(host.Kernel) == "" ||
		strings.TrimSpace(host.Device) == "" || strings.TrimSpace(host.MesaVersion) == "" ||
		strings.TrimSpace(host.Firmware) == "" || !strings.EqualFold(host.MesaDriver, "radv") ||
		identity.Backend != "vulkan" || host.Device != identity.Device {
		return false
	}
	vendor, device, ok := rawDecodeBackendPCIIdentity(identity.Driver)
	return ok && host.VendorID == vendor && host.DeviceID == device
}

// BoundVulkanHostEnvironment returns the host tuple only when it remains
// exactly bound to the backend-owned identity retained for the same run.
func BoundVulkanHostEnvironment(run Run) (compute.VulkanHostEnvironment, bool) {
	if run.HostEnvironment == nil || run.BackendExecution == nil {
		return compute.VulkanHostEnvironment{}, false
	}
	host := *run.HostEnvironment
	if !rawDecodeHostMatchesBackend(host, run.BackendExecution.Identity) {
		return compute.VulkanHostEnvironment{}, false
	}
	return host, true
}

func rawDecodeBackendPCIIdentity(driver string) (string, string, bool) {
	fields := strings.Fields(driver)
	if len(fields) != 3 {
		return "", "", false
	}
	values := make(map[string]string, len(fields))
	for _, field := range fields {
		key, value, ok := strings.Cut(field, "=")
		if !ok || values[key] != "" || (key != "driver" && key != "vendor" && key != "device") {
			return "", "", false
		}
		values[key] = value
	}
	canonical := func(value string) (string, bool) {
		value = strings.TrimSpace(strings.ToLower(value))
		if !strings.HasPrefix(value, "0x") {
			return "", false
		}
		parsed, err := strconv.ParseUint(strings.TrimPrefix(value, "0x"), 16, 16)
		if err != nil {
			return "", false
		}
		return fmt.Sprintf("0x%04x", parsed), true
	}
	if strings.TrimSpace(values["driver"]) == "" {
		return "", "", false
	}
	vendor, ok := canonical(values["vendor"])
	if !ok {
		return "", "", false
	}
	device, ok := canonical(values["device"])
	return vendor, device, ok
}

func executeRun(req Request, m loadedModel, be compute.Backend, executionSeal *generationExecutionSeal, repetition int, now func() time.Time, since func(time.Time) time.Duration) (Run, []float64, error) {
	start := now()
	s, err := m.NewCandidateSession(be, req)
	if err != nil {
		return Run{}, nil, fmt.Errorf("raw decode: create candidate session: %w", err)
	}
	setupDone := now()
	logits := s.Prefill(req.PromptTokenIDs)
	prefillDone := now()
	token, top1, top2, selectedTokenLogprob, err := greedySelection(logits)
	if err != nil {
		s.Close()
		return Run{}, nil, errors.New("raw decode: prefill produced non-finite logits")
	}
	run := Run{PrefillOutputID: token, GeneratedTokens: []int{token}, Steps: []Step{{Step: 0, TokenID: token, Top1: top1, Top2: top2, Margin: top1 - top2}}}
	selectedTokenLogprobs := []float64{selectedTokenLogprob}
	var candidateLogits [][]float32
	if req.VerifyCPU {
		candidateLogits = append(candidateLogits, slices.Clone(logits))
	}
	sampled := now()
	ignoreEOS := req.IgnoreEOS
	shouldStopOnEOS := func(token int) bool { return m.IsEOS(token) && !ignoreEOS }
	run.EOSStopped = shouldStopOnEOS(token)
	previous := token
	for step := 1; !run.EOSStopped && step < req.GeneratedTokenLimit && len(req.PromptTokenIDs)+len(run.GeneratedTokens) < req.ContextLimit; step++ {
		logits = s.Step(previous)
		token, top1, top2, selectedTokenLogprob, err = greedySelection(logits)
		if err != nil {
			s.Close()
			return Run{}, nil, fmt.Errorf("raw decode: step %d produced non-finite logits", step)
		}
		run.StepTokens = append(run.StepTokens, token)
		run.GeneratedTokens = append(run.GeneratedTokens, token)
		run.Steps = append(run.Steps, Step{Step: step, TokenID: token, Top1: top1, Top2: top2, Margin: top1 - top2})
		selectedTokenLogprobs = append(selectedTokenLogprobs, selectedTokenLogprob)
		if req.VerifyCPU {
			candidateLogits = append(candidateLogits, slices.Clone(logits))
		}
		previous = token
		run.EOSStopped = shouldStopOnEOS(token)
	}
	run.generation = GenerationObservation{
		observed:       true,
		executionSeal:  executionSeal,
		repetition:     repetition,
		ignoreEOS:      ignoreEOS,
		eosStopped:     run.EOSStopped,
		outputTokenIDs: slices.Clone(run.GeneratedTokens),
	}
	decoded := now()
	s.Close()
	closed := now()
	boundaries := [6]time.Time{start, setupDone, prefillDone, sampled, decoded, closed}
	if !monotonicTimingBoundaries(boundaries) {
		return Run{}, nil, errors.New("raw decode: timing clock regressed during repetition")
	}
	run.timing = TimingObservation{
		executionSeal: executionSeal,
		repetition:    repetition,
		boundaries:    boundaries,
	}
	run.SessionSetupDuration = run.timing.SessionSetupDuration()
	run.PrefillDuration = run.timing.PrefillDuration()
	run.FirstSampleDuration = run.timing.FirstSampleDuration()
	run.DecodeDuration = run.timing.DecodeDuration()
	run.TeardownDuration = run.timing.TeardownDuration()
	if req.VerifyCPU {
		verifyStart := now()
		verification, err := verifyCPU(req, m.NewCPUSession(req), run, candidateLogits)
		run.CPUVerifyDuration = since(verifyStart)
		if err != nil {
			return Run{}, nil, err
		}
		run.CPUVerification = &verification
	}
	return run, selectedTokenLogprobs, nil
}

func verifyCPU(req Request, s session, run Run, candidate [][]float32) (CPUVerification, error) {
	defer s.Close()
	logits := s.Prefill(req.PromptTokenIDs)
	if !allFinite(logits) {
		return CPUVerification{}, errors.New("raw verify cpu: CPU prefill produced non-finite logits")
	}
	verification := compareStep(0, run.PrefillOutputID, candidate[0], logits)
	result := CPUVerification{Passed: verification.Agree, AllAgree: verification.Agree, AllArgmaxAgree: verification.Agree, MinCosine: verification.Cosine, MaxDelta: verification.MaxDelta, Prefill: verification}
	previous := run.PrefillOutputID
	for i, token := range run.StepTokens {
		logits = s.Step(previous)
		if !allFinite(logits) {
			return CPUVerification{}, fmt.Errorf("raw verify cpu: CPU step %d produced non-finite logits", i+1)
		}
		step := compareStep(i+1, token, candidate[i+1], logits)
		result.Steps = append(result.Steps, step)
		result.Passed = result.Passed && step.Agree
		result.AllAgree, result.AllArgmaxAgree = result.Passed, result.Passed
		result.MinCosine = math.Min(result.MinCosine, step.Cosine)
		result.MaxDelta = math.Max(result.MaxDelta, step.MaxDelta)
		previous = token
	}
	return result, nil
}

func compareStep(step, candidateToken int, candidate, cpu []float32) VerificationStep {
	cpuToken, _, _ := logitTop2(cpu)
	return VerificationStep{Step: step, DeviceToken: candidateToken, CPUToken: cpuToken, Agree: candidateToken == cpuToken, Cosine: cosine(candidate, cpu), MaxDelta: maxAbsDelta(candidate, cpu)}
}

func allFinite(values []float32) bool {
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return len(values) > 0
}

func logitTop2(values []float32) (int, float32, float32) {
	top1, top2, index := -float32(math.MaxFloat32), -float32(math.MaxFloat32), 0
	for i, value := range values {
		if value > top1 {
			top2, top1, index = top1, value, i
		} else if value > top2 {
			top2 = value
		}
	}
	return index, top1, top2
}

func greedySelection(values []float32) (int, float32, float32, float64, error) {
	if len(values) == 0 {
		return 0, 0, 0, 0, errors.New("empty logits")
	}
	top1, top2, index := -float32(math.MaxFloat32), -float32(math.MaxFloat32), 0
	for i, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return 0, 0, 0, 0, errors.New("non-finite logits")
		}
		if value > top1 {
			top2, top1, index = top1, value, i
		} else if value > top2 {
			top2 = value
		}
	}
	maximum := float64(top1)
	var shiftedExpSum float64
	for _, value := range values {
		shiftedExpSum += math.Exp(float64(value) - maximum)
	}
	selectedTokenLogprob := -math.Log(shiftedExpSum)
	if selectedTokenLogprob == 0 {
		selectedTokenLogprob = 0
	}
	return index, top1, top2, selectedTokenLogprob, nil
}

func cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, aa, bb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot, aa, bb = dot+x*y, aa+x*x, bb+y*y
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return dot / math.Sqrt(aa*bb)
}

func maxAbsDelta(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var maximum float64
	for i := range a {
		maximum = math.Max(maximum, math.Abs(float64(a[i]-b[i])))
	}
	return maximum
}
