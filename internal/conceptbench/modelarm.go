// modelarm.go — the model-driver registry (#2731, epic #2721): one seam that
// resolves a model id to a callable arm, so the benchmark can contrast frontier
// vs small models on identical tasks. Three arms, one interface
// (Drive(task) -> Transcript):
//
//   - gateway — Anthropic models through the fak OpenAI/Anthropic-compatible
//     gateway (internal/gateway).
//   - serve   — local GGUF models through the in-kernel `fak serve --gguf` path
//     (docs/benchmarks/LOCAL-MODEL-CODING-WITNESS-2026-06-27.md).
//   - raw     — an OpenAI-compatible endpoint with no fak wrapping (the
//     LiveCodeBench raw path, #2104), registered explicitly per endpoint.
//
// The registry reuses the fleet's existing model-signal vocabulary instead of
// inventing one: an unknown id resolves to the typed model_unknown class
// (dispatchtick.NoCommitModelUnknown — the same token ModelSwitchableReason
// acts on, 36185e28), and a driven run whose error text names a
// usage/entitlement wall is CLASSIFIED via sessionsignals (through
// dispatchtick.ClassifyNoCommitReason) rather than scored as a concept
// failure — conflating "the model is walled" with "the model failed the
// concept" would misrepresent the benchmark.
//
// Live calls need a key/GPU and are gated: an arm with no bound Transport
// refuses with the typed arm_gated class, and a recorded (replay) Transport is
// labeled so report.go's honesty gate excludes it from headline claims.
package conceptbench

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TranscriptSchemaV1 is the versioned envelope a driven episode carries.
const TranscriptSchemaV1 = "fak.conceptbench.transcript.v1"

// ArmKind is the closed set of driver arms a model id can resolve to.
type ArmKind string

const (
	ArmGateway ArmKind = "gateway"
	ArmServe   ArmKind = "serve"
	ArmRaw     ArmKind = "raw"
)

// Arm-resolution/drive failure classes. ArmErrModelUnknown deliberately equals
// dispatchtick.NoCommitModelUnknown so a registry refusal reads the same as a
// launch-time model refusal everywhere the fleet classifies one (and stays a
// ModelSwitchableReason). ArmErrGated is the "live calls need a key/GPU" fence:
// the model id is known but no Transport is bound for its arm.
const (
	ArmErrModelUnknown = dispatchtick.NoCommitModelUnknown // "model_unknown"
	ArmErrGated        = "arm_gated"
)

// ArmError is the typed error the registry returns on resolution or drive
// failure. Class is machine-checkable (ArmErrModelUnknown | ArmErrGated).
type ArmError struct {
	Class string
	Model string
	Msg   string
}

func (e *ArmError) Error() string {
	return fmt.Sprintf("conceptbench arm %s: %s: %s", e.Model, e.Class, e.Msg)
}

// Provenance is the per-run {model, arm, source} triple the registry resolved,
// emitted into the report so every row states which arm produced it.
type Provenance struct {
	Model  string  `json:"model"`
	Arm    ArmKind `json:"arm"`
	Source string  `json:"source"`
}

// ArmTranscript is one driven episode: the resolved provenance, the model's
// output, and — when the run hit a wall — the sessionsignals-derived class.
// A walled transcript is recorded, never scored (Scoreable reports which).
// (Named apart from the grader's act-record Transcript in grade.go: this is
// the raw drive output the grader adapter (#2732) consumes.)
type ArmTranscript struct {
	Schema      string     `json:"schema"`
	TaskID      string     `json:"task_id"`
	Provenance  Provenance `json:"provenance"`
	Output      string     `json:"output"`
	ErrText     string     `json:"err_text,omitempty"`
	SignalClass string     `json:"signal_class,omitempty"`
	Replay      bool       `json:"replay,omitempty"`
}

// Scoreable reports whether the grader (#2732) may score this transcript's
// output against the task's referee. A run that hit a usage/entitlement wall
// (SignalClass != "") must be recorded as its class, not counted as a concept
// failure.
func (t ArmTranscript) Scoreable() bool { return t.SignalClass == "" }

// StampProvenance copies the transcript's resolved {model, arm, source} — and,
// for a walled run, its signal class — onto a report row, so the report carries
// per-run provenance and the honesty machinery in report.go can exclude a
// classified (walled) row from headline scoring.
func (t ArmTranscript) StampProvenance(row ReportRow) ReportRow {
	row.Model = t.Provenance.Model
	row.Arm = string(t.Provenance.Arm)
	row.ModelSource = t.Provenance.Source
	row.SignalClass = t.SignalClass
	if t.SignalClass != "" {
		// Fold the wall into the per-model no-commit reason mix too, so the
		// rollup's reason columns see it alongside guard refusals.
		row.NoCommitReason = t.SignalClass
	}
	if t.Replay {
		row.Replay = true
	}
	return row
}

// Transport is the callable edge of an arm: given a model id and a prompt,
// return the model's output text and, separately, any terminal ERROR text.
// errText is the error-channel contract sessionsignals classification keys off
// (never assistant prose); err is reserved for transport mechanics (the call
// could not be made at all). A recorded transcript replayer is a valid
// Transport — that is how the test exercises an arm with no key/GPU.
type Transport func(model, prompt string) (output, errText string, err error)

// ModelArm is the one interface all three arms implement.
type ModelArm interface {
	Kind() ArmKind
	Drive(task Task) (ArmTranscript, error)
}

// armEntry is one registered model id: the arm it resolves to, the provenance
// source string the report will carry, and the model-strength band the #5380
// affordance-hint injection reads (see affordance.go). Keeping the band on the
// entry means the registry stays the ONE list of model identity — a second table
// keyed by model id could drift away from this one silently.
type armEntry struct {
	kind   ArmKind
	source string
	tier   ArmTier
}

// Registry resolves a model id to a callable arm. Model-id matching is
// case-insensitive. Transports are bound per arm kind; an unbound arm is
// gated (ArmErrGated), never silently live.
type Registry struct {
	entries    map[string]armEntry
	transports map[ArmKind]Transport
	replay     map[ArmKind]bool
}

// NewRegistry returns a registry seeded with the fleet's known model identity:
// the Anthropic gateway set (the same ids cmd/fak/dispatch_model_policy.go's
// workerModelPolicy pins and downgrades across) and the in-kernel serve set
// proven in the local-model witness doc. Raw endpoints carry no fixed id set —
// register each with RegisterRaw.
func NewRegistry() *Registry {
	r := &Registry{
		entries:    map[string]armEntry{},
		transports: map[ArmKind]Transport{},
		replay:     map[ArmKind]bool{},
	}
	// The strength band per id (#5380). claude-3-5-haiku is the small arm the
	// replay findings actually measured falling off the commit_stamp concept; the
	// rest of the gateway set is the frontier side of that contrast. The in-kernel
	// serve set is the local single-workstation GGUF arm — the weak side of this
	// repo's own local-model coding witness — so it rates small too.
	for _, m := range []string{"claude-opus-5", "claude-opus-4-8", "fable", "claude-fable-5", "claude-sonnet-5"} {
		r.register(m, ArmGateway, "internal/gateway", TierFrontier)
	}
	r.register("claude-3-5-haiku", ArmGateway, "internal/gateway", TierSmall)
	for _, m := range []string{
		"qwen3.6", "smollm2", "glm-5.2",
		"qwen2.5-7b", "qwen2.5-coder-7b", "llama-3.2-3b", "gemma-4-4b", "qwen3-4b",
	} {
		r.register(m, ArmServe, "fak serve --gguf", TierSmall)
	}
	return r
}

func (r *Registry) register(model string, kind ArmKind, source string, tier ArmTier) {
	r.entries[strings.ToLower(strings.TrimSpace(model))] = armEntry{kind: kind, source: source, tier: tier}
}

// RegisterRaw registers an OpenAI-compatible model with no fak wrapping (the
// LiveCodeBench #2104 raw path). endpoint becomes the provenance source. The
// strength band is TierUnrated: the registry has no basis to rate an arbitrary
// endpoint, and an unrated arm is never treated by the #5380 hint injection.
func (r *Registry) RegisterRaw(model, endpoint string) {
	r.register(model, ArmRaw, endpoint, TierUnrated)
}

// Bind attaches the Transport for one arm kind. replay marks a recorded
// (non-live) transport: every transcript it produces is labeled Replay so the
// report's honesty gate excludes it from headline claims.
func (r *Registry) Bind(kind ArmKind, t Transport, replay bool) {
	r.transports[kind] = t
	r.replay[kind] = replay
}

// Models lists the registered model ids, sorted — the benchmark's model axis.
func (r *Registry) Models() []string {
	out := make([]string, 0, len(r.entries))
	for m := range r.entries {
		out = append(out, m)
	}
	sort.Strings(out)
	return out
}

// Resolve maps a model id to its callable arm. An unknown id is the typed
// model_unknown refusal; a known id whose arm has no bound Transport is gated.
func (r *Registry) Resolve(model string) (ModelArm, error) {
	key := strings.ToLower(strings.TrimSpace(model))
	e, ok := r.entries[key]
	if !ok {
		return nil, &ArmError{Class: ArmErrModelUnknown, Model: model, Msg: "no registered arm for this model id"}
	}
	t := r.transports[e.kind]
	if t == nil {
		return nil, &ArmError{Class: ArmErrGated, Model: model, Msg: fmt.Sprintf("arm %q has no bound transport (live calls need a key/GPU)", e.kind)}
	}
	return &armDriver{model: key, entry: e, transport: t, replay: r.replay[e.kind]}, nil
}

// Drive resolves model and drives task through its arm in one call.
func (r *Registry) Drive(model string, task Task) (ArmTranscript, error) {
	arm, err := r.Resolve(model)
	if err != nil {
		return ArmTranscript{}, err
	}
	return arm.Drive(task)
}

// armDriver is the one concrete ModelArm: an entry plus its bound transport.
// All three arms share it — they differ only in kind, source, and transport.
type armDriver struct {
	model     string
	entry     armEntry
	transport Transport
	replay    bool
}

func (a *armDriver) Kind() ArmKind { return a.entry.kind }

// Drive runs the task's prompt through the arm and classifies any terminal
// error text with the fleet's shared classifier (auth_wall > usage_cap >
// model_unknown > rate_limit — dispatchtick.ClassifyNoCommitReason, which
// wraps sessionsignals), so a walled model is recorded as its class.
func (a *armDriver) Drive(task Task) (ArmTranscript, error) {
	out, errText, err := a.transport(a.model, task.Prompt)
	if err != nil {
		return ArmTranscript{}, err
	}
	class := ""
	if strings.TrimSpace(errText) != "" {
		class = dispatchtick.ClassifyNoCommitReason(errText, int64(len(errText)))
	}
	return ArmTranscript{
		Schema:      TranscriptSchemaV1,
		TaskID:      task.ID,
		Provenance:  Provenance{Model: a.model, Arm: a.entry.kind, Source: a.entry.source},
		Output:      out,
		ErrText:     errText,
		SignalClass: class,
		Replay:      a.replay,
	}, nil
}
