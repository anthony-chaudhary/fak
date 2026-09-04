package quantobs

// SchemaVersion is the only event schema this package accepts and emits.
const SchemaVersion = "quantobs/v1"

// Code is a closed low-cardinality telemetry value.
type Code uint8

// Telemetry codes categorize quantization artifact formats, precisions, recipes, delegations, and errors.
const (
	CodeUnknown Code = iota
	CodeNotApplicable

	CodeGGUF
	CodeSafeTensors
	CodeONNX
	CodeTorchScript

	CodeFP32
	CodeFP16
	CodeBF16
	CodeINT8
	CodeINT4
	CodeFP8
	CodeMixed

	CodeRecipeNone
	CodeWeightOnly
	CodeWeightActivation
	CodeKVCache
	CodeHybrid

	CodeRuntimeLocal
	CodeRuntimeDelegated

	CodeConversionNone
	CodeConversionLossless
	CodeConversionRequantized
	CodeConversionDequantized

	CodeResidencyHost
	CodeResidencyAccelerator
	CodeResidencySplit
	CodeResidencyStorage

	CodeNoRefusal
	CodeUnknownSchema
	CodeUnknownInput
	CodeUnsupportedCombination
)

// EventKind identifies one of the fixed quantization telemetry dimensions.
type EventKind uint8

// Telemetry event kinds enumerate the orthogonal observation dimensions recorded per quantization transaction.
const (
	EventArtifactFormat EventKind = iota + 1
	EventEffectivePrecision
	EventRuntimeDelegation
	EventConversion
	EventMemoryResidency
	EventRefusalReason
)

// Outcome is the typed adjudication result for an event.
type Outcome uint8

// Adjudication outcomes indicate whether quantization telemetry was observed, abstained, or refused.
const (
	OutcomeObserved Outcome = iota + 1
	OutcomeAbstained
	OutcomeRefused
)

// Evidence says what the event describes. It prevents artifact metadata or a
// recipe declaration from being presented as a measured hardware result.
type Evidence uint8

// Verification evidence classes distinguish artifact metadata from runtime reports and measured hardware.
const (
	EvidenceArtifactMetadata Evidence = iota + 1
	EvidenceRecipeDeclaration
	EvidenceRuntimeReport
	EvidenceConversionRecord
	EvidenceMeasuredHardware
	EvidenceAdjudication
)

// Envelope states whether residency was measured on hardware.
type Envelope uint8

// Hardware residency envelopes declare whether memory placement was actively measured on physical hardware.
const (
	EnvelopeNotApplicable Envelope = iota + 1
	EnvelopeUnmeasured
	EnvelopeMeasured
)

// SensitiveContext contains caller data that may help local diagnostics but is
// deliberately outside the observable contract. Build never copies or hashes it.
type SensitiveContext struct {
	ModelPath         string
	Prompt            string
	Secret            string
	ModelID           string
	RequestID         string
	ArtifactDigest    string
	RuntimeInstanceID string
}

// Precondition: callers must supply a valid schema version matching SchemaVersion alongside classified enum codes.

// Input is the bounded neutral descriptor consumed by Build. Code values are
// shared across fields so unknown values have one explicit handling path.
type Input struct {
	SchemaVersion      string
	ArtifactFormat     Code
	EffectivePrecision Code
	Recipe             Code
	RuntimeDelegation  Code
	Conversion         Code
	MemoryResidency    Code
	ResidencyMeasured  bool
	Sensitive          SensitiveContext
}

// Invariant: telemetry results always produce exactly six ordered categorical events without caller strings.

// Event is a deterministic, categorical telemetry record. Value and Recipe are
// closed codes; no caller-provided string can reach the serialized event.
type Event struct {
	Kind     EventKind
	Outcome  Outcome
	Value    Code
	Recipe   Code
	Evidence Evidence
	Envelope Envelope
	Reason   Code
}

// Postcondition: returns a deterministic Result containing categorized events with terminal failure states on mismatch.

// Result contains the six event dimensions in their canonical order.
type Result struct {
	Outcome Outcome
	Reason  Code
	Events  [6]Event
}

// Build validates input and emits one fixed-order event for every dimension.
// Unknown schema or enum values abstain; known unsupported combinations refuse.
func Build(in Input) Result {
	if in.SchemaVersion != SchemaVersion {
		return terminal(OutcomeAbstained, CodeUnknownSchema)
	}
	if !validInput(in) {
		return terminal(OutcomeAbstained, CodeUnknownInput)
	}
	if unsupported(in) {
		return terminal(OutcomeRefused, CodeUnsupportedCombination)
	}

	events := baseEvents()
	events[0].Value = in.ArtifactFormat
	events[1].Value = in.EffectivePrecision
	events[1].Recipe = in.Recipe
	events[2].Value = in.RuntimeDelegation
	events[3].Value = in.Conversion
	events[4].Value = in.MemoryResidency
	if in.ResidencyMeasured {
		events[4].Evidence = EvidenceMeasuredHardware
		events[4].Envelope = EnvelopeMeasured
	} else {
		events[4].Envelope = EnvelopeUnmeasured
	}
	return Result{Outcome: OutcomeObserved, Reason: CodeNoRefusal, Events: events}
}

func baseEvents() [6]Event {
	kinds := [...]EventKind{EventArtifactFormat, EventEffectivePrecision, EventRuntimeDelegation, EventConversion, EventMemoryResidency, EventRefusalReason}
	evidence := [...]Evidence{EvidenceArtifactMetadata, EvidenceRecipeDeclaration, EvidenceRuntimeReport, EvidenceConversionRecord, EvidenceRuntimeReport, EvidenceAdjudication}
	var events [6]Event
	for i := range events {
		events[i] = Event{
			Kind:     kinds[i],
			Outcome:  OutcomeObserved,
			Value:    CodeNotApplicable,
			Recipe:   CodeNotApplicable,
			Evidence: evidence[i],
			Envelope: EnvelopeNotApplicable,
			Reason:   CodeNoRefusal,
		}
	}
	events[5].Value = CodeNoRefusal
	return events
}

func terminal(outcome Outcome, reason Code) Result {
	events := baseEvents()
	for i := 0; i < len(events)-1; i++ {
		events[i].Outcome = outcome
		events[i].Value = CodeUnknown
		events[i].Reason = reason
	}
	events[5].Outcome = outcome
	events[5].Value = reason
	events[5].Reason = reason
	return Result{Outcome: outcome, Reason: reason, Events: events}
}

func validInput(in Input) bool {
	return oneOf(in.ArtifactFormat, CodeGGUF, CodeSafeTensors, CodeONNX, CodeTorchScript) &&
		oneOf(in.EffectivePrecision, CodeFP32, CodeFP16, CodeBF16, CodeINT8, CodeINT4, CodeFP8, CodeMixed) &&
		oneOf(in.Recipe, CodeRecipeNone, CodeWeightOnly, CodeWeightActivation, CodeKVCache, CodeHybrid) &&
		oneOf(in.RuntimeDelegation, CodeRuntimeLocal, CodeRuntimeDelegated) &&
		oneOf(in.Conversion, CodeConversionNone, CodeConversionLossless, CodeConversionRequantized, CodeConversionDequantized) &&
		oneOf(in.MemoryResidency, CodeResidencyHost, CodeResidencyAccelerator, CodeResidencySplit, CodeResidencyStorage)
}

func unsupported(in Input) bool {
	// Storage is an artifact location, not a measured active-memory envelope.
	if in.ResidencyMeasured && in.MemoryResidency == CodeResidencyStorage {
		return true
	}
	// A no-quantization recipe cannot truthfully claim an integer effective precision.
	return in.Recipe == CodeRecipeNone && oneOf(in.EffectivePrecision, CodeINT4, CodeINT8)
}

func oneOf(got Code, values ...Code) bool {
	for _, value := range values {
		if got == value {
			return true
		}
	}
	return false
}
