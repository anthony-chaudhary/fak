package harnessmodelset

const (
	// SchemaV1 is the only model-set intent schema accepted by this package.
	SchemaV1 = "fak-harness-model-set-intent/1"
	// DiagnosticsSchemaV1 identifies the stable validation-report wire format.
	DiagnosticsSchemaV1 = "fak-harness-model-set-diagnostics/1"
)

// Intent declares the model capabilities a generated harness needs. It names
// requirements only; inventory observations and candidate selection belong to
// later model-set leaves.
type Intent struct {
	Schema string `json:"schema"`
	Roles  []Role `json:"roles"`
}

// Role is one stable harness responsibility and its ordered compatible
// alternatives. Required must be present on the JSON wire even when false.
type Role struct {
	ID           string               `json:"id"`
	Required     bool                 `json:"required"`
	Alternatives []Alternative        `json:"alternatives"`
	Preference   *SelectionPreference `json:"preference,omitempty"`
	Evidence     EvidencePolicy       `json:"evidence"`
}

// Alternative is one independently satisfiable hard-constraint set. Order in
// Role.Alternatives is semantic and is therefore preserved by canonical JSON.
type Alternative struct {
	ID           string                 `json:"id"`
	Capabilities ModelRequirements      `json:"capabilities,omitempty"`
	Operational  OperationalConstraints `json:"operational,omitempty"`
}

// ModelRequirements declares witnessed model behavior. Pointer booleans
// distinguish an omitted constraint from an explicit false, which is rejected
// as ambiguous for this positive-requirement schema.
type ModelRequirements struct {
	Family             string       `json:"family,omitempty"`
	Quantization       string       `json:"quantization,omitempty"`
	ToolCalling        *bool        `json:"tool_calling,omitempty"`
	StructuredOutput   *bool        `json:"structured_output,omitempty"`
	ToolProtocol       ToolProtocol `json:"tool_protocol,omitempty"`
	MinimumInputTokens *int64       `json:"minimum_input_tokens,omitempty"`
	Modalities         []Modality   `json:"modalities,omitempty"`
}

// OperationalConstraints bounds where and how a compatible candidate may run.
// Runtime is an immutable inventory vocabulary key; the remaining enumerations
// are schema-owned so unknown values fail closed before resolution.
type OperationalConstraints struct {
	Runtime          string          `json:"runtime,omitempty"`
	ServingProtocol  ServingProtocol `json:"serving_protocol,omitempty"`
	Platforms        []string        `json:"platforms,omitempty"`
	Accelerators     []string        `json:"accelerators,omitempty"`
	MaxMemoryBytes   *int64          `json:"max_memory_bytes,omitempty"`
	Locality         Locality        `json:"locality,omitempty"`
	Privacy          Privacy         `json:"privacy,omitempty"`
	LicenseAllowlist []string        `json:"license_allowlist,omitempty"`
}

// SelectionPreference applies only after every hard constraint is satisfied.
type SelectionPreference struct {
	Mode PreferenceMode `json:"mode"`
}

// EvidencePolicy bounds the age and accepted kinds of inventory evidence.
type EvidencePolicy struct {
	MaxAgeHours   int64          `json:"max_age_hours"`
	RequiredKinds []EvidenceKind `json:"required_kinds,omitempty"`
}

type ToolProtocol string

const (
	ToolProtocolOpenAI    ToolProtocol = "openai-tools"
	ToolProtocolAnthropic ToolProtocol = "anthropic-tools"
	ToolProtocolMCP       ToolProtocol = "mcp-tools"
)

type Modality string

const (
	ModalityText  Modality = "text"
	ModalityImage Modality = "image"
	ModalityAudio Modality = "audio"
)

type ServingProtocol string

const (
	ServingProtocolOpenAI    ServingProtocol = "openai-compatible"
	ServingProtocolAnthropic ServingProtocol = "anthropic-compatible"
	ServingProtocolGRPC      ServingProtocol = "grpc"
	ServingProtocolInProcess ServingProtocol = "in-process"
)

type Locality string

const (
	LocalityLocalOnly     Locality = "local-only"
	LocalityRemoteAllowed Locality = "remote-allowed"
)

type Privacy string

const (
	PrivacyNoEgress        Privacy = "no-egress"
	PrivacyPrivateEndpoint Privacy = "private-endpoint"
	PrivacyPublicEndpoint  Privacy = "public-endpoint"
)

type PreferenceMode string

const (
	PreferenceDeclaredOrder PreferenceMode = "declared-order"
	PreferenceLocalFirst    PreferenceMode = "local-first"
	PreferenceLowestMemory  PreferenceMode = "lowest-memory"
)

type EvidenceKind string

const (
	EvidenceModelBehaviorProbe  EvidenceKind = "model-behavior-probe"
	EvidenceRuntimeProbe        EvidenceKind = "runtime-probe"
	EvidenceOperatorAttestation EvidenceKind = "operator-attestation"
)
