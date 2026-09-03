package harnesskit

import "encoding/json"

// Contract describes the machine-readable public compatibility surface.
type Contract struct {
	SchemaVersion string                         `json:"schema_version"`
	GoImport      string                         `json:"go_import"`
	InternalRoot  string                         `json:"internal_root"`
	Planes        []ExtensionPlane               `json:"extension_planes"`
	Lifecycle     []LifecycleState               `json:"lifecycle"`
	Cancellation  string                         `json:"cancellation"`
	Streaming     string                         `json:"streaming"`
	Backpressure  string                         `json:"backpressure"`
	Errors        []Code                         `json:"errors"`
	Security      string                         `json:"security_reachability"`
	Compatibility string                         `json:"compatibility"`
	Ownership     map[string]Ownership           `json:"resource_ownership"`
	RunProtocol   ProtocolContract               `json:"run_protocol"`
	Instructions  InstructionCompositionContract `json:"instruction_composition"`
	Hardware      HardwareContract               `json:"hardware"`
	Tools         ToolContract                   `json:"tools,omitempty"`
}

// CompatibilityContract publishes the stable machine formats used at upgrade
// and activation boundaries.
type CompatibilityContract struct {
	ReportSchema string                `json:"report_schema"`
	DiffSchema   string                `json:"diff_schema"`
	PlanSchema   string                `json:"plan_schema"`
	Statuses     []CapabilityStatus    `json:"statuses"`
	Reasons      []CompatibilityReason `json:"reasons"`
	Absence      string                `json:"absence"`
	Planning     string                `json:"planning"`
}

// SupportedPlanes returns a fresh copy of all public extension planes.
func SupportedPlanes() []ExtensionPlane {
	return []ExtensionPlane{PlaneTools, PlaneModels, PlaneContext, PlaneInstructions, PlaneTransports, PlaneEvents, PlaneHardware}
}

// PublicCompatibilityContract returns the machine formats and closed
// vocabularies for negotiation and read-only upgrade planning. It is separate
// from Contract so adding this surface does not break unkeyed v1alpha1 literals.
func PublicCompatibilityContract() CompatibilityContract {
	return CompatibilityContract{
		ReportSchema: CompatibilityReportSchema,
		DiffSchema:   ContractDiffSchema,
		PlanSchema:   UpgradePlanSchema,
		Statuses:     []CapabilityStatus{StatusStable, StatusExperimental, StatusDeprecated},
		Reasons:      []CompatibilityReason{ReasonCompatible, ReasonContractMismatch, ReasonCapabilityAbsent, ReasonRevisionBelowMin, ReasonRevisionAboveMax, ReasonStatusMismatch, ReasonInvalidRequirement, ReasonInvalidOffer, ReasonDuplicateRequirement, ReasonDuplicateOffer, ReasonInvalidDeprecation},
		Absence:      "unknown or absent capabilities are incompatible; version provenance is not proof",
		Planning:     "upgrade planning is pure and never edits builder-owned code or configuration",
	}
}

// PublicContract returns the normative machine-readable contract.
func PublicContract() Contract {
	return Contract{
		SchemaVersion: ContractVersion,
		GoImport:      "github.com/anthony-chaudhary/fak/pkg/harnesskit",
		InternalRoot:  "github.com/anthony-chaudhary/fak/internal/",
		Planes:        SupportedPlanes(),
		Lifecycle:     []LifecycleState{StateDeclared, StateStarting, StateRunning, StateDraining, StateClosed, StateFailed},
		Cancellation:  "every blocking operation accepts context.Context and returns its cause",
		Streaming:     "Stream.Send and Stream.Recv are ordered; io.EOF is clean completion",
		Backpressure:  "Send blocks until accepted or context cancellation; adapters may return backpressure",
		Errors:        []Code{CodeInvalid, CodeUnsupported, CodeConflict, CodeDenied, CodeCanceled, CodeBackpressure, CodeInternal},
		Security:      "registration grants reachability, never authority; Services.Invoke is adjudicated per call",
		Compatibility: "same schema_version is additive; removals or semantic changes require a new schema_version",
		Ownership:     map[string]Ownership{"factory_runtime": OwnershipHost, "builder_inputs": OwnershipCaller, "service_inputs": OwnershipCaller},
		RunProtocol:   PublicProtocolContract(),
		Instructions:  PublicInstructionContract(),
		Hardware:      PublicHardwareContract(),
		Tools:         PublicToolContract(),
	}
}

// ContractJSON serializes the normative contract for tooling and fixtures.
func ContractJSON() ([]byte, error) { return json.MarshalIndent(PublicContract(), "", "  ") }
