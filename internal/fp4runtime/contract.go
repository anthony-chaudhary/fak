package fp4runtime

import (
	"encoding/hex"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/strictjson"
	"strings"
)

const (
	// SchemaV1 is the request/result contract implemented by this package.
	SchemaV1 = "fak.fp4runtime/v1"
	// MatrixSchemaV1 is the exact compatibility-matrix schema accepted here.
	MatrixSchemaV1 = "fak.fp4runtime-matrix/v1"

	// SanctionedGPUEvidenceCommand is the positive Blackwell compatibility
	// evidence. Run it on a sanctioned GPU node with TensorRT-LLM installed.
	// On a known non-Blackwell GPU, change the expected outcome to "refuse";
	// that negative run proves the architecture fence rather than a local
	// hardware limitation.
	SanctionedGPUEvidenceCommand = `FAK_FP4_GPU_EVIDENCE=1 FAK_FP4_EXPECT_OUTCOME=delegate FAK_FP4_RUNTIME_ID=tensorrt-llm-pytorch FAK_FP4_RUNTIME_VERSION="$(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)')" FAK_FP4_RUNTIME_FILE="$(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__file__)')" go test ./internal/fp4runtime -run '^TestSanctionedGPUEvidence$' -count=1 -v`
)

type Outcome string

const (
	OutcomeAllow    Outcome = "allow"
	OutcomeDelegate Outcome = "delegate"
	OutcomeAbstain  Outcome = "abstain"
	OutcomeRefuse   Outcome = "refuse"
)

type ReasonCode string

const (
	ReasonAdmitted                     ReasonCode = "FP4_RUNTIME_ADMITTED"
	ReasonRuntimeDelegationRequired    ReasonCode = "FP4_RUNTIME_DELEGATION_REQUIRED"
	ReasonInvalidJSON                  ReasonCode = "FP4_INVALID_JSON"
	ReasonUnknownField                 ReasonCode = "FP4_UNKNOWN_FIELD"
	ReasonUnknownSchema                ReasonCode = "FP4_UNKNOWN_SCHEMA"
	ReasonInvalidRequest               ReasonCode = "FP4_INVALID_REQUEST"
	ReasonInvalidMatrix                ReasonCode = "FP4_INVALID_MATRIX"
	ReasonUnknownArtifact              ReasonCode = "FP4_UNKNOWN_ARTIFACT"
	ReasonUnknownArtifactVersion       ReasonCode = "FP4_UNKNOWN_ARTIFACT_VERSION"
	ReasonArtifactSemanticsMismatch    ReasonCode = "FP4_ARTIFACT_SEMANTICS_MISMATCH"
	ReasonUnknownRuntime               ReasonCode = "FP4_UNKNOWN_RUNTIME"
	ReasonUnknownRuntimeVersion        ReasonCode = "FP4_UNKNOWN_RUNTIME_VERSION"
	ReasonRuntimeUnavailable           ReasonCode = "FP4_RUNTIME_UNAVAILABLE"
	ReasonUnknownGPUArchitecture       ReasonCode = "FP4_UNKNOWN_GPU_ARCHITECTURE"
	ReasonGPUArchitectureUnavailable   ReasonCode = "FP4_GPU_ARCHITECTURE_UNAVAILABLE"
	ReasonUnknownAccumulator           ReasonCode = "FP4_UNKNOWN_ACCUMULATOR"
	ReasonAccumulatorSemanticsMismatch ReasonCode = "FP4_ACCUMULATOR_SEMANTICS_MISMATCH"
	ReasonAccumulatorUnavailable       ReasonCode = "FP4_ACCUMULATOR_UNAVAILABLE"
	ReasonInvalidHardwareEvidence      ReasonCode = "FP4_INVALID_HARDWARE_EVIDENCE"
	ReasonHardwareEvidenceMismatch     ReasonCode = "FP4_HARDWARE_EVIDENCE_MISMATCH"
)

type ArtifactID string
type RuntimeID string
type ArchitectureID string
type AccumulatorID string
type ProfileID string
type Mode string

const (
	ModeNative   Mode = "native"
	ModeExternal Mode = "external"

	ArtifactNVIDIANVFP4 ArtifactID = "nvidia-nvfp4"
	ArtifactOCPMXFP4    ArtifactID = "ocp-mxfp4"

	RuntimeTensorRTLLMPyTorch RuntimeID = "tensorrt-llm-pytorch"
	RuntimeFakNative          RuntimeID = "fak-native"

	ArchitectureSM80  ArchitectureID = "sm_80"
	ArchitectureSM86  ArchitectureID = "sm_86"
	ArchitectureSM89  ArchitectureID = "sm_89"
	ArchitectureSM90  ArchitectureID = "sm_90"
	ArchitectureSM100 ArchitectureID = "sm_100"
	ArchitectureSM103 ArchitectureID = "sm_103"
	ArchitectureSM120 ArchitectureID = "sm_120"

	AccumulatorFP32BF16RNE AccumulatorID = "fp32-bf16-rne"
)

// Pin identifies one immutable artifact, recipe, or runtime build. A friendly
// name without a version and digest is not enough to support an admission.
type Pin struct {
	ID      string `json:"id"`
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
}

// Artifact describes the externally owned FP4/microscaling bytes. The
// element, scale, and block fields are matched independently: NVFP4 and
// MXFP4 are never treated as aliases merely because both contain E2M1 data.
type Artifact struct {
	Pin           Pin    `json:"pin"`
	ElementFormat string `json:"element_format"`
	ScaleFormat   string `json:"scale_format"`
	BlockSize     int    `json:"block_size"`
}

// GPU is the declared hardware target. Device is descriptive; Vendor and
// Architecture are the compatibility keys.
type GPU struct {
	Vendor       string         `json:"vendor"`
	Architecture ArchitectureID `json:"architecture"`
	Device       string         `json:"device,omitempty"`
}

// AccumulatorSemantics names all arithmetic properties this contract matches.
// A shared ID with different fields is a contradiction and refuses.
type AccumulatorSemantics struct {
	ID         AccumulatorID `json:"id"`
	Product    string        `json:"product"`
	Accumulate string        `json:"accumulate"`
	Output     string        `json:"output"`
	Rounding   string        `json:"rounding"`
}

// HardwareEvidence is independently read evidence for the hardware envelope.
// It does not assert model quality or performance. RuntimeSHA256,
// Architecture, and AccumulatorID must match the request exactly.
type HardwareEvidence struct {
	Source            string         `json:"source"`
	RunSHA256         string         `json:"run_sha256"`
	RuntimeSHA256     string         `json:"runtime_sha256"`
	Architecture      ArchitectureID `json:"architecture"`
	AccumulatorID     AccumulatorID  `json:"accumulator_id"`
	DeviceFingerprint string         `json:"device_fingerprint"`
	Command           string         `json:"command"`
}

type Request struct {
	Schema           string               `json:"schema"`
	Artifact         Artifact             `json:"artifact"`
	Recipe           Pin                  `json:"recipe"`
	Runtime          Pin                  `json:"runtime"`
	GPU              GPU                  `json:"gpu"`
	Accumulator      AccumulatorSemantics `json:"accumulator"`
	HardwareEvidence *HardwareEvidence    `json:"hardware_evidence,omitempty"`
}

// ArtifactSpec is one known serialization identity in a Matrix.
type ArtifactSpec struct {
	ID            ArtifactID `json:"id"`
	Version       string     `json:"version"`
	ElementFormat string     `json:"element_format"`
	ScaleFormat   string     `json:"scale_format"`
	BlockSize     int        `json:"block_size"`
}

type RuntimeSpec struct {
	ID      RuntimeID `json:"id"`
	Version string    `json:"version"`
}

type ArchitectureSpec struct {
	Vendor string         `json:"vendor"`
	ID     ArchitectureID `json:"id"`
	Class  string         `json:"class"`
}

// Profile is one exact compatibility row. No field is a wildcard. ModeNative
// says the caller owns execution; ModeExternal says a named external runtime
// owns it and the result must remain a delegation.
type Profile struct {
	ID              ProfileID      `json:"id"`
	ArtifactID      ArtifactID     `json:"artifact_id"`
	ArtifactVersion string         `json:"artifact_version"`
	RuntimeID       RuntimeID      `json:"runtime_id"`
	RuntimeVersion  string         `json:"runtime_version"`
	Architecture    ArchitectureID `json:"architecture"`
	AccumulatorID   AccumulatorID  `json:"accumulator_id"`
	Mode            Mode           `json:"mode"`
	Authority       string         `json:"authority"`
}

// Matrix separates known vocabulary from supported rows. That distinction
// lets the adjudicator return abstain for an unknown value and refuse for a
// known value in an unavailable combination.
type Matrix struct {
	Schema        string                 `json:"schema"`
	Artifacts     []ArtifactSpec         `json:"artifacts"`
	Runtimes      []RuntimeSpec          `json:"runtimes"`
	Architectures []ArchitectureSpec     `json:"architectures"`
	Accumulators  []AccumulatorSemantics `json:"accumulators"`
	Profiles      []Profile              `json:"profiles"`
}

type RuntimeClaim struct {
	Pin       Pin       `json:"pin"`
	ProfileID ProfileID `json:"profile_id,omitempty"`
	External  bool      `json:"external"`
}

type HardwareClaim struct {
	Vendor         string         `json:"vendor"`
	Architecture   ArchitectureID `json:"architecture"`
	Device         string         `json:"device,omitempty"`
	Observed       bool           `json:"observed"`
	EvidenceSHA256 string         `json:"evidence_sha256,omitempty"`
}

// Claims deliberately keeps the four user-facing authorities separate.
// Compatibility cannot turn an artifact identity into a recipe claim, a
// delegation into a native claim, or an unobserved target into measurement.
type Claims struct {
	Artifact Pin           `json:"artifact"`
	Recipe   Pin           `json:"recipe"`
	Runtime  RuntimeClaim  `json:"runtime"`
	Hardware HardwareClaim `json:"hardware"`
}

type Result struct {
	Outcome   Outcome    `json:"outcome"`
	Reason    ReasonCode `json:"reason"`
	Detail    string     `json:"detail,omitempty"`
	ProfileID ProfileID  `json:"profile_id,omitempty"`
	Authority string     `json:"authority,omitempty"`
	Claims    Claims     `json:"claims"`
}

// ParseAndNegotiate strictly parses both inputs and returns a typed result on
// every path. The error is supplementary for callers following Go's malformed
// input convention; Result remains the machine-readable authority.
func ParseAndNegotiate(requestJSON, matrixJSON []byte) (Result, error) {
	var request Request
	if err := decodeStrict(requestJSON, &request); err != nil {
		return parseFailure(err), fmt.Errorf("parse fp4 request: %w", err)
	}
	var matrix Matrix
	if err := decodeStrict(matrixJSON, &matrix); err != nil {
		result := parseFailure(err)
		if result.Reason == ReasonInvalidJSON {
			result.Detail = "matrix: " + result.Detail
		} else {
			result.Detail = "matrix: " + result.Detail
		}
		return result, fmt.Errorf("parse fp4 matrix: %w", err)
	}
	return Negotiate(request, matrix), nil
}

// Negotiate matches an artifact only when its serialization, runtime version,
// GPU architecture, and accumulator semantics all exactly match one profile.
//
// Invariant: FP4 runtime evaluations are fail-closed and deterministic across all artifact profiles.
// Guard: Incompatible matrix definitions, unverified evidence, or mismatched formats force explicit refusal.
func Negotiate(request Request, matrix Matrix) Result {
	result := baseResult(request)
	if matrix.Schema != MatrixSchemaV1 {
		return result.finish(OutcomeAbstain, ReasonUnknownSchema, "matrix schema "+matrix.Schema)
	}
	if err := validateMatrix(matrix); err != nil {
		return result.finish(OutcomeRefuse, ReasonInvalidMatrix, err.Error())
	}
	if request.Schema != SchemaV1 {
		return result.finish(OutcomeAbstain, ReasonUnknownSchema, "request schema "+request.Schema)
	}
	if err := validateRequest(request); err != nil {
		return result.finish(OutcomeRefuse, ReasonInvalidRequest, err.Error())
	}
	if request.HardwareEvidence != nil {
		if reason, err := validateHardwareEvidence(request); err != nil {
			return result.finish(OutcomeRefuse, reason, err.Error())
		}
		result.Claims.Hardware.Observed = true
		result.Claims.Hardware.EvidenceSHA256 = request.HardwareEvidence.RunSHA256
	}

	artifact, idKnown, versionKnown := findArtifact(matrix.Artifacts, request.Artifact.Pin.ID, request.Artifact.Pin.Version)
	if !idKnown {
		return result.finish(OutcomeAbstain, ReasonUnknownArtifact, request.Artifact.Pin.ID)
	}
	if !versionKnown {
		return result.finish(OutcomeAbstain, ReasonUnknownArtifactVersion, request.Artifact.Pin.ID+"@"+request.Artifact.Pin.Version)
	}
	if artifact.ElementFormat != request.Artifact.ElementFormat ||
		artifact.ScaleFormat != request.Artifact.ScaleFormat ||
		artifact.BlockSize != request.Artifact.BlockSize {
		return result.finish(OutcomeRefuse, ReasonArtifactSemanticsMismatch,
			fmt.Sprintf("%s@%s requires element=%s scale=%s block=%d",
				artifact.ID, artifact.Version, artifact.ElementFormat, artifact.ScaleFormat, artifact.BlockSize))
	}

	runtimeIDKnown, runtimeVersionKnown := findRuntime(matrix.Runtimes, request.Runtime.ID, request.Runtime.Version)
	if !runtimeIDKnown {
		return result.finish(OutcomeAbstain, ReasonUnknownRuntime, request.Runtime.ID)
	}
	if !runtimeVersionKnown {
		return result.finish(OutcomeAbstain, ReasonUnknownRuntimeVersion, request.Runtime.ID+"@"+request.Runtime.Version)
	}
	if !findArchitecture(matrix.Architectures, request.GPU.Vendor, request.GPU.Architecture) {
		return result.finish(OutcomeAbstain, ReasonUnknownGPUArchitecture,
			request.GPU.Vendor+"/"+string(request.GPU.Architecture))
	}
	accumulator, ok := findAccumulator(matrix.Accumulators, request.Accumulator.ID)
	if !ok {
		return result.finish(OutcomeAbstain, ReasonUnknownAccumulator, string(request.Accumulator.ID))
	}
	if accumulator != request.Accumulator {
		return result.finish(OutcomeRefuse, ReasonAccumulatorSemanticsMismatch,
			"accumulator "+string(request.Accumulator.ID)+" fields do not match the matrix vocabulary")
	}

	profiles := profilesForArtifactRuntime(matrix.Profiles, request)
	if len(profiles) == 0 {
		return result.finish(OutcomeRefuse, ReasonRuntimeUnavailable,
			request.Runtime.ID+"@"+request.Runtime.Version+" does not advertise "+request.Artifact.Pin.ID+"@"+request.Artifact.Pin.Version)
	}
	profiles = profilesForArchitecture(profiles, request.GPU.Architecture)
	if len(profiles) == 0 {
		return result.finish(OutcomeRefuse, ReasonGPUArchitectureUnavailable,
			string(request.GPU.Architecture)+" is known but absent from the artifact/runtime profile")
	}
	profiles = profilesForAccumulator(profiles, request.Accumulator.ID)
	if len(profiles) == 0 {
		return result.finish(OutcomeRefuse, ReasonAccumulatorUnavailable,
			string(request.Accumulator.ID)+" is known but absent from the artifact/runtime/architecture profile")
	}
	if len(profiles) != 1 {
		return result.finish(OutcomeRefuse, ReasonInvalidMatrix, "multiple exact profiles matched")
	}
	profile := profiles[0]
	result.ProfileID = profile.ID
	result.Authority = profile.Authority
	result.Claims.Runtime.ProfileID = profile.ID
	switch profile.Mode {
	case ModeNative:
		return result.finish(OutcomeAllow, ReasonAdmitted, "exact native profile match")
	case ModeExternal:
		result.Claims.Runtime.External = true
		return result.finish(OutcomeDelegate, ReasonRuntimeDelegationRequired, "exact external runtime profile match")
	default:
		return result.finish(OutcomeRefuse, ReasonInvalidMatrix, "profile "+string(profile.ID)+" has invalid mode")
	}
}

func baseResult(request Request) Result {
	return Result{Claims: Claims{
		Artifact: request.Artifact.Pin,
		Recipe:   request.Recipe,
		Runtime:  RuntimeClaim{Pin: request.Runtime},
		Hardware: HardwareClaim{
			Vendor:       request.GPU.Vendor,
			Architecture: request.GPU.Architecture,
			Device:       request.GPU.Device,
		},
	}}
}

func (result Result) finish(outcome Outcome, reason ReasonCode, detail string) Result {
	result.Outcome = outcome
	result.Reason = reason
	result.Detail = detail
	return result
}

func parseFailure(err error) Result {
	reason := ReasonInvalidJSON
	outcome := OutcomeRefuse
	if strings.Contains(err.Error(), "json: unknown field ") {
		reason = ReasonUnknownField
		outcome = OutcomeAbstain
	}
	return Result{Outcome: outcome, Reason: reason, Detail: err.Error()}
}

func decodeStrict(raw []byte, destination any) error {
	return strictjson.Decode(raw, destination, "multiple JSON values")
}

func validateRequest(request Request) error {
	if err := validatePin("artifact", request.Artifact.Pin); err != nil {
		return err
	}
	if err := validatePin("recipe", request.Recipe); err != nil {
		return err
	}
	if err := validatePin("runtime", request.Runtime); err != nil {
		return err
	}
	if strings.TrimSpace(request.Artifact.ElementFormat) == "" ||
		strings.TrimSpace(request.Artifact.ScaleFormat) == "" ||
		request.Artifact.BlockSize <= 0 {
		return fmt.Errorf("artifact element_format, scale_format, and positive block_size are required")
	}
	if strings.TrimSpace(request.GPU.Vendor) == "" || request.GPU.Architecture == "" {
		return fmt.Errorf("gpu vendor and architecture are required")
	}
	if err := validateAccumulator("request accumulator", request.Accumulator); err != nil {
		return err
	}
	return nil
}

func validatePin(name string, pin Pin) error {
	if strings.TrimSpace(pin.ID) == "" || strings.TrimSpace(pin.Version) == "" || !isSHA256(pin.SHA256) {
		return fmt.Errorf("%s pin requires id, version, and lowercase sha256", name)
	}
	return nil
}

func validateAccumulator(name string, accumulator AccumulatorSemantics) error {
	if accumulator.ID == "" ||
		strings.TrimSpace(accumulator.Product) == "" ||
		strings.TrimSpace(accumulator.Accumulate) == "" ||
		strings.TrimSpace(accumulator.Output) == "" ||
		strings.TrimSpace(accumulator.Rounding) == "" {
		return fmt.Errorf("%s requires id, product, accumulate, output, and rounding", name)
	}
	return nil
}

func validateHardwareEvidence(request Request) (ReasonCode, error) {
	evidence := request.HardwareEvidence
	if strings.TrimSpace(evidence.Source) == "" ||
		!isSHA256(evidence.RunSHA256) ||
		!isSHA256(evidence.RuntimeSHA256) ||
		strings.TrimSpace(evidence.DeviceFingerprint) == "" ||
		strings.TrimSpace(evidence.Command) == "" ||
		evidence.Architecture == "" ||
		evidence.AccumulatorID == "" {
		return ReasonInvalidHardwareEvidence, fmt.Errorf("hardware evidence requires source, run/runtime sha256, architecture, accumulator, device fingerprint, and command")
	}
	if evidence.RuntimeSHA256 != request.Runtime.SHA256 {
		return ReasonHardwareEvidenceMismatch, fmt.Errorf("evidence runtime digest does not match the requested runtime")
	}
	if evidence.Architecture != request.GPU.Architecture {
		return ReasonHardwareEvidenceMismatch, fmt.Errorf("evidence architecture does not match the requested GPU")
	}
	if evidence.AccumulatorID != request.Accumulator.ID {
		return ReasonHardwareEvidenceMismatch, fmt.Errorf("evidence accumulator does not match the requested semantics")
	}
	return "", nil
}

func validateMatrix(matrix Matrix) error {
	if len(matrix.Artifacts) == 0 || len(matrix.Runtimes) == 0 ||
		len(matrix.Architectures) == 0 || len(matrix.Accumulators) == 0 {
		return fmt.Errorf("matrix vocabulary must include artifacts, runtimes, architectures, and accumulators")
	}

	artifactKeys := map[string]bool{}
	for _, artifact := range matrix.Artifacts {
		key := string(artifact.ID) + "@" + artifact.Version
		if artifact.ID == "" || strings.TrimSpace(artifact.Version) == "" ||
			strings.TrimSpace(artifact.ElementFormat) == "" ||
			strings.TrimSpace(artifact.ScaleFormat) == "" || artifact.BlockSize <= 0 {
			return fmt.Errorf("artifact %q is incomplete", key)
		}
		if artifactKeys[key] {
			return fmt.Errorf("duplicate artifact %s", key)
		}
		artifactKeys[key] = true
	}

	runtimeKeys := map[string]bool{}
	for _, runtime := range matrix.Runtimes {
		key := string(runtime.ID) + "@" + runtime.Version
		if runtime.ID == "" || strings.TrimSpace(runtime.Version) == "" {
			return fmt.Errorf("runtime %q is incomplete", key)
		}
		if runtimeKeys[key] {
			return fmt.Errorf("duplicate runtime %s", key)
		}
		runtimeKeys[key] = true
	}

	architectureKeys := map[string]bool{}
	architectureIDs := map[ArchitectureID]bool{}
	for _, architecture := range matrix.Architectures {
		key := architecture.Vendor + "/" + string(architecture.ID)
		if strings.TrimSpace(architecture.Vendor) == "" || architecture.ID == "" || strings.TrimSpace(architecture.Class) == "" {
			return fmt.Errorf("architecture %q is incomplete", key)
		}
		if architectureKeys[key] {
			return fmt.Errorf("duplicate architecture %s", key)
		}
		architectureKeys[key] = true
		if architectureIDs[architecture.ID] {
			return fmt.Errorf("architecture id %s is ambiguous across vendors", architecture.ID)
		}
		architectureIDs[architecture.ID] = true
	}

	accumulatorKeys := map[AccumulatorID]bool{}
	for _, accumulator := range matrix.Accumulators {
		if err := validateAccumulator("matrix accumulator", accumulator); err != nil {
			return err
		}
		if accumulatorKeys[accumulator.ID] {
			return fmt.Errorf("duplicate accumulator %s", accumulator.ID)
		}
		accumulatorKeys[accumulator.ID] = true
	}

	profileIDs := map[ProfileID]bool{}
	exactKeys := map[string]bool{}
	for _, profile := range matrix.Profiles {
		if profile.ID == "" || strings.TrimSpace(profile.Authority) == "" {
			return fmt.Errorf("profile id and authority are required")
		}
		if profileIDs[profile.ID] {
			return fmt.Errorf("duplicate profile id %s", profile.ID)
		}
		profileIDs[profile.ID] = true
		if profile.Mode != ModeNative && profile.Mode != ModeExternal {
			return fmt.Errorf("profile %s has unknown mode %q", profile.ID, profile.Mode)
		}
		artifactKey := string(profile.ArtifactID) + "@" + profile.ArtifactVersion
		if !artifactKeys[artifactKey] {
			return fmt.Errorf("profile %s references unknown artifact %s", profile.ID, artifactKey)
		}
		runtimeKey := string(profile.RuntimeID) + "@" + profile.RuntimeVersion
		if !runtimeKeys[runtimeKey] {
			return fmt.Errorf("profile %s references unknown runtime %s", profile.ID, runtimeKey)
		}
		if !architectureIDKnown(matrix.Architectures, profile.Architecture) {
			return fmt.Errorf("profile %s references unknown architecture %s", profile.ID, profile.Architecture)
		}
		if !accumulatorKeys[profile.AccumulatorID] {
			return fmt.Errorf("profile %s references unknown accumulator %s", profile.ID, profile.AccumulatorID)
		}
		exact := strings.Join([]string{
			artifactKey, runtimeKey, string(profile.Architecture), string(profile.AccumulatorID),
		}, "|")
		if exactKeys[exact] {
			return fmt.Errorf("duplicate exact compatibility profile %s", exact)
		}
		exactKeys[exact] = true
	}
	return nil
}

func findArtifact(artifacts []ArtifactSpec, id, version string) (ArtifactSpec, bool, bool) {
	idKnown := false
	for _, artifact := range artifacts {
		if string(artifact.ID) != id {
			continue
		}
		idKnown = true
		if artifact.Version == version {
			return artifact, true, true
		}
	}
	return ArtifactSpec{}, idKnown, false
}

func findRuntime(runtimes []RuntimeSpec, id, version string) (bool, bool) {
	idKnown := false
	for _, runtime := range runtimes {
		if string(runtime.ID) != id {
			continue
		}
		idKnown = true
		if runtime.Version == version {
			return true, true
		}
	}
	return idKnown, false
}

func findArchitecture(architectures []ArchitectureSpec, vendor string, id ArchitectureID) bool {
	for _, architecture := range architectures {
		if architecture.Vendor == vendor && architecture.ID == id {
			return true
		}
	}
	return false
}

func architectureIDKnown(architectures []ArchitectureSpec, id ArchitectureID) bool {
	for _, architecture := range architectures {
		if architecture.ID == id {
			return true
		}
	}
	return false
}

func findAccumulator(accumulators []AccumulatorSemantics, id AccumulatorID) (AccumulatorSemantics, bool) {
	for _, accumulator := range accumulators {
		if accumulator.ID == id {
			return accumulator, true
		}
	}
	return AccumulatorSemantics{}, false
}

func profilesForArtifactRuntime(profiles []Profile, request Request) []Profile {
	var matched []Profile
	for _, profile := range profiles {
		if string(profile.ArtifactID) == request.Artifact.Pin.ID &&
			profile.ArtifactVersion == request.Artifact.Pin.Version &&
			string(profile.RuntimeID) == request.Runtime.ID &&
			profile.RuntimeVersion == request.Runtime.Version {
			matched = append(matched, profile)
		}
	}
	return matched
}

func profilesForArchitecture(profiles []Profile, architecture ArchitectureID) []Profile {
	var matched []Profile
	for _, profile := range profiles {
		if profile.Architecture == architecture {
			matched = append(matched, profile)
		}
	}
	return matched
}

func profilesForAccumulator(profiles []Profile, accumulator AccumulatorID) []Profile {
	var matched []Profile
	for _, profile := range profiles {
		if profile.AccumulatorID == accumulator {
			matched = append(matched, profile)
		}
	}
	return matched
}

func isSHA256(value string) bool {
	if len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}
