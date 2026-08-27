package newmodel

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/modeldescriptor"
)

const (
	ReleaseManifestSchema  = "fak.new-model-release-manifest/1"
	OnboardingPacketSchema = "fak.new-model-onboarding-packet/1"
	RefusalSchema          = "fak.new-model-refusal/1"

	RefusalMalformedManifest      = "MALFORMED_RELEASE_MANIFEST"
	RefusalUnresolvedSemanticAxis = "UNRESOLVED_SEMANTIC_AXIS"
	RefusalDescriptorMismatch     = "DESCRIPTOR_INCOMPATIBLE"
)

// ReleaseManifest is the offline, pinned input to model onboarding. It contains
// facts and obligations only; it cannot import code or execute model behavior.
type ReleaseManifest struct {
	Schema         string            `json:"schema"`
	ReleaseID      string            `json:"release_id"`
	Source         SourceIdentity    `json:"source"`
	Artifact       ArtifactIdentity  `json:"artifact"`
	Execution      ExecutionIdentity `json:"execution"`
	Descriptor     DescriptorInput   `json:"descriptor"`
	SemanticDeltas []SemanticDelta   `json:"semantic_deltas"`
	Obligations    []Obligation      `json:"obligations"`
	Coupling       CouplingInput     `json:"coupling"`
	Rollback       string            `json:"rollback"`
}

type SourceIdentity struct {
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	SHA256     string `json:"sha256"`
}

type ArtifactIdentity struct {
	ID     string `json:"id"`
	Format string `json:"format"`
	SHA256 string `json:"sha256"`
}

type ExecutionIdentity struct {
	Engine           string `json:"engine"`
	ExternalFallback bool   `json:"external_fallback"`
}

type DescriptorInput struct {
	ID           string                     `json:"id"`
	Revision     string                     `json:"revision"`
	Trust        string                     `json:"trust"`
	Aliases      []string                   `json:"aliases"`
	Topology     []string                   `json:"topology"`
	State        []modeldescriptor.Geometry `json:"state"`
	Quantization []string                   `json:"quantization"`
	Storage      []string                   `json:"storage"`
	Tokenizer    []string                   `json:"tokenizer"`
	Tools        []string                   `json:"tools"`
	Multimodal   []string                   `json:"multimodal"`
	Backends     []string                   `json:"backends"`
	Kernels      []string                   `json:"kernels"`
	Envelopes    []string                   `json:"envelopes"`
	Oracles      []string                   `json:"oracles"`
	Readiness    []string                   `json:"readiness"`
	Migration    []string                   `json:"migration"`
	Forbidden    [][]string                 `json:"forbidden"`
}

type SemanticDelta struct {
	Axis       string `json:"axis"`
	Value      string `json:"value"`
	Status     string `json:"status"`
	Resolution string `json:"resolution"`
}

type Obligation struct {
	Kind   string `json:"kind"`
	Target string `json:"target"`
	Status string `json:"status"`
}

type CouplingInput struct {
	CoreSwitches         int `json:"core_switches"`
	OutsideLeafFiles     int `json:"outside_leaf_files"`
	ArchitectureBranches int `json:"architecture_branches"`
	DuplicatedLifecycle  int `json:"duplicated_lifecycle"`
	DuplicatedMetrics    int `json:"duplicated_metrics"`
	Budget               struct {
		CoreSwitches         int `json:"core_switches"`
		OutsideLeafFiles     int `json:"outside_leaf_files"`
		ArchitectureBranches int `json:"architecture_branches"`
		DuplicatedLifecycle  int `json:"duplicated_lifecycle"`
		DuplicatedMetrics    int `json:"duplicated_metrics"`
	} `json:"budget"`
}

type OnboardingPacket struct {
	Schema              string                     `json:"schema"`
	ReleaseID           string                     `json:"release_id"`
	ManifestSHA256      string                     `json:"manifest_sha256"`
	Source              SourceIdentity             `json:"source"`
	Artifact            ArtifactIdentity           `json:"artifact"`
	Execution           ExecutionIdentity          `json:"execution"`
	Descriptor          modeldescriptor.Descriptor `json:"descriptor"`
	DescriptorSHA256    string                     `json:"descriptor_sha256"`
	SupportLadder       []SupportStage             `json:"support_ladder"`
	OpenObligations     []Obligation               `json:"open_obligations"`
	RegistrationClosure []string                   `json:"registration_closure"`
	CouplingReport      modeldescriptor.Report     `json:"coupling_report"`
	Rollback            string                     `json:"rollback"`
}

type SupportStage struct {
	Stage  string `json:"stage"`
	Status string `json:"status"`
}

type Refusal struct {
	Schema string `json:"schema"`
	Code   string `json:"code"`
	Phase  string `json:"phase"`
	Axis   string `json:"axis"`
	Detail string `json:"detail"`
}

type CompileError struct{ Refusal Refusal }

func (e *CompileError) Error() string {
	return fmt.Sprintf("%s: %s: %s", e.Refusal.Code, e.Refusal.Axis, e.Refusal.Detail)
}

// RefusalFor returns the stable refusal payload carried by a compile error.
func RefusalFor(err error) (Refusal, bool) {
	var compileErr *CompileError
	if !errors.As(err, &compileErr) {
		return Refusal{}, false
	}
	return compileErr.Refusal, true
}

// CompileReleaseManifest validates and canonicalizes a pinned release without
// fetching artifacts, allocating weights, or executing any model runtime.
func CompileReleaseManifest(data []byte) (OnboardingPacket, error) {
	manifest, err := decodeReleaseManifest(data)
	if err != nil {
		return OnboardingPacket{}, refuse(RefusalMalformedManifest, "manifest", err.Error())
	}
	if err := validateManifestIdentity(manifest); err != nil {
		return OnboardingPacket{}, err
	}
	if err := validateSemanticDeltas(manifest.SemanticDeltas); err != nil {
		return OnboardingPacket{}, err
	}

	descriptor := compileDescriptor(manifest)
	if err := validateDescriptorState(descriptor.State); err != nil {
		return OnboardingPacket{}, refuse(RefusalDescriptorMismatch, "state", err.Error())
	}
	if err := modeldescriptor.Validate(descriptor); err != nil {
		return OnboardingPacket{}, refuse(RefusalDescriptorMismatch, "descriptor", err.Error())
	}
	descriptorDigest, err := modeldescriptor.Digest(descriptor)
	if err != nil {
		return OnboardingPacket{}, refuse(RefusalDescriptorMismatch, "descriptor", err.Error())
	}

	obligations, err := canonicalObligations(manifest.Obligations)
	if err != nil {
		return OnboardingPacket{}, err
	}
	report := modeldescriptor.Check(modeldescriptor.Candidate{
		Descriptor:           descriptor,
		CoreSwitches:         manifest.Coupling.CoreSwitches,
		OutsideLeafFiles:     manifest.Coupling.OutsideLeafFiles,
		ArchitectureBranches: manifest.Coupling.ArchitectureBranches,
		DuplicatedLifecycle:  manifest.Coupling.DuplicatedLifecycle,
		DuplicatedMetrics:    manifest.Coupling.DuplicatedMetrics,
	}, modeldescriptor.Budget{
		CoreSwitches:         manifest.Coupling.Budget.CoreSwitches,
		OutsideLeafFiles:     manifest.Coupling.Budget.OutsideLeafFiles,
		ArchitectureBranches: manifest.Coupling.Budget.ArchitectureBranches,
		DuplicatedLifecycle:  manifest.Coupling.Budget.DuplicatedLifecycle,
		DuplicatedMetrics:    manifest.Coupling.Budget.DuplicatedMetrics,
	})

	manifestDigest := sha256.Sum256(data)
	return OnboardingPacket{
		Schema:              OnboardingPacketSchema,
		ReleaseID:           manifest.ReleaseID,
		ManifestSHA256:      hex.EncodeToString(manifestDigest[:]),
		Source:              manifest.Source,
		Artifact:            manifest.Artifact,
		Execution:           manifest.Execution,
		Descriptor:          descriptor,
		DescriptorSHA256:    descriptorDigest,
		SupportLadder:       supportLadder(),
		OpenObligations:     obligations,
		RegistrationClosure: []string{"aliases-normalized", "descriptor-validated", "digests-pinned", "external-fallback-disabled"},
		CouplingReport:      report,
		Rollback:            manifest.Rollback,
	}, nil
}

func decodeReleaseManifest(data []byte) (ReleaseManifest, error) {
	var manifest ReleaseManifest
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&manifest); err != nil {
		return manifest, err
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			return manifest, errors.New("multiple JSON values")
		}
		return manifest, err
	}
	return manifest, nil
}

func validateManifestIdentity(manifest ReleaseManifest) error {
	if manifest.Schema != ReleaseManifestSchema {
		return refuse(RefusalMalformedManifest, "schema", fmt.Sprintf("must be %q", ReleaseManifestSchema))
	}
	if strings.TrimSpace(manifest.ReleaseID) == "" || strings.TrimSpace(manifest.Source.Repository) == "" || strings.TrimSpace(manifest.Source.Revision) == "" {
		return refuse(RefusalMalformedManifest, "release_identity", "release_id, source.repository, and source.revision are required")
	}
	if !sha256Digest(manifest.Source.SHA256) {
		return refuse(RefusalMalformedManifest, "source.sha256", "must be a lowercase SHA-256 digest")
	}
	if strings.TrimSpace(manifest.Artifact.ID) == "" || strings.TrimSpace(manifest.Artifact.Format) == "" || !sha256Digest(manifest.Artifact.SHA256) {
		return refuse(RefusalMalformedManifest, "artifact", "id, format, and lowercase SHA-256 digest are required")
	}
	if manifest.Execution.Engine != "fak-native" || manifest.Execution.ExternalFallback {
		return refuse(RefusalDescriptorMismatch, "execution", "requires fak-native with external_fallback=false")
	}
	if strings.TrimSpace(manifest.Rollback) == "" {
		return refuse(RefusalMalformedManifest, "rollback", "an explicit rollback action is required")
	}
	return nil
}

var semanticAxes = map[string]bool{
	"architecture_alias": true,
	"attention_state":    true,
	"backend":            true,
	"kernel":             true,
	"multimodal":         true,
	"oracle":             true,
	"quantization":       true,
	"storage":            true,
	"tokenizer":          true,
	"tool_calling":       true,
	"topology":           true,
}

func validateSemanticDeltas(deltas []SemanticDelta) error {
	for _, delta := range deltas {
		axis := strings.TrimSpace(delta.Axis)
		if !semanticAxes[axis] {
			return refuse(RefusalUnresolvedSemanticAxis, axis, "unknown semantic axis")
		}
		if delta.Status != "resolved" || strings.TrimSpace(delta.Value) == "" || strings.TrimSpace(delta.Resolution) == "" {
			return refuse(RefusalUnresolvedSemanticAxis, axis, "semantic delta must name a resolved value and descriptor resolution")
		}
	}
	return nil
}

func compileDescriptor(manifest ReleaseManifest) modeldescriptor.Descriptor {
	in := manifest.Descriptor
	return modeldescriptor.Descriptor{
		Schema:       modeldescriptor.Schema,
		ID:           strings.TrimSpace(in.ID),
		Revision:     strings.TrimSpace(in.Revision),
		Provenance:   strings.TrimSpace(manifest.Source.Repository) + "@" + strings.TrimSpace(manifest.Source.Revision),
		Trust:        strings.TrimSpace(in.Trust),
		Aliases:      normalizeAliases(in.Aliases),
		Topology:     canonicalStrings(in.Topology),
		State:        canonicalState(in.State),
		Quantization: canonicalStrings(in.Quantization),
		Storage:      canonicalStrings(in.Storage),
		Tokenizer:    canonicalStrings(in.Tokenizer),
		Tools:        canonicalStrings(in.Tools),
		Multimodal:   canonicalStrings(in.Multimodal),
		Backends:     canonicalStrings(in.Backends),
		Kernels:      canonicalStrings(in.Kernels),
		Envelopes:    canonicalStrings(in.Envelopes),
		Oracles:      canonicalStrings(in.Oracles),
		Readiness:    canonicalStrings(in.Readiness),
		Migration:    canonicalStrings(in.Migration),
		NativeEngine: manifest.Execution.Engine,
		Forbidden:    canonicalForbidden(in.Forbidden),
	}
}

func canonicalStrings(values []string) []string {
	set := make(map[string]struct{}, len(values))
	for _, value := range values {
		if value = strings.TrimSpace(value); value != "" {
			set[value] = struct{}{}
		}
	}
	out := make([]string, 0, len(set))
	for value := range set {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func normalizeAliases(values []string) []string {
	aliases := make([]string, 0, len(values))
	for _, value := range values {
		var b strings.Builder
		for _, r := range strings.ToLower(strings.TrimSpace(value)) {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				b.WriteRune(r)
			}
		}
		aliases = append(aliases, b.String())
	}
	return canonicalStrings(aliases)
}

func canonicalState(state []modeldescriptor.Geometry) []modeldescriptor.Geometry {
	out := append([]modeldescriptor.Geometry(nil), state...)
	sort.Slice(out, func(i, j int) bool { return out[i].Kind < out[j].Kind })
	return out
}

func canonicalForbidden(combos [][]string) [][]string {
	out := make([][]string, 0, len(combos))
	for _, combo := range combos {
		out = append(out, canonicalStrings(combo))
	}
	sort.Slice(out, func(i, j int) bool { return strings.Join(out[i], "\x00") < strings.Join(out[j], "\x00") })
	return out
}

func validateDescriptorState(state []modeldescriptor.Geometry) error {
	if len(state) == 0 {
		return errors.New("at least one state geometry is required")
	}
	for _, geometry := range state {
		if strings.TrimSpace(geometry.Kind) == "" || len(geometry.Shape) == 0 || geometry.BytesPerElement <= 0 {
			return fmt.Errorf("geometry %q requires kind, shape, and positive bytes_per_element", geometry.Kind)
		}
		for _, dimension := range geometry.Shape {
			if dimension <= 0 {
				return fmt.Errorf("geometry %q has a non-positive shape dimension", geometry.Kind)
			}
		}
	}
	return nil
}

func canonicalObligations(obligations []Obligation) ([]Obligation, error) {
	required := map[string]bool{"backend": false, "docs": false, "oracle": false, "performance": false, "test": false}
	out := append([]Obligation(nil), obligations...)
	for i := range out {
		out[i].Kind = strings.TrimSpace(out[i].Kind)
		out[i].Target = strings.TrimSpace(out[i].Target)
		out[i].Status = strings.TrimSpace(out[i].Status)
		if _, ok := required[out[i].Kind]; !ok || out[i].Target == "" || out[i].Status != "open" {
			return nil, refuse(RefusalMalformedManifest, "obligations", "each obligation must be an open backend, docs, oracle, performance, or test target")
		}
		required[out[i].Kind] = true
	}
	for kind, present := range required {
		if !present {
			return nil, refuse(RefusalMalformedManifest, "obligations."+kind, "required open obligation is missing")
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind == out[j].Kind {
			return out[i].Target < out[j].Target
		}
		return out[i].Kind < out[j].Kind
	})
	return out, nil
}

func supportLadder() []SupportStage {
	return []SupportStage{
		{Stage: "descriptor", Status: "validated"},
		{Stage: "registration", Status: "open"},
		{Stage: "correctness", Status: "open"},
		{Stage: "backend", Status: "open"},
		{Stage: "performance", Status: "open"},
		{Stage: "promotion", Status: "blocked"},
	}
}

func sha256Digest(value string) bool {
	if len(value) != sha256.Size*2 || value != strings.ToLower(value) {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func refuse(code, axis, detail string) error {
	return &CompileError{Refusal: Refusal{Schema: RefusalSchema, Code: code, Phase: "pre-allocation", Axis: axis, Detail: detail}}
}
