package newmodel

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/modeldescriptor"
)

const (
	ManifestSchema = "fak.new-model-manifest/1"
	PacketSchema   = "fak.new-model-onboarding-packet/1"
	RefusalSchema  = "fak.new-model-refusal/1"
)

// RefusalReason is the closed reason vocabulary returned before any model
// allocation, import, or execution is possible.
type RefusalReason string

const (
	RefusalManifestInvalid       RefusalReason = "MANIFEST_INVALID"
	RefusalPinInvalid            RefusalReason = "PIN_INVALID"
	RefusalUnknownSemanticDelta  RefusalReason = "UNKNOWN_SEMANTIC_DELTA"
	RefusalContradictorySemantic RefusalReason = "CONTRADICTORY_SEMANTIC_DELTA"
	RefusalDescriptorInvalid     RefusalReason = "DESCRIPTOR_INVALID"
	RefusalObligationsIncomplete RefusalReason = "OBLIGATIONS_INCOMPLETE"
)

// Refusal is a machine-readable fail-closed result from manifest compilation.
type Refusal struct {
	Schema string        `json:"schema"`
	Reason RefusalReason `json:"reason"`
	Axis   string        `json:"axis"`
	Detail string        `json:"detail"`
}

func (r *Refusal) Error() string {
	return fmt.Sprintf("%s on %s: %s", r.Reason, r.Axis, r.Detail)
}

type ReleaseManifest struct {
	Schema         string              `json:"schema"`
	Release        ReleaseIdentity     `json:"release"`
	Source         SourcePin           `json:"source"`
	Artifact       ArtifactPin         `json:"artifact"`
	Descriptor     DescriptorInput     `json:"descriptor"`
	SemanticDeltas []SemanticDelta     `json:"semantic_deltas"`
	Obligations    []Obligation        `json:"obligations"`
	Coupling       CouplingDeclaration `json:"coupling"`
}

type ReleaseIdentity struct {
	ID            string `json:"id"`
	Family        string `json:"family"`
	Revision      string `json:"revision"`
	EvidenceClass string `json:"evidence_class"`
}

type SourcePin struct {
	URI            string `json:"uri"`
	Revision       string `json:"revision"`
	ManifestSHA256 string `json:"manifest_sha256"`
}

type ArtifactPin struct {
	URI                string `json:"uri"`
	SHA256             string `json:"sha256"`
	TokenizerSHA256    string `json:"tokenizer_sha256"`
	ChatTemplateSHA256 string `json:"chat_template_sha256"`
}

type DescriptorInput struct {
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
	Axis  string `json:"axis"`
	Value string `json:"value"`
}

type Obligation struct {
	ID          string `json:"id"`
	Kind        string `json:"kind"`
	Description string `json:"description"`
}

type CouplingValues struct {
	CoreSwitches         int `json:"core_switches"`
	OutsideLeafFiles     int `json:"outside_leaf_files"`
	ArchitectureBranches int `json:"architecture_branches"`
	DuplicatedLifecycle  int `json:"duplicated_lifecycle"`
	DuplicatedMetrics    int `json:"duplicated_metrics"`
}

type CouplingDeclaration struct {
	Counts CouplingValues `json:"counts"`
	Budget CouplingValues `json:"budget"`
}

type Packet struct {
	Schema                  string              `json:"schema"`
	ManifestDigest          string              `json:"manifest_digest"`
	Engine                  string              `json:"engine"`
	ExternalRuntimeFallback bool                `json:"external_runtime_fallback"`
	Release                 ReleaseIdentity     `json:"release"`
	Source                  SourcePin           `json:"source"`
	Artifact                ArtifactPin         `json:"artifact"`
	Descriptor              DescriptorPacket    `json:"descriptor"`
	SemanticDeltas          []SemanticDelta     `json:"semantic_deltas"`
	SupportLadder           []SupportRung       `json:"support_ladder"`
	Obligations             []Obligation        `json:"obligations"`
	RegistrationClosure     RegistrationClosure `json:"registration_closure"`
	Coupling                CouplingReport      `json:"coupling"`
}

type CouplingReport struct {
	Schema           string         `json:"schema"`
	DescriptorDigest string         `json:"descriptor_digest"`
	Counts           CouplingValues `json:"counts"`
	Budget           CouplingValues `json:"budget"`
	Missing          []string       `json:"missing"`
	WithinBudget     bool           `json:"within_budget"`
}

type DescriptorPacket struct {
	Schema       string                     `json:"schema"`
	ID           string                     `json:"id"`
	Revision     string                     `json:"revision"`
	Provenance   string                     `json:"provenance"`
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
	Engine       string                     `json:"engine"`
	Forbidden    [][]string                 `json:"forbidden"`
}

func (d DescriptorPacket) ModelDescriptor() modeldescriptor.Descriptor {
	return modeldescriptor.BindFakNative(modeldescriptor.Descriptor{
		Schema: d.Schema, ID: d.ID, Revision: d.Revision, Provenance: d.Provenance, Trust: d.Trust,
		Aliases: d.Aliases, Topology: d.Topology, State: d.State, Quantization: d.Quantization,
		Storage: d.Storage, Tokenizer: d.Tokenizer, Tools: d.Tools, Multimodal: d.Multimodal,
		Backends: d.Backends, Kernels: d.Kernels, Envelopes: d.Envelopes, Oracles: d.Oracles,
		Readiness: d.Readiness, Migration: d.Migration, Forbidden: d.Forbidden,
	})
}

type SupportRung struct {
	Name        string   `json:"name"`
	Status      string   `json:"status"`
	Obligations []string `json:"obligations,omitempty"`
}

type RegistrationClosure struct {
	Complete []string `json:"complete"`
	Open     []string `json:"open"`
	Closed   bool     `json:"closed"`
}

var semanticVocabulary = map[string]map[string]bool{
	"attention":     set("gqa", "mha", "mla", "sliding-window", "hybrid"),
	"ffn":           set("swiglu", "gelu", "moe"),
	"normalization": set("rmsnorm", "layernorm"),
	"position":      set("rope", "alibi", "learned"),
	"routing":       set("dense", "moe-topk"),
	"state":         set("kv-cache", "recurrent", "hybrid"),
}

var obligationKinds = []string{"semantic", "oracle", "backend", "test", "docs", "performance"}

// CompileManifest deterministically turns a pinned, offline manifest into an
// inspection packet. It imports no model package and cannot allocate model state.
func CompileManifest(raw []byte) (Packet, error) {
	var manifest ReleaseManifest
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&manifest); err != nil {
		return Packet{}, refuse(RefusalManifestInvalid, "json", err.Error())
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		return Packet{}, refuse(RefusalManifestInvalid, "json", "multiple JSON values")
	}
	if manifest.Schema != ManifestSchema || manifest.Release.ID == "" || manifest.Release.Family == "" || manifest.Release.Revision == "" {
		return Packet{}, refuse(RefusalManifestInvalid, "identity", "schema, release.id, release.family, and release.revision are required")
	}
	if manifest.Release.EvidenceClass != "pinned-release" && manifest.Release.EvidenceClass != "synthetic-non-claiming" {
		return Packet{}, refuse(RefusalManifestInvalid, "release.evidence_class", "must be pinned-release or synthetic-non-claiming")
	}
	if err := validatePins(manifest); err != nil {
		return Packet{}, err
	}

	normalizeManifest(&manifest)
	if err := validateSemanticDeltas(manifest.SemanticDeltas); err != nil {
		return Packet{}, err
	}
	if err := validateObligations(manifest.Obligations); err != nil {
		return Packet{}, err
	}
	if err := validateDescriptorInput(manifest.Descriptor); err != nil {
		return Packet{}, err
	}
	if err := validateCoupling(manifest.Coupling); err != nil {
		return Packet{}, err
	}

	descriptor := descriptorFrom(manifest)
	if err := modeldescriptor.Validate(descriptor); err != nil {
		return Packet{}, refuse(RefusalDescriptorInvalid, "descriptor", err.Error())
	}
	descriptorDigest, err := modeldescriptor.Digest(descriptor)
	if err != nil {
		return Packet{}, refuse(RefusalDescriptorInvalid, "descriptor", err.Error())
	}
	canonicalManifest, err := json.Marshal(manifest)
	if err != nil {
		return Packet{}, refuse(RefusalManifestInvalid, "json", err.Error())
	}
	manifestSum := sha256.Sum256(canonicalManifest)

	candidate := modeldescriptor.Candidate{
		Descriptor:           descriptor,
		CoreSwitches:         manifest.Coupling.Counts.CoreSwitches,
		OutsideLeafFiles:     manifest.Coupling.Counts.OutsideLeafFiles,
		ArchitectureBranches: manifest.Coupling.Counts.ArchitectureBranches,
		DuplicatedLifecycle:  manifest.Coupling.Counts.DuplicatedLifecycle,
		DuplicatedMetrics:    manifest.Coupling.Counts.DuplicatedMetrics,
	}
	budget := modeldescriptor.Budget{
		CoreSwitches:         manifest.Coupling.Budget.CoreSwitches,
		OutsideLeafFiles:     manifest.Coupling.Budget.OutsideLeafFiles,
		ArchitectureBranches: manifest.Coupling.Budget.ArchitectureBranches,
		DuplicatedLifecycle:  manifest.Coupling.Budget.DuplicatedLifecycle,
		DuplicatedMetrics:    manifest.Coupling.Budget.DuplicatedMetrics,
	}
	report := modeldescriptor.Check(candidate, budget)
	report.DescriptorDigest = descriptorDigest
	appendCouplingOverages(&report)

	packet := Packet{
		Schema: PacketSchema, ManifestDigest: hex.EncodeToString(manifestSum[:]), Engine: "fak-native",
		ExternalRuntimeFallback: false, Release: manifest.Release, Source: manifest.Source, Artifact: manifest.Artifact,
		Descriptor: descriptorPacket(descriptor), SemanticDeltas: manifest.SemanticDeltas,
		Obligations: manifest.Obligations, Coupling: couplingPacket(report),
	}
	packet.SupportLadder = supportLadder(packet.Obligations)
	packet.RegistrationClosure = closure(packet.Obligations)
	return packet, nil
}

func CompileManifestJSON(raw []byte) ([]byte, error) {
	packet, err := CompileManifest(raw)
	if err != nil {
		return nil, err
	}
	out, err := json.MarshalIndent(packet, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(out, '\n'), nil
}

func refuse(reason RefusalReason, axis, detail string) error {
	return &Refusal{Schema: RefusalSchema, Reason: reason, Axis: axis, Detail: detail}
}

func validatePins(m ReleaseManifest) error {
	for axis, value := range map[string]string{
		"source.uri": m.Source.URI, "artifact.uri": m.Artifact.URI,
	} {
		if strings.TrimSpace(value) == "" {
			return refuse(RefusalPinInvalid, axis, "value is required")
		}
	}
	if !isHexLen(m.Source.Revision, 40, 64) {
		return refuse(RefusalPinInvalid, "source.revision", "must be a 40- or 64-character hexadecimal revision")
	}
	for axis, value := range map[string]string{
		"source.manifest_sha256":        m.Source.ManifestSHA256,
		"artifact.sha256":               m.Artifact.SHA256,
		"artifact.tokenizer_sha256":     m.Artifact.TokenizerSHA256,
		"artifact.chat_template_sha256": m.Artifact.ChatTemplateSHA256,
	} {
		if !isHexLen(value, 64) {
			return refuse(RefusalPinInvalid, axis, "must be a 64-character hexadecimal digest")
		}
	}
	return nil
}

func isHexLen(s string, lengths ...int) bool {
	ok := false
	for _, n := range lengths {
		ok = ok || len(s) == n
	}
	if !ok {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

func validateSemanticDeltas(deltas []SemanticDelta) error {
	if len(deltas) == 0 {
		return refuse(RefusalManifestInvalid, "semantic_deltas", "at least one semantic delta is required")
	}
	seen := map[string]string{}
	for _, delta := range deltas {
		values, ok := semanticVocabulary[delta.Axis]
		if !ok || !values[delta.Value] {
			return refuse(RefusalUnknownSemanticDelta, delta.Axis, fmt.Sprintf("unsupported semantic value %q", delta.Value))
		}
		if previous, ok := seen[delta.Axis]; ok && previous != delta.Value {
			return refuse(RefusalContradictorySemantic, delta.Axis, fmt.Sprintf("both %q and %q were declared", previous, delta.Value))
		}
		seen[delta.Axis] = delta.Value
	}
	return nil
}

func validateObligations(obligations []Obligation) error {
	seenIDs := map[string]bool{}
	kinds := map[string]bool{}
	for _, obligation := range obligations {
		if obligation.ID == "" || obligation.Description == "" || !contains(obligationKinds, obligation.Kind) {
			return refuse(RefusalObligationsIncomplete, obligation.Kind, "every obligation needs a unique id, a known kind, and a description")
		}
		if seenIDs[obligation.ID] {
			return refuse(RefusalObligationsIncomplete, obligation.ID, "duplicate obligation id")
		}
		seenIDs[obligation.ID] = true
		kinds[obligation.Kind] = true
	}
	for _, kind := range obligationKinds {
		if !kinds[kind] {
			return refuse(RefusalObligationsIncomplete, kind, "manifest cannot skip this support obligation class")
		}
	}
	return nil
}

func validateDescriptorInput(d DescriptorInput) error {
	if len(d.Aliases) == 0 {
		return refuse(RefusalDescriptorInvalid, "descriptor.aliases", "at least one normalized architecture alias is required")
	}
	seenState := map[string]bool{}
	for _, geometry := range d.State {
		if geometry.Kind == "" || len(geometry.Shape) == 0 || geometry.BytesPerElement <= 0 {
			return refuse(RefusalDescriptorInvalid, "descriptor.state", "state geometry requires a unique kind, a non-empty shape, and positive bytes_per_element")
		}
		if seenState[geometry.Kind] {
			return refuse(RefusalDescriptorInvalid, "descriptor.state", "state geometry kinds must be unique")
		}
		seenState[geometry.Kind] = true
		for _, dimension := range geometry.Shape {
			if dimension <= 0 {
				return refuse(RefusalDescriptorInvalid, "descriptor.state", "state dimensions must be positive")
			}
		}
	}
	return nil
}

func validateCoupling(c CouplingDeclaration) error {
	checks := []struct {
		axis  string
		value int
	}{
		{"coupling.counts.core_switches", c.Counts.CoreSwitches},
		{"coupling.counts.outside_leaf_files", c.Counts.OutsideLeafFiles},
		{"coupling.counts.architecture_branches", c.Counts.ArchitectureBranches},
		{"coupling.counts.duplicated_lifecycle", c.Counts.DuplicatedLifecycle},
		{"coupling.counts.duplicated_metrics", c.Counts.DuplicatedMetrics},
		{"coupling.budget.core_switches", c.Budget.CoreSwitches},
		{"coupling.budget.outside_leaf_files", c.Budget.OutsideLeafFiles},
		{"coupling.budget.architecture_branches", c.Budget.ArchitectureBranches},
		{"coupling.budget.duplicated_lifecycle", c.Budget.DuplicatedLifecycle},
		{"coupling.budget.duplicated_metrics", c.Budget.DuplicatedMetrics},
	}
	for _, check := range checks {
		if check.value < 0 {
			return refuse(RefusalManifestInvalid, check.axis, "coupling values cannot be negative")
		}
	}
	return nil
}

func normalizeManifest(m *ReleaseManifest) {
	m.Release.ID = strings.TrimSpace(m.Release.ID)
	m.Release.Family = normalizeToken(m.Release.Family)
	m.Release.Revision = strings.TrimSpace(m.Release.Revision)
	m.Release.EvidenceClass = normalizeToken(m.Release.EvidenceClass)
	m.Source.URI = strings.TrimSpace(m.Source.URI)
	m.Source.Revision = strings.ToLower(strings.TrimSpace(m.Source.Revision))
	m.Source.ManifestSHA256 = strings.ToLower(strings.TrimSpace(m.Source.ManifestSHA256))
	m.Artifact.URI = strings.TrimSpace(m.Artifact.URI)
	m.Artifact.SHA256 = strings.ToLower(strings.TrimSpace(m.Artifact.SHA256))
	m.Artifact.TokenizerSHA256 = strings.ToLower(strings.TrimSpace(m.Artifact.TokenizerSHA256))
	m.Artifact.ChatTemplateSHA256 = strings.ToLower(strings.TrimSpace(m.Artifact.ChatTemplateSHA256))
	m.Descriptor.Aliases = normalizeAliases(m.Descriptor.Aliases)
	m.Descriptor.Topology = normalizeList(m.Descriptor.Topology)
	for i := range m.Descriptor.State {
		m.Descriptor.State[i].Kind = normalizeToken(m.Descriptor.State[i].Kind)
	}
	sort.Slice(m.Descriptor.State, func(i, j int) bool { return m.Descriptor.State[i].Kind < m.Descriptor.State[j].Kind })
	m.Descriptor.Quantization = normalizeList(m.Descriptor.Quantization)
	m.Descriptor.Storage = normalizeList(m.Descriptor.Storage)
	m.Descriptor.Tokenizer = normalizeList(m.Descriptor.Tokenizer)
	m.Descriptor.Tools = normalizeList(m.Descriptor.Tools)
	m.Descriptor.Multimodal = normalizeList(m.Descriptor.Multimodal)
	m.Descriptor.Backends = normalizeList(m.Descriptor.Backends)
	m.Descriptor.Kernels = normalizeList(m.Descriptor.Kernels)
	m.Descriptor.Envelopes = normalizeList(m.Descriptor.Envelopes)
	m.Descriptor.Oracles = normalizeList(m.Descriptor.Oracles)
	m.Descriptor.Readiness = normalizeList(m.Descriptor.Readiness)
	m.Descriptor.Migration = normalizeList(m.Descriptor.Migration)
	for i := range m.Descriptor.Forbidden {
		m.Descriptor.Forbidden[i] = normalizeList(m.Descriptor.Forbidden[i])
	}
	sort.Slice(m.Descriptor.Forbidden, func(i, j int) bool {
		return strings.Join(m.Descriptor.Forbidden[i], "\x00") < strings.Join(m.Descriptor.Forbidden[j], "\x00")
	})
	for i := range m.SemanticDeltas {
		m.SemanticDeltas[i].Axis = normalizeToken(m.SemanticDeltas[i].Axis)
		m.SemanticDeltas[i].Value = normalizeToken(m.SemanticDeltas[i].Value)
	}
	sort.Slice(m.SemanticDeltas, func(i, j int) bool {
		if m.SemanticDeltas[i].Axis == m.SemanticDeltas[j].Axis {
			return m.SemanticDeltas[i].Value < m.SemanticDeltas[j].Value
		}
		return m.SemanticDeltas[i].Axis < m.SemanticDeltas[j].Axis
	})
	for i := range m.Obligations {
		m.Obligations[i].ID = normalizeToken(m.Obligations[i].ID)
		m.Obligations[i].Kind = normalizeToken(m.Obligations[i].Kind)
		m.Obligations[i].Description = strings.TrimSpace(m.Obligations[i].Description)
	}
	sort.Slice(m.Obligations, func(i, j int) bool {
		if m.Obligations[i].Kind == m.Obligations[j].Kind {
			return m.Obligations[i].ID < m.Obligations[j].ID
		}
		return m.Obligations[i].Kind < m.Obligations[j].Kind
	})
}

func descriptorFrom(m ReleaseManifest) modeldescriptor.Descriptor {
	d := m.Descriptor
	return modeldescriptor.BindFakNative(modeldescriptor.Descriptor{
		Schema: modeldescriptor.Schema, ID: m.Release.ID, Revision: m.Release.Revision,
		Provenance: m.Source.URI + "@" + m.Source.Revision, Trust: "witnessed",
		Aliases: d.Aliases, Topology: d.Topology, State: d.State, Quantization: d.Quantization,
		Storage: d.Storage, Tokenizer: d.Tokenizer, Tools: d.Tools, Multimodal: d.Multimodal,
		Backends: d.Backends, Kernels: d.Kernels, Envelopes: d.Envelopes, Oracles: d.Oracles,
		Readiness: d.Readiness, Migration: d.Migration, Forbidden: d.Forbidden,
	})
}

func descriptorPacket(d modeldescriptor.Descriptor) DescriptorPacket {
	return DescriptorPacket{
		Schema: d.Schema, ID: d.ID, Revision: d.Revision, Provenance: d.Provenance, Trust: d.Trust,
		Aliases: d.Aliases, Topology: d.Topology, State: d.State, Quantization: d.Quantization,
		Storage: d.Storage, Tokenizer: d.Tokenizer, Tools: d.Tools, Multimodal: d.Multimodal,
		Backends: d.Backends, Kernels: d.Kernels, Envelopes: d.Envelopes, Oracles: d.Oracles,
		Readiness: d.Readiness, Migration: d.Migration, Engine: "fak-native", Forbidden: d.Forbidden,
	}
}

func appendCouplingOverages(report *modeldescriptor.Report) {
	checks := []struct {
		name       string
		count, max int
	}{
		{"core_switches", report.Counts.CoreSwitches, report.Budget.CoreSwitches},
		{"outside_leaf_files", report.Counts.OutsideLeafFiles, report.Budget.OutsideLeafFiles},
		{"architecture_branches", report.Counts.ArchitectureBranches, report.Budget.ArchitectureBranches},
		{"duplicated_lifecycle", report.Counts.DuplicatedLifecycle, report.Budget.DuplicatedLifecycle},
		{"duplicated_metrics", report.Counts.DuplicatedMetrics, report.Budget.DuplicatedMetrics},
	}
	for _, check := range checks {
		if check.count > check.max {
			report.Missing = append(report.Missing, "coupling."+check.name)
		}
	}
}

func couplingPacket(report modeldescriptor.Report) CouplingReport {
	values := func(b modeldescriptor.Budget) CouplingValues {
		return CouplingValues{
			CoreSwitches: b.CoreSwitches, OutsideLeafFiles: b.OutsideLeafFiles,
			ArchitectureBranches: b.ArchitectureBranches, DuplicatedLifecycle: b.DuplicatedLifecycle,
			DuplicatedMetrics: b.DuplicatedMetrics,
		}
	}
	return CouplingReport{
		Schema: report.Schema, DescriptorDigest: report.DescriptorDigest,
		Counts: values(report.Counts), Budget: values(report.Budget),
		Missing: append([]string(nil), report.Missing...), WithinBudget: report.WithinBudget,
	}
}

func supportLadder(obligations []Obligation) []SupportRung {
	byKind := map[string][]string{}
	for _, obligation := range obligations {
		byKind[obligation.Kind] = append(byKind[obligation.Kind], obligation.ID)
	}
	return []SupportRung{
		{Name: "release-pinned", Status: "complete"},
		{Name: "descriptor-validated", Status: "complete"},
		{Name: "semantic-reference", Status: "pending", Obligations: append(byKind["semantic"], byKind["oracle"]...)},
		{Name: "fak-native", Status: "pending", Obligations: append(append(byKind["backend"], byKind["test"]...), byKind["docs"]...)},
		{Name: "optimized", Status: "pending", Obligations: byKind["performance"]},
	}
}

func closure(obligations []Obligation) RegistrationClosure {
	open := make([]string, 0, len(obligations))
	for _, obligation := range obligations {
		open = append(open, obligation.Kind+":"+obligation.ID)
	}
	sort.Strings(open)
	return RegistrationClosure{Complete: []string{"descriptor-validation", "source-and-artifact-pins"}, Open: open, Closed: len(open) == 0}
}

func normalizeAliases(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		var b strings.Builder
		for _, r := range strings.TrimSpace(value) {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				b.WriteRune(unicode.ToLower(r))
			}
		}
		if b.Len() > 0 {
			out = append(out, b.String())
		}
	}
	return uniqueSorted(out)
}

func normalizeList(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value = normalizeToken(value); value != "" {
			out = append(out, value)
		}
	}
	return uniqueSorted(out)
}

func normalizeToken(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.NewReplacer("_", "-", " ", "-", "/", "-").Replace(value)
	return strings.Trim(value, "-")
}

func uniqueSorted(values []string) []string {
	sort.Strings(values)
	out := values[:0]
	for _, value := range values {
		if len(out) == 0 || out[len(out)-1] != value {
			out = append(out, value)
		}
	}
	return out
}

func contains(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func set(values ...string) map[string]bool {
	out := make(map[string]bool, len(values))
	for _, value := range values {
		out[value] = true
	}
	return out
}
