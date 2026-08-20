package serverproduct

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"net/url"
	pathpkg "path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	// SchemaV1 is the only server-product contract understood by this package.
	SchemaV1 = "fak.server-product/v1"

	LifecycleLocalProcess = "local-process"
	ProtocolOpenAIHTTP    = "openai-http"
	AuthNone              = "none"
	AuthCredentialRef     = "credential-ref"
	ProvenanceAuthored    = "authored"
	ProvenanceObserved    = "observed"
	ReceiptStateReady     = "ready"
)

var (
	digestPattern        = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
	namePattern          = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,63}$`)
	capabilityPattern    = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,63}$`)
	credentialRefPattern = regexp.MustCompile(`^[A-Z][A-Z0-9_]{0,127}$`)
	jwtPattern           = regexp.MustCompile(`[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}`)
)

// ServerSpec contains authored intent. It carries references and constraints,
// never resolved runtime observations or credential values.
type ServerSpec struct {
	Schema            string           `json:"schema"`
	ServerName        string           `json:"server_name"`
	InstanceDirectory string           `json:"instance_directory"`
	Artifact          ArtifactSpec     `json:"artifact"`
	Adapter           AdapterSpec      `json:"adapter"`
	Protocol          ProtocolSpec     `json:"protocol"`
	Endpoint          EndpointSpec     `json:"endpoint"`
	Auth              AuthReference    `json:"auth"`
	Lifecycle         string           `json:"lifecycle"`
	Provenance        Provenance       `json:"provenance"`
	Resource          ResourceEnvelope `json:"resource"`
}

type ArtifactSpec struct {
	Reference string `json:"reference"`
	Digest    string `json:"digest"`
}

type AdapterSpec struct {
	Name              string `json:"name"`
	VersionConstraint string `json:"version_constraint"`
}

type ProtocolSpec struct {
	Family               string   `json:"family"`
	Revision             string   `json:"revision"`
	RequiredCapabilities []string `json:"required_capabilities"`
}

type EndpointSpec struct {
	BindHost      string `json:"bind_host"`
	RequestedPort uint16 `json:"requested_port"`
}

// AuthReference names where a later runtime may resolve a credential. The
// symbolic reference is intentionally unable to represent a bearer value.
type AuthReference struct {
	Mode          string `json:"mode"`
	CredentialRef string `json:"credential_ref,omitempty"`
}

// ResourceEnvelope keeps the v1 intent bounded without becoming a deployment
// DSL. Zero means the operator did not place that optional bound.
type ResourceEnvelope struct {
	MaxMemoryBytes uint64 `json:"max_memory_bytes,omitempty"`
	GPUCount       uint16 `json:"gpu_count,omitempty"`
}

type Provenance struct {
	Kind   string `json:"kind"`
	Source string `json:"source"`
}

// ServerReceipt contains runtime facts observed after a readiness probe.
type ServerReceipt struct {
	Schema     string              `json:"schema"`
	State      string              `json:"state"`
	Identity   ServerIdentity      `json:"identity"`
	SpecDigest string              `json:"spec_digest"`
	Generation uint64              `json:"generation"`
	CreatedAt  string              `json:"created_at"`
	Artifact   ArtifactIdentity    `json:"artifact"`
	Adapter    AdapterIdentity     `json:"adapter"`
	Endpoint   LoopbackEndpoint    `json:"endpoint"`
	ModelAlias string              `json:"model_alias"`
	Auth       AuthReference       `json:"auth"`
	Protocol   ProtocolObservation `json:"protocol"`
	Readiness  ReadinessEvidence   `json:"readiness"`
	Ownership  OwnershipReference  `json:"ownership"`
	Provenance ReceiptProvenance   `json:"provenance"`
}

type ServerIdentity struct {
	ServerName string `json:"server_name"`
	InstanceID string `json:"instance_id"`
}

type ArtifactIdentity struct {
	Reference string `json:"reference"`
	Digest    string `json:"digest"`
}

type AdapterIdentity struct {
	Name             string `json:"name"`
	Version          string `json:"version"`
	ExecutableDigest string `json:"executable_digest"`
}

type LoopbackEndpoint struct {
	BaseURL string `json:"base_url"`
}

type ProtocolObservation struct {
	Family       string   `json:"family"`
	Revision     string   `json:"revision"`
	Capabilities []string `json:"capabilities"`
}

type ReadinessEvidence struct {
	Probe       string `json:"probe"`
	ProbeDigest string `json:"probe_digest"`
	ObservedAt  string `json:"observed_at"`
}

type OwnershipReference struct {
	InstanceID           string `json:"instance_id"`
	ProcessID            int    `json:"process_id"`
	ProcessStartIdentity string `json:"process_start_identity"`
}

// ReceiptProvenance makes the authored-versus-observed boundary explicit for
// each receipt group instead of relying on field-name convention.
type ReceiptProvenance struct {
	Spec      Provenance `json:"spec"`
	Artifact  Provenance `json:"artifact"`
	Adapter   Provenance `json:"adapter"`
	Endpoint  Provenance `json:"endpoint"`
	Readiness Provenance `json:"readiness"`
	Ownership Provenance `json:"ownership"`
}

// ValidateSpec rejects ambiguous, unsafe, or secret-bearing schema-v1 intent.
func ValidateSpec(spec ServerSpec) error {
	if spec.Schema != SchemaV1 {
		return fmt.Errorf("schema must be %q", SchemaV1)
	}
	if !namePattern.MatchString(spec.ServerName) {
		return errors.New("server_name must be a lowercase stable name")
	}
	if err := validateInstanceDirectory(spec.InstanceDirectory); err != nil {
		return err
	}
	if err := validateArtifactReference(spec.Artifact.Reference); err != nil {
		return err
	}
	if !digestPattern.MatchString(spec.Artifact.Digest) {
		return errors.New("artifact digest must be lowercase sha256:<64 hex>")
	}
	if !namePattern.MatchString(spec.Adapter.Name) {
		return errors.New("adapter name must be a lowercase stable name")
	}
	if err := validateText("adapter version_constraint", spec.Adapter.VersionConstraint); err != nil {
		return err
	}
	if spec.Protocol.Family != ProtocolOpenAIHTTP {
		return fmt.Errorf("protocol family must be %q", ProtocolOpenAIHTTP)
	}
	if err := validateText("protocol revision", spec.Protocol.Revision); err != nil {
		return err
	}
	if err := validateCapabilities("required capabilities", spec.Protocol.RequiredCapabilities); err != nil {
		return err
	}
	if err := validateBindHost(spec.Endpoint.BindHost); err != nil {
		return err
	}
	if err := validateAuth(spec.Auth); err != nil {
		return err
	}
	if spec.Lifecycle != LifecycleLocalProcess {
		return fmt.Errorf("lifecycle must be %q", LifecycleLocalProcess)
	}
	if err := validateProvenance("spec", spec.Provenance, ProvenanceAuthored); err != nil {
		return err
	}
	return nil
}

// ValidateReceipt validates receipt-internal invariants. Use
// CheckCompatibility as well before a consumer trusts it for a spec.
func ValidateReceipt(receipt ServerReceipt) error {
	if receipt.Schema != SchemaV1 {
		return fmt.Errorf("schema must be %q", SchemaV1)
	}
	if receipt.State != ReceiptStateReady {
		return fmt.Errorf("receipt state must be %q", ReceiptStateReady)
	}
	if !namePattern.MatchString(receipt.Identity.ServerName) {
		return errors.New("identity server_name must be a lowercase stable name")
	}
	if !namePattern.MatchString(receipt.Identity.InstanceID) {
		return errors.New("identity instance_id must be a lowercase stable name")
	}
	if !digestPattern.MatchString(receipt.SpecDigest) {
		return errors.New("spec_digest must be lowercase sha256:<64 hex>")
	}
	if receipt.Generation == 0 {
		return errors.New("generation must be positive")
	}
	createdAt, err := validateTimestamp("created_at", receipt.CreatedAt)
	if err != nil {
		return err
	}
	if err := validateArtifactReference(receipt.Artifact.Reference); err != nil {
		return err
	}
	if !digestPattern.MatchString(receipt.Artifact.Digest) {
		return errors.New("artifact digest must be lowercase sha256:<64 hex>")
	}
	if !namePattern.MatchString(receipt.Adapter.Name) {
		return errors.New("adapter name must be a lowercase stable name")
	}
	if err := validateText("adapter version", receipt.Adapter.Version); err != nil {
		return err
	}
	if !digestPattern.MatchString(receipt.Adapter.ExecutableDigest) {
		return errors.New("adapter executable_digest must be lowercase sha256:<64 hex>")
	}
	if err := validateBaseURL(receipt.Endpoint.BaseURL); err != nil {
		return err
	}
	if err := validateText("model_alias", receipt.ModelAlias); err != nil {
		return err
	}
	if err := validateAuth(receipt.Auth); err != nil {
		return err
	}
	if receipt.Protocol.Family != ProtocolOpenAIHTTP {
		return fmt.Errorf("protocol family must be %q", ProtocolOpenAIHTTP)
	}
	if err := validateText("protocol revision", receipt.Protocol.Revision); err != nil {
		return err
	}
	if err := validateCapabilities("probed capabilities", receipt.Protocol.Capabilities); err != nil {
		return err
	}
	if err := validateText("readiness probe", receipt.Readiness.Probe); err != nil {
		return err
	}
	if !digestPattern.MatchString(receipt.Readiness.ProbeDigest) {
		return errors.New("readiness probe_digest must be lowercase sha256:<64 hex>")
	}
	observedAt, err := validateTimestamp("readiness observed_at", receipt.Readiness.ObservedAt)
	if err != nil {
		return err
	}
	if observedAt.After(createdAt) {
		return errors.New("readiness observed_at must not be after receipt created_at")
	}
	if receipt.Ownership.InstanceID != receipt.Identity.InstanceID {
		return errors.New("ownership instance_id must match identity instance_id")
	}
	if receipt.Ownership.ProcessID <= 0 {
		return errors.New("ownership process_id must be positive")
	}
	if err := validateText("ownership process_start_identity", receipt.Ownership.ProcessStartIdentity); err != nil {
		return err
	}
	if err := validateProvenance("provenance spec", receipt.Provenance.Spec, ProvenanceAuthored); err != nil {
		return err
	}
	for field, provenance := range map[string]Provenance{
		"artifact":  receipt.Provenance.Artifact,
		"adapter":   receipt.Provenance.Adapter,
		"endpoint":  receipt.Provenance.Endpoint,
		"readiness": receipt.Provenance.Readiness,
		"ownership": receipt.Provenance.Ownership,
	} {
		if err := validateProvenance("provenance "+field, provenance, ProvenanceObserved); err != nil {
			return err
		}
	}
	return nil
}

// CheckCompatibility binds a ready receipt to the exact authored spec that a
// consumer selected. Adapter constraint interpretation belongs to the adapter
// leaf; schema v1 still binds the observed adapter name and version evidence.
func CheckCompatibility(spec ServerSpec, receipt ServerReceipt) error {
	if err := ValidateSpec(spec); err != nil {
		return fmt.Errorf("spec: %w", err)
	}
	if err := ValidateReceipt(receipt); err != nil {
		return fmt.Errorf("receipt: %w", err)
	}
	digest, err := DigestSpec(spec)
	if err != nil {
		return err
	}
	if receipt.SpecDigest != digest {
		return fmt.Errorf("receipt spec_digest does not match the authored spec: got %q, want %q", receipt.SpecDigest, digest)
	}
	if receipt.Identity.ServerName != spec.ServerName {
		return errors.New("receipt server_name does not match the authored spec")
	}
	if receipt.Artifact.Reference != spec.Artifact.Reference || receipt.Artifact.Digest != spec.Artifact.Digest {
		return errors.New("receipt artifact does not match the authored spec")
	}
	if receipt.Adapter.Name != spec.Adapter.Name {
		return errors.New("receipt adapter does not match the authored spec")
	}
	if receipt.Protocol.Family != spec.Protocol.Family || receipt.Protocol.Revision != spec.Protocol.Revision {
		return errors.New("receipt protocol does not match the authored spec")
	}
	if receipt.Auth != spec.Auth {
		return errors.New("receipt auth reference does not match the authored spec")
	}
	if receipt.Provenance.Spec != spec.Provenance {
		return errors.New("receipt authored provenance does not match the authored spec")
	}
	baseURL, _ := url.Parse(receipt.Endpoint.BaseURL)
	if !net.ParseIP(baseURL.Hostname()).Equal(net.ParseIP(spec.Endpoint.BindHost)) {
		return errors.New("receipt endpoint host does not match the authored spec")
	}
	port, _ := strconv.ParseUint(baseURL.Port(), 10, 16)
	if spec.Endpoint.RequestedPort != 0 && uint16(port) != spec.Endpoint.RequestedPort {
		return errors.New("receipt endpoint port does not match the authored spec")
	}
	observed := make(map[string]struct{}, len(receipt.Protocol.Capabilities))
	for _, capability := range receipt.Protocol.Capabilities {
		observed[capability] = struct{}{}
	}
	for _, required := range spec.Protocol.RequiredCapabilities {
		if _, ok := observed[required]; !ok {
			return fmt.Errorf("receipt is missing required capability %q", required)
		}
	}
	return nil
}

// DigestSpec returns the content address of the canonical schema-v1 spec JSON.
func DigestSpec(spec ServerSpec) (string, error) {
	encoded, err := EncodeSpec(spec)
	if err != nil {
		return "", fmt.Errorf("encode spec for digest: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}

func validateInstanceDirectory(path string) error {
	if err := validateText("instance_directory", path); err != nil {
		return err
	}
	portableAbsolute := strings.HasPrefix(path, "/") && pathpkg.Clean(path) == path
	nativeAbsolute := filepath.IsAbs(path) && filepath.Clean(path) == path
	if !portableAbsolute && !nativeAbsolute {
		return errors.New("instance_directory must be an absolute clean path")
	}
	return nil
}

func validateArtifactReference(path string) error {
	if err := validateText("artifact reference", path); err != nil {
		return err
	}
	portableAbsolute := strings.HasPrefix(path, "/") && pathpkg.Clean(path) == path
	nativeAbsolute := filepath.IsAbs(path) && filepath.Clean(path) == path
	if !portableAbsolute && !nativeAbsolute {
		return errors.New("artifact reference must be an absolute clean local path")
	}
	return nil
}

func validateBindHost(host string) error {
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return errors.New("endpoint bind_host must be a literal loopback IP")
	}
	return nil
}

func validateBaseURL(raw string) error {
	if err := validateText("endpoint base_url", raw); err != nil {
		return err
	}
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "http" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || (u.Path != "" && u.Path != "/") {
		return errors.New("endpoint base_url must be a plain loopback http origin")
	}
	host := u.Hostname()
	if err := validateBindHost(host); err != nil {
		return errors.New("endpoint base_url host must be a literal loopback IP")
	}
	port, err := strconv.ParseUint(u.Port(), 10, 16)
	if err != nil || port == 0 {
		return errors.New("endpoint base_url must include a nonzero port")
	}
	return nil
}

func validateAuth(auth AuthReference) error {
	switch auth.Mode {
	case AuthNone:
		if auth.CredentialRef != "" {
			return errors.New("auth mode none must not carry credential_ref")
		}
	case AuthCredentialRef:
		if !credentialRefPattern.MatchString(auth.CredentialRef) {
			return errors.New("credential_ref must be an uppercase symbolic name, never a credential value")
		}
	default:
		return errors.New("auth mode must be none or credential-ref")
	}
	return nil
}

func validateCapabilities(field string, capabilities []string) error {
	if len(capabilities) == 0 {
		return fmt.Errorf("%s must contain at least one capability", field)
	}
	seen := make(map[string]struct{}, len(capabilities))
	for _, capability := range capabilities {
		if !capabilityPattern.MatchString(capability) {
			return fmt.Errorf("%s contains invalid capability %q", field, capability)
		}
		if _, ok := seen[capability]; ok {
			return fmt.Errorf("%s contains duplicate capability %q", field, capability)
		}
		seen[capability] = struct{}{}
	}
	return nil
}

func validateProvenance(field string, provenance Provenance, wantKind string) error {
	if provenance.Kind != wantKind {
		return fmt.Errorf("%s kind must be %q", field, wantKind)
	}
	return validateText(field+" source", provenance.Source)
}

func validateTimestamp(field, value string) (time.Time, error) {
	if value == "" {
		return time.Time{}, fmt.Errorf("%s is required", field)
	}
	timestamp, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return time.Time{}, fmt.Errorf("%s must be RFC3339: %w", field, err)
	}
	return timestamp, nil
}

func validateText(field, value string) error {
	if value == "" {
		return fmt.Errorf("%s is required", field)
	}
	if strings.TrimSpace(value) != value || strings.IndexFunc(value, func(r rune) bool { return r < 0x20 || r == 0x7f }) >= 0 {
		return fmt.Errorf("%s contains whitespace or control characters at its boundary", field)
	}
	lower := strings.ToLower(value)
	for _, marker := range []string{"bearer ", "-----begin ", "api_key=", "apikey=", "token=", "secret="} {
		if strings.Contains(lower, marker) {
			return fmt.Errorf("%s appears to contain secret material", field)
		}
	}
	if strings.HasPrefix(lower, "sk-") || jwtPattern.MatchString(value) {
		return fmt.Errorf("%s appears to contain secret material", field)
	}
	return nil
}

func canonicalSpec(spec ServerSpec) ServerSpec {
	spec.Protocol.RequiredCapabilities = append([]string(nil), spec.Protocol.RequiredCapabilities...)
	sort.Strings(spec.Protocol.RequiredCapabilities)
	return spec
}

func canonicalReceipt(receipt ServerReceipt) ServerReceipt {
	receipt.Protocol.Capabilities = append([]string(nil), receipt.Protocol.Capabilities...)
	sort.Strings(receipt.Protocol.Capabilities)
	return receipt
}
