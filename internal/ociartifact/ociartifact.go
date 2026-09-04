// Package ociartifact implements the activation-neutral fak collection profile for OCI 1.1.
package ociartifact

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	// ProfileVersion identifies the canonical OCI collection schema profile version string.
	ProfileVersion = "fak.oci.collection/v1"
	// CollectionArtifactType defines the OCI media type identifier for top-level collection manifests.
	CollectionArtifactType = "application/vnd.fak.collection.v1"
	// ManifestMediaType designates standard OCI image manifest serialization envelopes.
	ManifestMediaType = "application/vnd.oci.image.manifest.v1+json"
	// IndexMediaType designates standard OCI image index descriptor structures.
	IndexMediaType = "application/vnd.oci.image.index.v1+json"
	// ConfigMediaType defines the media type used for collection configuration blobs.
	ConfigMediaType = "application/vnd.fak.collection.config.v1+json"
	// SkillMediaType defines the media type identifying packaged agent skill descriptors.
	SkillMediaType = "application/vnd.fak.skill.v1+json"
	// WorkflowMediaType defines the media type identifying declarative workflow execution descriptors.
	WorkflowMediaType = "application/vnd.fak.workflow.v1+json"
	// PolicyMediaType defines the media type identifying runtime capability policy configurations.
	PolicyMediaType = "application/vnd.fak.policy.v1+json"
	// MCPServerMediaType defines the media type identifying Model Context Protocol server packages.
	MCPServerMediaType = "application/vnd.fak.mcp.server.v1+json"
	// SignatureMediaType designates detached Cosign signature payload layers.
	SignatureMediaType = "application/vnd.dev.cosign.simplesigning.v1+json"
	// AttestationMediaType designates in-toto supply-chain attestation statement layers.
	AttestationMediaType = "application/vnd.in-toto+json"
	// SBOMMediaType designates CycloneDX software bill of materials metadata layers.
	SBOMMediaType = "application/vnd.cyclonedx+json"
	// StatementMediaType designates structured evaluation assertion records.
	StatementMediaType = "application/vnd.fak.collection.statement.v1+json"
	annotationTitle    = "org.opencontainers.image.title"
	annotationKind     = "io.fak.object.kind"
)

// Error describes structured operational failures with machine-readable error codes and targeted fields.
type Error struct{ Code, Operation, Field, Message string }

// Error formats the structured failure into a colon-delimited diagnostic message string.
func (e *Error) Error() string                   { return e.Code + ": " + e.Operation + ": " + e.Message }
func fail(code, op, field, message string) error { return &Error{code, op, field, message} }

// Code extracts the machine-readable failure classifier token from an error value if present.
func Code(err error) string {
	var e *Error
	if errors.As(err, &e) {
		return e.Code
	}
	return ""
}

// Descriptor conforms to the OCI content descriptor specification identifying media types and content digests.
type Descriptor struct {
	MediaType    string            `json:"mediaType"`
	Digest       string            `json:"digest"`
	Size         int64             `json:"size"`
	Annotations  map[string]string `json:"annotations,omitempty"`
	ArtifactType string            `json:"artifactType,omitempty"`
}

// Manifest adheres to the OCI image manifest specification version 2 describing layers and configuration.
type Manifest struct {
	SchemaVersion int               `json:"schemaVersion"`
	MediaType     string            `json:"mediaType"`
	ArtifactType  string            `json:"artifactType,omitempty"`
	Config        Descriptor        `json:"config"`
	Layers        []Descriptor      `json:"layers"`
	Subject       *Descriptor       `json:"subject,omitempty"`
	Annotations   map[string]string `json:"annotations,omitempty"`
}

// Object models an individual resource entry bundled within a collection manifest.
type Object struct {
	Name         string   `json:"name"`
	Kind         string   `json:"kind"`
	MediaType    string   `json:"mediaType"`
	Digest       string   `json:"digest"`
	Path         string   `json:"path"`
	Dependencies []string `json:"dependencies,omitempty"`
}

// Config specifies collection metadata, profile schema version, and ordered component object descriptors.
type Config struct {
	Schema  string   `json:"schema"`
	Name    string   `json:"name"`
	Version string   `json:"version"`
	Objects []Object `json:"objects"`
}

// Artifact encapsulates an in-memory OCI collection bundle alongside parsed manifest structures and blob content.
type Artifact struct {
	Manifest    Descriptor
	RawManifest []byte
	Parsed      Manifest
	Blobs       map[string][]byte
}

// Receipt summarizes inspection outcomes, manifest digests, and activation state flags.
type Receipt struct {
	ManifestDigest string   `json:"manifest_digest"`
	LayerDigests   []string `json:"layer_digests"`
	Activated      bool     `json:"activated"`
}

// Digest computes the canonical sha256 hex string with sha256 prefix for arbitrary payload byte slices.
func Digest(b []byte) string { s := sha256.Sum256(b); return "sha256:" + hex.EncodeToString(s[:]) }
func descriptor(mt string, b []byte) Descriptor {
	return Descriptor{MediaType: mt, Digest: Digest(b), Size: int64(len(b))}
}

func validateDescriptor(d Descriptor, op, field string) error {
	if d.MediaType == "" {
		return fail("MEDIA_TYPE_INVALID", op, field+".mediaType", "descriptor media type is empty")
	}
	if d.Size < 0 {
		return fail("SIZE_MISMATCH", op, field+".size", "descriptor size is negative")
	}
	if !validDigest(d.Digest) {
		return fail("DIGEST_INVALID", op, field+".digest", "descriptor digest must be sha256:<64 lowercase hex>")
	}
	return nil
}

func validDigest(d string) bool {
	if len(d) != len("sha256:")+sha256.Size*2 || !strings.HasPrefix(d, "sha256:") {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(d, "sha256:"))
	return err == nil
}

func verifyBlob(d Descriptor, b []byte, op, field string) error {
	if err := validateDescriptor(d, op, field); err != nil {
		return err
	}
	if int64(len(b)) != d.Size {
		return fail("SIZE_MISMATCH", op, field, "descriptor size does not match bytes")
	}
	if Digest(b) != d.Digest {
		return fail("DIGEST_MISMATCH", op, field, "descriptor digest does not match bytes")
	}
	return nil
}
func canonical(v any) ([]byte, error) { return json.Marshal(v) }

// Invariant: collection building preserves content digests deterministically and produces valid OCI manifest layouts.
// Build synthesizes an OCI collection artifact from configuration objects and payload byte maps.
func Build(c Config, payloads map[string][]byte, annotations map[string]string) (Artifact, error) {
	if c.Schema != ProfileVersion {
		return Artifact{}, fail("CONFIG_VERSION", "build", "schema", "unsupported collection profile")
	}
	seen, paths := map[string]bool{}, map[string]bool{}
	layers := make([]Descriptor, 0, len(c.Objects))
	blobs := map[string][]byte{}
	for i := range c.Objects {
		o := &c.Objects[i]
		if o.Name == "" || o.Kind == "" || !validPath(o.Path) || o.MediaType == "" {
			return Artifact{}, fail("OBJECT_INVALID", "build", o.Name, "object requires name, kind, safe relative path, and media type")
		}
		if seen[o.Name] {
			return Artifact{}, fail("OBJECT_DUPLICATE", "build", o.Name, "duplicate object name")
		}
		if paths[o.Path] {
			return Artifact{}, fail("PATH_DUPLICATE", "build", o.Path, "duplicate object path")
		}
		seen[o.Name], paths[o.Path] = true, true
		b, ok := payloads[o.Path]
		if !ok {
			b, ok = payloads[o.Name]
		}
		if !ok {
			return Artifact{}, fail("LAYER_MISSING", "build", o.Name, "native payload missing")
		}
		d := descriptor(o.MediaType, b)
		d.Annotations = map[string]string{annotationTitle: o.Path, annotationKind: o.Kind}
		o.Digest = d.Digest
		layers = append(layers, d)
		blobs[d.Digest] = append([]byte(nil), b...)
	}
	if err := checkDependencies(c); err != nil {
		return Artifact{}, err
	}
	cb, err := canonical(c)
	if err != nil {
		return Artifact{}, err
	}
	cd := descriptor(ConfigMediaType, cb)
	blobs[cd.Digest] = cb
	m := Manifest{SchemaVersion: 2, MediaType: ManifestMediaType, ArtifactType: CollectionArtifactType, Config: cd, Layers: layers, Annotations: cloneMap(annotations)}
	mb, err := canonical(m)
	if err != nil {
		return Artifact{}, err
	}
	md := descriptor(ManifestMediaType, mb)
	md.ArtifactType = CollectionArtifactType
	return Artifact{Manifest: md, RawManifest: mb, Parsed: m, Blobs: blobs}, nil
}

// Precondition: manifest payload and blob maps must provide valid JSON data and sha256 content hashes.
// Postcondition: returns validated artifact records and an inert execution receipt without activating layers.
// Inspect parses and validates an OCI manifest against payload blobs without executing runtime activations.
func Inspect(raw []byte, blobs map[string][]byte) (Artifact, Receipt, error) {
	var m Manifest
	if err := json.Unmarshal(raw, &m); err != nil {
		return Artifact{}, Receipt{}, fail("MANIFEST_INVALID", "inspect", "manifest", err.Error())
	}
	if m.SchemaVersion != 2 || m.MediaType != ManifestMediaType || (m.ArtifactType != CollectionArtifactType && m.ArtifactType != ConfigMediaType && m.Subject == nil) {
		return Artifact{}, Receipt{}, fail("MEDIA_TYPE_INVALID", "inspect", "manifest", "not a fak OCI collection manifest")
	}
	if m.Config.MediaType != ConfigMediaType {
		return Artifact{}, Receipt{}, fail("MEDIA_TYPE_INVALID", "inspect", "config", "unexpected config media type")
	}
	all := append([]Descriptor{m.Config}, m.Layers...)
	for i, d := range all {
		field := fmt.Sprintf("descriptors[%d]", i)
		b, ok := blobs[d.Digest]
		if !ok {
			return Artifact{}, Receipt{}, fail("MISSING_LAYER", "inspect", field, "descriptor blob is absent")
		}
		if err := verifyBlob(d, b, "inspect", field); err != nil {
			return Artifact{}, Receipt{}, err
		}
	}
	if m.Subject != nil {
		if err := validateDescriptor(*m.Subject, "inspect", "subject"); err != nil {
			return Artifact{}, Receipt{}, err
		}
	}
	var c Config
	if err := json.Unmarshal(blobs[m.Config.Digest], &c); err != nil || c.Schema != ProfileVersion {
		return Artifact{}, Receipt{}, fail("CONFIG_INVALID", "inspect", "config", "invalid or unsupported config")
	}
	if len(c.Objects) != len(m.Layers) {
		return Artifact{}, Receipt{}, fail("CONFIG_INVALID", "inspect", "objects", "object/layer count differs")
	}
	ids, paths := map[string]bool{}, map[string]bool{}
	for i, o := range c.Objects {
		if o.Name == "" || o.Kind == "" {
			return Artifact{}, Receipt{}, fail("OBJECT_INVALID", "inspect", fmt.Sprintf("objects[%d]", i), "object name or kind is empty")
		}
		if !validPath(o.Path) {
			return Artifact{}, Receipt{}, fail("PATH_ESCAPE", "inspect", o.Path, "object path is not safe and relative")
		}
		if o.MediaType != m.Layers[i].MediaType || o.Digest != m.Layers[i].Digest {
			return Artifact{}, Receipt{}, fail("UNSUPPORTED_MEDIA_TYPE", "inspect", o.Name, "config object differs from native layer descriptor")
		}
		if ids[o.Name] {
			return Artifact{}, Receipt{}, fail("OBJECT_DUPLICATE", "inspect", o.Name, "duplicate object name")
		}
		if paths[o.Path] {
			return Artifact{}, Receipt{}, fail("PATH_DUPLICATE", "inspect", o.Path, "duplicate object path")
		}
		ids[o.Name], paths[o.Path] = true, true
	}
	if err := checkDependencies(c); err != nil {
		return Artifact{}, Receipt{}, err
	}
	layers := make([]string, len(m.Layers))
	for i := range m.Layers {
		layers[i] = m.Layers[i].Digest
	}
	md := descriptor(ManifestMediaType, raw)
	md.ArtifactType = m.ArtifactType
	return Artifact{Manifest: md, RawManifest: append([]byte(nil), raw...), Parsed: m, Blobs: copyBlobs(blobs)}, Receipt{ManifestDigest: md.Digest, LayerDigests: layers, Activated: false}, nil
}

func checkDependencies(c Config) error {
	nodes := map[string]Object{}
	for _, o := range c.Objects {
		nodes[o.Digest] = o
	}
	state := map[string]int{}
	var visit func(string) error
	visit = func(d string) error {
		if state[d] == 1 {
			return fail("DEPENDENCY_CYCLE", "inspect", d, "dependency graph contains a cycle")
		}
		if state[d] == 2 {
			return nil
		}
		o, ok := nodes[d]
		if !ok {
			return fail("DEPENDENCY_NOT_FOUND", "inspect", d, "dependency does not name an object in the collection")
		}
		state[d] = 1
		for _, x := range o.Dependencies {
			if err := visit(x); err != nil {
				return err
			}
		}
		state[d] = 2
		return nil
	}
	for d := range nodes {
		if err := visit(d); err != nil {
			return err
		}
	}
	return nil
}
func validPath(p string) bool {
	return p != "" && !strings.HasPrefix(p, "/") && !strings.Contains(p, "\\") && filepath.ToSlash(filepath.Clean(p)) == p && p != "." && !strings.HasPrefix(p, "../")
}
func cloneMap(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	n := map[string]string{}
	for k, v := range m {
		n[k] = v
	}
	return n
}
func copyBlobs(m map[string][]byte) map[string][]byte {
	n := map[string][]byte{}
	for k, v := range m {
		n[k] = append([]byte(nil), v...)
	}
	return n
}

// Precondition: artifact manifest and constituent blobs must pass full structural validation before layout export.
// Postcondition: creates an OCI image-layout compliant directory containing verified blobs and index manifest.
// ExportLayout serializes an artifact into a conformant on-disk OCI layout directory hierarchy.
func ExportLayout(dir string, a Artifact) error {
	if _, _, err := Inspect(a.RawManifest, a.Blobs); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(dir, "blobs", "sha256"), 0755); err != nil {
		return err
	}
	all := copyBlobs(a.Blobs)
	all[a.Manifest.Digest] = a.RawManifest
	for d, b := range all {
		if err := os.WriteFile(filepath.Join(dir, "blobs", "sha256", strings.TrimPrefix(d, "sha256:")), b, 0644); err != nil {
			return err
		}
	}
	idx := struct {
		SchemaVersion int          `json:"schemaVersion"`
		MediaType     string       `json:"mediaType"`
		Manifests     []Descriptor `json:"manifests"`
	}{2, IndexMediaType, []Descriptor{a.Manifest}}
	ib, _ := json.Marshal(idx)
	if err := os.WriteFile(filepath.Join(dir, "index.json"), ib, 0644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "oci-layout"), []byte("{\"imageLayoutVersion\":\"1.0.0\"}\n"), 0644)
}

// Precondition: directory must contain index.json, oci-layout, and verified blob storage paths.
// Postcondition: returns the parsed artifact alongside an inert receipt without side-effect activation.
// ImportLayout reads and validates an OCI layout directory returning parsed artifact payloads.
func ImportLayout(dir string) (Artifact, Receipt, error) {
	b, err := os.ReadFile(filepath.Join(dir, "index.json"))
	if err != nil {
		return Artifact{}, Receipt{}, err
	}
	var idx struct {
		Manifests []Descriptor `json:"manifests"`
	}
	if json.Unmarshal(b, &idx) != nil || len(idx.Manifests) != 1 {
		return Artifact{}, Receipt{}, fail("MALFORMED_INDEX", "import-layout", "index.json", "exactly one manifest required")
	}
	read := func(d string) ([]byte, error) {
		return os.ReadFile(filepath.Join(dir, "blobs", "sha256", strings.TrimPrefix(d, "sha256:")))
	}
	raw, err := read(idx.Manifests[0].Digest)
	if err != nil {
		return Artifact{}, Receipt{}, fail("MISSING_LAYER", "import-layout", idx.Manifests[0].Digest, err.Error())
	}
	var m Manifest
	if json.Unmarshal(raw, &m) != nil {
		return Artifact{}, Receipt{}, fail("MALFORMED_MANIFEST", "import-layout", "manifest", "invalid JSON")
	}
	bl := map[string][]byte{}
	for _, d := range append([]Descriptor{m.Config}, m.Layers...) {
		x, e := read(d.Digest)
		if e != nil {
			return Artifact{}, Receipt{}, fail("MISSING_LAYER", "import-layout", d.Digest, e.Error())
		}
		bl[d.Digest] = x
	}
	return Inspect(raw, bl)
}

// Client manages network transport operations against an OCI Distribution 1.1 compliant remote registry.
type Client struct {
	Base       string
	HTTP       *http.Client
	Repository string
}

func (c Client) do(method, path, ct string, body []byte) ([]byte, http.Header, int, error) {
	u := strings.TrimRight(c.Base, "/") + "/v2/" + c.Repository + path
	req, _ := http.NewRequest(method, u, bytes.NewReader(body))
	if ct != "" {
		req.Header.Set("Content-Type", ct)
	}
	r, e := c.client().Do(req)
	if e != nil {
		return nil, nil, 0, e
	}
	defer r.Body.Close()
	b, _ := io.ReadAll(r.Body)
	return b, r.Header, r.StatusCode, nil
}
func (c Client) client() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return &http.Client{Timeout: 30 * time.Second}
}

// Precondition: artifact manifest and layers must pass structural validation before initiating upload.
// Postcondition: blobs and manifest are successfully uploaded to the target repository reference.
// Push uploads artifact blobs and manifests to the configured remote OCI registry endpoint.
func (c Client) Push(reference string, a Artifact) error {
	if _, _, e := Inspect(a.RawManifest, a.Blobs); e != nil {
		return e
	}
	for d, b := range a.Blobs {
		q := "/blobs/uploads/?digest=" + url.QueryEscape(d)
		_, _, s, e := c.do("POST", q, "application/octet-stream", b)
		if e != nil || s/100 != 2 {
			return fail("REGISTRY_ERROR", "push", d, fmt.Sprintf("status %d: %v", s, e))
		}
	}
	_, _, s, e := c.do("PUT", "/manifests/"+url.PathEscape(reference), ManifestMediaType, a.RawManifest)
	if e != nil || s/100 != 2 {
		return fail("REGISTRY_ERROR", "push", reference, fmt.Sprintf("status %d: %v", s, e))
	}
	return nil
}

// Invariant: client registry resolution strictly requires canonical sha256 lowercase digest identifiers.
// Resolve queries the remote registry via HEAD request to determine the content digest of a tag or reference.
func (c Client) Resolve(reference string) (Descriptor, error) {
	_, h, s, e := c.do("HEAD", "/manifests/"+url.PathEscape(reference), "", nil)
	if e != nil || s/100 != 2 {
		return Descriptor{}, fail("REGISTRY_ERROR", "resolve", reference, fmt.Sprintf("status %d: %v", s, e))
	}
	d := h.Get("Docker-Content-Digest")
	if !strings.HasPrefix(d, "sha256:") {
		return Descriptor{}, fail("DIGEST_REQUIRED", "resolve", reference, "registry did not resolve reference to a digest")
	}
	mediaType := strings.TrimSpace(strings.Split(h.Get("Content-Type"), ";")[0])
	if mediaType != ManifestMediaType {
		return Descriptor{}, fail("MEDIA_TYPE_INVALID", "resolve", reference, "registry resolved a non-manifest media type")
	}
	return Descriptor{MediaType: mediaType, Digest: d}, nil
}

// Precondition: reference must be an inspected sha256 digest string preventing ambiguous tag pulls.
// Postcondition: downloads and returns the fully validated artifact alongside an inert receipt.
// Pull downloads manifest descriptors and associated layer blobs by digest from the remote registry.
func (c Client) Pull(reference string) (Artifact, Receipt, error) {
	if !strings.HasPrefix(reference, "sha256:") {
		return Artifact{}, Receipt{}, fail("DIGEST_REQUIRED", "pull", reference, "pull requires an inspected digest")
	}
	raw, _, s, e := c.do("GET", "/manifests/"+url.PathEscape(reference), "", nil)
	if e != nil || s/100 != 2 {
		return Artifact{}, Receipt{}, fail("REGISTRY_ERROR", "pull", reference, fmt.Sprintf("status %d: %v", s, e))
	}
	var m Manifest
	if json.Unmarshal(raw, &m) != nil {
		return Artifact{}, Receipt{}, fail("MALFORMED_MANIFEST", "pull", "manifest", "invalid JSON")
	}
	bl := map[string][]byte{}
	for _, d := range append([]Descriptor{m.Config}, m.Layers...) {
		b, _, st, x := c.do("GET", "/blobs/"+url.PathEscape(d.Digest), "", nil)
		if x != nil || st/100 != 2 {
			return Artifact{}, Receipt{}, fail("MISSING_LAYER", "pull", d.Digest, "registry blob absent")
		}
		bl[d.Digest] = b
	}
	return Inspect(raw, bl)
}

// Precondition: subject must be a valid sha256 digest string of an existing artifact manifest.
// Postcondition: returns all matching content descriptors referring to the specified subject digest.
// Referrers queries the registry for artifacts linked to the specified subject digest via referrers API or fallback tags.
func (c Client) Referrers(subject string) ([]Descriptor, error) {
	parse := func(b []byte) ([]Descriptor, error) {
		var x struct {
			Manifests []Descriptor `json:"manifests"`
		}
		if e := json.Unmarshal(b, &x); e != nil {
			return nil, e
		}
		return x.Manifests, nil
	}
	b, _, s, e := c.do("GET", "/referrers/"+url.PathEscape(subject), "", nil)
	if e == nil && s/100 == 2 {
		return parse(b)
	}
	tag := strings.Replace(subject, ":", "-", 1)
	b, _, s, e = c.do("GET", "/manifests/"+url.PathEscape(tag), "", nil)
	if e != nil || s/100 != 2 {
		return nil, fail("REFERRERS_UNAVAILABLE", "referrers", subject, "standard API and sha256-<hex> fallback unavailable")
	}
	return parse(b)
}

// MCPServer models the external server configuration schema for Model Context Protocol service definitions.
type MCPServer struct {
	Schema     string          `json:"$schema,omitempty"`
	Name       string          `json:"name"`
	Version    string          `json:"version"`
	Packages   json.RawMessage `json:"packages,omitempty"`
	Remotes    json.RawMessage `json:"remotes,omitempty"`
	Repository json.RawMessage `json:"repository,omitempty"`
	Status     json.RawMessage `json:"status,omitempty"`
}

// Precondition: kind parameter must equal "mcp-service" and payload must contain valid JSON bytes.
// Postcondition: returns the populated and validated MCPServer configuration structure.
// ImportServerJSON parses and validates raw server.json payloads into typed MCPServer structures.
func ImportServerJSON(kind string, b []byte) (MCPServer, error) {
	if kind != "mcp-service" {
		return MCPServer{}, fail("BRIDGE_SCOPE", "server-json", "kind", "server.json is only valid for MCP service objects")
	}
	var s MCPServer
	if e := json.Unmarshal(b, &s); e != nil {
		return s, fail("MALFORMED_SERVER_JSON", "server-json", "payload", e.Error())
	}
	if s.Name == "" || s.Version == "" {
		return s, fail("MALFORMED_SERVER_JSON", "server-json", "identity", "name and version required")
	}
	return s, nil
}

// Precondition: kind parameter must equal "mcp-service" and server configuration must be initialized.
// Postcondition: returns indented JSON configuration bytes representing the MCP server definition.
// ExportServerJSON formats typed MCPServer structures into indented server.json configuration bytes.
func ExportServerJSON(kind string, s MCPServer) ([]byte, error) {
	if kind != "mcp-service" {
		return nil, fail("BRIDGE_SCOPE", "server-json", "kind", "server.json is only valid for MCP service objects")
	}
	return json.MarshalIndent(s, "", "  ")
}
