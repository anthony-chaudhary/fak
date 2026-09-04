// Package managedinventory owns the registered managed-agent object taxonomy and
// the deterministic portability inventory used by the portability spine.
package managedinventory

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strings"
)

const (
	// Schema identifies the versioned managed-agent object inventory format.
	Schema = "fak-managed-agent-object-inventory/v1"
	// DefaultSourceRel is the repository-relative path to the canonical JSON inventory source.
	DefaultSourceRel = "docs/portability/managed-agent-objects.json"
	// DefaultReportRel is the repository-relative path to the generated Markdown report.
	DefaultReportRel = "docs/portability/managed-agent-object-inventory.md"
)

// Catalog is the authored portability source. The Markdown report is a pure
// projection of this value; it is never an independent source of truth.
type Catalog struct {
	Schema                   string                   `json:"schema"`
	Discovery                Discovery                `json:"discovery"`
	RepresentativeCollection RepresentativeCollection `json:"representative_collection"`
	Objects                  []Object                 `json:"objects"`
}

// Discovery makes the repository archaeology reproducible at one immutable git
// revision. The checker replays every query and compares both line and file counts.
type Discovery struct {
	Revision string           `json:"revision"`
	Scope    []string         `json:"scope"`
	Queries  []DiscoveryQuery `json:"queries"`
}

// DiscoveryQuery records one reproducible repository search query and its expected matches.
type DiscoveryQuery struct {
	ID            string   `json:"id"`
	Pattern       string   `json:"pattern"`
	Paths         []string `json:"paths"`
	ExpectedLines int      `json:"expected_lines"`
	ExpectedFiles int      `json:"expected_files"`
}

// RepresentativeCollection records the minimal reference subset of managed object types.
type RepresentativeCollection struct {
	IDs       []string `json:"ids"`
	Rationale string   `json:"rationale"`
}

// Object is one managed-agent object type, not one instance. Storage roots use
// symbolic, host-neutral paths; no live path or credential value belongs here.
type Object struct {
	ID              string   `json:"id"`
	Name            string   `json:"name"`
	Owner           string   `json:"owner"`
	Scope           string   `json:"scope"`
	StorageRoots    []string `json:"storage_roots"`
	IdentityVersion string   `json:"identity_version"`
	Sensitivity     string   `json:"sensitivity"`
	Dependencies    []string `json:"dependencies"`
	Lifecycle       string   `json:"lifecycle"`
	Export          string   `json:"export"`
	Import          string   `json:"import"`
	Sync            string   `json:"sync"`
	Portability     string   `json:"portability"`
	Gap             string   `json:"gap"`
	Evidence        []string `json:"evidence"`
}

// Registration is deliberately separate from an inventory row. A future object
// adapter must register its type here and add a catalog row; CI fails in between.
type Registration struct {
	ID string
}

var registrations = []Registration{
	{ID: "checkpoint"},
	{ID: "credential-reference"},
	{ID: "harness-profile"},
	{ID: "history"},
	{ID: "lease"},
	{ID: "loop"},
	{ID: "memory"},
	{ID: "model-binding"},
	{ID: "plan"},
	{ID: "policy"},
	{ID: "receipt"},
	{ID: "session"},
	{ID: "skill"},
	{ID: "task"},
	{ID: "tool-binding"},
	{ID: "trajectory"},
	{ID: "workflow"},
}

// Registrations returns a copy of the registered managed object types.
func Registrations() []Registration {
	out := make([]Registration, len(registrations))
	copy(out, registrations)
	return out
}

// Load reads and parses a Catalog from a JSON file, rejecting unknown or trailing data.
func Load(path string) (Catalog, error) {
	var c Catalog
	b, err := os.ReadFile(path)
	if err != nil {
		return c, err
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if err := d.Decode(&c); err != nil {
		return c, fmt.Errorf("decode managed inventory: %w", err)
	}
	var extra any
	if err := d.Decode(&extra); err != io.EOF {
		if err == nil {
			return c, fmt.Errorf("decode managed inventory: trailing JSON value")
		}
		return c, fmt.Errorf("decode managed inventory trailing data: %w", err)
	}
	return c, nil
}

// Diagnostic describes a validation failure or structural inconsistency in a catalog.
type Diagnostic struct {
	Code   string
	Object string
	Field  string
	Detail string
}

// Error formats the diagnostic code, location, and detail.
func (d Diagnostic) Error() string {
	where := d.Field
	if d.Object != "" {
		where = "object " + d.Object + "." + d.Field
	}
	return fmt.Sprintf("%s: %s: %s", d.Code, where, d.Detail)
}

var fullSHA = regexp.MustCompile(`^[0-9a-f]{40}$`)
var hostAbsolutePath = regexp.MustCompile(`(?i)([a-z]:[\\/]|/(home|users|root)/[^<[:space:]]+)`)

var allowedPortability = map[string]bool{
	"portable": true, "partial": true, "local-only": true, "unknown": true,
}

var allowedSensitivity = map[string]bool{
	"public": true, "internal": true, "sensitive": true,
	"secret-reference": true, "mixed": true,
}

// Validate checks total registration coverage, closed vocabularies, dependency
// closure, required fields, and the public-artifact path/secret floor.
func Validate(c Catalog, regs []Registration) []Diagnostic {
	var out []Diagnostic
	add := func(code, object, field, detail string) {
		out = append(out, Diagnostic{Code: code, Object: object, Field: field, Detail: detail})
	}
	if c.Schema != Schema {
		add("SCHEMA", "", "schema", fmt.Sprintf("got %q, want %q", c.Schema, Schema))
	}
	// The authored JSON is public. Check the complete value, not only storage_roots,
	// so a host path or credential shape cannot hide in rationale or lifecycle prose.
	if raw, err := json.Marshal(c); err == nil {
		text := string(raw)
		if containsSecretShape(text) || hostAbsolutePath.MatchString(text) {
			add("PUBLIC_ARTIFACT_LEAK", "", "catalog", "remove host-absolute paths, credential shapes, and private-repository references")
		}
	}
	if !fullSHA.MatchString(c.Discovery.Revision) {
		add("DISCOVERY_REVISION", "", "discovery.revision", "record one full lowercase git commit SHA")
	}
	if len(c.Discovery.Scope) == 0 {
		add("DISCOVERY_SCOPE", "", "discovery.scope", "record the repository paths searched")
	}
	queryIDs := map[string]bool{}
	for i, q := range c.Discovery.Queries {
		field := fmt.Sprintf("discovery.queries[%d]", i)
		if strings.TrimSpace(q.ID) == "" || strings.TrimSpace(q.Pattern) == "" || len(q.Paths) == 0 {
			add("DISCOVERY_QUERY", "", field, "id, pattern, and paths are required")
		}
		if queryIDs[q.ID] {
			add("DUPLICATE_DISCOVERY_QUERY", "", field, "query id "+q.ID+" appears more than once")
		}
		queryIDs[q.ID] = true
		if q.ExpectedLines <= 0 || q.ExpectedFiles <= 0 {
			add("DISCOVERY_COUNTS", "", field, "expected line and file counts must be positive")
		}
		for _, p := range q.Paths {
			if unsafeArtifactText(p) {
				add("UNSAFE_PATH", "", field, "query path must be repository-relative and public-safe: "+p)
			}
		}
	}
	if len(c.Discovery.Queries) == 0 {
		add("DISCOVERY_QUERY", "", "discovery.queries", "record at least one reproducible search query")
	}

	regSeen := map[string]bool{}
	for _, r := range regs {
		id := strings.TrimSpace(r.ID)
		if id == "" {
			add("EMPTY_REGISTRATION", "", "registrations", "registered type id is empty")
			continue
		}
		if regSeen[id] {
			add("DUPLICATE_REGISTRATION", id, "id", "registered more than once")
		}
		regSeen[id] = true
	}

	rows := map[string]Object{}
	for i, o := range c.Objects {
		id := strings.TrimSpace(o.ID)
		if id == "" {
			add("MISSING_FIELD", fmt.Sprintf("row-%d", i), "id", "stable type id is required")
			continue
		}
		if _, ok := rows[id]; ok {
			add("DUPLICATE_ROW", id, "id", "inventory contains more than one row")
		}
		rows[id] = o
		if !regSeen[id] {
			add("UNREGISTERED_ROW", id, "id", "inventory row has no managed-type registration")
		}
		required := map[string]string{
			"name": o.Name, "owner": o.Owner, "scope": o.Scope,
			"identity_version": o.IdentityVersion, "sensitivity": o.Sensitivity,
			"lifecycle": o.Lifecycle, "export": o.Export, "import": o.Import,
			"sync": o.Sync, "portability": o.Portability, "gap": o.Gap,
		}
		for field, value := range required {
			if strings.TrimSpace(value) == "" {
				add("MISSING_FIELD", id, field, "explicit value is required")
			}
		}
		if o.StorageRoots == nil || len(o.StorageRoots) == 0 {
			add("MISSING_FIELD", id, "storage_roots", "record at least one symbolic root, including runtime-memory when appropriate")
		}
		if o.Dependencies == nil {
			add("MISSING_FIELD", id, "dependencies", "record [] when the type has no managed-type dependency")
		}
		if len(o.Evidence) == 0 {
			add("MISSING_FIELD", id, "evidence", "record at least one repository-relative grounding path")
		}
		if !allowedPortability[o.Portability] {
			add("PORTABILITY_STATUS", id, "portability", "use portable|partial|local-only|unknown")
		}
		if !allowedSensitivity[o.Sensitivity] {
			add("SENSITIVITY_CLASS", id, "sensitivity", "use public|internal|sensitive|secret-reference|mixed")
		}
		for _, value := range append(append([]string{}, o.StorageRoots...), o.Evidence...) {
			if unsafeArtifactText(value) {
				add("UNSAFE_PATH", id, "storage_roots/evidence", "use a public-safe repository-relative or symbolic path: "+value)
			}
		}
		for _, value := range []string{o.IdentityVersion, o.Lifecycle, o.Export, o.Import, o.Sync, o.Gap} {
			if containsSecretShape(value) {
				add("SECRET_SHAPE", id, "text", "credential/private-repository shaped content is forbidden")
			}
		}
	}
	for id := range regSeen {
		if _, ok := rows[id]; !ok {
			add("MISSING_INVENTORY_ROW", id, "id", "registered managed type has no inventory row")
		}
	}
	for id, o := range rows {
		for _, dep := range o.Dependencies {
			if !regSeen[dep] {
				add("UNKNOWN_DEPENDENCY", id, "dependencies", "dependency "+dep+" is not a registered managed type")
			}
		}
	}
	if len(c.RepresentativeCollection.IDs) == 0 || strings.TrimSpace(c.RepresentativeCollection.Rationale) == "" {
		add("SPINE_COLLECTION", "", "representative_collection", "ids and smallest-set rationale are required")
	}
	spineSeen := map[string]bool{}
	for _, id := range c.RepresentativeCollection.IDs {
		if spineSeen[id] {
			add("SPINE_COLLECTION", id, "representative_collection.ids", "duplicate type")
		}
		spineSeen[id] = true
		if _, ok := rows[id]; !ok {
			add("SPINE_COLLECTION", id, "representative_collection.ids", "type has no inventory row")
		}
	}

	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		if a.Code != b.Code {
			return a.Code < b.Code
		}
		if a.Object != b.Object {
			return a.Object < b.Object
		}
		if a.Field != b.Field {
			return a.Field < b.Field
		}
		return a.Detail < b.Detail
	})
	return out
}

func unsafeArtifactText(s string) bool {
	n := strings.ToLower(strings.TrimSpace(strings.ReplaceAll(s, `\`, "/")))
	return n == "" || strings.HasPrefix(n, "/") || strings.HasPrefix(n, "../") ||
		strings.Contains(n, "/../") || regexp.MustCompile(`^[a-z]:/`).MatchString(n) ||
		strings.Contains(n, "fak-private") || strings.Contains(n, "anthony-chaudhary/fak-private")
}

func containsSecretShape(s string) bool {
	n := strings.ToLower(s)
	return strings.Contains(n, "-----begin private key-----") ||
		strings.Contains(n, "sk-ant-") || strings.Contains(n, "sk-proj-") ||
		strings.Contains(n, "ghp_") || strings.Contains(n, "github_pat_") ||
		strings.Contains(n, "anthony-chaudhary/fak-private")
}

// CountGrepOutput returns the stable line/file cardinality of `git grep -n`
// output. A revision-prefixed row has the form REV:path:line:match.
func CountGrepOutput(out []byte) (lines, files int) {
	set := map[string]bool{}
	for _, raw := range bytes.Split(out, []byte{'\n'}) {
		if len(raw) == 0 {
			continue
		}
		lines++
		parts := bytes.SplitN(raw, []byte{':'}, 4)
		if len(parts) >= 3 {
			set[string(parts[1])] = true
		}
	}
	return lines, len(set)
}
