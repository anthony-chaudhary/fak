package gateway

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// specPathFor maps each ServeMux registration pattern in routeTable() to the
// exact OpenAPI path key the committed spec (docs/fak/openapi.yaml) must
// document for it. A subtree pattern (a trailing "/") maps to its templated
// OpenAPI path.
//
// Keeping this mapping in the test — keyed off the live route table rather than
// a free-standing list — is what makes the gate non-circular: a route added to
// routeTable() with no entry here fails TestServedRouteMappingIsExhaustive, and
// a mapped route the spec does not document fails
// TestOpenAPISpecDocumentsEveryServedRoute. Together they force a new served
// endpoint, its mapping, and its OpenAPI path to land in the same change.
//
// This is the F-007 (#205) "Generated OpenAPI spec" acceptance gate: the spec
// the Python/TypeScript/Go client SDKs are generated from cannot silently drift
// behind the surface `fak serve` actually exposes.
var specPathFor = map[string]string{
	"/v1/chat/completions":      "/v1/chat/completions",
	"/v1/embeddings":            "/v1/embeddings",
	"/v1/moderations":           "/v1/moderations",
	"/v1/messages":              "/v1/messages",
	"/v1/messages/count_tokens": "/v1/messages/count_tokens",
	"/v1beta/":                  "/v1beta/models/{model}:generateContent",
	"/v1/fak/syscall":           "/v1/fak/syscall",
	"/v1/fak/adjudicate":        "/v1/fak/adjudicate",
	"/v1/fak/admit":             "/v1/fak/admit",
	"/v1/fak/changes":           "/v1/fak/changes",
	"/v1/fak/events":            "/v1/fak/events",
	"/v1/fak/revoke":            "/v1/fak/revoke",
	"/v1/fak/context/change":    "/v1/fak/context/change",
	"/v1/fak/policy/reload":     "/v1/fak/policy/reload",
	"/v1/fak/trace/reset":       "/v1/fak/trace/reset",
	"/v1/fak/trace/":            "/v1/fak/trace/{trace_id}",
	"/v1/fak/session/":          "/v1/fak/session/{trace_id}",
	"/v1/fak/sessions":          "/v1/fak/sessions",
	"/v1/fak/tasks":             "/v1/fak/tasks",
	"/v1/models":                "/v1/models",
	"/mcp":                      "/mcp",
	"/healthz":                  "/healthz",
	"/metrics":                  "/metrics",
	"/debug/vars":               "/debug/vars",
}

// openAPISpecPath is the committed OpenAPI document, relative to this package.
const openAPISpecPath = "../../docs/fak/openapi.yaml"

// TestServedRouteMappingIsExhaustive fails if routeTable() registers a route
// that specPathFor does not map. It is the first half of the drift gate: a new
// endpoint must declare which OpenAPI path documents it (and, via the sibling
// test, that path must actually exist in the spec).
func TestServedRouteMappingIsExhaustive(t *testing.T) {
	for _, rt := range (&Server{}).routeTable() {
		if _, ok := specPathFor[rt.pattern]; !ok {
			t.Errorf("route %q is served by routeTable() but unmapped in specPathFor — add the mapping AND document the path in %s",
				rt.pattern, openAPISpecPath)
		}
	}
}

// TestOpenAPISpecDocumentsEveryServedRoute is the second half of the gate: every
// route the gateway actually serves must appear as a path key in the committed
// OpenAPI spec. This keeps the generated-SDK source of truth honest against the
// live HTTP surface (#205, F-007).
func TestOpenAPISpecDocumentsEveryServedRoute(t *testing.T) {
	raw, err := os.ReadFile(filepath.FromSlash(openAPISpecPath))
	if err != nil {
		t.Fatalf("read %s: %v", openAPISpecPath, err)
	}
	spec := string(raw)

	for _, rt := range (&Server{}).routeTable() {
		want, ok := specPathFor[rt.pattern]
		if !ok {
			// Reported by TestServedRouteMappingIsExhaustive; skip here.
			continue
		}
		if !specHasPathKey(spec, want) {
			t.Errorf("docs/fak/openapi.yaml does not document served route %q (expected an OpenAPI path key %q)", rt.pattern, want)
		}
	}
}

// specHasPathKey reports whether the OpenAPI document declares path as a mapping
// key (i.e. a `paths:` entry). It deliberately avoids a YAML dependency (the repo
// is zero-dep): a path key is the only "/"-leading mapping key in the document,
// so a line that — once indentation, the key-terminating colon, and any
// surrounding quotes are stripped — equals the path is an unambiguous match.
// This tolerates both bare (`/v1/models:`) and quoted
// (`'/v1beta/models/{model}:generateContent':`) key styles.
func specHasPathKey(spec, path string) bool {
	for _, line := range strings.Split(spec, "\n") {
		key := strings.TrimSpace(line)
		key = strings.TrimSuffix(key, ":")
		key = strings.Trim(key, "'\"")
		if key == path {
			return true
		}
	}
	return false
}
