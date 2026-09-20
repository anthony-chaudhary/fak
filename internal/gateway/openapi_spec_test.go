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
	"/": "/",
	// A2A Agent-to-Agent protocol surface (#1019).
	"/a2a/v1/messages":   "/a2a/v1/messages",
	"/a2a/v1/tasks":      "/a2a/v1/tasks",
	"/a2a/v1/agent-card": "/a2a/v1/agent-card",
	"/a2a/v1/tasks/":     "/a2a/v1/tasks/{task_id}",
	// OpenAI-compatible surface.
	"/v1/chat/completions":          "/v1/chat/completions",
	"/v1/completions":               "/v1/completions",
	"/v1/responses":                 "/v1/responses",
	"/v1/embeddings":                "/v1/embeddings",
	"/v1/moderations":               "/v1/moderations",
	"/v1/messages":                  "/v1/messages",
	"/v1/messages/count_tokens":     "/v1/messages/count_tokens",
	"/v1beta/":                      "/v1beta/models/{model}:generateContent",
	"/v1/fak/syscall":               "/v1/fak/syscall",
	"/v1/fak/features":              "/v1/fak/features",
	"/v1/fak/adjudicate":            "/v1/fak/adjudicate",
	"/v1/fak/admit":                 "/v1/fak/admit",
	"/v1/fak/cache/posture":         "/v1/fak/cache/posture",
	"/v1/fak/changes":               "/v1/fak/changes",
	"/v1/fak/events":                "/v1/fak/events",
	"/v1/fak/vcache/score":          "/v1/fak/vcache/score",
	"/v1/fak/vcache/actions":        "/v1/fak/vcache/actions",
	"/v1/fak/usage/cache-alignment": "/v1/fak/usage/cache-alignment",
	"/v1/fak/session-audit/actions": "/v1/fak/session-audit/actions",
	"/v1/fak/ctxvalue":              "/v1/fak/ctxvalue",
	"/v1/fak/revoke":                "/v1/fak/revoke",
	"/v1/fak/context/change":        "/v1/fak/context/change",
	"/v1/fak/policy":                "/v1/fak/policy",
	"/v1/fak/policy/reload":         "/v1/fak/policy/reload",
	"/v1/control/config":            "/v1/control/config",
	"/v1/fak/control/config":        "/v1/fak/control/config",
	"/v1/control/apply":             "/v1/control/apply",
	"/v1/fak/control/apply":         "/v1/fak/control/apply",
	"/v1/control/events":            "/v1/control/events",
	"/v1/fak/control/events":        "/v1/fak/control/events",
	"/v1/control/telemetry":         "/v1/control/telemetry",
	"/v1/fak/control/telemetry":     "/v1/fak/control/telemetry",
	"/v1/fak/route/reload":          "/v1/fak/route/reload",
	"/v1/fak/trace/reset":           "/v1/fak/trace/reset",
	"/v1/fak/trace/":                "/v1/fak/trace/{trace_id}",
	"/v1/fak/session/changes":       "/v1/fak/session/changes",
	"/v1/fak/session/":              "/v1/fak/session/{trace_id}",
	"/v1/fak/fleet":                 "/v1/fak/fleet",
	"/v1/fak/sessions":              "/v1/fak/sessions",
	"/v1/fak/observation":           "/v1/fak/observation",
	"/v1/fak/observation/requests":  "/v1/fak/observation/requests",
	"/v1/fak/features/proof":        "/v1/fak/features/proof",
	"/v1/fak/loops":                 "/v1/fak/loops",
	"/v1/fak/tasks":                 "/v1/fak/tasks",
	"/v1/fak/sharedtask/":           "/v1/fak/sharedtask/{task_id}",
	"/v1/fak/agent/sessions":        "/v1/fak/agent/sessions",
	"/v1/fak/account/rehome":        "/v1/fak/account/rehome",
	"/v1/fak/discovery/":            "/v1/fak/discovery/{session_id}",
	"/v1/models":                    "/v1/models",
	// Multi-node dev-server read plane (#2297).
	"/v1/leases": "/v1/leases",
	// Multi-node dev-server write plane (#2299): POST /v1/leases/{acquire,renew,release}.
	"/v1/leases/":        "/v1/leases/{op}",
	"/v1/sessions":       "/v1/sessions",
	"/mcp":               "/mcp",
	"/healthz":           "/healthz",
	"/metrics":           "/metrics",
	"/debug/vars":        "/debug/vars",
	"/debug/guard-audit": "/debug/guard-audit",
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

// TestNativeTokenizeOpenAPIContract is the focused schema-first witness for
// fak#13382: it declares the shipped `POST /v1/fak/tokenize` wire in the
// committed OpenAPI document WITHOUT registering a served route.
//
// It proves the declaration is complete and honest:
//   - the path exists and documents POST (the shipped `fak up` method);
//   - the request schema fixes the 1 MiB body bound;
//   - the response schema declares the identity fields, the exact token
//     IDs/count, and the rendered-prompt hash;
//   - the source prompt text is never echoed by the declared response; and
//   - the ordinary-serving adapter subset is declared as its own fence.
//
// routeTable() must remain UNCHANGED: schema-first means the contract lands
// before the route, and gateway invokability is tracked separately (#13389).
// TestServedRouteMappingIsExhaustive/TestOpenAPISpecDocumentsEveryServedRoute
// stay green because the extra path is a permitted addition, not a served one.
func TestNativeTokenizeOpenAPIContract(t *testing.T) {
	raw, err := os.ReadFile(filepath.FromSlash(openAPISpecPath))
	if err != nil {
		t.Fatalf("read %s: %v", openAPISpecPath, err)
	}
	spec := string(raw)

	// The path and method are declared, matching the shipped `fak up` wire.
	if !specHasPathKey(spec, "/v1/fak/tokenize") {
		t.Fatalf("docs/fak/openapi.yaml does not declare the /v1/fak/tokenize path")
	}
	if !specDeclaresPostForPath(spec, "/v1/fak/tokenize") {
		t.Errorf("docs/fak/openapi.yaml declares /v1/fak/tokenize but not a POST operation")
	}

	// The request schema is declared, fixes the 1 MiB bound, and is wired to
	// the path's requestBody.
	for _, want := range []string{
		"NativeTokenizeRequest:",
		"1 MiB",
		"'#/components/schemas/NativeTokenizeRequest'",
	} {
		if !strings.Contains(spec, want) {
			t.Errorf("docs/fak/openapi.yaml is missing %q for the native tokenize request", want)
		}
	}

	// The response schema is declared and wires the path's 200 response.
	if !strings.Contains(spec, "PromptEncoding:") {
		t.Errorf("docs/fak/openapi.yaml is missing the PromptEncoding schema")
	}
	if !strings.Contains(spec, "'#/components/schemas/PromptEncoding'") {
		t.Errorf("docs/fak/openapi.yaml does not reference PromptEncoding from the tokenize 200 response")
	}

	// The response declares the identity fields, exact token IDs/count, and the
	// rendered hash — and declares that the source prompt is never echoed.
	promptEncoding := specSchemaBlock(spec, "PromptEncoding")
	if promptEncoding == "" {
		t.Fatalf("PromptEncoding schema block not found")
	}
	for _, field := range []string{
		"model_id:", "renderer_id:", "tokenizer_id:",
		"token_ids:", "prompt_tokens:",
		"context_window_tokens:", "reserved_output_tokens:",
		"rendered_sha256:",
	} {
		if !strings.Contains(promptEncoding, field) {
			t.Errorf("PromptEncoding schema does not declare %q", field)
		}
	}
	if !strings.Contains(promptEncoding, "equals `len(token_ids)`") {
		t.Errorf("PromptEncoding must tie prompt_tokens to the exact len(token_ids)")
	}
	if !strings.Contains(promptEncoding, "NEVER echoed") {
		t.Errorf("PromptEncoding must declare that the source prompt text is never echoed")
	}

	// The ordinary-serving adapter subset is declared as its own fence.
	req := specSchemaBlock(spec, "NativeTokenizeRequest")
	if !strings.Contains(req, "ordinary-serving adapter subset") {
		t.Errorf("NativeTokenizeRequest must declare the ordinary-serving adapter subset")
	}

	// The leaf is schema-first: it must NOT register the route. A served route
	// without a specPathFor entry fails the sibling exhaustiveness test.
	for _, rt := range (&Server{}).routeTable() {
		if rt.pattern == "/v1/fak/tokenize" {
			t.Errorf("routeTable() registers /v1/fak/tokenize; #13382 is schema-only (registration is #13389)")
		}
	}
}

// specDeclaresPostForPath reports whether the OpenAPI document declares a POST
// operation under the given path key. It scans for the path key line and then,
// before the next sibling path key, looks for an operation line `post:`.
func specDeclaresPostForPath(spec, path string) bool {
	lines := strings.Split(spec, "\n")
	inPath := false
	for _, line := range lines {
		key := strings.TrimSpace(line)
		trimmed := strings.Trim(strings.TrimSuffix(key, ":"), "'\"")
		switch {
		case trimmed == path && strings.HasSuffix(key, ":"):
			inPath = true
		case inPath && strings.HasPrefix(key, "/") && strings.HasSuffix(key, ":"):
			// Reached the next sibling path key without finding POST.
			return false
		case inPath && key == "post:":
			return true
		}
	}
	return false
}

// specSchemaBlock returns the text of a components/schemas block (from its
// `Name:` line up to the next schema at the same indentation), or "" if absent.
func specSchemaBlock(spec, name string) string {
	lines := strings.Split(spec, "\n")
	start := -1
	for i, line := range lines {
		if strings.TrimSpace(line) == name+":" {
			start = i
			break
		}
	}
	if start < 0 {
		return ""
	}
	end := len(lines)
	for i := start + 1; i < len(lines); i++ {
		trimmed := strings.TrimSpace(lines[i])
		if trimmed == "" {
			continue
		}
		// A sibling schema is indented exactly like the schema name line.
		if len(lines[i])-len(strings.TrimLeft(lines[i], " ")) ==
			len(lines[start])-len(strings.TrimLeft(lines[start], " ")) &&
			strings.HasSuffix(trimmed, ":") {
			end = i
			break
		}
	}
	return strings.Join(lines[start:end], "\n")
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
