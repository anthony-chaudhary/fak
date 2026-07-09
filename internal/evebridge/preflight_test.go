package evebridge

import (
	"testing"
)

// hasDiag reports whether the report carries a diagnostic with the given code
// (and, when non-empty, connection/operation).
func hasDiag(t *testing.T, r Report, code, conn, op string) bool {
	t.Helper()
	for _, d := range r.Diagnostics {
		if d.Code != code {
			continue
		}
		if conn != "" && d.Connection != conn {
			continue
		}
		if op != "" && d.Operation != op {
			continue
		}
		if d.Remediation == "" {
			t.Fatalf("diagnostic %s has no remediation text: %+v", code, d)
		}
		return true
	}
	return false
}

// TestPreflightNoAuthRemoteFails is the first acceptance fixture: a connection
// with no auth and a non-local URL fails with the typed EVE_CONNECTION_NO_AUTH
// diagnostic and admits nothing.
func TestPreflightNoAuthRemoteFails(t *testing.T) {
	m := Manifest{Connections: []Connection{{
		Name: "crm", Type: "openapi", URL: "https://api.example.com/v1", Auth: "no-auth",
		Operations: []Operation{{Name: "list_contacts", Method: "GET"}},
	}}}
	r := Preflight(m, Options{})
	if r.OK {
		t.Fatalf("expected fail-closed report, got OK: %+v", r)
	}
	if !hasDiag(t, r, CodeNoAuth, "crm", "") {
		t.Fatalf("expected %s for crm, got %+v", CodeNoAuth, r.Diagnostics)
	}
	if len(r.AdmittedTools) != 0 {
		t.Fatalf("a red preflight must admit no tools, got %v", r.AdmittedTools)
	}
}

// TestPreflightNoAuthLocalPasses is the counter-fixture the confusion-risk
// note demands: the same no-auth posture on a genuinely local endpoint passes.
func TestPreflightNoAuthLocalPasses(t *testing.T) {
	for _, u := range []string{"http://127.0.0.1:8123/mcp", "http://localhost:9000", "unix:///tmp/eve.sock", ""} {
		m := Manifest{Connections: []Connection{{
			Name: "localtool", Type: "mcp", URL: u, Auth: "local",
			Operations: []Operation{{Name: "list_things", ReadOnly: true}},
		}}}
		r := Preflight(m, Options{})
		if !r.OK {
			t.Fatalf("local no-auth connection at %q should pass, got %+v", u, r.Diagnostics)
		}
		if len(r.AdmittedTools) != 1 || r.AdmittedTools[0] != "localtool__list_things" {
			t.Fatalf("expected the exact admitted namespace [localtool__list_things] for %q, got %v", u, r.AdmittedTools)
		}
	}
}

// TestPreflightAuthUndeclaredFails: an implicit posture is refused.
func TestPreflightAuthUndeclaredFails(t *testing.T) {
	m := Manifest{Connections: []Connection{{
		Name: "vague", Type: "mcp", URL: "https://mcp.example.com", Auth: "",
	}}}
	r := Preflight(m, Options{})
	if r.OK || !hasDiag(t, r, CodeAuthUndeclared, "vague", "") {
		t.Fatalf("expected %s, got %+v", CodeAuthUndeclared, r.Diagnostics)
	}
}

// TestPreflightMutatingWithoutApprovalFailsClosed is the second acceptance
// fixture: mutating operations without approval or allowlist fail closed,
// whether declared by MCP annotation, HTTP method, or operation-name verb.
func TestPreflightMutatingWithoutApprovalFailsClosed(t *testing.T) {
	ops := []Operation{
		{Name: "delete_contact", Method: "DELETE"}, // openapi method
		{Name: "sendMessage"},                      // mcp name verb (camelCase)
		{Name: "innocuous", Mutating: true},        // explicit annotation
		{Name: "read_secrets", Sensitive: true},    // sensitive-read
	}
	for _, op := range ops {
		m := Manifest{Connections: []Connection{{
			Name: "crm", Type: "openapi", URL: "https://api.example.com", Auth: "static",
			Operations: []Operation{op},
		}}}
		r := Preflight(m, Options{})
		if r.OK {
			t.Fatalf("mutating op %+v without approval/allowlist must fail closed", op)
		}
		if !hasDiag(t, r, CodeMutationUnapproved, "crm", op.Name) {
			t.Fatalf("expected %s for %q, got %+v", CodeMutationUnapproved, op.Name, r.Diagnostics)
		}
		if len(r.AdmittedTools) != 0 {
			t.Fatalf("fail-closed report must admit nothing, got %v", r.AdmittedTools)
		}
	}
}

// TestPreflightMutatingGates: each of the three sanctioned gates (approval
// policy, exact allowlist entry, fak policy override) admits the mutation.
func TestPreflightMutatingGates(t *testing.T) {
	base := Connection{
		Name: "shop", Type: "openapi", URL: "https://api.shop.example", Auth: "app",
		Operations: []Operation{{Name: "purchase_item", Method: "POST"}},
	}
	cases := []struct {
		name string
		mut  func(*Connection)
		opts Options
	}{
		{"approval always", func(c *Connection) { c.Approval = "always" }, Options{}},
		{"approval mutating", func(c *Connection) { c.Approval = "mutating" }, Options{}},
		{"exact allowlist", func(c *Connection) { c.Allowlist = []string{"purchase_item"} }, Options{}},
		{"fak override", func(c *Connection) {}, Options{Overrides: []string{"shop__purchase_item"}}},
	}
	for _, tc := range cases {
		c := base
		tc.mut(&c)
		r := Preflight(Manifest{Connections: []Connection{c}}, tc.opts)
		if !r.OK {
			t.Fatalf("%s: expected pass, got %+v", tc.name, r.Diagnostics)
		}
		if len(r.AdmittedTools) != 1 || r.AdmittedTools[0] != "shop__purchase_item" {
			t.Fatalf("%s: expected [shop__purchase_item], got %v", tc.name, r.AdmittedTools)
		}
	}
}

// TestPreflightReadOnlyAllowlistedPasses is the third acceptance fixture: a
// read-only allowlisted connection passes and the report carries the EXACT
// generated tool namespace fak will admit; unlisted operations are filtered,
// not failed.
func TestPreflightReadOnlyAllowlistedPasses(t *testing.T) {
	m := Manifest{Connections: []Connection{{
		Name: "wiki", Type: "mcp", URL: "https://mcp.wiki.example", Auth: "app",
		Allowlist: []string{"search_pages", "get_page"},
		Operations: []Operation{
			{Name: "search_pages", ReadOnly: true},
			{Name: "get_page", ReadOnly: true},
			{Name: "delete_page", Mutating: true}, // not allowlisted -> filtered out
		},
	}}}
	r := Preflight(m, Options{})
	if !r.OK {
		t.Fatalf("read-only allowlisted connection should pass, got %+v", r.Diagnostics)
	}
	want := []string{"wiki__get_page", "wiki__search_pages"}
	if len(r.AdmittedTools) != len(want) {
		t.Fatalf("expected exactly %v, got %v", want, r.AdmittedTools)
	}
	for i := range want {
		if r.AdmittedTools[i] != want[i] {
			t.Fatalf("expected exactly %v, got %v", want, r.AdmittedTools)
		}
	}
}

// TestPreflightBlocklistWithoutAllowlistRemoteFails: a broad remote surface
// filtered by blocklist only fails; the same shape on a local surface only
// warns.
func TestPreflightBlocklistWithoutAllowlistRemoteFails(t *testing.T) {
	remote := Manifest{Connections: []Connection{{
		Name: "broad", Type: "mcp", URL: "https://mcp.example.com", Auth: "app",
		Blocklist:  []string{"drop_tables"},
		Operations: []Operation{{Name: "list_rows", ReadOnly: true}},
	}}}
	r := Preflight(remote, Options{})
	if r.OK || !hasDiag(t, r, CodeBlocklistWithoutAllowlist, "broad", "") {
		t.Fatalf("remote blocklist-only surface must fail, got %+v", r.Diagnostics)
	}

	local := remote
	local.Connections = []Connection{{
		Name: "narrow", Type: "mcp", URL: "http://127.0.0.1:9", Auth: "local",
		Blocklist:  []string{"drop_tables"},
		Operations: []Operation{{Name: "list_rows", ReadOnly: true}},
	}}
	lr := Preflight(local, Options{})
	if !lr.OK {
		t.Fatalf("local blocklist-only surface should warn, not fail: %+v", lr.Diagnostics)
	}
	if !hasDiag(t, lr, CodeBlocklistWithoutAllowlist, "narrow", "") {
		t.Fatalf("expected a %s warn on the local surface, got %+v", CodeBlocklistWithoutAllowlist, lr.Diagnostics)
	}
}

// TestPreflightUserAuthSchedule: a schedule reaching a user-scoped connection
// without a user principal fails; with a principal it passes; and a schedule
// naming an undeclared connection fails closed.
func TestPreflightUserAuthSchedule(t *testing.T) {
	conn := Connection{
		Name: "gmail", Type: "mcp", URL: "https://mcp.gmail.example", Auth: "user",
		Approval:   "always",
		Operations: []Operation{{Name: "list_labels", ReadOnly: true}},
	}
	m := Manifest{
		Connections: []Connection{conn},
		Schedules:   []Schedule{{Name: "nightly", Connections: []string{"gmail"}}},
	}
	r := Preflight(m, Options{})
	if r.OK || !hasDiag(t, r, CodeUserAuthNoPrincipal, "gmail", "") {
		t.Fatalf("expected %s, got %+v", CodeUserAuthNoPrincipal, r.Diagnostics)
	}

	m.Schedules[0].UserPrincipal = true
	if r := Preflight(m, Options{}); !r.OK {
		t.Fatalf("schedule with a user principal should pass, got %+v", r.Diagnostics)
	}

	m.Schedules = []Schedule{{Name: "nightly", Connections: []string{"ghost"}, UserPrincipal: true}}
	if r := Preflight(m, Options{}); r.OK || !hasDiag(t, r, CodeScheduleUnknownConnection, "ghost", "") {
		t.Fatalf("expected %s, got %+v", CodeScheduleUnknownConnection, r.Diagnostics)
	}
}

// TestPreflightToolNameUnsafe: a remote-shaped operation or connection name
// never becomes a fak tool name — refused, not munged.
func TestPreflightToolNameUnsafe(t *testing.T) {
	m := Manifest{Connections: []Connection{{
		Name: "files", Type: "mcp", URL: "http://127.0.0.1:1", Auth: "local",
		Operations: []Operation{{Name: "rm -rf /", ReadOnly: true}},
	}}}
	r := Preflight(m, Options{})
	if r.OK || !hasDiag(t, r, CodeToolNameUnsafe, "files", "rm -rf /") {
		t.Fatalf("expected %s for the shell-shaped op name, got %+v", CodeToolNameUnsafe, r.Diagnostics)
	}

	m.Connections[0].Name = "weird name!"
	m.Connections[0].Operations = []Operation{{Name: "ok_op", ReadOnly: true}}
	r = Preflight(m, Options{})
	if r.OK || !hasDiag(t, r, CodeToolNameUnsafe, "weird name!", "") {
		t.Fatalf("expected %s for the connection name, got %+v", CodeToolNameUnsafe, r.Diagnostics)
	}
}

// TestPreflightToolNameCollision: two operations mapping onto one generated
// name are refused as ambiguous.
func TestPreflightToolNameCollision(t *testing.T) {
	m := Manifest{Connections: []Connection{{
		Name: "dup", Type: "mcp", URL: "http://localhost:1", Auth: "local",
		Operations: []Operation{
			{Name: "get_item", ReadOnly: true},
			{Name: "get_item", ReadOnly: true},
		},
	}}}
	r := Preflight(m, Options{})
	if r.OK || !hasDiag(t, r, CodeToolNameCollision, "dup", "") {
		t.Fatalf("expected %s, got %+v", CodeToolNameCollision, r.Diagnostics)
	}
}

// TestPreflightReadOnlyHintCannotHideMutatingShape: a remote read-only hint
// does not clear an operation whose method or name says it mutates.
func TestPreflightReadOnlyHintCannotHideMutatingShape(t *testing.T) {
	for _, op := range []Operation{
		{Name: "cleanup", Method: "DELETE", ReadOnly: true},
		{Name: "send_report", ReadOnly: true},
	} {
		if mut, _ := operationMutating(op); !mut {
			t.Fatalf("read-only hint must not clear %+v", op)
		}
	}
}

// TestPreflightTypeUnknownFails: a surface the preflight cannot reason about
// fails closed.
func TestPreflightTypeUnknownFails(t *testing.T) {
	m := Manifest{Connections: []Connection{{
		Name: "mystery", Type: "grpc", URL: "http://localhost:1", Auth: "local",
	}}}
	r := Preflight(m, Options{})
	if r.OK || !hasDiag(t, r, CodeTypeUnknown, "mystery", "") {
		t.Fatalf("expected %s, got %+v", CodeTypeUnknown, r.Diagnostics)
	}
}

// TestParseManifestLenientButNotSilent: unknown fields are tolerated (the
// artifact carries more than the preflight reads); malformed JSON errors.
func TestParseManifestLenientButNotSilent(t *testing.T) {
	m, err := ParseManifest([]byte(`{"version":9,"connections":[{"name":"a","type":"mcp","auth":"local","future_field":true}]}`))
	if err != nil || len(m.Connections) != 1 {
		t.Fatalf("lenient parse failed: %v %+v", err, m)
	}
	if _, err := ParseManifest([]byte(`{"connections":`)); err == nil {
		t.Fatal("malformed manifest must error, not pass empty")
	}
}
