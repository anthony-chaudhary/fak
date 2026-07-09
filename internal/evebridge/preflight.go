// Package evebridge is the pure core of the fak <-> eve bridge (#2600). This file
// is the connection-security preflight (#2602): a mechanical fold over eve's
// MCP/OpenAPI connection manifest (`eve info --json` / compiled discovery
// artifacts) that turns the deployer checklist in eve's connection docs
// (auth posture, tool filters, approval policies, connection scopes) into
// typed pass/fail diagnostics, so a scoping mistake fails closed at preflight
// instead of being left to deployer discipline.
//
// The package deliberately parses only the SUBSET of the manifest the
// preflight needs (connections + schedules); the full `eve info --json`
// compiler is #2601's seam and this fold consumes whatever it lands, since a
// lenient decode ignores the fields it does not check.
//
// All of it is pure: manifest in, report out, no I/O and no clock. The impure
// shell (file/stdin/`eve info --json` exec, flags, exit codes) is
// cmd/fak/eve.go.
package evebridge

import (
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"regexp"
	"sort"
	"strings"
)

// SchemaPreflight stamps every preflight report so a consumer (CI log, issue
// body, dispatcher) can bind to the shape.
const SchemaPreflight = "fak-eve-connection-preflight/1"

// The typed diagnostic codes the preflight emits. They are the closed
// vocabulary CI and issue bodies match on — never free text alone.
const (
	// CodeNoAuth: a connection declares the local/no-auth posture (or is
	// missing credentials) but points at a NON-local URL — an unauthenticated
	// remote surface.
	CodeNoAuth = "EVE_CONNECTION_NO_AUTH"
	// CodeAuthUndeclared: a connection has no explicit auth posture at all.
	// Every connection must say which of app|user|static|local it is; an
	// implicit posture is exactly the deployer-discipline gap this preflight
	// exists to close.
	CodeAuthUndeclared = "EVE_CONNECTION_AUTH_UNDECLARED"
	// CodeTypeUnknown: a connection is neither mcp nor openapi, so the
	// preflight cannot reason about its surface. Fail closed.
	CodeTypeUnknown = "EVE_CONNECTION_TYPE_UNKNOWN"
	// CodeUserAuthNoPrincipal: a schedule (an unattended runtime path)
	// references a user-scoped connection without carrying a user principal —
	// user credentials must be unreachable from unattended paths.
	CodeUserAuthNoPrincipal = "EVE_USER_AUTH_NO_PRINCIPAL"
	// CodeScheduleUnknownConnection: a schedule references a connection the
	// manifest does not declare; an unresolvable reference cannot be scoped,
	// so it fails closed rather than silently passing.
	CodeScheduleUnknownConnection = "EVE_SCHEDULE_UNKNOWN_CONNECTION"
	// CodeBlocklistWithoutAllowlist: a broad remote surface filters tools by
	// blocklist only. Blocklists fail open (a new remote tool is admitted by
	// default); remote surfaces must prefer allowlists.
	CodeBlocklistWithoutAllowlist = "EVE_BLOCKLIST_WITHOUT_ALLOWLIST"
	// CodeMutationUnapproved: a write/delete/purchase/message/transmit or
	// sensitive-read operation is reachable without an approval policy, an
	// exact allowlist entry, or a fak policy override.
	CodeMutationUnapproved = "EVE_MUTATION_UNAPPROVED"
	// CodeToolNameUnsafe: a connection or operation name does not survive the
	// strict tool-name shape, so the generated <connection>__<tool> fak name
	// would trust a remote input shape. Refused, never silently munged.
	CodeToolNameUnsafe = "EVE_TOOL_NAME_UNSAFE"
	// CodeToolNameCollision: two operations map to the same generated fak
	// tool name — an ambiguous admission is refused.
	CodeToolNameCollision = "EVE_TOOL_NAME_COLLISION"
)

// Severities. A "fail" fails the preflight closed (the connection admits no
// tools); a "warn" is surfaced but does not gate.
const (
	SeverityFail = "fail"
	SeverityWarn = "warn"
)

// The explicit auth postures the issue names. "no-auth" and "none" are
// accepted spellings of the local posture.
const (
	AuthApp    = "app"
	AuthUser   = "user"
	AuthStatic = "static"
	AuthLocal  = "local"
)

// Manifest is the connection-relevant subset of `eve info --json` / a
// compiled discovery artifact. Decoding is lenient (unknown fields ignored)
// so the #2601 reader can grow around it; NAME validation is strict.
type Manifest struct {
	Connections []Connection `json:"connections"`
	Schedules   []Schedule   `json:"schedules,omitempty"`
}

// Connection is one MCP or OpenAPI connection as eve declares it.
type Connection struct {
	Name string `json:"name"`
	Type string `json:"type"` // "mcp" | "openapi"
	URL  string `json:"url"`
	// Auth is the explicit posture: app | user | static | local (accepting
	// "no-auth"/"none" for local). Empty means undeclared — a fail.
	Auth string `json:"auth"`
	// Allowlist admits only the named operations (exact names; a mutating
	// operation must be named exactly to count as deliberately admitted).
	Allowlist []string `json:"allowlist,omitempty"`
	// Blocklist rejects the named operations. On a remote surface a blocklist
	// without an allowlist fails: it admits unknown future tools by default.
	Blocklist []string `json:"blocklist,omitempty"`
	// Approval is the connection's approval policy: "always" (every call
	// gated), "mutating" (mutating calls gated), or ""/"never" (no gate).
	Approval   string      `json:"approval,omitempty"`
	Operations []Operation `json:"operations,omitempty"`
}

// Operation is one exposed tool/operation. The boolean hints come from MCP
// tool annotations; Method comes from an OpenAPI operation. All of them are
// remote-declared, so the classifier treats "mutating" hints as binding but
// never lets a remote "read-only" hint override a mutating method or verb.
type Operation struct {
	Name      string `json:"name"`
	Method    string `json:"method,omitempty"`
	ReadOnly  bool   `json:"read_only,omitempty"`
	Mutating  bool   `json:"mutating,omitempty"`
	Sensitive bool   `json:"sensitive,omitempty"` // sensitive-read: gated like a mutation
}

// Schedule is an unattended runtime path that can reach connections without
// a human in the loop.
type Schedule struct {
	Name          string   `json:"name"`
	Connections   []string `json:"connections,omitempty"`
	UserPrincipal bool     `json:"user_principal,omitempty"`
}

// Options tunes one preflight run.
type Options struct {
	// Overrides is the fak policy override list: exact generated tool names
	// (<connection>__<operation>) allowed to mutate without a connection
	// approval policy. It is the deliberate, named escape hatch the issue
	// requires — never a blanket flag.
	Overrides []string
}

// Diagnostic is one typed finding. Remediation is written to be pasted into
// an issue body or CI log as the do-this-next line.
type Diagnostic struct {
	Code        string `json:"code"`
	Severity    string `json:"severity"`
	Connection  string `json:"connection,omitempty"`
	Operation   string `json:"operation,omitempty"`
	Detail      string `json:"detail"`
	Remediation string `json:"remediation"`
}

// Report is the JSON-first preflight verdict. AdmittedTools is the EXACT
// generated tool namespace fak will admit — a connection with any fail
// diagnostic admits nothing (fail closed).
type Report struct {
	Schema        string       `json:"schema"`
	OK            bool         `json:"ok"`
	Connections   int          `json:"connections_checked"`
	AdmittedTools []string     `json:"admitted_tools"`
	Diagnostics   []Diagnostic `json:"diagnostics,omitempty"`
}

// ParseManifest decodes the manifest subset. Unknown fields are ignored on
// purpose (the artifact carries far more than the preflight checks); a
// malformed document is an error, never an empty pass.
func ParseManifest(data []byte) (Manifest, error) {
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return Manifest{}, fmt.Errorf("eve manifest: %w", err)
	}
	return m, nil
}

// toolNamePart is the strict shape each side of <connection>__<operation>
// must have before it becomes a fak tool name. Anything else is a remote
// input shape we refuse to trust.
var toolNamePart = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_-]{0,63}$`)

// mutatingWords are the operation-name segments that classify an operation as
// mutating when no explicit hint or HTTP method decides first — the verbs the
// issue names (write, delete, purchase, message, transmit) plus their common
// API spellings.
var mutatingWords = map[string]bool{
	"write": true, "delete": true, "remove": true, "purchase": true, "buy": true,
	"pay": true, "order": true, "message": true, "send": true, "transmit": true,
	"create": true, "update": true, "set": true, "upload": true, "post": true,
	"put": true, "patch": true, "exec": true, "execute": true, "run": true,
}

// Preflight folds one manifest into one report. Pure: no I/O, no clock.
func Preflight(m Manifest, opts Options) Report {
	overrides := make(map[string]bool, len(opts.Overrides))
	for _, o := range opts.Overrides {
		overrides[strings.TrimSpace(o)] = true
	}

	var diags []Diagnostic
	var admitted []string
	byName := make(map[string]Connection, len(m.Connections))

	for _, c := range m.Connections {
		byName[c.Name] = c
		connDiags, tools := preflightConnection(c, overrides)
		diags = append(diags, connDiags...)
		if !hasFail(connDiags) {
			admitted = append(admitted, tools...)
		}
	}

	for _, s := range m.Schedules {
		diags = append(diags, preflightSchedule(s, byName)...)
	}

	sort.Strings(admitted)
	report := Report{
		Schema:        SchemaPreflight,
		OK:            !hasFail(diags),
		Connections:   len(m.Connections),
		AdmittedTools: admitted,
		Diagnostics:   diags,
	}
	if !report.OK {
		// Fail closed globally: a red preflight admits no namespace at all,
		// so a CI consumer can never mount the green half of a red manifest.
		report.AdmittedTools = []string{}
	}
	return report
}

// preflightConnection checks one connection and returns its diagnostics plus
// the generated tool names it would admit if it stays green.
func preflightConnection(c Connection, overrides map[string]bool) ([]Diagnostic, []string) {
	var diags []Diagnostic

	if !toolNamePart.MatchString(c.Name) {
		diags = append(diags, Diagnostic{
			Code: CodeToolNameUnsafe, Severity: SeverityFail, Connection: c.Name,
			Detail:      fmt.Sprintf("connection name %q is not a safe tool-name segment (%s)", c.Name, toolNamePart.String()),
			Remediation: "rename the connection to a letter-led [A-Za-z0-9_-] identifier of at most 64 characters; fak never munges remote-shaped names",
		})
		// A connection we cannot even name admits nothing; stop here.
		return diags, nil
	}

	switch strings.ToLower(strings.TrimSpace(c.Type)) {
	case "mcp", "openapi":
	default:
		diags = append(diags, Diagnostic{
			Code: CodeTypeUnknown, Severity: SeverityFail, Connection: c.Name,
			Detail:      fmt.Sprintf("connection type %q is not a surface this preflight can scope (want mcp|openapi)", c.Type),
			Remediation: `declare the connection "type" as "mcp" or "openapi"`,
		})
	}

	local := isLocalURL(c.URL)
	switch normalizeAuth(c.Auth) {
	case AuthApp, AuthUser, AuthStatic:
	case AuthLocal:
		if !local {
			diags = append(diags, Diagnostic{
				Code: CodeNoAuth, Severity: SeverityFail, Connection: c.Name,
				Detail:      fmt.Sprintf("connection %q declares the local/no-auth posture but its URL %q is not a local endpoint", c.Name, c.URL),
				Remediation: `an unauthenticated remote surface is refused: give the connection an explicit auth posture ("app", "user", or "static"), or move it to a loopback/unix endpoint if it is genuinely local`,
			})
		}
	default:
		diags = append(diags, Diagnostic{
			Code: CodeAuthUndeclared, Severity: SeverityFail, Connection: c.Name,
			Detail:      fmt.Sprintf("connection %q has no explicit auth posture (got %q)", c.Name, c.Auth),
			Remediation: `declare "auth" as one of app | user | static | local (local only for loopback/unix endpoints); an implicit posture is refused`,
		})
	}

	if len(c.Blocklist) > 0 && len(c.Allowlist) == 0 {
		sev, rem := SeverityFail, "replace the blocklist with an allowlist naming the operations this deployment actually needs; a remote blocklist admits every future tool by default"
		if local {
			sev, rem = SeverityWarn, "prefer an allowlist even on a local surface; a blocklist admits every future tool by default"
		}
		diags = append(diags, Diagnostic{
			Code: CodeBlocklistWithoutAllowlist, Severity: sev, Connection: c.Name,
			Detail:      fmt.Sprintf("connection %q filters its surface with a blocklist only (%d entries, no allowlist)", c.Name, len(c.Blocklist)),
			Remediation: rem,
		})
	}

	opDiags, tools := preflightOperations(c, overrides)
	return append(diags, opDiags...), tools
}

// preflightOperations classifies each operation the connection's filters
// admit, gates mutations, and generates the fak tool namespace.
func preflightOperations(c Connection, overrides map[string]bool) ([]Diagnostic, []string) {
	allow := make(map[string]bool, len(c.Allowlist))
	for _, a := range c.Allowlist {
		allow[strings.TrimSpace(a)] = true
	}
	block := make(map[string]bool, len(c.Blocklist))
	for _, b := range c.Blocklist {
		block[strings.TrimSpace(b)] = true
	}
	approval := strings.ToLower(strings.TrimSpace(c.Approval))
	approvalCoversMutations := approval == "always" || approval == "mutating"

	var diags []Diagnostic
	var tools []string
	seen := make(map[string]string, len(c.Operations)) // generated name -> operation name

	for _, op := range c.Operations {
		full := c.Name + "__" + op.Name
		if block[op.Name] || block[full] {
			continue // filtered out by the deployer, nothing to admit or gate
		}
		if len(allow) > 0 && !allow[op.Name] && !allow[full] {
			continue // allowlist mode: unlisted operations are simply not admitted
		}

		if !toolNamePart.MatchString(op.Name) {
			diags = append(diags, Diagnostic{
				Code: CodeToolNameUnsafe, Severity: SeverityFail, Connection: c.Name, Operation: op.Name,
				Detail:      fmt.Sprintf("operation %q does not survive the strict tool-name shape (%s); fak will not trust a remote input shape", op.Name, toolNamePart.String()),
				Remediation: "exclude the operation via the allowlist, or fix the upstream name to a letter-led [A-Za-z0-9_-] identifier of at most 64 characters",
			})
			continue
		}
		if prev, dup := seen[full]; dup {
			diags = append(diags, Diagnostic{
				Code: CodeToolNameCollision, Severity: SeverityFail, Connection: c.Name, Operation: op.Name,
				Detail:      fmt.Sprintf("operations %q and %q both map to the generated tool name %q", prev, op.Name, full),
				Remediation: "rename one of the colliding operations upstream or exclude one via the allowlist; an ambiguous admission is refused",
			})
			continue
		}
		seen[full] = op.Name

		if mutating, why := operationMutating(op); mutating {
			gated := approvalCoversMutations || allow[op.Name] || allow[full] || overrides[full]
			if !gated {
				diags = append(diags, Diagnostic{
					Code: CodeMutationUnapproved, Severity: SeverityFail, Connection: c.Name, Operation: op.Name,
					Detail:      fmt.Sprintf("operation %q is %s but connection %q has no approval policy, no exact allowlist entry for it, and no fak policy override", op.Name, why, c.Name),
					Remediation: fmt.Sprintf(`set the connection's "approval" to "always" or "mutating", allowlist %q explicitly, or pass a deliberate fak override for %q`, op.Name, full),
				})
				continue
			}
		}
		tools = append(tools, full)
	}
	return diags, tools
}

// preflightSchedule checks that no unattended schedule can reach a
// user-scoped credential without a user principal.
func preflightSchedule(s Schedule, byName map[string]Connection) []Diagnostic {
	var diags []Diagnostic
	for _, ref := range s.Connections {
		c, ok := byName[ref]
		if !ok {
			diags = append(diags, Diagnostic{
				Code: CodeScheduleUnknownConnection, Severity: SeverityFail, Connection: ref,
				Detail:      fmt.Sprintf("schedule %q references connection %q, which the manifest does not declare", s.Name, ref),
				Remediation: "declare the connection in the manifest or drop the schedule reference; an unresolvable reference cannot be scoped",
			})
			continue
		}
		if normalizeAuth(c.Auth) == AuthUser && !s.UserPrincipal {
			diags = append(diags, Diagnostic{
				Code: CodeUserAuthNoPrincipal, Severity: SeverityFail, Connection: c.Name,
				Detail:      fmt.Sprintf("schedule %q reaches user-scoped connection %q without a user principal", s.Name, c.Name),
				Remediation: "bind the schedule to a user principal, or move the connection to an app/static credential scoped for unattended use",
			})
		}
	}
	return diags
}

// operationMutating classifies one operation. Precedence is fail-closed
// against remote hints: an explicit mutating/sensitive hint or a mutating
// HTTP method binds unconditionally, and a remote read-only hint can never
// clear an operation whose NAME carries a mutation verb it could be hiding
// behind — only a genuinely read-shaped operation classifies as read-only.
func operationMutating(op Operation) (bool, string) {
	if op.Mutating {
		return true, "declared mutating"
	}
	if op.Sensitive {
		return true, "a sensitive-read"
	}
	switch strings.ToUpper(strings.TrimSpace(op.Method)) {
	case "POST", "PUT", "PATCH", "DELETE":
		return true, fmt.Sprintf("a mutating %s operation", strings.ToUpper(strings.TrimSpace(op.Method)))
	case "GET", "HEAD", "OPTIONS":
		return false, ""
	}
	for _, seg := range splitNameSegments(op.Name) {
		if mutatingWords[seg] {
			return true, fmt.Sprintf("named as a %q operation", seg)
		}
	}
	// No hint, no mutating method, no mutation verb in the name: read-only.
	// op.ReadOnly needs no branch of its own — a read-only hint is only ever
	// advisory and can never OVERRIDE the mutating signals above.
	return false, ""
}

// splitNameSegments lower-cases and splits an operation name on the
// non-alphanumeric separators and camelCase humps API names use, so
// "sendMessage", "send_message", and "send-message" all yield "send".
func splitNameSegments(name string) []string {
	var segs []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			segs = append(segs, strings.ToLower(cur.String()))
			cur.Reset()
		}
	}
	for _, r := range name {
		switch {
		case r >= 'A' && r <= 'Z':
			flush()
			cur.WriteRune(r)
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			cur.WriteRune(r)
		default:
			flush()
		}
	}
	flush()
	return segs
}

// isLocalURL reports whether the connection endpoint is genuinely local: a
// loopback host, a unix/stdio transport, or an empty URL (a stdio MCP server
// launched by eve itself). Anything unparseable is NOT local — fail closed.
func isLocalURL(raw string) bool {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return true
	}
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	switch strings.ToLower(u.Scheme) {
	case "unix", "stdio", "npipe":
		return true
	}
	host := u.Hostname()
	if host == "" {
		return false
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	if ip := net.ParseIP(host); ip != nil {
		return ip.IsLoopback()
	}
	return false
}

// normalizeAuth folds the accepted spellings onto the four postures; an
// unknown or empty value returns "" (undeclared).
func normalizeAuth(auth string) string {
	switch strings.ToLower(strings.TrimSpace(auth)) {
	case AuthApp:
		return AuthApp
	case AuthUser:
		return AuthUser
	case AuthStatic:
		return AuthStatic
	case AuthLocal, "no-auth", "noauth", "none":
		return AuthLocal
	}
	return ""
}

// hasFail reports whether any diagnostic is a fail (the fail-closed gate).
func hasFail(diags []Diagnostic) bool {
	for _, d := range diags {
		if d.Severity == SeverityFail {
			return true
		}
	}
	return false
}
