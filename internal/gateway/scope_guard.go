package gateway

import (
	"net/http"
	"strings"
)

// X-Fak-Auth-Scope is the caller-DECLARED auth-scope claim header (#12761): an
// agent announces the plane it operates on so the gateway can bound its own way
// in - `read` (observation-polling watchers) or `write` (mutating agents and
// admin plumbing). This is a containment contract between cooperating agents,
// NOT a cryptographic credential: a compromised watcher could simply omit the
// header, so the guard's job is making scoping auditable and refusing
// well-behaved read agents before they reach a mutating handler, with a
// machine-readable reason. The header is ABSENT by default - every existing
// client - and absence preserves the legacy behavior byte-for-byte.
const authScopeHeader = "X-Fak-Auth-Scope"

const (
	authScopeRead  = "read"
	authScopeWrite = "write"
)

// Scope route classes: the closed label set fak_gateway_scope_refusals_total
// buckets refusals by. "admin" is the admin-grade control plane (/v1/admin/*)
// where a GET can be just as operational as a POST (the companion server daemon
// drains via POST; even a GET-form admin probe refuses a read-scoped caller),
// and the observation family (aggregate read + per-request read) is its own
// class so a refusal on the polling surface is distinguishable from one on a
// swallowed generic fak route.
const (
	scopeRouteClassObservation         = "observation"
	scopeRouteClassObservationRequests = "observation_requests"
	scopeRouteClassAdmin               = "admin"
	scopeRouteClassOther               = "other"
)

// authScopeClaim is one parsed X-Fak-Auth-Scope header value.
type authScopeClaim struct {
	scope    string
	declared bool
}

// authScopeVerdict is the scope/verb/route-class matrix outcome for one request.
type authScopeVerdict struct {
	allowed    bool
	scope      string
	routeClass string
}

// parseAuthScope reads the declared scope off the request. Absent or
// whitespace-only -> undeclared (legacy path, no enforcement). The value is
// matched case-insensitively after trimming (Header.Get already canonicalizes
// the key). `write` is the only claim that grants writes; `read` - and any
// unrecognized value - normalizes to the least-privilege read scope, so a
// future or typo'd token can never widen access by failing to parse.
func parseAuthScope(r *http.Request) authScopeClaim {
	raw := strings.TrimSpace(r.Header.Get(authScopeHeader))
	if raw == "" {
		return authScopeClaim{}
	}
	if strings.EqualFold(raw, authScopeWrite) {
		return authScopeClaim{scope: authScopeWrite, declared: true}
	}
	return authScopeClaim{scope: authScopeRead, declared: true}
}

// scopeRouteClass classifies a request path into the closed route-class label
// set. /v1/admin/* is the write-required admin plane; the observation family
// (aggregate read + per-request read) is its own class so a refusal on the
// polling surface is distinguishable from one on a swallowed generic /v1/fak
// route.
func scopeRouteClass(path string) string {
	switch {
	case strings.HasPrefix(path, "/v1/admin"):
		return scopeRouteClassAdmin
	case path == "/v1/fak/observation":
		return scopeRouteClassObservation
	case path == "/v1/fak/observation/requests":
		return scopeRouteClassObservationRequests
	default:
		return scopeRouteClassOther
	}
}

// evaluateAuthScope applies the verb/scope matrix. Undeclared calls never
// reach it (handled at parseAuthScope). Matrix: write claims pass everything;
// read claims get GET/HEAD reads only on non-admin routes; /v1/admin/*
// requires a write claim for EVERY verb including GET. Unreachable by
// construction: the rows compose so a refusal always exists in the matrix.
func evaluateAuthScope(scope, method, path string) authScopeVerdict {
	routeClass := scopeRouteClass(path)
	if routeClass == scopeRouteClassAdmin && scope != authScopeWrite {
		return authScopeVerdict{scope: scope, routeClass: routeClass}
	}
	if scope == authScopeWrite || method == http.MethodGet || method == http.MethodHead {
		return authScopeVerdict{allowed: true}
	}
	return authScopeVerdict{scope: scope, routeClass: routeClass}
}

// writeScopeForbidden emits the machine-auditable refusal body: a flat JSON
// object naming the reason code, the scope that was evaluated, and the method
// that was refused, so an agent loop can branch on error == "scope_forbidden"
// instead of string-matching prose.
func writeScopeForbidden(w http.ResponseWriter, scope, method string) {
	writeJSON(w, http.StatusForbidden, map[string]string{
		"error":  "scope_forbidden",
		"scope":  scope,
		"method": method,
	})
}

// enforceAuthScope applies the X-Fak-Auth-Scope matrix to one request inside
// the auth middleware: undeclared -> legacy pass-through (bit-for-bit); an
// allowed verdict -> pass; a refused verdict -> count it and answer 403
// scope_forbidden without reaching the route handler. Returns false when the
// request was refused and the response already written.
func (s *Server) enforceAuthScope(w http.ResponseWriter, r *http.Request) bool {
	claim := parseAuthScope(r)
	if !claim.declared {
		return true
	}
	verdict := evaluateAuthScope(claim.scope, r.Method, r.URL.Path)
	if verdict.allowed {
		return true
	}
	s.metrics.observeScopeRefusal(verdict.scope, verdict.routeClass)
	writeScopeForbidden(w, verdict.scope, r.Method)
	return false
}
