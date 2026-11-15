package gateway

// session_owner.go — the tenant-ownership floor on the session-control plane
// (#14442, part of the multi-tenant isolation epic #3263).
//
// Per-tenant keys (--key-principal, #5332) authenticate each tenant's turns and
// attribute them to that tenant's principal, but the session-control plane was
// never retrofitted with an owner check: any authenticated tenant could observe,
// steer, pause, or stop another tenant's session, clear its guard taint, or list
// every tenant's tasks. Authentication did not imply authorization on the
// session routes.
//
// This file closes that gap by reusing the EXISTING trace-owner binding
// (bindTraceOwner / traceOwnerOf, ctxrestore.go) as the authorization key for
// every session-plane route: the principal that drove a session's first turn
// owns that session, and only that tenant — or the operator scope below — may
// address it. The observable outcome is a cross-tenant caller drawing 404 on a
// foreign trace (no existence leak, no 403 that confirms the target exists)
// while list routes return only the caller's rows.
//
// Enforcement arms ONLY when a keyset is configured (Config.KeyPrincipals):
// that is the deployment in which tenant identity is AUTHENTICATED at the front
// door rather than claimed by caller text. Without a keyset (single-key /
// loopback deployments) every caller shares the "" principal and the session
// plane keeps today's behaviour byte-for-byte.

import "net/http"

// sessionOwnerAdmitted reports whether the caller's authenticated tenant may
// address a session-plane resource owned by `owner`. A nil-keyset server or an
// un-stamped caller is admitted everywhere: under an armed keyset an empty
// principal denotes the OPERATOR SCOPE — the RequireKey bearer, or an
// auth-exempt loopback/LAN path — both of which already hold full visibility
// today, so admitting them changes nothing. Under an armed keyset a stamped
// tenant is admitted only on resources it owns.
func (s *Server) sessionOwnerAdmitted(caller, owner string) bool {
	if s.keyset == nil || caller == "" {
		return true
	}
	return caller == owner
}

// tenantPrincipalOf returns the request's authenticated tenant principal
// ("") and whether tenancy enforcement is armed for this request. Armed means a
// keyset is configured: only then does an empty principal denote the operator
// scope instead of an unidentified caller, because the front door has already
// refused every non-exempt request without an accepted credential.
func (s *Server) tenantPrincipalOf(r *http.Request) (string, bool) {
	if s.keyset == nil {
		return "", false
	}
	return principalFromContext(r.Context()), true
}

// foreignTraceRefused reports whether the request must be refused as a
// cross-tenant address of trace: tenancy armed, the caller stamped, and neither
// ownership nor the operator scope holds. Handlers answer the refusal with 404
// (session/trace not found) so a foreign tenant learns nothing about traces it
// cannot address.
func (s *Server) foreignTraceRefused(r *http.Request, trace string) bool {
	caller, armed := s.tenantPrincipalOf(r)
	if !armed {
		return false
	}
	owner, _ := s.traceOwnerOf(trace)
	return !s.sessionOwnerAdmitted(caller, owner)
}
