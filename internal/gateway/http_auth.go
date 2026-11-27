package gateway

import (
	"crypto/sha256"
	"crypto/subtle"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/gatewayauth"
)

// withAuth enforces the configured secret on every route except the auth-exempt
// set (authExempt) when RequireKey is set. With no key configured it is a
// pass-through (drop-in, loopback default), except that terminal-outcome reads
// always require an accepted configured credential. The comparison is constant-time over
// SHA-256 digests so the reject latency leaks neither the secret's bytes nor its
// length — this is the gateway's only auth primitive on a network-reachable
// security kernel.
func (s *Server) withAuth(next http.Handler) http.Handler {
	want := sha256.Sum256([]byte(s.requireKey))
	wantRead := sha256.Sum256([]byte(s.readBearer))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The X-Fak-Auth-Scope guard (#12761) runs BEFORE the bearer check so a
		// scope-refused request answers 403 without ever consulting credentials.
		// An absent header falls through unchanged (the legacy path is byte-for-
		// byte identical); enforceAuthScope returns false only when it already
		// wrote the 403 scope_forbidden response.
		if !s.enforceAuthScope(w, r) {
			return
		}
		if requiresRequestOutcomeAuth(r) || ((s.requireKey != "" || s.keyset != nil) && !s.authExempt(r)) {
			tok, ok := gatewayCredential(r)
			got := sha256.Sum256([]byte(tok))
			// The single RequireKey bearer authenticates the anonymous single-tenant
			// caller; the keyset (#5332) authenticates a bound org/project principal.
			// Both are the SAME constant-time SHA-256 compare — a request presenting ANY
			// accepted key is authenticated. A keyset match additionally ATTRIBUTES the
			// turn to its tenant principal, stamped onto the context below so principalFor
			// / traceOwner / the access log / /v1/fak/events all name the same tenant. On
			// the RequireKey-only path (keyset == nil) lookup is a nil no-op and this is
			// byte-for-byte the prior behavior.
			authed := ok && s.requireKey != "" && subtle.ConstantTimeCompare(got[:], want[:]) == 1
			principal := ""
			if ok {
				if p, sc, matched := s.keyset.lookupScoped(tok); matched {
					authed = true
					principal = p
					// Per-key scope enforcement (#14448): a tenant key without the
					// admin scope may not reach the host-wide operator plane — policy
					// / route reload, control directives, revoke, /v1/admin/* — which
					// answers 403 scope_forbidden here, before any handler runs. Every
					// other route stays tenant-reachable (the inference wire plus the
					// tenant self-service surface, #14442), and the RequireKey operator
					// door stays unscoped on every route.
					if sc&keyScopeAdmin == 0 && operatorPlaneRoute(r.URL.Path) {
						writeScopeForbidden(w, sc.token(), r.Method)
						return
					}
				}
			}
			// The read-scoped bearer (Config.ReadBearer) is consulted only AFTER the
			// full-strength credential has already failed, and only on the read-only
			// observability paths. That ordering is what keeps it strictly widening: it
			// can admit a caller the main key would have rejected, but it can never
			// reject one the main key accepted, and it is never reachable from a
			// mutating route. Guarding on a non-empty readBearer is load-bearing, not
			// belt-and-braces — without it an UNSET read bearer would hash to the same
			// digest as an empty presented bearer and silently authorize `Bearer `.
			if !authed && s.readBearer != "" && ok && readScopedPath(r) {
				authed = subtle.ConstantTimeCompare(got[:], wantRead[:]) == 1
			}
			if !authed {
				writeAuthError(w, r, ok)
				return
			}
			if principal != "" {
				r = r.WithContext(WithPrincipal(r.Context(), principal))
			}
		} else if s.keyset != nil {
			if tok, ok := gatewayCredential(r); ok {
				if p, _, matched := s.keyset.lookupScoped(tok); matched {
					r = r.WithContext(WithPrincipal(r.Context(), p))
				}
			}
		}
		// Tenant floor for operator-grade observability (#14452): whatever the
		// source address, a request that authenticated AS a named tenant principal
		// (keyset-bound, stamped above on either branch) is refused on the
		// operator surfaces — a tenant key must not open /metrics, /debug/vars,
		// the observation family or /slots even from loopback. Operators scrape
		// with the full gateway key, the dedicated metrics bearer, or on-box
		// without any tenant credential; both doors authenticate WITHOUT a
		// principal, so this check never touches them.
		if p := principalFromContext(r.Context()); p != "" && operatorObservationPath(r.URL.Path) {
			s.logf("gateway: tenant %q refused operator-scoped route %s", p, r.URL.Path)
			writeErr(w, http.StatusForbidden, "operator-scoped route: tenant keys cannot read "+r.URL.Path+"; use the operator or metrics bearer")
			return
		}
		if s.startup.childStartupPending() && strings.HasPrefix(r.URL.Path, "/v1/") && !strings.HasPrefix(r.URL.Path, "/v1/fak/") {
			s.MarkChildUsable(time.Now())
		}
		next.ServeHTTP(w, r)
	})
}

// authExempt reports whether a request may skip the bearer check on an
// authenticated gateway.
func (s *Server) authExempt(r *http.Request) bool {
	// A LAN peer must still present the configured credential
	// before it can submit or read a provisional control directive.
	if isControlDirectivePath(r.URL.Path) {
		return false
	}
	if r.URL.Path == gatewayauth.KeyProofPath {
		return true
	}
	if r.URL.Path == "/healthz" || r.URL.Path == "/readyz" {
		return true
	}
	if requestViaProxy(r) {
		return false
	}
	if s.allowLAN && requestFromLAN(r) {
		return true
	}
	// Human/agent discovery is directly clickable from the loopback URL shown
	// in the TUI, but remains bearer-gated when the gateway is exposed off-box.
	if (r.URL.Path == "/" || r.URL.Path == "/a2a/v1/agent-card") && requestFromLoopback(r) {
		return true
	}
	if readScopedPath(r) {
		return requestFromLoopback(r)
	}
	if s.richDashboards != nil && s.richDashboards.proxyGrafana && requestFromLoopback(r) {
		prefix := s.richDashboards.proxyPrefix
		if prefix == "" {
			prefix = "/grafana"
		}
		prefix = "/" + strings.Trim(prefix, "/")
		if r.URL.Path == prefix || strings.HasPrefix(r.URL.Path, prefix+"/") {
			return true
		}
	}
	return false
}

var proxyHeaders = []string{
	"CF-Connecting-IP", "CF-Ray", "CF-Visitor",
	"X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto",
	"X-Real-IP", "Forwarded", "True-Client-IP",
}

// requestViaProxy reports whether the request carries a forwarding/edge header,
// meaning an on-box proxy (e.g. cloudflared) may be relaying an off-box caller.
// These headers are client-spoofable, but presence can only REMOVE a peer-address
// exemption, so it fails closed.
func requestViaProxy(r *http.Request) bool {
	for _, h := range proxyHeaders {
		if _, ok := r.Header[http.CanonicalHeaderKey(h)]; ok {
			return true
		}
	}
	return false
}

// requestFromLAN reports whether the request's peer is on a private/local network
// interface (RFC 1918 IPv4, link-local IPv4/IPv6, unique-local IPv6, or loopback).
func requestFromLAN(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	host = strings.Trim(host, "[]")
	if zone := strings.IndexByte(host, '%'); zone >= 0 {
		host = host[:zone]
	}
	ip := net.ParseIP(host)
	return ip != nil && (ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast())
}

// readScopedPath reports whether the path is one of the read-only observability
// surfaces — the set the loopback exemption opens, and the same set the read-scoped
// bearer may unlock off-loopback. Both callers share this one predicate so the two
// grants can never drift into disagreeing about what "read-only" means: adding a
// surface here widens both at once, which is the intent.
func readScopedPath(r *http.Request) bool {
	switch r.URL.Path {
	case "/v1/fak/features/proof", "/metrics", "/debug/vars", "/v1/fak/observation", "/v1/fak/observation/requests", "/v1/fak/observation/engine", "/v1/fak/perf/recent", "/v1/fak/perf/phases", "/v1/fak/arms", "/v1/fak/arms/traffic",
		// /props and /slots are the llama-server-shaped engine introspection
		// pair, served from the same live state as /metrics and carrying the
		// same class of information: counts, ratios, and build labels. They join
		// /metrics here so the read-scoped floor cannot drift — an engine that
		// admits /metrics must admit the introspection that explains it.
		"/props", "/slots", "/v1/serve/status":
		return true
	}
	return false
}

// operatorObservationPath reports whether the path is an OPERATOR-GRADE
// observability surface (#14452): the process-wide /metrics and /debug/vars,
// the observation family, and the engine-introspection /slots. These carry
// other tenants' tool names, verdicts, trace ids and customer names, so they
// stay closed to keyset-bound TENANT principals: the doors are the on-box
// loopback exemption, the full-strength RequireKey bearer, and the dedicated
// read-only metrics bearer (Config.ReadBearer, armed on `fak serve` via
// --metrics-key-env). The set is deliberately NARROWER than readScopedPath —
// /props, /v1/fak/perf/* and /v1/fak/arms* expose process-level counts and
// build labels, not per-tenant content, and keep their read-scoped access.
func operatorObservationPath(path string) bool {
	switch path {
	case "/metrics", "/debug/vars", "/v1/fak/observation", "/v1/fak/observation/requests", "/v1/fak/observation/engine", "/slots":
		return true
	}
	return false
}

// requestFromLoopback reports whether the request's peer is the loopback
// interface, classifying by IP VALUE (net.ParseIP + IsLoopback) rather than a
// string prefix so a spoofed RemoteAddr host cannot masquerade as local. An
// unparseable RemoteAddr is treated as NOT loopback (fail closed). RemoteAddr is
// the kernel-observed peer of the TCP connection, set by net/http — it is not a
// client-supplied header (unlike X-Forwarded-For, which is deliberately ignored
// here so a proxied header can never grant the exemption).
func requestFromLoopback(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	host = strings.Trim(host, "[]")
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// gatewayCredential extracts the presented secret from any of the auth schemes a
// fak gateway fronts. The OpenAI/fak-native surfaces send
// "Authorization: Bearer <tok>"; the native Anthropic surface (/v1/messages) is
// driven by clients — Claude Code, the Anthropic SDKs — that authenticate with the
// "x-api-key: <tok>" header instead; the native Gemini surface
// (/v1beta/models/{model}:generateContent) is driven by clients — Gemini CLI, the
// google-genai SDKs — that authenticate with "x-goog-api-key: <tok>" (or, for raw
// REST, "?key=<tok>"). Accepting all of them is what lets an authenticated
// (non-loopback) gateway serve any native client wire over its base-URL redirect;
// without the matching arm every such client 401s even though the gateway speaks
// its wire. All schemes compare against the same single secret in constant time at
// the call site.
func gatewayCredential(r *http.Request) (string, bool) {
	if tok, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer "); ok {
		return tok, true
	}
	if k := r.Header.Get("X-Api-Key"); k != "" {
		return k, true
	}
	// Control directives use only their declared bearer and X-Api-Key doors.
	// A query-string secret can be logged or copied with a URL; Gemini's
	// X-Goog-Api-Key scheme is not a credential for this operator route.
	if isControlDirectivePath(r.URL.Path) {
		return "", false
	}
	if g := r.Header.Get("X-Goog-Api-Key"); g != "" {
		return g, true
	}
	if q := r.URL.Query().Get("key"); q != "" {
		return q, true
	}
	return "", false
}
