// serve_bind_safety.go — the `fak serve` bind-admission rule (#5373, a child of the
// air-gapped deployment work in #3279): refuse to come up listening on an interface
// that is reachable from OFF this host while the gateway would still answer requests
// nobody had to authenticate. The January 2026 internet-wide scan #3279 cites found
// 175,108 publicly reachable, auth-less local-model servers; a convention written in a
// runbook does not prevent that shape, a startup refusal does.
//
// The rule fires only on the CONJUNCTION — reachable off-host AND no token door named.
// Every other combination is admitted untouched, which is what keeps the check cheap to
// live with: the shipped `--addr` default is `127.0.0.1:8080`, so a local dev run, a
// `--stdio` launch, and every test that binds `127.0.0.1:0` never see it.
//
// It is deliberately NOT the same predicate as the gateway's own `loopbackOnly`
// (internal/gateway/tool_routing.go) or its `fak guard` mirror `guardLoopbackOnly`.
// Those two decide whether to print a WARNING, so they resolve every ambiguous host
// (a DNS name, a typo) toward "exposed" — a needless warning costs nothing. This one
// decides whether to REFUSE TO START, so it resolves ambiguity the other way: a host
// that cannot be PROVEN reachable off-host is admitted, and the pre-existing gateway
// warning still covers it. A malformed-but-harmless `--addr` must not wedge a boot.
package main

import (
	"fmt"
	"io"
	"net"
	"strings"
)

// serveBindRefusalToken is the stable, structured reason `fak serve` names when it
// refuses an unauthenticated off-host bind, so an operator (or a log scrape) can match
// the refusal by token rather than by prose that may be reworded.
const serveBindRefusalToken = "UNAUTHENTICATED_OFF_HOST_BIND"

// serveUnsafeBindFlag is the operator opt-out's flag name, held as a constant so the
// refusal text and the flag registration in serve.go can never drift apart.
const serveUnsafeBindFlag = "unsafe-allow-unauthenticated-bind"

// serveBindReachesOffHost reports whether addr binds an interface a peer on the network
// can reach. It classifies by IP VALUE, never by string prefix, and answers false
// whenever the address cannot be proven off-host reachable:
//
//   - "" — no listen address at all; run() refuses that separately.
//   - "localhost", any loopback IP (127.0.0.0/8, ::1) — this host only.
//   - a host that is not a parseable IP (a DNS name, or a lookalike such as
//     "127.0.0.1.evil.example") — the interface it would resolve to is not knowable
//     here without a boot-time DNS lookup, and net.Listen fails loudly on its own if
//     the name does not resolve, so an unproven host is not a refusal.
//
// It answers true for the wildcard forms, because net.Listen binds those to ALL
// interfaces: a bare ":8080" (empty host), "0.0.0.0", and "::". It also answers true
// for any other parseable IP — a specific routable address, and a link-local one, are
// both reachable by someone who is not this host.
func serveBindReachesOffHost(addr string) bool {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return false
	}
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr // no port present, e.g. a bare "127.0.0.1" or "::1"
	}
	host = strings.TrimSpace(strings.Trim(host, "[]"))
	if host == "" {
		return true // ":8080" — every interface
	}
	if strings.EqualFold(host, "localhost") {
		return false
	}
	// A scoped IPv6 literal ("fe80::1%eth0") carries its zone after a '%'; net.ParseIP
	// rejects the zone, so drop it before classifying. The zone names which link the
	// address lives on, never whether it is loopback.
	if zone := strings.IndexByte(host, '%'); zone >= 0 {
		host = host[:zone]
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return false // unproven, so not refused
	}
	return !ip.IsLoopback()
}

// serveAuthConfigured reports whether the operator named an inbound token door on this
// `fak serve` invocation. It mirrors the gateway's own admission condition
// (internal/gateway/http.go withAuth: `s.requireKey != "" || s.keyset != nil`) across
// the two flags that feed it, and there is no third door — `--api-key-env` and
// `--engine-cache-admin-key-env` are OUTBOUND upstream credentials, and the read-scoped
// bearer only ever widens access on the observability paths, never restricts it.
//
// It reads flag PRESENCE rather than the resolved secret, on purpose. Naming either
// flag is the operator declaring a door; if the env var it names turns out unset or
// empty, resolveSessionPlane (--require-key-env) and serveKeyPrincipals
// (--key-principal) each already refuse to boot with a message specific to that
// failure. Reading presence here therefore never lets an unauthenticated listener bind
// and never pre-empts the more precise diagnostic.
func serveAuthConfigured(sf *serveFlags) bool {
	return strings.TrimSpace(*sf.requireKeyEnv) != "" || len(sf.keyPrincipal.Values()) > 0 || (sf.allowLAN != nil && *sf.allowLAN)
}

// serveBindRefusal returns the operator-facing refusal text for a listen address, or ""
// when the bind is admissible. Pure and total: it performs no I/O and binds no socket,
// so the whole matrix is table-testable without a listener.
//
// override is the `--unsafe-allow-unauthenticated-bind` escape. It suppresses the
// refusal for the operator who genuinely means it (an isolated lab segment, a host
// firewall doing the work); the caller is expected to say so loudly on stderr, because
// a safety default with no escape becomes a patch people carry out-of-tree.
func serveBindRefusal(addr string, authConfigured, override bool) string {
	if authConfigured || override || !serveBindReachesOffHost(addr) {
		return ""
	}
	return fmt.Sprintf(
		"fak serve: %s — refusing to bind %s, which is reachable from off this host, with no inbound token door: "+
			"every request would be served unauthenticated. Fix it one of three ways: bind loopback (--addr 127.0.0.1:8080, the default), "+
			"require a bearer (--require-key-env FAK_API_KEY, with that env var set), or bind per-tenant keys (--key-principal acme=ACME_KEY). "+
			"If this host really is meant to serve an unauthenticated interface, pass --%s to proceed anyway.\n"+
			"  next:   fak recover %s",
		serveBindRefusalToken, addr, serveUnsafeBindFlag, serveBindRefusalToken)
}

// admitServeBind applies the bind-admission rule to the parsed serve flags, writing the
// refusal (or the override's warning) to stderr. It returns false when the caller must
// NOT continue to boot.
//
// --stdio is exempt: that surface speaks MCP over stdin/stdout and binds no socket at
// all, so there is no interface to be reachable on.
func admitServeBind(sf *serveFlags, stderr io.Writer) bool {
	if *sf.stdio {
		return true
	}
	override := *sf.unsafeUnauthedBind
	if refusal := serveBindRefusal(*sf.addr, serveAuthConfigured(sf), override); refusal != "" {
		fmt.Fprintln(stderr, refusal)
		return false
	}
	if override && !serveAuthConfigured(sf) && serveBindReachesOffHost(*sf.addr) {
		fmt.Fprintf(stderr,
			"fak serve: WARNING — --%s was passed: binding %s, reachable from off this host, with NO inbound authentication. "+
				"Every request this gateway serves is unauthenticated. This is the %s refusal, deliberately overridden.\n",
			serveUnsafeBindFlag, *sf.addr, serveBindRefusalToken)
	}
	return true
}
