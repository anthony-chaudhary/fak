// l3_serve_bind_safety.go — the `fak l3-serve` bind-admission rule (issue
// #14345): refuse to come up listening on an interface reachable from OFF this
// host while the L3 tier would still answer every request unauthenticated.
// This mirrors the `fak serve` gateway rule (serve_bind_safety.go, #5373) for
// the KV-cache tier, whose shipped default was `0.0.0.0:18000` with no
// credential anywhere in the protocol.
//
// The rule fires only on the CONJUNCTION — reachable off-host AND no auth door
// configured. Loopback binds (the new default), binds with an auth secret or
// admin token configured, and the explicit operator override are all admitted
// untouched. It reuses the gateway's `serveBindReachesOffHost` predicate
// verbatim, so both verbs classify addresses identically.
package main

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// l3BindRefusalToken is the stable, structured reason `fak l3-serve` names when
// it refuses an unauthenticated off-host bind, so an operator (or a log scrape)
// can match the refusal by token rather than by prose that may be reworded.
const l3BindRefusalToken = "L3_UNAUTHENTICATED_OFF_HOST_BIND"

// l3UnsafeBindFlag is the operator opt-out's flag name, held as a constant so
// the refusal text and the flag registration in l3_serve.go can never drift.
const l3UnsafeBindFlag = "unsafe-allow-unauthenticated-bind"

// l3ServeBindRefusal returns the operator-facing refusal text for one listen
// address, or "" when the bind is admissible.
func l3ServeBindRefusal(addr string, authConfigured, override bool) string {
	if authConfigured || override || !serveBindReachesOffHost(addr) {
		return ""
	}
	return fmt.Sprintf(
		"fak l3-serve: %s — refusing to bind %s, which is reachable from off this host, with no auth door configured: "+
			"every request would be served unauthenticated. Fix it one of three ways: bind loopback (--listen 127.0.0.1:18000, the default), "+
			"require a handshake secret (--auth-secret-env FAK_L3_AUTH_SECRET, with that env var set), or name an admin token (--admin-token-env FAK_L3_ADMIN_TOKEN). "+
			"If this host really is meant to serve an unauthenticated interface, pass --%s to proceed anyway.\n"+
			"  next:   fak recover %s",
		l3BindRefusalToken, addr, l3UnsafeBindFlag, l3BindRefusalToken)
}

// admitL3ServeBind applies the bind-admission rule to every configured listen
// address, writing refusals (or the override's warning) to stderr. It returns
// false when the caller must NOT continue to boot.
func admitL3ServeBind(addrs []string, authConfigured, override bool, stderr io.Writer) bool {
	refused := false
	for _, addr := range addrs {
		if refusal := l3ServeBindRefusal(addr, authConfigured, override); refusal != "" {
			fmt.Fprintln(stderr, refusal)
			refused = true
		}
	}
	if refused {
		return false
	}
	if override && !authConfigured {
		for _, addr := range addrs {
			if serveBindReachesOffHost(addr) {
				fmt.Fprintf(stderr,
					"fak l3-serve: WARNING — --%s was passed: binding %s, reachable from off this host, with NO authentication. "+
						"Every request this server serves is unauthenticated. This is the %s refusal, deliberately overridden.\n",
					l3UnsafeBindFlag, addr, l3BindRefusalToken)
			}
		}
	}
	return true
}

// resolveSecretFlag resolves a credential from a literal flag and an env-var
// name flag. Precedence: exactly one of the two must be given — a literal value
// or the NAME of an environment variable holding it. Giving both is refused
// (ambiguous security configuration), and a named env var that is unset or
// empty is a fatal configuration error: the operator asked for that door by
// name, and silently booting without it would defeat the point.
func resolveSecretFlag(name, literal, envName string) (string, error) {
	literal = trimSecret(literal)
	envName = trimSecret(envName)
	if literal != "" && envName != "" {
		return "", fmt.Errorf("--%s and --%s-env are mutually exclusive; give one", name, name)
	}
	if envName != "" {
		v := trimSecret(os.Getenv(envName))
		if v == "" {
			return "", fmt.Errorf("--%s-env names %q but it is unset or empty", name, envName)
		}
		return v, nil
	}
	return literal, nil
}

// trimSecret strips surrounding whitespace from a credential as typed or read
// from the environment — copy-paste artifacts are far more likely than
// meaningful leading/trailing spaces in a shared secret.
func trimSecret(s string) string {
	return strings.TrimSpace(s)
}
