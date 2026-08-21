// Package cloudroute detects a request-signed cloud model route (AWS Bedrock via
// SigV4, Google Vertex via ADC) whose base-URL repoint cannot take effect.
//
// # Why this exists
//
// `fak guard` fronts a wrapped agent by pointing it at fak's local gateway with
// ANTHROPIC_BASE_URL and then adjudicating the traffic that arrives. A child
// running with CLAUDE_CODE_USE_BEDROCK=1 signs each request itself and goes
// straight to a regional cloud endpoint, so it never reads that variable: the
// repoint is silently inert and the session is adjudicated by nothing. Worse, such
// a box has no .credentials.json and no ANTHROPIC_API_KEY, so the launch gates
// fire on the wrong cause — they report a missing subscription token, or park for
// up to a day waiting for a re-login to land, for a session that holds a perfectly
// good working credential.
//
// This package is the sensor that lets the launch path name the real posture
// instead. It is pure: Detect takes the environment as a "NAME=VALUE" snapshot and
// returns a Route, so the guard can ask the question about the environment the
// CHILD will receive, and every case is testable without touching the process env.
//
// It reports NAMES only, never values. A Route is safe to print in a refusal, a
// config-bail file, or a doctor report.
//
// # Scope
//
// Detection, not support. A detected route means "fak's gateway cannot adjudicate
// this wire, and here is the alternative that does" — the working MCP-server path
// (`fak serve --stdio`), which is provider-agnostic because the agent calls fak
// rather than fak fronting the agent. Making the signed wire itself adjudicable
// needs a declared per-request credential provider, which is deliberately a
// follow-up and not in this package.
//
// # Tier
//
// Tier: primitive (1) - see internal/architest. This package may import only
// packages whose tier is <= 1; an upward import fails the architest gate. It
// imports nothing internal.
// See AGENTS.md and internal/architest for the layering contract.
package cloudroute
