// Package codexmcpdiag classifies Codex MCP startup evidence without exposing log bodies.
package codexmcpdiag

import "strings"

const (
	// VerdictFalseNegative signals that all target MCP servers initialized cleanly
	// despite client-side warning banners suggesting offline status.
	VerdictFalseNegative = "CLIENT_STATUS_FALSE_NEGATIVE"

	// VerdictServerFailure indicates fatal startup errors or missing initialization
	// for one or more target MCP servers.
	VerdictServerFailure = "SERVER_FAILURE"

	// VerdictRuntimeCancellation indicates initialization was interrupted by
	// a concurrent runtime configuration refresh or context reload.
	VerdictRuntimeCancellation = "RUNTIME_REFRESH_CANCELLATION"

	// VerdictInsufficient indicates recorded telemetry logs lack definitive
	// startup completion or failure markers for requested servers.
	VerdictInsufficient = "INSUFFICIENT_EVIDENCE"
)

// Event captures a single filtered telemetry entry from startup logs,
// preserving message metadata without logging sensitive prompt payloads.
type Event struct{ Level, Target, Body string }

// Server describes the resolved health state and operational status
// for a single named MCP server instance.
type Server struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}

// Report summarizes the aggregate diagnostic verdict and individual status
// determinations across all evaluated MCP server targets.
type Report struct {
	Verdict string   `json:"verdict"`
	Servers []Server `json:"servers"`
}

// Invariant: Diagnostic classification never echoes raw event bodies or private tokens into the report.

// Precondition: names specifies target server identifiers; events supplies captured telemetry records.

// Postcondition: Every requested server identifier receives exactly one classified status in output order.

// Classify evaluates captured startup log events against target server names to
// produce an aggregate health diagnosis without leaking sensitive log contents.
func Classify(names []string, events []Event) Report {
	r := Report{Verdict: VerdictInsufficient, Servers: make([]Server, 0, len(names))}
	allReady := len(names) > 0
	hasFailure, hasCancellation := false, false
	for _, name := range names {
		status := "missing"
		needle := strings.ToLower(strings.TrimSpace(name))
		for _, e := range events {
			text := strings.ToLower(e.Target + " " + e.Body)
			if needle == "" || !strings.Contains(text, needle) {
				continue
			}
			switch {
			case containsAny(text, "cancelled", "canceled", "runtime refresh", "configuration reload", "startup interrupted"):
				if status != "failure" {
					status = "cancellation"
				}
			case strings.EqualFold(e.Level, "error") || containsAny(text, "failed to initialize", "startup failed", "timed out", "timeout", "connection refused"):
				status = "failure"
			case containsAny(text, "initialized", "initialize success", "startup complete", "server ready", "tools/list completed"):
				if status == "missing" {
					status = "ready"
				}
			}
		}
		r.Servers = append(r.Servers, Server{Name: name, Status: status})
		allReady = allReady && status == "ready"
		hasFailure = hasFailure || status == "failure"
		hasCancellation = hasCancellation || status == "cancellation"
	}
	switch {
	case hasFailure:
		r.Verdict = VerdictServerFailure
	case hasCancellation:
		r.Verdict = VerdictRuntimeCancellation
	case allReady:
		r.Verdict = VerdictFalseNegative
	}
	return r
}
func containsAny(s string, xs ...string) bool {
	for _, x := range xs {
		if strings.Contains(s, x) {
			return true
		}
	}
	return false
}
