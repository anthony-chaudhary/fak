// Package codexmcpdiag classifies Codex MCP startup evidence without exposing log bodies.
package codexmcpdiag

import "strings"

const (
	VerdictFalseNegative       = "CLIENT_STATUS_FALSE_NEGATIVE"
	VerdictServerFailure       = "SERVER_FAILURE"
	VerdictRuntimeCancellation = "RUNTIME_REFRESH_CANCELLATION"
	VerdictInsufficient        = "INSUFFICIENT_EVIDENCE"
)

type Event struct{ Level, Target, Body string }
type Server struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}
type Report struct {
	Verdict string   `json:"verdict"`
	Servers []Server `json:"servers"`
}

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
