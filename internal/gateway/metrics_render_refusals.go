package gateway

import (
	"fmt"
	"sort"
	"strings"
)

// writeScopeRefusalMetrics renders the auth-scope refusal family (#12761): each
// X-Fak-Auth-Scope claim the gateway refused with 403 scope_forbidden, keyed by
// the declared scope (read/write-normalized) and the coarse route class the
// refusal happened on (observation/observation_requests/admin/other), so an
// operator can tell an observation-polling watcher from someone probing the
// admin plane. Snapshot under the lock, then render in sorted (scope,
// route_class) order so the scrape is byte-stable; a refusal-free session
// renders the header alone.
func writeScopeRefusalMetrics(b *strings.Builder, m *gatewayMetrics) {
	snap := m.scopeRefusalSnapshot()
	writeHelpType(b, "fak_gateway_scope_refusals_total",
		"Requests refused by the X-Fak-Auth-Scope guard with 403 scope_forbidden, by declared scope and route class (observation, observation_requests, admin, other). A read-scoped agent hitting a mutating verb or any /v1/admin route lands here.",
		"counter")
	keys := make([]scopeRefusalKey, 0, len(snap))
	for k := range snap {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].scope != keys[j].scope {
			return keys[i].scope < keys[j].scope
		}
		return keys[i].routeClass < keys[j].routeClass
	})
	for _, k := range keys {
		fmt.Fprintf(b, "fak_gateway_scope_refusals_total{scope=%q,route_class=%q} %d\n", k.scope, k.routeClass, snap[k])
	}
}

// writeToolCallParseMetrics renders the tool-call parse-outcome family: what
// happened to the tool calls the upstream announced on a served turn, keyed by
// outcome (fail_closed, lifted, structured), the text-lift dialect that produced
// a lift (empty otherwise), and the closed abi drop-reason name (empty unless
// fail_closed). Snapshot under the lock, then render in sorted (outcome,
// dialect, reason) order so the scrape is byte-stable; a session with zero parse
// events renders the header alone.
func writeToolCallParseMetrics(b *strings.Builder, m *gatewayMetrics) {
	snap := m.toolCallParseSnapshot()
	writeHelpType(b, "fak_gateway_toolcall_parse_total",
		"Tool-call parse outcomes per served turn. fail_closed = the upstream announced tool calls but none survived parsing + the text-lift fallback, so the turn was refused; lifted = a text-form tool-call dialect was promoted into structured calls; structured = the turn reached adjudication carrying structured calls. dialect names the lift dialect (empty otherwise); reason is the closed abi drop-reason name (empty unless fail_closed).",
		"counter")
	keys := make([]toolCallParseKey, 0, len(snap))
	for k := range snap {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].outcome != keys[j].outcome {
			return keys[i].outcome < keys[j].outcome
		}
		if keys[i].dialect != keys[j].dialect {
			return keys[i].dialect < keys[j].dialect
		}
		return keys[i].reason < keys[j].reason
	})
	for _, k := range keys {
		fmt.Fprintf(b, "fak_gateway_toolcall_parse_total{outcome=%q,dialect=%q,reason=%q} %d\n", k.outcome, k.dialect, k.reason, snap[k])
	}
}
