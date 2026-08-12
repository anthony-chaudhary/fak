package capindex

import (
	"sort"
	"strings"
	"unicode"
)

// ProductOutcome is one operator-language answer to "what can fak do?". This
// catalog is intentionally stdlib-only and runtime-safe: both the shipped fak
// binary and repository self-query can consume it without importing devindex.
type ProductOutcome struct {
	ID      string   `json:"id"`
	Name    string   `json:"name"`
	Summary string   `json:"summary"`
	Tags    []string `json:"tags"`
	Effect  string   `json:"effect"`
	Command []string `json:"command"`
	Detail  string   `json:"detail_ref"`
	Witness string   `json:"witness"`
}

// ProductOutcomes returns the small performance-first product catalog. Security
// remains indexed as a supporting floor and sorts behind performance outcomes
// unless the query asks for it.
func ProductOutcomes() []ProductOutcome {
	return []ProductOutcome{
		{ID: "turn-savings", Name: "Avoid unnecessary model turns", Summary: "measure turn tax and elide kernel-known work instead of paying for another model round trip", Tags: []string{"turn control", "turn tax", "turn savings", "fewer turns", "fused turn", "elision", "token efficiency", "latency"}, Effect: "read", Command: []string{"go", "run", "./cmd/turntaxdemo", "-selfcheck"}, Detail: "docs/CAPABILITIES.md#turn-savings", Witness: "internal/turntaxmeter + internal/fusedturn + cmd/turntaxdemo"},
		{ID: "context-reuse", Name: "Reuse stable prompt and context work", Summary: "reuse stable prefixes, manage resident context, and price replay versus cut or reset after cache expiry", Tags: []string{"token savings", "save tokens", "prompt cache", "prefix reuse", "context compaction", "ctxmmu", "vdso", "resume", "cache efficiency"}, Effect: "read", Command: []string{"fak", "resume", "plan", "--resident-tokens", "250000", "--idle-seconds", "7200", "--json"}, Detail: "docs/CAPABILITIES.md#context-reuse", Witness: "internal/ctxmmu + internal/vdso + docs/managed-context-continuous-usage.md"},
		{ID: "session-control", Name: "Control a live session out of band", Summary: "budget, pause, resume, throttle, steer, or stop a served session without spending another prompt turn", Tags: []string{"turn control", "turn budget", "token budget", "context budget", "session control", "steer", "pause", "throttle", "cancel"}, Effect: "mutate", Command: []string{"fak", "session", "budget", "<id>", "--turns", "N", "--tokens", "N", "--context-tokens", "N"}, Detail: "docs/CAPABILITIES.md#session-control", Witness: "internal/sessionctl + internal/sessionsignals + docs/operator-control-plane.md"},
		{ID: "model-routing", Name: "Route each call to an appropriate model", Summary: "select cheaper or specialized inference per call instead of pinning a whole session to one expensive model", Tags: []string{"token savings", "cost efficiency", "model routing", "per call model", "model ladder", "cheap model", "inference efficiency"}, Effect: "read", Command: []string{"fak", "model", "--help"}, Detail: "docs/CAPABILITIES.md#model-routing", Witness: "internal/modelroute + internal/modelladder + docs/model-routing.md"},
		{ID: "savings-observability", Name: "Attribute cache and token savings", Summary: "show reused tokens, effective cost, and total savings; ablate one frozen trace to attribute the gain", Tags: []string{"token savings", "cache savings", "cost savings", "cache value", "observability", "ablate", "same trace", "efficiency"}, Effect: "read", Command: []string{"fak", "info", "--once"}, Detail: "docs/CAPABILITIES.md#savings-observability", Witness: "internal/cachevalue + internal/cachevaluereport + docs/cache-value-rollup.md"},
		{ID: "capability-floor", Name: "Enforce the supporting capability floor", Summary: "check tool authority before execution so efficiency changes remain bounded and auditable", Tags: []string{"security", "policy", "capability floor", "preflight", "audit", "default deny"}, Effect: "read", Command: []string{"fak", "preflight", "--policy", "<file>", "--tool", "<name>", "--args", "{}"}, Detail: "docs/CAPABILITIES.md#capability-floor", Witness: "internal/policy + internal/adjudicator + docs/fak/security.md"},
	}
}

// QueryProductOutcomes ranks outcomes by operator intent. Exact phrases and
// complete token matches beat partial matches; stable declaration order breaks
// ties so the performance-first default remains deliberate and reproducible.
func QueryProductOutcomes(query string, limit int) []ProductOutcome {
	outcomes := ProductOutcomes()
	query = strings.TrimSpace(strings.ToLower(query))
	if query == "" {
		if limit > 0 && len(outcomes) > limit {
			return outcomes[:limit]
		}
		return outcomes
	}
	tokens := words(query)
	type ranked struct {
		outcome      ProductOutcome
		score, order int
	}
	var found []ranked
	for i, outcome := range outcomes {
		hay := strings.ToLower(outcome.Name + " " + outcome.Summary + " " + strings.Join(outcome.Tags, " "))
		score := 0
		if strings.Contains(hay, query) {
			score += 100
		}
		for _, token := range tokens {
			if containsWord(hay, token) {
				score += 12
			} else if strings.Contains(hay, token) {
				score += 3
			}
		}
		if score > 0 {
			found = append(found, ranked{outcome: outcome, score: score, order: i})
		}
	}
	sort.SliceStable(found, func(i, j int) bool {
		if found[i].score != found[j].score {
			return found[i].score > found[j].score
		}
		return found[i].order < found[j].order
	})
	result := make([]ProductOutcome, 0, len(found))
	for _, item := range found {
		result = append(result, item.outcome)
	}
	if limit > 0 && len(result) > limit {
		result = result[:limit]
	}
	return result
}

func words(s string) []string {
	return strings.FieldsFunc(s, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) })
}
func containsWord(hay, needle string) bool {
	for _, word := range words(hay) {
		if word == needle {
			return true
		}
	}
	return false
}
