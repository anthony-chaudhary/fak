// Package egresslist is the nuanced, adblock-style site allow/block layer that sits
// ABOVE the hardwired cloud-metadata egress floor (internal/egressfloor) and BELOW the
// restrictive WebFetch research allowlist. Where egressfloor answers one frozen
// question — "is this destination the cloud-metadata SSRF class?" — this leaf answers
// the operator-shaped question: "does a maintained list of sites say block or allow
// this host?"
//
// WHY A SECOND LAYER (the "chrome-adblock for agent egress" concept). A coding agent's
// network reach is not a single yes/no. Some destinations are always hostile (metadata
// endpoints — egressfloor's job). Most are fine. But a large, churning middle exists:
// ad/tracker/malware domains a community list already curates, an operator's own
// deny-these-sites list, and a small set of explicitly-sanctioned docs hosts that must
// stay reachable even when everything else is tightened. Browser ad blockers solved the
// same shape years ago: big refreshable BLOCK lists (EasyList, StevenBlack hosts) plus
// user EXCEPTION rules (adblock's `@@`) that carve a hole back open. This leaf ports
// that model to egress adjudication: a compiled List holds block rules and allow
// (exception) rules, and Decide(host) resolves one host with adblock precedence —
// an allow rule WINS over a block rule.
//
// TIER-1 DISCIPLINE. Like egressfloor, this is a pure, allocation-light classifier that
// imports only the stdlib (strings): no os/exec, no net calls, no cgo. Fetching and
// refreshing the community lists is deliberately NOT here — that is a separate,
// network-touching concern (see the expansion tickets). This leaf only COMPILES rule
// text into a matcher and DECIDES a host against it, so it is safe to fold on the live
// tool-call decision path the adjudicator's egress rung runs it from.
//
// MATCHING. A rule is a domain. It matches its own host and every subdomain
// (rule "example.com" matches "example.com" and "a.b.example.com"), the same
// subdomain-suffix convention the research allowlist already uses. Lookup walks the
// host's parent domains (host, then strip the leftmost label, repeat) against a map, so
// a decision is O(labels-in-host), not O(rules) — the list can hold hundreds of
// thousands of entries and a decision is still a handful of map probes.
package egresslist

import "strings"

// Kind is the outcome of resolving one host against a List.
type Kind int

const (
	// None means no rule matched: the list is silent on this host and the caller
	// falls through to the next egress layer (research allowlist / default posture).
	None Kind = iota
	// Block means a block rule matched and no allow (exception) rule overrode it:
	// the host is on a deny/community list and egress to it is refused.
	Block
	// Allow means an allow (exception) rule matched: the host is explicitly
	// sanctioned. Allow wins over Block (adblock `@@` semantics), so an allow rule
	// carves a host back open even when a block rule would otherwise match it.
	Allow
)

func (k Kind) String() string {
	switch k {
	case Block:
		return "block"
	case Allow:
		return "allow"
	default:
		return "none"
	}
}

// Decision is the resolved verdict for one host, naming the rule and the source list
// that decided it so a refusal (or an allow) can be recorded with a bounded, auditable
// witness — never the full URL, only the host and which list spoke.
type Decision struct {
	Kind   Kind
	Rule   string // the domain rule that matched ("" when Kind == None)
	Source string // the list/source the rule came from ("" when Kind == None)
}

// entry is one compiled rule: the domain and the source list it came from.
type entry struct {
	rule   string
	source string
}

// List is a compiled, immutable set of block and allow rules. The zero value and a nil
// *List are valid and Decide to None for every host, so a policy with no list configured
// costs nothing and never blocks.
type List struct {
	block map[string]entry
	allow map[string]entry
}

// Decide resolves one host against the list. Precedence is adblock-standard: an allow
// (exception) rule WINS over a block rule, so a host on both a community block list and
// the operator's allow_hosts is Allowed. A nil/empty list Decides None. The host is
// normalized (lower-cased, de-ported, de-bracketed) before matching, so a caller may
// pass a raw authority.
//
// Invariant: egress list matching is fail-closed and deterministic.
// Guard: host evaluation normalizes authorities purely via string operations without network lookup.
func (l *List) Decide(host string) Decision {
	if l == nil {
		return Decision{}
	}
	h := normalizeHost(host)
	if h == "" {
		return Decision{}
	}
	// Allow is checked first and wins: an explicit exception overrides any block.
	if e, ok := matchDomain(l.allow, h); ok {
		return Decision{Kind: Allow, Rule: e.rule, Source: e.source}
	}
	if e, ok := matchDomain(l.block, h); ok {
		return Decision{Kind: Block, Rule: e.rule, Source: e.source}
	}
	return Decision{}
}

// Counts reports how many block and allow rules the list compiled to, for the policy
// Explain/summary surface.
func (l *List) Counts() (block, allow int) {
	if l == nil {
		return 0, 0
	}
	return len(l.block), len(l.allow)
}

// Empty reports whether the list carries no rules at all (so callers can skip the layer
// entirely and treat it as absent).
func (l *List) Empty() bool {
	b, a := l.Counts()
	return b == 0 && a == 0
}

// matchDomain resolves a normalized host against a rule map by walking the host and each
// of its parent domains: "a.b.example.com" probes "a.b.example.com", "b.example.com",
// "example.com", "com". The first hit wins, and because a rule is stored by its bare
// domain, a rule "example.com" matches the host and all its subdomains in O(labels).
func matchDomain(set map[string]entry, host string) (entry, bool) {
	if len(set) == 0 {
		return entry{}, false
	}
	h := host
	for {
		if e, ok := set[h]; ok {
			return e, true
		}
		i := strings.IndexByte(h, '.')
		if i < 0 {
			return entry{}, false
		}
		h = h[i+1:]
	}
}

// normalizeHost reduces a raw host or authority to the bare, comparable host: trimmed,
// lower-cased, de-bracketed ([::1] -> ::1), and de-ported (example.com:443 ->
// example.com). It is pure string work — never DNS — mirroring egressfloor's host
// reduction so the two layers agree on what "the host" is. A trailing dot (the DNS root,
// "example.com.") is stripped so "example.com." and "example.com" compare equal.
func normalizeHost(h string) string {
	h = strings.ToLower(strings.TrimSpace(h))
	h = strings.Trim(h, "[]")
	// Strip a :port only when the colon is not part of an IPv6 literal. A bracketed
	// IPv6 authority has already lost its brackets above; a bare IPv6 has multiple
	// colons and no port here, so only split when exactly one colon is present.
	if strings.Count(h, ":") == 1 {
		if i := strings.LastIndexByte(h, ':'); i >= 0 {
			h = h[:i]
		}
	}
	h = strings.TrimSuffix(h, ".")
	return h
}
