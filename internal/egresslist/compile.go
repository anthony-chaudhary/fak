package egresslist

import "strings"

// Builder accumulates rules from several sources — explicit operator rules and parsed
// community filter lists — and compiles them into one immutable List. It exists so the
// policy loader can fold many inputs (allow_hosts, block_hosts, and each named
// block_list) into a single matcher built once at policy-load time, never on the hot
// path. A rule added as Allow always wins at Decide time regardless of add order, so the
// builder never has to reason about precedence — that lives in Decide.
type Builder struct {
	block map[string]entry
	allow map[string]entry
}

// NewBuilder returns an empty Builder.
func NewBuilder() *Builder {
	return &Builder{block: map[string]entry{}, allow: map[string]entry{}}
}

// AddRules folds a slice of already-bare domain rules (an operator's allow_hosts /
// block_hosts) into the builder under the given kind. Blank and comment entries are
// skipped; each domain is normalized so "Example.COM" and "example.com" collapse. A
// later add for the same domain overwrites the source label but not the outcome, so the
// most recent source of a duplicated rule is the one reported.
func (b *Builder) AddRules(source string, rules []string, kind Kind) *Builder {
	for _, r := range rules {
		b.add(source, r, kind)
	}
	return b
}

// AddFilterText parses one community filter list's raw text and folds its rules in. Two
// grammars are recognized, covering the overwhelming majority of host-level entries in
// the popular lists:
//
//   - hosts-file lines ("0.0.0.0 ads.example.com", "127.0.0.1 tracker.example.net",
//     or a bare "ads.example.com"): the sink IP is dropped and each remaining field is a
//     BLOCK domain. localhost/broadcast noise is skipped.
//   - adblock domain-anchor lines ("||ads.example.com^" -> block "ads.example.com";
//     "@@||docs.example.com^" -> allow "docs.example.com").
//
// Lines this leaf does not model — element-hiding ("##"), option-bearing ("$..."),
// regex ("/.../"), and comments ("!" / "#") — are skipped, not guessed at: a host-level
// egress floor must never invent a block from a rule it did not actually understand.
// Parsing the full adblock grammar is an explicit expansion ticket.
func (b *Builder) AddFilterText(source, text string) *Builder {
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || line[0] == '#' || line[0] == '!' {
			continue // blank or comment (hosts '#', adblock '!')
		}
		if strings.Contains(line, "##") || strings.Contains(line, "#@#") {
			continue // adblock element-hiding ("domain##selector") — not a host rule.
		}
		if strings.HasPrefix(line, "||") || strings.HasPrefix(line, "@@") {
			b.addAdblock(source, line)
			continue
		}
		b.addHostsLine(source, line)
	}
	return b
}

// addAdblock folds one adblock domain-anchor line. Only pure host-anchor rules are
// taken: "||domain^" (block) and "@@||domain^" (allow). A rule bearing options ("$..."),
// a path ("/"), a wildcard ("*"), or element-hiding ("##") is skipped — it is not a
// whole-host rule, and a host-level egress floor must never approximate a scope it did
// not actually parse (honoring the "$" option grammar is an expansion ticket).
func (b *Builder) addAdblock(source, line string) {
	kind := Block
	if strings.HasPrefix(line, "@@") {
		kind = Allow
		line = line[2:]
	}
	if !strings.HasPrefix(line, "||") {
		return // "@@" without a host anchor — a URL/regex exception we don't model.
	}
	line = line[2:]
	// Any option / path / wildcard / element-hiding marker means this is not a bare
	// whole-host rule; skip it rather than guess which hosts or resource types it scopes.
	if strings.ContainsAny(line, "$*/") || strings.Contains(line, "##") {
		return
	}
	// A bare host rule may end at the '^' separator; strip a single trailing one. A '^'
	// anywhere but the end is a mid-rule separator we do not model.
	line = strings.TrimSuffix(line, "^")
	if strings.ContainsAny(line, "^") {
		return
	}
	b.add(source, line, kind)
}

// addHostsLine folds one hosts-file line: an optional sink IP followed by one or more
// host names, all treated as BLOCK rules. The sink IP (0.0.0.0 / 127.0.0.1 / ::) and the
// localhost/broadcast bookkeeping names carried by every hosts file are dropped.
func (b *Builder) addHostsLine(source, line string) {
	// Trim a trailing inline comment.
	if i := strings.IndexByte(line, '#'); i >= 0 {
		line = strings.TrimSpace(line[:i])
	}
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return
	}
	// Drop a leading sink IP so "0.0.0.0 ads.example.com" yields "ads.example.com".
	if isSinkIP(fields[0]) {
		fields = fields[1:]
	}
	for _, f := range fields {
		if isNoiseHost(f) {
			continue
		}
		b.add(source, f, Block)
	}
}

// add normalizes one domain and stores it under kind. A normalized-empty, noise, or
// implausible host is dropped so a malformed list line (element-hiding text, a regex, a
// leftover path) can never inject a rule — and, worse, never inject one that matches
// broadly. This is the single choke point every parser path funnels through.
func (b *Builder) add(source, domain string, kind Kind) {
	d := normalizeHost(domain)
	if d == "" || isNoiseHost(d) || !isPlausibleHost(d) {
		return
	}
	e := entry{rule: d, source: source}
	switch kind {
	case Allow:
		b.allow[d] = e
	case Block:
		b.block[d] = e
	}
}

// Build freezes the accumulated rules into an immutable List. The returned List shares
// no state with the Builder, so the Builder may be discarded.
func (b *Builder) Build() *List {
	return &List{block: b.block, allow: b.allow}
}

// isSinkIP reports whether a token is one of the null-route sink addresses a hosts file
// uses as the left field ("0.0.0.0", "127.0.0.1", "::", "::1").
func isSinkIP(tok string) bool {
	switch tok {
	case "0.0.0.0", "127.0.0.1", "255.255.255.255", "::", "::1", "0:0:0:0:0:0:0:1":
		return true
	}
	return false
}

// isPlausibleHost reports whether a normalized token looks like a real DNS host or IPv4
// literal an egress rule could target: at least one dot, and only DNS-valid characters
// (lower-case letters, digits, '.', '-'). It rejects the adblock artifacts that reach the
// hosts-file path — element-hiding selectors ("example.com##.banner"), regex rules
// ("/ad/"), and leftover paths — which all carry a character a hostname never does.
// IPv6 literals (colons) are rejected here; an operator blocks those via the exact
// deny_hosts path. Requiring a dot also drops bare single-label bookkeeping tokens.
func isPlausibleHost(h string) bool {
	if h == "" || !strings.Contains(h, ".") {
		return false
	}
	for i := 0; i < len(h); i++ {
		c := h[i]
		switch {
		case c >= 'a' && c <= 'z', c >= '0' && c <= '9', c == '.', c == '-':
		default:
			return false
		}
	}
	return true
}

// isNoiseHost reports whether a host token is hosts-file bookkeeping (localhost and the
// broadcast/ip6 names) that must never become a real block rule.
func isNoiseHost(h string) bool {
	switch strings.ToLower(strings.TrimSuffix(h, ".")) {
	case "", "localhost", "localhost.localdomain", "local", "broadcasthost",
		"ip6-localhost", "ip6-loopback", "ip6-localnet", "ip6-mcastprefix",
		"ip6-allnodes", "ip6-allrouters", "ip6-allhosts":
		return true
	}
	return false
}
