package negframe

import (
	"regexp"
	"sort"
	"strings"
)

const maxL1DomainMembers = 256

var l1BanPattern = regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t|never)\s+use\s+([^.!?;\n]+)`)

// L1Result records an exact structural positive-state rewrite. Refused counts
// clauses recognized as bans but retained because equivalence was unprovable.
type L1Result struct {
	Text     string `json:"text"`
	Admitted int    `json:"admitted"`
	Refused  int    `json:"refused"`
}

// RewriteL1 converts bounded "do not use A, B" clauses into the exact positive
// allow-set of domain minus the banned members. An absent, open, malformed, or
// mismatched domain leaves the source byte-identical. Fenced code is opaque.
func RewriteL1(text string, domain Domain) L1Result {
	if !validL1Domain(domain) {
		return L1Result{Text: text}
	}

	parts := strings.SplitAfter(text, "\n")
	inFence := false
	admitted, refused := 0, 0
	for i, raw := range parts {
		line, newline := strings.TrimSuffix(raw, "\n"), ""
		if strings.HasSuffix(raw, "\n") {
			newline = "\n"
		}
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			inFence = !inFence
			continue
		}
		if inFence || strings.Contains(line, "`") {
			continue
		}

		loc := l1BanPattern.FindStringSubmatchIndex(line)
		if loc == nil {
			continue
		}
		excluded, ok := parseL1Members(line[loc[2]:loc[3]], domain)
		if !ok {
			refused++
			continue
		}
		allowed := l1Complement(domain, excluded)
		if len(allowed) == 0 || !L1Equivalent(domain, excluded, allowed) {
			refused++
			continue
		}
		replacement := "use only " + strings.Join(allowed, ", ")
		if startsUpper(line[loc[0]:loc[1]]) {
			replacement = "Use" + strings.TrimPrefix(replacement, "use")
		}
		parts[i] = line[:loc[0]] + replacement + line[loc[1]:] + newline
		admitted++
	}
	return L1Result{Text: strings.Join(parts, ""), Admitted: admitted, Refused: refused}
}

// L1Equivalent admits an allow-set exactly when it is the closed domain minus
// the excluded set. Order is irrelevant for checking; emitted order remains the
// domain's deterministic authored order.
func L1Equivalent(domain Domain, excluded, allowed []string) bool {
	if !validL1Domain(domain) || len(excluded) == 0 {
		return false
	}
	want := l1Complement(domain, excluded)
	return sameMemberSet(want, allowed) && len(want) == len(allowed)
}

func validL1Domain(domain Domain) bool {
	if strings.TrimSpace(domain.Name) == "" || len(domain.Members) == 0 || len(domain.Members) > maxL1DomainMembers {
		return false
	}
	seen := make(map[string]struct{}, len(domain.Members))
	for _, member := range domain.Members {
		key := strings.ToLower(strings.TrimSpace(member))
		if key == "" {
			return false
		}
		if _, duplicate := seen[key]; duplicate {
			return false
		}
		seen[key] = struct{}{}
	}
	return true
}

func parseL1Members(raw string, domain Domain) ([]string, bool) {
	if strings.Contains(strings.ToLower(raw), " and ") {
		return nil, false
	}
	raw = strings.NewReplacer(" or ", ",", " OR ", ",", " Or ", ",").Replace(raw)
	fields := strings.Split(raw, ",")
	members := make([]string, 0, len(fields))
	seen := map[string]bool{}
	for _, field := range fields {
		candidate := strings.Trim(strings.TrimSpace(field), `"'()[]{}`)
		canonical, ok := memberOf(domain, candidate)
		if !ok || seen[strings.ToLower(canonical)] {
			return nil, false
		}
		seen[strings.ToLower(canonical)] = true
		members = append(members, canonical)
	}
	return members, len(members) > 0
}

func l1Complement(domain Domain, excluded []string) []string {
	ban := make(map[string]bool, len(excluded))
	for _, member := range excluded {
		ban[strings.ToLower(strings.TrimSpace(member))] = true
	}
	allowed := make([]string, 0, len(domain.Members)-len(excluded))
	for _, member := range domain.Members {
		if !ban[strings.ToLower(strings.TrimSpace(member))] {
			allowed = append(allowed, member)
		}
	}
	return allowed
}

func sameMemberSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	normalize := func(in []string) []string {
		out := make([]string, len(in))
		for i, value := range in {
			out[i] = strings.ToLower(strings.TrimSpace(value))
		}
		sort.Strings(out)
		return out
	}
	aa, bb := normalize(a), normalize(b)
	for i := range aa {
		if aa[i] != bb[i] {
			return false
		}
	}
	return true
}

func startsUpper(s string) bool {
	for _, r := range s {
		return r >= 'A' && r <= 'Z'
	}
	return false
}
