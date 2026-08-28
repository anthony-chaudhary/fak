package agentreadinessscore

import (
	"regexp"
	"sort"
	"strings"
)

// dispatchVerbFns names the top-level verb-routing functions main() delegates to in
// cmd/fak/main.go. Each listed helper switches on the same top-level verb name; they
// are routing-table splits, not subcommand switches
// like cmdPolicy's argv[0] switch — those must NOT leak in). If the routing table is
// split across a new helper, add its `func <name>(` header here so real verbs keep
// resolving.
var dispatchVerbFns = []string{
	"func main()",
	"func dispatchCoreVerbA(",
	"func dispatchCoreVerbB(",
	"func dispatchExtendedVerbA(",
	"func dispatchExtendedVerbB(",
	"func dispatchPrimaryVerb(",
}

// dispatchVerbs is the set of top-level verbs the binary dispatches, parsed from every
// top-level routing switch in cmd/fak/main.go (func main() plus the routing-table
// helpers it delegates to).
func dispatchVerbs(mainGoText string) map[string]bool {
	verbs := map[string]bool{}
	if mainGoText == "" {
		return verbs
	}
	lines := strings.Split(mainGoText, "\n")
	for _, fn := range dispatchVerbFns {
		collectCaseVerbs(lines, fn, verbs)
	}
	return verbs
}

// collectCaseVerbs adds every quoted `case "verb":` label in the body of the function
// whose header line begins with fnPrefix. The body runs from the header to the first
// column-0 `}` (the function's own closing brace); nested switch/closure braces are
// indented, so they never end the scan early.
func collectCaseVerbs(lines []string, fnPrefix string, verbs map[string]bool) {
	start := -1
	for i, ln := range lines {
		if strings.HasPrefix(ln, fnPrefix) {
			start = i
			break
		}
	}
	if start < 0 {
		return
	}
	for _, ln := range lines[start+1:] {
		if ln == "}" {
			break
		}
		m := caseRe.FindStringSubmatch(ln)
		if m == nil {
			continue
		}
		for _, sm := range quotedRe.FindAllStringSubmatch(m[1], -1) {
			verbs[sm[1]] = true
		}
	}
}

// commandVerbs returns every CLI verb an agent would paste from this doc, in appearance order.
func commandVerbs(text string) []string {
	var verbs []string
	fromSegment := func(seg string) {
		s := strings.TrimSpace(splitOnceWSHash(strings.TrimSpace(seg)))
		s = promptRe.ReplaceAllString(s, "")
		for {
			loc := envPrefixRe.FindStringIndex(s)
			if loc == nil {
				break
			}
			s = s[loc[1]:]
		}
		for _, pre := range cmdPrefixes {
			if strings.HasPrefix(s, pre) {
				rest := strings.TrimLeft(s[len(pre):], " \t")
				m := verbTokenRe.FindStringSubmatch(rest)
				if m != nil {
					end := len(m[1])
					next := ""
					if end < len(rest) {
						next = rest[end : end+1]
					}
					if next != ":" {
						verbs = append(verbs, m[1])
					}
				}
				return
			}
		}
	}
	for _, block := range fencedBlocks(text) {
		for _, line := range strings.Split(block, "\n") {
			if strings.HasPrefix(strings.TrimLeft(line, " \t"), "#") {
				continue
			}
			for _, seg := range segSepRe.Split(line, -1) {
				fromSegment(seg)
			}
		}
	}
	for _, m := range inlineCodeRe.FindAllStringSubmatch(proseOutsideFences(text), -1) {
		fromSegment(m[1])
	}
	return verbs
}

func dosReasonTokens(dosText string) []string {
	if dosText == "" {
		return []string{}
	}
	set := map[string]bool{}
	for _, m := range reasonRe.FindAllStringSubmatch(dosText, -1) {
		set[m[1]] = true
	}
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func queryBackedRefusalRecovery(tokens []string, dosText, recoveryText string) []string {
	if !has(recoveryText, "dos man wedge", "--explain") || !has(recoveryText, "fak recover") {
		return unmappedRefusalTokens(tokens, recoveryText)
	}
	unmapped := []string{}
	for _, token := range tokens {
		section := "[reasons." + token + "]"
		start := strings.Index(dosText, section)
		if start < 0 {
			unmapped = append(unmapped, token)
			continue
		}
		end := strings.Index(dosText[start+len(section):], "\n[reasons.")
		block := dosText[start:]
		if end >= 0 {
			block = dosText[start : start+len(section)+end]
		}
		if !regexp.MustCompile(`(?m)^summary\s*=`).MatchString(block) || !regexp.MustCompile(`(?m)^fix\s*=`).MatchString(block) {
			unmapped = append(unmapped, token)
		}
	}
	return unmapped
}
func unmappedRefusalTokens(tokens []string, recoveryText string) []string {
	if recoveryText == "" {
		return append([]string{}, tokens...)
	}
	lines := strings.Split(recoveryText, "\n")
	var cueLines []int
	for i, ln := range lines {
		if has(ln, recoveryCues...) {
			cueLines = append(cueLines, i)
		}
	}
	unmapped := []string{}
	for _, t := range tokens {
		var hits []int
		for i, ln := range lines {
			if strings.Contains(ln, t) {
				hits = append(hits, i)
			}
		}
		mapped := false
		for _, h := range hits {
			for _, c := range cueLines {
				d := h - c
				if d < 0 {
					d = -d
				}
				if d <= recoveryWindow {
					mapped = true
					break
				}
			}
			if mapped {
				break
			}
		}
		if !mapped {
			unmapped = append(unmapped, t)
		}
	}
	return unmapped
}

// quickstartSignal returns (found, hasSignal) for the 60-second proof block.
func quickstartSignal(texts map[string]string) (bool, bool) {
	for _, doc := range firstCommandDocs {
		for _, block := range fencedBlocks(texts[doc]) {
			if has(block, firstCommandTokens...) {
				return true, has(block, successSignalTokens...)
			}
		}
	}
	return false, false
}

// ---------------------------------------------------------------------------
// Per-KPI pure checks. Each returns a KPI (defects = HARD friction-debt units; soft =
// score-only judgment nudges). Slices are always non-nil so JSON emits [] not null.
// ---------------------------------------------------------------------------
