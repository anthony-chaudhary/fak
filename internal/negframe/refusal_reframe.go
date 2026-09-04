package negframe

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// RefusalReframeResult is the outcome of a refusal reframe pass (#11044).
type RefusalReframeResult struct {
	Text            string   `json:"text"`
	Applied         int      `json:"applied"`
	PreservedTokens []string `json:"preserved_tokens,omitempty"`
}

// RefusalReframe transforms negatively-framed refusal summaries, boundaries, and error
// guidance into affordance-first, prescriptive, constructive guidance (#11044).
// It converts negative idioms (e.g. "DENIED:", "CANNOT", "REFUSED: do not edit X",
// "modify only implementation files; test modifications are forbidden") into affirmative
// instructions while preserving typed machine-readable tokens (ReasonCode, rule keys,
// paths, and error identifiers like TEST_TAMPER_REFUSED).
func RefusalReframe(prose string) string {
	return RefusalReframePass(prose).Text
}

// ReframeRefusalProse is an alias for RefusalReframe matching ticket #11044.
func ReframeRefusalProse(prose string) string {
	return RefusalReframe(prose)
}

// RefusalReframePass runs the refusal reframe pass and returns the rewritten prose
// along with telemetry on applied transformations and preserved tokens.
func RefusalReframePass(prose string) RefusalReframeResult {
	if prose == "" {
		return RefusalReframeResult{}
	}

	lines := strings.Split(prose, "\n")
	outLines := make([]string, len(lines))
	totalApplied := 0
	tokenSet := make(map[string]bool)

	inFence := false
	for i, raw := range lines {
		trimmed := strings.TrimSpace(raw)
		isFence := strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~")
		if isFence {
			inFence = !inFence
		}
		if isFence || inFence || trimmed == "" {
			outLines[i] = raw
			continue
		}

		rewritten, applied, tokens := reframeRefusalLine(raw)
		outLines[i] = rewritten
		totalApplied += applied
		for _, tok := range tokens {
			tokenSet[tok] = true
		}
	}

	var preserved []string
	for tok := range tokenSet {
		preserved = append(preserved, tok)
	}
	sort.Strings(preserved)

	return RefusalReframeResult{
		Text:            strings.Join(outLines, "\n"),
		Applied:         totalApplied,
		PreservedTokens: preserved,
	}
}

type tokenSpan struct {
	start int
	end   int
	text  string
}

var (
	backtickTokenRE  = regexp.MustCompile("`[^`\n]+`")
	quotedTokenRE    = regexp.MustCompile(`"[^"\n]+"|'[^'\n]+'`)
	keyValueTokenRE  = regexp.MustCompile(`\b[a-zA-Z_][a-zA-Z0-9_]*=[^\s,;)]+`)
	keyColonTokenRE  = regexp.MustCompile(`\b(?:rule|target|lane|reason|ReasonCode|code|disposition|tool):\s*[^\s,;)]+`)
	reasonNameRE     = regexp.MustCompile(`\bReason[A-Z][a-zA-Z0-9]+\b`)
	compoundReasonRE = regexp.MustCompile(`\b[A-Z0-9]+_[A-Z0-9_]+\b`)
	pathTokenRE      = regexp.MustCompile(`\b[\w.-]+[/\\][\w.*-]+(?:[/\\][\w.*-]+)*\b`)
	wildcardTokenRE  = regexp.MustCompile(`\b\*\.[\w]+\b|\b\*\*/[\w.*-]+\b`)
	filenameTokenRE  = regexp.MustCompile(`\b[a-zA-Z0-9_.-]+\.(?:go|ts|js|py|json|toml|yaml|yml|md|txt|sh|ps1|rs|c|cpp|h|proto)\b`)
	numericCodeRE    = regexp.MustCompile(`\b\d{3,}\b`)
)

func extractRefusalTokens(line string) (string, []string) {
	var claimed []tokenSpan
	overlaps := func(s, e int) bool {
		for _, c := range claimed {
			if s < c.end && c.start < e {
				return true
			}
		}
		return false
	}

	tokenPatterns := []*regexp.Regexp{
		backtickTokenRE,
		quotedTokenRE,
		keyValueTokenRE,
		keyColonTokenRE,
		reasonNameRE,
		compoundReasonRE,
		pathTokenRE,
		wildcardTokenRE,
		filenameTokenRE,
		numericCodeRE,
	}

	for _, pat := range tokenPatterns {
		for _, loc := range pat.FindAllStringIndex(line, -1) {
			s, e := loc[0], loc[1]
			if overlaps(s, e) {
				continue
			}
			claimed = append(claimed, tokenSpan{start: s, end: e, text: line[s:e]})
		}
	}

	if len(claimed) == 0 {
		return line, nil
	}

	sort.Slice(claimed, func(i, j int) bool {
		return claimed[i].start < claimed[j].start
	})

	tokens := make([]string, len(claimed))
	for i, c := range claimed {
		tokens[i] = c.text
	}

	out := line
	for i := len(claimed) - 1; i >= 0; i-- {
		c := claimed[i]
		placeholder := fmt.Sprintf("\x00__FAK_REFUSAL_TOK_%d__\x00", i)
		out = out[:c.start] + placeholder + out[c.end:]
	}

	return out, tokens
}

func restoreRefusalTokens(line string, tokens []string) string {
	out := line
	for i, tok := range tokens {
		placeholder := fmt.Sprintf("\x00__FAK_REFUSAL_TOK_%d__\x00", i)
		out = strings.ReplaceAll(out, placeholder, tok)
	}
	return out
}

type refusalPatternRule struct {
	pattern *regexp.Regexp
	repl    string
}

var refusalRules = []refusalPatternRule{
	// 1. Specific test-immunity / implementation-lane compound phrasing
	{
		pattern: regexp.MustCompile(`(?i)\b(?:modify\s+only\s+implementation\s+files|modify\s+implementation\s+files)(?:\s+in\s+your\s+assigned\s+lane)?\s*;\s*test\s+modifications\s+are\s+forbidden\b`),
		repl:    "modify implementation files in your assigned lane; preserve test files intact",
	},
	{
		pattern: regexp.MustCompile(`(?i)\btest\s+modifications\s+are\s+forbidden\b`),
		repl:    "preserve test files intact",
	},
	{
		pattern: regexp.MustCompile(`(?i)\btest\s+modifications\s+are\s+not\s+allowed\b`),
		repl:    "preserve test files intact",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(\w+(?:\s+\w+)*)\s+modifications\s+are\s+forbidden\b`),
		repl:    "preserve $1 files intact",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(\w+(?:\s+\w+)*)\s+modifications\s+are\s+not\s+allowed\b`),
		repl:    "preserve $1 files intact",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bmodify\s+only\s+implementation\s+files\b`),
		repl:    "modify implementation files in your assigned lane",
	},

	// 2. Headings and prefixes
	{
		pattern: regexp.MustCompile(`(?i)(?:^|\b)(?:DENIED|REFUSED):\s+`),
		repl:    "ACTION REQUIRED: ",
	},
	{
		pattern: regexp.MustCompile(`(?i)(?:^|\b)(?:DENIED|REFUSED):$`),
		repl:    "ACTION REQUIRED:",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bthe\s+kernel\s+refused\s+this\s+tool\s+call\b`),
		repl:    "the kernel held this tool call",
	},

	// 3. CANNOT / cannot
	{
		pattern: regexp.MustCompile(`(?i)\bcannot\s+be\s+(\w+)(?:ed|d)\b`),
		repl:    "must remain intact",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bcannot\s+(?:edit|modify|change|touch|tamper\s+with)\s+`),
		repl:    "preserve ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bcannot\s+(?:delete|remove)\s+`),
		repl:    "retain ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bcannot\s+write\s+to\s+`),
		repl:    "preserve ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bcannot\s+(?:execute|run)\s+`),
		repl:    "use permitted tools instead of executing ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bcannot\s+access\s+`),
		repl:    "obtain authorization before accessing ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bcannot\s+(?:bypass|override)\s+`),
		repl:    "must honor ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bcannot\s+hold\s+`),
		repl:    "is insufficient to hold ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bcannot\s+(\w+)\b`),
		repl:    "must use permitted alternatives to $1",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bCANNOT\b`),
		repl:    "must use permitted alternatives",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bcannot\b`),
		repl:    "must use permitted alternatives",
	},

	// 4. "do not" / "don't"
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+forget\s+to\s+(\w+)`),
		repl:    "remember to $1",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+hesitate\s+to\s+(\w+)`),
		repl:    "feel free to $1",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bno\s+need\s+to\s+(\w+)`),
		repl:    "you can skip $1",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bmake\s+sure\s+(?:that\s+)?you\s+do\s+not\s+(\w+)`),
		repl:    "ensure you preserve $1",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+(?:edit|modify|change|touch|tamper\s+with)\s+`),
		repl:    "preserve ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+(?:delete|remove)\s+`),
		repl:    "retain ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+write\s+to\s+`),
		repl:    "preserve ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+(?:execute|run)\s+`),
		repl:    "use permitted tools instead of executing ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+dial\s+`),
		repl:    "connect only to authorized endpoints instead of dialing ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+(?:bypass|override)\s+`),
		repl:    "honor ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+re-?propose\s+`),
		repl:    "choose an allowed alternative to ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+retry\b`),
		repl:    "address root cause before retrying",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+lower\s+`),
		repl:    "maintain ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+raise\s+`),
		repl:    "maintain ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+clear\s+`),
		repl:    "retain ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+consume\s+`),
		repl:    "rely on verified ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+admit\s+`),
		repl:    "require verification before admitting ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+work\s+around\s+`),
		repl:    "resolve directly rather than circumventing ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+push\b`),
		repl:    "keep commits local",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(?:do\s+not|don'?t)\s+(\w+)\b`),
		repl:    "use permitted alternatives instead of $1ing",
	},

	// 5. "forbidden"
	{
		pattern: regexp.MustCompile(`(?i)\b(\w+(?:\s+\w+)*)\s+is\s+forbidden\b`),
		repl:    "preserve $1 intact",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(\w+(?:\s+\w+)*)\s+are\s+forbidden\b`),
		repl:    "preserve $1 intact",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bis\s+forbidden\b`),
		repl:    "requires explicit authorization",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bare\s+forbidden\b`),
		repl:    "require explicit authorization",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bforbidden\b`),
		repl:    "restricted",
	},

	// 6. "not allowed"
	{
		pattern: regexp.MustCompile(`(?i)\b(\w+(?:\s+\w+)*)\s+is\s+not\s+allowed\b`),
		repl:    "$1 requires explicit authorization",
	},
	{
		pattern: regexp.MustCompile(`(?i)\b(\w+(?:\s+\w+)*)\s+are\s+not\s+allowed\b`),
		repl:    "$1 require explicit authorization",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bnot\s+allowed\s+to\s+(\w+)\b`),
		repl:    "must obtain authorization to $1",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bis\s+not\s+allowed\b`),
		repl:    "requires explicit authorization",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bare\s+not\s+allowed\b`),
		repl:    "require explicit authorization",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bnot\s+allowed\b`),
		repl:    "restricted to authorized callers",
	},

	// 7. "may not" / "must not"
	{
		pattern: regexp.MustCompile(`(?i)\bmay\s+not\s+(?:edit|modify|change|write\s+to)\s+`),
		repl:    "must preserve ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bmay\s+not\s+(\w+)\b`),
		repl:    "requires authorization to $1",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bmust\s+not\s+(?:edit|modify|change|write\s+to)\s+`),
		repl:    "must preserve ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bmust\s+not\s+enter\s+`),
		repl:    "must remain outside ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bmust\s+not\s+regress\s+`),
		repl:    "must maintain or improve ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bmust\s+not\s+(\w+)\b`),
		repl:    "must use permitted alternatives to $1",
	},

	// 8. "never"
	{
		pattern: regexp.MustCompile(`(?i)\bnever\s+(?:edit|modify|change|touch|delete)\s+`),
		repl:    "preserve ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bnever\s+(?:bypass|override)\s+`),
		repl:    "always honor ",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bnever\s+force-?push\b`),
		repl:    "push with standard verification",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bnever\s+merge\s+without\s+review\b`),
		repl:    "merge only with required review",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bis\s+never\s+emitted\b`),
		repl:    "is withheld",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bnever\s+(\w+)\b`),
		repl:    "ensure you $1 only through authorized channels",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bnever\b`),
		repl:    "ensure compliance rather than",
	},

	// 9. "avoid"
	{
		pattern: regexp.MustCompile(`(?i)\bavoid\s+raw\s+git\s+commit\b`),
		repl:    "prefer fak commit",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bavoid\s+(\w+)ing\b`),
		repl:    "prefer permitted alternatives to $1ing",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bavoid\s+(\w+)\b`),
		repl:    "prefer permitted alternatives to $1",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bavoid\b`),
		repl:    "prefer permitted alternatives to",
	},

	// 10. "refuse to" / "refuses to"
	{
		pattern: regexp.MustCompile(`(?i)\brefuses?\s+to\s+(\w+)\b`),
		repl:    "requires authorization to $1",
	},
	{
		pattern: regexp.MustCompile(`(?i)\brefused\s+because\b`),
		repl:    "held because",
	},

	// 11. "without"
	{
		pattern: regexp.MustCompile(`(?i)\bwithout\s+review\b`),
		repl:    "with required review",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bwithout\s+authorization\b`),
		repl:    "with required authorization",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bwithout\s+permission\b`),
		repl:    "with required permission",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bwithout\s+a\s+spine\s+witness\b`),
		repl:    "with a confirmed spine witness",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bwithout\s+meaningful\s+([a-zA-Z\s]+)\s+proof\b`),
		repl:    "with meaningful $1 proof",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bwithout\s+(\w+)\b`),
		repl:    "with required $1",
	},

	// 12. "fails to" / "failed to"
	{
		pattern: regexp.MustCompile(`(?i)\bfails?\s+to\s+(\w+)\b`),
		repl:    "must $1",
	},
	{
		pattern: regexp.MustCompile(`(?i)\bfailed\s+to\s+(\w+)\b`),
		repl:    "must $1",
	},

	// 13. Hedges / double negatives
	{
		pattern: regexp.MustCompile(`(?i)\bnot\s+un(readable|usual|common|clear|likely|necessary|important|safe|able|available|aware|certain|reasonable|realistic|helpful|related|expected|documented|reachable|recoverable|ambiguous|desirable|acceptable|familiar|known|wise|fair|kind|true|real|stable|bounded|intended|wanted|used|changed|tested|defined|limited|restricted|planned|warranted|justified|founded|biased|happy|healthy|even)\b`),
		repl:    "$1",
	},
}

func matchCase(orig, repl string) string {
	if len(orig) > 0 && orig[0] >= 'A' && orig[0] <= 'Z' {
		if len(repl) > 0 && repl[0] >= 'a' && repl[0] <= 'z' {
			return strings.ToUpper(repl[:1]) + repl[1:]
		}
	}
	return repl
}

func reframeRefusalLine(line string) (string, int, []string) {
	masked, tokens := extractRefusalTokens(line)
	applied := 0

	cur := masked
	for _, rule := range refusalRules {
		if rule.pattern.MatchString(cur) {
			cur = rule.pattern.ReplaceAllStringFunc(cur, func(match string) string {
				applied++
				expanded := match
				if strings.Contains(rule.repl, "$") {
					expanded = rule.pattern.ReplaceAllString(match, rule.repl)
				} else {
					expanded = rule.repl
				}
				return matchCase(match, expanded)
			})
		}
	}

	restored := restoreRefusalTokens(cur, tokens)
	return restored, applied, tokens
}
