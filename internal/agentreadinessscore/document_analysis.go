package agentreadinessscore

import (
	"strconv"
	"strings"
)

// round1 rounds to one decimal the way Python round(x, 1) does (correctly-rounded,
// half-to-even) by round-tripping through fixed-precision formatting.
func round1(x float64) float64 {
	v, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', 1, 64), 64)
	return v
}

// experienceFrontier is the UNBOUNDED agent-experience frontier: sum of weight*count over
// every dimension, plus the per-dimension breakdown. A missing fact counts as zero.
func experienceFrontier(facts map[string]int) (int, map[string]int) {
	byTerm := make(map[string]int, len(frontierDims))
	total := 0
	for _, dim := range frontierDims {
		v := facts[dim]
		if v < 0 {
			v = 0
		}
		byTerm[dim] = frontierUnits[dim] * v
		total += byTerm[dim]
	}
	return total, byTerm
}

// has reports whether text (case-insensitive) contains any of the tokens.
func has(text string, tokens ...string) bool {
	if text == "" {
		return false
	}
	low := strings.ToLower(text)
	for _, t := range tokens {
		if strings.Contains(low, strings.ToLower(t)) {
			return true
		}
	}
	return false
}

// fencedBlocks returns the contents of every fenced code block.
func fencedBlocks(text string) []string {
	var blocks []string
	var cur []string
	inFence := false
	for _, raw := range strings.Split(text, "\n") {
		if fenceRe.MatchString(strings.TrimSpace(raw)) {
			if inFence {
				blocks = append(blocks, strings.Join(cur, "\n"))
				cur = nil
			}
			inFence = !inFence
			continue
		}
		if inFence {
			cur = append(cur, raw)
		}
	}
	return blocks
}

// proseOutsideFences returns the document with fenced blocks (and fence lines) removed.
func proseOutsideFences(text string) string {
	var out []string
	inFence := false
	for _, raw := range strings.Split(text, "\n") {
		if fenceRe.MatchString(strings.TrimSpace(raw)) {
			inFence = !inFence
			continue
		}
		if !inFence {
			out = append(out, raw)
		}
	}
	return strings.Join(out, "\n")
}

func isTemplateSlot(tok string) bool {
	return bracketSlotRe.MatchString(tok) || envSlotRe.MatchString(tok)
}

// splitOnceWSHash returns the segment before the first ` # ` inline comment.
func splitOnceWSHash(s string) string {
	if loc := wsHashRe.FindStringIndex(s); loc != nil {
		return s[:loc[0]]
	}
	return s
}

// pathOperands returns the repo-relative-looking path operands in a fenced command block.
func pathOperands(block string) []string {
	var ops []string
	for _, raw := range strings.Split(block, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		var code string
		if strings.HasPrefix(strings.TrimLeft(line, " \t"), "\"") {
			code = line
		} else {
			code = splitOnceWSHash(line)
		}
		for _, tok := range wsRe.Split(code, -1) {
			rawTok := strings.TrimSpace(tok)
			t := strings.Trim(rawTok, "\"'`,;\\")
			if t == "" || isTemplateSlot(t) || strings.ContainsAny(rawTok, "\"[]{}") {
				continue
			}
			low := strings.ToLower(t)
			isRepoRel := (strings.HasPrefix(low, "./") && strings.Contains(low[2:], "/"))
			if !isRepoRel {
				for _, d := range repoTopDirs {
					if strings.HasPrefix(low, d) {
						isRepoRel = true
						break
					}
				}
			}
			if !isRepoRel {
				for _, p := range stalePathPrefixes {
					if strings.HasPrefix(low, p) {
						isRepoRel = true
						break
					}
				}
			}
			if isRepoRel {
				ops = append(ops, t)
			}
		}
		if m := cdRe.FindStringSubmatch(code); m != nil {
			tgt := strings.Trim(m[1], "\"'`")
			if tgt != "" && !isTemplateSlot(tgt) && !strings.HasPrefix(tgt, "/") &&
				!strings.HasPrefix(tgt, "~") && !strings.HasPrefix(tgt, "$") {
				ops = append(ops, tgt)
			}
		}
	}
	return ops
}

// findFirstCommand reports whether a no-key/no-model/no-GPU first command sits inside a
// fenced block of an adoption doc, and where.
func findFirstCommand(texts map[string]string) (bool, string) {
	for _, doc := range firstCommandDocs {
		for _, block := range fencedBlocks(texts[doc]) {
			if has(block, firstCommandTokens...) {
				return true, doc
			}
		}
	}
	return false, ""
}

func findInstallOneliner(texts map[string]string) (bool, string) {
	for _, doc := range installDocs {
		t := texts[doc]
		all := true
		for _, tok := range installTokens {
			if !has(t, tok) {
				all = false
				break
			}
		}
		if all {
			return true, doc
		}
	}
	return false, ""
}

type identityEvidence struct {
	Doc       string
	Line      int
	Statement string
}

type identityParagraph struct {
	Line int
	Text string
}

func findIdentity(texts map[string]string) (present, missing []string) {
	evidence, missing := findIdentityEvidence(texts)
	present = make([]string, 0, len(evidence))
	for _, match := range evidence {
		present = append(present, match.Doc)
	}
	return present, missing
}

func findIdentityEvidence(texts map[string]string) (matches []identityEvidence, missing []string) {
	for _, doc := range identityDocs {
		matched := false
		for _, paragraph := range identityParagraphs(texts[doc]) {
			statement := normalizeIdentityText(paragraph.Text)
			if !identityStatementMatches(statement) {
				continue
			}
			matches = append(matches, identityEvidence{Doc: doc, Line: paragraph.Line, Statement: boundedIdentityStatement(statement)})
			matched = true
			break
		}
		if !matched {
			missing = append(missing, doc)
		}
	}
	return matches, missing
}

// identityParagraphs keeps the ordinary near-the-top bound, but also admits the
// first prose under an explicitly named identity heading. This lets a command-dense
// AGENTS.md put operational rules first without making its "What this project is"
// section invisible to the readiness gate.
func identityParagraphs(text string) []identityParagraph {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	paragraphs := []identityParagraph{}
	parts := []string{}
	startLine := 0
	eligible := false
	inIdentitySection := false
	flush := func() {
		if eligible && len(parts) > 0 {
			paragraphs = append(paragraphs, identityParagraph{Line: startLine, Text: strings.Join(parts, " ")})
		}
		parts = parts[:0]
		startLine = 0
		eligible = false
	}
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "#") {
			flush()
			inIdentitySection = identityHeading(strings.TrimSpace(strings.TrimLeft(trimmed, "#")))
			continue
		}
		if trimmed == "" {
			flush()
			continue
		}
		if len(parts) == 0 {
			startLine = i + 1
			eligible = startLine <= identityHeadLines || inIdentitySection
		}
		parts = append(parts, trimmed)
	}
	flush()
	return paragraphs
}

func identityHeading(heading string) bool {
	heading = strings.ToLower(normalizeIdentityText(heading))
	return heading == "what this project is" || heading == "what fak is" || heading == "about fak" || heading == "fak identity"
}

func normalizeIdentityText(text string) string {
	replacer := strings.NewReplacer("**", "", "__", "", "*", "", "_", "", "`", "")
	return strings.Join(strings.Fields(replacer.Replace(text)), " ")
}

func identityStatementMatches(statement string) bool {
	if identityDefinitionRe.MatchString(statement) || identityEnableRe.MatchString(statement) {
		return true
	}
	return identityTaskConfigRe.MatchString(statement) && identityBoundaryRe.MatchString(statement)
}

func boundedIdentityStatement(statement string) string {
	const maxRunes = 180
	runes := []rune(statement)
	if len(runes) <= maxRunes {
		return statement
	}
	return string(runes[:maxRunes-1]) + "…"
}

func missingRecipes(present map[string]bool) []string {
	out := []string{}
	for _, r := range requiredRecipes {
		if !present[r.label] {
			out = append(out, r.label)
		}
	}
	return out
}

func missingAgentConfigs(present map[string]bool) []string {
	out := []string{}
	for _, c := range agentConfigs {
		if !present[c.label] {
			out = append(out, c.label)
		}
	}
	return out
}

func missingGuardrails(agentsText string) []string {
	out := []string{}
	for _, g := range guardrailClusters {
		if !has(agentsText, g.paths...) {
			out = append(out, g.label)
		}
	}
	return out
}

func codexRecipeGaps(text string, present bool) []string {
	if !present {
		return []string{"missing " + codexFile + " — the Codex/OpenAI recipe an agent follows"}
	}
	gaps := []string{}
	for _, cl := range codexRecipeClusters {
		var missing []string
		for _, tok := range cl.tokens {
			if !has(text, tok) {
				missing = append(missing, tok)
			}
		}
		if len(missing) > 0 {
			gaps = append(gaps, codexFile+" missing "+cl.label+": "+strings.Join(missing, ", "))
		}
	}
	for _, tok := range staleCodexRecipeTokens {
		if has(text, tok) {
			gaps = append(gaps, codexFile+" still carries stale Codex-era copy: "+tok)
		}
	}
	return gaps
}
