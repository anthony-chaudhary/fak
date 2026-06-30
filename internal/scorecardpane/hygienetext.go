package scorecardpane

// hygienetext.go — the pure text/prose helpers the hygiene fold runs over each
// reader-facing doc: prose-only stripping, k-word shingles, AI-tell matching,
// reading ease, image alt-text, acronym detection. Ported from the Python module
// helpers (_prose_only, shingles, image_alt_defects, flesch, etc.) so the same doc
// scores identically.

import (
	"fmt"
	"hash/fnv"
	"regexp"
	"sort"
	"strings"
)

var (
	linkRE          = regexp.MustCompile(`\[([^\]]+)\]\(([^)]+)\)`)
	datedRE         = regexp.MustCompile(`(20\d\d-\d\d-\d\d)|(-\d{3,}\.md$)`)
	fenceRE         = regexp.MustCompile("^(```|~~~)")
	acronymRE       = regexp.MustCompile(`\b([A-Z]{2,5})s?\b`)
	inlineCodeRE    = regexp.MustCompile("`[^`]*`")
	wordRE          = regexp.MustCompile(`[a-z0-9]+`)
	wordCountRE     = regexp.MustCompile(`[A-Za-z0-9][A-Za-z0-9'×%/-]*`)
	mdImgRE         = regexp.MustCompile(`!\[([^\]]*)\]\(([^)\s]+)`)
	htmlImgRE       = regexp.MustCompile(`(?is)<img\b[^>]*>`)
	htmlAltRE       = regexp.MustCompile(`(?is)\balt\s*=\s*(?:"([^"]*)"|'([^']*)')`)
	htmlSrcRE       = regexp.MustCompile(`(?is)\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)')`)
	letterRE        = regexp.MustCompile(`[A-Za-z]+`)
	sentSplitRE     = regexp.MustCompile(`[.!?]+\s`)
	vowelRunRE      = regexp.MustCompile(`[aeiouy]+`)
	syllableCleanRE = regexp.MustCompile(`[^a-z]`)
)

// clichePhrases / llmScaffoldPhrases are the vetted doc_appeal lists (the Python
// fallback copies used when the sibling import is unavailable).
var clichePhrases = []string{
	"delve", "leverage", "seamless", "robust", "cutting-edge",
	"in today's", "landscape of", "tapestry", "boasts", "myriad",
	"supercharge", "effortless", "world-class", "paradigm shift",
}
var llmScaffoldPhrases = []string{
	"here's the thing", "the key insight", "at its core",
	"in a nutshell", "the bottom line", "it's important to note",
	"when it comes to", "the reality is", "let's be clear",
}

// jargonTerms is the docs-scorecard jargon list (fallback copy).
var jargonTerms = []string{
	"vDSO", "context-MMU", "RadixAttention", "KV cache", "KV-cache",
	"idempoten", "quiesce", "default-deny", "fail-open", "adjudicat",
}

// contextSensitiveTells are dropped from the corpus-wide AI-tell set (just as often
// plain technical English corpus-wide).
var contextSensitiveTells = map[string]bool{"robust": true, "leverage": true}

// aiTellPhrases is the corpus-wide HARD set: the vetted lists minus the
// context-sensitive bare words.
var aiTellPhrases = buildAITellPhrases()

func buildAITellPhrases() []string {
	var out []string
	for _, p := range append(append([]string{}, clichePhrases...), llmScaffoldPhrases...) {
		if !contextSensitiveTells[strings.ToLower(p)] {
			out = append(out, p)
		}
	}
	return out
}

// literalIdioms are figurative phrases a literal reader / translator stumbles on.
var literalIdioms = []string{
	"rule of thumb", "low-hanging fruit", "boils down to", "under the hood",
	"out of the box", "move the needle", "bite the bullet", "the elephant in the room",
	"ballpark", "a far cry", "hand-wavy", "bread and butter", "silver bullet",
	"wild goose chase", "back of the envelope",
	"tip of the iceberg", "apples to apples", "apples to oranges", "moving target",
	"north star", "happy path", "raise the bar", "drop in the bucket",
	"throw over the wall", "cut corners", "in the weeds", "on the same page",
	"draw a line in the sand", "the lay of the land", "barking up the wrong tree",
	"piece of cake", "the whole nine yards", "foot the bill", "yak shaving",
	"the long pole in the tent", "boiling the ocean", "secret sauce",
}

// commonAcronyms — acronyms common enough that using them unexpanded does not block
// a reader (general computing + this project's domain vocabulary).
var commonAcronyms = map[string]bool{
	"API": true, "CLI": true, "URL": true, "URI": true, "HTTP": true, "HTTPS": true,
	"JSON": true, "YAML": true, "TOML": true, "HTML": true, "CSS": true, "CPU": true,
	"GPU": true, "RAM": true, "OS": true, "SDK": true, "ID": true, "UI": true, "UX": true,
	"IO": true, "TLS": true, "SSL": true, "DNS": true, "TCP": true, "UDP": true, "IP": true,
	"RPC": true, "REST": true, "RPS": true, "QPS": true, "VM": true, "FS": true, "DB": true,
	"SQL": true, "OK": true, "FAQ": true, "TOC": true, "TODO": true, "FIXME": true, "CI": true,
	"CD": true, "PR": true, "VCS": true, "DSL": true, "AST": true, "SIMD": true, "AVX": true,
	"NUMA": true, "OOM": true, "GEMM": true, "TLB": true, "FFI": true, "JIT": true, "AOT": true,
	"EOF": true, "UTF": true, "ASCII": true, "RNG": true, "UUID": true, "SHA": true, "HMAC": true,
	"RSA": true, "AES": true, "JWT": true, "CORS": true, "RBAC": true, "ACL": true, "PII": true,
	"E2E": true, "SLA": true, "SLO": true, "HA": true, "P0": true, "P1": true, "P2": true, "OSS": true,
	"MIT": true, "BSD": true, "GNU": true, "LGPL": true, "GPL": true, "DCO": true, "CLA": true,
	"GDPR": true, "ABI": true, "DAG": true, "WAL": true, "CAS": true, "TTL": true, "MMU": true,
	"DOS": true, "SOTA": true, "AEO": true, "SEO": true, "RSI": true, "FAK": true, "AMD": true,
	"ARM": true, "DGX": true, "TP": true, "PP": true, "DP": true, "AI": true, "ML": true,
	"LLM": true, "KV": true, "MCP": true, "GGUF": true, "AWQ": true, "GPTQ": true, "RoPE": true,
	"ROPE": true, "MoE": true, "LoRA": true, "RAG": true, "NLP": true, "OWASP": true, "CUDA": true,
	"HAL": true, "IFC": true, "CFI": true, "KPI": true, "MVP": true, "ETA": true, "AKA": true,
	"FYI": true, "TLDR": true,
}

// aiTellMatchers precompiles a word-boundary matcher for each AI-tell phrase. A
// single bare word is guarded against hyphen compounds; a multi-word/hyphenated
// phrase matches whole on plain word boundaries.
type tellMatcher struct {
	phrase string
	re     *regexp.Regexp
}

var aiTellMatchers = buildTellMatchers()

func buildTellMatchers() []tellMatcher {
	var out []tellMatcher
	for _, ph := range aiTellPhrases {
		esc := regexp.QuoteMeta(strings.ToLower(ph))
		var pat string
		if strings.Contains(ph, " ") || strings.Contains(ph, "-") {
			pat = `\b` + esc + `\b`
		} else {
			// (?<![\w-]) … (?![\w-]) — Go's RE2 has no lookbehind; emulate with a
			// scan in tellHits instead. Here keep a plain match and refine in code.
			pat = esc
		}
		out = append(out, tellMatcher{phrase: ph, re: regexp.MustCompile(pat)})
	}
	return out
}

// isWordOrHyphen reports whether b is a word char or hyphen (the [\w-] class the
// Python single-word tell guard excludes on each side).
func isWordOrHyphen(b byte) bool {
	return b == '-' || b == '_' ||
		(b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
}

// tellHits returns the AI-tell phrase occurrences in lowercased prose, mirroring the
// Python per-phrase finditer (one entry per occurrence). Single bare words enforce
// the (?<![\w-])…(?![\w-]) boundary by hand since RE2 lacks lookbehind.
func tellHits(low string) []string {
	var hits []string
	for _, m := range aiTellMatchers {
		single := !strings.Contains(m.phrase, " ") && !strings.Contains(m.phrase, "-")
		for _, loc := range m.re.FindAllStringIndex(low, -1) {
			if single {
				start, end := loc[0], loc[1]
				if start > 0 && isWordOrHyphen(low[start-1]) {
					continue
				}
				if end < len(low) && isWordOrHyphen(low[end]) {
					continue
				}
			}
			hits = append(hits, m.phrase)
		}
	}
	return hits
}

// hardJunkRE / softJunkRE are the FILE_ADMISSION junk patterns (the Python inline
// fallback the tool uses when check_committed_files is unimportable).
var hardJunkRE = []*regexp.Regexp{
	regexp.MustCompile(`\.(exe|dll|so|dylib|pyc|pyo|class|o|a|obj)$`),
	regexp.MustCompile(`~$`),
	regexp.MustCompile(`(^|/)__pycache__/`),
}
var softJunkRE = []*regexp.Regexp{
	regexp.MustCompile(`\.log$`),
	regexp.MustCompile(`\.tmp$`),
	regexp.MustCompile(`(^|/)(report|agent-report)\.json$`),
}

// classifyJunk returns the junk reason for a tracked path, or "" — reuses the
// FILE_ADMISSION rules. Ported from the Python classify_junk.
func classifyJunk(rel string) string {
	for _, rx := range hardJunkRE {
		if rx.MatchString(rel) {
			return "build artifact / cache / compiled output"
		}
	}
	exempt := false
	for _, d := range exemptDataDirs {
		if strings.HasPrefix(rel, d) {
			exempt = true
			break
		}
	}
	if !exempt {
		for _, rx := range softJunkRE {
			if rx.MatchString(rel) {
				return "log / temp / report output (regenerable)"
			}
		}
	}
	return ""
}

// isDatedDoc reports a basename carrying a YYYY-MM-DD stamp or trailing issue number.
func isDatedDoc(name string) bool { return datedRE.MatchString(name) }

// isReaderFacing reports a discovery-surface doc (root front-door .md, or a docs/**
// .md outside the archival subtrees).
func isReaderFacing(rel string) bool {
	if !strings.HasSuffix(rel, ".md") || rel == generatedSnapshot {
		return false
	}
	parts := strings.Split(rel, "/")
	if len(parts) == 1 {
		return allowedRootMD[parts[0]]
	}
	if parts[0] != "docs" {
		return false
	}
	return !(len(parts) >= 3 && archiveDirs[parts[1]])
}

// proseOnly drops fenced code, inline code, and YAML front-matter so prose checks
// see only what a reader reads. Ported from _prose_only.
func proseOnly(text string) string {
	lines := strings.Split(text, "\n")
	var out []string
	inFence := false
	start := 0
	if len(lines) > 0 && strings.TrimSpace(lines[0]) == "---" {
		for j := 1; j < len(lines); j++ {
			if strings.TrimSpace(lines[j]) == "---" {
				start = j + 1
				break
			}
		}
	}
	for _, raw := range lines[start:] {
		if fenceRE.MatchString(strings.TrimSpace(raw)) {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		out = append(out, inlineCodeRE.ReplaceAllString(raw, ""))
	}
	return strings.Join(out, "\n")
}

// shingles is a set of hashed k-word shingles over the prose (deterministic).
func shingles(text string) map[uint64]struct{} {
	words := wordRE.FindAllString(strings.ToLower(proseOnly(text)), -1)
	out := map[uint64]struct{}{}
	if len(words) < shingleK {
		if len(words) > 0 {
			out[hashWords(words)] = struct{}{}
		}
		return out
	}
	for i := 0; i+shingleK <= len(words); i++ {
		out[hashWords(words[i:i+shingleK])] = struct{}{}
	}
	return out
}

func hashWords(words []string) uint64 {
	h := fnv.New64a()
	for i, w := range words {
		if i > 0 {
			h.Write([]byte{0})
		}
		h.Write([]byte(w))
	}
	return h.Sum64()
}

func wordCount(text string) int {
	return len(wordCountRE.FindAllString(text, -1))
}

func syllables(word string) int {
	w := syllableCleanRE.ReplaceAllString(strings.ToLower(word), "")
	if w == "" {
		return 0
	}
	n := len(vowelRunRE.FindAllString(w, -1))
	if strings.HasSuffix(w, "e") && n > 1 {
		n--
	}
	if n < 1 {
		return 1
	}
	return n
}

// flesch is the Flesch reading-ease score. Ported from the Python flesch().
func flesch(text string) float64 {
	var sents []string
	for _, s := range sentSplitRE.Split(text, -1) {
		if strings.TrimSpace(s) != "" {
			sents = append(sents, s)
		}
	}
	words := letterRE.FindAllString(text, -1)
	if len(sents) == 0 || len(words) == 0 {
		return 100.0
	}
	syll := 0
	for _, w := range words {
		syll += syllables(w)
	}
	return 206.835 - 1.015*(float64(len(words))/float64(len(sents))) -
		84.6*(float64(syll)/float64(len(words)))
}

// undefinedAcronyms returns recurring (2+) acronyms in prose that are not in the
// common set and never expanded anywhere in the doc. Ported from the Python
// undefined-acronym detection: a one-off mention or a glossed term is not flagged.
func undefinedAcronyms(prose string) []string {
	all := acronymRE.FindAllStringSubmatch(prose, -1)
	counts := map[string]int{}
	for _, m := range all {
		counts[m[1]]++
	}
	var keys []string
	for a := range counts {
		keys = append(keys, a)
	}
	sort.Strings(keys)
	var out []string
	for _, a := range keys {
		if counts[a] < 2 || commonAcronyms[a] {
			continue
		}
		if strings.Contains(prose, "("+a+")") {
			continue
		}
		if acronymExpansionRE(a).MatchString(prose) {
			continue
		}
		out = append(out, a)
	}
	return out
}

// acronymExpansionRE builds the "<word> (<acro>" expansion matcher for a given
// acronym, mirroring the Python rf"\b\w+(?:\s+\w+){{1,4}}\s*\(\s*{a}\b".
func acronymExpansionRE(a string) *regexp.Regexp {
	return regexp.MustCompile(`\b\w+(?:\s+\w+){1,4}\s*\(\s*` + regexp.QuoteMeta(a) + `\b`)
}

// fmtRound0 renders a float rounded to a whole number (the Python "%.0f").
func fmtRound0(f float64) string {
	return fmt.Sprintf("%.0f", f)
}

// imageAltDefects returns image sources whose alt-text is missing or empty. Ported
// from image_alt_defects (markdown bracket alt + HTML alt= attribute).
func imageAltDefects(text string) []string {
	var out []string
	for _, m := range mdImgRE.FindAllStringSubmatch(text, -1) {
		if strings.TrimSpace(m[1]) == "" {
			out = append(out, strings.TrimSpace(m[2]))
		}
	}
	for _, tag := range htmlImgRE.FindAllString(text, -1) {
		alt := htmlAltRE.FindStringSubmatch(tag)
		if alt == nil || strings.TrimSpace(altValue(alt)) == "" {
			src := htmlSrcRE.FindStringSubmatch(tag)
			if src != nil {
				out = append(out, strings.TrimSpace(altValue(src)))
			} else {
				out = append(out, "<img>")
			}
		}
	}
	return out
}

// altValue returns the matched quoted attribute value from a double-or-single-quote
// alternation submatch (group 1 = double-quoted, group 2 = single-quoted).
func altValue(m []string) string {
	if len(m) > 1 && m[1] != "" {
		return m[1]
	}
	if len(m) > 2 {
		return m[2]
	}
	return ""
}
