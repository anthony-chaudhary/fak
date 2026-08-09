package breath

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode"
)

// ScopeNotice is the written refusal of the judgement half. Every report this package
// renders — human or JSON — carries it verbatim, because a gate whose output does not
// say which half it declined to judge silently implies it judged both.
const ScopeNotice = "breath judges the COUNTABLE half of the contract only: presence, position, " +
	"sentence count, sentence length, banned punctuation, unexpanded acronyms. It does NOT judge " +
	"accuracy, completeness, or faithfulness to the page body, and must not be extended to: good " +
	"plain-language writing ADDS content absent from the source (the elaborative explanation " +
	"phenomenon, PlainQAFact arXiv 2503.08890), so an entailment check scores the BEST-written " +
	"blocks as the LEAST faithful. Whether the block is TRUE is review's job. See docs/ONE-BREATH-CONTRACT.md."

// Kind is the closed vocabulary of breath findings.
//
// Closed is load-bearing twice over. It lets a loop route on a finding the way it routes
// on a refusal reason, and it makes the scope refusal MECHANICAL: TestNoJudgementHalfKind
// walks Kinds() and fails on any token naming accuracy, faithfulness, entailment, or
// completeness, so the judgement half cannot be smuggled in as "just one more kind".
type Kind string

const (
	// BreathMissing: the page carries no `> **In one breath:**` block at all.
	BreathMissing Kind = "BREATH_MISSING"
	// BreathMisplaced: the block exists but sits after the first `##` section heading.
	// A summary a reader reaches after three sections is not a summary.
	BreathMisplaced Kind = "BREATH_MISPLACED"
	// BreathSentenceCount: the block is outside the 2..4 sentence range.
	BreathSentenceCount Kind = "BREATH_SENTENCE_COUNT"
	// BreathSentenceLength: one sentence in the block exceeds the word ceiling.
	BreathSentenceLength Kind = "BREATH_SENTENCE_LENGTH"
	// BreathEmDash: an em-dash appends a second idea without admitting it is one.
	BreathEmDash Kind = "BREATH_EM_DASH"
	// BreathParentheses: a parenthetical is nuance the reader must hold mid-sentence.
	BreathParentheses Kind = "BREATH_PARENTHESES"
	// BreathUnexpandedAcronym: an ALLCAPS acronym is never spelled out inside the block.
	BreathUnexpandedAcronym Kind = "BREATH_UNEXPANDED_ACRONYM"
	// BreathMissingOneLine: the block has no `**One line:**` sibling — a crude version
	// with no precise version standing behind it.
	BreathMissingOneLine Kind = "BREATH_MISSING_ONE_LINE"
	// BreathScanFloor: the run examined fewer pages than its floor, so its verdict is
	// worth nothing. Reported against the gate, not against a page.
	BreathScanFloor Kind = "BREATH_SCAN_FLOOR"
)

// Kinds returns the closed vocabulary in a stable order. Used by the CLI to print the
// finding taxonomy and by TestNoJudgementHalfKind to police the scope boundary.
func Kinds() []Kind {
	return []Kind{
		BreathMissing, BreathMisplaced, BreathSentenceCount, BreathSentenceLength,
		BreathEmDash, BreathParentheses, BreathUnexpandedAcronym, BreathMissingOneLine,
		BreathScanFloor,
	}
}

// Contract is the written contract from docs/ONE-BREATH-CONTRACT.md, named once here so
// the numbers the gate enforces and the numbers the page promises can be compared by a
// test (TestContractNumbersMatchThisPage) instead of by a human retyping them.
type Contract struct {
	// Roots are the tracked-path prefixes whose `.md` pages carry the contract.
	Roots []string
	// Exempt names pages inside Roots that are not judged — the page that STATES the
	// contract necessarily quotes the marker, and keying on the marker would make the
	// page defining the rule the first page to break it.
	Exempt []string
	// MinSentences / MaxSentences bound the block's sentence count.
	MinSentences, MaxSentences int
	// MaxWords is the per-sentence word ceiling.
	MaxWords int
	// AllowAcronyms are capitalized tokens that are names, not acronyms, so they need
	// no expansion. Empty by default, deliberately: the block's reader has no background.
	AllowAcronyms []string
	// Floor is the smallest number of pages a run may examine before its verdict counts.
	Floor int
}

// The contract's own numbers, quoted from docs/ONE-BREATH-CONTRACT.md.
const (
	DefaultMinSentences = 2
	DefaultMaxSentences = 4
	DefaultMaxWords     = 15
	// DefaultFloor sits well below the page count under contract today (59 pages under
	// docs/explainers/ at the landing SHA) so ordinary deletion does not trip it, and
	// far above zero so a scan whose root was renamed out from under it cannot pass.
	DefaultFloor = 40
	// ContractDoc is the page that states the contract, and the one page exempt from it.
	ContractDoc = "docs/ONE-BREATH-CONTRACT.md"
)

// DefaultContract is the contract as docs/ONE-BREATH-CONTRACT.md states it, scoped to the
// explainer tree — the pages whose whole job is to be readable by someone with no
// background, and therefore the tree where the block earns its keep first.
func DefaultContract() Contract {
	return Contract{
		Roots:        []string{"docs/explainers"},
		Exempt:       []string{ContractDoc},
		MinSentences: DefaultMinSentences,
		MaxSentences: DefaultMaxSentences,
		MaxWords:     DefaultMaxWords,
		Floor:        DefaultFloor,
	}
}

// Finding is one countable defect: which rule, which page, the line the block starts on
// (1 when there is no block), and a message that says what to do about it.
type Finding struct {
	Kind   Kind   `json:"kind"`
	Path   string `json:"path"`
	Line   int    `json:"line,omitempty"`
	Detail string `json:"detail"`
}

// Key is the ratchet key: kind + path. It is stable under editing — inserting a sentence
// above a finding does not renumber it — which is the property that lets the baseline
// store a COUNT rather than a line number.
func (f Finding) Key() string { return string(f.Kind) + "\t" + f.Path }

// Doc is one page to judge: its repo-relative path and its bytes. Taking bytes (not a
// path to open) is what makes every rule table-testable with no tree and no git.
type Doc struct {
	Path string
	Body []byte
}

var (
	// markerRe matches the block's opening line. Matched EXACTLY, not tolerantly: a page
	// that spells the marker differently is reported as missing the block rather than
	// quietly parsed, because the marker is also read by eye and keyed on by the loader.
	markerRe = regexp.MustCompile(`(?m)^> \*\*In one breath:\*\*[ \t]*`)
	// oneLineSiblingRe matches the precise-version paragraph only when it is the
	// next non-blank content after the block. A `One line` paragraph elsewhere in
	// the page cannot stand behind this particular crude summary.
	oneLineSiblingRe = regexp.MustCompile(`^\s*\*\*One line:\*\*`)
	// sectionRe matches the first `##` section heading; the block must precede it.
	sectionRe = regexp.MustCompile(`(?m)^##+ `)
	// inlineRe strips markdown emphasis and code fences so word counts measure words
	// rather than markup: `**boat**` is one word.
	inlineRe = regexp.MustCompile("[*_`]+")
	// acronymRe matches an ALLCAPS token of two or more letters, with an optional
	// lowercase plural `s` (GPUs). Mixed-case coinages (IoU, KVCache) are prose.
	acronymRe = regexp.MustCompile(`\b([A-Z][A-Z0-9]*[A-Z0-9])s?\b`)
)

// Block is the extracted summary block: its text joined onto one line, the 1-based line
// the marker sits on, and whether it precedes the first section heading.
type Block struct {
	Text              string
	Line              int
	BeforeFirstHead   bool
	HasOneLineSibling bool
}

// Extract pulls the block out of a page body.
//
// A block is the marker line plus every immediately following blockquote line, which is
// how the contract renders it. Joining with a single space is what makes the word and
// sentence counts independent of where the author hard-wrapped, and that independence is
// the point: a rewrap is not a content change and must not move the verdict.
func Extract(body []byte) (Block, bool) {
	loc := markerRe.FindIndex(body)
	if loc == nil {
		return Block{}, false
	}
	b := Block{
		Line:            lineOf(body, loc[0]),
		BeforeFirstHead: true,
	}
	if h := sectionRe.FindIndex(body); h != nil && h[0] < loc[0] {
		b.BeforeFirstHead = false
	}
	rest := string(body[loc[1]:])
	var parts []string
	consumed := 0
	for i, raw := range strings.SplitAfter(rest, "\n") {
		l := strings.TrimSuffix(strings.TrimSuffix(raw, "\n"), "\r")
		if i > 0 {
			if !strings.HasPrefix(l, ">") {
				break
			}
			l = strings.TrimPrefix(l, ">")
		}
		if s := strings.TrimSpace(l); s != "" {
			parts = append(parts, s)
		} else if i > 0 {
			consumed += len(raw)
			break // a bare `>` line ends the block
		}
		consumed += len(raw)
	}
	b.Text = strings.Join(parts, " ")
	b.HasOneLineSibling = oneLineSiblingRe.MatchString(rest[consumed:])
	return b, true
}

// lineOf returns the 1-based line number of a byte offset.
func lineOf(body []byte, off int) int {
	if off > len(body) {
		off = len(body)
	}
	return 1 + strings.Count(string(body[:off]), "\n")
}

// Check judges one page and returns every countable rule it breaks, in rule order so a
// report reads top-down. It never opens a file and never looks at the page body for
// anything but the block and the `**One line:**` sibling — see ScopeNotice.
func (c Contract) Check(d Doc) []Finding {
	for _, e := range c.Exempt {
		if d.Path == e {
			return nil
		}
	}
	blk, ok := Extract(d.Body)
	if !ok {
		return []Finding{{
			Kind: BreathMissing, Path: d.Path, Line: 1,
			Detail: "no `> **In one breath:**` block. It is the only part of the page written for a " +
				"reader — or a model — with no background at all: " + c.sentenceRule() + ". Write it " +
				"above the `**One line:**` paragraph, before the first `##` heading. Contract: " + ContractDoc,
		}}
	}

	var out []Finding
	add := func(k Kind, format string, args ...any) {
		out = append(out, Finding{Kind: k, Path: d.Path, Line: blk.Line, Detail: fmt.Sprintf(format, args...)})
	}

	if !blk.BeforeFirstHead {
		add(BreathMisplaced, "the `In one breath` block sits after the first `##` section heading. "+
			"It belongs in the lede: a summary a reader reaches after three sections is not a summary, "+
			"and a budget-aware loader that serves the lede would not find it. Contract: %s", ContractDoc)
	}
	sents := Sentences(blk.Text)
	if n := len(sents); n < c.MinSentences || n > c.MaxSentences {
		add(BreathSentenceCount, "the `In one breath` block is %d sentence(s); the contract in %s is %d to %d. "+
			"The ceiling is not a style preference: the block's job is that someone with no background comes "+
			"away with the RIGHT idea before reading on, and a block long enough to carry the nuance has "+
			"become a second summary. Move the nuance into `**One line:**`",
			n, ContractDoc, c.MinSentences, c.MaxSentences)
	}
	for i, s := range sents {
		if w := Words(s); w > c.MaxWords {
			add(BreathSentenceLength, "`In one breath` sentence %d is %d words; the contract in %s is %d or "+
				"fewer. Split it, or move the clause to `**One line:**`. The sentence is: %q",
				i+1, w, ContractDoc, c.MaxWords, trunc(s))
		}
	}
	if strings.Contains(blk.Text, "—") {
		add(BreathEmDash, "the `In one breath` block has an em-dash, which %s forbids there. An em-dash is "+
			"how a second idea is appended to a sentence without admitting it is one, so it defeats the "+
			"one-idea-per-sentence rule the word ceiling exists to enforce. Make it a second sentence", ContractDoc)
	}
	if strings.ContainsAny(blk.Text, "()") {
		add(BreathParentheses, "the `In one breath` block has parentheses, which %s forbids there. A "+
			"parenthetical is nuance the reader is asked to hold while finishing the sentence, and this "+
			"block's whole premise is a reader who cannot yet do that. State it as its own sentence", ContractDoc)
	}
	for _, a := range c.UnexpandedAcronyms(blk.Text) {
		add(BreathUnexpandedAcronym, "the `In one breath` block uses %q and never spells it out. %s requires "+
			"an acronym to be expanded on the spot, because the block's reader has no background to supply "+
			"it: write \"%s stands for …\", or use the plain words. If %q is a name and not an acronym, "+
			"declare it with --allow-acronym", a, ContractDoc, a, a)
	}
	if !blk.HasOneLineSibling {
		add(BreathMissingOneLine, "the `In one breath` block has no `**One line:**` paragraph beneath it. The "+
			"pairing is the contract in %s: the block is deliberately cruder than the page, and `**One "+
			"line:**` is the precise version where nuance is allowed. A crude version with no precise "+
			"version standing behind it is where an oversimplification survives, because nothing on the "+
			"page contradicts it", ContractDoc)
	}
	return out
}

func (c Contract) sentenceRule() string {
	return fmt.Sprintf("%d to %d sentences, each %d words or fewer, one idea per sentence",
		c.MinSentences, c.MaxSentences, c.MaxWords)
}

// abbrev are the abbreviations whose internal period must not end a sentence.
//
// Without this the sentence COUNT breaks, and it breaks in the LENIENT direction: "e.g. a
// boat" splits into two sentences, so a block already at the ceiling reads as one over and
// a compliant block reads as one under. Decimals need no entry because Sentences only
// splits on a terminator FOLLOWED by whitespace, and "1.0" has none.
var abbrev = []string{"e.g.", "i.e.", "vs.", "etc.", "cf.", "approx.", "Dr.", "Mr.", "Ms.", "No."}

// Sentences splits a block into sentences on a terminator followed by whitespace or
// end-of-text. A trailing fragment with no terminal punctuation counts as a sentence,
// deliberately: a reader reads it as one, and dropping it would let a block hide an
// over-long clause by omitting its full stop.
func Sentences(text string) []string {
	var out []string
	var cur strings.Builder
	rs := []rune(text)
	for i, r := range rs {
		cur.WriteRune(r)
		if r != '.' && r != '!' && r != '?' {
			continue
		}
		if i+1 < len(rs) && !unicode.IsSpace(rs[i+1]) {
			continue // mid-token: a decimal, or the next dot of an ellipsis
		}
		if endsWithAbbrev(cur.String()) {
			continue
		}
		if s := strings.TrimSpace(cur.String()); s != "" {
			out = append(out, s)
		}
		cur.Reset()
	}
	if s := strings.TrimSpace(cur.String()); s != "" {
		out = append(out, s)
	}
	return out
}

func endsWithAbbrev(s string) bool {
	f := strings.Fields(s)
	if len(f) == 0 {
		return false
	}
	last := inlineRe.ReplaceAllString(f[len(f)-1], "")
	for _, a := range abbrev {
		if strings.EqualFold(last, a) {
			return true
		}
	}
	return false
}

// Words counts the words in one sentence with markdown emphasis removed, so markup is
// never mistaken for content.
func Words(sentence string) int {
	return len(strings.Fields(inlineRe.ReplaceAllString(sentence, "")))
}

// UnexpandedAcronyms returns the ALLCAPS acronyms in a block that the block never spells
// out, in first-appearance order.
//
// "Spelled out on the spot" is made countable this way: an acronym of n letters is
// expanded when the block contains n consecutive words whose initials spell it, and the
// window does not itself contain the acronym. That last clause is what keeps "AI is" from
// expanding "AI". A token immediately followed by a lowercase extension (AGENTS.md) is a
// filename, not an acronym, and is skipped.
//
// The rule is deliberately mechanical and will occasionally be wrong in the strict
// direction — REST is four letters over three words — which is what Contract.AllowAcronyms
// is for. It is never wrong in the LENIENT direction, which is the one that matters for a
// gate whose failure mode is a green tick.
func (c Contract) UnexpandedAcronyms(text string) []string {
	allow := map[string]bool{}
	for _, a := range c.AllowAcronyms {
		allow[strings.ToUpper(strings.TrimSpace(a))] = true
	}
	clean := inlineRe.ReplaceAllString(text, "")
	words := splitWords(clean)
	var out []string
	seen := map[string]bool{}
	for _, loc := range acronymRe.FindAllStringSubmatchIndex(clean, -1) {
		tok := clean[loc[2]:loc[3]]
		if seen[tok] || allow[tok] {
			continue
		}
		// A filename, not an acronym: the match is followed by "." + a lowercase letter.
		if rest := clean[loc[1]:]; len(rest) >= 2 && rest[0] == '.' && rest[1] >= 'a' && rest[1] <= 'z' {
			continue
		}
		seen[tok] = true
		if !expandedIn(words, tok) {
			out = append(out, tok)
		}
	}
	return out
}

// splitWords lowercases and splits on every non-letter, so "Key-Value" contributes two
// words and can expand "KV".
func splitWords(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool { return !unicode.IsLetter(r) })
}

// expandedIn reports whether some window of len(acronym) consecutive words has those
// initials, without the window containing the acronym itself.
func expandedIn(words []string, acronym string) bool {
	letters := []rune(strings.ToLower(acronym))
	n := len(letters)
	if n == 0 || n > len(words) {
		return false
	}
	for i := 0; i+n <= len(words); i++ {
		ok := true
		for j := 0; j < n; j++ {
			w := []rune(words[i+j])
			if len(w) == 0 || w[0] != letters[j] || words[i+j] == strings.ToLower(acronym) {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}

// trunc keeps a quoted sentence from swamping a finding while still quoting enough of it
// to locate.
func trunc(s string) string {
	const max = 90
	if len([]rune(s)) <= max {
		return s
	}
	return string([]rune(s)[:max]) + "…"
}

// Census is the three-numbers-with-a-denominator measurement the contract asks for:
// of Pages pages under contract, how many carry a CONFORMING block, how many carry a
// block that FAILS a countable rule, and how many carry NONE at all.
type Census struct {
	Pages      int `json:"pages"`
	Conforming int `json:"conforming"`
	Failing    int `json:"failing"`
	Missing    int `json:"missing"`
	Exempt     int `json:"exempt"`
	// Notice is ScopeNotice, carried in the serialized census so a machine consumer
	// cannot read the numbers without reading which half was not judged.
	Notice string `json:"scope_notice"`
}

// Scan judges a whole corpus and returns the census plus every finding, sorted by path
// then kind for a stable, diffable report.
//
// The floor check comes first and is reported against the gate, not a page: this package
// reports an ABSENCE, so a scan that examined zero pages prints "clean" and is
// byte-identical to a tree whose every block is perfect.
func (c Contract) Scan(docs []Doc) (Census, []Finding) {
	cen := Census{Notice: ScopeNotice}
	exempt := map[string]bool{}
	for _, e := range c.Exempt {
		exempt[e] = true
	}
	var out []Finding
	for _, d := range docs {
		if exempt[d.Path] {
			cen.Exempt++
			continue
		}
		cen.Pages++
		fs := c.Check(d)
		out = append(out, fs...)
		switch {
		case len(fs) == 0:
			cen.Conforming++
		case fs[0].Kind == BreathMissing:
			cen.Missing++
		default:
			cen.Failing++
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Path != out[j].Path {
			return out[i].Path < out[j].Path
		}
		return out[i].Kind < out[j].Kind
	})
	if cen.Pages < c.Floor {
		out = append([]Finding{{
			Kind: BreathScanFloor, Path: "internal/promptlint/breath", Line: 0,
			Detail: fmt.Sprintf("examined %d page(s) under %s, which is below the floor of %d. This check "+
				"reports an ABSENCE, so a scan that found nothing prints `clean` and is indistinguishable "+
				"from a tree whose every block obeys the contract. Either the roots no longer name the "+
				"directories the pages live in, or the file list was read empty. Fix the scan before "+
				"trusting any verdict from it; lower the bar deliberately with --floor",
				cen.Pages, strings.Join(c.Roots, ", "), c.Floor),
		}}, out...)
	}
	return cen, out
}
