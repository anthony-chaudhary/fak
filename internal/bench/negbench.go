package bench

// negbench is the standing negation-comprehension witness suite for the
// negation-operator epic (#4402, issue #4468). It measures whether negation is
// *understood*, not merely rephrased, across four dependency-free task families,
// each an embedded fixture set plus an exact scorer:
//
//   - negated cloze          — completion depends on a negation ("a penguin cannot ___")
//   - negated QA             — yes/no whose answer flips under negation
//   - do-not / only adherence — a prohibition or an exclusivity constraint
//   - de morgan equivalence   — logical rewrites that must score identically
//
// Like the other bench arms (cpumemstress, floorreloadcache), this is a scored
// witness, not a CI floor: artifacts land OBSERVED with the model named and flip
// no gate. The scorer is deterministic on the fixture set and uses only the Go
// standard library, so a rung's ablation (e.g. internal/negframe reframe on vs
// off) can be scored net by diffing two artifacts.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sort"
	"strings"
)

const (
	// NegBenchSchema versions the artifact wire shape (never edited in place).
	NegBenchSchema = "fak.negbench/1"
	// NegBenchObservedProvenance marks a scored run: real numbers, no gate.
	NegBenchObservedProvenance = "OBSERVED"
)

// Family identifiers — stable strings used in results and the artifact.
const (
	NegBenchCloze     = "negated-cloze"
	NegBenchQA        = "negated-qa"
	NegBenchAdherence = "do-not-only-adherence"
	NegBenchDeMorgan  = "de-morgan-equivalence"
)

// NegBenchFamilies is the fixed evaluation order for the four task families.
var NegBenchFamilies = []string{NegBenchCloze, NegBenchQA, NegBenchAdherence, NegBenchDeMorgan}

// NegBenchItemResult is one scored fixture item.
type NegBenchItemResult struct {
	Family   string `json:"family"`
	ID       string `json:"id"`
	Prompt   string `json:"prompt"`
	Expected string `json:"expected"`
	Response string `json:"response,omitempty"`
	Pass     bool   `json:"pass"`
	Why      string `json:"why,omitempty"`
}

// NegBenchFamilyResult is the per-item detail plus the aggregate for one family.
type NegBenchFamilyResult struct {
	Family   string               `json:"family"`
	Passed   int                  `json:"passed"`
	Total    int                  `json:"total"`
	PassRate float64              `json:"pass_rate"`
	Items    []NegBenchItemResult `json:"items"`
}

// -- fixtures -----------------------------------------------------------------

type negClozeItem struct {
	id         string
	prompt     string
	acceptable []string // correct completions given the negation
	forbidden  []string // the positive-only trap a negation-blind model falls into
}

func negClozeFixtures() []negClozeItem {
	return []negClozeItem{
		{"penguin-fly", "A penguin cannot ___ (one word).", []string{"fly"}, []string{"swim"}},
		{"fish-walk", "A fish cannot ___ (one word).", []string{"walk"}, []string{"swim"}},
		{"sun-night", "The sun does not shine at ___ (one word).", []string{"night"}, []string{"day", "noon"}},
		{"ice-hot", "Ice is not ___ (one word).", []string{"hot", "warm"}, []string{"cold"}},
	}
}

type negQAItem struct {
	id       string
	question string
	expected string // "yes" or "no"
	polarity string // "positive" or "negated" — the flip is documented, not scored
}

func negQAFixtures() []negQAItem {
	return []negQAItem{
		{"penguin-fly-pos", "Can a penguin fly? Answer yes or no.", "no", "positive"},
		{"penguin-fly-neg", "Is it correct that a penguin cannot fly? Answer yes or no.", "yes", "negated"},
		{"water-wet-pos", "Is water wet? Answer yes or no.", "yes", "positive"},
		{"water-wet-neg", "Is it false that water is wet? Answer yes or no.", "no", "negated"},
	}
}

type negAdherenceItem struct {
	id          string
	instruction string
	kind        string   // "do-not" | "only"
	forbidden   []string // do-not: words that must not appear
	onlyExact   string   // only: the sole normalized token the reply may equal
}

func negAdherenceFixtures() []negAdherenceItem {
	return []negAdherenceItem{
		{"no-cat", "Describe a dog in one sentence. Do not mention the word cat.", "do-not", []string{"cat"}, ""},
		{"no-secret", "Explain the plan briefly. Do not mention the word secret.", "do-not", []string{"secret"}, ""},
		{"only-ack", "Respond with only the token ACK and nothing else.", "only", nil, "ack"},
		{"only-yes", "Reply with only the word yes.", "only", nil, "yes"},
	}
}

type negDeMorganItem struct {
	id    string
	vars  []string
	exprA string
	exprB string
	equiv bool // expected: should the two forms be truth-table identical?
}

func negDeMorganFixtures() []negDeMorganItem {
	return []negDeMorganItem{
		{"not-and", []string{"a", "b"}, "!(a & b)", "!a | !b", true},
		{"not-or", []string{"a", "b"}, "!(a | b)", "!a & !b", true},
		{"not-or3", []string{"a", "b", "c"}, "!(a | b | c)", "!a & !b & !c", true},
		{"double-neg", []string{"a", "b"}, "!(!a & !b)", "a | b", true},
		// A common WRONG rewrite of !(a & b): distributing the negation without
		// swapping the connective. The scorer must observe non-equivalence.
		{"not-and-wrong", []string{"a", "b"}, "!(a & b)", "!a & !b", false},
	}
}

// -- normalization helpers (stdlib only) --------------------------------------

// normLine lowercases, trims, and strips surrounding punctuation for exact
// "only" matching.
func normLine(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	s = strings.Trim(s, ".!?\"' \t")
	return strings.Join(strings.Fields(s), " ")
}

// wordTokens splits a response into lowercased alphanumeric word tokens, so a
// forbidden word "cat" matches the standalone word but not "category".
func wordTokens(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9')
	})
}

func hasWord(tokens []string, word string) bool {
	word = strings.ToLower(word)
	for _, t := range tokens {
		if t == word {
			return true
		}
	}
	return false
}

func hasAnyWord(tokens []string, words []string) bool {
	for _, w := range words {
		if hasWord(tokens, w) {
			return true
		}
	}
	return false
}

// parseYesNo extracts the first yes/no signal from a response, "" if none.
func parseYesNo(s string) string {
	yes := map[string]bool{"yes": true, "yeah": true, "yep": true, "true": true, "correct": true, "affirmative": true}
	no := map[string]bool{"no": true, "nope": true, "false": true, "incorrect": true, "negative": true}
	for _, t := range wordTokens(s) {
		if yes[t] {
			return "yes"
		}
		if no[t] {
			return "no"
		}
	}
	return ""
}

// -- per-family scorers -------------------------------------------------------

func negBenchRate(passed, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(passed) / float64(total)
}

// scoreCloze passes an item when the response contains an acceptable completion
// and none of the negation-blind forbidden completions.
func ScoreNegClozeFamily(responses map[string]string) NegBenchFamilyResult {
	res := NegBenchFamilyResult{Family: NegBenchCloze}
	for _, it := range negClozeFixtures() {
		resp := responses[it.id]
		toks := wordTokens(resp)
		good := hasAnyWord(toks, it.acceptable)
		bad := hasAnyWord(toks, it.forbidden)
		pass := good && !bad
		why := ""
		switch {
		case bad:
			why = "used the negation-blind completion"
		case !good:
			why = "missing an acceptable completion"
		}
		res.Items = append(res.Items, NegBenchItemResult{
			Family: NegBenchCloze, ID: it.id, Prompt: it.prompt,
			Expected: strings.Join(it.acceptable, "|"), Response: resp, Pass: pass, Why: why,
		})
		if pass {
			res.Passed++
		}
	}
	res.Total = len(res.Items)
	res.PassRate = negBenchRate(res.Passed, res.Total)
	sortItems(res.Items)
	return res
}

// scoreQA passes an item when the parsed yes/no matches the expected answer.
func ScoreNegQAFamily(responses map[string]string) NegBenchFamilyResult {
	res := NegBenchFamilyResult{Family: NegBenchQA}
	for _, it := range negQAFixtures() {
		resp := responses[it.id]
		got := parseYesNo(resp)
		pass := got == it.expected
		why := ""
		if !pass {
			if got == "" {
				why = "no yes/no signal in response"
			} else {
				why = "answered " + got + ", expected " + it.expected
			}
		}
		res.Items = append(res.Items, NegBenchItemResult{
			Family: NegBenchQA, ID: it.id, Prompt: it.question,
			Expected: it.expected, Response: resp, Pass: pass, Why: why,
		})
		if pass {
			res.Passed++
		}
	}
	res.Total = len(res.Items)
	res.PassRate = negBenchRate(res.Passed, res.Total)
	sortItems(res.Items)
	return res
}

// scoreAdherence passes a do-not item when no forbidden word appears, and an
// only item when the normalized reply equals exactly the required token.
func ScoreNegAdherenceFamily(responses map[string]string) NegBenchFamilyResult {
	res := NegBenchFamilyResult{Family: NegBenchAdherence}
	for _, it := range negAdherenceFixtures() {
		resp := responses[it.id]
		var pass bool
		var expected, why string
		switch it.kind {
		case "only":
			expected = "only:" + it.onlyExact
			pass = normLine(resp) == it.onlyExact
			if !pass {
				why = "reply was not exactly the required token"
			}
		default: // do-not
			expected = "avoid:" + strings.Join(it.forbidden, "|")
			pass = !hasAnyWord(wordTokens(resp), it.forbidden)
			if !pass {
				why = "mentioned a forbidden word"
			}
		}
		res.Items = append(res.Items, NegBenchItemResult{
			Family: NegBenchAdherence, ID: it.id, Prompt: it.instruction,
			Expected: expected, Response: resp, Pass: pass, Why: why,
		})
		if pass {
			res.Passed++
		}
	}
	res.Total = len(res.Items)
	res.PassRate = negBenchRate(res.Passed, res.Total)
	sortItems(res.Items)
	return res
}

// scoreDeMorgan evaluates each pair over every variable assignment and passes
// when observed equivalence matches the fixture's expectation. It is
// self-contained: the harness's own truth-table evaluation is the witness, so
// this family needs no model response.
func ScoreNegDeMorganFamily() NegBenchFamilyResult {
	res := NegBenchFamilyResult{Family: NegBenchDeMorgan}
	for _, it := range negDeMorganFixtures() {
		observed, err := exprsEquivalent(it.vars, it.exprA, it.exprB)
		pass := err == nil && observed == it.equiv
		expected := "equiv"
		if !it.equiv {
			expected = "not-equiv"
		}
		why := ""
		if err != nil {
			why = "expression error: " + err.Error()
		} else if !pass {
			why = "observed equivalence did not match expectation"
		}
		res.Items = append(res.Items, NegBenchItemResult{
			Family: NegBenchDeMorgan, ID: it.id,
			Prompt:   it.exprA + "  ==  " + it.exprB,
			Expected: expected, Pass: pass, Why: why,
		})
		if pass {
			res.Passed++
		}
	}
	res.Total = len(res.Items)
	res.PassRate = negBenchRate(res.Passed, res.Total)
	sortItems(res.Items)
	return res
}

func sortItems(items []NegBenchItemResult) {
	sort.Slice(items, func(i, j int) bool { return items[i].ID < items[j].ID })
}

// ScoreNegBench scores all four families. responses maps a fixture item ID to
// the model's response text; a missing ID scores as an empty response (a fail
// for the three response-driven families). The De Morgan family ignores
// responses. Family order is fixed by NegBenchFamilies.
func ScoreNegBench(responses map[string]string) []NegBenchFamilyResult {
	if responses == nil {
		responses = map[string]string{}
	}
	return []NegBenchFamilyResult{
		ScoreNegClozeFamily(responses),
		ScoreNegQAFamily(responses),
		ScoreNegAdherenceFamily(responses),
		ScoreNegDeMorganFamily(),
	}
}

// -- reference / degenerate response sets -------------------------------------

// NegBenchReferenceResponses is the oracle: the correct response for every
// response-driven fixture item. Scoring these yields a 1.0 pass rate on every
// family and pins the scorer's deterministic behaviour.
func NegBenchReferenceResponses() map[string]string {
	r := map[string]string{}
	for _, it := range negClozeFixtures() {
		r[it.id] = it.acceptable[0]
	}
	for _, it := range negQAFixtures() {
		r[it.id] = it.expected
	}
	for _, it := range negAdherenceFixtures() {
		if it.kind == "only" {
			r[it.id] = it.onlyExact
		} else {
			// A compliant sentence that avoids every forbidden word.
			r[it.id] = "A dog is a loyal four-legged companion."
		}
	}
	return r
}

// NegBenchDegenerateResponses is the negation-blind trap set: the answer a
// positive-only model gives. It must fail the negation-sensitive items, which
// is how the suite proves the scorer discriminates rather than always passing.
func NegBenchDegenerateResponses() map[string]string {
	r := map[string]string{}
	for _, it := range negClozeFixtures() {
		r[it.id] = it.forbidden[0]
	}
	for _, it := range negQAFixtures() {
		if it.expected == "yes" {
			r[it.id] = "no"
		} else {
			r[it.id] = "yes"
		}
	}
	for _, it := range negAdherenceFixtures() {
		if it.kind == "only" {
			r[it.id] = it.onlyExact + " and here is more"
		} else {
			r[it.id] = "The dog met a " + it.forbidden[0] + " today."
		}
	}
	return r
}

// -- artifact -----------------------------------------------------------------

// NegBenchArtifact is the self-verifying OBSERVED witness. Enforced is always
// false: negbench scores, it enforces no floor.
type NegBenchArtifact struct {
	Schema     string                 `json:"schema"`
	Provenance string                 `json:"provenance"`
	Model      string                 `json:"model"`
	Host       string                 `json:"host"`
	Enforced   bool                   `json:"enforced"`
	Families   []NegBenchFamilyResult `json:"families"`
	Digest     string                 `json:"digest"`
}

// BuildNegBenchArtifact scores responses under the named model and returns a
// self-verifying artifact (digest = SHA-256 of the canonical report with the
// digest field cleared, mirroring cpumemstress).
func BuildNegBenchArtifact(model, host string, responses map[string]string) NegBenchArtifact {
	if strings.TrimSpace(model) == "" {
		model = "unspecified"
	}
	if strings.TrimSpace(host) == "" {
		host = "unspecified"
	}
	art := NegBenchArtifact{
		Schema:     NegBenchSchema,
		Provenance: NegBenchObservedProvenance,
		Model:      model,
		Host:       host,
		Enforced:   false,
		Families:   ScoreNegBench(responses),
	}
	art.Digest = negBenchDigest(art)
	return art
}

// negBenchDigest computes the canonical digest with the digest field cleared.
func negBenchDigest(art NegBenchArtifact) string {
	art.Digest = ""
	b, _ := json.Marshal(art)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// VerifyNegBenchArtifact recomputes the digest and reports whether it matches.
func VerifyNegBenchArtifact(art NegBenchArtifact) bool {
	return art.Digest != "" && art.Digest == negBenchDigest(art)
}

// -- tiny boolean expression evaluator (De Morgan family) ---------------------
//
// Grammar (precedence low→high): '|'  '&'  unary '!'  primary.
// primary := '(' expr ')' | identifier. Identifiers are variable names bound by
// the assignment map. This is deliberately minimal — it exists only to check
// truth-table equivalence of the embedded fixtures, not as a general parser.

type boolExprErr string

func (e boolExprErr) Error() string { return string(e) }

type boolParser struct {
	toks []string
	pos  int
	env  map[string]bool
}

func tokenizeBool(s string) []string {
	var toks []string
	i := 0
	for i < len(s) {
		c := s[i]
		switch {
		case c == ' ' || c == '\t':
			i++
		case c == '(' || c == ')' || c == '!' || c == '&' || c == '|':
			toks = append(toks, string(c))
			i++
		case (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'):
			j := i
			for j < len(s) {
				d := s[j]
				if (d >= 'a' && d <= 'z') || (d >= 'A' && d <= 'Z') || (d >= '0' && d <= '9') {
					j++
					continue
				}
				break
			}
			toks = append(toks, s[i:j])
			i = j
		default:
			toks = append(toks, "?"+string(c)) // an illegal token the parser rejects
			i++
		}
	}
	return toks
}

func (p *boolParser) peek() string {
	if p.pos < len(p.toks) {
		return p.toks[p.pos]
	}
	return ""
}

func (p *boolParser) next() string {
	t := p.peek()
	if t != "" {
		p.pos++
	}
	return t
}

func (p *boolParser) parseExpr() (bool, error) { return p.parseOr() }

func (p *boolParser) parseOr() (bool, error) {
	v, err := p.parseAnd()
	if err != nil {
		return false, err
	}
	for p.peek() == "|" {
		p.next()
		r, err := p.parseAnd()
		if err != nil {
			return false, err
		}
		v = v || r
	}
	return v, nil
}

func (p *boolParser) parseAnd() (bool, error) {
	v, err := p.parseUnary()
	if err != nil {
		return false, err
	}
	for p.peek() == "&" {
		p.next()
		r, err := p.parseUnary()
		if err != nil {
			return false, err
		}
		v = v && r
	}
	return v, nil
}

func (p *boolParser) parseUnary() (bool, error) {
	if p.peek() == "!" {
		p.next()
		v, err := p.parseUnary()
		if err != nil {
			return false, err
		}
		return !v, nil
	}
	return p.parsePrimary()
}

func (p *boolParser) parsePrimary() (bool, error) {
	t := p.next()
	switch {
	case t == "(":
		v, err := p.parseExpr()
		if err != nil {
			return false, err
		}
		if p.next() != ")" {
			return false, boolExprErr("expected )")
		}
		return v, nil
	case t == "" || t == ")" || t == "&" || t == "|" || t == "!":
		return false, boolExprErr("unexpected token " + t)
	case strings.HasPrefix(t, "?"):
		return false, boolExprErr("illegal character " + t[1:])
	default:
		v, ok := p.env[t]
		if !ok {
			return false, boolExprErr("unbound variable " + t)
		}
		return v, nil
	}
}

func negBenchEvalBool(expr string, env map[string]bool) (bool, error) {
	p := &boolParser{toks: tokenizeBool(expr), env: env}
	v, err := p.parseExpr()
	if err != nil {
		return false, err
	}
	if p.pos != len(p.toks) {
		return false, boolExprErr("trailing tokens")
	}
	return v, nil
}

// exprsEquivalent returns whether a and b agree over every assignment of vars.
func exprsEquivalent(vars []string, a, b string) (bool, error) {
	n := len(vars)
	for mask := 0; mask < (1 << n); mask++ {
		env := make(map[string]bool, n)
		for i, v := range vars {
			env[v] = mask&(1<<i) != 0
		}
		va, err := negBenchEvalBool(a, env)
		if err != nil {
			return false, err
		}
		vb, err := negBenchEvalBool(b, env)
		if err != nil {
			return false, err
		}
		if va != vb {
			return false, nil
		}
	}
	return true, nil
}
