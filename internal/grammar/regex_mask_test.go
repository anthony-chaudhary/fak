package grammar

import (
	"errors"
	"math"
	"math/rand"
	"regexp"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/model"
)

var _ model.LogitMask = (*RegexMask)(nil)

type regexTestDecoder [][]byte

func (d regexTestDecoder) TokenBytes(id int) []byte {
	if id < 0 || id >= len(d) {
		return nil
	}
	return d[id]
}

// regexOracleAlphabet spans 1-, 2- and 3-byte UTF-8 runes plus the newline that
// Perl `.` excludes. é (C3 A9) and U+4A69 (E4 A9 A9) share their only
// continuation byte, so no recombination of the alphabet's bytes spells a rune
// outside it.
var regexOracleAlphabet = []string{"a", "b", "é", "䩩", "\n"}

const (
	regexOracleEOS   = 0
	regexOracleEmpty = 1
	regexOracleRunes = 6
)

// regexOracleTokens holds every single byte of the alphabet plus tokens that
// cross rune boundaries. The EOS id spells "a" so admission must key on the id.
func regexOracleTokens() regexTestDecoder {
	tokens := regexTestDecoder{[]byte("a"), {}}
	seen := map[byte]bool{}
	for _, r := range regexOracleAlphabet {
		for _, b := range []byte(r) {
			if !seen[b] {
				seen[b] = true
				tokens = append(tokens, []byte{b})
			}
		}
	}
	for _, multi := range []string{"ab", "é", "䩩", "aé", "\xa9b", "\xa9\xa9", "é䩩", "\na", "ba"} {
		tokens = append(tokens, []byte(multi))
	}
	return tokens
}

type regexOracle struct {
	matched map[string]bool
	viable  map[string]bool
}

func newRegexOracle(pattern string, maxRunes int) regexOracle {
	re := regexp.MustCompile(`^(?:` + pattern + `)$`)
	o := regexOracle{matched: map[string]bool{}, viable: map[string]bool{}}
	var walk func(s string, runes int)
	walk = func(s string, runes int) {
		if re.MatchString(s) {
			o.matched[s] = true
			for i := 0; i <= len(s); i++ {
				o.viable[s[:i]] = true
			}
		}
		if runes == maxRunes {
			return
		}
		for _, r := range regexOracleAlphabet {
			walk(s+r, runes+1)
		}
	}
	walk("", 0)
	return o
}

func regexOracleHistories(maxRunes int) []string {
	seen := map[string]bool{}
	var out []string
	var walk func(s string, runes int)
	walk = func(s string, runes int) {
		for i := 0; i <= len(s); i++ {
			if !seen[s[:i]] {
				seen[s[:i]] = true
				out = append(out, s[:i])
			}
		}
		if runes == maxRunes {
			return
		}
		for _, r := range regexOracleAlphabet {
			walk(s+r, runes+1)
		}
	}
	walk("", 0)
	return out
}

func regexByteHistory(t *testing.T, dec regexTestDecoder, text string) []int {
	t.Helper()
	history := make([]int, 0, len(text))
	for _, b := range []byte(text) {
		found := -1
		for id := regexOracleEmpty + 1; id < len(dec); id++ {
			if len(dec[id]) == 1 && dec[id][0] == b {
				found = id
				break
			}
		}
		if found < 0 {
			t.Fatalf("no single-byte token for %#x", b)
		}
		history = append(history, found)
	}
	return history
}

// checkRegexAgainstOracle compares the mask with stdlib full-string matching.
// bound is an upper bound, in runes, on the shortest completion of any viable
// prefix, so prefixes of at most regexOracleRunes-2-bound runes plus one token
// (at most two more runes) always have a witness completion inside the oracle.
// Every atom matches some alphabet rune, so an oracle with no match means the
// language is empty and CompileRegex must refuse it. It reports whether the
// language is non-empty.
func checkRegexAgainstOracle(t *testing.T, pattern string, bound int) bool {
	t.Helper()
	historyRunes := regexOracleRunes - 2 - bound
	if historyRunes < 1 {
		t.Fatalf("pattern %q completion bound %d leaves no history", pattern, bound)
	}
	dec := regexOracleTokens()
	mask, err := CompileRegex(pattern, dec, CompileOptions{EOS: regexOracleEOS})
	oracle := newRegexOracle(pattern, regexOracleRunes)
	if len(oracle.matched) == 0 {
		if !errors.Is(err, ErrUnsupportedSchema) || mask != nil {
			t.Fatalf("CompileRegex(%q) built a mask=%t, err=%v for an empty language; want ErrUnsupportedSchema", pattern, mask != nil, err)
		}
		return false
	}
	if err != nil {
		t.Fatalf("CompileRegex(%q): %v", pattern, err)
	}
	logits := make([]float32, len(dec))
	for _, h := range regexOracleHistories(historyRunes) {
		for id := range logits {
			logits[id] = float32(id + 1)
		}
		mask.MaskLogits(regexByteHistory(t, dec, h), logits)
		if !oracle.viable[h] {
			for id, logit := range logits {
				if !math.IsInf(float64(logit), -1) {
					t.Fatalf("%q after dead prefix %q admitted token %d", pattern, h, id)
				}
			}
			continue
		}
		for id, logit := range logits {
			var want bool
			switch id {
			case regexOracleEOS:
				want = oracle.matched[h]
			case regexOracleEmpty:
				want = false
			default:
				want = oracle.viable[h+string(dec[id])]
			}
			got := !math.IsInf(float64(logit), -1)
			if got != want {
				t.Fatalf("%q after %q: token %d (%q) admitted=%v, oracle=%v", pattern, h, id, dec[id], got, want)
			}
			if got && logit != float32(id+1) {
				t.Fatalf("%q after %q: admitted token %d logit changed to %v", pattern, h, id, logit)
			}
		}
	}
	return true
}

type regexAtom struct {
	text  string
	bound int
}

// regexAnchorAtoms are zero-width, so they never lengthen a completion; every
// other atom matches at least one rune of its alphabet.
var regexAnchorAtoms = []regexAtom{{"^", 0}, {"$", 0}, {`\A`, 0}, {`\z`, 0}, {"(?:)", 0}}

var regexOracleAtoms = append([]regexAtom{
	{"a", 1}, {"b", 1}, {"é", 1}, {`\x{4A69}`, 1}, {`\n`, 1}, {".", 1}, {"(?s:.)", 1},
	{"[ab]", 1}, {"[^a]", 1}, {"[a-é]", 1}, {`[é\x{4A69}]`, 1}, {`\p{Han}`, 1},
	{"(?i:A)", 1}, {"(?i:É)", 1}, {`[^\n]`, 1},
}, regexAnchorAtoms...)

type regexOracleGen struct {
	r     *rand.Rand
	atoms []regexAtom
}

// pattern returns a random pattern over the generator's atoms and an upper
// bound on the runes needed to complete any viable prefix of its language.
func (g regexOracleGen) pattern(depth int) (string, int) {
	if depth == 0 || g.r.Intn(3) == 0 {
		atom := g.atoms[g.r.Intn(len(g.atoms))]
		return atom.text, atom.bound
	}
	x, bx := g.pattern(depth - 1)
	switch g.r.Intn(8) {
	case 0, 1:
		y, by := g.pattern(depth - 1)
		return "(?:" + x + y + ")", bx + by
	case 2:
		y, by := g.pattern(depth - 1)
		return "(?:" + x + "|" + y + ")", max(bx, by)
	case 3:
		return "(?:" + x + "*)", bx
	case 4:
		return "(?:" + x + "+)", bx
	case 5:
		return "(?:" + x + "?)", bx
	case 6:
		return "(?:" + x + "{0,2})", bx
	default:
		return "(?:" + x + "{2})", 2 * bx
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestCompileRegexAgreesWithStdlibOracle(t *testing.T) {
	curated := []struct {
		pattern string
		bound   int
	}{
		{".*", 1},
		{"(?s).*", 1},
		{"[^a]*", 1},
		{`é+\x{4A69}?`, 1},
		{"(?i)A+b", 2},
		{"^a*$", 1},
		{"(?:a|$)b?", 2},
		{`\A(?:ab|é)\z`, 2},
		{"[a-é]{2}", 2},
		{`\p{Han}|\n`, 1},
		{"(?s).{1,2}", 2},
		{".{1,2}", 2},
		{`(?:a|é)*\x{4A69}`, 2},
		{"䩩+", 1},
		{`[^\x00-\x7F]+`, 1},
		{"(?U)a*?b", 2},
		{"(?:)|a", 1},
		{"(?i)É", 1},
		{"(a)(?P<x>b)", 2},
		{"a(?:b$a|é)", 2},
		{"(?:a|b$)*", 1},
		{"$^|a", 1},
		{"(?:^a)*", 1},
		{`a*\z\A`, 1},
		{`(?:a|\z)\A`, 1},
		{`(?:\A|a)\Ab`, 2},
		{`\n$\A`, 1},
		{"a$^", 1},
	}
	for _, tc := range curated {
		t.Run(tc.pattern, func(t *testing.T) {
			checkRegexAgainstOracle(t, tc.pattern, tc.bound)
		})
	}
	gen := regexOracleGen{r: rand.New(rand.NewSource(20261021)), atoms: regexOracleAtoms}
	gen.check(t, 3, 32, 2, checkRegexAgainstOracle)
}

// check runs distinct generated patterns of at most maxBound until wantMatched
// of them have a non-empty language and wantEmpty have an empty one.
func (g regexOracleGen) check(t *testing.T, maxBound, wantMatched, wantEmpty int, check func(*testing.T, string, int) bool) {
	t.Helper()
	seen := map[string]bool{}
	matched, empty := 0, 0
	for tries := 0; matched < wantMatched || empty < wantEmpty; tries++ {
		if tries == 100000 {
			t.Fatalf("generator produced %d matching and %d empty languages in %d tries", matched, empty, tries)
		}
		pattern, bound := g.pattern(3)
		if bound > maxBound || seen[pattern] {
			continue
		}
		seen[pattern] = true
		var nonEmpty bool
		t.Run(pattern, func(t *testing.T) {
			nonEmpty = check(t, pattern, bound)
		})
		if nonEmpty {
			matched++
		} else {
			empty++
		}
	}
}

// regexWordOracle enumerates every word of at most maxRunes alphabet runes.
// Each atom checked against it matches at least one alphabet rune, so any
// completion can be respelled inside the alphabet.
type regexWordOracle struct {
	alphabet []string
	maxRunes int
}

// regexFoldOracle holds two three-member case-fold orbits (k K U+212A and
// s S U+017F), the runes either side of the 2-/3-byte UTF-8 boundary, and a
// 4-byte rune. Its bytes recombine into runes outside it, so it is checked on
// whole runes only.
var regexFoldOracle = regexWordOracle{
	alphabet: []string{"k", "K", "\u212A", "s", "S", "\u017F", "\u07FF", "\u0800", "\U0001F600"},
	maxRunes: 4,
}

var regexFoldAtoms = append([]regexAtom{
	{"k", 1}, {"s", 1}, {`\x{212A}`, 1}, {`\x{17F}`, 1}, {`\x{7FF}`, 1}, {`\x{800}`, 1}, {`\x{1F600}`, 1},
	{"(?i:k)", 1}, {"(?i:S)", 1}, {`(?i:\x{212A})`, 1}, {`(?i:\x{17F})`, 1}, {"(?i:[k-s])", 1},
	{"(?i:[^k])", 1}, {"[^s]", 1}, {".", 1}, {`[\x{7FF}-\x{800}]`, 1}, {`[^\x00-\x{7FF}]`, 1}, {`\p{So}`, 1},
}, regexAnchorAtoms...)

// check compares the automaton with stdlib full-string matching on every word:
// a word ends in an accepting state exactly when stdlib matches it, and a word
// of at most maxRunes-bound runes stays live exactly when it extends to a
// match. An empty language must be refused. It reports whether the language is
// non-empty.
func (o regexWordOracle) check(t *testing.T, pattern string, bound int) bool {
	t.Helper()
	re := regexp.MustCompile(`\A(?:` + pattern + `)\z`)
	matched, viable := map[string]bool{}, map[string]bool{}
	var words []string
	var walk func(w string, runes int)
	walk = func(w string, runes int) {
		words = append(words, w)
		if re.MatchString(w) {
			matched[w] = true
			for i := range w {
				viable[w[:i]] = true
			}
			viable[w] = true
		}
		if runes == o.maxRunes {
			return
		}
		for _, r := range o.alphabet {
			walk(w+r, runes+1)
		}
	}
	walk("", 0)
	mask, err := CompileRegex(pattern, regexByteDecoder(), CompileOptions{EOS: 256})
	if len(matched) == 0 {
		if !errors.Is(err, ErrUnsupportedSchema) || mask != nil {
			t.Fatalf("CompileRegex(%q) built a mask=%t, err=%v for an empty language; want ErrUnsupportedSchema", pattern, mask != nil, err)
		}
		return false
	}
	if err != nil {
		t.Fatalf("CompileRegex(%q): %v", pattern, err)
	}
	for _, w := range words {
		end := mask.step(0, []byte(w))
		if got := end >= 0 && mask.accept[end]; got != matched[w] {
			t.Fatalf("%q on %+q: automaton match=%v, stdlib=%v", pattern, w, got, matched[w])
		}
		if utf8.RuneCountInString(w) <= o.maxRunes-bound {
			if got := end >= 0; got != viable[w] {
				t.Fatalf("%q after %+q: live=%v, stdlib viable=%v", pattern, w, got, viable[w])
			}
		}
	}
	return true
}

// fak-test:runtime fast est=150ms lane=default
func TestCompileRegexFullMatchAgreesWithStdlibAcrossFoldOrbits(t *testing.T) {
	for _, tc := range []struct {
		pattern string
		bound   int
	}{
		{"(?i)k", 1},
		{"(?i)s", 1},
		{"(?i)K+", 1},
		{`(?i)\x{17F}`, 1},
		{`(?i:\x{212A})s`, 2},
		{"(?:(?i:k)*){2}", 2},
		{"(?i)[k-s]", 1},
		{"(?i)[^k]", 1},
		{`[\x{7FF}\x{800}]+`, 1},
		{`\x{7FF}\x{800}`, 2},
		{`[\x{7FF}-\x{800}]`, 1},
		{`[^\x00-\x{7FF}]`, 1},
		{`[\x{800}-\x{10FFFF}]{2}`, 2},
		{`\x{1F600}|\x{17F}`, 1},
		{".", 1},
		{".*", 1},
		{`k*\z\A`, 1},
		{`(?:k|\z)\A`, 1},
		{"k$^", 1},
		{`\x{1F600}$\A`, 1},
		{"^(?i)k*$", 1},
		{"(?:^s|k)+", 1},
	} {
		t.Run(tc.pattern, func(t *testing.T) {
			regexFoldOracle.check(t, tc.pattern, tc.bound)
		})
	}
	gen := regexOracleGen{r: rand.New(rand.NewSource(20261022)), atoms: regexFoldAtoms}
	gen.check(t, 2, 32, 2, regexFoldOracle.check)
}

// fak-test:runtime fast est=60ms lane=default
func TestCompileRegexAnchorOrdersAgreeWithStdlib(t *testing.T) {
	oracle := regexWordOracle{alphabet: []string{"k", "\u212A", "s"}, maxRunes: 4}
	items := []regexAtom{{"k", 1}, {"k*", 1}, {`(?:s|\z)`, 1}, {"^", 0}, {"$", 0}, {`\A`, 0}, {`\z`, 0}}
	var walk func(pattern string, bound, depth int)
	walk = func(pattern string, bound, depth int) {
		if depth > 0 {
			oracle.check(t, pattern, bound)
		}
		if depth == 3 {
			return
		}
		for _, item := range items {
			walk(pattern+item.text, bound+item.bound, depth+1)
		}
	}
	walk("", 0, 0)
}

// fak-test:runtime fast est=5ms lane=default
func TestRegexMaskSplitsUTF8AcrossTokens(t *testing.T) {
	dec := regexTestDecoder{[]byte("<eos>"), {0xC3}, {0xA9}, []byte("é"), []byte("e"), {0xC3, 0xA8}, {0xFF}}
	const eos, lead, cont, whole, ascii, other, invalid = 0, 1, 2, 3, 4, 5, 6
	mask, err := CompileRegex("é+", dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatal(err)
	}
	admitted := func(history []int) []int {
		logits := make([]float32, len(dec))
		mask.MaskLogits(history, logits)
		var ids []int
		for id, logit := range logits {
			if !math.IsInf(float64(logit), -1) {
				ids = append(ids, id)
			}
		}
		return ids
	}
	for _, tc := range []struct {
		name    string
		history []int
		want    []int
	}{
		{"start", nil, []int{lead, whole}},
		{"after_lead_byte", []int{lead}, []int{cont}},
		{"after_split_rune", []int{lead, cont}, []int{eos, lead, whole}},
		{"after_whole_rune", []int{whole}, []int{eos, lead, whole}},
		{"after_wrong_rune", []int{other}, nil},
		{"after_invalid_byte", []int{invalid}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := admitted(tc.history)
			if len(got) != len(tc.want) {
				t.Fatalf("admitted %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("admitted %v, want %v", got, tc.want)
				}
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestRegexMaskAdmitsExactlyValidUTF8(t *testing.T) {
	mask, err := CompileRegex("(?s).", regexByteDecoder(), CompileOptions{EOS: 256})
	if err != nil {
		t.Fatal(err)
	}
	viable := [3][]bool{make([]bool, 1<<8), make([]bool, 1<<16), make([]bool, 1<<24)}
	var buf [utf8.UTFMax]byte
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if !utf8.ValidRune(r) {
			continue
		}
		n := utf8.EncodeRune(buf[:], r)
		if end := mask.step(0, buf[:n]); end < 0 || !mask.accept[end] {
			t.Fatalf("valid rune %U (% x) is not accepted", r, buf[:n])
		}
		key := uint32(0)
		for i := 0; i < n-1; i++ {
			key = key<<8 | uint32(buf[i])
			viable[i][key] = true
		}
	}
	for b0 := range 256 {
		if got := mask.step(0, []byte{byte(b0)}); (got >= 0) != (viable[0][uint32(b0)] || b0 < utf8.RuneSelf) {
			t.Fatalf("lead byte %#x admitted=%v", b0, got >= 0)
		}
		if b0 < 0xC0 {
			continue
		}
		for b1 := range 256 {
			prefix := []byte{byte(b0), byte(b1)}
			key := uint32(b0)<<8 | uint32(b1)
			want := viable[1][key] || (b0 < 0xE0 && utf8.Valid(prefix))
			if got := mask.step(0, prefix); (got >= 0) != want {
				t.Fatalf("prefix % x admitted=%v, want %v", prefix, got >= 0, want)
			}
			if b0 < 0xE0 || !viable[1][key] {
				continue
			}
			for b2 := range 256 {
				prefix := []byte{byte(b0), byte(b1), byte(b2)}
				want := viable[2][key<<8|uint32(b2)] || (b0 < 0xF0 && utf8.Valid(prefix))
				if got := mask.step(0, prefix); (got >= 0) != want {
					t.Fatalf("prefix % x admitted=%v, want %v", prefix, got >= 0, want)
				}
			}
		}
	}
}

func regexByteDecoder() regexTestDecoder {
	dec := make(regexTestDecoder, 257)
	for b := range 256 {
		dec[b] = []byte{byte(b)}
	}
	dec[256] = []byte("<eos>")
	return dec
}

// fak-test:runtime fast est=60ms lane=default
func TestRegexMaskEveryReachableStateCanAccept(t *testing.T) {
	dec := regexByteDecoder()
	const eos = 256
	for _, pattern := range []string{
		"[A-Z]{2}-[0-9]{3}",
		`\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z`,
		`[a-z0-9._]+@[a-z0-9]+\.[a-z]{2,4}`,
		`[^"\n]{0,64}`,
		"(?:a|$)b?",
		"é{1,3}|中+",
		`\p{Greek}+`,
		"ab",
		"(?s).*",
	} {
		t.Run(pattern, func(t *testing.T) {
			mask, err := CompileRegex(pattern, dec, CompileOptions{EOS: eos})
			if err != nil {
				t.Fatal(err)
			}
			paths := map[int32][]int{0: nil}
			queue := []int32{0}
			for len(queue) > 0 {
				state := queue[0]
				queue = queue[1:]
				if !regexStateCanAccept(mask, state) {
					t.Fatalf("reachable state %d cannot reach an accepting state", state)
				}
				logits := make([]float32, len(dec))
				mask.MaskLogits(paths[state], logits)
				if got := !math.IsInf(float64(logits[eos]), -1); got != mask.accept[state] {
					t.Fatalf("state %d: EOS admitted=%v, accepting=%v", state, got, mask.accept[state])
				}
				outgoing := false
				for b := range 256 {
					next := mask.trans[int(state)*mask.numClasses+int(mask.classOf[b])]
					if admitted := !math.IsInf(float64(logits[b]), -1); admitted != (next >= 0) {
						t.Fatalf("state %d byte %#x: admitted=%v, transition=%d", state, b, admitted, next)
					}
					if next < 0 {
						continue
					}
					outgoing = true
					if _, seen := paths[next]; !seen {
						paths[next] = append(append([]int(nil), paths[state]...), b)
						queue = append(queue, next)
					}
				}
				if !outgoing && !mask.accept[state] {
					t.Fatalf("state %d is a non-accepting dead end", state)
				}
			}
			if len(paths) > maxRegexDFAStates {
				t.Fatalf("%d reachable states exceed bound %d", len(paths), maxRegexDFAStates)
			}
		})
	}
}

func regexStateCanAccept(m *RegexMask, from int32) bool {
	seen := map[int32]bool{from: true}
	queue := []int32{from}
	for len(queue) > 0 {
		state := queue[0]
		queue = queue[1:]
		if m.accept[state] {
			return true
		}
		for c := range m.numClasses {
			next := m.trans[int(state)*m.numClasses+c]
			if next >= 0 && !seen[next] {
				seen[next] = true
				queue = append(queue, next)
			}
		}
	}
	return false
}

// fak-test:runtime fast est=5ms lane=default
func TestRegexMaskTerminalAcceptAdmitsOnlyEOS(t *testing.T) {
	dec := regexByteDecoder()
	const eos = 256
	mask, err := CompileRegex("ab", dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, len(dec))
	mask.MaskLogits([]int{'a', 'b'}, logits)
	for id, logit := range logits {
		if finite := !math.IsInf(float64(logit), -1); finite != (id == eos) {
			t.Fatalf("token %d finite=%v after a complete match", id, finite)
		}
	}
	mask.MaskLogits([]int{'a', 'b', 'a'}, logits)
	for id, logit := range logits {
		if !math.IsInf(float64(logit), -1) {
			t.Fatalf("token %d admitted after a history past the match", id)
		}
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestCompileRegexRefusesOutsideBoundedSurface(t *testing.T) {
	dec := regexByteDecoder()
	for _, tc := range []struct{ name, pattern string }{
		{"word_boundary", `\bab`},
		{"not_word_boundary", `a\B`},
		{"line_anchors", `(?m)^a$`},
		{"backreference", `(a)\1`},
		{"lookahead", `a(?=b)`},
		{"lookbehind", `(?<=a)b`},
		{"parse_error", `(`},
		{"empty_class", `[^\x00-\x{10FFFF}]`},
		{"begin_after_byte", `a^b`},
		{"end_before_byte", `a$b`},
		{"begin_after_end", `a$^`},
		{"begin_text_after_end", `\n$\A`},
		{"dfa_state_cap", `(a|b)*a(a|b){16}`},
		{"nested_repeat_count", `(?:(?:a{100}){100}){100}`},
		{"expanded_size", `(?:abcdefghij){1000}`},
		{"byte_automaton_size", `[\p{L}\p{N}]{1,32}`},
		{"pattern_bytes", strings.Repeat("a", maxRegexPatternBytes+1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mask, err := CompileRegex(tc.pattern, dec, CompileOptions{EOS: 256})
			if !errors.Is(err, ErrUnsupportedSchema) || mask != nil {
				t.Fatalf("CompileRegex(%q) built a mask=%t, err=%v; want ErrUnsupportedSchema", tc.pattern, mask != nil, err)
			}
		})
	}
	if _, err := CompileRegex("a", nil, CompileOptions{EOS: 0}); err == nil {
		t.Fatal("CompileRegex without a TokenDecoder succeeded")
	}
}

// fak-test:runtime fast est=40ms lane=default
func TestRegexMaskCacheStaysBounded(t *testing.T) {
	dec := regexByteDecoder()
	const eos = 256
	mask, err := CompileRegex("[ab]{0,150}", dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, len(dec))
	var history []int
	for step := range 150 {
		mask.MaskLogits(history, logits)
		if math.IsInf(float64(logits['a']), -1) || math.IsInf(float64(logits[eos]), -1) || !math.IsInf(float64(logits['c']), -1) {
			t.Fatalf("step %d: a=%v eos=%v c=%v", step, logits['a'], logits[eos], logits['c'])
		}
		mask.cacheMu.Lock()
		cached := len(mask.cache)
		mask.cacheMu.Unlock()
		if cached > maxRegexMaskCacheStates {
			t.Fatalf("step %d: %d cached states exceed bound %d", step, cached, maxRegexMaskCacheStates)
		}
		history = append(history, 'a'+step%2)
	}
	mask.MaskLogits(history, logits)
	if math.IsInf(float64(logits[eos]), -1) || !math.IsInf(float64(logits['a']), -1) {
		t.Fatalf("after the repeat bound: eos=%v a=%v", logits[eos], logits['a'])
	}
}
