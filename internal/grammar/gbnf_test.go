package grammar

import (
	"errors"
	"math"
	"slices"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"
)

type gbnfTestDecoder [][]byte

func (d gbnfTestDecoder) TokenBytes(id int) []byte {
	if id < 0 || id >= len(d) {
		return nil
	}
	return d[id]
}
func gbnfByteDecoder() gbnfTestDecoder {
	d := make(gbnfTestDecoder, 258)
	d[0] = []byte("not EOS text") // admission must use ID, not these bytes
	for b := 0; b < 256; b++ {
		d[b+2] = []byte{byte(b)}
	}
	return d
}
func gbnfCompileTest(t *testing.T, source string, dec TokenDecoder) *GBNFMask {
	t.Helper()
	m, err := CompileGBNF(source, dec, CompileOptions{EOS: 0})
	if err != nil {
		t.Fatal(err)
	}
	return m
}
func gbnfTestLogits(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i % 5)
	}
	return out
}
func gbnfAllowed(logit float32) bool { return !math.IsInf(float64(logit), -1) }
func gbnfTestHistory(s string) []int {
	out := make([]int, len(s))
	for i := 0; i < len(s); i++ {
		out[i] = int(s[i]) + 2
	}
	return out
}

// fak-test:runtime fast est=1s lane=default
func TestGBNFFiniteLanguagePrefixOracle(t *testing.T) {
	// This oracle enumerates whole finite languages independently of the parser
	// and checks every byte-prefix mask, EOS and every byte ID, not just output.
	for _, tc := range []struct {
		source string
		words  []string
	}{
		{`root ::= "a" | "bc"`, []string{"a", "bc"}},
		{`root ::= ("a" | "b") ("c" | "")`, []string{"a", "ac", "b", "bc"}},
		{`root ::= "x"{1,3}`, []string{"x", "xx", "xxx"}},
		{`root ::= "x"{0,2}`, []string{"", "x", "xx"}},
		{`root ::= a? b
a ::= "a"
b ::= "b"`, []string{"b", "ab"}},
		{`root ::= [a-c] [x-z]`, []string{"ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz"}},
		{`root ::= "é" | "日本" | "\u03BB"`, []string{"é", "日本", "λ"}},
		{`# comments, grouped newlines and newline after alternative
root ::= ("a"
 | "b") | # comment
 "c"`, []string{"a", "b", "c"}},
	} {
		m := gbnfCompileTest(t, tc.source, gbnfByteDecoder())
		prefixes := map[string]bool{}
		for _, word := range tc.words {
			for i := 0; i <= len(word); i++ {
				prefixes[word[:i]] = true
			}
		}
		for prefix := range prefixes {
			logits := gbnfTestLogits(258)
			if err := m.MaskLogitsChecked(gbnfTestHistory(prefix), logits); err != nil {
				t.Fatalf("prefix %x: %v", prefix, err)
			}
			if got, want := gbnfAllowed(logits[0]), slices.Contains(tc.words, prefix); got != want {
				t.Fatalf("EOS %x=%v want %v", prefix, got, want)
			}
			if gbnfAllowed(logits[1]) {
				t.Fatal("empty token admitted")
			}
			for b := 0; b < 256; b++ {
				candidate := prefix + string([]byte{byte(b)})
				want := false
				for _, word := range tc.words {
					want = want || strings.HasPrefix(word, candidate)
				}
				if gbnfAllowed(logits[b+2]) != want {
					t.Fatalf("prefix %x byte %x: want %v", prefix, b, want)
				}
			}
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestGBNFRecursiveReferencesAndRepetition(t *testing.T) {
	for _, tc := range []struct {
		source  string
		yes, no []string
	}{
		{`root ::= "(" root ")" root | ""`, []string{"", "()", "()()", "(())", "(()())"}, []string{")", "(()", "())", "(x)"}},
		{`root ::= "a" root "b" | ""`, []string{"", "ab", "aabb", "aaabbb"}, []string{"aab", "abb", "abab"}},
		{`root ::= "a"+ "b"* "c"{2,}`, []string{"acc", "aabbcccc", "abcc"}, []string{"cc", "ac", "acccb"}},
		{`root ::= [^a-c] .`, []string{"zé", "日本", "\nλ"}, []string{"a!", "b?", "c.", "z"}},
		{`root ::= "\x41\u03bb\U0001F642"`, []string{"Aλ🙂"}, []string{"Aλ", "Aλx"}},
	} {
		m := gbnfCompileTest(t, tc.source, gbnfByteDecoder())
		for _, set := range []struct {
			words []string
			want  bool
		}{{tc.yes, true}, {tc.no, false}} {
			for _, word := range set.words {
				s, err := m.program.feed(m.initial, []byte(word), &gbnfWork{left: gbnfMaxWork})
				if err != nil {
					t.Fatal(err)
				}
				if got := s.accepts(); got != set.want {
					t.Fatalf("grammar %q word %q accepts=%v", tc.source, word, got)
				}
			}
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestGBNFParseRefusalsAreBoundedAndRedacted(t *testing.T) {
	cases := []string{
		"", "root ::= \"secret unfinished", "root ::= missing",
		"root ::= \"a\"\nroot ::= \"b\"", "other ::= \"a\"",
		"root ::= root \"a\" | \"\"", "root ::= a\na ::= root | \"x\"",
		"root ::= a root | \"x\"\na ::= \"\"", "root ::= (\"\")*",
		"root ::= a\na ::= a", "root ::= [z-a]", "root ::= []",
		"root ::= [^\\x00-\\U0010FFFF]", "root ::= \"\\uD800\"",
		"root ::= \"\\U00110000\"", "root ::= \"\\q\"", "root ::= \"\\x0\"",
		"root ::= \"a\"{2,1}", "root ::= \"a\"{257}", "root ::= \"a\"**",
		"root ::= <[17]>", "root ::= <token>", "root ::= !<[17]>", "root ::= \"a\" ;",
		"root ::= \"a\"\x00", "root ::= \"\xff\"",
		strings.Repeat("#", gbnfMaxSourceBytes+1),
		"root ::= " + strings.Repeat("(", gbnfMaxNesting+1) + "\"a\"" + strings.Repeat(")", gbnfMaxNesting+1),
	}
	for i, source := range cases {
		if _, err := CompileGBNF(source, gbnfByteDecoder(), CompileOptions{EOS: 0}); !errors.Is(err, ErrGBNF) {
			t.Fatalf("case %d: %v", i, err)
		} else if strings.Contains(err.Error(), "secret") {
			t.Fatalf("untrusted source leaked: %v", err)
		}
	}
	if _, err := CompileGBNF(`root ::= "a"`, nil, CompileOptions{EOS: 0}); !errors.Is(err, ErrGBNF) {
		t.Fatal(err)
	}
	if _, err := CompileGBNF(`root ::= "a"`, gbnfByteDecoder(), CompileOptions{EOS: -1}); !errors.Is(err, ErrGBNF) {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestGBNFUTF8PartialTokenAndEOS(t *testing.T) {
	dec := gbnfTestDecoder{
		[]byte("a"), nil, []byte{0xE6}, []byte{0x97}, []byte{0xA5}, []byte("日"),
		[]byte{0x80}, []byte{0xC0, 0xAF}, []byte{0xED, 0xA0, 0x80}, []byte{0xF4, 0x90, 0x80, 0x80},
		[]byte{0xE6, 0x97}, []byte{0x97, 0xA5}, []byte("日x"), []byte{0xE6, 0x98},
	}
	m := gbnfCompileTest(t, `root ::= "日"`, dec)
	for _, tc := range []struct{ history, allow []int }{
		{nil, []int{2, 5, 10}},
		{[]int{2}, []int{3, 11}},
		{[]int{2, 3}, []int{4}},
		{[]int{10}, []int{4}},
		{[]int{5}, []int{0}},
		{[]int{2, 11}, []int{0}},
	} {
		logits := gbnfTestLogits(len(dec))
		if err := m.MaskLogitsChecked(tc.history, logits); err != nil {
			t.Fatal(err)
		}
		for id, v := range logits {
			if gbnfAllowed(v) != slices.Contains(tc.allow, id) {
				t.Fatalf("history %v token %d allowed=%v", tc.history, id, gbnfAllowed(v))
			}
		}
	}
	for _, history := range [][]int{{0}, {1}, {6}, {7}, {8}, {9}, {13}, {999}} {
		logits := gbnfTestLogits(len(dec))
		if err := m.MaskLogitsChecked(history, logits); !errors.Is(err, ErrGBNF) {
			t.Fatalf("history %v: %v", history, err)
		}
		for _, v := range logits {
			if gbnfAllowed(v) {
				t.Fatal("invalid history left logits unmasked")
			}
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestGBNFUTF8PrefixInterval(t *testing.T) {
	// Compare every prefix of representative scalar-boundary encodings;
	// invalid prefixes never
	// create a scalar interval that includes an overlong/surrogate value.
	for _, r := range []rune{0, 0x7F, 0x80, 0x7FF, 0x800, 0xD7FF, 0xE000, 0xFFFF, 0x10000, 0x10FFFF} {
		piece := []byte(string(r))
		for n := 1; n <= len(piece); n++ {
			lo, hi, complete, ok := gbnfUTF8Prefix(piece[:n])
			if !ok || r < lo || r > hi || complete != (n == len(piece)) {
				t.Fatalf("rune %x prefix %x interval %x..%x", r, piece[:n], lo, hi)
			}
			if n == len(piece) && (lo != r || hi != r) {
				t.Fatal("complete interval not exact")
			}
		}
	}
	for _, piece := range [][]byte{{0xC0}, {0xC1}, {0x80}, {0xF5}, {0xE0, 0x80}, {0xED, 0xA0}, {0xF0, 0x80}, {0xF4, 0x90}, {0xE6, 0x41}} {
		if _, _, _, ok := gbnfUTF8Prefix(piece); ok {
			t.Fatalf("invalid prefix %x", piece)
		}
	}
	m := gbnfCompileTest(t, `root ::= [^\u0080-\u07FF]`, gbnfByteDecoder())
	logits := gbnfTestLogits(258)
	if err := m.MaskLogitsChecked(nil, logits); err != nil {
		t.Fatal(err)
	}
	for b := 0xC2; b <= 0xDF; b++ {
		if gbnfAllowed(logits[b+2]) {
			t.Fatalf("negation admitted excluded partial prefix %x", b)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestGBNFSharedMaskRollbackAndConcurrentHistories(t *testing.T) {
	m := gbnfCompileTest(t, `root ::= "ab" | "cd"`, gbnfByteDecoder())
	check := func(history string, want string) error {
		logits := gbnfTestLogits(258)
		if err := m.MaskLogitsChecked(gbnfTestHistory(history), logits); err != nil {
			return err
		}
		for b := 0; b < 256; b++ {
			if gbnfAllowed(logits[b+2]) != strings.ContainsRune(want, rune(b)) {
				return errors.New("branch mask mismatch")
			}
		}
		return nil
	}
	for _, tc := range []struct{ h, w string }{{"a", "b"}, {"c", "d"}, {"", "ac"}, {"a", "b"}, {"ab", ""}} {
		if err := check(tc.h, tc.w); err != nil {
			t.Fatal(err)
		}
	}
	var wg sync.WaitGroup
	errs := make(chan error, 4)
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				h, w := "a", "b"
				if (i+j)%2 == 0 {
					h, w = "c", "d"
				}
				if err := check(h, w); err != nil {
					errs <- err
					return
				}
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestGBNFResourceExhaustionFailsClosed(t *testing.T) {
	// 256 indistinguishable first bytes lead to 512 distinct remaining stacks.
	// The runtime refuses ambiguity, rather than dropping valid alternatives
	// and pretending the smaller language is the requested grammar.
	alternatives := make([]string, 256)
	for i := range alternatives {
		alternatives[i] = `"a" ("b" | "c")`
	}
	m := gbnfCompileTest(t, "root ::= "+strings.Join(alternatives, " | "), gbnfByteDecoder())
	logits := gbnfTestLogits(258)
	if err := m.MaskLogitsChecked(nil, logits); !errors.Is(err, ErrGBNF) {
		t.Fatalf("ambiguity: %v", err)
	}
	for _, v := range logits {
		if gbnfAllowed(v) {
			t.Fatal("resource refusal left an allowed token")
		}
	}
	m = gbnfCompileTest(t, `root ::= "a"*`, gbnfByteDecoder())
	logits = gbnfTestLogits(258)
	if err := m.MaskLogitsChecked(make([]int, gbnfMaxHistoryTokens+1), logits); !errors.Is(err, ErrGBNF) {
		t.Fatal(err)
	}
	large := gbnfTestDecoder{nil, []byte(strings.Repeat("a", gbnfMaxTokenBytes+1))}
	m = gbnfCompileTest(t, `root ::= "a"*`, large)
	logits = gbnfTestLogits(2)
	m.MaskLogits(nil, logits)
	if gbnfAllowed(logits[0]) || gbnfAllowed(logits[1]) {
		t.Fatal("generic interface failed open")
	}
	m = gbnfCompileTest(t, `root ::= "a"`, gbnfByteDecoder())
	logits = gbnfTestLogits(258)
	logits['a'+2] = float32(math.Inf(-1))
	if err := m.MaskLogitsChecked(nil, logits); err != nil {
		t.Fatal(err)
	}
	if gbnfAllowed(logits['a'+2]) {
		t.Fatal("grammar unmasked an existing minus infinity")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestGBNFWholeTokenMatchesByteSegmentation(t *testing.T) {
	words := []string{"", "ab", "aé", "日本"}
	dec := gbnfByteDecoder()
	dec = append(dec, []byte("ab"), []byte("aé"), []byte("日本"), []byte("a"), []byte("é"))
	m := gbnfCompileTest(t, `root ::= "ab" | "aé" | "日本" | ""`, dec)
	for _, word := range words {
		history := gbnfTestHistory(word)
		byteMask := gbnfTestLogits(len(dec))
		if err := m.MaskLogitsChecked(history, byteMask); err != nil {
			t.Fatal(err)
		}
		if word == "" {
			continue
		}
		id := -1
		for i, piece := range dec {
			if string(piece) == word {
				id = i
				break
			}
		}
		if id < 0 {
			t.Fatal("missing fixture token")
		}
		wholeMask := gbnfTestLogits(len(dec))
		if err := m.MaskLogitsChecked([]int{id}, wholeMask); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(byteMask, wholeMask) {
			t.Fatalf("segmentation changed mask for %q", word)
		}
		if !utf8.ValidString(word) {
			t.Fatal("fixture encoding")
		}
	}
}
