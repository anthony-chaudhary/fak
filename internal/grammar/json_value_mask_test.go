package grammar

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"
)

const jsonValueTestEOS = 0

// jsonValueTestVocab returns EOS (empty) at id 0, an empty non-EOS token at id
// 1, pieces that cross structural, escape and UTF-8 boundaries, every single
// byte from the returned base, then random JSON-shaped pieces.
func jsonValueTestVocab(rng *rand.Rand, random int) (jsonSchemaTokenDecoder, int) {
	fixed := []string{
		"",
		`{"`, `":`, `":"`, `","`, `"}`, `"]`, `}}`, `]}`, `}]`, `}]}`, `[{`, `[[`, `[]`, `{}`, `{},`, `[],`,
		`\u00`, `\"`, `\\`, `\n`, `\u12aB`, `\uZ`, `\x`, `\/`,
		"é", "€", "😀", "\xc3", "\xa9", "\xe2\x82", "\xed\xa0\x80", "\xed\x9f\xbf", "\xf4\x90", "\xf4\x8f\xbf\xbf",
		"\xe0\x80", "\xe0\xa0\x80", "\xf0\x80", "\xf0\x90\x80\x80", "\xc0\x80", "\xc2\x80", "\xff", "a\xc3",
		` "`, `, `, `: `, "\n  ", "\n    \"", "  }", "\t]", "\r\n", "  ", " \t\n\r ",
		`1e`, `-0.`, `.5`, `e+1`, `E-`, `00`, `01`, `-1`, `0,`, `10]`, `1.5e3}`, `0e0`,
		`true`, `tru`, `e,`, `nul`, `l}`, `fa`, `lse`, `null,`, `false]`,
		`"a":`, `"a":1}`, `{"a":[1,"x",true,null]}`, `}x`, `} `, "}\n", "\x00", "\x1f", "\x7f",
	}
	dec := jsonSchemaTokenDecoder{nil}
	for _, s := range fixed {
		dec = append(dec, []byte(s))
	}
	base := len(dec)
	for b := range 256 {
		dec = append(dec, []byte{byte(b)})
	}
	alphabet := []string{"{", "}", "[", "]", `"`, ":", ",", ".", "-", "+", "0", "1", "9", "e", "E", "t", "r", "u", "f", "a", "l", "s", "n", `\`, "/", " ", "\n", "\t", "é", "\xe2"}
	for range random {
		var piece strings.Builder
		for range 1 + rng.Intn(6) {
			piece.WriteString(alphabet[rng.Intn(len(alphabet))])
		}
		dec = append(dec, []byte(piece.String()))
	}
	return dec, base
}

var jsonValueProperUTF8Prefixes = func() map[string]bool {
	out := map[string]bool{}
	var buf [utf8.UTFMax]byte
	for r := rune(0x80); r <= utf8.MaxRune; r++ {
		if !utf8.ValidRune(r) {
			continue
		}
		n := utf8.EncodeRune(buf[:], r)
		for k := 1; k < n; k++ {
			out[string(buf[:k])] = true
		}
	}
	return out
}()

// jsonValueUTF8Viable reports whether p is valid UTF-8 up to a final rune that
// some continuation bytes can still complete.
func jsonValueUTF8Viable(p []byte) bool {
	if utf8.Valid(p) {
		return true
	}
	for k := 1; k < utf8.UTFMax && k <= len(p); k++ {
		if start := len(p) - k; utf8.RuneStart(p[start]) {
			return utf8.Valid(p[:start]) && jsonValueProperUTF8Prefixes[string(p[start:])]
		}
	}
	return false
}

// jsonValueOracleViable is the independent prefix oracle: encoding/json's
// streaming decoder reports io.ErrUnexpectedEOF exactly when more input could
// still complete a value. It does not apply the depth or whitespace caps, so
// callers keep p within maxJSONValueDepth bytes.
func jsonValueOracleViable(p []byte) (viable, complete bool) {
	if len(p) == 0 {
		return true, false
	}
	if p[0] != '{' || !jsonValueUTF8Viable(p) {
		return false, false
	}
	dec := json.NewDecoder(bytes.NewReader(p))
	var value json.RawMessage
	err := dec.Decode(&value)
	switch {
	case err == nil:
		done := dec.InputOffset() == int64(len(p))
		return done, done
	case errors.Is(err, io.ErrUnexpectedEOF):
		return true, false
	}
	return false, false
}

func jsonValueOracleDepth(doc []byte) int {
	dec := json.NewDecoder(bytes.NewReader(doc))
	depth, deepest := 0, 0
	for {
		token, err := dec.Token()
		if err != nil {
			return deepest
		}
		switch token {
		case json.Delim('{'), json.Delim('['):
			depth++
			deepest = max(deepest, depth)
		case json.Delim('}'), json.Delim(']'):
			depth--
		}
	}
}

// jsonValueOracleWhitespaceRun returns the longest whitespace run outside
// strings: deleting such a run leaves the compacted document unchanged, while
// deleting a run inside a string changes that string.
func jsonValueOracleWhitespaceRun(doc []byte) int {
	space := func(b byte) bool { return b == ' ' || b == '\t' || b == '\n' || b == '\r' }
	var want bytes.Buffer
	if json.Compact(&want, doc) != nil {
		return math.MaxInt
	}
	longest := 0
	for i := 0; i < len(doc); {
		if !space(doc[i]) {
			i++
			continue
		}
		j := i
		for j < len(doc) && space(doc[j]) {
			j++
		}
		var got bytes.Buffer
		trimmed := append(append([]byte(nil), doc[:i]...), doc[j:]...)
		if json.Compact(&got, trimmed) == nil && bytes.Equal(got.Bytes(), want.Bytes()) {
			longest = max(longest, j-i)
		}
		i = j
	}
	return longest
}

// jsonValueOracleDocument accepts exactly one JSON object with no surrounding
// whitespace, valid UTF-8, at most maxJSONValueDepth nested containers and no
// whitespace run outside strings longer than maxJSONValueWhitespaceRun.
func jsonValueOracleDocument(doc []byte) bool {
	if len(doc) == 0 || doc[0] != '{' || doc[len(doc)-1] != '}' || !utf8.Valid(doc) || !json.Valid(doc) {
		return false
	}
	return jsonValueOracleDepth(doc) <= maxJSONValueDepth && jsonValueOracleWhitespaceRun(doc) <= maxJSONValueWhitespaceRun
}

func jsonValueHistory(base int, prefix []byte) []int {
	history := make([]int, len(prefix))
	for i, b := range prefix {
		history[i] = base + int(b)
	}
	return history
}

// jsonValueMaskAccepts feeds doc one byte token at a time and requires EOS at
// the end.
func jsonValueMaskAccepts(mask *JSONValueMask, vocab, base int, doc []byte) bool {
	logits := make([]float32, vocab)
	for i := 0; i <= len(doc); i++ {
		clear(logits)
		mask.MaskLogits(jsonValueHistory(base, doc[:i]), logits)
		id := jsonValueTestEOS
		if i < len(doc) {
			id = base + int(doc[i])
		}
		if math.IsInf(float64(logits[id]), -1) {
			return false
		}
	}
	return true
}

func compileJSONValueTestMask(t testing.TB, dec TokenDecoder) *JSONValueMask {
	t.Helper()
	mask, err := CompileJSONObject(dec, CompileOptions{EOS: jsonValueTestEOS})
	if err != nil {
		t.Fatal(err)
	}
	return mask
}

// jsonValueRandomDocument writes a random object whose strings, numbers and
// whitespace cover the grammar; it is valid JSON by construction.
func jsonValueRandomDocument(rng *rand.Rand) []byte {
	var out bytes.Buffer
	space := func() {
		if rng.Intn(3) == 0 {
			for range rng.Intn(4) {
				out.WriteByte(" \t\n\r"[rng.Intn(4)])
			}
		}
	}
	str := func() {
		out.WriteByte('"')
		pieces := []string{"a", "Z", " ", "é", "€", "😀", "\u4a69", `\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, `\u00e9`, `\uD83D\uDE00`, `\ud800`, "{", "]", ":", ",", "\x7f"}
		for range rng.Intn(5) {
			out.WriteString(pieces[rng.Intn(len(pieces))])
		}
		out.WriteByte('"')
	}
	number := func() {
		forms := []string{"0", "-0", "7", "-12", "105", "0.5", "-3.25", "1e5", "1E+5", "2e-3", "-0.0e0", "6.02E23", "9007199254740993"}
		out.WriteString(forms[rng.Intn(len(forms))])
	}
	var value func(depth int)
	object := func(depth int) {
		out.WriteByte('{')
		n := rng.Intn(4)
		for i := range n {
			space()
			if i > 0 {
				out.WriteByte(',')
				space()
			}
			str()
			space()
			out.WriteByte(':')
			space()
			value(depth + 1)
		}
		space()
		out.WriteByte('}')
	}
	value = func(depth int) {
		switch r := rng.Intn(9); {
		case r == 0 && depth < 5:
			object(depth)
		case r == 1 && depth < 5:
			out.WriteByte('[')
			for i := range rng.Intn(4) {
				space()
				if i > 0 {
					out.WriteByte(',')
					space()
				}
				value(depth + 1)
			}
			space()
			out.WriteByte(']')
		case r <= 3:
			str()
		case r <= 6:
			number()
		default:
			out.WriteString([]string{"true", "false", "null"}[rng.Intn(3)])
		}
	}
	object(1)
	return out.Bytes()
}

func jsonValueMutate(rng *rand.Rand, doc []byte) []byte {
	const nasty = "{}[]\":,\\/u0e+-.tfn \t\n\x00\x1f\xc3\xa9\xed\xa0\xf4\x90\xff"
	out := append([]byte(nil), doc...)
	i := rng.Intn(len(out) + 1)
	b := nasty[rng.Intn(len(nasty))]
	switch {
	case rng.Intn(3) == 0 || i == len(out):
		return append(out[:i], append([]byte{b}, out[i:]...)...)
	case rng.Intn(2) == 0:
		return append(out[:i], out[i+1:]...)
	}
	out[i] = b
	return out
}

func jsonValueNested(open, close string, n int) string {
	return strings.Repeat(open, n) + strings.Repeat(close, n)
}

var jsonValueCuratedDocuments = []string{
	`{}`, `{"a":1}`, `{"":""}`, `{"a":[]}`, `{"a":{}}`, `{"a":[1,2,[3,{"b":null}]],"c":true,"d":false}`,
	`{"s":"\"\\\/\b\f\n\r\t\u00e9\uABCD\ud800"}`, `{"s":"é€😀\u0000"}`, "{\"s\":\"\x7f\"}",
	`{"n":[0,-0,1,-1,10,0.5,-0.5,1e5,1E5,1e+5,1e-5,1.25E-07,-9.5e+300]}`,
	"{ \"a\" : [ 1 , 2 ] ,\n\t\"b\"\r\n:\n{ } }", `{"a" :1}`, "{\"a\":1 }", "{\n}",
	"{" + strings.Repeat(" ", maxJSONValueWhitespaceRun) + "}",
	"{" + strings.Repeat(" ", maxJSONValueWhitespaceRun+1) + "}",
	"{\"a\":" + strings.Repeat("\n", maxJSONValueWhitespaceRun) + "1}",
	"{\"a\":" + strings.Repeat("\t", maxJSONValueWhitespaceRun+1) + "1}",
	`{"a":"` + strings.Repeat(" ", 3*maxJSONValueWhitespaceRun) + `"}`,
	`{"a":` + jsonValueNested("[", "]", maxJSONValueDepth-1) + `}`,
	`{"a":` + jsonValueNested("[", "]", maxJSONValueDepth) + `}`,
	strings.Repeat(`{"a":`, maxJSONValueDepth-1) + `{}` + strings.Repeat("}", maxJSONValueDepth-1),
	strings.Repeat(`{"a":`, maxJSONValueDepth) + `{}` + strings.Repeat("}", maxJSONValueDepth),
	`[]`, `"a"`, `1`, `true`, `null`, ` {}`, `{} `, "{}\n", `{}{}`, `{},`, `{}}`,
	`{a:1}`, `{"a"}`, `{"a":}`, `{"a":1,}`, `{,"a":1}`, `{"a":1 "b":2}`, `{"a":[1,]}`, `{"a":[,1]}`, `{"a":[1 2]}`, `{"a":[}`, `{"a":{]}`,
	`{"a":01}`, `{"a":-}`, `{"a":1.}`, `{"a":.5}`, `{"a":+1}`, `{"a":1e}`, `{"a":1e+}`, `{"a":0x1}`, `{"a":1.5.2}`, `{"a":--1}`, `{"a":-01}`,
	`{"a":tru}`, `{"a":True}`, `{"a":nul}`, `{"a":NaN}`, `{"a":Infinity}`,
	`{"s":"\x"}`, `{"s":"\u12"}`, `{"s":"\u12G4"}`, `{"s":"\U0041"}`, `{"s":"\'"}`,
	"{\"s\":\"\x00\"}", "{\"s\":\"\x1f\"}", "{\"s\":\"\n\"}",
	"{\"s\":\"\xff\"}", "{\"s\":\"\xc3\"}", "{\"s\":\"\xc0\x80\"}", "{\"s\":\"\xc1\xbf\"}", "{\"s\":\"\xe0\x80\x80\"}",
	"{\"s\":\"\xed\xa0\x80\"}", "{\"s\":\"\xed\x9f\xbf\"}", "{\"s\":\"\xf0\x8f\xbf\xbf\"}", "{\"s\":\"\xf4\x90\x80\x80\"}", "{\"s\":\"\xf4\x8f\xbf\xbf\"}", "{\"s\":\"\xf5\x80\x80\x80\"}", "{\"s\":\"\x80\"}",
	"{\"\xc3\xa9\":1}", "{\"\xe9\":1}", "{\xc3\xa9:1}",
}

// fak-test:runtime fast est=100ms lane=default
func TestJSONValueMaskDocumentOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(61))
	dec, base := jsonValueTestVocab(rng, 64)
	mask := compileJSONValueTestMask(t, dec)
	docs := make([][]byte, 0, 2000)
	for _, doc := range jsonValueCuratedDocuments {
		docs = append(docs, []byte(doc))
	}
	accepted, rejected := 0, 0
	for range 300 {
		doc := jsonValueRandomDocument(rng)
		if !jsonValueOracleDocument(doc) {
			t.Fatalf("generator produced a document the oracle rejects: %q", doc)
		}
		docs = append(docs, doc)
		for range 4 {
			docs = append(docs, jsonValueMutate(rng, doc))
		}
	}
	bad := 0
	for _, doc := range docs {
		want := jsonValueOracleDocument(doc)
		if want {
			accepted++
		} else {
			rejected++
		}
		if got := jsonValueMaskAccepts(mask, len(dec), base, doc); got != want {
			t.Errorf("document %q: mask accepts=%v, oracle=%v", doc, got, want)
			if bad++; bad > 10 {
				t.FailNow()
			}
		}
	}
	if accepted < 300 || rejected < 300 {
		t.Fatalf("oracle split too lopsided: accepted=%d rejected=%d", accepted, rejected)
	}
}

// jsonValueCapsCannotBind reports whether q has too few container openings
// and too short a whitespace run for the depth or whitespace cap to matter.
func jsonValueCapsCannotBind(q []byte) bool {
	run, longest := 0, 0
	for _, b := range q {
		if b == ' ' || b == '\t' || b == '\n' || b == '\r' {
			run++
			longest = max(longest, run)
		} else {
			run = 0
		}
	}
	return bytes.Count(q, []byte("{"))+bytes.Count(q, []byte("[")) <= maxJSONValueDepth && longest <= maxJSONValueWhitespaceRun
}

// checkJSONValuePrefix asserts that after prefix the mask admits exactly the
// tokens that keep a document reachable, and EOS exactly at a complete one.
// Pairs where a cap could bind are left to the explicit limit tests.
func checkJSONValuePrefix(t *testing.T, mask *JSONValueMask, dec jsonSchemaTokenDecoder, base int, prefix []byte, logits []float32) bool {
	t.Helper()
	clear(logits)
	mask.MaskLogits(jsonValueHistory(base, prefix), logits)
	// Viability is prefix-closed, so a piece whose first byte already strands
	// the prefix needs no decode.
	var firstViable [256]bool
	for b := range firstViable {
		firstViable[b], _ = jsonValueOracleViable(append(append([]byte(nil), prefix...), byte(b)))
	}
	ok := true
	for id := range logits {
		admitted := !math.IsInf(float64(logits[id]), -1)
		var want bool
		if id == jsonValueTestEOS {
			_, want = jsonValueOracleViable(prefix)
		} else if piece := dec[id]; len(piece) == 1 {
			want = firstViable[piece[0]]
		} else if len(piece) > 1 && firstViable[piece[0]] {
			q := append(append([]byte(nil), prefix...), piece...)
			if !jsonValueCapsCannotBind(q) {
				continue
			}
			want, _ = jsonValueOracleViable(q)
		}
		if admitted != want {
			t.Errorf("after %q token %d %q admitted=%v, oracle %v", prefix, id, dec[id], admitted, want)
			ok = false
		}
	}
	return ok
}

// fak-test:runtime fast est=1500ms lane=default
func TestJSONValueMaskPrefixOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(67))
	dec, base := jsonValueTestVocab(rng, 48)
	const walkLen = 24
	mask := compileJSONValueTestMask(t, dec)
	logits := make([]float32, len(dec))
	failures := 0
	check := func(prefix []byte) {
		if !checkJSONValuePrefix(t, mask, dec, base, prefix, logits) {
			if failures++; failures > 4 {
				t.FailNow()
			}
		}
	}

	const expand = "{}[]\":,01-.e+t\\u \n\xc3\xa9\xed\xa0"
	frontier := [][]byte{nil}
	for length := 0; length <= 4; length++ {
		var next [][]byte
		for _, prefix := range frontier {
			check(prefix)
			if length == 4 {
				continue
			}
			for i := range len(expand) {
				candidate := append(append([]byte(nil), prefix...), expand[i])
				if viable, _ := jsonValueOracleViable(candidate); viable {
					next = append(next, candidate)
				}
			}
		}
		frontier = next
	}

	visited := 0
	for range 120 {
		var prefix []byte
		for len(prefix) < walkLen {
			clear(logits)
			mask.MaskLogits(jsonValueHistory(base, prefix), logits)
			var options, favored []byte
			for b := range 256 {
				if !math.IsInf(float64(logits[base+b]), -1) {
					options = append(options, byte(b))
					// Structural, numeric and escape bytes are rare among 256
					// options; favor them so walks reach nested states.
					if strings.IndexByte(`{}[]":,0123456789-.eE+\utfn `, byte(b)) >= 0 {
						favored = append(favored, byte(b))
					}
				}
			}
			if len(options) == 0 {
				break
			}
			if len(favored) > 0 && rng.Intn(4) != 0 {
				options = favored
			}
			prefix = append(prefix, options[rng.Intn(len(options))])
			check(prefix)
			visited++
		}
	}
	if visited < 1000 {
		t.Fatalf("random walks visited only %d prefixes", visited)
	}
}

// fak-test:runtime fast est=1000ms lane=default
func TestJSONValueMaskDocumentPrefixOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(71))
	dec, base := jsonValueTestVocab(rng, 48)
	mask := compileJSONValueTestMask(t, dec)
	logits := make([]float32, len(dec))
	failures, prefixes, keyStrings, valueStrings, nested := 0, 0, 0, 0, 0
	for range 40 {
		doc := jsonValueRandomDocument(rng)
		for i := 0; i <= len(doc); i++ {
			// The mask's own cursor only meters coverage; admission is judged by
			// the encoding/json oracle.
			if state, ok := mask.stateAfter(jsonValueHistory(base, doc[:i])); ok {
				if state.lex == jvString && state.key {
					keyStrings++
				}
				if state.lex == jvString && !state.key {
					valueStrings++
				}
				if state.depth >= 2 {
					nested++
				}
			}
			prefixes++
			if !checkJSONValuePrefix(t, mask, dec, base, doc[:i], logits) {
				if failures++; failures > 4 {
					t.FailNow()
				}
			}
		}
	}
	if prefixes < 800 || keyStrings < 100 || valueStrings < 25 || nested < 100 {
		t.Fatalf("document prefixes too narrow: prefixes=%d key strings=%d value strings=%d nested=%d", prefixes, keyStrings, valueStrings, nested)
	}
}

// jsonValueNextBytes lists the single bytes admitted after prefix and whether
// EOS is admitted.
func jsonValueNextBytes(mask *JSONValueMask, vocab, base int, prefix string) (string, bool) {
	logits := make([]float32, vocab)
	mask.MaskLogits(jsonValueHistory(base, []byte(prefix)), logits)
	var allowed []byte
	for b := range 256 {
		if !math.IsInf(float64(logits[base+b]), -1) {
			allowed = append(allowed, byte(b))
		}
	}
	return string(allowed), !math.IsInf(float64(logits[jsonValueTestEOS]), -1)
}

func jsonValueAdmits(mask *JSONValueMask, dec jsonSchemaTokenDecoder, base int, prefix, piece string) bool {
	id := -1
	for i, candidate := range dec {
		if i != jsonValueTestEOS && string(candidate) == piece {
			id = i
			break
		}
	}
	if id < 0 {
		panic("jsonValueAdmits: piece not in vocabulary: " + strconv.Quote(piece))
	}
	logits := make([]float32, len(dec))
	mask.MaskLogits(jsonValueHistory(base, []byte(prefix)), logits)
	return !math.IsInf(float64(logits[id]), -1)
}

// fak-test:runtime fast est=20ms lane=default
func TestJSONValueMaskDepthAndWhitespaceLimits(t *testing.T) {
	dec, base := jsonValueTestVocab(rand.New(rand.NewSource(71)), 0)
	extra := []string{
		strings.Repeat(" ", maxJSONValueWhitespaceRun), strings.Repeat(" ", maxJSONValueWhitespaceRun+1),
		strings.Repeat(" ", 3*maxJSONValueWhitespaceRun), strings.Repeat(" ", 2) + "}", strings.Repeat(" ", 3) + "}",
	}
	for _, piece := range extra {
		dec = append(dec, []byte(piece))
	}
	mask := compileJSONValueTestMask(t, dec)
	const values = "\"-0123456789fnt"
	const space = "\t\n\r "
	sorted := func(s string) string {
		b := []byte(s)
		for i := range b {
			for j := i + 1; j < len(b); j++ {
				if b[j] < b[i] {
					b[i], b[j] = b[j], b[i]
				}
			}
		}
		return string(b)
	}
	deepest := `{"a":` + strings.Repeat("[", maxJSONValueDepth-1)
	for _, tc := range []struct {
		prefix string
		want   string
		eos    bool
	}{
		{"", "{", false},
		{"{}", "", true},
		{`{"a":` + strings.Repeat("[", maxJSONValueDepth-2), sorted(space + values + "[]{"), false},
		{deepest, sorted(space + values + "]"), false},
		{deepest + "1", sorted(space + "0123456789.eE,]"), false},
		{deepest + `"`, "", false},
		{strings.Repeat(`{"a":`, maxJSONValueDepth-1), sorted(space + values + "[{"), false},
		{strings.Repeat(`{"a":`, maxJSONValueDepth-1) + "{", sorted(space + "\"}"), false},
		{"{" + strings.Repeat(" ", maxJSONValueWhitespaceRun-1), sorted(space + "\"}"), false},
		{"{" + strings.Repeat(" ", maxJSONValueWhitespaceRun), "\"}", false},
		{`{"a":1` + strings.Repeat("\n", maxJSONValueWhitespaceRun), ",}", false},
		{`{"a":[1,` + strings.Repeat("\r", maxJSONValueWhitespaceRun), sorted(values + "[{"), false},
	} {
		got, eos := jsonValueNextBytes(mask, len(dec), base, tc.prefix)
		if tc.prefix == deepest+`"` {
			if eos || !strings.Contains(got, "[") || strings.Contains(got, "\x1f") {
				t.Errorf("inside a string at the depth cap: bytes %q eos=%v, want string content", got, eos)
			}
			continue
		}
		if got != tc.want || eos != tc.eos {
			t.Errorf("after %q: bytes %q eos=%v, want %q eos=%v", tc.prefix, got, eos, tc.want, tc.eos)
		}
	}

	for _, tc := range []struct {
		prefix, piece string
		want          bool
	}{
		{"{" + strings.Repeat(" ", maxJSONValueWhitespaceRun-2), "  }", true},
		{"{" + strings.Repeat(" ", maxJSONValueWhitespaceRun-2), "   }", false},
		{"{" + strings.Repeat(" ", maxJSONValueWhitespaceRun-2), "  ", true},
		{"{" + strings.Repeat(" ", maxJSONValueWhitespaceRun-2), " \t\n\r ", false},
		{"{", strings.Repeat(" ", maxJSONValueWhitespaceRun), true},
		{"{", strings.Repeat(" ", maxJSONValueWhitespaceRun+1), false},
		{"{ ", strings.Repeat(" ", maxJSONValueWhitespaceRun), false},
		{`{"a":"`, strings.Repeat(" ", 3*maxJSONValueWhitespaceRun), true},
		{`{"a":"` + strings.Repeat(" ", maxJSONValueWhitespaceRun), strings.Repeat(" ", 3*maxJSONValueWhitespaceRun), true},
		{`{"a":1` + strings.Repeat(" ", maxJSONValueWhitespaceRun-2), "  }", true},
		{`{"a":1` + strings.Repeat(" ", maxJSONValueWhitespaceRun-2), "   }", false},
		{`{"a":` + strings.Repeat("[", maxJSONValueDepth-2), "[[", false},
		{`{"a":` + strings.Repeat("[", maxJSONValueDepth-3), "[[", true},
		{`{"a":` + strings.Repeat("[", maxJSONValueDepth-2), "[]", true},
		{"{}", "} ", false},
		{"{}", "}\n", false},
		{"", "{}", true},
		{"", " \t\n\r ", false},
	} {
		if got := jsonValueAdmits(mask, dec, base, tc.prefix, tc.piece); got != tc.want {
			t.Errorf("after %q piece %q admitted=%v, want %v", tc.prefix, tc.piece, got, tc.want)
		}
	}

	nested := `{"a":` + jsonValueNested("[", "]", maxJSONValueDepth-1) + `}`
	if !jsonValueMaskAccepts(mask, len(dec), base, []byte(nested)) {
		t.Fatalf("document at the depth cap rejected")
	}
	if jsonValueMaskAccepts(mask, len(dec), base, []byte(`{"a":`+jsonValueNested("[", "]", maxJSONValueDepth)+`}`)) {
		t.Fatalf("document past the depth cap accepted")
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestJSONValueMaskRefusesEmptyAndDeadTokens(t *testing.T) {
	if _, err := CompileJSONObject(nil, CompileOptions{EOS: 0}); err == nil {
		t.Fatal("CompileJSONObject accepted a nil decoder")
	}
	dec, base := jsonValueTestVocab(rand.New(rand.NewSource(73)), 0)
	mask := compileJSONValueTestMask(t, dec)
	logits := make([]float32, len(dec))
	for _, prefix := range []string{"", "{", `{"a":"`, `{"a":1`, "{}"} {
		clear(logits)
		mask.MaskLogits(jsonValueHistory(base, []byte(prefix)), logits)
		if !math.IsInf(float64(logits[1]), -1) {
			t.Fatalf("after %q the empty non-EOS token is admitted", prefix)
		}
	}
	for _, history := range [][]int{
		jsonValueHistory(base, []byte("[")),
		jsonValueHistory(base, []byte(`{"a":01`)),
		append(jsonValueHistory(base, []byte(`{"a":"`)), 1),
		jsonValueHistory(base, []byte("{} ")),
	} {
		clear(logits)
		mask.MaskLogits(history, logits)
		for id, logit := range logits {
			if !math.IsInf(float64(logit), -1) {
				t.Fatalf("dead history %v admits token %d %q", history, id, dec[id])
			}
		}
	}
}

// fak-test:runtime fast est=150ms lane=default
func TestJSONValueMaskLongValuesKeepCacheFinite(t *testing.T) {
	dec, base := jsonValueTestVocab(rand.New(rand.NewSource(79)), 32)
	mask := compileJSONValueTestMask(t, dec)
	visit := func(doc string, from int) {
		for i := from; i <= len(doc); i++ {
			jsonValueNextBytes(mask, len(dec), base, doc[:i])
		}
	}
	item := `{"name":"widget é😀 \"q\" \u00e9","qty":-12.5e+3,"tags":[1,true,null,"x"]}`
	doc := `{"n":7,"text":"` + strings.Repeat("lorem ipsum é ", 40) + `","items":[` + item
	for range 6 {
		doc += "," + item
	}
	visit(doc, 0)
	before := len(mask.cache)
	more := doc + strings.Repeat(","+item, 3) + `],"digits":` + strings.Repeat("9", 200) + `,"s":"` + strings.Repeat("ab", 300)
	visit(more, len(doc))
	if len(mask.cache) != before {
		t.Fatalf("longer strings, numbers and arrays added cache states: %d -> %d", before, len(mask.cache))
	}
	if before > maxJSONValueMaskCacheStates {
		t.Fatalf("cache holds %d states, over the cap %d", before, maxJSONValueMaskCacheStates)
	}
	if !jsonValueMaskAccepts(mask, len(dec), base, []byte(more+`"}`)) {
		t.Fatal("long document rejected")
	}
}

// BenchmarkJSONValueMaskLargeVocab measures one masking step inside a free
// string nested two containers deep over a 151,936-token vocabulary: cold
// builds a fresh mask (vocabulary decode plus the state's admission set); miss
// computes the admission set over an already-decoded vocabulary, the cost of
// each new state; warm reuses the cached state.
func BenchmarkJSONValueMaskLargeVocab(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	dec := make(jsonSchemaTokenDecoder, 151936)
	for i := range 256 {
		dec[i] = []byte{byte(i)}
	}
	const alphabet = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\":,.{}[]\\\n"
	for i := 256; i < len(dec); i++ {
		piece := make([]byte, 2+rng.Intn(7))
		for j := range piece {
			piece[j] = alphabet[rng.Intn(len(alphabet))]
		}
		dec[i] = piece
	}
	var history []int
	for _, c := range []byte(`{"a":[{"b":"some free text`) {
		history = append(history, int(c))
	}
	logits := make([]float32, len(dec))
	b.Run("cold", func(b *testing.B) {
		for range b.N {
			mask, err := CompileJSONObject(dec, CompileOptions{EOS: -1})
			if err != nil {
				b.Fatal(err)
			}
			mask.MaskLogits(history, logits)
		}
	})
	b.Run("miss", func(b *testing.B) {
		mask, err := CompileJSONObject(dec, CompileOptions{EOS: -1})
		if err != nil {
			b.Fatal(err)
		}
		mask.MaskLogits(history, logits)
		b.ResetTimer()
		for range b.N {
			mask.cache = nil
			mask.MaskLogits(history, logits)
		}
	})
	b.Run("warm", func(b *testing.B) {
		mask, err := CompileJSONObject(dec, CompileOptions{EOS: -1})
		if err != nil {
			b.Fatal(err)
		}
		mask.MaskLogits(history, logits)
		b.ResetTimer()
		for range b.N {
			mask.MaskLogits(history, logits)
		}
	})
}
