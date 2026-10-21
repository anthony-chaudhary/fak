package grammar

import (
	"math"
	"math/rand"
	"testing"
)

// referenceMaskLogits is the previous whole-prefix rescan mask, retained as the
// oracle the cursor mask must match exactly.
func referenceMaskLogits(m *JSONSchemaMask, history []int, logits []float32) {
	m.cacheDecoded(len(logits))
	prefix := make([]byte, 0, len(history))
	for _, id := range history {
		piece := m.tokenBytes(id)
		if len(piece) == 0 {
			maskAll(logits)
			return
		}
		prefix = append(prefix, piece...)
	}
	argsAllowed := func(buf []byte) (map[byte]bool, bool) {
		allowed := map[byte]bool{}
		complete := false
		for i := range m.objects {
			next, done := m.objects[i].argsAllowed(buf)
			for b := range next {
				allowed[b] = true
			}
			complete = complete || done
		}
		return allowed, complete
	}
	allowed, complete := argsAllowed(prefix)
	if complete && len(allowed) == 0 {
		maskAllExcept(logits, m.eos)
		return
	}
	if len(allowed) == 0 {
		maskAll(logits)
		return
	}
	neg := float32(math.Inf(-1))
	for id := range logits {
		if id == m.eos {
			if !complete {
				logits[id] = neg
			}
			continue
		}
		token := m.tokenBytes(id)
		ok := len(token) > 0
		buf := append([]byte(nil), prefix...)
		next := allowed
		for _, b := range token {
			if !ok || !next[b] {
				ok = false
				break
			}
			buf = append(buf, b)
			next, _ = argsAllowed(buf)
		}
		if !ok {
			logits[id] = neg
		}
	}
}

// cursorTestVocab returns a JSON-shaped vocabulary: EOS (empty) at id 0, fixed
// boundary-crossing pieces, every single byte at cursorSingleByteBase+b, then
// random pieces.
func cursorTestVocab(rng *rand.Rand) jsonSchemaTokenDecoder {
	const alphabet = `{}":,.-0123456789abtrufelsxyz!` + " "
	fixed := []string{
		"", `{"`, `":"`, `","`, `"}`, `"},`, `":`, `,"`, `}`, `1}`, `10}`, `100`, `0.`, `.5`, `-0`, `-1`,
		`true`, `false`, `tru`, `e}`, `"a"`, `"a!"`, `"ab"`, `a!`, `{"n":1`, `{"n":10}`, `"}tail`, `null`,
		`1,"`, `0.5,`, `10,"`, `-0}`, `"",`, `true,"`, `1000`, `"a b"`, `""`,
	}
	dec := jsonSchemaTokenDecoder{nil} // id 0 is EOS and decodes to nothing
	for _, s := range fixed {
		dec = append(dec, []byte(s))
	}
	if len(dec) != cursorSingleByteBase {
		panic("cursorTestVocab: fixed piece count changed")
	}
	for i := 0; i < 256; i++ {
		dec = append(dec, []byte{byte(i)})
	}
	for len(dec) < 900 {
		n := 1 + rng.Intn(6)
		b := make([]byte, n)
		for i := range b {
			b[i] = alphabet[rng.Intn(len(alphabet))]
		}
		dec = append(dec, b)
	}
	return dec
}

var cursorTestSchemas = []string{
	`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`,
	`{"type":"object","properties":{"s":{"type":"string","enum":["a","a!","ab","b"]},"t":{"type":"boolean"}},"required":["s","t"],"additionalProperties":false}`,
	`{"type":"object","properties":{"n":{"type":"integer","enum":[1,10,100,-5]},"x":{"type":"number"}},"required":["n"],"additionalProperties":false}`,
	`{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"},"c":{"type":"boolean"},"d":{"type":"number"},"e":{"type":"string","const":"z"}},"required":["a"],"additionalProperties":false}`,
	`{"type":"object","properties":{},"additionalProperties":false}`,
	`{"type":"object","properties":{"e":{"type":"string","enum":["","a b","a!","a"]},"n":{"type":"integer","enum":[1,10,100,1000,-1,-10,0,5]},"o":{"type":"number"},"p":{"type":"integer"},"q":{"type":"boolean"}},"required":["e"],"additionalProperties":false}`,
	`{"type":"object","properties":{"o":{"type":"object","properties":{"n":{"type":"integer"},"s":{"type":"string"}},"required":["n","s"]},"z":{"type":"boolean"}},"required":["o","z"]}`,
	`{"type":"object","properties":{"a":{"type":"integer"},"m":{"type":"object","properties":{"k":{"type":"string","enum":["a","b"]},"x":{"type":"number"}},"required":["k"]}},"required":["a"]}`,
}

// fak-test:runtime fast est=2s lane=default
func TestJSONSchemaCursorMaskMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	dec := cursorTestVocab(rng)
	for si, schema := range cursorTestSchemas {
		mask, err := CompileJSONSchema([]byte(schema), dec, CompileOptions{EOS: 0})
		if err != nil {
			t.Fatalf("schema %d: %v", si, err)
		}
		reference, err := CompileJSONSchema([]byte(schema), dec, CompileOptions{EOS: 0})
		if err != nil {
			t.Fatalf("schema %d: %v", si, err)
		}
		for walk := 0; walk < 40; walk++ {
			var history []int
			for step := 0; step < 40; step++ {
				got := make([]float32, len(dec))
				want := make([]float32, len(dec))
				for i := range got {
					got[i] = float32(i % 7)
					want[i] = got[i]
				}
				mask.MaskLogits(history, got)
				referenceMaskLogits(reference, history, want)
				var survivors []int
				for id := range got {
					if got[id] != want[id] {
						t.Fatalf("schema %d history %q token %d %q: cursor=%v reference=%v", si, historyText(dec, history), id, dec[id], got[id], want[id])
					}
					if !math.IsInf(float64(got[id]), -1) {
						survivors = append(survivors, id)
					}
				}
				if len(survivors) == 0 || (len(survivors) == 1 && survivors[0] == 0) {
					break
				}
				next := survivors[rng.Intn(len(survivors))]
				if rng.Intn(10) == 0 {
					next = 1 + rng.Intn(len(dec)-1) // also probe dead-end histories
				}
				if next == 0 {
					break
				}
				history = append(history, next)
			}
		}
	}
}

func historyText(dec jsonSchemaTokenDecoder, history []int) string {
	var out []byte
	for _, id := range history {
		out = append(out, dec.TokenBytes(id)...)
	}
	return string(out)
}

// fak-test:runtime fast est=50ms lane=default
func TestJSONSchemaCursorMaskEdgeCalls(t *testing.T) {
	dec := cursorTestVocab(rand.New(rand.NewSource(5)))
	mask, err := CompileJSONSchema([]byte(cursorTestSchemas[0]), dec, CompileOptions{EOS: 0})
	if err != nil {
		t.Fatal(err)
	}
	mask.MaskLogits(nil, nil)
	reference, err := CompileJSONSchema([]byte(cursorTestSchemas[0]), dec, CompileOptions{EOS: 0})
	if err != nil {
		t.Fatal(err)
	}
	for _, vocab := range []int{len(dec), 70, len(dec)} {
		got := make([]float32, vocab)
		want := make([]float32, vocab)
		mask.MaskLogits(nil, got)
		referenceMaskLogits(reference, nil, want)
		for id := range got {
			if got[id] != want[id] {
				t.Fatalf("vocab %d token %d: cursor=%v reference=%v", vocab, id, got[id], want[id])
			}
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestJSONSchemaCursorMaskReusesStringState(t *testing.T) {
	dec := cursorTestVocab(rand.New(rand.NewSource(3)))
	mask, err := CompileJSONSchema([]byte(cursorTestSchemas[0]), dec, CompileOptions{EOS: 0})
	if err != nil {
		t.Fatal(err)
	}
	history := []int{}
	for _, b := range []byte(`{"answer":"abc`) {
		history = append(history, cursorSingleByteBase+int(b))
	}
	first := make([]float32, len(dec))
	mask.MaskLogits(history, first)
	states := len(mask.cache)
	longer := append(append([]int(nil), history...), cursorSingleByteBase+int('x'))
	second := make([]float32, len(dec))
	mask.MaskLogits(longer, second)
	if len(mask.cache) != states {
		t.Fatalf("free-string continuation added cache states: %d -> %d", states, len(mask.cache))
	}
	for id := range first {
		if first[id] != second[id] {
			t.Fatalf("free-string masks differ at token %d %q", id, dec[id])
		}
	}
}

const cursorSingleByteBase = 38

// BenchmarkJSONSchemaMaskLargeVocab measures one masking step inside a free
// string over a 151,936-token vocabulary: cold builds a fresh mask (vocabulary
// decode plus the state's admission set); warm reuses the cached state.
func BenchmarkJSONSchemaMaskLargeVocab(b *testing.B) {
	rng := rand.New(rand.NewSource(7))
	dec := make(jsonSchemaTokenDecoder, 151936)
	for i := range 256 {
		dec[i] = []byte{byte(i)}
	}
	const alphabet = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\":,.{}"
	for i := 256; i < len(dec); i++ {
		piece := make([]byte, 2+rng.Intn(7))
		for j := range piece {
			piece[j] = alphabet[rng.Intn(len(alphabet))]
		}
		dec[i] = piece
	}
	schema := []byte(`{"type":"object","properties":{"answer":{"type":"string"},"a":{"type":"string"},"b":{"type":"string"},"c":{"type":"integer"},"d":{"type":"boolean"}},"required":["answer"],"additionalProperties":false}`)
	var history []int
	for _, c := range []byte(`{"a":"some free text`) {
		history = append(history, int(c))
	}
	logits := make([]float32, len(dec))
	b.Run("cold", func(b *testing.B) {
		for range b.N {
			mask, err := CompileJSONSchema(schema, dec, CompileOptions{EOS: -1})
			if err != nil {
				b.Fatal(err)
			}
			mask.MaskLogits(history, logits)
		}
	})
	b.Run("warm", func(b *testing.B) {
		mask, err := CompileJSONSchema(schema, dec, CompileOptions{EOS: -1})
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
