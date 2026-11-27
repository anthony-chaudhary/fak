package grammar

import (
	"bytes"
	"math/rand"
	"slices"
	"testing"
)

// TestVocabTrieSubtreeIsPrefixClosed pins subtreeEnd to a true prefix test:
// with a substring test "acab" would land inside the subtree of "ab".
func TestVocabTrieSubtreeIsPrefixClosed(t *testing.T) {
	vocab := [][]byte{[]byte("ab"), []byte("abc"), []byte("acab"), []byte("b"), []byte("ab"), nil}
	tr := newVocabTrie(len(vocab), func(id int) []byte { return vocab[id] })
	if len(tr.ids) != 5 {
		t.Fatalf("empty token must be excluded, got %d entries", len(tr.ids))
	}
	for i, p := range tr.pieces {
		end := tr.subtreeEnd[i]
		for j := range tr.pieces {
			in := j > i && j < end
			if want := j > i && bytes.HasPrefix(tr.pieces[j], p); in != want {
				t.Fatalf("entry %q: index %d (%q) in subtree=%v, want %v", p, j, tr.pieces[j], in, want)
			}
		}
	}
}

// TestJSONSchemaMaskTrieMatchesLinearScan requires the trie fill to be
// bit-identical to the linear scan over varied states of a random,
// prefix-heavy vocabulary with duplicate and empty tokens.
func TestJSONSchemaMaskTrieMatchesLinearScan(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	dec := make(jsonSchemaTokenDecoder, 4000)
	for i := range 256 {
		dec[i] = []byte{byte(i)}
	}
	const alphabet = "ab\":,{} 0159truefalsnl"
	for i := 256; i < len(dec); i++ {
		if i%97 == 0 {
			continue // empty token
		}
		piece := make([]byte, 1+rng.Intn(8))
		for j := range piece {
			piece[j] = alphabet[rng.Intn(len(alphabet))]
		}
		if i%53 == 0 {
			piece = slices.Clone(dec[i-1]) // duplicate bytes, distinct id
		}
		dec[i] = piece
	}
	schemas := []string{
		`{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"},"c":{"type":"boolean"}},"required":["a"],"additionalProperties":false}`,
		`{"type":"object","properties":{"n":{"type":"object","properties":{"x":{"type":"number"}},"required":["x"]},"l":{"type":"array","items":{"type":"string"}}},"required":["n"],"additionalProperties":false}`,
	}
	checked := 0
	for _, sc := range schemas {
		mask, err := CompileJSONSchema([]byte(sc), dec, CompileOptions{EOS: -1})
		if err != nil {
			t.Fatal(err)
		}
		mask.cacheDecoded(len(dec))
		for _, doc := range []string{``, `{`, `{"a`, `{"a":`, `{"a":"x`, `{"a":"x",`, `{"a":"x","b":1`, `{"n":{"x":1`, `{"n":{"x":1},"l":["q",`} {
			var hist []int
			for _, c := range []byte(doc) {
				hist = append(hist, int(c))
			}
			state, ok := mask.stateAfter(hist)
			if !ok {
				continue
			}
			want := mask.admittedTokensLinear(state, len(dec))
			got := mask.fillTrie(state, len(dec))
			if !slices.Equal(got, want) {
				t.Fatalf("schema %.30s state %q: trie bitset differs from linear scan", sc, doc)
			}
			checked++
		}
	}
	if checked < 6 {
		t.Fatalf("only %d states compared", checked)
	}
}

// BenchmarkJSONSchemaMaskMissFill compares the linear and trie cache-miss
// fills over several structural states of a 151,936-token vocabulary.
func BenchmarkJSONSchemaMaskMissFill(b *testing.B) {
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
	mask, err := CompileJSONSchema(schema, dec, CompileOptions{EOS: -1})
	if err != nil {
		b.Fatal(err)
	}
	mask.cacheDecoded(len(dec))
	var states []jsonMaskState
	for _, doc := range []string{``, `{`, `{"a`, `{"a":`, `{"a":"x`, `{"a":"x",`, `{"a":"x","b":"y`, `{"a":"x","c":1`} {
		var hist []int
		for _, c := range []byte(doc) {
			hist = append(hist, int(c))
		}
		if st, ok := mask.stateAfter(hist); ok {
			states = append(states, st)
		}
	}
	b.Run("linear", func(b *testing.B) {
		for range b.N {
			for _, st := range states {
				mask.admittedTokensLinear(st, len(dec))
			}
		}
	})
	b.Run("trie", func(b *testing.B) {
		mask.fillTrie(states[0], len(dec)) // build the trie outside the timed loop
		b.ResetTimer()
		for range b.N {
			for _, st := range states {
				mask.fillTrie(st, len(dec))
			}
		}
	})
}
