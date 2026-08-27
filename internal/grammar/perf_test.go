package grammar

import (
	"math"
	"strconv"
	"testing"
)

// perfVocab models the two token classes that make constrained masking costly:
// every byte as a fallback token plus common multi-byte JSON/name/value pieces.
// It also records decoder traffic so the benchmark can attribute repeated
// vocabulary synchronization separately from FSM transition work.
type perfVocab struct {
	pieces [][]byte
	calls  int
}

func newPerfVocab() *perfVocab {
	pieces := make([][]byte, 0, 4096)
	for i := 0; i < 256; i++ {
		pieces = append(pieces, []byte{byte(i)})
	}
	common := []string{
		`{"name":"`, `","arguments":{`, `"query":"`, `","limit":`, `,"verbose":`, `}}`,
		`search`, `lookup`, `weather`, `false`, `true`, `null`, `"`, `,`, `:`, `{`, `}`,
		`café`, `東京`, `🙂`, `\u`, `\n`, `-1`, `0`, `10`, `100`, `3.14`,
	}
	for _, s := range common {
		pieces = append(pieces, []byte(s))
	}
	for len(pieces) < 4096 {
		i := len(pieces)
		pieces = append(pieces, []byte("token_"+strconv.Itoa(i)))
	}
	return &perfVocab{pieces: pieces}
}

func (v *perfVocab) TokenBytes(id int) []byte {
	v.calls++
	if id < 0 || id >= len(v.pieces) {
		return nil
	}
	return v.pieces[id]
}

var perfSpecs = []ToolSpec{
	{Name: "search", Schema: []byte(`{"type":"object","properties":{"limit":{"type":"integer"},"query":{"type":"string"},"verbose":{"type":"boolean"}},"required":["query","limit","verbose"]}`)},
	{Name: "weather", Schema: []byte(`{"type":"object","properties":{"city":{"type":"string"},"units":{"type":"string","enum":["c","f"]}},"required":["city","units"]}`)},
}

func perfMask(b testing.TB) (*CallMask, *perfVocab) {
	b.Helper()
	v := newPerfVocab()
	m, err := Compile(perfSpecs, v, CompileOptions{EOS: -1})
	if err != nil {
		b.Fatal(err)
	}
	return m, v
}

func allowedIDs(m *CallMask, history []int, vocab int) []int {
	logits := make([]float32, vocab)
	m.MaskLogits(history, logits)
	out := make([]int, 0, vocab)
	for id, logit := range logits {
		if !math.IsInf(float64(logit), -1) {
			out = append(out, id)
		}
	}
	return out
}

func TestCallMaskPerformanceCorpusExactness(t *testing.T) {
	m, v := perfMask(t)
	cases := []struct {
		name   string
		prefix string
	}{
		{name: "tool-schema", prefix: `{"name":"search","arguments":{"limit":10,"query":"tea","verbose":false}}`[:28]},
		{name: "nested-json-dead-end", prefix: `{"name":"search","arguments":{"limit":{"nested":1}`},
		{name: "unicode-token-boundary", prefix: `{"name":"search","arguments":{"limit":10,"query":"caf`},
		{name: "impossible-grammar", prefix: `{"name":"missing"`},
		{name: "complete-envelope", prefix: `{"name":"weather","arguments":{"city":"Paris","units":"c"}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := allowedIDs(m, byteIDs(tc.prefix), len(v.pieces))
			after := allowedIDs(m, byteIDs(tc.prefix), len(v.pieces))
			if len(before) != len(after) {
				t.Fatalf("allowed-token count changed: %d != %d", len(before), len(after))
			}
			for i := range before {
				if before[i] != after[i] {
					t.Fatalf("allowed token %d changed: %d != %d", i, before[i], after[i])
				}
			}
		})
	}
}

func BenchmarkCallMaskCorpus(b *testing.B) {
	prefixes := map[string]string{
		"tool-schema":          `{"name":"search","arguments":{"limit":10,"query":"tea","verbose":false}}`[:28],
		"nested-json-dead-end": `{"name":"search","arguments":{"limit":{"nested":1}`,
		"unicode":              `{"name":"search","arguments":{"limit":10,"query":"caf` + "é",
		"impossible":           `{"name":"missing"`,
		"complete":             `{"name":"weather","arguments":{"city":"Paris","units":"c"}}`,
	}
	for name, prefix := range prefixes {
		b.Run(name, func(b *testing.B) {
			m, v := perfMask(b)
			history := byteIDs(prefix)
			logits := make([]float32, len(v.pieces))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				clear(logits)
				m.MaskLogits(history, logits)
			}
		})
	}
	b.Run("unconstrained-control", func(b *testing.B) {
		logits := make([]float32, 4096)
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			clear(logits)
		}
	})
}

func BenchmarkCallMaskComponents(b *testing.B) {
	b.Run("compile", func(b *testing.B) {
		v := newPerfVocab()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := Compile(perfSpecs, v, CompileOptions{EOS: -1}); err != nil {
				b.Fatal(err)
			}
		}
	})
	m, v := perfMask(b)
	prefix := []byte(`{"name":"search","arguments":{"limit":10,"query":"tea`)
	b.Run("allowed-token-construction", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _ = m.allowedNext(prefix)
		}
	})
	allowed, _ := m.allowedNext(prefix)
	piece := v.TokenBytes(256)
	b.Run("transition", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = m.admits(prefix, allowed, piece)
		}
	})
	b.Run("mask", func(b *testing.B) {
		history := byteIDs(string(prefix))
		logits := make([]float32, len(v.pieces))
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			clear(logits)
			m.MaskLogits(history, logits)
		}
	})
}
