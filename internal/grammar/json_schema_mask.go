package grammar

import (
	"bytes"
	"errors"
	"math"
	"sync"
)

// JSONSchemaMask constrains generated tokens to one bare JSON object. It reuses
// the required-scalar object grammar used by CallMask, without the tool-call
// envelope. Unsupported schema features are refused by CompileJSONSchema.
type JSONSchemaMask struct {
	object compiledTool
	dec    TokenDecoder
	eos    int

	decodedOnce sync.Once
	decoded     [][]byte
}

// CompileJSONSchema compiles the currently supported required-scalar object
// subset into a tokenizer-aware mask. The accepted schema surface is exactly
// normalizeFields' surface; unsupported keywords return ErrUnsupportedSchema.
func CompileJSONSchema(schema []byte, dec TokenDecoder, opts CompileOptions) (*JSONSchemaMask, error) {
	if dec == nil {
		return nil, errors.New("grammar: CompileJSONSchema needs a TokenDecoder (inject tokenizer TokenBytes)")
	}
	fields, err := normalizeFields("response", schema)
	if err != nil {
		return nil, err
	}
	segs := buildSegments(fields)
	last := len(segs) - 1
	if last < 0 || segs[last].field != nil || len(segs[last].lit) < 2 ||
		!bytes.Equal(segs[last].lit[len(segs[last].lit)-2:], []byte("}}")) {
		return nil, errors.New("grammar: CompileJSONSchema internal object grammar is incomplete")
	}
	// buildSegments closes both the arguments object and its tool envelope.
	// A response-format mask emits the bare arguments object, so remove only
	// the envelope's closing brace from the final immutable segment.
	segs[last].lit = append([]byte(nil), segs[last].lit[:len(segs[last].lit)-1]...)
	return &JSONSchemaMask{
		object: compiledTool{fields: fields, segs: segs},
		dec:    dec,
		eos:    opts.EOS,
	}, nil
}

// MaskLogits implements model.LogitMask structurally. Every candidate token is
// checked byte by byte, so a token that starts legally but crosses the complete
// object boundary is rejected. Dead-end histories mask the whole vocabulary.
func (m *JSONSchemaMask) MaskLogits(history []int, logits []float32) {
	if m == nil || m.dec == nil {
		return
	}
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
	allowed, complete := m.object.argsAllowed(prefix)
	if complete {
		maskAllExcept(logits, m.eos)
		return
	}
	if len(allowed) == 0 {
		maskAll(logits)
		return
	}
	neg := float32(math.Inf(-1))
	for id := range logits {
		if id == m.eos || !m.admits(prefix, allowed, m.tokenBytes(id)) {
			logits[id] = neg
		}
	}
}

func (m *JSONSchemaMask) cacheDecoded(vocabSize int) {
	m.decodedOnce.Do(func() {
		m.decoded = make([][]byte, vocabSize)
		for id := range m.decoded {
			m.decoded[id] = append([]byte(nil), m.dec.TokenBytes(id)...)
		}
	})
}

func (m *JSONSchemaMask) tokenBytes(id int) []byte {
	if id >= 0 && id < len(m.decoded) {
		return m.decoded[id]
	}
	return m.dec.TokenBytes(id)
}

func (m *JSONSchemaMask) admits(prefix []byte, first map[byte]bool, token []byte) bool {
	if len(token) == 0 {
		return false
	}
	buf := append([]byte(nil), prefix...)
	allowed, complete := first, false
	for _, b := range token {
		if complete || !allowed[b] {
			return false
		}
		buf = append(buf, b)
		allowed, complete = m.object.argsAllowed(buf)
	}
	return true
}

func maskAll(logits []float32) {
	neg := float32(math.Inf(-1))
	for i := range logits {
		logits[i] = neg
	}
}

func maskAllExcept(logits []float32, keep int) {
	neg := float32(math.Inf(-1))
	for i := range logits {
		if i != keep {
			logits[i] = neg
		}
	}
}
