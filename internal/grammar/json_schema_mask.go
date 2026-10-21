package grammar

import (
	"errors"
	"math"
	"sync"
)

// JSONSchemaMask constrains generated tokens to one bare JSON object built from
// literal and value segments like CallMask's, without the tool-call envelope.
// Unsupported schema features are refused by CompileJSONSchema.
type JSONSchemaMask struct {
	objects []compiledTool
	dec     TokenDecoder
	eos     int

	decodedOnce sync.Once
	decoded     [][]byte

	programs   []jsonShapeProgram
	cacheMu    sync.Mutex
	cacheVocab int
	cache      map[string][]uint64
}

const (
	maxJSONSchemaBytes      = 65_536
	maxJSONSchemaProperties = 16
	maxJSONSchemaOptional   = 4
	// maxJSONSchemaMaskCacheStates bounds one request's cached admission sets.
	maxJSONSchemaMaskCacheStates = 64
)

// CompileJSONSchema compiles the supported response subset (scalars, scalar
// arrays and nested objects) into a tokenizer-aware mask. Up to four optional
// properties across the tree expand into at most 16 canonical shapes;
// unsupported keywords return ErrUnsupportedSchema.
func CompileJSONSchema(schema []byte, dec TokenDecoder, opts CompileOptions) (*JSONSchemaMask, error) {
	if dec == nil {
		return nil, errors.New("grammar: CompileJSONSchema needs a TokenDecoder (inject tokenizer TokenBytes)")
	}
	if len(schema) > maxJSONSchemaBytes {
		return nil, unsupported("response", "", "schema size %d exceeds limit %d", len(schema), maxJSONSchemaBytes)
	}
	shapes, err := normalizeJSONSchemaShapes("response", schema, maxJSONSchemaProperties, maxJSONSchemaOptional)
	if err != nil {
		return nil, err
	}
	objects := make([]compiledTool, 0, len(shapes))
	for _, shape := range shapes {
		last := len(shape.segs) - 1
		if last < 0 || shape.segs[last].field != nil || len(shape.segs[last].lit) == 0 ||
			shape.segs[last].lit[len(shape.segs[last].lit)-1] != '}' {
			return nil, errors.New("grammar: CompileJSONSchema internal object grammar is incomplete")
		}
		objects = append(objects, compiledTool{fields: shape.fields, segs: shape.segs})
	}
	programs := make([]jsonShapeProgram, len(objects))
	for i := range objects {
		programs[i] = newJSONShapeProgram(objects[i])
	}
	return &JSONSchemaMask{
		objects:  objects,
		dec:      dec,
		eos:      opts.EOS,
		programs: programs,
	}, nil
}

// MaskLogits implements model.LogitMask structurally. Every candidate token is
// checked byte by byte, so a token that starts legally but crosses the complete
// object boundary is rejected. Dead-end histories mask the whole vocabulary.
func (m *JSONSchemaMask) MaskLogits(history []int, logits []float32) {
	if m == nil || m.dec == nil || len(logits) == 0 {
		return
	}
	m.cacheDecoded(len(logits))
	state, ok := m.stateAfter(history)
	if !ok {
		maskAll(logits)
		return
	}
	complete := false
	for i := range state.cursors {
		if m.programs[i].complete(&state.cursors[i]) {
			complete = true
		}
	}
	admitted := m.admittedTokens(state, len(logits))
	neg := float32(math.Inf(-1))
	for id := range logits {
		if id == m.eos {
			if !complete {
				logits[id] = neg
			}
			continue
		}
		if admitted[id>>6]&(1<<(uint(id)&63)) == 0 {
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

// stateAfter steps every shape through the history bytes. It reports false for
// an undecodable token or a history no shape accepts.
func (m *JSONSchemaMask) stateAfter(history []int) (jsonMaskState, bool) {
	state := jsonMaskState{cursors: make([]jsonCursor, len(m.programs))}
	for i := range m.programs {
		state.cursors[i] = m.programs[i].start()
	}
	for _, id := range history {
		piece := m.tokenBytes(id)
		if len(piece) == 0 {
			return state, false
		}
		for _, b := range piece {
			for i := range state.cursors {
				m.programs[i].step(&state.cursors[i], b)
			}
		}
		if !state.alive() {
			return state, false
		}
	}
	return state, true
}

// admittedTokens returns the bitset of tokens whose every byte keeps at least
// one shape alive from state. Results are cached per state and vocabulary size.
func (m *JSONSchemaMask) admittedTokens(state jsonMaskState, vocabSize int) []uint64 {
	key := state.key()
	m.cacheMu.Lock()
	if m.cacheVocab == vocabSize {
		if bits, ok := m.cache[key]; ok {
			m.cacheMu.Unlock()
			return bits
		}
	}
	m.cacheMu.Unlock()

	var firstOK [256]bool
	scratch := make([]jsonCursor, len(state.cursors))
	survives := func(piece []byte) bool {
		copy(scratch, state.cursors)
		for _, b := range piece {
			alive := false
			for i := range scratch {
				if m.programs[i].step(&scratch[i], b) {
					alive = true
				}
			}
			if !alive {
				return false
			}
		}
		return true
	}
	for b := range firstOK {
		firstOK[b] = survives([]byte{byte(b)})
	}
	bits := make([]uint64, (vocabSize+63)/64)
	for id := 0; id < vocabSize; id++ {
		piece := m.tokenBytes(id)
		if len(piece) == 0 || !firstOK[piece[0]] {
			continue
		}
		if len(piece) == 1 || survives(piece) {
			bits[id>>6] |= 1 << (uint(id) & 63)
		}
	}

	m.cacheMu.Lock()
	// A full cache restarts rather than freezing, so a later long free string
	// still reaches the cached path after many structural states.
	if m.cacheVocab != vocabSize || m.cache == nil || len(m.cache) >= maxJSONSchemaMaskCacheStates {
		m.cache, m.cacheVocab = map[string][]uint64{}, vocabSize
	}
	m.cache[key] = bits
	m.cacheMu.Unlock()
	return bits
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
