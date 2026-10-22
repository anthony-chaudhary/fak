package grammar

import (
	"errors"
	"math"
	"sync"
)

// JSONValueMask constrains generated tokens to exactly one JSON object, the
// OpenAI JSON mode contract: any nested objects, arrays, strings, numbers,
// true, false and null, with the full RFC 8259 string and number syntax.
//
// Output surface, chosen so the grammar has finitely many cache states and no
// unbounded loop the model can fall into:
//   - the object starts at the first byte, and EOS is the only token admitted
//     once its closing brace is emitted, so no leading or trailing whitespace;
//   - whitespace (space, tab, LF, CR) is allowed between tokens inside the
//     object, at most maxJSONValueWhitespaceRun consecutive bytes;
//   - strings accept every escape (\" \\ \/ \b \f \n \r \t and \u with four hex
//     digits, unpaired surrogates included as in RFC 8259's grammar) and raw
//     bytes from 0x20 up only as valid UTF-8;
//   - numbers accept sign, fraction and exponent;
//   - containers nest at most maxJSONValueDepth deep, the object included; an
//     opening bracket past the cap is never admitted.
//
// Adapted from DeepInfer's dispatch of JSON output to an XGrammar builtin
// any-JSON grammar with a bounded whitespace count (Apache-2.0,
// deepseek-ai/DeepInfer deepinfer/structure_output/compiler.py at
// 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01); no upstream source is copied.
type JSONValueMask struct {
	dec TokenDecoder
	eos int

	decodedOnce sync.Once
	decoded     [][]byte
	leadSpace   []uint8

	cacheMu    sync.Mutex
	cacheVocab int
	cache      map[jsonValueCursor][]uint64
}

const (
	maxJSONValueDepth         = 32
	maxJSONValueWhitespaceRun = 32
	// maxJSONValueMaskCacheStates bounds one request's cached admission sets.
	maxJSONValueMaskCacheStates = 128
)

// CompileJSONObject returns a mask for one JSON object over dec's vocabulary.
// The grammar is fixed, so only the decoder and EOS vary between masks.
func CompileJSONObject(dec TokenDecoder, opts CompileOptions) (*JSONValueMask, error) {
	if dec == nil {
		return nil, errors.New("grammar: CompileJSONObject needs a TokenDecoder (inject tokenizer TokenBytes)")
	}
	return &JSONValueMask{dec: dec, eos: opts.EOS}, nil
}

const (
	jvTop          = iota // before the object: only '{'
	jvObjectFirst         // after '{': key or '}'
	jvKey                 // after ',' in an object: key
	jvColon               // after a key
	jvValue               // after ':' or ',' in an array
	jvArrayFirst          // after '[': value or ']'
	jvAfterValue          // ',' or the open container's closing bracket
	jvString              // string body; key marks an object key
	jvEscape              // after '\'
	jvHex                 // inside \uXXXX; aux hex digits read
	jvUTF8                // inside a multi-byte rune; aux bytes left, next in [lo, hi]
	jvLiteral             // inside true, false or null; aux bytes read
	jvNumMinus            // after '-'
	jvNumZero             // integer part "0"
	jvNumInt              // integer part [1-9][0-9]*
	jvNumDot              // after '.'
	jvNumFrac             // fraction digits
	jvNumExp              // after 'e' or 'E'
	jvNumExpSign          // after the exponent sign
	jvNumExpDigits        // exponent digits
	jvDone                // the object is closed
)

var jsonValueLiterals = [...]string{"true", "false", "null"}

// jsonValueCursor is the whole parse state, O(1) per byte. Every field but ws
// is the cache key: the open containers fit in stack and depth, and a string
// body or digit run is one state however long it grows. Unused fields stay
// zero so equal states compare equal.
type jsonValueCursor struct {
	stack uint32 // bit d set: the container at depth d (0 is the object) is an array
	depth uint8
	lex   uint8
	aux   uint8
	lit   uint8
	lo    byte
	hi    byte
	key   bool
	ws    uint8
}

// step advances c by b and reports whether the document is still viable.
// Every viable state can still be completed, so a live cursor never strands
// the response.
func (c *jsonValueCursor) step(b byte) bool {
	switch c.lex {
	case jvTop:
		return b == '{' && c.open(false)
	case jvDone:
		return false
	case jvString:
		switch {
		case b == '"':
			c.lex = jvAfterValue
			if c.key {
				c.lex, c.key = jvColon, false
			}
		case b == '\\':
			c.lex = jvEscape
		case b >= 0x80:
			return c.startRune(b)
		case b < 0x20:
			return false
		}
		return true
	case jvEscape:
		switch b {
		case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
			c.lex = jvString
		case 'u':
			c.lex = jvHex
		default:
			return false
		}
		return true
	case jvHex:
		if !(b >= '0' && b <= '9' || b >= 'a' && b <= 'f' || b >= 'A' && b <= 'F') {
			return false
		}
		if c.aux++; c.aux == 4 {
			c.lex, c.aux = jvString, 0
		}
		return true
	case jvUTF8:
		if b < c.lo || b > c.hi {
			return false
		}
		c.lo, c.hi = 0x80, 0xBF
		if c.aux--; c.aux == 0 {
			c.lex, c.lo, c.hi = jvString, 0, 0
		}
		return true
	case jvLiteral:
		lit := jsonValueLiterals[c.lit]
		if lit[c.aux] != b {
			return false
		}
		if c.aux++; int(c.aux) == len(lit) {
			c.lex, c.aux, c.lit = jvAfterValue, 0, 0
		}
		return true
	case jvNumMinus:
		switch {
		case b == '0':
			c.lex = jvNumZero
		case b >= '1' && b <= '9':
			c.lex = jvNumInt
		default:
			return false
		}
		return true
	case jvNumZero, jvNumInt, jvNumFrac:
		switch {
		case b >= '0' && b <= '9' && c.lex != jvNumZero:
			return true
		case b == '.' && c.lex != jvNumFrac:
			c.lex = jvNumDot
			return true
		case b == 'e' || b == 'E':
			c.lex = jvNumExp
			return true
		}
		// The number can end here; b belongs to what follows it.
		c.lex = jvAfterValue
	case jvNumDot:
		if b < '0' || b > '9' {
			return false
		}
		c.lex = jvNumFrac
		return true
	case jvNumExp, jvNumExpSign:
		if c.lex == jvNumExp && (b == '+' || b == '-') {
			c.lex = jvNumExpSign
			return true
		}
		if b < '0' || b > '9' {
			return false
		}
		c.lex = jvNumExpDigits
		return true
	case jvNumExpDigits:
		if b >= '0' && b <= '9' {
			return true
		}
		c.lex = jvAfterValue
	}

	if b == ' ' || b == '\t' || b == '\n' || b == '\r' {
		if c.ws == maxJSONValueWhitespaceRun {
			return false
		}
		c.ws++
		return true
	}
	c.ws = 0
	switch c.lex {
	case jvObjectFirst, jvKey:
		if b == '}' && c.lex == jvObjectFirst {
			c.close()
			return true
		}
		if b != '"' {
			return false
		}
		c.lex, c.key = jvString, true
		return true
	case jvColon:
		if b != ':' {
			return false
		}
		c.lex = jvValue
		return true
	case jvArrayFirst, jvValue:
		if b == ']' && c.lex == jvArrayFirst {
			c.close()
			return true
		}
		return c.startValue(b)
	default: // jvAfterValue
		array := c.stack>>(c.depth-1)&1 == 1
		switch {
		case b == ',' && array:
			c.lex = jvValue
		case b == ',':
			c.lex = jvKey
		case b == ']' && array, b == '}' && !array:
			c.close()
		default:
			return false
		}
		return true
	}
}

func (c *jsonValueCursor) startValue(b byte) bool {
	switch {
	case b == '{':
		return c.open(false)
	case b == '[':
		return c.open(true)
	case b == '"':
		c.lex = jvString
	case b == '-':
		c.lex = jvNumMinus
	case b == '0':
		c.lex = jvNumZero
	case b >= '1' && b <= '9':
		c.lex = jvNumInt
	case b == 't':
		c.lex, c.aux, c.lit = jvLiteral, 1, 0
	case b == 'f':
		c.lex, c.aux, c.lit = jvLiteral, 1, 1
	case b == 'n':
		c.lex, c.aux, c.lit = jvLiteral, 1, 2
	default:
		return false
	}
	return true
}

func (c *jsonValueCursor) open(array bool) bool {
	if c.depth == maxJSONValueDepth {
		return false
	}
	if array {
		c.stack |= 1 << c.depth
		c.lex = jvArrayFirst
	} else {
		c.lex = jvObjectFirst
	}
	c.depth++
	return true
}

func (c *jsonValueCursor) close() {
	c.depth--
	c.stack &^= 1 << c.depth
	c.lex = jvAfterValue
	if c.depth == 0 {
		c.lex = jvDone
	}
}

// startRune enters a multi-byte UTF-8 rune at its lead byte. The first
// continuation byte's range excludes overlong forms, surrogates and code
// points past U+10FFFF.
func (c *jsonValueCursor) startRune(b byte) bool {
	c.lex, c.lo, c.hi = jvUTF8, 0x80, 0xBF
	switch {
	case b >= 0xC2 && b <= 0xDF:
		c.aux = 1
	case b == 0xE0:
		c.aux, c.lo = 2, 0xA0
	case b == 0xED:
		c.aux, c.hi = 2, 0x9F
	case b >= 0xE1 && b <= 0xEF:
		c.aux = 2
	case b == 0xF0:
		c.aux, c.lo = 3, 0x90
	case b == 0xF4:
		c.aux, c.hi = 3, 0x8F
	case b >= 0xF1 && b <= 0xF3:
		c.aux = 3
	default:
		return false
	}
	return true
}

// MaskLogits implements model.LogitMask structurally. EOS stays finite only
// once the object is closed; every other token must step all of its bytes
// through viable states. Tokens decoding to no bytes and dead-end histories
// are masked.
func (m *JSONValueMask) MaskLogits(history []int, logits []float32) {
	if m == nil || m.dec == nil || len(logits) == 0 {
		return
	}
	m.cacheDecoded(len(logits))
	state, ok := m.stateAfter(history)
	if !ok {
		maskAll(logits)
		return
	}
	// A whitespace run only shortens what may follow: a token is admitted iff
	// it is admitted at the run's start and its own leading whitespace still
	// fits. So the run length stays out of the cache key.
	run := state.ws
	state.ws = 0
	admitted := m.admittedTokens(state, len(logits))
	neg := float32(math.Inf(-1))
	eos := m.eos >= 0 && m.eos < len(logits)
	var eosLogit float32
	if eos {
		eosLogit = logits[m.eos]
	}
	for word, bits := range admitted {
		if bits == math.MaxUint64 {
			continue
		}
		for id := word << 6; id < min(word<<6+64, len(logits)); id++ {
			if bits&(1<<(uint(id)&63)) == 0 {
				logits[id] = neg
			}
		}
	}
	if run > 0 {
		room := uint8(maxJSONValueWhitespaceRun) - run
		for id, lead := range m.leadSpace[:min(len(m.leadSpace), len(logits))] {
			if lead > room {
				logits[id] = neg
			}
		}
	}
	if eos {
		logits[m.eos] = neg
		if state.lex == jvDone {
			logits[m.eos] = eosLogit
		}
	}
}

func (m *JSONValueMask) cacheDecoded(vocabSize int) {
	m.decodedOnce.Do(func() {
		m.decoded = make([][]byte, vocabSize)
		m.leadSpace = make([]uint8, vocabSize)
		for id := range m.decoded {
			piece := append([]byte(nil), m.dec.TokenBytes(id)...)
			m.decoded[id] = piece
			n := 0
			for n < len(piece) && n < math.MaxUint8 && (piece[n] == ' ' || piece[n] == '\t' || piece[n] == '\n' || piece[n] == '\r') {
				n++
			}
			m.leadSpace[id] = uint8(n)
		}
	})
}

func (m *JSONValueMask) tokenBytes(id int) []byte {
	if id >= 0 && id < len(m.decoded) {
		return m.decoded[id]
	}
	return m.dec.TokenBytes(id)
}

func (m *JSONValueMask) stateAfter(history []int) (jsonValueCursor, bool) {
	var c jsonValueCursor
	for _, id := range history {
		piece := m.tokenBytes(id)
		if len(piece) == 0 {
			return c, false
		}
		for _, b := range piece {
			if !c.step(b) {
				return c, false
			}
		}
	}
	return c, true
}

// admittedTokens returns the bitset of non-empty tokens whose every byte keeps
// the document viable from state, cached per state and vocabulary size.
func (m *JSONValueMask) admittedTokens(state jsonValueCursor, vocabSize int) []uint64 {
	m.cacheMu.Lock()
	if m.cacheVocab == vocabSize {
		if bits, ok := m.cache[state]; ok {
			m.cacheMu.Unlock()
			return bits
		}
	}
	m.cacheMu.Unlock()

	var firstOK [256]bool
	for b := range firstOK {
		next := state
		firstOK[b] = next.step(byte(b))
	}
	bits := make([]uint64, (vocabSize+63)/64)
	for id := range vocabSize {
		piece := m.tokenBytes(id)
		if len(piece) == 0 || !firstOK[piece[0]] {
			continue
		}
		next, ok := state, true
		for _, b := range piece {
			if ok = next.step(b); !ok {
				break
			}
		}
		if ok {
			bits[id>>6] |= 1 << (uint(id) & 63)
		}
	}

	m.cacheMu.Lock()
	if m.cacheVocab != vocabSize || m.cache == nil || len(m.cache) >= maxJSONValueMaskCacheStates {
		m.cache, m.cacheVocab = map[jsonValueCursor][]uint64{}, vocabSize
	}
	m.cache[state] = bits
	m.cacheMu.Unlock()
	return bits
}
