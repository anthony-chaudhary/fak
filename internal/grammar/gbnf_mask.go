// GBNF pushdown matching and partial-codepoint admission are adapted from
// llama.cpp cd8427d018c07eefab6c110f7f115c31ec4c6157 (MIT).
// See LLAMA-GBNF-LICENSE.txt and gbnf.README.md.
package grammar

import (
	"encoding/binary"
	"math"
	"slices"
	"sync"
	"unicode/utf8"
)

const (
	gbnfMaxStacks        = 256
	gbnfMaxStackDepth    = 256
	gbnfMaxClosureStates = 1024
	gbnfMaxVocabulary    = 262144
	gbnfMaxTokenBytes    = 4096
	gbnfMaxDecodedBytes  = 16 * 1024 * 1024
	gbnfMaxHistoryTokens = 32768
	gbnfMaxOutputBytes   = 256 * 1024
)

// GBNFMask is a bounded character-rule GBNF pushdown matcher. Its program is
// immutable; each call derives its cursor from the complete generated history.
// A synchronized one-history cache accelerates append-only decode and is safe
// when native speculative forks share this mask but use different histories.
// TokenDecoder must be immutable for the lifetime of the mask.
type GBNFMask struct {
	program *gbnfProgram
	decoder TokenDecoder
	eos     int
	initial gbnfState

	mu            sync.Mutex
	decoded       [][]byte
	decodedReady  bool
	cachedHistory []int
	cachedState   gbnfState
	cachedBytes   int
}

type gbnfState struct {
	stacks     [][]gbnfSymbol
	pending    [utf8.UTFMax]byte
	pendingLen int
}

func gbnfRefusal(reason string) error { return &GBNFError{Reason: reason, Offset: 0} }

// CompileGBNF accepts the documented bounded character-rule subset with a
// required root rule. Token-ID extensions, lazy triggers and other grammar
// dialects are refused. EOS is matched by ID, never by its decoded spelling.
func CompileGBNF(source string, dec TokenDecoder, opts CompileOptions) (*GBNFMask, error) {
	if dec == nil {
		return nil, gbnfRefusal("missing_token_decoder")
	}
	if opts.EOS < 0 || opts.EOS >= gbnfMaxVocabulary {
		return nil, gbnfRefusal("invalid_eos")
	}
	program, err := parseGBNF(source)
	if err != nil {
		return nil, err
	}
	work := &gbnfWork{left: gbnfMaxWork}
	stacks, err := program.closure([][]gbnfSymbol{{{ref: program.root, class: -1}}}, work)
	if err != nil {
		return nil, err
	}
	initial := gbnfState{stacks: stacks}
	return &GBNFMask{program: program, decoder: dec, eos: opts.EOS, initial: initial, cachedState: initial}, nil
}

func gbnfStackKey(stack []gbnfSymbol) string {
	buf := make([]byte, 4*len(stack))
	for i, s := range stack {
		id := uint32(s.ref) << 1
		if s.ref < 0 {
			id = uint32(s.class)<<1 | 1
		}
		binary.LittleEndian.PutUint32(buf[4*i:], id)
	}
	return string(buf)
}

// closure expands only the top reference. Terminals remain on the stack;
// empty stacks are accepting. The parser excludes nullable-prefix cycles, and
// these independent bounds also refuse exponential ambiguity before allocation.
func (p *gbnfProgram) closure(seeds [][]gbnfSymbol, work *gbnfWork) ([][]gbnfSymbol, error) {
	seen := make(map[string]bool)
	queue := make([][]gbnfSymbol, 0, min(len(seeds), gbnfMaxClosureStates))
	add := func(stack []gbnfSymbol) error {
		if len(stack) > gbnfMaxStackDepth {
			return gbnfRefusal("stack_depth_limit")
		}
		if err := work.spend(len(stack) + 1); err != nil {
			return err
		}
		key := gbnfStackKey(stack)
		if seen[key] {
			return nil
		}
		if len(seen) >= gbnfMaxClosureStates {
			return gbnfRefusal("closure_state_limit")
		}
		seen[key] = true
		queue = append(queue, stack)
		return nil
	}
	for _, s := range seeds {
		if err := add(s); err != nil {
			return nil, err
		}
	}
	var out [][]gbnfSymbol
	for at := 0; at < len(queue); at++ {
		stack := queue[at]
		if len(stack) == 0 || stack[len(stack)-1].ref < 0 {
			if len(out) >= gbnfMaxStacks {
				return nil, gbnfRefusal("active_stack_limit")
			}
			out = append(out, stack)
			continue
		}
		top := stack[len(stack)-1]
		tail := stack[:len(stack)-1]
		for _, seq := range p.rules[top.ref].alternatives {
			if len(seq) > gbnfMaxStackDepth-len(tail) {
				return nil, gbnfRefusal("stack_depth_limit")
			}
			if err := work.spend(len(tail) + len(seq) + 1); err != nil {
				return nil, err
			}
			next := make([]gbnfSymbol, len(tail), len(tail)+len(seq))
			copy(next, tail)
			for i := len(seq) - 1; i >= 0; i-- {
				next = append(next, seq[i])
			}
			if err := add(next); err != nil {
				return nil, err
			}
		}
	}
	return out, nil
}

func (p *gbnfProgram) classOverlaps(id int, lo, hi rune, work *gbnfWork) (bool, error) {
	for _, r := range p.classes[id] {
		if err := work.spend(1); err != nil {
			return false, err
		}
		if r.lo > hi {
			return false, nil
		}
		if r.hi >= lo {
			return true, nil
		}
	}
	return false, nil
}
func (p *gbnfProgram) acceptRune(state gbnfState, r rune, work *gbnfWork) (gbnfState, error) {
	seeds := make([][]gbnfSymbol, 0, len(state.stacks))
	for _, stack := range state.stacks {
		if err := work.spend(1); err != nil {
			return gbnfState{}, err
		}
		if len(stack) == 0 {
			continue
		}
		ok, err := p.classOverlaps(stack[len(stack)-1].class, r, r, work)
		if err != nil {
			return gbnfState{}, err
		}
		if ok {
			seeds = append(seeds, stack[:len(stack)-1])
		}
	}
	stacks, err := p.closure(seeds, work)
	return gbnfState{stacks: stacks}, err
}

// gbnfUTF8Prefix returns the exact scalar interval represented by a legal
// nonempty UTF-8 prefix. It rejects overlong encodings, surrogate encodings,
// stray continuations and values above U+10FFFF before admitting a token.
func gbnfUTF8Prefix(buf []byte) (lo, hi rune, complete, ok bool) {
	if len(buf) == 0 || len(buf) > utf8.UTFMax {
		return
	}
	first := buf[0]
	need := 0
	value := rune(0)
	switch {
	case first < 0x80:
		need = 1
		value = rune(first)
	case first >= 0xC2 && first <= 0xDF:
		need = 2
		value = rune(first & 0x1F)
	case first >= 0xE0 && first <= 0xEF:
		need = 3
		value = rune(first & 0x0F)
	case first >= 0xF0 && first <= 0xF4:
		need = 4
		value = rune(first & 0x07)
	default:
		return
	}
	if len(buf) > need {
		return
	}
	for i := 1; i < len(buf); i++ {
		b := buf[i]
		if b < 0x80 || b > 0xBF {
			return
		}
		if i == 1 && ((first == 0xE0 && b < 0xA0) || (first == 0xED && b > 0x9F) || (first == 0xF0 && b < 0x90) || (first == 0xF4 && b > 0x8F)) {
			return
		}
		value = value<<6 | rune(b&0x3F)
	}
	lo, hi = value, value
	for i := len(buf); i < need; i++ {
		a, b := byte(0x80), byte(0xBF)
		if i == 1 {
			switch first {
			case 0xE0:
				a = 0xA0
			case 0xED:
				b = 0x9F
			case 0xF0:
				a = 0x90
			case 0xF4:
				b = 0x8F
			}
		}
		lo = lo<<6 | rune(a&0x3F)
		hi = hi<<6 | rune(b&0x3F)
	}
	return lo, hi, len(buf) == need, true
}
func (p *gbnfProgram) feed(state gbnfState, piece []byte, work *gbnfWork) (gbnfState, error) {
	for _, b := range piece {
		if err := work.spend(1); err != nil {
			return gbnfState{}, err
		}
		if len(state.stacks) == 0 {
			return gbnfState{}, nil
		}
		if state.pendingLen >= utf8.UTFMax {
			return gbnfState{}, nil
		}
		state.pending[state.pendingLen] = b
		state.pendingLen++
		lo, hi, complete, valid := gbnfUTF8Prefix(state.pending[:state.pendingLen])
		if !valid {
			return gbnfState{}, nil
		}
		if complete {
			var err error
			state, err = p.acceptRune(state, lo, work)
			if err != nil {
				return gbnfState{}, err
			}
			continue
		}
		kept := make([][]gbnfSymbol, 0, len(state.stacks))
		for _, stack := range state.stacks {
			if err := work.spend(1); err != nil {
				return gbnfState{}, err
			}
			if len(stack) == 0 {
				continue
			}
			ok, err := p.classOverlaps(stack[len(stack)-1].class, lo, hi, work)
			if err != nil {
				return gbnfState{}, err
			}
			if ok {
				kept = append(kept, stack)
			}
		}
		state.stacks = kept
	}
	return state, nil
}
func (s gbnfState) accepts() bool {
	if s.pendingLen != 0 {
		return false
	}
	for _, stack := range s.stacks {
		if len(stack) == 0 {
			return true
		}
	}
	return false
}
func (m *GBNFMask) prepareTokens(vocab int, work *gbnfWork) error {
	if vocab <= 0 || vocab > gbnfMaxVocabulary || m.eos >= vocab {
		return gbnfRefusal("vocabulary_limit_or_eos")
	}
	if m.decodedReady {
		if len(m.decoded) != vocab {
			return gbnfRefusal("changed_vocabulary")
		}
		return nil
	}
	decoded := make([][]byte, vocab)
	total := 0
	for id := range decoded {
		if err := work.spend(1); err != nil {
			return err
		}
		if id == m.eos {
			continue
		}
		piece := m.decoder.TokenBytes(id)
		if len(piece) > gbnfMaxTokenBytes || len(piece) > gbnfMaxDecodedBytes-total {
			return gbnfRefusal("token_byte_limit")
		}
		if err := work.spend(len(piece)); err != nil {
			return err
		}
		total += len(piece)
		decoded[id] = slices.Clone(piece)
	}
	m.decoded = decoded
	m.decodedReady = true
	return nil
}
func (m *GBNFMask) stateFor(history []int, work *gbnfWork) (gbnfState, int, error) {
	if len(history) > gbnfMaxHistoryTokens {
		return gbnfState{}, 0, gbnfRefusal("history_limit")
	}
	if err := work.spend(len(history)); err != nil {
		return gbnfState{}, 0, err
	}
	state, start, size := m.initial, 0, 0
	if len(history) >= len(m.cachedHistory) && slices.Equal(history[:len(m.cachedHistory)], m.cachedHistory) {
		state, start, size = m.cachedState, len(m.cachedHistory), m.cachedBytes
	}
	for _, id := range history[start:] {
		if id < 0 || id >= len(m.decoded) || id == m.eos {
			return gbnfState{}, 0, gbnfRefusal("invalid_history_token")
		}
		piece := m.decoded[id]
		if len(piece) == 0 {
			return gbnfState{}, 0, gbnfRefusal("empty_history_token")
		}
		if len(piece) > gbnfMaxOutputBytes-size {
			return gbnfState{}, 0, gbnfRefusal("output_byte_limit")
		}
		size += len(piece)
		var err error
		state, err = m.program.feed(state, piece, work)
		if err != nil {
			return gbnfState{}, 0, err
		}
		if len(state.stacks) == 0 {
			return gbnfState{}, 0, gbnfRefusal("history_outside_language")
		}
	}
	return state, size, nil
}

// MaskLogitsChecked is used by the native structured consumer to stop before
// selection on resource exhaustion. It never leaves an unconstrained fallback.
// The generic LogitMask adapter below also fails closed, but consumers must not
// sample an all-minus-infinity vector as an ordinary distribution.
func (m *GBNFMask) MaskLogitsChecked(history []int, logits []float32) (err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	defer func() {
		if err != nil {
			gbnfMaskAll(logits)
		}
	}()
	work := &gbnfWork{left: gbnfMaxWork}
	if err = m.prepareTokens(len(logits), work); err != nil {
		return err
	}
	state, size, err := m.stateFor(history, work)
	if err != nil {
		return err
	}
	for id := range logits {
		if err = work.spend(1); err != nil {
			return err
		}
		if math.IsInf(float64(logits[id]), -1) {
			continue
		}
		allowed := false
		if id == m.eos {
			allowed = state.accepts()
		} else if len(history) < gbnfMaxHistoryTokens {
			piece := m.decoded[id]
			if len(piece) > 0 && len(piece) <= gbnfMaxOutputBytes-size {
				next, feedErr := m.program.feed(state, piece, work)
				if feedErr != nil {
					return feedErr
				}
				allowed = len(next.stacks) > 0
			}
		}
		if !allowed {
			logits[id] = float32(math.Inf(-1))
		}
	}
	m.cachedHistory = append(m.cachedHistory[:0], history...)
	m.cachedState = state
	m.cachedBytes = size
	return nil
}
func gbnfMaskAll(logits []float32) {
	for i := range logits {
		logits[i] = float32(math.Inf(-1))
	}
}
func (m *GBNFMask) MaskLogits(history []int, logits []float32) {
	_ = m.MaskLogitsChecked(history, logits)
}
