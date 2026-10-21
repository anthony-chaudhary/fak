package grammar

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"regexp/syntax"
	"slices"
	"sync"
	"unicode"
	"unicode/utf8"
)

// RegexMask constrains generated tokens so the whole response fully matches one
// regular expression. The pattern is compiled ahead of decode into a byte-level
// DFA over UTF-8: Go's rune-level program is expanded into byte-range chains, so
// a token may end inside a multi-byte rune and only valid UTF-8 is ever admitted.
// States that cannot reach a match are pruned, so every admitted token keeps a
// completion reachable. Patterns outside the bounded surface are refused with
// ErrUnsupportedSchema.
//
// Adapted from DeepInfer structured-output regex dispatch and full-match regex
// semantics (Apache-2.0, deepseek-ai/DeepInfer deepinfer/structure_output/compiler.py
// at 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01); no upstream source is copied.
type RegexMask struct {
	dec        TokenDecoder
	eos        int
	classOf    [256]uint8
	numClasses int
	// trans holds numClasses next states per state; -1 is dead. State 0 is the start.
	trans  []int32
	accept []bool

	decodedOnce sync.Once
	decoded     [][]byte

	cacheMu    sync.Mutex
	cacheVocab int
	cache      map[int32][]uint64
}

const (
	maxRegexPatternBytes = 4096
	// maxRegexProgramSize bounds the repeat-expanded syntax tree and the compiled
	// rune program; maxRegexNFASize bounds the byte NFA's nodes plus edges.
	maxRegexProgramSize = 8192
	maxRegexNFASize     = 32768
	maxRegexDFAStates   = 4096
	maxRegexStateNodes  = 1024
	// maxRegexMaskCacheStates bounds one request's cached admission sets.
	maxRegexMaskCacheStates = 64
)

// CompileRegex compiles a Perl-syntax pattern (Go regexp/syntax) with
// full-match semantics: the generated text must match from its first byte to
// its last. Text anchors (^ $ \A \z) are honored at those two positions; line
// anchors, word boundaries, and anything regexp/syntax cannot parse
// (backreferences, lookaround) are refused.
func CompileRegex(pattern string, dec TokenDecoder, opts CompileOptions) (*RegexMask, error) {
	if dec == nil {
		return nil, errors.New("grammar: CompileRegex needs a TokenDecoder (inject tokenizer TokenBytes)")
	}
	if len(pattern) > maxRegexPatternBytes {
		return nil, regexUnsupported("pattern size %d exceeds limit %d", len(pattern), maxRegexPatternBytes)
	}
	re, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return nil, regexUnsupported("pattern does not parse: %v", err)
	}
	if size := regexExpandedSize(re); size > maxRegexProgramSize {
		return nil, regexUnsupported("expanded pattern exceeds %d nodes", maxRegexProgramSize)
	}
	prog, err := syntax.Compile(re.Simplify())
	if err != nil {
		return nil, regexUnsupported("pattern does not compile: %v", err)
	}
	if len(prog.Inst) > maxRegexProgramSize {
		return nil, regexUnsupported("program size %d exceeds limit %d", len(prog.Inst), maxRegexProgramSize)
	}
	nfa, err := newRegexNFA(prog)
	if err != nil {
		return nil, err
	}
	m := &RegexMask{dec: dec, eos: opts.EOS}
	if err := m.determinize(nfa, int32(prog.Start)); err != nil {
		return nil, err
	}
	return m, nil
}

func regexUnsupported(format string, a ...any) error {
	return fmt.Errorf("grammar: regex: %s: %w", fmt.Sprintf(format, a...), ErrUnsupportedSchema)
}

// regexExpandedSize estimates the syntax tree size after repeat expansion,
// saturating just past the limit so nested counted repeats are refused before
// Simplify materializes them.
func regexExpandedSize(re *syntax.Regexp) int {
	size := 1
	switch re.Op {
	case syntax.OpLiteral:
		size += len(re.Rune)
	case syntax.OpRepeat:
		count := re.Max
		if count < 0 {
			count = re.Min + 1
		}
		size += max(count, 1) * regexExpandedSize(re.Sub[0])
	default:
		for _, sub := range re.Sub {
			size += regexExpandedSize(sub)
		}
	}
	return min(size, maxRegexProgramSize+1)
}

type regexEdge struct {
	lo, hi byte
	to     int32
}

// regexNode is one byte-NFA node. Program instructions keep their pc as node
// id; UTF-8 continuation chains are appended after them.
type regexNode struct {
	eps   []int32
	cond  syntax.EmptyOp
	edges []regexEdge
	match bool
}

type regexNFA struct {
	nodes  []regexNode
	size   int
	shared map[regexEdge]int32
}

func newRegexNFA(prog *syntax.Prog) (*regexNFA, error) {
	n := &regexNFA{nodes: make([]regexNode, len(prog.Inst)), size: len(prog.Inst), shared: map[regexEdge]int32{}}
	for pc := range prog.Inst {
		inst := &prog.Inst[pc]
		node := &n.nodes[pc]
		switch inst.Op {
		case syntax.InstAlt, syntax.InstAltMatch:
			node.eps = []int32{int32(inst.Out), int32(inst.Arg)}
		case syntax.InstCapture, syntax.InstNop:
			node.eps = []int32{int32(inst.Out)}
		case syntax.InstEmptyWidth:
			cond := syntax.EmptyOp(inst.Arg)
			if cond&^(syntax.EmptyBeginText|syntax.EmptyEndText) != 0 {
				return nil, regexUnsupported("assertion %s", regexAssertionName(cond))
			}
			node.cond, node.eps = cond, []int32{int32(inst.Out)}
		case syntax.InstMatch:
			node.match = true
		case syntax.InstFail:
		case syntax.InstRune, syntax.InstRune1, syntax.InstRuneAny, syntax.InstRuneAnyNotNL:
			for _, r := range regexInstRanges(inst) {
				for _, seq := range appendUTF8Sequences(nil, r[0], r[1]) {
					if err := n.addSequence(pc, seq, int32(inst.Out)); err != nil {
						return nil, err
					}
				}
			}
		default:
			return nil, regexUnsupported("program instruction %v", inst.Op)
		}
	}
	return n, nil
}

func regexAssertionName(cond syntax.EmptyOp) string {
	switch {
	case cond&(syntax.EmptyBeginLine|syntax.EmptyEndLine) != 0:
		return "multi-line ^ or $"
	case cond&syntax.EmptyWordBoundary != 0:
		return `\b`
	case cond&syntax.EmptyNoWordBoundary != 0:
		return `\B`
	}
	return fmt.Sprintf("%#x", uint8(cond))
}

// regexInstRanges returns the inclusive rune ranges one rune instruction
// matches, mirroring syntax.Inst.MatchRune.
func regexInstRanges(inst *syntax.Inst) [][2]rune {
	switch inst.Op {
	case syntax.InstRune1:
		return [][2]rune{{inst.Rune[0], inst.Rune[0]}}
	case syntax.InstRuneAny:
		return [][2]rune{{0, unicode.MaxRune}}
	case syntax.InstRuneAnyNotNL:
		return [][2]rune{{0, '\n' - 1}, {'\n' + 1, unicode.MaxRune}}
	}
	if len(inst.Rune) == 1 {
		r := inst.Rune[0]
		out := [][2]rune{{r, r}}
		if syntax.Flags(inst.Arg)&syntax.FoldCase != 0 {
			for f := unicode.SimpleFold(r); f != r; f = unicode.SimpleFold(f) {
				out = append(out, [2]rune{f, f})
			}
		}
		return out
	}
	out := make([][2]rune, 0, len(inst.Rune)/2)
	for i := 0; i+1 < len(inst.Rune); i += 2 {
		out = append(out, [2]rune{inst.Rune[i], inst.Rune[i+1]})
	}
	return out
}

// addSequence links pc to out through one UTF-8 byte-range sequence. Chain
// nodes are shared by (range, target), so the continuation suffixes of
// different sequences into the same target collapse to one node each.
func (n *regexNFA) addSequence(pc int, seq []regexEdge, out int32) error {
	target := out
	for i := len(seq) - 1; i >= 1; i-- {
		key := regexEdge{lo: seq[i].lo, hi: seq[i].hi, to: target}
		id, ok := n.shared[key]
		if !ok {
			id = int32(len(n.nodes))
			n.nodes = append(n.nodes, regexNode{edges: []regexEdge{key}})
			n.shared[key] = id
			n.size += 2
		}
		target = id
	}
	n.nodes[pc].edges = append(n.nodes[pc].edges, regexEdge{lo: seq[0].lo, hi: seq[0].hi, to: target})
	n.size++
	if n.size > maxRegexNFASize {
		return regexUnsupported("byte automaton exceeds %d nodes and edges", maxRegexNFASize)
	}
	return nil
}

// appendUTF8Sequences splits the valid scalar values in [lo, hi] into byte-range
// sequences whose cross product is exactly their UTF-8 encodings: the range is
// split at encoded-length boundaries and then wherever lo and hi disagree above
// a continuation-byte boundary, so every byte position spans a contiguous range.
func appendUTF8Sequences(dst [][]regexEdge, lo, hi rune) [][]regexEdge {
	lo, hi = max(lo, 0), min(hi, unicode.MaxRune)
	if lo > hi {
		return dst
	}
	if lo <= 0xDFFF && hi >= 0xD800 {
		dst = appendUTF8Sequences(dst, lo, 0xD7FF)
		return appendUTF8Sequences(dst, 0xE000, hi)
	}
	for _, edge := range []rune{0x7F, 0x7FF, 0xFFFF} {
		if lo <= edge && hi > edge {
			dst = appendUTF8Sequences(dst, lo, edge)
			return appendUTF8Sequences(dst, edge+1, hi)
		}
	}
	if hi <= utf8.RuneSelf-1 {
		return append(dst, []regexEdge{{lo: byte(lo), hi: byte(hi)}})
	}
	for i := 1; i < utf8.UTFMax; i++ {
		low := rune(1)<<(6*i) - 1
		if lo&^low == hi&^low {
			continue
		}
		if lo&low != 0 {
			dst = appendUTF8Sequences(dst, lo, lo|low)
			return appendUTF8Sequences(dst, (lo|low)+1, hi)
		}
		if hi&low != low {
			dst = appendUTF8Sequences(dst, lo, (hi&^low)-1)
			return appendUTF8Sequences(dst, hi&^low, hi)
		}
	}
	var a, b [utf8.UTFMax]byte
	n := utf8.EncodeRune(a[:], lo)
	utf8.EncodeRune(b[:], hi)
	seq := make([]regexEdge, n)
	for i := range seq {
		seq[i] = regexEdge{lo: a[i], hi: b[i]}
	}
	return append(dst, seq)
}

// regexDeterminizer runs the subset construction over byte classes.
type regexDeterminizer struct {
	nfa     *regexNFA
	mark    []uint32
	gen     uint32
	stack   []int32
	ids     map[string]int32
	sets    [][]int32
	buckets [][]int32
}

// closure follows epsilon edges from seeds, crossing an empty-width gate only
// when allowed holds it. It returns the sorted nodes that can still act: byte
// nodes, matches, and end-of-text gates awaiting the end.
func (d *regexDeterminizer) closure(seeds []int32, allowed syntax.EmptyOp) []int32 {
	d.gen++
	d.stack = append(d.stack[:0], seeds...)
	var out []int32
	for len(d.stack) > 0 {
		id := d.stack[len(d.stack)-1]
		d.stack = d.stack[:len(d.stack)-1]
		if d.mark[id] == d.gen {
			continue
		}
		d.mark[id] = d.gen
		node := &d.nfa.nodes[id]
		switch {
		case node.match || len(node.edges) > 0:
			out = append(out, id)
		case node.cond != 0 && node.cond&^allowed != 0:
			if node.cond == syntax.EmptyEndText {
				out = append(out, id)
			}
		default:
			d.stack = append(d.stack, node.eps...)
		}
	}
	slices.Sort(out)
	return out
}

func (d *regexDeterminizer) accepts(set []int32, allowed syntax.EmptyOp) bool {
	for _, id := range d.closure(set, allowed|syntax.EmptyEndText) {
		if d.nfa.nodes[id].match {
			return true
		}
	}
	return false
}

func regexSetKey(set []int32) string {
	key := make([]byte, 4*len(set))
	for i, id := range set {
		binary.LittleEndian.PutUint32(key[4*i:], uint32(id))
	}
	return string(key)
}

func (m *RegexMask) determinize(nfa *regexNFA, start int32) error {
	var boundary [257]bool
	boundary[0] = true
	for _, node := range nfa.nodes {
		for _, e := range node.edges {
			boundary[e.lo], boundary[int(e.hi)+1] = true, true
		}
	}
	m.numClasses = 0
	for b := range 256 {
		if boundary[b] && b > 0 {
			m.numClasses++
		}
		m.classOf[b] = uint8(m.numClasses)
	}
	m.numClasses++

	d := &regexDeterminizer{nfa: nfa, mark: make([]uint32, len(nfa.nodes)), ids: map[string]int32{}, buckets: make([][]int32, m.numClasses)}
	startSet := d.closure([]int32{start}, syntax.EmptyBeginText)
	// The start state is never shared: only it may still cross a begin-text gate
	// when testing acceptance.
	d.sets = append(d.sets, startSet)
	m.accept = append(m.accept, d.accepts(startSet, syntax.EmptyBeginText))
	for state := 0; state < len(d.sets); state++ {
		set := d.sets[state]
		if len(set) > maxRegexStateNodes {
			return regexUnsupported("a state holds %d nodes, over the limit %d", len(set), maxRegexStateNodes)
		}
		for c := range d.buckets {
			d.buckets[c] = d.buckets[c][:0]
		}
		for _, id := range set {
			for _, e := range nfa.nodes[id].edges {
				for c := m.classOf[e.lo]; ; c++ {
					d.buckets[c] = append(d.buckets[c], e.to)
					if c == m.classOf[e.hi] {
						break
					}
				}
			}
		}
		row := make([]int32, m.numClasses)
		for c := range row {
			row[c] = -1
			if len(d.buckets[c]) == 0 {
				continue
			}
			next := d.closure(d.buckets[c], 0)
			if len(next) == 0 {
				continue
			}
			key := regexSetKey(next)
			id, ok := d.ids[key]
			if !ok {
				if len(d.sets) >= maxRegexDFAStates {
					return regexUnsupported("automaton exceeds %d states", maxRegexDFAStates)
				}
				id = int32(len(d.sets))
				d.ids[key] = id
				d.sets = append(d.sets, next)
				m.accept = append(m.accept, d.accepts(next, 0))
			}
			row[c] = id
		}
		m.trans = append(m.trans, row...)
	}
	return m.pruneDead()
}

// pruneDead removes transitions into states that cannot reach an accepting
// state, so the mask never admits a token that strands the response.
func (m *RegexMask) pruneDead() error {
	states := len(m.accept)
	reverse := make([][]int32, states)
	for s := range states {
		for _, next := range m.trans[s*m.numClasses : (s+1)*m.numClasses] {
			if next >= 0 {
				reverse[next] = append(reverse[next], int32(s))
			}
		}
	}
	live := make([]bool, states)
	var queue []int32
	for s, ok := range m.accept {
		if ok {
			live[s] = true
			queue = append(queue, int32(s))
		}
	}
	for len(queue) > 0 {
		s := queue[0]
		queue = queue[1:]
		for _, prev := range reverse[s] {
			if !live[prev] {
				live[prev] = true
				queue = append(queue, prev)
			}
		}
	}
	if !live[0] {
		return regexUnsupported("pattern matches no text")
	}
	for i, next := range m.trans {
		if next >= 0 && !live[next] {
			m.trans[i] = -1
		}
	}
	return nil
}

// MaskLogits implements model.LogitMask structurally. EOS stays finite only in
// an accepting state; every other token must step every byte through live
// states. A history the automaton rejects masks the whole vocabulary.
func (m *RegexMask) MaskLogits(history []int, logits []float32) {
	if m == nil || m.dec == nil || len(logits) == 0 {
		return
	}
	m.cacheDecoded(len(logits))
	state, ok := m.stateAfter(history)
	if !ok {
		maskAll(logits)
		return
	}
	admitted := m.admittedTokens(state, len(logits))
	neg := float32(math.Inf(-1))
	for id := range logits {
		if id == m.eos {
			if !m.accept[state] {
				logits[id] = neg
			}
			continue
		}
		if admitted[id>>6]&(1<<(uint(id)&63)) == 0 {
			logits[id] = neg
		}
	}
}

func (m *RegexMask) cacheDecoded(vocabSize int) {
	m.decodedOnce.Do(func() {
		m.decoded = make([][]byte, vocabSize)
		for id := range m.decoded {
			m.decoded[id] = append([]byte(nil), m.dec.TokenBytes(id)...)
		}
	})
}

func (m *RegexMask) tokenBytes(id int) []byte {
	if id >= 0 && id < len(m.decoded) {
		return m.decoded[id]
	}
	return m.dec.TokenBytes(id)
}

func (m *RegexMask) step(state int32, piece []byte) int32 {
	for _, b := range piece {
		state = m.trans[int(state)*m.numClasses+int(m.classOf[b])]
		if state < 0 {
			return -1
		}
	}
	return state
}

func (m *RegexMask) stateAfter(history []int) (int32, bool) {
	state := int32(0)
	for _, id := range history {
		piece := m.tokenBytes(id)
		if len(piece) == 0 {
			return -1, false
		}
		if state = m.step(state, piece); state < 0 {
			return -1, false
		}
	}
	return state, true
}

// admittedTokens returns the bitset of non-empty tokens whose bytes all step
// through live states from state, cached per state and vocabulary size.
func (m *RegexMask) admittedTokens(state int32, vocabSize int) []uint64 {
	m.cacheMu.Lock()
	if m.cacheVocab == vocabSize {
		if bits, ok := m.cache[state]; ok {
			m.cacheMu.Unlock()
			return bits
		}
	}
	m.cacheMu.Unlock()

	bits := make([]uint64, (vocabSize+63)/64)
	for id := range vocabSize {
		if piece := m.tokenBytes(id); len(piece) > 0 && m.step(state, piece) >= 0 {
			bits[id>>6] |= 1 << (uint(id) & 63)
		}
	}

	m.cacheMu.Lock()
	if m.cacheVocab != vocabSize || m.cache == nil || len(m.cache) >= maxRegexMaskCacheStates {
		m.cache, m.cacheVocab = map[int32][]uint64{}, vocabSize
	}
	m.cache[state] = bits
	m.cacheMu.Unlock()
	return bits
}
