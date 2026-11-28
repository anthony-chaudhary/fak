// Clean-room Go re-expression of the compressed token-trie + byte-level Earley
// parser state machine emitting per-step token allow-list masks for structured
// output. No upstream Rust source is copied.
package gateway

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"
	"regexp/syntax"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

var (
	// ErrUnsupportedGrammar is returned when a schema, regex, or CFG construct is
	// outside the supported bounded grammar surface.
	ErrUnsupportedGrammar = errors.New("gateway: unsupported grammar construct")
	// ErrInvalidGrammar is returned when a grammar definition is malformed or
	// produces an empty language.
	ErrInvalidGrammar = errors.New("gateway: invalid grammar")
	// ErrInvalidTokenTrie is returned when token-trie construction receives an
	// invalid vocabulary, EOS id, or decoder.
	ErrInvalidTokenTrie = errors.New("gateway: invalid token trie")
	// ErrGrammarRejectedToken is returned when CommitToken receives a token ID
	// disallowed by the current parser state.
	ErrGrammarRejectedToken = errors.New("gateway: token rejected by grammar")
	// ErrGrammarStopped is returned when advancing an engine that already
	// committed its stop/EOS token.
	ErrGrammarStopped = errors.New("gateway: grammar parser already stopped")
)

const (
	maxGrammarVocabSize    = 512 * 1024
	maxGrammarTokenBytes   = 4096
	maxGrammarSchemaBytes  = 64 * 1024
	maxGrammarSchemaDepth  = 32
	maxGrammarRules        = 4096
	maxGrammarProds        = 16384
	maxGrammarRegexBytes   = 4096
	maxGrammarRegexProgram = 8192
	maxGrammarNFASize      = 32768
	maxGrammarDFAStates    = 4096
	maxGrammarArrayBound   = 64
	maxGrammarStringBound  = 64
	maxGrammarChartBytes   = 256 * 1024
	maxGrammarChartItems   = 8192
	maxGrammarForcedBytes  = 1024
)

// TokenByteDecoder maps a vocabulary token ID to its decoded byte sequence.
// Callers may inject any tokenizer implementing TokenBytes(id int) []byte.
type TokenByteDecoder interface {
	TokenBytes(id int) []byte
}

// GrammarRecognizer is the byte-level push/pop parser contract traversed by
// TokenTrie during mask computation (adapted from toktrie::Recognizer).
type GrammarRecognizer interface {
	TryPushByte(b byte) bool
	PopBytes(n int)
	AllowEOS() bool
}

// TokenMaskBitset is a compact 64-bit word bitset over [0, vocabSize) storing
// allowed token IDs for one decoding step (adapted from toktrie::SimpleVob).
type TokenMaskBitset struct {
	vocabSize int
	words     []uint64
}

// NewTokenMaskBitset allocates an empty allow-list bitset for vocabSize tokens.
func NewTokenMaskBitset(vocabSize int) *TokenMaskBitset {
	if vocabSize < 0 {
		vocabSize = 0
	}
	numWords := (vocabSize + 63) / 64
	return &TokenMaskBitset{
		vocabSize: vocabSize,
		words:     make([]uint64, numWords),
	}
}

// VocabSize returns the vocabulary capacity covered by this bitset.
func (b *TokenMaskBitset) VocabSize() int {
	if b == nil {
		return 0
	}
	return b.vocabSize
}

// Clear resets every token bit to disallowed.
func (b *TokenMaskBitset) Clear() {
	if b == nil {
		return
	}
	clear(b.words)
}

// SetAll allows every token ID in [0, vocabSize).
func (b *TokenMaskBitset) SetAll() {
	if b == nil || b.vocabSize == 0 {
		return
	}
	for i := range b.words {
		b.words[i] = ^uint64(0)
	}
	if rem := b.vocabSize & 63; rem != 0 {
		b.words[len(b.words)-1] = (uint64(1) << rem) - 1
	}
}

// Allow marks tokenID as permitted if within [0, vocabSize).
func (b *TokenMaskBitset) Allow(tokenID int) {
	if b == nil || tokenID < 0 || tokenID >= b.vocabSize {
		return
	}
	b.words[tokenID>>6] |= uint64(1) << (tokenID & 63)
}

// Disallow marks tokenID as forbidden if within [0, vocabSize).
func (b *TokenMaskBitset) Disallow(tokenID int) {
	if b == nil || tokenID < 0 || tokenID >= b.vocabSize {
		return
	}
	b.words[tokenID>>6] &^= uint64(1) << (tokenID & 63)
}

// Contains reports whether tokenID is permitted in the bitset.
func (b *TokenMaskBitset) Contains(tokenID int) bool {
	if b == nil || tokenID < 0 || tokenID >= b.vocabSize {
		return false
	}
	return (b.words[tokenID>>6] & (uint64(1) << (tokenID & 63))) != 0
}

// Count returns the number of allowed token IDs.
func (b *TokenMaskBitset) Count() int {
	if b == nil {
		return 0
	}
	total := 0
	for _, w := range b.words {
		total += bits.OnesCount64(w)
	}
	return total
}

// AllowedIDs materializes the sorted slice of allowed token IDs.
func (b *TokenMaskBitset) AllowedIDs() []int {
	if b == nil {
		return nil
	}
	out := make([]int, 0, b.Count())
	for wi, w := range b.words {
		base := wi << 6
		for w != 0 {
			bit := bits.TrailingZeros64(w)
			out = append(out, base+bit)
			w &= w - 1
		}
	}
	return out
}

// AllowedMap materializes the allow-list as a map[int]bool compatible with
// ApplyGrammarMaskWithMonotonicity.
func (b *TokenMaskBitset) AllowedMap() map[int]bool {
	if b == nil {
		return nil
	}
	out := make(map[int]bool, b.Count())
	for wi, w := range b.words {
		base := wi << 6
		for w != 0 {
			bit := bits.TrailingZeros64(w)
			out[base+bit] = true
			w &= w - 1
		}
	}
	return out
}

// Clone returns an independent copy of the bitset.
func (b *TokenMaskBitset) Clone() *TokenMaskBitset {
	if b == nil {
		return nil
	}
	cp := &TokenMaskBitset{
		vocabSize: b.vocabSize,
		words:     make([]uint64, len(b.words)),
	}
	copy(cp.words, b.words)
	return cp
}

// Equal reports whether b and other represent the exact same vocab size and
// allowed token set.
func (b *TokenMaskBitset) Equal(other *TokenMaskBitset) bool {
	if b == nil || other == nil {
		return b == other
	}
	if b.vocabSize != other.vocabSize {
		return false
	}
	return slices.Equal(b.words, other.words)
}

// ApplyToLogits masks logits in place using the bitset while enforcing domain
// monotonicity: tokens that were already -inf stay -inf even if allowed by
// the grammar mask.
func (b *TokenMaskBitset) ApplyToLogits(logits []float32) (GrammarMaskDomainMonotonicityReceipt, error) {
	var receipt GrammarMaskDomainMonotonicityReceipt
	if b == nil {
		return receipt, fmt.Errorf("%w: nil token mask bitset", ErrInvalidTokenTrie)
	}
	n := len(logits)
	if n == 0 {
		return receipt, fmt.Errorf("logits must not be empty")
	}
	negInf := float32(math.Inf(-1))
	origInfCount := 0
	finalInfCount := 0
	for i, v := range logits {
		wasInf := math.IsInf(float64(v), -1)
		if wasInf {
			origInfCount++
			finalInfCount++
			continue
		}
		if !b.Contains(i) {
			logits[i] = negInf
			finalInfCount++
		}
	}
	receipt = GrammarMaskDomainMonotonicityReceipt{
		VocabSize:              n,
		OriginalInfiniteTokens: origInfCount,
		FinalInfiniteTokens:    finalInfCount,
		WidenedTokens:          0,
		DomainMonotonicityOK:   true,
	}
	return receipt, nil
}

type trieNode struct {
	byteVal     byte
	tokenID     int32
	dupIndex    int32
	subtreeSize int32
}

// TokenTrie is an immutable compressed pre-order prefix trie over vocabulary
// token bytes (adapted from toktrie::TokTrie). During ComputeBias, rejecting a
// byte transition skips the entire child subtree in O(1) via subtreeSize.
type TokenTrie struct {
	vocabSize   int
	eosToken    int
	maxTokenLen int
	special     map[int]bool
	nodes       []trieNode
	dupTokens   [][]int
	pieces      [][]byte
}

type trieEntry struct {
	piece []byte
	id    int
}

// NewTokenTrie builds an immutable TokenTrie from a TokenByteDecoder over token
// IDs [0, vocabSize). eosTokenID < 0 disables EOS admission; eosTokenID >= 0
// binds EOS exclusively to GrammarRecognizer.AllowEOS() and excludes it (along
// with any specialTokens) from byte-prefix matching.
func NewTokenTrie(vocabSize, eosTokenID int, dec TokenByteDecoder, specialTokens ...int) (*TokenTrie, error) {
	if dec == nil {
		return nil, fmt.Errorf("%w: nil TokenByteDecoder", ErrInvalidTokenTrie)
	}
	if vocabSize <= 0 || vocabSize > maxGrammarVocabSize {
		return nil, fmt.Errorf("%w: vocab size %d out of range [1, %d]", ErrInvalidTokenTrie, vocabSize, maxGrammarVocabSize)
	}
	if eosTokenID >= vocabSize {
		return nil, fmt.Errorf("%w: eos token %d >= vocab size %d", ErrInvalidTokenTrie, eosTokenID, vocabSize)
	}
	specMap := make(map[int]bool, len(specialTokens)+1)
	if eosTokenID >= 0 {
		specMap[eosTokenID] = true
	}
	for _, id := range specialTokens {
		if id < 0 || id >= vocabSize {
			return nil, fmt.Errorf("%w: special token %d out of range [0, %d)", ErrInvalidTokenTrie, id, vocabSize)
		}
		specMap[id] = true
	}

	pieces := make([][]byte, vocabSize)
	entries := make([]trieEntry, 0, vocabSize)
	maxLen := 0
	for id := 0; id < vocabSize; id++ {
		raw := dec.TokenBytes(id)
		if len(raw) > maxGrammarTokenBytes {
			return nil, fmt.Errorf("%w: token %d byte length %d exceeds %d", ErrInvalidTokenTrie, id, len(raw), maxGrammarTokenBytes)
		}
		if len(raw) > 0 {
			cp := append([]byte(nil), raw...)
			pieces[id] = cp
			if !specMap[id] {
				entries = append(entries, trieEntry{piece: cp, id: id})
				if len(cp) > maxLen {
					maxLen = len(cp)
				}
			}
		}
	}
	slices.SortFunc(entries, func(a, b trieEntry) int {
		if c := bytes.Compare(a.piece, b.piece); c != 0 {
			return c
		}
		return a.id - b.id
	})

	t := &TokenTrie{
		vocabSize:   vocabSize,
		eosToken:    eosTokenID,
		maxTokenLen: maxLen,
		special:     specMap,
		pieces:      pieces,
		nodes:       make([]trieNode, 1, len(entries)+1),
	}
	t.nodes[0] = trieNode{tokenID: -1, dupIndex: -1}
	t.buildSubtree(0, entries, 0)
	t.nodes[0].subtreeSize = int32(len(t.nodes))
	return t, nil
}

// NewTokenTrieFromPieces constructs a TokenTrie from an explicit slice of token
// byte pieces.
func NewTokenTrieFromPieces(pieces [][]byte, eosTokenID int, specialTokens ...int) (*TokenTrie, error) {
	return NewTokenTrie(len(pieces), eosTokenID, sliceTokenDecoder(pieces), specialTokens...)
}

type sliceTokenDecoder [][]byte

func (d sliceTokenDecoder) TokenBytes(id int) []byte {
	if id < 0 || id >= len(d) {
		return nil
	}
	return d[id]
}

func (t *TokenTrie) buildSubtree(nodeIdx int, entries []trieEntry, depth int) {
	i := 0
	for i < len(entries) && len(entries[i].piece) == depth {
		id := entries[i].id
		if t.nodes[nodeIdx].tokenID < 0 {
			t.nodes[nodeIdx].tokenID = int32(id)
		} else {
			if t.nodes[nodeIdx].dupIndex < 0 {
				t.nodes[nodeIdx].dupIndex = int32(len(t.dupTokens))
				t.dupTokens = append(t.dupTokens, []int{id})
			} else {
				idx := t.nodes[nodeIdx].dupIndex
				t.dupTokens[idx] = append(t.dupTokens[idx], id)
			}
		}
		i++
	}
	for i < len(entries) {
		b := entries[i].piece[depth]
		j := i + 1
		for j < len(entries) && entries[j].piece[depth] == b {
			j++
		}
		childIdx := len(t.nodes)
		t.nodes = append(t.nodes, trieNode{
			byteVal:  b,
			tokenID:  -1,
			dupIndex: -1,
		})
		t.buildSubtree(childIdx, entries[i:j], depth+1)
		t.nodes[childIdx].subtreeSize = int32(len(t.nodes) - childIdx)
		i = j
	}
}

// VocabSize returns the vocabulary size covered by this trie.
func (t *TokenTrie) VocabSize() int { return t.vocabSize }

// EOSTokenID returns the configured EOS token ID, or < 0 if disabled.
func (t *TokenTrie) EOSTokenID() int { return t.eosToken }

// NodeCount returns the total number of compressed prefix nodes in the trie.
func (t *TokenTrie) NodeCount() int { return len(t.nodes) }

// TokenBytes returns a defensive copy of tokenID's decoded bytes, or nil if
// unknown or zero-byte.
func (t *TokenTrie) TokenBytes(tokenID int) []byte {
	if t == nil || tokenID < 0 || tokenID >= len(t.pieces) || len(t.pieces[tokenID]) == 0 {
		return nil
	}
	return append([]byte(nil), t.pieces[tokenID]...)
}

// AllowedTokens allocates a fresh TokenMaskBitset populated via ComputeBias.
func (t *TokenTrie) AllowedTokens(r GrammarRecognizer) (*TokenMaskBitset, error) {
	if t == nil {
		return nil, fmt.Errorf("%w: nil TokenTrie", ErrInvalidTokenTrie)
	}
	out := NewTokenMaskBitset(t.vocabSize)
	if err := t.ComputeBias(r, out); err != nil {
		return nil, err
	}
	return out, nil
}

// ComputeBias traverses the prefix trie with r, writing every token ID admitted
// by r into out. Rejected prefix subtrees are skipped in O(1) via subtreeSize,
// and r is popped back to its exact entry state on return.
func (t *TokenTrie) ComputeBias(r GrammarRecognizer, out *TokenMaskBitset) error {
	if t == nil {
		return fmt.Errorf("%w: nil TokenTrie", ErrInvalidTokenTrie)
	}
	if r == nil {
		return fmt.Errorf("%w: nil GrammarRecognizer", ErrInvalidTokenTrie)
	}
	if out == nil || out.vocabSize != t.vocabSize {
		return fmt.Errorf("%w: bitset vocab size mismatch", ErrInvalidTokenTrie)
	}
	out.Clear()
	if t.eosToken >= 0 && r.AllowEOS() {
		out.Allow(t.eosToken)
	}
	if len(t.nodes) <= 1 {
		return nil
	}

	type walkFrame struct {
		nextChild int
		end       int
		pushed    bool
	}
	stackCap := min(t.maxTokenLen+1, 64)
	stack := make([]walkFrame, 0, stackCap)
	stack = append(stack, walkFrame{
		nextChild: 1,
		end:       int(t.nodes[0].subtreeSize),
		pushed:    false,
	})
	for len(stack) > 0 {
		top := &stack[len(stack)-1]
		if top.nextChild >= top.end {
			if top.pushed {
				r.PopBytes(1)
			}
			stack = stack[:len(stack)-1]
			continue
		}
		childIdx := top.nextChild
		child := &t.nodes[childIdx]
		childEnd := childIdx + int(child.subtreeSize)
		top.nextChild = childEnd
		if !r.TryPushByte(child.byteVal) {
			continue
		}
		if child.tokenID >= 0 {
			out.Allow(int(child.tokenID))
			if child.dupIndex >= 0 {
				for _, dupID := range t.dupTokens[child.dupIndex] {
					out.Allow(dupID)
				}
			}
		}
		if childIdx+1 < childEnd {
			stack = append(stack, walkFrame{
				nextChild: childIdx + 1,
				end:       childEnd,
				pushed:    true,
			})
		} else {
			r.PopBytes(1)
		}
	}
	return nil
}

// GreedyMatchForced consumes as much of forced as possible using longest valid
// token prefixes present in the trie without overshooting forced.
func (t *TokenTrie) GreedyMatchForced(forced []byte) []int {
	if t == nil || len(forced) == 0 || len(t.nodes) <= 1 {
		return nil
	}
	var tokens []int
	pos := 0
	for pos < len(forced) {
		nodeIdx := 0
		bestToken := -1
		bestEnd := pos
		for cur := pos; cur < len(forced); cur++ {
			b := forced[cur]
			childIdx := t.findChild(nodeIdx, b)
			if childIdx < 0 {
				break
			}
			nodeIdx = childIdx
			if t.nodes[nodeIdx].tokenID >= 0 {
				bestToken = int(t.nodes[nodeIdx].tokenID)
				bestEnd = cur + 1
			}
		}
		if bestToken < 0 {
			break
		}
		tokens = append(tokens, bestToken)
		pos = bestEnd
	}
	return tokens
}

func (t *TokenTrie) findChild(nodeIdx int, b byte) int {
	child := nodeIdx + 1
	end := nodeIdx + int(t.nodes[nodeIdx].subtreeSize)
	for child < end {
		n := &t.nodes[child]
		if n.byteVal == b {
			return child
		}
		if n.byteVal > b {
			return -1
		}
		child += int(n.subtreeSize)
	}
	return -1
}

// --- Byte-level CFG & Earley Parser State Machine ---

type byteSet [4]uint64

func (s *byteSet) add(b byte) {
	s[b>>6] |= uint64(1) << (b & 63)
}

func (s *byteSet) addRange(lo, hi byte) {
	for b := int(lo); b <= int(hi); b++ {
		s.add(byte(b))
	}
}

func (s byteSet) has(b byte) bool {
	return (s[b>>6] & (uint64(1) << (b & 63))) != 0
}

func (s byteSet) empty() bool {
	return (s[0] | s[1] | s[2] | s[3]) == 0
}

func (s byteSet) singleByte() (byte, bool) {
	count := bits.OnesCount64(s[0]) + bits.OnesCount64(s[1]) + bits.OnesCount64(s[2]) + bits.OnesCount64(s[3])
	if count != 1 {
		return 0, false
	}
	for wi := 0; wi < 4; wi++ {
		if s[wi] != 0 {
			return byte((wi << 6) + bits.TrailingZeros64(s[wi])), true
		}
	}
	return 0, false
}

func (s *byteSet) orWith(other byteSet) {
	s[0] |= other[0]
	s[1] |= other[1]
	s[2] |= other[2]
	s[3] |= other[3]
}

func singleByteSet(b byte) byteSet {
	var s byteSet
	s.add(b)
	return s
}

type cfgSym struct {
	isTerm bool
	rule   int
	term   byteSet
}

type cfgProd struct {
	lhs int
	rhs []cfgSym
}

type cfgRule struct {
	name  string
	prods []int
}

// CompiledGrammar is an immutable, pruned byte-level context-free grammar ready
// to drive GrammarParser / GrammarMaskEngine.
type CompiledGrammar struct {
	startRule int
	rules     []cfgRule
	prods     []cfgProd
	nullable  []bool
}

// RuleCount returns the number of grammar rules in the compiled CFG.
func (g *CompiledGrammar) RuleCount() int {
	if g == nil {
		return 0
	}
	return len(g.rules)
}

// ProdCount returns the number of grammar productions in the compiled CFG.
func (g *CompiledGrammar) ProdCount() int {
	if g == nil {
		return 0
	}
	return len(g.prods)
}

type cfgBuilder struct {
	rules   []cfgRule
	prods   []cfgProd
	ruleMap map[string]int
}

func newCFGBuilder() *cfgBuilder {
	return &cfgBuilder{ruleMap: make(map[string]int)}
}

func (b *cfgBuilder) newRule(name string) (int, error) {
	if len(b.rules) >= maxGrammarRules {
		return -1, fmt.Errorf("%w: rule count exceeds limit %d", ErrUnsupportedGrammar, maxGrammarRules)
	}
	id := len(b.rules)
	b.rules = append(b.rules, cfgRule{name: name})
	return id, nil
}

func (b *cfgBuilder) namedRule(name string) (int, error) {
	if id, ok := b.ruleMap[name]; ok {
		return id, nil
	}
	id, err := b.newRule(name)
	if err != nil {
		return -1, err
	}
	b.ruleMap[name] = id
	return id, nil
}

func (b *cfgBuilder) addProd(lhs int, rhs ...cfgSym) error {
	if len(b.prods) >= maxGrammarProds {
		return fmt.Errorf("%w: production count exceeds limit %d", ErrUnsupportedGrammar, maxGrammarProds)
	}
	if len(rhs) > 4095 {
		return fmt.Errorf("%w: production rhs length exceeds limit", ErrUnsupportedGrammar)
	}
	pid := len(b.prods)
	b.prods = append(b.prods, cfgProd{lhs: lhs, rhs: rhs})
	b.rules[lhs].prods = append(b.rules[lhs].prods, pid)
	return nil
}

func (b *cfgBuilder) addLiteralProd(lhs int, lit []byte) error {
	rhs := make([]cfgSym, len(lit))
	for i, ch := range lit {
		rhs[i] = cfgSym{isTerm: true, term: singleByteSet(ch)}
	}
	return b.addProd(lhs, rhs...)
}

func (b *cfgBuilder) finalize(startRule int) (*CompiledGrammar, error) {
	if startRule < 0 || startRule >= len(b.rules) {
		return nil, fmt.Errorf("%w: invalid start rule", ErrInvalidGrammar)
	}
	numRules := len(b.rules)
	productive := make([]bool, numRules)
	changed := true
	for changed {
		changed = false
		for _, p := range b.prods {
			if productive[p.lhs] {
				continue
			}
			ok := true
			for _, sym := range p.rhs {
				if sym.isTerm {
					if sym.term.empty() {
						ok = false
						break
					}
					continue
				}
				if !productive[sym.rule] {
					ok = false
					break
				}
			}
			if ok {
				productive[p.lhs] = true
				changed = true
			}
		}
	}
	if !productive[startRule] {
		return nil, fmt.Errorf("%w: grammar produces empty language", ErrInvalidGrammar)
	}

	// Keep only productive productions and rules reachable from startRule.
	prodOK := make([]bool, len(b.prods))
	for pid, p := range b.prods {
		if !productive[p.lhs] {
			continue
		}
		ok := true
		for _, sym := range p.rhs {
			if sym.isTerm {
				if sym.term.empty() {
					ok = false
					break
				}
			} else if !productive[sym.rule] {
				ok = false
				break
			}
		}
		prodOK[pid] = ok
	}

	reachable := make([]bool, numRules)
	reachable[startRule] = true
	queue := []int{startRule}
	for len(queue) > 0 {
		r := queue[0]
		queue = queue[1:]
		for _, pid := range b.rules[r].prods {
			if !prodOK[pid] {
				continue
			}
			for _, sym := range b.prods[pid].rhs {
				if !sym.isTerm && !reachable[sym.rule] {
					reachable[sym.rule] = true
					queue = append(queue, sym.rule)
				}
			}
		}
	}

	remap := make([]int, numRules)
	for i := range remap {
		remap[i] = -1
	}
	var rules []cfgRule
	for oldID := 0; oldID < numRules; oldID++ {
		if reachable[oldID] && productive[oldID] {
			remap[oldID] = len(rules)
			rules = append(rules, cfgRule{name: b.rules[oldID].name})
		}
	}
	var prods []cfgProd
	for pid, p := range b.prods {
		if !prodOK[pid] || remap[p.lhs] < 0 {
			continue
		}
		newLHS := remap[p.lhs]
		newRHS := make([]cfgSym, len(p.rhs))
		for i, sym := range p.rhs {
			if sym.isTerm {
				newRHS[i] = sym
			} else {
				newRHS[i] = cfgSym{rule: remap[sym.rule]}
			}
		}
		newPID := len(prods)
		prods = append(prods, cfgProd{lhs: newLHS, rhs: newRHS})
		rules[newLHS].prods = append(rules[newLHS].prods, newPID)
	}

	nullable := make([]bool, len(rules))
	changed = true
	for changed {
		changed = false
		for _, p := range prods {
			if nullable[p.lhs] {
				continue
			}
			ok := true
			for _, sym := range p.rhs {
				if sym.isTerm || !nullable[sym.rule] {
					ok = false
					break
				}
			}
			if ok {
				nullable[p.lhs] = true
				changed = true
			}
		}
	}

	return &CompiledGrammar{
		startRule: remap[startRule],
		rules:     rules,
		prods:     prods,
		nullable:  nullable,
	}, nil
}

type earleyItem struct {
	prod   uint32
	dot    uint16
	origin uint32
}

func packEarleyItem(it earleyItem) uint64 {
	return (uint64(it.prod) << 44) | (uint64(it.dot) << 32) | uint64(it.origin)
}

type earleyChart struct {
	items     []earleyItem
	allowed   byteSet
	accepting bool
}

// GrammarParser is the stateful byte-level Earley parser machine implementing
// GrammarRecognizer for incremental token-trie mask computation, commit, and
// speculative fork.
type GrammarParser struct {
	g              *CompiledGrammar
	charts         []earleyChart
	matched        []byte
	seenEpoch      map[uint64]int
	predictedEpoch []int
	epoch          int
}

// NewGrammarParser creates a fresh parser state initialized at byte offset 0.
func NewGrammarParser(g *CompiledGrammar) (*GrammarParser, error) {
	if g == nil || len(g.rules) == 0 {
		return nil, fmt.Errorf("%w: nil or empty compiled grammar", ErrInvalidGrammar)
	}
	p := &GrammarParser{
		g:              g,
		charts:         make([]earleyChart, 1, 32),
		seenEpoch:      make(map[uint64]int, 64),
		predictedEpoch: make([]int, len(g.rules)),
	}
	p.Reset()
	return p, nil
}

// Reset restores the parser back to byte offset 0.
func (p *GrammarParser) Reset() {
	if p == nil {
		return
	}
	p.charts = p.charts[:1]
	p.charts[0].items = p.charts[0].items[:0]
	p.charts[0].allowed = byteSet{}
	p.charts[0].accepting = false
	p.matched = p.matched[:0]
	for _, pid := range p.g.rules[p.g.startRule].prods {
		p.charts[0].items = append(p.charts[0].items, earleyItem{
			prod:   uint32(pid),
			dot:    0,
			origin: 0,
		})
	}
	_ = p.closeChart(0)
}

// Fork returns an independent copy of the parser state sharing the immutable
// CompiledGrammar.
func (p *GrammarParser) Fork() *GrammarParser {
	if p == nil {
		return nil
	}
	chartsCopy := make([]earleyChart, len(p.charts))
	for i := range p.charts {
		items := make([]earleyItem, len(p.charts[i].items))
		copy(items, p.charts[i].items)
		chartsCopy[i] = earleyChart{
			items:     items,
			allowed:   p.charts[i].allowed,
			accepting: p.charts[i].accepting,
		}
	}
	matchedCopy := make([]byte, len(p.matched))
	copy(matchedCopy, p.matched)
	return &GrammarParser{
		g:              p.g,
		charts:         chartsCopy,
		matched:        matchedCopy,
		seenEpoch:      make(map[uint64]int, 64),
		predictedEpoch: make([]int, len(p.g.rules)),
	}
}

// TryPushByte advances the parser by one byte b, returning false without
// modifying state if b is not admitted by the current grammar position.
func (p *GrammarParser) TryPushByte(b byte) bool {
	if p == nil || len(p.charts) == 0 {
		return false
	}
	if len(p.matched) >= maxGrammarChartBytes {
		return false
	}
	curIdx := len(p.charts) - 1
	cur := &p.charts[curIdx]
	if !cur.allowed.has(b) {
		return false
	}
	nextIdx := curIdx + 1
	if nextIdx < cap(p.charts) {
		p.charts = p.charts[:nextIdx+1]
	} else {
		p.charts = append(p.charts, earleyChart{})
	}
	next := &p.charts[nextIdx]
	next.items = next.items[:0]
	next.allowed = byteSet{}
	next.accepting = false

	prods := p.g.prods
	for _, it := range p.charts[curIdx].items {
		rhs := prods[it.prod].rhs
		if int(it.dot) < len(rhs) && rhs[it.dot].isTerm && rhs[it.dot].term.has(b) {
			next.items = append(next.items, earleyItem{
				prod:   it.prod,
				dot:    it.dot + 1,
				origin: it.origin,
			})
		}
	}
	if !p.closeChart(nextIdx) {
		p.charts = p.charts[:curIdx+1]
		return false
	}
	p.matched = append(p.matched, b)
	return true
}

// TryPushBytes atomically attempts to advance all bytes of s, rolling back on
// the first rejected byte.
func (p *GrammarParser) TryPushBytes(s []byte) bool {
	pushed := 0
	for _, b := range s {
		if !p.TryPushByte(b) {
			if pushed > 0 {
				p.PopBytes(pushed)
			}
			return false
		}
		pushed++
	}
	return true
}

// PopBytes rolls back the last n pushed bytes in O(1) chart slice truncation.
func (p *GrammarParser) PopBytes(n int) {
	if p == nil || n <= 0 {
		return
	}
	if n >= len(p.charts) {
		p.Reset()
		return
	}
	p.charts = p.charts[:len(p.charts)-n]
	if n <= len(p.matched) {
		p.matched = p.matched[:len(p.matched)-n]
	} else {
		p.matched = p.matched[:0]
	}
}

// AllowEOS reports whether the parser has matched a complete grammar sentence.
func (p *GrammarParser) AllowEOS() bool {
	return p.IsAccepting()
}

// IsAccepting reports whether the current prefix is a complete valid sentence.
func (p *GrammarParser) IsAccepting() bool {
	if p == nil || len(p.charts) == 0 {
		return false
	}
	return p.charts[len(p.charts)-1].accepting
}

// MatchedBytes returns a defensive copy of all committed bytes so far.
func (p *GrammarParser) MatchedBytes() []byte {
	if p == nil || len(p.matched) == 0 {
		return nil
	}
	return append([]byte(nil), p.matched...)
}

// AllowedNextBytes returns the 256-entry bitmap of valid next bytes at the
// current parser position.
func (p *GrammarParser) AllowedNextBytes() [256]bool {
	var out [256]bool
	if p == nil || len(p.charts) == 0 {
		return out
	}
	allowed := p.charts[len(p.charts)-1].allowed
	for wi := 0; wi < 4; wi++ {
		w := allowed[wi]
		base := wi << 6
		for w != 0 {
			bit := bits.TrailingZeros64(w)
			out[base+bit] = true
			w &= w - 1
		}
	}
	return out
}

// ForcedBytes computes the longest deterministic byte continuation available
// from the current state without mutating parser position.
func (p *GrammarParser) ForcedBytes() []byte {
	if p == nil || len(p.charts) == 0 {
		return nil
	}
	var forced []byte
	pushed := 0
	for len(forced) < maxGrammarForcedBytes {
		cur := &p.charts[len(p.charts)-1]
		if cur.accepting {
			break
		}
		b, ok := cur.allowed.singleByte()
		if !ok {
			break
		}
		if !p.TryPushByte(b) {
			break
		}
		forced = append(forced, b)
		pushed++
	}
	if pushed > 0 {
		p.PopBytes(pushed)
	}
	return forced
}

func (p *GrammarParser) nextEpoch() int {
	p.epoch++
	if p.epoch == 0 || len(p.seenEpoch) > maxGrammarChartItems*2 {
		clear(p.seenEpoch)
		clear(p.predictedEpoch)
		p.epoch = 1
	}
	return p.epoch
}

func (p *GrammarParser) closeChart(k int) bool {
	ch := &p.charts[k]
	if len(ch.items) == 0 {
		return false
	}
	epoch := p.nextEpoch()
	// Deduplicate initial seed items before running closure.
	deduped := ch.items[:0]
	for _, it := range ch.items {
		key := packEarleyItem(it)
		if p.seenEpoch[key] != epoch {
			p.seenEpoch[key] = epoch
			deduped = append(deduped, it)
		}
	}
	ch.items = deduped

	enqueue := func(it earleyItem) bool {
		key := packEarleyItem(it)
		if p.seenEpoch[key] == epoch {
			return true
		}
		if len(p.charts[k].items) >= maxGrammarChartItems {
			return false
		}
		p.seenEpoch[key] = epoch
		p.charts[k].items = append(p.charts[k].items, it)
		return true
	}

	prods := p.g.prods
	rules := p.g.rules
	nullable := p.g.nullable
	startRule := p.g.startRule

	for idx := 0; idx < len(p.charts[k].items); idx++ {
		it := p.charts[k].items[idx]
		prod := &prods[it.prod]
		if int(it.dot) == len(prod.rhs) {
			if prod.lhs == startRule && it.origin == 0 {
				p.charts[k].accepting = true
			}
			origin := int(it.origin)
			// Read by index rather than caching a slice header because origin == k
			// may grow while enqueue appends to p.charts[k].items.
			for j := 0; j < len(p.charts[origin].items); j++ {
				parent := p.charts[origin].items[j]
				parentRHS := prods[parent.prod].rhs
				if int(parent.dot) < len(parentRHS) {
					sym := parentRHS[parent.dot]
					if !sym.isTerm && sym.rule == prod.lhs {
						if !enqueue(earleyItem{
							prod:   parent.prod,
							dot:    parent.dot + 1,
							origin: parent.origin,
						}) {
							return false
						}
					}
				}
			}
			continue
		}

		sym := prod.rhs[it.dot]
		if sym.isTerm {
			p.charts[k].allowed.orWith(sym.term)
			continue
		}

		subRule := sym.rule
		if p.predictedEpoch[subRule] != epoch {
			p.predictedEpoch[subRule] = epoch
			for _, cp := range rules[subRule].prods {
				if !enqueue(earleyItem{
					prod:   uint32(cp),
					dot:    0,
					origin: uint32(k),
				}) {
					return false
				}
			}
		}
		if nullable[subRule] {
			if !enqueue(earleyItem{
				prod:   it.prod,
				dot:    it.dot + 1,
				origin: it.origin,
			}) {
				return false
			}
		}
	}
	return !p.charts[k].allowed.empty() || p.charts[k].accepting
}

// --- Grammar Mask Engine (TokenTrie + GrammarParser) ---

// GrammarMaskStep captures one complete constrained-decoding step emitted by
// GrammarMaskEngine.StepMask.
type GrammarMaskStep struct {
	Allowed      *TokenMaskBitset
	Accepting    bool
	Stopped      bool
	ForcedBytes  []byte
	ForcedTokens []int
}

// GrammarMaskEngine pairs an immutable TokenTrie with a mutable GrammarParser
// state machine to compute per-step token allow-lists, detect deterministic
// jump-forward tokens, apply logit masks, and commit generated tokens.
type GrammarMaskEngine struct {
	trie    *TokenTrie
	grammar *CompiledGrammar
	parser  *GrammarParser
	maskBuf *TokenMaskBitset
	stopped bool
}

// NewGrammarMaskEngine constructs a constrained-decoding engine for trie and
// compiled grammar g.
func NewGrammarMaskEngine(trie *TokenTrie, g *CompiledGrammar) (*GrammarMaskEngine, error) {
	if trie == nil {
		return nil, fmt.Errorf("%w: nil TokenTrie", ErrInvalidTokenTrie)
	}
	parser, err := NewGrammarParser(g)
	if err != nil {
		return nil, err
	}
	return &GrammarMaskEngine{
		trie:    trie,
		grammar: g,
		parser:  parser,
		maskBuf: NewTokenMaskBitset(trie.vocabSize),
	}, nil
}

// Reset restores the engine to byte offset 0 and clears the stopped state.
func (e *GrammarMaskEngine) Reset() {
	if e == nil {
		return
	}
	e.stopped = false
	e.parser.Reset()
	e.maskBuf.Clear()
}

// Fork returns an independent copy of the engine sharing the immutable trie and
// compiled grammar, suitable for speculative decoding or beam search.
func (e *GrammarMaskEngine) Fork() *GrammarMaskEngine {
	if e == nil {
		return nil
	}
	return &GrammarMaskEngine{
		trie:    e.trie,
		grammar: e.grammar,
		parser:  e.parser.Fork(),
		maskBuf: NewTokenMaskBitset(e.trie.vocabSize),
		stopped: e.stopped,
	}
}

// IsAccepting reports whether the committed output forms a complete valid
// sentence under the grammar.
func (e *GrammarMaskEngine) IsAccepting() bool {
	if e == nil {
		return false
	}
	return e.parser.IsAccepting()
}

// IsStopped reports whether EOS has been committed.
func (e *GrammarMaskEngine) IsStopped() bool {
	if e == nil {
		return false
	}
	return e.stopped
}

// MatchedBytes returns a copy of the committed output bytes so far.
func (e *GrammarMaskEngine) MatchedBytes() []byte {
	if e == nil {
		return nil
	}
	return e.parser.MatchedBytes()
}

// ComputeMask computes and returns a fresh bitset of all token IDs permitted at
// the current parser position.
func (e *GrammarMaskEngine) ComputeMask() (*TokenMaskBitset, error) {
	if e == nil {
		return nil, fmt.Errorf("%w: nil GrammarMaskEngine", ErrInvalidGrammar)
	}
	if e.stopped {
		return NewTokenMaskBitset(e.trie.vocabSize), nil
	}
	if err := e.trie.ComputeBias(e.parser, e.maskBuf); err != nil {
		return nil, err
	}
	return e.maskBuf.Clone(), nil
}

// StepMask computes the current per-step token bitset together with
// accepting/stopped status and any deterministic fast-forward bytes/tokens.
func (e *GrammarMaskEngine) StepMask() (GrammarMaskStep, error) {
	if e == nil {
		return GrammarMaskStep{}, fmt.Errorf("%w: nil GrammarMaskEngine", ErrInvalidGrammar)
	}
	if e.stopped {
		return GrammarMaskStep{
			Allowed:   NewTokenMaskBitset(e.trie.vocabSize),
			Accepting: e.parser.IsAccepting(),
			Stopped:   true,
		}, nil
	}
	if err := e.trie.ComputeBias(e.parser, e.maskBuf); err != nil {
		return GrammarMaskStep{}, err
	}
	forced := e.parser.ForcedBytes()
	var forcedTokens []int
	if len(forced) > 0 {
		forcedTokens = e.trie.GreedyMatchForced(forced)
	}
	return GrammarMaskStep{
		Allowed:      e.maskBuf.Clone(),
		Accepting:    e.parser.IsAccepting(),
		Stopped:      false,
		ForcedBytes:  forced,
		ForcedTokens: forcedTokens,
	}, nil
}

// NextAllowedTokens returns the sorted per-step allow-list of valid token IDs.
func (e *GrammarMaskEngine) NextAllowedTokens() ([]int, error) {
	mask, err := e.ComputeMask()
	if err != nil {
		return nil, err
	}
	return mask.AllowedIDs(), nil
}

// CommitToken transactionally advances the parser state by tokenID. Illegal,
// out-of-range, non-EOS special, or zero-byte tokens leave state unchanged and
// return ErrGrammarRejectedToken.
func (e *GrammarMaskEngine) CommitToken(tokenID int) error {
	if e == nil {
		return fmt.Errorf("%w: nil GrammarMaskEngine", ErrInvalidGrammar)
	}
	if e.stopped {
		return ErrGrammarStopped
	}
	if tokenID < 0 || tokenID >= e.trie.vocabSize {
		return fmt.Errorf("%w: token %d out of range [0, %d)", ErrGrammarRejectedToken, tokenID, e.trie.vocabSize)
	}
	if tokenID == e.trie.eosToken {
		if !e.parser.IsAccepting() {
			return fmt.Errorf("%w: EOS token %d disallowed before accepting state", ErrGrammarRejectedToken, tokenID)
		}
		e.stopped = true
		return nil
	}
	if e.trie.special[tokenID] {
		return fmt.Errorf("%w: special token %d disallowed", ErrGrammarRejectedToken, tokenID)
	}
	piece := e.trie.pieces[tokenID]
	if len(piece) == 0 {
		return fmt.Errorf("%w: zero-byte token %d disallowed", ErrGrammarRejectedToken, tokenID)
	}
	if !e.parser.TryPushBytes(piece) {
		return fmt.Errorf("%w: token %d (%q) violates grammar", ErrGrammarRejectedToken, tokenID, piece)
	}
	return nil
}

// CommitBytes transactionally advances the parser state by raw bytes b.
func (e *GrammarMaskEngine) CommitBytes(b []byte) error {
	if e == nil {
		return fmt.Errorf("%w: nil GrammarMaskEngine", ErrInvalidGrammar)
	}
	if e.stopped {
		return ErrGrammarStopped
	}
	if !e.parser.TryPushBytes(b) {
		return fmt.Errorf("%w: bytes %q violate grammar", ErrGrammarRejectedToken, b)
	}
	return nil
}

// ApplyMask computes the current token mask and applies it to logits in place
// while preserving domain monotonicity.
func (e *GrammarMaskEngine) ApplyMask(logits []float32) (GrammarMaskDomainMonotonicityReceipt, error) {
	var receipt GrammarMaskDomainMonotonicityReceipt
	if e == nil {
		return receipt, fmt.Errorf("%w: nil GrammarMaskEngine", ErrInvalidGrammar)
	}
	if e.stopped {
		return receipt, ErrGrammarStopped
	}
	if err := e.trie.ComputeBias(e.parser, e.maskBuf); err != nil {
		return receipt, err
	}
	return e.maskBuf.ApplyToLogits(logits)
}

// --- UTF-8 Byte-NFA / Byte-DFA Compiler (Regex & JSON String Patterns) ---

type regexNFANode struct {
	eps       []int32
	beginOnly []int32
	endOnly   []int32
	edges     []regexByteEdge
	match     bool
}

type regexByteEdge struct {
	lo, hi byte
	to     int32
}

type regexNFA struct {
	nodes []regexNFANode
}

func (n *regexNFA) newNode() (int32, error) {
	if len(n.nodes) >= maxGrammarNFASize {
		return -1, fmt.Errorf("%w: regex NFA exceeds %d states", ErrUnsupportedGrammar, maxGrammarNFASize)
	}
	id := int32(len(n.nodes))
	n.nodes = append(n.nodes, regexNFANode{})
	return id, nil
}

func (n *regexNFA) addSequence(from int32, seq []byteRange, to int32) error {
	cur := from
	for i, br := range seq {
		next := to
		if i+1 < len(seq) {
			var err error
			next, err = n.newNode()
			if err != nil {
				return err
			}
		}
		n.nodes[cur].edges = append(n.nodes[cur].edges, regexByteEdge{
			lo: br.lo,
			hi: br.hi,
			to: next,
		})
		cur = next
	}
	return nil
}

type byteRange struct {
	lo, hi byte
}

type runeRange struct {
	lo, hi rune
}

// appendUTF8Sequences decomposes a Unicode codepoint interval [lo, hi] into exact
// RFC 3629 UTF-8 byte-range sequences, skipping surrogate codepoints.
func appendUTF8Sequences(dst [][]byteRange, lo, hi rune) [][]byteRange {
	if lo > hi {
		return dst
	}
	// Clamp and split around UTF-16 surrogates [0xD800, 0xDFFF].
	if lo < 0 {
		lo = 0
	}
	if hi > utf8.MaxRune {
		hi = utf8.MaxRune
	}
	if lo <= 0xDFFF && hi >= 0xD800 {
		if lo < 0xD800 {
			dst = appendUTF8Sequences(dst, lo, 0xD7FF)
		}
		if hi > 0xDFFF {
			dst = appendUTF8Sequences(dst, 0xE000, hi)
		}
		return dst
	}

	buckets := [][2]rune{
		{0x00, 0x7F},
		{0x80, 0x7FF},
		{0x800, 0xD7FF},
		{0xE000, 0xFFFF},
		{0x10000, utf8.MaxRune},
	}
	for _, b := range buckets {
		a := max(lo, b[0])
		z := min(hi, b[1])
		if a > z {
			continue
		}
		var sb [utf8.UTFMax]byte
		var eb [utf8.UTFMax]byte
		sn := utf8.EncodeRune(sb[:], a)
		en := utf8.EncodeRune(eb[:], z)
		dst = splitFixedUTF8Ranges(dst, sb[:sn], eb[:en], nil)
	}
	return dst
}

func splitFixedUTF8Ranges(dst [][]byteRange, start, end []byte, prefix []byteRange) [][]byteRange {
	if len(start) == 0 {
		seq := make([]byteRange, len(prefix))
		copy(seq, prefix)
		return append(dst, seq)
	}
	if start[0] == end[0] {
		return splitFixedUTF8Ranges(dst, start[1:], end[1:], append(prefix, byteRange{lo: start[0], hi: start[0]}))
	}
	if len(start) == 1 {
		seq := make([]byteRange, len(prefix)+1)
		copy(seq, prefix)
		seq[len(prefix)] = byteRange{lo: start[0], hi: end[0]}
		return append(dst, seq)
	}

	midLo := start[0]
	midHi := end[0]
	isMinCont := true
	for _, b := range start[1:] {
		if b != 0x80 {
			isMinCont = false
			break
		}
	}
	isMaxCont := true
	for _, b := range end[1:] {
		if b != 0xBF {
			isMaxCont = false
			break
		}
	}

	if !isMinCont {
		maxTail := make([]byte, len(start)-1)
		for i := range maxTail {
			maxTail[i] = 0xBF
		}
		dst = splitFixedUTF8Ranges(dst, start[1:], maxTail, append(prefix, byteRange{lo: start[0], hi: start[0]}))
		midLo++
	}
	if midLo <= midHi {
		effHi := midHi
		if !isMaxCont {
			effHi--
		}
		if midLo <= effHi {
			seq := make([]byteRange, len(prefix)+len(start))
			copy(seq, prefix)
			seq[len(prefix)] = byteRange{lo: midLo, hi: effHi}
			for i := 1; i < len(start); i++ {
				seq[len(prefix)+i] = byteRange{lo: 0x80, hi: 0xBF}
			}
			dst = append(dst, seq)
		}
	}
	if !isMaxCont {
		minTail := make([]byte, len(end)-1)
		for i := range minTail {
			minTail[i] = 0x80
		}
		dst = splitFixedUTF8Ranges(dst, minTail, end[1:], append(prefix, byteRange{lo: end[0], hi: end[0]}))
	}
	return dst
}

func buildRegexNFA(prog *syntax.Prog) (*regexNFA, error) {
	n := &regexNFA{nodes: make([]regexNFANode, len(prog.Inst))}
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
				return nil, fmt.Errorf("%w: unsupported regex assertion", ErrUnsupportedGrammar)
			}
			if cond == 0 {
				node.eps = append(node.eps, int32(inst.Out))
			} else if cond == syntax.EmptyBeginText {
				node.beginOnly = append(node.beginOnly, int32(inst.Out))
			} else if cond == syntax.EmptyEndText {
				node.endOnly = append(node.endOnly, int32(inst.Out))
			} else {
				// Both begin and end required on empty text.
				inter, err := n.newNode()
				if err != nil {
					return nil, err
				}
				n.nodes[pc].beginOnly = append(n.nodes[pc].beginOnly, inter)
				n.nodes[inter].endOnly = append(n.nodes[inter].endOnly, int32(inst.Out))
			}
		case syntax.InstMatch:
			node.match = true
		case syntax.InstFail:
		case syntax.InstRune, syntax.InstRune1, syntax.InstRuneAny, syntax.InstRuneAnyNotNL:
			for _, rr := range regexRuneRanges(inst) {
				for _, seq := range appendUTF8Sequences(nil, rr.lo, rr.hi) {
					if err := n.addSequence(int32(pc), seq, int32(inst.Out)); err != nil {
						return nil, err
					}
				}
			}
		default:
			return nil, fmt.Errorf("%w: unsupported regex opcode %d", ErrUnsupportedGrammar, inst.Op)
		}
	}
	return n, nil
}

func regexRuneRanges(inst *syntax.Inst) []runeRange {
	switch inst.Op {
	case syntax.InstRune1:
		return []runeRange{{lo: inst.Rune[0], hi: inst.Rune[0]}}
	case syntax.InstRuneAny:
		return []runeRange{{lo: 0, hi: utf8.MaxRune}}
	case syntax.InstRuneAnyNotNL:
		return []runeRange{{lo: 0, hi: '\n' - 1}, {lo: '\n' + 1, hi: utf8.MaxRune}}
	default:
		out := make([]runeRange, 0, len(inst.Rune)/2)
		for i := 0; i+1 < len(inst.Rune); i += 2 {
			out = append(out, runeRange{lo: inst.Rune[i], hi: inst.Rune[i+1]})
		}
		return out
	}
}

type byteDFA struct {
	start  int
	accept []bool
	trans  [][256]int
}

func determinizeRegexNFA(n *regexNFA, startPC int32) (*byteDFA, error) {
	closure := func(seeds []int32, atBegin bool) []int32 {
		seen := make(map[int32]bool, len(seeds)*2+4)
		queue := make([]int32, 0, len(seeds)*2+4)
		for _, s := range seeds {
			if !seen[s] {
				seen[s] = true
				queue = append(queue, s)
			}
		}
		for i := 0; i < len(queue); i++ {
			cur := queue[i]
			node := &n.nodes[cur]
			for _, next := range node.eps {
				if !seen[next] {
					seen[next] = true
					queue = append(queue, next)
				}
			}
			if atBegin {
				for _, next := range node.beginOnly {
					if !seen[next] {
						seen[next] = true
						queue = append(queue, next)
					}
				}
			}
		}
		slices.Sort(queue)
		return queue
	}

	canAccept := func(state []int32, atBegin bool) bool {
		seen := make(map[int32]bool, len(state)*2)
		queue := make([]int32, 0, len(state)*2)
		for _, s := range state {
			seen[s] = true
			queue = append(queue, s)
		}
		for i := 0; i < len(queue); i++ {
			cur := queue[i]
			node := &n.nodes[cur]
			if node.match {
				return true
			}
			for _, next := range node.eps {
				if !seen[next] {
					seen[next] = true
					queue = append(queue, next)
				}
			}
			for _, next := range node.endOnly {
				if !seen[next] {
					seen[next] = true
					queue = append(queue, next)
				}
			}
			if atBegin {
				for _, next := range node.beginOnly {
					if !seen[next] {
						seen[next] = true
						queue = append(queue, next)
					}
				}
			}
		}
		return false
	}

	setKey := func(s []int32, atBegin bool) string {
		buf := make([]byte, 1+4*len(s))
		if atBegin {
			buf[0] = 1
		}
		for i, v := range s {
			o := 1 + 4*i
			buf[o] = byte(v)
			buf[o+1] = byte(v >> 8)
			buf[o+2] = byte(v >> 16)
			buf[o+3] = byte(v >> 24)
		}
		return string(buf)
	}

	startSet := closure([]int32{startPC}, true)
	stateMap := map[string]int{setKey(startSet, true): 0}
	type workItem struct {
		set     []int32
		atBegin bool
	}
	work := []workItem{{set: startSet, atBegin: true}}
	dfa := &byteDFA{start: 0}

	for q := 0; q < len(work); q++ {
		item := work[q]
		dfa.accept = append(dfa.accept, canAccept(item.set, item.atBegin))
		var row [256]int
		for b := range row {
			row[b] = -1
		}

		// Partition bytes [0..255] by target NFA seed sets for this state.
		var buckets [256][]int32
		for _, s := range item.set {
			for _, e := range n.nodes[s].edges {
				for b := int(e.lo); b <= int(e.hi); b++ {
					buckets[b] = append(buckets[b], e.to)
				}
			}
		}

		// Memoize closures within this state across bytes with identical target seeds.
		localCache := make(map[string]int)
		for b := 0; b < 256; b++ {
			if len(buckets[b]) == 0 {
				continue
			}
			slices.Sort(buckets[b])
			buckets[b] = slices.Compact(buckets[b])
			nextSet := closure(buckets[b], false)
			k := setKey(nextSet, false)
			if tid, ok := localCache[k]; ok {
				row[b] = tid
				continue
			}
			tid, exists := stateMap[k]
			if !exists {
				if len(work) >= maxGrammarDFAStates {
					return nil, fmt.Errorf("%w: regex DFA exceeds %d states", ErrUnsupportedGrammar, maxGrammarDFAStates)
				}
				tid = len(work)
				stateMap[k] = tid
				work = append(work, workItem{set: nextSet, atBegin: false})
			}
			localCache[k] = tid
			row[b] = tid
		}
		dfa.trans = append(dfa.trans, row)
	}
	return dfa, nil
}

func trimByteDFA(dfa *byteDFA) *byteDFA {
	numStates := len(dfa.accept)
	if numStates == 0 {
		return dfa
	}
	rev := make([][]int, numStates)
	for q := 0; q < numStates; q++ {
		for b := 0; b < 256; b++ {
			to := dfa.trans[q][b]
			if to >= 0 {
				rev[to] = append(rev[to], q)
			}
		}
	}
	coAccess := make([]bool, numStates)
	var queue []int
	for q := 0; q < numStates; q++ {
		if dfa.accept[q] {
			coAccess[q] = true
			queue = append(queue, q)
		}
	}
	for i := 0; i < len(queue); i++ {
		q := queue[i]
		for _, prev := range rev[q] {
			if !coAccess[prev] {
				coAccess[prev] = true
				queue = append(queue, prev)
			}
		}
	}
	if !coAccess[dfa.start] {
		return &byteDFA{start: -1}
	}

	remap := make([]int, numStates)
	for i := range remap {
		remap[i] = -1
	}
	var out byteDFA
	out.start = 0
	for q := 0; q < numStates; q++ {
		if coAccess[q] {
			remap[q] = len(out.accept)
			out.accept = append(out.accept, dfa.accept[q])
		}
	}
	out.trans = make([][256]int, len(out.accept))
	for q := 0; q < numStates; q++ {
		nq := remap[q]
		if nq < 0 {
			continue
		}
		for b := 0; b < 256; b++ {
			to := dfa.trans[q][b]
			if to >= 0 && remap[to] >= 0 {
				out.trans[nq][b] = remap[to]
			} else {
				out.trans[nq][b] = -1
			}
		}
	}
	out.start = remap[dfa.start]
	return &out
}

// addDFALeftLinear emits left-linear CFG rules for a trimmed DFA so Earley
// tracks active DFA origins in O(1) per byte regardless of match length.
func (b *cfgBuilder) addDFALeftLinear(name string, dfa *byteDFA, jsonEscaped bool) (int, error) {
	if dfa.start < 0 || len(dfa.accept) == 0 {
		return -1, fmt.Errorf("%w: pattern %q cannot match any valid output", ErrInvalidGrammar, name)
	}
	numStates := len(dfa.accept)
	entryRules := make([]int, numStates)
	for q := 0; q < numStates; q++ {
		r, err := b.newRule(fmt.Sprintf("%s_s%d", name, q))
		if err != nil {
			return -1, err
		}
		entryRules[q] = r
	}
	if err := b.addProd(entryRules[dfa.start]); err != nil {
		return -1, err
	}

	for q := 0; q < numStates; q++ {
		byTarget := make(map[int]byteSet)
		for ch := 0; ch < 256; ch++ {
			to := dfa.trans[q][ch]
			if to < 0 {
				continue
			}
			if jsonEscaped {
				if esc, ok := jsonEscapeForByte(byte(ch)); ok {
					if err := b.addProd(
						entryRules[to],
						cfgSym{rule: entryRules[q]},
						cfgSym{isTerm: true, term: singleByteSet('\\')},
						cfgSym{isTerm: true, term: singleByteSet(esc)},
					); err != nil {
						return -1, err
					}
					continue
				}
				if ch < 0x20 || ch == '"' || ch == '\\' {
					continue
				}
			}
			bs := byTarget[to]
			bs.add(byte(ch))
			byTarget[to] = bs
		}
		targets := make([]int, 0, len(byTarget))
		for to := range byTarget {
			targets = append(targets, to)
		}
		slices.Sort(targets)
		for _, to := range targets {
			if err := b.addProd(
				entryRules[to],
				cfgSym{rule: entryRules[q]},
				cfgSym{isTerm: true, term: byTarget[to]},
			); err != nil {
				return -1, err
			}
		}
	}

	matchRule, err := b.newRule(name)
	if err != nil {
		return -1, err
	}
	for q := 0; q < numStates; q++ {
		if dfa.accept[q] {
			if err := b.addProd(matchRule, cfgSym{rule: entryRules[q]}); err != nil {
				return -1, err
			}
		}
	}
	return matchRule, nil
}

func jsonEscapeForByte(b byte) (byte, bool) {
	switch b {
	case '"':
		return '"', true
	case '\\':
		return '\\', true
	case '\b':
		return 'b', true
	case '\f':
		return 'f', true
	case '\n':
		return 'n', true
	case '\r':
		return 'r', true
	case '\t':
		return 't', true
	default:
		return 0, false
	}
}

func compileRegexSyntaxDFA(re *syntax.Regexp) (*byteDFA, error) {
	simplified := re.Simplify()
	if regexASTSize(simplified) > maxGrammarRegexProgram {
		return nil, fmt.Errorf("%w: regex syntax tree exceeds %d nodes", ErrUnsupportedGrammar, maxGrammarRegexProgram)
	}
	prog, err := syntax.Compile(simplified)
	if err != nil {
		return nil, fmt.Errorf("%w: regex compile failed: %v", ErrInvalidGrammar, err)
	}
	if len(prog.Inst) > maxGrammarRegexProgram {
		return nil, fmt.Errorf("%w: regex program exceeds %d instructions", ErrUnsupportedGrammar, maxGrammarRegexProgram)
	}
	nfa, err := buildRegexNFA(prog)
	if err != nil {
		return nil, err
	}
	dfa, err := determinizeRegexNFA(nfa, int32(prog.Start))
	if err != nil {
		return nil, err
	}
	trimmed := trimByteDFA(dfa)
	if trimmed.start < 0 {
		return nil, fmt.Errorf("%w: regex matches empty language", ErrInvalidGrammar)
	}
	return trimmed, nil
}

func regexASTSize(re *syntax.Regexp) int {
	if re == nil {
		return 0
	}
	size := 1
	switch re.Op {
	case syntax.OpLiteral:
		size += len(re.Rune)
	case syntax.OpRepeat:
		cnt := re.Max
		if cnt < 0 {
			cnt = re.Min + 1
		}
		size += max(cnt, 1) * regexASTSize(re.Sub[0])
	default:
		for _, sub := range re.Sub {
			size += regexASTSize(sub)
		}
	}
	return min(size, maxGrammarRegexProgram+1)
}

// CompileRegexGrammar compiles a full-match Perl-syntax regular expression
// (via Go regexp/syntax) into a UTF-8 byte-level CompiledGrammar.
func CompileRegexGrammar(pattern string) (*CompiledGrammar, error) {
	if len(pattern) > maxGrammarRegexBytes {
		return nil, fmt.Errorf("%w: regex length %d exceeds limit %d", ErrUnsupportedGrammar, len(pattern), maxGrammarRegexBytes)
	}
	re, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return nil, fmt.Errorf("%w: invalid regex: %v", ErrInvalidGrammar, err)
	}
	if regexASTSize(re) > maxGrammarRegexProgram {
		return nil, fmt.Errorf("%w: expanded regex exceeds limit", ErrUnsupportedGrammar)
	}
	dfa, err := compileRegexSyntaxDFA(re)
	if err != nil {
		return nil, err
	}
	b := newCFGBuilder()
	startRule, err := b.addDFALeftLinear("regex", dfa, false)
	if err != nil {
		return nil, err
	}
	return b.finalize(startRule)
}

// CompileChoiceGrammar compiles a finite closed set of literal string choices
// into a prefix-shared CompiledGrammar.
func CompileChoiceGrammar(choices []string) (*CompiledGrammar, error) {
	if len(choices) == 0 {
		return nil, fmt.Errorf("%w: choices must not be empty", ErrInvalidGrammar)
	}
	sorted := append([]string(nil), choices...)
	slices.Sort(sorted)
	sorted = slices.Compact(sorted)

	b := newCFGBuilder()
	root, err := b.newRule("choice_root")
	if err != nil {
		return nil, err
	}
	if err := buildChoiceTrieRules(b, root, sorted, 0); err != nil {
		return nil, err
	}
	return b.finalize(root)
}

func buildChoiceTrieRules(b *cfgBuilder, ruleID int, choices []string, depth int) error {
	i := 0
	if i < len(choices) && len(choices[i]) == depth {
		if err := b.addProd(ruleID); err != nil {
			return err
		}
		i++
	}
	for i < len(choices) {
		ch := choices[i][depth]
		j := i + 1
		for j < len(choices) && choices[j][depth] == ch {
			j++
		}
		childRule, err := b.newRule(fmt.Sprintf("choice_%d_%d", ruleID, ch))
		if err != nil {
			return err
		}
		if err := b.addProd(ruleID, cfgSym{isTerm: true, term: singleByteSet(ch)}, cfgSym{rule: childRule}); err != nil {
			return err
		}
		if err := buildChoiceTrieRules(b, childRule, choices[i:j], depth+1); err != nil {
			return err
		}
		i = j
	}
	return nil
}

// --- JSON Schema Compiler ---

// GrammarOptions configures structured-output JSON Schema compilation.
type GrammarOptions struct {
	// CompactJSON omits structural JSON whitespace around punctuation so forced
	// literal prefixes can be fast-forwarded cleanly. When false (default),
	// standard RFC 8259 JSON whitespace [ \t\n\r]* is admitted.
	CompactJSON bool
	// StrictAdditionalProperties treats omitted additionalProperties on object
	// schemas with declared properties as false.
	StrictAdditionalProperties bool
}

type jsonSchemaCompiler struct {
	opts     GrammarOptions
	rootRaw  json.RawMessage
	g        *cfgBuilder
	refRules map[string]int

	wsRule     int
	nullRule   int
	boolRule   int
	intRule    int
	numRule    int
	stringRule int
	anyRule    int
	seqCounter int
}

// CompileJSONSchemaGrammar compiles a JSON Schema document into a byte-level
// CompiledGrammar suitable for GrammarMaskEngine.
func CompileJSONSchemaGrammar(schema []byte, opts GrammarOptions) (*CompiledGrammar, error) {
	trimmed := bytes.TrimSpace(schema)
	if len(trimmed) == 0 {
		return nil, fmt.Errorf("%w: empty schema", ErrInvalidGrammar)
	}
	if len(trimmed) > maxGrammarSchemaBytes {
		return nil, fmt.Errorf("%w: schema size %d exceeds %d bytes", ErrUnsupportedGrammar, len(trimmed), maxGrammarSchemaBytes)
	}
	if !json.Valid(trimmed) {
		return nil, fmt.Errorf("%w: malformed JSON schema", ErrInvalidGrammar)
	}

	c := &jsonSchemaCompiler{
		opts:     opts,
		rootRaw:  trimmed,
		g:        newCFGBuilder(),
		refRules: make(map[string]int),
		wsRule:   -1,
		nullRule: -1,
		boolRule: -1,
		intRule:  -1,
		numRule:  -1,
		anyRule:  -1,
	}
	c.stringRule = -1
	rootRule, err := c.compileNode(trimmed, 0)
	if err != nil {
		return nil, err
	}
	return c.g.finalize(rootRule)
}

func (c *jsonSchemaCompiler) getWSRule() (int, error) {
	if c.wsRule >= 0 {
		return c.wsRule, nil
	}
	r, err := c.g.newRule("json_ws")
	if err != nil {
		return -1, err
	}
	if err := c.g.addProd(r); err != nil {
		return -1, err
	}
	if !c.opts.CompactJSON {
		var ws byteSet
		ws.add(' ')
		ws.add('\t')
		ws.add('\n')
		ws.add('\r')
		if err := c.g.addProd(r, cfgSym{rule: r}, cfgSym{isTerm: true, term: ws}); err != nil {
			return -1, err
		}
	}
	c.wsRule = r
	return r, nil
}

func (c *jsonSchemaCompiler) getNullRule() (int, error) {
	if c.nullRule >= 0 {
		return c.nullRule, nil
	}
	r, err := c.g.newRule("json_null")
	if err != nil {
		return -1, err
	}
	if err := c.g.addLiteralProd(r, []byte("null")); err != nil {
		return -1, err
	}
	c.nullRule = r
	return r, nil
}

func (c *jsonSchemaCompiler) getBoolRule() (int, error) {
	if c.boolRule >= 0 {
		return c.boolRule, nil
	}
	r, err := c.g.newRule("json_bool")
	if err != nil {
		return -1, err
	}
	if err := c.g.addLiteralProd(r, []byte("true")); err != nil {
		return -1, err
	}
	if err := c.g.addLiteralProd(r, []byte("false")); err != nil {
		return -1, err
	}
	c.boolRule = r
	return r, nil
}

func (c *jsonSchemaCompiler) getDigitsRules() (int, int, error) {
	nz := byteSet{}
	nz.addRange('1', '9')
	dig := byteSet{}
	dig.addRange('0', '9')

	digStar, err := c.g.namedRule("json_digits_star")
	if err != nil {
		return -1, -1, err
	}
	if len(c.g.rules[digStar].prods) == 0 {
		if err := c.g.addProd(digStar); err != nil {
			return -1, -1, err
		}
		if err := c.g.addProd(digStar, cfgSym{rule: digStar}, cfgSym{isTerm: true, term: dig}); err != nil {
			return -1, -1, err
		}
	}

	intMag, err := c.g.namedRule("json_int_mag")
	if err != nil {
		return -1, -1, err
	}
	if len(c.g.rules[intMag].prods) == 0 {
		if err := c.g.addProd(intMag, cfgSym{isTerm: true, term: singleByteSet('0')}); err != nil {
			return -1, -1, err
		}
		if err := c.g.addProd(intMag, cfgSym{isTerm: true, term: nz}, cfgSym{rule: digStar}); err != nil {
			return -1, -1, err
		}
	}
	return intMag, digStar, nil
}

func (c *jsonSchemaCompiler) getIntRule() (int, error) {
	if c.intRule >= 0 {
		return c.intRule, nil
	}
	intMag, _, err := c.getDigitsRules()
	if err != nil {
		return -1, err
	}
	r, err := c.g.newRule("json_int")
	if err != nil {
		return -1, err
	}
	if err := c.g.addProd(r, cfgSym{rule: intMag}); err != nil {
		return -1, err
	}
	if err := c.g.addProd(r, cfgSym{isTerm: true, term: singleByteSet('-')}, cfgSym{rule: intMag}); err != nil {
		return -1, err
	}
	c.intRule = r
	return r, nil
}

func (c *jsonSchemaCompiler) getNumberRule() (int, error) {
	if c.numRule >= 0 {
		return c.numRule, nil
	}
	intRule, err := c.getIntRule()
	if err != nil {
		return -1, err
	}
	_, digStar, err := c.getDigitsRules()
	if err != nil {
		return -1, err
	}
	var dig, sign, exp byteSet
	dig.addRange('0', '9')
	sign.add('+')
	sign.add('-')
	exp.add('e')
	exp.add('E')

	fracRule, err := c.g.newRule("json_frac")
	if err != nil {
		return -1, err
	}
	if err := c.g.addProd(fracRule); err != nil {
		return -1, err
	}
	if err := c.g.addProd(fracRule, cfgSym{isTerm: true, term: singleByteSet('.')}, cfgSym{isTerm: true, term: dig}, cfgSym{rule: digStar}); err != nil {
		return -1, err
	}

	expRule, err := c.g.newRule("json_exp")
	if err != nil {
		return -1, err
	}
	if err := c.g.addProd(expRule); err != nil {
		return -1, err
	}
	if err := c.g.addProd(expRule, cfgSym{isTerm: true, term: exp}, cfgSym{isTerm: true, term: dig}, cfgSym{rule: digStar}); err != nil {
		return -1, err
	}
	if err := c.g.addProd(expRule, cfgSym{isTerm: true, term: exp}, cfgSym{isTerm: true, term: sign}, cfgSym{isTerm: true, term: dig}, cfgSym{rule: digStar}); err != nil {
		return -1, err
	}

	r, err := c.g.newRule("json_number")
	if err != nil {
		return -1, err
	}
	if err := c.g.addProd(r, cfgSym{rule: intRule}, cfgSym{rule: fracRule}, cfgSym{rule: expRule}); err != nil {
		return -1, err
	}
	c.numRule = r
	return r, nil
}

func (c *jsonSchemaCompiler) getStringCharRule() (int, error) {
	r, err := c.g.namedRule("json_str_char")
	if err != nil {
		return -1, err
	}
	if len(c.g.rules[r].prods) > 0 {
		return r, nil
	}

	var safeASCII byteSet
	safeASCII.addRange(0x20, 0x7F)
	safeASCII[uint8('"')>>6] &^= uint64(1) << (uint8('"') & 63)
	safeASCII[uint8('\\')>>6] &^= uint64(1) << (uint8('\\') & 63)
	if err := c.g.addProd(r, cfgSym{isTerm: true, term: safeASCII}); err != nil {
		return -1, err
	}
	for _, seq := range appendUTF8Sequences(nil, 0x80, utf8.MaxRune) {
		rhs := make([]cfgSym, len(seq))
		for i, br := range seq {
			var bs byteSet
			bs.addRange(br.lo, br.hi)
			rhs[i] = cfgSym{isTerm: true, term: bs}
		}
		if err := c.g.addProd(r, rhs...); err != nil {
			return -1, err
		}
	}

	var esc byteSet
	for _, ch := range []byte{'"', '\\', '/', 'b', 'f', 'n', 'r', 't'} {
		esc.add(ch)
	}
	if err := c.g.addProd(r, cfgSym{isTerm: true, term: singleByteSet('\\')}, cfgSym{isTerm: true, term: esc}); err != nil {
		return -1, err
	}

	var hexSet byteSet
	hexSet.addRange('0', '9')
	hexSet.addRange('a', 'f')
	hexSet.addRange('A', 'F')
	hexSym := cfgSym{isTerm: true, term: hexSet}
	if err := c.g.addProd(r,
		cfgSym{isTerm: true, term: singleByteSet('\\')},
		cfgSym{isTerm: true, term: singleByteSet('u')},
		hexSym, hexSym, hexSym, hexSym,
	); err != nil {
		return -1, err
	}
	return r, nil
}

func (c *jsonSchemaCompiler) getStringRule() (int, error) {
	if c.stringRule >= 0 {
		return c.stringRule, nil
	}
	chRule, err := c.getStringCharRule()
	if err != nil {
		return -1, err
	}
	body, err := c.g.newRule("json_str_body")
	if err != nil {
		return -1, err
	}
	if err := c.g.addProd(body); err != nil {
		return -1, err
	}
	if err := c.g.addProd(body, cfgSym{rule: body}, cfgSym{rule: chRule}); err != nil {
		return -1, err
	}
	r, err := c.g.newRule("json_string")
	if err != nil {
		return -1, err
	}
	q := cfgSym{isTerm: true, term: singleByteSet('"')}
	if err := c.g.addProd(r, q, cfgSym{rule: body}, q); err != nil {
		return -1, err
	}
	c.stringRule = r
	return r, nil
}

func (c *jsonSchemaCompiler) getAnyValueRule() (int, error) {
	if c.anyRule >= 0 {
		return c.anyRule, nil
	}
	r, err := c.g.newRule("json_any")
	if err != nil {
		return -1, err
	}
	c.anyRule = r

	nullR, err := c.getNullRule()
	if err != nil {
		return -1, err
	}
	boolR, err := c.getBoolRule()
	if err != nil {
		return -1, err
	}
	numR, err := c.getNumberRule()
	if err != nil {
		return -1, err
	}
	strR, err := c.getStringRule()
	if err != nil {
		return -1, err
	}
	wsR, err := c.getWSRule()
	if err != nil {
		return -1, err
	}

	arrR, err := c.compileArrayRule(r, 0, -1)
	if err != nil {
		return -1, err
	}

	pairR, err := c.g.newRule("json_any_pair")
	if err != nil {
		return -1, err
	}
	if err := c.g.addProd(pairR,
		cfgSym{rule: strR},
		cfgSym{rule: wsR},
		cfgSym{isTerm: true, term: singleByteSet(':')},
		cfgSym{rule: wsR},
		cfgSym{rule: r},
	); err != nil {
		return -1, err
	}
	pairsTail, err := c.g.newRule("json_any_pairs_tail")
	if err != nil {
		return -1, err
	}
	if err := c.g.addProd(pairsTail); err != nil {
		return -1, err
	}
	if err := c.g.addProd(pairsTail,
		cfgSym{rule: pairsTail},
		cfgSym{rule: wsR},
		cfgSym{isTerm: true, term: singleByteSet(',')},
		cfgSym{rule: wsR},
		cfgSym{rule: pairR},
	); err != nil {
		return -1, err
	}
	objR, err := c.g.newRule("json_any_obj")
	if err != nil {
		return -1, err
	}
	if err := c.g.addProd(objR,
		cfgSym{isTerm: true, term: singleByteSet('{')},
		cfgSym{rule: wsR},
		cfgSym{isTerm: true, term: singleByteSet('}')},
	); err != nil {
		return -1, err
	}
	if err := c.g.addProd(objR,
		cfgSym{isTerm: true, term: singleByteSet('{')},
		cfgSym{rule: wsR},
		cfgSym{rule: pairR},
		cfgSym{rule: pairsTail},
		cfgSym{rule: wsR},
		cfgSym{isTerm: true, term: singleByteSet('}')},
	); err != nil {
		return -1, err
	}

	for _, alt := range []int{nullR, boolR, numR, strR, arrR, objR} {
		if err := c.g.addProd(r, cfgSym{rule: alt}); err != nil {
			return -1, err
		}
	}
	return r, nil
}

func decodeExactJSONObject(raw []byte) (map[string]json.RawMessage, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil || tok != json.Delim('{') {
		return nil, fmt.Errorf("%w: schema node must be a JSON object or boolean", ErrInvalidGrammar)
	}
	out := make(map[string]json.RawMessage)
	for dec.More() {
		ktok, err := dec.Token()
		key, ok := ktok.(string)
		if err != nil || !ok {
			return nil, fmt.Errorf("%w: invalid object key", ErrInvalidGrammar)
		}
		if _, dup := out[key]; dup {
			return nil, fmt.Errorf("%w: duplicate key %q in schema", ErrInvalidGrammar, key)
		}
		var val json.RawMessage
		if err := dec.Decode(&val); err != nil {
			return nil, fmt.Errorf("%w: invalid value for key %q: %v", ErrInvalidGrammar, key, err)
		}
		out[key] = bytes.TrimSpace(val)
	}
	tok, err = dec.Token()
	if err != nil || tok != json.Delim('}') {
		return nil, fmt.Errorf("%w: malformed JSON object", ErrInvalidGrammar)
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, fmt.Errorf("%w: trailing JSON content", ErrInvalidGrammar)
	}
	return out, nil
}

func resolveJSONPointer(root json.RawMessage, ref string) (json.RawMessage, error) {
	if ref == "#" {
		return root, nil
	}
	if !strings.HasPrefix(ref, "#/") {
		return nil, fmt.Errorf("%w: external or non-pointer $ref %q is not supported", ErrUnsupportedGrammar, ref)
	}
	parts := strings.Split(ref[2:], "/")
	cur := bytes.TrimSpace(root)
	for _, rawPart := range parts {
		part := strings.ReplaceAll(strings.ReplaceAll(rawPart, "~1", "/"), "~0", "~")
		if len(cur) == 0 {
			return nil, fmt.Errorf("%w: unresolved $ref %q", ErrInvalidGrammar, ref)
		}
		switch cur[0] {
		case '{':
			obj, err := decodeExactJSONObject(cur)
			if err != nil {
				return nil, err
			}
			next, ok := obj[part]
			if !ok {
				return nil, fmt.Errorf("%w: $ref %q target not found", ErrInvalidGrammar, ref)
			}
			cur = next
		case '[':
			var arr []json.RawMessage
			if err := json.Unmarshal(cur, &arr); err != nil {
				return nil, fmt.Errorf("%w: invalid array during $ref %q lookup", ErrInvalidGrammar, ref)
			}
			idx, err := strconv.Atoi(part)
			if err != nil || idx < 0 || idx >= len(arr) {
				return nil, fmt.Errorf("%w: $ref %q index %q out of range", ErrInvalidGrammar, ref, part)
			}
			cur = bytes.TrimSpace(arr[idx])
		default:
			return nil, fmt.Errorf("%w: $ref %q traverses non-container", ErrInvalidGrammar, ref)
		}
	}
	return cur, nil
}

func isSchemaAnnotationKey(k string) bool {
	switch k {
	case "$schema", "$id", "$comment", "$vocabulary", "$anchor",
		"title", "description", "default", "examples",
		"readOnly", "writeOnly", "deprecated",
		"$defs", "definitions":
		return true
	default:
		return false
	}
}

func (c *jsonSchemaCompiler) compileNode(raw json.RawMessage, depth int) (int, error) {
	if depth > maxGrammarSchemaDepth {
		return -1, fmt.Errorf("%w: schema nesting exceeds depth %d", ErrUnsupportedGrammar, maxGrammarSchemaDepth)
	}
	trimmed := bytes.TrimSpace(raw)
	if bytes.Equal(trimmed, []byte("true")) {
		return c.getAnyValueRule()
	}
	if bytes.Equal(trimmed, []byte("false")) {
		return -1, fmt.Errorf("%w: false schema is unsatisfiable", ErrInvalidGrammar)
	}
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return -1, fmt.Errorf("%w: schema must be an object or boolean", ErrInvalidGrammar)
	}

	obj, err := decodeExactJSONObject(trimmed)
	if err != nil {
		return -1, err
	}

	for k := range obj {
		if isSchemaAnnotationKey(k) {
			continue
		}
		switch k {
		case "$ref", "type", "const", "enum", "anyOf", "oneOf", "allOf",
			"properties", "required", "additionalProperties",
			"items", "prefixItems", "minItems", "maxItems",
			"minLength", "maxLength", "pattern":
		default:
			return -1, fmt.Errorf("%w: JSON Schema keyword %q", ErrUnsupportedGrammar, k)
		}
	}

	if refRaw, hasRef := obj["$ref"]; hasRef {
		for k := range obj {
			if k != "$ref" && !isSchemaAnnotationKey(k) {
				return -1, fmt.Errorf("%w: $ref with sibling constraining keyword %q", ErrUnsupportedGrammar, k)
			}
		}
		var ref string
		if err := json.Unmarshal(refRaw, &ref); err != nil || ref == "" {
			return -1, fmt.Errorf("%w: invalid $ref value", ErrInvalidGrammar)
		}
		if rid, ok := c.refRules[ref]; ok {
			return rid, nil
		}
		target, err := resolveJSONPointer(c.rootRaw, ref)
		if err != nil {
			return -1, err
		}
		rid, err := c.g.newRule("ref_" + ref)
		if err != nil {
			return -1, err
		}
		c.refRules[ref] = rid
		targetRule, err := c.compileNode(target, depth+1)
		if err != nil {
			return -1, err
		}
		if err := c.g.addProd(rid, cfgSym{rule: targetRule}); err != nil {
			return -1, err
		}
		return rid, nil
	}

	if allRaw, hasAll := obj["allOf"]; hasAll {
		for k := range obj {
			if k != "allOf" && !isSchemaAnnotationKey(k) {
				return -1, fmt.Errorf("%w: allOf with sibling keyword %q", ErrUnsupportedGrammar, k)
			}
		}
		var subs []json.RawMessage
		if err := json.Unmarshal(allRaw, &subs); err != nil || len(subs) == 0 {
			return -1, fmt.Errorf("%w: invalid allOf", ErrInvalidGrammar)
		}
		if len(subs) > 1 {
			return -1, fmt.Errorf("%w: multi-branch allOf intersection is not supported", ErrUnsupportedGrammar)
		}
		return c.compileNode(subs[0], depth+1)
	}

	for _, comb := range []string{"anyOf", "oneOf"} {
		if combRaw, ok := obj[comb]; ok {
			for k := range obj {
				if k != comb && !isSchemaAnnotationKey(k) {
					return -1, fmt.Errorf("%w: %s with sibling constraining keyword %q", ErrUnsupportedGrammar, comb, k)
				}
			}
			var subs []json.RawMessage
			if err := json.Unmarshal(combRaw, &subs); err != nil || len(subs) == 0 {
				return -1, fmt.Errorf("%w: invalid %s array", ErrInvalidGrammar, comb)
			}
			r, err := c.g.newRule(comb)
			if err != nil {
				return -1, err
			}
			for _, sub := range subs {
				sr, err := c.compileNode(sub, depth+1)
				if err != nil {
					return -1, err
				}
				if err := c.g.addProd(r, cfgSym{rule: sr}); err != nil {
					return -1, err
				}
			}
			return r, nil
		}
	}

	if constRaw, hasConst := obj["const"]; hasConst {
		for k := range obj {
			if k != "const" && k != "type" && !isSchemaAnnotationKey(k) {
				return -1, fmt.Errorf("%w: const with sibling keyword %q", ErrUnsupportedGrammar, k)
			}
		}
		return c.compileLiteralValues([]json.RawMessage{constRaw}, obj["type"])
	}

	if enumRaw, hasEnum := obj["enum"]; hasEnum {
		for k := range obj {
			if k != "enum" && k != "type" && !isSchemaAnnotationKey(k) {
				return -1, fmt.Errorf("%w: enum with sibling keyword %q", ErrUnsupportedGrammar, k)
			}
		}
		var vals []json.RawMessage
		if err := json.Unmarshal(enumRaw, &vals); err != nil || len(vals) == 0 {
			return -1, fmt.Errorf("%w: enum must be a non-empty array", ErrInvalidGrammar)
		}
		return c.compileLiteralValues(vals, obj["type"])
	}

	types, hasType, err := parseSchemaTypes(obj)
	if err != nil {
		return -1, err
	}
	if !hasType {
		// No explicit type: infer object/array/string if shape keywords exist, else any JSON value.
		switch {
		case hasAnyKey(obj, "properties", "required", "additionalProperties"):
			return c.compileObjectSchema(obj, depth)
		case hasAnyKey(obj, "items", "prefixItems", "minItems", "maxItems"):
			return c.compileArraySchema(obj, depth)
		case hasAnyKey(obj, "pattern", "minLength", "maxLength"):
			return c.compileStringSchema(obj)
		default:
			return c.getAnyValueRule()
		}
	}

	if len(types) == 1 {
		return c.compileTypedSchema(types[0], obj, depth)
	}
	r, err := c.g.newRule("type_union")
	if err != nil {
		return -1, err
	}
	for _, typ := range types {
		subR, err := c.compileTypedSchema(typ, obj, depth)
		if err != nil {
			return -1, err
		}
		if err := c.g.addProd(r, cfgSym{rule: subR}); err != nil {
			return -1, err
		}
	}
	return r, nil
}

func hasAnyKey(m map[string]json.RawMessage, keys ...string) bool {
	for _, k := range keys {
		if _, ok := m[k]; ok {
			return true
		}
	}
	return false
}

func parseSchemaTypes(obj map[string]json.RawMessage) ([]string, bool, error) {
	raw, ok := obj["type"]
	if !ok {
		return nil, false, nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if err := validateSchemaTypeName(single); err != nil {
			return nil, false, err
		}
		return []string{single}, true, nil
	}
	var multi []string
	if err := json.Unmarshal(raw, &multi); err == nil && len(multi) > 0 {
		seen := make(map[string]bool, len(multi))
		out := make([]string, 0, len(multi))
		for _, typ := range multi {
			if err := validateSchemaTypeName(typ); err != nil {
				return nil, false, err
			}
			if !seen[typ] {
				seen[typ] = true
				out = append(out, typ)
			}
		}
		return out, true, nil
	}
	return nil, false, fmt.Errorf("%w: invalid type keyword", ErrInvalidGrammar)
}

func validateSchemaTypeName(typ string) error {
	switch typ {
	case "object", "array", "string", "integer", "number", "boolean", "null":
		return nil
	default:
		return fmt.Errorf("%w: unsupported JSON Schema type %q", ErrUnsupportedGrammar, typ)
	}
}

func (c *jsonSchemaCompiler) compileTypedSchema(typ string, obj map[string]json.RawMessage, depth int) (int, error) {
	switch typ {
	case "null":
		return c.getNullRule()
	case "boolean":
		return c.getBoolRule()
	case "integer":
		return c.getIntRule()
	case "number":
		return c.getNumberRule()
	case "string":
		return c.compileStringSchema(obj)
	case "array":
		return c.compileArraySchema(obj, depth)
	case "object":
		return c.compileObjectSchema(obj, depth)
	default:
		return -1, fmt.Errorf("%w: unsupported JSON Schema type %q", ErrUnsupportedGrammar, typ)
	}
}

func (c *jsonSchemaCompiler) compileLiteralValues(rawVals []json.RawMessage, typeRaw json.RawMessage) (int, error) {
	var allowedTypes map[string]bool
	if len(typeRaw) > 0 {
		types, _, err := parseSchemaTypes(map[string]json.RawMessage{"type": typeRaw})
		if err != nil {
			return -1, err
		}
		allowedTypes = make(map[string]bool, len(types))
		for _, t := range types {
			allowedTypes[t] = true
		}
	}

	r, err := c.g.newRule("json_literals")
	if err != nil {
		return -1, err
	}
	seen := make(map[string]bool, len(rawVals))
	matched := 0
	for _, rawVal := range rawVals {
		lit, kind, err := canonicalJSONLiteral(rawVal)
		if err != nil {
			return -1, err
		}
		if allowedTypes != nil && !matchesJSONType(kind, rawVal, allowedTypes) {
			continue
		}
		if seen[lit] {
			continue
		}
		seen[lit] = true
		matched++
		if err := c.g.addLiteralProd(r, []byte(lit)); err != nil {
			return -1, err
		}
	}
	if matched == 0 {
		return -1, fmt.Errorf("%w: enum/const has no values satisfying type constraint", ErrInvalidGrammar)
	}
	return r, nil
}

func canonicalJSONLiteral(raw json.RawMessage) (string, string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return "", "", fmt.Errorf("%w: invalid enum/const literal: %v", ErrInvalidGrammar, err)
	}
	if _, err := dec.Token(); err != io.EOF {
		return "", "", fmt.Errorf("%w: trailing token in enum/const literal", ErrInvalidGrammar)
	}
	switch x := v.(type) {
	case nil:
		return "null", "null", nil
	case bool:
		if x {
			return "true", "boolean", nil
		}
		return "false", "boolean", nil
	case string:
		b, err := json.Marshal(x)
		if err != nil {
			return "", "", err
		}
		return string(b), "string", nil
	case json.Number:
		return x.String(), "number", nil
	default:
		var buf bytes.Buffer
		if err := json.Compact(&buf, raw); err != nil {
			return "", "", fmt.Errorf("%w: invalid literal: %v", ErrInvalidGrammar, err)
		}
		s := buf.String()
		if strings.HasPrefix(s, "{") {
			return s, "object", nil
		}
		return s, "array", nil
	}
}

func matchesJSONType(kind string, raw json.RawMessage, allowed map[string]bool) bool {
	if allowed[kind] {
		return true
	}
	if kind == "number" && allowed["integer"] {
		trimmed := bytes.TrimSpace(raw)
		return !bytes.ContainsAny(trimmed, ".eE")
	}
	return false
}

func (c *jsonSchemaCompiler) compileStringSchema(obj map[string]json.RawMessage) (int, error) {
	if patRaw, hasPat := obj["pattern"]; hasPat {
		if _, hasMin := obj["minLength"]; hasMin {
			return -1, fmt.Errorf("%w: combining pattern with minLength/maxLength is not supported", ErrUnsupportedGrammar)
		}
		if _, hasMax := obj["maxLength"]; hasMax {
			return -1, fmt.Errorf("%w: combining pattern with minLength/maxLength is not supported", ErrUnsupportedGrammar)
		}
		var pat string
		if err := json.Unmarshal(patRaw, &pat); err != nil {
			return -1, fmt.Errorf("%w: invalid string pattern", ErrInvalidGrammar)
		}
		if len(pat) > maxGrammarRegexBytes {
			return -1, fmt.Errorf("%w: string pattern exceeds %d bytes", ErrUnsupportedGrammar, maxGrammarRegexBytes)
		}
		re, err := syntax.Parse(pat, syntax.Perl)
		if err != nil {
			return -1, fmt.Errorf("%w: invalid string pattern: %v", ErrInvalidGrammar, err)
		}
		// JSON Schema patterns match anywhere inside the string unless anchored
		// with ^ / $.
		anyChar := &syntax.Regexp{Op: syntax.OpAnyChar}
		anyStar := &syntax.Regexp{Op: syntax.OpStar, Sub: []*syntax.Regexp{anyChar}}
		wrapped := &syntax.Regexp{
			Op:  syntax.OpConcat,
			Sub: []*syntax.Regexp{anyStar, re, anyStar},
		}
		dfa, err := compileRegexSyntaxDFA(wrapped)
		if err != nil {
			return -1, err
		}
		// Trim transitions not representable without \uXXXX before left-linear encoding.
		for q := range dfa.trans {
			for b := 0; b < 256; b++ {
				if dfa.trans[q][b] < 0 {
					continue
				}
				if _, isEsc := jsonEscapeForByte(byte(b)); isEsc {
					continue
				}
				if b < 0x20 {
					dfa.trans[q][b] = -1
				}
			}
		}
		dfa = trimByteDFA(dfa)
		bodyRule, err := c.g.addDFALeftLinear("json_pat_body", dfa, true)
		if err != nil {
			return -1, err
		}
		r, err := c.g.newRule("json_pattern_string")
		if err != nil {
			return -1, err
		}
		q := cfgSym{isTerm: true, term: singleByteSet('"')}
		if err := c.g.addProd(r, q, cfgSym{rule: bodyRule}, q); err != nil {
			return -1, err
		}
		return r, nil
	}

	minLen, hasMin, err := parseNonNegBound(obj, "minLength", maxGrammarStringBound)
	if err != nil {
		return -1, err
	}
	maxLen, hasMax, err := parseNonNegBound(obj, "maxLength", maxGrammarStringBound)
	if err != nil {
		return -1, err
	}
	if !hasMin && !hasMax {
		return c.getStringRule()
	}
	if hasMax && minLen > maxLen {
		return -1, fmt.Errorf("%w: minLength %d > maxLength %d", ErrInvalidGrammar, minLen, maxLen)
	}
	chRule, err := c.getStringCharRule()
	if err != nil {
		return -1, err
	}
	optChar, err := c.g.newRule("json_str_opt_char")
	if err != nil {
		return -1, err
	}
	if len(c.g.rules[optChar].prods) == 0 {
		if err := c.g.addProd(optChar); err != nil {
			return -1, err
		}
		if err := c.g.addProd(optChar, cfgSym{rule: chRule}); err != nil {
			return -1, err
		}
	}

	r, err := c.g.newRule("json_bounded_string")
	if err != nil {
		return -1, err
	}
	q := cfgSym{isTerm: true, term: singleByteSet('"')}
	rhs := make([]cfgSym, 0, minLen+maxLen+4)
	rhs = append(rhs, q)
	for i := 0; i < minLen; i++ {
		rhs = append(rhs, cfgSym{rule: chRule})
	}
	if hasMax {
		for i := minLen; i < maxLen; i++ {
			rhs = append(rhs, cfgSym{rule: optChar})
		}
	} else {
		body, err := c.g.newRule("json_str_tail_body")
		if err != nil {
			return -1, err
		}
		if err := c.g.addProd(body); err != nil {
			return -1, err
		}
		if err := c.g.addProd(body, cfgSym{rule: body}, cfgSym{rule: chRule}); err != nil {
			return -1, err
		}
		rhs = append(rhs, cfgSym{rule: body})
	}
	rhs = append(rhs, q)
	if err := c.g.addProd(r, rhs...); err != nil {
		return -1, err
	}
	return r, nil
}

func parseNonNegBound(obj map[string]json.RawMessage, key string, limit int) (int, bool, error) {
	raw, ok := obj[key]
	if !ok {
		return 0, false, nil
	}
	var val int
	if err := json.Unmarshal(raw, &val); err != nil || val < 0 {
		return 0, false, fmt.Errorf("%w: invalid %s", ErrInvalidGrammar, key)
	}
	if val > limit {
		return 0, false, fmt.Errorf("%w: %s %d exceeds limit %d", ErrUnsupportedGrammar, key, val, limit)
	}
	return val, true, nil
}

func (c *jsonSchemaCompiler) compileArraySchema(obj map[string]json.RawMessage, depth int) (int, error) {
	minItems, _, err := parseNonNegBound(obj, "minItems", maxGrammarArrayBound)
	if err != nil {
		return -1, err
	}
	maxItems, hasMax, err := parseNonNegBound(obj, "maxItems", maxGrammarArrayBound)
	if err != nil {
		return -1, err
	}
	if !hasMax {
		maxItems = -1
	} else if minItems > maxItems {
		return -1, fmt.Errorf("%w: minItems %d > maxItems %d", ErrInvalidGrammar, minItems, maxItems)
	}

	if prefixRaw, hasPrefix := obj["prefixItems"]; hasPrefix {
		if _, hasItems := obj["items"]; hasItems {
			return -1, fmt.Errorf("%w: combining prefixItems with items is not supported", ErrUnsupportedGrammar)
		}
		var tupleSchemas []json.RawMessage
		if err := json.Unmarshal(prefixRaw, &tupleSchemas); err != nil {
			return -1, fmt.Errorf("%w: invalid prefixItems", ErrInvalidGrammar)
		}
		elemRules := make([]int, len(tupleSchemas))
		for i, s := range tupleSchemas {
			er, err := c.compileNode(s, depth+1)
			if err != nil {
				return -1, err
			}
			elemRules[i] = er
		}
		return c.compileTupleArrayRule(elemRules, minItems, maxItems)
	}

	var itemRule int
	if itemsRaw, hasItems := obj["items"]; hasItems {
		itemRule, err = c.compileNode(itemsRaw, depth+1)
		if err != nil {
			return -1, err
		}
	} else {
		itemRule, err = c.getAnyValueRule()
		if err != nil {
			return -1, err
		}
	}
	return c.compileArrayRule(itemRule, minItems, maxItems)
}

func (c *jsonSchemaCompiler) compileTupleArrayRule(elemRules []int, minItems, maxItems int) (int, error) {
	if maxItems >= 0 && maxItems < len(elemRules) {
		elemRules = elemRules[:maxItems]
	}
	if minItems > len(elemRules) {
		return -1, fmt.Errorf("%w: minItems %d exceeds prefixItems length %d", ErrInvalidGrammar, minItems, len(elemRules))
	}
	wsR, err := c.getWSRule()
	if err != nil {
		return -1, err
	}
	arrR, err := c.g.newRule("json_tuple_array")
	if err != nil {
		return -1, err
	}
	maxCount := len(elemRules)
	for n := minItems; n <= maxCount; n++ {
		if n == 0 {
			if err := c.g.addProd(arrR,
				cfgSym{isTerm: true, term: singleByteSet('[')},
				cfgSym{rule: wsR},
				cfgSym{isTerm: true, term: singleByteSet(']')},
			); err != nil {
				return -1, err
			}
			continue
		}
		rhs := []cfgSym{{isTerm: true, term: singleByteSet('[')}, {rule: wsR}}
		for i := 0; i < n; i++ {
			if i > 0 {
				rhs = append(rhs,
					cfgSym{rule: wsR},
					cfgSym{isTerm: true, term: singleByteSet(',')},
					cfgSym{rule: wsR},
				)
			}
			rhs = append(rhs, cfgSym{rule: elemRules[i]})
		}
		rhs = append(rhs, cfgSym{rule: wsR}, cfgSym{isTerm: true, term: singleByteSet(']')})
		if err := c.g.addProd(arrR, rhs...); err != nil {
			return -1, err
		}
	}
	return arrR, nil
}

func (c *jsonSchemaCompiler) compileArrayRule(itemRule, minItems, maxItems int) (int, error) {
	wsR, err := c.getWSRule()
	if err != nil {
		return -1, err
	}
	arrR, err := c.g.newRule("json_array")
	if err != nil {
		return -1, err
	}
	if maxItems == 0 {
		if err := c.g.addProd(arrR,
			cfgSym{isTerm: true, term: singleByteSet('[')},
			cfgSym{rule: wsR},
			cfgSym{isTerm: true, term: singleByteSet(']')},
		); err != nil {
			return -1, err
		}
		return arrR, nil
	}

	if maxItems >= 0 {
		for n := minItems; n <= maxItems; n++ {
			if n == 0 {
				if err := c.g.addProd(arrR,
					cfgSym{isTerm: true, term: singleByteSet('[')},
					cfgSym{rule: wsR},
					cfgSym{isTerm: true, term: singleByteSet(']')},
				); err != nil {
					return -1, err
				}
				continue
			}
			rhs := []cfgSym{{isTerm: true, term: singleByteSet('[')}, {rule: wsR}}
			for i := 0; i < n; i++ {
				if i > 0 {
					rhs = append(rhs,
						cfgSym{rule: wsR},
						cfgSym{isTerm: true, term: singleByteSet(',')},
						cfgSym{rule: wsR},
					)
				}
				rhs = append(rhs, cfgSym{rule: itemRule})
			}
			rhs = append(rhs, cfgSym{rule: wsR}, cfgSym{isTerm: true, term: singleByteSet(']')})
			if err := c.g.addProd(arrR, rhs...); err != nil {
				return -1, err
			}
		}
		return arrR, nil
	}

	// Unbounded maxItems (-1) with optional minItems >= 0.
	c.seqCounter++
	tailR, err := c.g.newRule(fmt.Sprintf("json_arr_tail_%d", c.seqCounter))
	if err != nil {
		return -1, err
	}
	if err := c.g.addProd(tailR); err != nil {
		return -1, err
	}
	if err := c.g.addProd(tailR,
		cfgSym{rule: tailR},
		cfgSym{rule: wsR},
		cfgSym{isTerm: true, term: singleByteSet(',')},
		cfgSym{rule: wsR},
		cfgSym{rule: itemRule},
	); err != nil {
		return -1, err
	}

	if minItems == 0 {
		if err := c.g.addProd(arrR,
			cfgSym{isTerm: true, term: singleByteSet('[')},
			cfgSym{rule: wsR},
			cfgSym{isTerm: true, term: singleByteSet(']')},
		); err != nil {
			return -1, err
		}
		if err := c.g.addProd(arrR,
			cfgSym{isTerm: true, term: singleByteSet('[')},
			cfgSym{rule: wsR},
			cfgSym{rule: itemRule},
			cfgSym{rule: tailR},
			cfgSym{rule: wsR},
			cfgSym{isTerm: true, term: singleByteSet(']')},
		); err != nil {
			return -1, err
		}
		return arrR, nil
	}

	rhs := []cfgSym{{isTerm: true, term: singleByteSet('[')}, {rule: wsR}}
	for i := 0; i < minItems; i++ {
		if i > 0 {
			rhs = append(rhs,
				cfgSym{rule: wsR},
				cfgSym{isTerm: true, term: singleByteSet(',')},
				cfgSym{rule: wsR},
			)
		}
		rhs = append(rhs, cfgSym{rule: itemRule})
	}
	rhs = append(rhs, cfgSym{rule: tailR}, cfgSym{rule: wsR}, cfgSym{isTerm: true, term: singleByteSet(']')})
	if err := c.g.addProd(arrR, rhs...); err != nil {
		return -1, err
	}
	return arrR, nil
}

func (c *jsonSchemaCompiler) compileObjectSchema(obj map[string]json.RawMessage, depth int) (int, error) {
	wsR, err := c.getWSRule()
	if err != nil {
		return -1, err
	}

	var props map[string]json.RawMessage
	if propsRaw, ok := obj["properties"]; ok {
		if !bytes.Equal(bytes.TrimSpace(propsRaw), []byte("null")) {
			props, err = decodeExactJSONObject(propsRaw)
			if err != nil {
				return -1, fmt.Errorf("%w: properties must be an object", ErrInvalidGrammar)
			}
		}
	}

	reqSet := make(map[string]bool)
	if reqRaw, ok := obj["required"]; ok {
		var reqList []string
		if err := json.Unmarshal(reqRaw, &reqList); err != nil {
			return -1, fmt.Errorf("%w: required must be an array of strings", ErrInvalidGrammar)
		}
		for _, name := range reqList {
			reqSet[name] = true
		}
	}

	allowExtra := !c.opts.StrictAdditionalProperties
	extraValRule := -1
	if addRaw, ok := obj["additionalProperties"]; ok {
		trimmedAdd := bytes.TrimSpace(addRaw)
		switch {
		case bytes.Equal(trimmedAdd, []byte("false")):
			allowExtra = false
		case bytes.Equal(trimmedAdd, []byte("true")):
			allowExtra = true
			extraValRule, err = c.getAnyValueRule()
			if err != nil {
				return -1, err
			}
		default:
			allowExtra = true
			extraValRule, err = c.compileNode(trimmedAdd, depth+1)
			if err != nil {
				return -1, err
			}
		}
	} else if allowExtra {
		extraValRule, err = c.getAnyValueRule()
		if err != nil {
			return -1, err
		}
	}

	for reqName := range reqSet {
		if _, declared := props[reqName]; !declared {
			if !allowExtra {
				return -1, fmt.Errorf("%w: required property %q not declared when additionalProperties is false", ErrInvalidGrammar, reqName)
			}
			if props == nil {
				props = make(map[string]json.RawMessage)
			}
			if addRaw, hasAdd := obj["additionalProperties"]; hasAdd && !bytes.Equal(bytes.TrimSpace(addRaw), []byte("true")) {
				props[reqName] = addRaw
			} else {
				props[reqName] = json.RawMessage("true")
			}
		}
	}

	names := make([]string, 0, len(props))
	for name := range props {
		names = append(names, name)
	}
	slices.Sort(names)

	type compiledProp struct {
		keyLit []byte
		valR   int
		req    bool
	}
	cProps := make([]compiledProp, len(names))
	for i, name := range names {
		valR, err := c.compileNode(props[name], depth+1)
		if err != nil {
			return -1, err
		}
		keyBytes, err := json.Marshal(name)
		if err != nil {
			return -1, err
		}
		cProps[i] = compiledProp{
			keyLit: keyBytes,
			valR:   valR,
			req:    reqSet[name],
		}
	}

	c.seqCounter++
	seqID := c.seqCounter

	// Compile linear 2*(N+1) tail chain for canonical sorted property order
	// with optional/required properties, then optionally allow additionalPairs.
	var extraPairRule int
	if allowExtra {
		strR, err := c.getStringRule()
		if err != nil {
			return -1, err
		}
		extraPairRule, err = c.g.newRule(fmt.Sprintf("json_extra_pair_%d", seqID))
		if err != nil {
			return -1, err
		}
		if err := c.g.addProd(extraPairRule,
			cfgSym{rule: strR},
			cfgSym{rule: wsR},
			cfgSym{isTerm: true, term: singleByteSet(':')},
			cfgSym{rule: wsR},
			cfgSym{rule: extraValRule},
		); err != nil {
			return -1, err
		}
	}

	n := len(cProps)
	tailFalse := make([]int, n+1)
	tailTrue := make([]int, n+1)
	for i := 0; i <= n; i++ {
		tf, err := c.g.newRule(fmt.Sprintf("json_obj_%d_t0_%d", seqID, i))
		if err != nil {
			return -1, err
		}
		tt, err := c.g.newRule(fmt.Sprintf("json_obj_%d_t1_%d", seqID, i))
		if err != nil {
			return -1, err
		}
		tailFalse[i] = tf
		tailTrue[i] = tt
	}

	// Base state i == n: close object, or accept arbitrary additional pairs when allowed.
	if err := c.g.addProd(tailFalse[n], cfgSym{isTerm: true, term: singleByteSet('}')}); err != nil {
		return -1, err
	}
	if err := c.g.addProd(tailTrue[n], cfgSym{rule: wsR}, cfgSym{isTerm: true, term: singleByteSet('}')}); err != nil {
		return -1, err
	}
	if allowExtra {
		if err := c.g.addProd(tailFalse[n],
			cfgSym{rule: extraPairRule},
			cfgSym{rule: tailTrue[n]},
		); err != nil {
			return -1, err
		}
		if err := c.g.addProd(tailTrue[n],
			cfgSym{rule: wsR},
			cfgSym{isTerm: true, term: singleByteSet(',')},
			cfgSym{rule: wsR},
			cfgSym{rule: extraPairRule},
			cfgSym{rule: tailTrue[n]},
		); err != nil {
			return -1, err
		}
	}

	for i := n - 1; i >= 0; i-- {
		prop := cProps[i]
		keySyms := make([]cfgSym, len(prop.keyLit))
		for bi, ch := range prop.keyLit {
			keySyms[bi] = cfgSym{isTerm: true, term: singleByteSet(ch)}
		}
		// Case needComma == false:
		rhs0 := append([]cfgSym(nil), keySyms...)
		rhs0 = append(rhs0,
			cfgSym{rule: wsR},
			cfgSym{isTerm: true, term: singleByteSet(':')},
			cfgSym{rule: wsR},
			cfgSym{rule: prop.valR},
			cfgSym{rule: tailTrue[i+1]},
		)
		if err := c.g.addProd(tailFalse[i], rhs0...); err != nil {
			return -1, err
		}
		// Case needComma == true:
		rhs1 := []cfgSym{
			{rule: wsR},
			{isTerm: true, term: singleByteSet(',')},
			{rule: wsR},
		}
		rhs1 = append(rhs1, keySyms...)
		rhs1 = append(rhs1,
			cfgSym{rule: wsR},
			cfgSym{isTerm: true, term: singleByteSet(':')},
			cfgSym{rule: wsR},
			cfgSym{rule: prop.valR},
			cfgSym{rule: tailTrue[i+1]},
		)
		if err := c.g.addProd(tailTrue[i], rhs1...); err != nil {
			return -1, err
		}
		// Optional properties can also be skipped without advancing comma state.
		if !prop.req {
			if err := c.g.addProd(tailFalse[i], cfgSym{rule: tailFalse[i+1]}); err != nil {
				return -1, err
			}
			if err := c.g.addProd(tailTrue[i], cfgSym{rule: tailTrue[i+1]}); err != nil {
				return -1, err
			}
		}
	}

	objR, err := c.g.newRule(fmt.Sprintf("json_object_%d", seqID))
	if err != nil {
		return -1, err
	}
	if err := c.g.addProd(objR,
		cfgSym{isTerm: true, term: singleByteSet('{')},
		cfgSym{rule: wsR},
		cfgSym{rule: tailFalse[0]},
	); err != nil {
		return -1, err
	}
	return objR, nil
}

// --- EBNF / CFG Text Grammar Compiler ---

type cfgTokKind int

const (
	cfgTokEOF cfgTokKind = iota
	cfgTokIdent
	cfgTokDef
	cfgTokPipe
	cfgTokLParen
	cfgTokRParen
	cfgTokStar
	cfgTokPlus
	cfgTokQuest
	cfgTokLiteral
	cfgTokCharClass
)

type cfgToken struct {
	kind   cfgTokKind
	text   string
	negate bool
	ranges []runeRange
	pos    int
}

// CompileCFGGrammar compiles an EBNF/GBNF-style context-free grammar into a
// CompiledGrammar. Supports rule definitions (name ::= ... or name: ...),
// alternation (|), grouping (...), postfix quantifiers (*, +, ?), quoted string
// literals ("..." or '...'), and character classes ([a-z], [^"\\]).
func CompileCFGGrammar(source string) (*CompiledGrammar, error) {
	if len(source) > maxGrammarSchemaBytes {
		return nil, fmt.Errorf("%w: CFG source exceeds %d bytes", ErrUnsupportedGrammar, maxGrammarSchemaBytes)
	}
	toks, err := lexCFGGrammar(source)
	if err != nil {
		return nil, err
	}
	if len(toks) == 0 || (len(toks) == 1 && toks[0].kind == cfgTokEOF) {
		return nil, fmt.Errorf("%w: empty CFG grammar", ErrInvalidGrammar)
	}
	p := &cfgTextParser{
		toks: toks,
		b:    newCFGBuilder(),
	}
	if err := p.parseAll(); err != nil {
		return nil, err
	}
	startRule := -1
	if id, ok := p.b.ruleMap["root"]; ok {
		startRule = id
	} else if id, ok := p.b.ruleMap["start"]; ok {
		startRule = id
	} else if p.firstRule >= 0 {
		startRule = p.firstRule
	}
	if startRule < 0 {
		return nil, fmt.Errorf("%w: no rules declared", ErrInvalidGrammar)
	}
	for name, id := range p.b.ruleMap {
		if len(p.b.rules[id].prods) == 0 {
			return nil, fmt.Errorf("%w: undefined rule %q", ErrInvalidGrammar, name)
		}
	}
	return p.b.finalize(startRule)
}

func lexCFGGrammar(src string) ([]cfgToken, error) {
	var toks []cfgToken
	i := 0
	for i < len(src) {
		ch := src[i]
		if ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n' {
			i++
			continue
		}
		if ch == '#' {
			for i < len(src) && src[i] != '\n' {
				i++
			}
			continue
		}
		if ch == '/' && i+1 < len(src) && src[i+1] == '/' {
			for i < len(src) && src[i] != '\n' {
				i++
			}
			continue
		}
		if strings.HasPrefix(src[i:], "::=") {
			toks = append(toks, cfgToken{kind: cfgTokDef, pos: i})
			i += 3
			continue
		}
		switch ch {
		case ':':
			toks = append(toks, cfgToken{kind: cfgTokDef, pos: i})
			i++
		case '|':
			toks = append(toks, cfgToken{kind: cfgTokPipe, pos: i})
			i++
		case '(':
			toks = append(toks, cfgToken{kind: cfgTokLParen, pos: i})
			i++
		case ')':
			toks = append(toks, cfgToken{kind: cfgTokRParen, pos: i})
			i++
		case '*':
			toks = append(toks, cfgToken{kind: cfgTokStar, pos: i})
			i++
		case '+':
			toks = append(toks, cfgToken{kind: cfgTokPlus, pos: i})
			i++
		case '?':
			toks = append(toks, cfgToken{kind: cfgTokQuest, pos: i})
			i++
		case '"', '\'':
			lit, next, err := lexCFGLiteral(src, i)
			if err != nil {
				return nil, err
			}
			toks = append(toks, cfgToken{kind: cfgTokLiteral, text: lit, pos: i})
			i = next
		case '[':
			neg, ranges, next, err := lexCFGCharClass(src, i)
			if err != nil {
				return nil, err
			}
			toks = append(toks, cfgToken{kind: cfgTokCharClass, negate: neg, ranges: ranges, pos: i})
			i = next
		default:
			if isCFGIdentStart(ch) {
				start := i
				i++
				for i < len(src) && isCFGIdentCont(src[i]) {
					i++
				}
				toks = append(toks, cfgToken{kind: cfgTokIdent, text: src[start:i], pos: start})
				continue
			}
			return nil, fmt.Errorf("%w: unexpected character %q at offset %d", ErrInvalidGrammar, ch, i)
		}
	}
	toks = append(toks, cfgToken{kind: cfgTokEOF, pos: len(src)})
	return toks, nil
}

func isCFGIdentStart(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || b == '_'
}

func isCFGIdentCont(b byte) bool {
	return isCFGIdentStart(b) || (b >= '0' && b <= '9') || b == '-'
}

func lexCFGEscape(src string, i int) (rune, int, error) {
	if i >= len(src) {
		return 0, i, fmt.Errorf("%w: unterminated escape at offset %d", ErrInvalidGrammar, i)
	}
	switch src[i] {
	case 'n':
		return '\n', i + 1, nil
	case 'r':
		return '\r', i + 1, nil
	case 't':
		return '\t', i + 1, nil
	case '\\':
		return '\\', i + 1, nil
	case '"':
		return '"', i + 1, nil
	case '\'':
		return '\'', i + 1, nil
	case ']':
		return ']', i + 1, nil
	case '[':
		return '[', i + 1, nil
	case '-':
		return '-', i + 1, nil
	case 'x':
		if i+3 > len(src) {
			return 0, i, fmt.Errorf("%w: short \\x escape at offset %d", ErrInvalidGrammar, i)
		}
		b, err := hex.DecodeString(src[i+1 : i+3])
		if err != nil {
			return 0, i, fmt.Errorf("%w: invalid \\x escape at offset %d", ErrInvalidGrammar, i)
		}
		return rune(b[0]), i + 3, nil
	case 'u':
		if i+5 > len(src) {
			return 0, i, fmt.Errorf("%w: short \\u escape at offset %d", ErrInvalidGrammar, i)
		}
		v, err := strconv.ParseUint(src[i+1:i+5], 16, 32)
		if err != nil {
			return 0, i, fmt.Errorf("%w: invalid \\u escape at offset %d", ErrInvalidGrammar, i)
		}
		return rune(v), i + 5, nil
	default:
		r, w := utf8.DecodeRuneInString(src[i:])
		return r, i + w, nil
	}
}

func lexCFGLiteral(src string, start int) (string, int, error) {
	quote := src[start]
	i := start + 1
	var buf strings.Builder
	for i < len(src) {
		if src[i] == quote {
			return buf.String(), i + 1, nil
		}
		if src[i] == '\\' {
			if i+2 <= len(src) && src[i+1] == 'x' {
				if i+4 > len(src) {
					return "", i, fmt.Errorf("%w: short \\x escape at offset %d", ErrInvalidGrammar, i)
				}
				b, err := hex.DecodeString(src[i+2 : i+4])
				if err != nil {
					return "", i, fmt.Errorf("%w: invalid \\x escape at offset %d", ErrInvalidGrammar, i)
				}
				buf.WriteByte(b[0])
				i += 4
				continue
			}
			r, next, err := lexCFGEscape(src, i+1)
			if err != nil {
				return "", i, err
			}
			buf.WriteRune(r)
			i = next
			continue
		}
		r, w := utf8.DecodeRuneInString(src[i:])
		buf.WriteRune(r)
		i += w
	}
	return "", start, fmt.Errorf("%w: unterminated string literal at offset %d", ErrInvalidGrammar, start)
}

func lexCFGCharClass(src string, start int) (bool, []runeRange, int, error) {
	i := start + 1
	negate := false
	if i < len(src) && src[i] == '^' {
		negate = true
		i++
	}
	var ranges []runeRange
	first := true
	for i < len(src) {
		if src[i] == ']' && !first {
			return negate, mergeRuneRanges(ranges), i + 1, nil
		}
		first = false
		var lo rune
		if src[i] == '\\' {
			if i+1 < len(src) {
				switch src[i+1] {
				case 'd':
					ranges = append(ranges, runeRange{lo: '0', hi: '9'})
					i += 2
					continue
				case 's':
					ranges = append(ranges,
						runeRange{lo: '\t', hi: '\n'},
						runeRange{lo: '\r', hi: '\r'},
						runeRange{lo: ' ', hi: ' '},
					)
					i += 2
					continue
				case 'w':
					ranges = append(ranges,
						runeRange{lo: '0', hi: '9'},
						runeRange{lo: 'A', hi: 'Z'},
						runeRange{lo: '_', hi: '_'},
						runeRange{lo: 'a', hi: 'z'},
					)
					i += 2
					continue
				}
			}
			r, next, err := lexCFGEscape(src, i+1)
			if err != nil {
				return false, nil, i, err
			}
			lo = r
			i = next
		} else {
			r, w := utf8.DecodeRuneInString(src[i:])
			lo = r
			i += w
		}

		if i+1 < len(src) && src[i] == '-' && src[i+1] != ']' {
			i++
			var hi rune
			if src[i] == '\\' {
				r, next, err := lexCFGEscape(src, i+1)
				if err != nil {
					return false, nil, i, err
				}
				hi = r
				i = next
			} else {
				r, w := utf8.DecodeRuneInString(src[i:])
				hi = r
				i += w
			}
			if hi < lo {
				return false, nil, i, fmt.Errorf("%w: descending range in character class at offset %d", ErrInvalidGrammar, start)
			}
			ranges = append(ranges, runeRange{lo: lo, hi: hi})
		} else {
			ranges = append(ranges, runeRange{lo: lo, hi: lo})
		}
	}
	return false, nil, start, fmt.Errorf("%w: unterminated character class at offset %d", ErrInvalidGrammar, start)
}

func mergeRuneRanges(in []runeRange) []runeRange {
	if len(in) == 0 {
		return nil
	}
	cp := append([]runeRange(nil), in...)
	slices.SortFunc(cp, func(a, b runeRange) int {
		if a.lo != b.lo {
			return int(a.lo - b.lo)
		}
		return int(a.hi - b.hi)
	})
	out := cp[:1]
	for _, r := range cp[1:] {
		last := &out[len(out)-1]
		if r.lo <= last.hi+1 {
			if r.hi > last.hi {
				last.hi = r.hi
			}
		} else {
			out = append(out, r)
		}
	}
	return out
}

func complementRuneRanges(in []runeRange) []runeRange {
	var out []runeRange
	cur := rune(0)
	for _, r := range in {
		if r.lo > cur {
			out = append(out, runeRange{lo: cur, hi: r.lo - 1})
		}
		if r.hi+1 > cur {
			cur = r.hi + 1
		}
	}
	if cur <= utf8.MaxRune {
		out = append(out, runeRange{lo: cur, hi: utf8.MaxRune})
	}
	// Strip UTF-16 surrogates.
	var clean []runeRange
	for _, r := range out {
		if r.lo <= 0xDFFF && r.hi >= 0xD800 {
			if r.lo < 0xD800 {
				clean = append(clean, runeRange{lo: r.lo, hi: 0xD7FF})
			}
			if r.hi > 0xDFFF {
				clean = append(clean, runeRange{lo: 0xE000, hi: r.hi})
			}
			continue
		}
		clean = append(clean, r)
	}
	return clean
}

type cfgTextParser struct {
	toks      []cfgToken
	pos       int
	b         *cfgBuilder
	firstRule int
	synthSeq  int
}

func (p *cfgTextParser) peek(off int) cfgToken {
	if p.pos+off >= len(p.toks) {
		return p.toks[len(p.toks)-1]
	}
	return p.toks[p.pos+off]
}

func (p *cfgTextParser) parseAll() error {
	p.firstRule = -1
	for p.peek(0).kind != cfgTokEOF {
		if p.peek(0).kind != cfgTokIdent || p.peek(1).kind != cfgTokDef {
			return fmt.Errorf("%w: expected rule definition at offset %d", ErrInvalidGrammar, p.peek(0).pos)
		}
		name := p.peek(0).text
		p.pos += 2
		lhs, err := p.b.namedRule(name)
		if err != nil {
			return err
		}
		if p.firstRule < 0 {
			p.firstRule = lhs
		}
		if err := p.parseAlternatives(lhs, false); err != nil {
			return err
		}
	}
	return nil
}

func (p *cfgTextParser) parseAlternatives(lhs int, insideGroup bool) error {
	for {
		seq, err := p.parseSequence(insideGroup)
		if err != nil {
			return err
		}
		if err := p.b.addProd(lhs, seq...); err != nil {
			return err
		}
		if p.peek(0).kind != cfgTokPipe {
			break
		}
		p.pos++
	}
	return nil
}

func (p *cfgTextParser) parseSequence(insideGroup bool) ([]cfgSym, error) {
	var seq []cfgSym
	for {
		tok := p.peek(0)
		if tok.kind == cfgTokEOF || tok.kind == cfgTokPipe {
			break
		}
		if tok.kind == cfgTokRParen {
			if !insideGroup {
				return nil, fmt.Errorf("%w: unmatched ')' at offset %d", ErrInvalidGrammar, tok.pos)
			}
			break
		}
		if !insideGroup && tok.kind == cfgTokIdent && p.peek(1).kind == cfgTokDef {
			break
		}
		sym, hasAtom, err := p.parseFactor()
		if err != nil {
			return nil, err
		}
		if !hasAtom {
			break
		}
		seq = append(seq, sym...)
	}
	return seq, nil
}

func (p *cfgTextParser) parseFactor() ([]cfgSym, bool, error) {
	atom, ok, err := p.parseAtom()
	if err != nil || !ok {
		return nil, ok, err
	}
	for {
		switch p.peek(0).kind {
		case cfgTokStar:
			p.pos++
			p.synthSeq++
			r, err := p.b.newRule(fmt.Sprintf("__star_%d", p.synthSeq))
			if err != nil {
				return nil, false, err
			}
			if err := p.b.addProd(r); err != nil {
				return nil, false, err
			}
			rhs := append([]cfgSym{{rule: r}}, atom...)
			if err := p.b.addProd(r, rhs...); err != nil {
				return nil, false, err
			}
			atom = []cfgSym{{rule: r}}
		case cfgTokPlus:
			p.pos++
			p.synthSeq++
			r, err := p.b.newRule(fmt.Sprintf("__plus_%d", p.synthSeq))
			if err != nil {
				return nil, false, err
			}
			if err := p.b.addProd(r, atom...); err != nil {
				return nil, false, err
			}
			rhs := append([]cfgSym{{rule: r}}, atom...)
			if err := p.b.addProd(r, rhs...); err != nil {
				return nil, false, err
			}
			atom = []cfgSym{{rule: r}}
		case cfgTokQuest:
			p.pos++
			p.synthSeq++
			r, err := p.b.newRule(fmt.Sprintf("__opt_%d", p.synthSeq))
			if err != nil {
				return nil, false, err
			}
			if err := p.b.addProd(r); err != nil {
				return nil, false, err
			}
			if err := p.b.addProd(r, atom...); err != nil {
				return nil, false, err
			}
			atom = []cfgSym{{rule: r}}
		default:
			return atom, true, nil
		}
	}
}

func (p *cfgTextParser) parseAtom() ([]cfgSym, bool, error) {
	tok := p.peek(0)
	switch tok.kind {
	case cfgTokIdent:
		if p.peek(1).kind == cfgTokDef {
			return nil, false, nil
		}
		p.pos++
		rid, err := p.b.namedRule(tok.text)
		if err != nil {
			return nil, false, err
		}
		return []cfgSym{{rule: rid}}, true, nil
	case cfgTokLiteral:
		p.pos++
		raw := []byte(tok.text)
		syms := make([]cfgSym, len(raw))
		for i, ch := range raw {
			syms[i] = cfgSym{isTerm: true, term: singleByteSet(ch)}
		}
		return syms, true, nil
	case cfgTokCharClass:
		p.pos++
		ranges := tok.ranges
		if tok.negate {
			ranges = complementRuneRanges(ranges)
		}
		if len(ranges) == 0 {
			return nil, false, fmt.Errorf("%w: empty character class at offset %d", ErrInvalidGrammar, tok.pos)
		}
		var asciiSet byteSet
		var multiSeqs [][]byteRange
		for _, rr := range ranges {
			for _, seq := range appendUTF8Sequences(nil, rr.lo, rr.hi) {
				if len(seq) == 1 {
					asciiSet.addRange(seq[0].lo, seq[0].hi)
				} else {
					multiSeqs = append(multiSeqs, seq)
				}
			}
		}
		if len(multiSeqs) == 0 {
			return []cfgSym{{isTerm: true, term: asciiSet}}, true, nil
		}
		p.synthSeq++
		r, err := p.b.newRule(fmt.Sprintf("__cls_%d", p.synthSeq))
		if err != nil {
			return nil, false, err
		}
		if !asciiSet.empty() {
			if err := p.b.addProd(r, cfgSym{isTerm: true, term: asciiSet}); err != nil {
				return nil, false, err
			}
		}
		for _, seq := range multiSeqs {
			rhs := make([]cfgSym, len(seq))
			for i, br := range seq {
				var bs byteSet
				bs.addRange(br.lo, br.hi)
				rhs[i] = cfgSym{isTerm: true, term: bs}
			}
			if err := p.b.addProd(r, rhs...); err != nil {
				return nil, false, err
			}
		}
		return []cfgSym{{rule: r}}, true, nil
	case cfgTokLParen:
		p.pos++
		p.synthSeq++
		grp, err := p.b.newRule(fmt.Sprintf("__grp_%d", p.synthSeq))
		if err != nil {
			return nil, false, err
		}
		if err := p.parseAlternatives(grp, true); err != nil {
			return nil, false, err
		}
		if p.peek(0).kind != cfgTokRParen {
			return nil, false, fmt.Errorf("%w: expected ')' matching '(' at offset %d", ErrInvalidGrammar, tok.pos)
		}
		p.pos++
		return []cfgSym{{rule: grp}}, true, nil
	default:
		return nil, false, fmt.Errorf("%w: unexpected token at offset %d", ErrInvalidGrammar, tok.pos)
	}
}
