package toolgrammar

import (
	"errors"
	"fmt"
	"strings"
)

// ErrGrammarMismatch is returned when input text or tokens violate the compiled JSON tool grammar.
var ErrGrammarMismatch = errors.New("toolgrammar: input violates compiled tool-call grammar")

// VocabMatcher maps forced literal text spans to vocabulary token IDs.
// Implementations may wrap a GGUF/BPE tokenizer (including byte-level BPE).
type VocabMatcher interface {
	// MatchPrefix returns the longest token ID in the vocabulary that is an exact
	// prefix of text, along with the number of bytes consumed.
	// If no token matches a non-empty prefix of text, it returns (0, 0, false).
	MatchPrefix(text string) (tokenID int32, byteLen int, ok bool)
}

// VocabToken pairs a token ID with its decoded UTF-8 literal text for building a PrefixVocab.
type VocabToken struct {
	ID   int32
	Text string
}

type trieNode struct {
	children [256]*trieNode
	hasToken bool
	tokenID  int32
}

// PrefixVocab is a zero-allocation byte-trie VocabMatcher for greedy longest-prefix tokenization.
type PrefixVocab struct {
	root trieNode
}

// NewPrefixVocab builds a PrefixVocab from a slice of vocabulary tokens.
// Empty-text tokens are ignored; duplicate texts keep the lowest token ID or latest entry deterministically.
func NewPrefixVocab(tokens []VocabToken) *PrefixVocab {
	v := &PrefixVocab{}
	for _, tok := range tokens {
		if tok.Text == "" {
			continue
		}
		cur := &v.root
		for i := 0; i < len(tok.Text); i++ {
			b := tok.Text[i]
			if cur.children[b] == nil {
				cur.children[b] = &trieNode{}
			}
			cur = cur.children[b]
		}
		cur.hasToken = true
		cur.tokenID = tok.ID
	}
	return v
}

// MatchPrefix returns the longest token in the trie matching a prefix of text.
func (v *PrefixVocab) MatchPrefix(text string) (int32, int, bool) {
	if v == nil || len(text) == 0 {
		return 0, 0, false
	}
	cur := &v.root
	var bestID int32
	bestLen := 0
	for i := 0; i < len(text); i++ {
		next := cur.children[text[i]]
		if next == nil {
			break
		}
		cur = next
		if cur.hasToken {
			bestID = cur.tokenID
			bestLen = i + 1
		}
	}
	if bestLen == 0 {
		return 0, 0, false
	}
	return bestID, bestLen, true
}

type jfPhase uint8

const (
	jfPhaseEnvelope jfPhase = iota
	jfPhaseArgsOpen
	jfPhaseDiscKey
	jfPhaseDiscColon
	jfPhaseDiscValOpen
	jfPhaseDiscValBody
	jfPhaseParamKey
	jfPhaseOptKeyBody
	jfPhaseParamColon
	jfPhaseParamValStart
	jfPhaseStringBody
	jfPhaseIntBody
	jfPhaseNumBody
	jfPhaseBoolBody
	jfPhaseAfterParam
	jfPhaseArgsClose
	jfPhaseOuterClose
	jfPhaseDone
)

// JumpForwardState tracks the deterministic parser/drafter position in a compiled DialectJSON tool grammar.
type JumpForwardState struct {
	grammar      *CompiledGrammar
	phase        jfPhase
	stepIdx      int
	activeBranch int
	paramIdx     int
	usedMask     uint64
	usedExtra    []bool

	// Sub-cursors for literal matching & scalar/string parsing.
	litCursor int
	escaped   bool
	numHasDot bool
	numDigits int
	valBuf    []byte
	draftBuf  []byte
}

// NewJumpForwardState initializes a JumpForwardState at the start of a compiled JSON tool grammar.
func NewJumpForwardState(cg *CompiledGrammar) (*JumpForwardState, error) {
	if cg == nil {
		return nil, fmt.Errorf("%w: nil compiled grammar", ErrInvalidSchema)
	}
	if cg.Dialect != DialectJSON {
		return nil, fmt.Errorf("%w: jump-forward requires DialectJSON, got %q", ErrUnsupportedDialect, cg.Dialect)
	}
	if len(cg.Branches) == 0 {
		return nil, fmt.Errorf("%w: compiled grammar has no branches", ErrNoDiscriminator)
	}

	s := &JumpForwardState{
		grammar:      cg,
		activeBranch: -1,
		valBuf:       make([]byte, 0, 64),
		draftBuf:     make([]byte, 0, 128),
	}
	if len(cg.Branches) == 1 {
		s.activeBranch = 0
	}
	if cg.IncludeToolTag {
		s.phase = jfPhaseEnvelope
	} else {
		s.phase = jfPhaseArgsOpen
	}
	return s, nil
}

// Clone returns an independent copy of the state for speculative branching or rollback.
func (s *JumpForwardState) Clone() *JumpForwardState {
	cp := *s
	if len(s.usedExtra) > 0 {
		cp.usedExtra = make([]bool, len(s.usedExtra))
		copy(cp.usedExtra, s.usedExtra)
	}
	cp.valBuf = make([]byte, len(s.valBuf), cap(s.valBuf))
	copy(cp.valBuf, s.valBuf)
	cp.draftBuf = make([]byte, 0, cap(s.draftBuf))
	return &cp
}

// IsComplete reports whether the full JSON tool-call envelope has been matched.
func (s *JumpForwardState) IsComplete() bool {
	return s.phase == jfPhaseDone
}

// ForcedDraft computes the maximal deterministic forced literal span from the current state
// (with canonical compact JSON formatting) and greedily tokenizes it with vocab.
// If vocab is nil, tokens is nil and text is the full forced literal span.
// Note: ForcedDraft does NOT mutate state; call AdvanceText or AdvanceTokens after verification.
func ForcedDraft(state *JumpForwardState, vocab VocabMatcher) (tokens []int32, text string) {
	return ForcedDraftInto(state, vocab, nil)
}

// ForcedDraftInto is the allocation-lean variant of ForcedDraft that appends matched token IDs into dstTokens.
func ForcedDraftInto(state *JumpForwardState, vocab VocabMatcher, dstTokens []int32) (tokens []int32, text string) {
	if state == nil || state.phase == jfPhaseDone {
		return dstTokens, ""
	}

	// Simulate forward on a lightweight stack-allocated copy of the state scalars (no slice copies needed).
	sim := *state
	sim.valBuf = nil
	sim.draftBuf = nil
	buf := state.draftBuf[:0]

	for {
		chunk := sim.forcedChunk(&buf)
		if len(chunk) == 0 {
			break
		}
		if err := sim.advanceForcedChunk(chunk); err != nil {
			break
		}
	}

	if len(buf) == 0 {
		return dstTokens, ""
	}
	state.draftBuf = buf
	fullText := string(buf)

	if vocab == nil {
		return dstTokens, fullText
	}

	rem := fullText
	matchedBytes := 0
	for len(rem) > 0 {
		tokID, n, ok := vocab.MatchPrefix(rem)
		if !ok || n <= 0 || n > len(rem) {
			break
		}
		dstTokens = append(dstTokens, tokID)
		rem = rem[n:]
		matchedBytes += n
	}
	if matchedBytes == 0 {
		return dstTokens, ""
	}
	return dstTokens, fullText[:matchedBytes]
}

// AdvanceText advances the grammar state by consuming s.
// On grammar violation, it returns ErrGrammarMismatch and leaves state unchanged.
func (s *JumpForwardState) AdvanceText(sText string) error {
	if s == nil {
		return fmt.Errorf("%w: nil state", ErrInvalidSchema)
	}
	cp := s.Clone()
	for i := 0; i < len(sText); i++ {
		if err := cp.stepByte(sText[i]); err != nil {
			return err
		}
	}
	*s = *cp
	return nil
}

// Accepts reports whether payload is a complete, valid JSON tool call under cg.
func Accepts(cg *CompiledGrammar, payload string) bool {
	st, err := NewJumpForwardState(cg)
	if err != nil {
		return false
	}
	if err := st.AdvanceText(payload); err != nil {
		return false
	}
	return st.IsComplete()
}

func (s *JumpForwardState) currentParam() (ParameterDef, bool) {
	if s.activeBranch < 0 || s.activeBranch >= len(s.grammar.Branches) {
		return ParameterDef{}, false
	}
	params := s.grammar.Branches[s.activeBranch].Parameters
	if s.paramIdx < 0 || s.paramIdx >= len(params) {
		return ParameterDef{}, false
	}
	return params[s.paramIdx], true
}

func (s *JumpForwardState) isParamUsed(idx int) bool {
	if idx < 64 {
		return (s.usedMask & (1 << uint(idx))) != 0
	}
	extraIdx := idx - 64
	if extraIdx < len(s.usedExtra) {
		return s.usedExtra[extraIdx]
	}
	return false
}

func (s *JumpForwardState) markParamUsed(idx int) {
	if idx < 64 {
		s.usedMask |= 1 << uint(idx)
		return
	}
	extraIdx := idx - 64
	if extraIdx >= len(s.usedExtra) {
		grow := make([]bool, extraIdx+1)
		copy(grow, s.usedExtra)
		s.usedExtra = grow
	}
	s.usedExtra[extraIdx] = true
}

func (s *JumpForwardState) remainingOptionalCount() (count int, singleIdx int) {
	if s.activeBranch < 0 || s.activeBranch >= len(s.grammar.Branches) {
		return 0, -1
	}
	params := s.grammar.Branches[s.activeBranch].Parameters
	singleIdx = -1
	for i := 0; i < len(params); i++ {
		if !params[i].Required && !s.isParamUsed(i) {
			count++
			singleIdx = i
		}
	}
	return count, singleIdx
}

func (s *JumpForwardState) nextRequiredIndex() (int, bool) {
	if s.activeBranch < 0 || s.activeBranch >= len(s.grammar.Branches) {
		return -1, false
	}
	params := s.grammar.Branches[s.activeBranch].Parameters
	for i := s.paramIdx; i < len(params); i++ {
		if params[i].Required && !s.isParamUsed(i) {
			return i, true
		}
	}
	return -1, false
}

// forcedChunk appends the next deterministic literal chunk to dst and returns the appended slice.
// Returns empty slice when at a branching or value position.
func (s *JumpForwardState) forcedChunk(dst *[]byte) []byte {
	start := len(*dst)
	switch s.phase {
	case jfPhaseEnvelope:
		// Canonical envelope: {"name":"<ToolName>","arguments":
		env := s.envelopeStepLiteral()
		if s.litCursor < len(env) {
			*dst = append(*dst, env[s.litCursor:]...)
		}
	case jfPhaseArgsOpen:
		*dst = append(*dst, '{')
	case jfPhaseDiscKey:
		var tmp [64]byte
		lit := appendJSONQuoted(tmp[:0], s.grammar.Discriminator)
		if s.litCursor < len(lit) {
			*dst = append(*dst, lit[s.litCursor:]...)
		}
	case jfPhaseDiscColon:
		*dst = append(*dst, ':')
	case jfPhaseDiscValOpen:
		*dst = append(*dst, '"')
	case jfPhaseParamKey:
		var tmp [64]byte
		if p, ok := s.currentParam(); ok && p.Required {
			lit := appendJSONQuoted(tmp[:0], p.Name)
			if s.litCursor < len(lit) {
				*dst = append(*dst, lit[s.litCursor:]...)
			}
		} else {
			cnt, idx := s.remainingOptionalCount()
			if cnt == 1 {
				p := s.grammar.Branches[s.activeBranch].Parameters[idx]
				lit := appendJSONQuoted(tmp[:0], p.Name)
				if s.litCursor < len(lit) {
					*dst = append(*dst, lit[s.litCursor:]...)
				}
			} else if cnt > 1 && s.litCursor == 0 {
				// Every key starts with a quote, then branches on key name.
				*dst = append(*dst, '"')
			}
		}
	case jfPhaseParamColon:
		*dst = append(*dst, ':')
	case jfPhaseParamValStart:
		p, ok := s.currentParam()
		if !ok {
			return nil
		}
		if p.ConstValue != "" {
			var tmp [64]byte
			lit := appendJSONQuoted(tmp[:0], p.ConstValue)
			if s.litCursor < len(lit) {
				*dst = append(*dst, lit[s.litCursor:]...)
			}
		} else if p.Type != "integer" && p.Type != "number" && p.Type != "boolean" {
			*dst = append(*dst, '"')
		}
	case jfPhaseAfterParam:
		if _, hasReq := s.nextRequiredIndex(); hasReq {
			*dst = append(*dst, ',')
		} else if optCnt, _ := s.remainingOptionalCount(); optCnt == 0 {
			*dst = append(*dst, '}')
		}
	case jfPhaseArgsClose:
		*dst = append(*dst, '}')
	case jfPhaseOuterClose:
		if s.grammar.IncludeToolTag {
			*dst = append(*dst, '}')
		}
	}
	return (*dst)[start:]
}

func (s *JumpForwardState) advanceForcedChunk(chunk []byte) error {
	for i := 0; i < len(chunk); i++ {
		if err := s.stepByte(chunk[i]); err != nil {
			return err
		}
	}
	return nil
}

func (s *JumpForwardState) envelopeStepLiteral() string {
	switch s.stepIdx {
	case 0:
		return "{"
	case 1:
		return `"name"`
	case 2:
		return ":"
	case 3:
		return string(appendJSONQuoted(nil, s.grammar.ToolName))
	case 4:
		return ","
	case 5:
		return `"arguments"`
	case 6:
		return ":"
	default:
		return ""
	}
}

func (s *JumpForwardState) stepByte(b byte) error {
	switch s.phase {
	case jfPhaseEnvelope:
		lit := s.envelopeStepLiteral()
		if s.litCursor == 0 && isJSONWS(b) {
			return nil
		}
		if s.litCursor < len(lit) && b == lit[s.litCursor] {
			s.litCursor++
			if s.litCursor == len(lit) {
				s.litCursor = 0
				s.stepIdx++
				if s.stepIdx > 6 {
					s.phase = jfPhaseArgsOpen
				}
			}
			return nil
		}
		return ErrGrammarMismatch

	case jfPhaseArgsOpen:
		if isJSONWS(b) {
			return nil
		}
		if b != '{' {
			return ErrGrammarMismatch
		}
		s.litCursor = 0
		if s.activeBranch < 0 {
			s.phase = jfPhaseDiscKey
			return nil
		}
		if reqIdx, ok := s.nextRequiredIndex(); ok {
			s.paramIdx = reqIdx
			s.phase = jfPhaseParamKey
			return nil
		}
		if optCnt, _ := s.remainingOptionalCount(); optCnt > 0 {
			s.phase = jfPhaseAfterParam
			s.stepIdx = 1 // inside '{', either '}' or a key
			return nil
		}
		s.phase = jfPhaseArgsClose
		return nil

	case jfPhaseDiscKey:
		if s.litCursor == 0 && isJSONWS(b) {
			return nil
		}
		lit := appendJSONQuoted(nil, s.grammar.Discriminator)
		if s.litCursor < len(lit) && b == lit[s.litCursor] {
			s.litCursor++
			if s.litCursor == len(lit) {
				s.litCursor = 0
				s.phase = jfPhaseDiscColon
			}
			return nil
		}
		return ErrGrammarMismatch

	case jfPhaseDiscColon:
		if isJSONWS(b) {
			return nil
		}
		if b != ':' {
			return ErrGrammarMismatch
		}
		s.phase = jfPhaseDiscValOpen
		return nil

	case jfPhaseDiscValOpen:
		if isJSONWS(b) {
			return nil
		}
		if b != '"' {
			return ErrGrammarMismatch
		}
		s.valBuf = s.valBuf[:0]
		s.escaped = false
		s.phase = jfPhaseDiscValBody
		return nil

	case jfPhaseDiscValBody:
		if s.escaped {
			s.valBuf = append(s.valBuf, b)
			s.escaped = false
			return nil
		}
		if b == '\\' {
			s.escaped = true
			return nil
		}
		if b == '"' {
			val := string(s.valBuf)
			matched := -1
			for i, br := range s.grammar.Branches {
				if br.DiscriminatorValue == val {
					matched = i
					break
				}
			}
			if matched < 0 {
				return ErrGrammarMismatch
			}
			s.activeBranch = matched
			s.markParamUsed(0)
			s.paramIdx = 1
			s.phase = jfPhaseAfterParam
			s.stepIdx = 0
			return nil
		}
		s.valBuf = append(s.valBuf, b)
		return nil

	case jfPhaseParamKey:
		if s.litCursor == 0 && isJSONWS(b) {
			return nil
		}
		if p, ok := s.currentParam(); ok && p.Required {
			lit := appendJSONQuoted(nil, p.Name)
			if s.litCursor < len(lit) && b == lit[s.litCursor] {
				s.litCursor++
				if s.litCursor == len(lit) {
					s.litCursor = 0
					s.phase = jfPhaseParamColon
				}
				return nil
			}
			return ErrGrammarMismatch
		}
		// Optional parameter key
		optCnt, singleIdx := s.remainingOptionalCount()
		if optCnt == 1 {
			s.paramIdx = singleIdx
			p := s.grammar.Branches[s.activeBranch].Parameters[singleIdx]
			lit := appendJSONQuoted(nil, p.Name)
			if s.litCursor < len(lit) && b == lit[s.litCursor] {
				s.litCursor++
				if s.litCursor == len(lit) {
					s.litCursor = 0
					s.phase = jfPhaseParamColon
				}
				return nil
			}
			return ErrGrammarMismatch
		}
		if optCnt > 1 {
			if b != '"' {
				return ErrGrammarMismatch
			}
			s.valBuf = s.valBuf[:0]
			s.escaped = false
			s.phase = jfPhaseOptKeyBody
			return nil
		}
		return ErrGrammarMismatch

	case jfPhaseOptKeyBody:
		if s.escaped {
			s.valBuf = append(s.valBuf, b)
			s.escaped = false
			return nil
		}
		if b == '\\' {
			s.escaped = true
			return nil
		}
		if b == '"' {
			key := string(s.valBuf)
			params := s.grammar.Branches[s.activeBranch].Parameters
			matched := -1
			for i, p := range params {
				if !p.Required && !s.isParamUsed(i) && p.Name == key {
					matched = i
					break
				}
			}
			if matched < 0 {
				return ErrGrammarMismatch
			}
			s.paramIdx = matched
			s.litCursor = 0
			s.phase = jfPhaseParamColon
			return nil
		}
		s.valBuf = append(s.valBuf, b)
		return nil

	case jfPhaseParamColon:
		if isJSONWS(b) {
			return nil
		}
		if b != ':' {
			return ErrGrammarMismatch
		}
		s.litCursor = 0
		s.phase = jfPhaseParamValStart
		return nil

	case jfPhaseParamValStart:
		p, ok := s.currentParam()
		if !ok {
			return ErrGrammarMismatch
		}
		if p.ConstValue != "" {
			if s.litCursor == 0 && isJSONWS(b) {
				return nil
			}
			lit := appendJSONQuoted(nil, p.ConstValue)
			if s.litCursor < len(lit) && b == lit[s.litCursor] {
				s.litCursor++
				if s.litCursor == len(lit) {
					s.litCursor = 0
					s.finishCurrentParam()
				}
				return nil
			}
			return ErrGrammarMismatch
		}
		if isJSONWS(b) {
			return nil
		}
		switch p.Type {
		case "integer":
			s.numDigits = 0
			s.valBuf = s.valBuf[:0]
			s.phase = jfPhaseIntBody
			return s.stepByte(b)
		case "number":
			s.numDigits = 0
			s.numHasDot = false
			s.valBuf = s.valBuf[:0]
			s.phase = jfPhaseNumBody
			return s.stepByte(b)
		case "boolean":
			s.litCursor = 0
			s.valBuf = s.valBuf[:0]
			s.phase = jfPhaseBoolBody
			return s.stepByte(b)
		default:
			if b != '"' {
				return ErrGrammarMismatch
			}
			s.escaped = false
			s.phase = jfPhaseStringBody
			return nil
		}

	case jfPhaseStringBody:
		if s.escaped {
			s.escaped = false
			return nil
		}
		if b == '\\' {
			s.escaped = true
			return nil
		}
		if b == '"' {
			s.finishCurrentParam()
			return nil
		}
		return nil

	case jfPhaseIntBody:
		if b == '-' && s.numDigits == 0 && len(s.valBuf) == 0 {
			s.valBuf = append(s.valBuf[:0], '-')
			return nil
		}
		if b >= '0' && b <= '9' {
			s.numDigits++
			return nil
		}
		if s.numDigits > 0 && (isJSONWS(b) || b == ',' || b == '}') {
			s.valBuf = s.valBuf[:0]
			s.finishCurrentParam()
			return s.stepByte(b)
		}
		return ErrGrammarMismatch

	case jfPhaseNumBody:
		if b == '-' && s.numDigits == 0 && !s.numHasDot && len(s.valBuf) == 0 {
			s.valBuf = append(s.valBuf[:0], '-')
			return nil
		}
		if b >= '0' && b <= '9' {
			s.numDigits++
			return nil
		}
		if b == '.' && !s.numHasDot && s.numDigits > 0 {
			s.numHasDot = true
			s.numDigits = 0 // require at least 1 digit after '.'
			return nil
		}
		if s.numDigits > 0 && (isJSONWS(b) || b == ',' || b == '}') {
			s.valBuf = s.valBuf[:0]
			s.finishCurrentParam()
			return s.stepByte(b)
		}
		return ErrGrammarMismatch

	case jfPhaseBoolBody:
		s.valBuf = append(s.valBuf, b)
		cur := string(s.valBuf)
		if cur == "true" || cur == "false" {
			s.valBuf = s.valBuf[:0]
			s.finishCurrentParam()
			return nil
		}
		if strings.HasPrefix("true", cur) || strings.HasPrefix("false", cur) {
			return nil
		}
		return ErrGrammarMismatch

	case jfPhaseAfterParam:
		if isJSONWS(b) {
			return nil
		}
		if s.stepIdx == 1 {
			// Immediately after '{' with only optional params remaining: either '}' or first optional key.
			if b == '}' {
				s.stepIdx = 0
				s.advancePastArgsClose()
				return nil
			}
			if b == '"' {
				s.stepIdx = 0
				s.phase = jfPhaseParamKey
				return s.stepByte(b)
			}
			return ErrGrammarMismatch
		}
		if reqIdx, hasReq := s.nextRequiredIndex(); hasReq {
			if b != ',' {
				return ErrGrammarMismatch
			}
			s.paramIdx = reqIdx
			s.litCursor = 0
			s.phase = jfPhaseParamKey
			return nil
		}
		if optCnt, _ := s.remainingOptionalCount(); optCnt > 0 {
			if b == ',' {
				s.litCursor = 0
				s.phase = jfPhaseParamKey
				return nil
			}
			if b == '}' {
				s.advancePastArgsClose()
				return nil
			}
			return ErrGrammarMismatch
		}
		if b == '}' {
			s.advancePastArgsClose()
			return nil
		}
		return ErrGrammarMismatch

	case jfPhaseArgsClose:
		if isJSONWS(b) {
			return nil
		}
		if b != '}' {
			return ErrGrammarMismatch
		}
		s.advancePastArgsClose()
		return nil

	case jfPhaseOuterClose:
		if isJSONWS(b) {
			return nil
		}
		if b != '}' {
			return ErrGrammarMismatch
		}
		s.phase = jfPhaseDone
		return nil

	case jfPhaseDone:
		if isJSONWS(b) {
			return nil
		}
		return ErrGrammarMismatch
	}
	return ErrGrammarMismatch
}

func (s *JumpForwardState) finishCurrentParam() {
	s.markParamUsed(s.paramIdx)
	if reqIdx, hasReq := s.nextRequiredIndex(); hasReq {
		s.paramIdx = reqIdx
	} else {
		s.paramIdx = len(s.grammar.Branches[s.activeBranch].Parameters)
	}
	s.phase = jfPhaseAfterParam
	s.stepIdx = 0
}

func (s *JumpForwardState) advancePastArgsClose() {
	if s.grammar.IncludeToolTag {
		s.phase = jfPhaseOuterClose
	} else {
		s.phase = jfPhaseDone
	}
}

func isJSONWS(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

func appendJSONQuoted(dst []byte, s string) []byte {
	dst = append(dst, '"')
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"':
			dst = append(dst, '\\', '"')
		case '\\':
			dst = append(dst, '\\', '\\')
		case '\n':
			dst = append(dst, '\\', 'n')
		case '\r':
			dst = append(dst, '\\', 'r')
		case '\t':
			dst = append(dst, '\\', 't')
		default:
			dst = append(dst, s[i])
		}
	}
	dst = append(dst, '"')
	return dst
}
