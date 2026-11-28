package toolgrammar

import (
	"bytes"
	"math"
	"strings"
	"sync"
)

func (s *JumpForwardState) hasNextRequiredAfterCurrent() bool {
	if s.activeBranch < 0 || s.activeBranch >= len(s.grammar.Branches) {
		return false
	}
	params := s.grammar.Branches[s.activeBranch].Parameters
	for i := s.paramIdx + 1; i < len(params); i++ {
		if params[i].Required && !s.isParamUsed(i) {
			return true
		}
	}
	return false
}

func (s *JumpForwardState) hasNextOptionalAfterCurrent() bool {
	if s.activeBranch < 0 || s.activeBranch >= len(s.grammar.Branches) {
		return false
	}
	params := s.grammar.Branches[s.activeBranch].Parameters
	for i, p := range params {
		if i != s.paramIdx && !p.Required && !s.isParamUsed(i) {
			return true
		}
	}
	return false
}

func setJSONWSBytes(allowed *[256]bool) {
	allowed[' '] = true
	allowed['\t'] = true
	allowed['\n'] = true
	allowed['\r'] = true
}

func bytesHasPrefixString(s string, prefix []byte) bool {
	if len(s) < len(prefix) {
		return false
	}
	for i := 0; i < len(prefix); i++ {
		if s[i] != prefix[i] {
			return false
		}
	}
	return true
}

func bytesEqualString(s string, b []byte) bool {
	if len(s) != len(b) {
		return false
	}
	for i := 0; i < len(b); i++ {
		if s[i] != b[i] {
			return false
		}
	}
	return true
}

func (s *JumpForwardState) canAcceptDiscByte(b byte) bool {
	if s.escaped {
		for _, br := range s.grammar.Branches {
			if len(br.DiscriminatorValue) > len(s.valBuf) &&
				bytesHasPrefixString(br.DiscriminatorValue, s.valBuf) &&
				br.DiscriminatorValue[len(s.valBuf)] == b {
				return true
			}
		}
		return false
	}
	if b == '"' {
		for _, br := range s.grammar.Branches {
			if bytesEqualString(br.DiscriminatorValue, s.valBuf) {
				return true
			}
		}
		return false
	}
	if b == '\\' {
		for _, br := range s.grammar.Branches {
			if len(br.DiscriminatorValue) > len(s.valBuf) &&
				bytesHasPrefixString(br.DiscriminatorValue, s.valBuf) {
				next := br.DiscriminatorValue[len(s.valBuf)]
				if next == '"' || next == '\\' {
					return true
				}
			}
		}
		return false
	}
	for _, br := range s.grammar.Branches {
		if len(br.DiscriminatorValue) > len(s.valBuf) &&
			bytesHasPrefixString(br.DiscriminatorValue, s.valBuf) &&
			br.DiscriminatorValue[len(s.valBuf)] == b {
			return true
		}
	}
	return false
}

func (s *JumpForwardState) canAcceptOptKeyByte(b byte) bool {
	if s.activeBranch < 0 || s.activeBranch >= len(s.grammar.Branches) {
		return false
	}
	params := s.grammar.Branches[s.activeBranch].Parameters
	if s.escaped {
		for i, p := range params {
			if !p.Required && !s.isParamUsed(i) &&
				len(p.Name) > len(s.valBuf) &&
				bytesHasPrefixString(p.Name, s.valBuf) &&
				p.Name[len(s.valBuf)] == b {
				return true
			}
		}
		return false
	}
	if b == '"' {
		for i, p := range params {
			if !p.Required && !s.isParamUsed(i) && bytesEqualString(p.Name, s.valBuf) {
				return true
			}
		}
		return false
	}
	if b == '\\' {
		for i, p := range params {
			if !p.Required && !s.isParamUsed(i) &&
				len(p.Name) > len(s.valBuf) &&
				bytesHasPrefixString(p.Name, s.valBuf) {
				next := p.Name[len(s.valBuf)]
				if next == '"' || next == '\\' {
					return true
				}
			}
		}
		return false
	}
	for i, p := range params {
		if !p.Required && !s.isParamUsed(i) &&
			len(p.Name) > len(s.valBuf) &&
			bytesHasPrefixString(p.Name, s.valBuf) &&
			p.Name[len(s.valBuf)] == b {
			return true
		}
	}
	return false
}

// AllowedNextBytes returns a 256-entry lookup table indicating which raw UTF-8
// bytes are valid next continuations from the current grammar state.
func (s *JumpForwardState) AllowedNextBytes() [256]bool {
	var allowed [256]bool
	if s == nil {
		return allowed
	}

	switch s.phase {
	case jfPhaseEnvelope:
		lit := s.envelopeStepLiteral()
		if s.litCursor == 0 {
			setJSONWSBytes(&allowed)
		}
		if s.litCursor < len(lit) {
			allowed[lit[s.litCursor]] = true
		}

	case jfPhaseArgsOpen:
		setJSONWSBytes(&allowed)
		allowed['{'] = true

	case jfPhaseDiscKey:
		if s.litCursor == 0 {
			setJSONWSBytes(&allowed)
		}
		var tmp [64]byte
		lit := appendJSONQuoted(tmp[:0], s.grammar.Discriminator)
		if s.litCursor < len(lit) {
			allowed[lit[s.litCursor]] = true
		}

	case jfPhaseDiscColon:
		setJSONWSBytes(&allowed)
		allowed[':'] = true

	case jfPhaseDiscValOpen:
		setJSONWSBytes(&allowed)
		allowed['"'] = true

	case jfPhaseDiscValBody:
		if s.escaped {
			for _, br := range s.grammar.Branches {
				if len(br.DiscriminatorValue) > len(s.valBuf) && bytesHasPrefixString(br.DiscriminatorValue, s.valBuf) {
					allowed[br.DiscriminatorValue[len(s.valBuf)]] = true
				}
			}
			return allowed
		}
		for _, br := range s.grammar.Branches {
			if bytesEqualString(br.DiscriminatorValue, s.valBuf) {
				allowed['"'] = true
			} else if len(br.DiscriminatorValue) > len(s.valBuf) && bytesHasPrefixString(br.DiscriminatorValue, s.valBuf) {
				next := br.DiscriminatorValue[len(s.valBuf)]
				if next == '"' || next == '\\' {
					allowed['\\'] = true
				} else {
					allowed[next] = true
				}
			}
		}

	case jfPhaseParamKey:
		if s.litCursor == 0 {
			setJSONWSBytes(&allowed)
		}
		if p, ok := s.currentParam(); ok && p.Required {
			var tmp [64]byte
			lit := appendJSONQuoted(tmp[:0], p.Name)
			if s.litCursor < len(lit) {
				allowed[lit[s.litCursor]] = true
			}
			return allowed
		}
		optCnt, singleIdx := s.remainingOptionalCount()
		if optCnt == 1 {
			p := s.grammar.Branches[s.activeBranch].Parameters[singleIdx]
			var tmp [64]byte
			lit := appendJSONQuoted(tmp[:0], p.Name)
			if s.litCursor < len(lit) {
				allowed[lit[s.litCursor]] = true
			}
		} else if optCnt > 1 {
			allowed['"'] = true
		}

	case jfPhaseOptKeyBody:
		if s.activeBranch < 0 || s.activeBranch >= len(s.grammar.Branches) {
			return allowed
		}
		params := s.grammar.Branches[s.activeBranch].Parameters
		if s.escaped {
			for i, p := range params {
				if !p.Required && !s.isParamUsed(i) &&
					len(p.Name) > len(s.valBuf) && bytesHasPrefixString(p.Name, s.valBuf) {
					allowed[p.Name[len(s.valBuf)]] = true
				}
			}
			return allowed
		}
		for i, p := range params {
			if p.Required || s.isParamUsed(i) {
				continue
			}
			if bytesEqualString(p.Name, s.valBuf) {
				allowed['"'] = true
			} else if len(p.Name) > len(s.valBuf) && bytesHasPrefixString(p.Name, s.valBuf) {
				next := p.Name[len(s.valBuf)]
				if next == '"' || next == '\\' {
					allowed['\\'] = true
				} else {
					allowed[next] = true
				}
			}
		}

	case jfPhaseParamColon:
		setJSONWSBytes(&allowed)
		allowed[':'] = true

	case jfPhaseParamValStart:
		p, ok := s.currentParam()
		if !ok {
			return allowed
		}
		if p.ConstValue != "" {
			if s.litCursor == 0 {
				setJSONWSBytes(&allowed)
			}
			var tmp [64]byte
			lit := appendJSONQuoted(tmp[:0], p.ConstValue)
			if s.litCursor < len(lit) {
				allowed[lit[s.litCursor]] = true
			}
			return allowed
		}
		setJSONWSBytes(&allowed)
		switch p.Type {
		case "integer", "number":
			allowed['-'] = true
			for b := byte('0'); b <= '9'; b++ {
				allowed[b] = true
			}
		case "boolean":
			allowed['t'] = true
			allowed['f'] = true
		default:
			allowed['"'] = true
		}

	case jfPhaseStringBody:
		for i := range allowed {
			allowed[i] = true
		}

	case jfPhaseIntBody:
		if s.numDigits == 0 && len(s.valBuf) == 0 {
			allowed['-'] = true
		}
		for b := byte('0'); b <= '9'; b++ {
			allowed[b] = true
		}
		if s.numDigits > 0 {
			setJSONWSBytes(&allowed)
			hasReq := s.hasNextRequiredAfterCurrent()
			hasOpt := s.hasNextOptionalAfterCurrent()
			if hasReq || hasOpt {
				allowed[','] = true
			}
			if !hasReq {
				allowed['}'] = true
			}
		}

	case jfPhaseNumBody:
		if s.numDigits == 0 && !s.numHasDot && len(s.valBuf) == 0 {
			allowed['-'] = true
		}
		for b := byte('0'); b <= '9'; b++ {
			allowed[b] = true
		}
		if !s.numHasDot && s.numDigits > 0 {
			allowed['.'] = true
		}
		if s.numDigits > 0 {
			setJSONWSBytes(&allowed)
			hasReq := s.hasNextRequiredAfterCurrent()
			hasOpt := s.hasNextOptionalAfterCurrent()
			if hasReq || hasOpt {
				allowed[','] = true
			}
			if !hasReq {
				allowed['}'] = true
			}
		}

	case jfPhaseBoolBody:
		switch {
		case len(s.valBuf) == 0:
			allowed['t'] = true
			allowed['f'] = true
		case bytesEqualString("t", s.valBuf):
			allowed['r'] = true
		case bytesEqualString("tr", s.valBuf):
			allowed['u'] = true
		case bytesEqualString("tru", s.valBuf):
			allowed['e'] = true
		case bytesEqualString("f", s.valBuf):
			allowed['a'] = true
		case bytesEqualString("fa", s.valBuf):
			allowed['l'] = true
		case bytesEqualString("fal", s.valBuf):
			allowed['s'] = true
		case bytesEqualString("fals", s.valBuf):
			allowed['e'] = true
		}

	case jfPhaseAfterParam:
		setJSONWSBytes(&allowed)
		if s.stepIdx == 1 {
			allowed['}'] = true
			allowed['"'] = true
			return allowed
		}
		if _, hasReq := s.nextRequiredIndex(); hasReq {
			allowed[','] = true
			return allowed
		}
		if optCnt, _ := s.remainingOptionalCount(); optCnt > 0 {
			allowed[','] = true
			allowed['}'] = true
			return allowed
		}
		allowed['}'] = true

	case jfPhaseArgsClose, jfPhaseOuterClose:
		setJSONWSBytes(&allowed)
		allowed['}'] = true

	case jfPhaseDone:
		setJSONWSBytes(&allowed)
	}

	return allowed
}

func (s *JumpForwardState) stepByteStrict(b byte) error {
	switch s.phase {
	case jfPhaseDiscValBody:
		if !s.canAcceptDiscByte(b) {
			return ErrGrammarMismatch
		}
	case jfPhaseOptKeyBody:
		if !s.canAcceptOptKeyByte(b) {
			return ErrGrammarMismatch
		}
	}
	return s.stepByte(b)
}

// CanAcceptText reports whether text is a valid continuation prefix from the current
// grammar state without mutating s.
func (s *JumpForwardState) CanAcceptText(text string) bool {
	if s == nil {
		return false
	}
	if len(text) == 0 {
		return true
	}

	cp := *s
	cp.draftBuf = nil
	var vb [64]byte
	if len(s.valBuf) <= len(vb) {
		cp.valBuf = append(vb[:0], s.valBuf...)
	} else {
		cp.valBuf = append([]byte(nil), s.valBuf...)
	}
	if len(s.usedExtra) > 0 {
		cp.usedExtra = make([]bool, len(s.usedExtra))
		copy(cp.usedExtra, s.usedExtra)
	}

	for i := 0; i < len(text); i++ {
		if err := cp.stepByteStrict(text[i]); err != nil {
			return false
		}
	}
	return true
}

// MaskLogits masks all invalid token IDs in logits to -Inf according to the current grammar state.
// Any token ID in tokens whose text is a valid continuation preserves its logit; all other positions
// in logits are set to -Inf.
func (s *JumpForwardState) MaskLogits(tokens []VocabToken, logits []float32) {
	if s == nil || len(logits) == 0 {
		return
	}
	negInf := float32(math.Inf(-1))
	allowed := s.AllowedNextBytes()
	isStringBody := s.phase == jfPhaseStringBody

	var stackKeep [512]bool
	var keep []bool
	if len(logits) <= len(stackKeep) {
		keep = stackKeep[:len(logits)]
	} else {
		keep = make([]bool, len(logits))
	}

	for _, tok := range tokens {
		id := int(tok.ID)
		if id < 0 || id >= len(logits) || len(tok.Text) == 0 {
			continue
		}
		if !allowed[tok.Text[0]] {
			continue
		}
		if isStringBody && strings.IndexByte(tok.Text, '"') < 0 {
			keep[id] = true
			continue
		}
		if s.CanAcceptText(tok.Text) {
			keep[id] = true
		}
	}

	for i := range logits {
		if !keep[i] {
			logits[i] = negInf
		}
	}
}

type maskKeepBuf struct {
	b []bool
}

// MaskCache pre-indexes a vocabulary by first byte and quote presence to accelerate
// repeated JumpForwardState logit masking safely across concurrent callers.
type MaskCache struct {
	tokens      []VocabToken
	byFirst     [256][]VocabToken
	noQuoteIDs  []int32
	quoteTokens []VocabToken
	keepPool    sync.Pool
}

// NewMaskCache builds a reusable MaskCache from a slice of vocabulary tokens.
func NewMaskCache(tokens []VocabToken) *MaskCache {
	mc := &MaskCache{
		tokens: append([]VocabToken(nil), tokens...),
	}
	for _, tok := range mc.tokens {
		if tok.ID < 0 || len(tok.Text) == 0 {
			continue
		}
		mc.byFirst[tok.Text[0]] = append(mc.byFirst[tok.Text[0]], tok)
		if bytes.IndexByte([]byte(tok.Text), '"') < 0 {
			mc.noQuoteIDs = append(mc.noQuoteIDs, tok.ID)
		} else {
			mc.quoteTokens = append(mc.quoteTokens, tok)
		}
	}
	return mc
}

// Tokens returns the cached vocabulary tokens.
func (mc *MaskCache) Tokens() []VocabToken {
	if mc == nil {
		return nil
	}
	return mc.tokens
}

// MaskLogits masks all invalid token IDs in logits to -Inf using the pre-indexed vocabulary.
func (mc *MaskCache) MaskLogits(s *JumpForwardState, logits []float32) {
	if mc == nil || s == nil || len(logits) == 0 {
		return
	}
	negInf := float32(math.Inf(-1))
	var keep []bool
	var stackKeep [256]bool
	if len(logits) <= len(stackKeep) {
		keep = stackKeep[:len(logits)]
	} else {
		pooled, _ := mc.keepPool.Get().(*maskKeepBuf)
		if pooled == nil {
			pooled = &maskKeepBuf{b: make([]bool, len(logits))}
		} else if cap(pooled.b) < len(logits) {
			pooled.b = make([]bool, len(logits))
		} else {
			pooled.b = pooled.b[:len(logits)]
			clear(pooled.b)
		}
		keep = pooled.b
		defer mc.keepPool.Put(pooled)
	}

	if s.phase == jfPhaseStringBody {
		for _, id32 := range mc.noQuoteIDs {
			id := int(id32)
			if id >= 0 && id < len(logits) {
				keep[id] = true
			}
		}
		for _, tok := range mc.quoteTokens {
			id := int(tok.ID)
			if id >= 0 && id < len(logits) && s.CanAcceptText(tok.Text) {
				keep[id] = true
			}
		}
	} else {
		allowed := s.AllowedNextBytes()
		for b := 0; b < 256; b++ {
			if !allowed[b] {
				continue
			}
			for _, tok := range mc.byFirst[b] {
				id := int(tok.ID)
				if id >= 0 && id < len(logits) && s.CanAcceptText(tok.Text) {
					keep[id] = true
				}
			}
		}
	}

	for i := range logits {
		if !keep[i] {
			logits[i] = negInf
		}
	}
}
