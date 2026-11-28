package agent

import "context"

type nativeThinkingConstraintContextKey struct{}

// nativeThinkingConstraint is owned by one request, including its OOM retry. The
// singleton choice at exhaustion adapts SGLang's strict reasoning mask and atomic
// closing-token validation (Apache-2.0, SGLang Team, reasoner_grammar_backend.py at
type nativeThinkingConstraint struct {
	budget      *ThinkBudget
	limit       int
	startInSpan bool
	closeID     int
	forced      bool
	decodeToken func(int) (string, error)
	// spanByToken counts reasoning by the structured constraint's token-ID span
	// instead of decoded marker text, so a spelled-out close cannot end the budget
	// while the schema still waits for the atomic close token.
	spanByToken bool
	spanCount   int
}

func nativeThinkingConstraintFromContext(ctx context.Context) *nativeThinkingConstraint {
	constraint, _ := ctx.Value(nativeThinkingConstraintContextKey{}).(*nativeThinkingConstraint)
	return constraint
}

func (p *InKernelPlanner) newNativeThinkingConstraint(limit int, startInSpan bool, stops map[int]bool) (*nativeThinkingConstraint, error) {
	ids, err := p.tok.Encode(thinkClose)
	if err != nil || len(ids) != 1 {
		return nil, &NativeThinkingBudgetUnsupportedError{Reason: "closing delimiter must encode as exactly one token"}
	}
	id := ids[0]
	if id < 0 || id >= p.tok.Vocab() || id >= p.m.Cfg.VocabSize {
		return nil, &NativeThinkingBudgetUnsupportedError{Reason: "closing token is outside the model vocabulary"}
	}
	piece, err := p.tok.Decode(ids)
	if err != nil || piece != thinkClose {
		return nil, &NativeThinkingBudgetUnsupportedError{Reason: "closing token does not decode to the exact delimiter"}
	}
	if stops[id] {
		return nil, &NativeThinkingBudgetUnsupportedError{Reason: "closing token conflicts with a token-ID stop"}
	}
	tok := p.tok
	constraint := &nativeThinkingConstraint{
		limit: limit, startInSpan: startInSpan, closeID: id,
		decodeToken: func(token int) (string, error) { return tok.Decode([]int{token}) },
	}
	constraint.reset()
	return constraint, nil
}

func (c *nativeThinkingConstraint) reset() {
	c.budget = NewThinkBudget(c.limit, c.startInSpan)
	c.forced = false
	c.spanCount = 0
}

// closingTokenInSpan is closingToken for a token-ID span: inSpan comes from the
// structured constraint, and the limit counts tokens accepted inside it.
func (c *nativeThinkingConstraint) closingTokenInSpan(inSpan bool, vocab int) (int, bool, error) {
	if c.closeID < 0 || c.closeID >= vocab {
		return 0, false, &NativeThinkingBudgetUnsupportedError{Reason: "closing token is outside the logits vocabulary"}
	}
	if inSpan && c.limit >= 0 && c.spanCount >= c.limit {
		c.forced = true
		return c.closeID, true, nil
	}
	return 0, false, nil
}

// acceptTokenInSpan counts one reasoning token selected inside a token-ID span.
// The close token ends the span and is not reasoning.
func (c *nativeThinkingConstraint) acceptTokenInSpan(inSpan bool, token int) {
	if inSpan && token != c.closeID {
		c.spanCount++
	}
}

// acceptToken counts the actual selected token before it can be emitted or
// forwarded. A decode error must fail closed instead of silently undercounting
// reasoning tokens; the unconstrained legacy emitter remains unchanged.
func (c *nativeThinkingConstraint) acceptToken(token int) error {
	piece, err := c.decodeToken(token)
	if err != nil {
		return &NativeThinkingBudgetUnsupportedError{Reason: "selected token could not be decoded for budget accounting"}
	}
	if c.budget.Observe(piece) {
		return &NativeThinkingBudgetUnsupportedError{Reason: "selected token exceeded the reasoning budget"}
	}
	return nil
}

// closingToken returns the only allowed next token once the reasoning budget is
// spent. Choosing a singleton before sampling is equivalent to a hard allow mask;
// repetition penalties and user bias cannot re-enable a reasoning token. Raw
// logits are left untouched for other consumers and cached-prefix reuse.
func (c *nativeThinkingConstraint) closingToken(vocab int) (int, bool, error) {
	if c.closeID < 0 || c.closeID >= vocab {
		return 0, false, &NativeThinkingBudgetUnsupportedError{Reason: "closing token is outside the logits vocabulary"}
	}
	if c.budget.InSpan() && !c.budget.canThinkMore() {
		c.forced = true
		return c.closeID, true, nil
	}
	return 0, false, nil
}
