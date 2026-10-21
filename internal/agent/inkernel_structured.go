package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/grammar"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var (
	ErrNativeStructuredUnsupported = errors.New("native structured output unsupported")
	ErrNativeStructuredNoToken     = errors.New("native structured output has no allowed token")
)

type nativeStructuredConstraintContextKey struct{}

// nativeStructuredPhase places a structured response relative to an optional
// reasoning span. Adapted from DeepInfer's structured-output matcher, which leaves
// tokens unconstrained until a trigger ends reasoning and forces that trigger once
// a bound fires (Apache-2.0, deepseek-ai/DeepInfer,
// deepinfer/structure_output/matcher.py at 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01);
// no upstream source is copied. The trigger here is the reasoning-close token ID,
// never its spelled-out text, and the bound is the native thinking budget.
type nativeStructuredPhase int

const (
	nativeStructuredJSON nativeStructuredPhase = iota
	nativeStructuredReasoning
	nativeStructuredOpenOrJSON
)

type nativeStructuredConstraint struct {
	mask       model.LogitMask
	stopID     int
	history    []int
	maskBuffer []float32

	openID  int
	closeID int
	initial nativeStructuredPhase
	phase   nativeStructuredPhase
	closed  bool
}

func nativeStructuredConstraintFromContext(ctx context.Context) *nativeStructuredConstraint {
	constraint, _ := ctx.Value(nativeStructuredConstraintContextKey{}).(*nativeStructuredConstraint)
	return constraint
}

func nativeStructuredJSONSchemaRequested(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return false
	}
	var response struct {
		Type string `json:"type"`
	}
	return json.Unmarshal(trimmed, &response) == nil && response.Type == "json_schema"
}

func (p *InKernelPlanner) newNativeStructuredConstraint(raw json.RawMessage, stops map[int]bool) (*nativeStructuredConstraint, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil
	}
	var response struct {
		Type       string `json:"type"`
		JSONSchema *struct {
			Strict *bool           `json:"strict"`
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	}
	if err := json.Unmarshal(trimmed, &response); err != nil {
		return nil, fmt.Errorf("%w: invalid response_format: %v", ErrNativeStructuredUnsupported, err)
	}
	if response.Type == "text" {
		return nil, nil
	}
	if response.Type != "json_schema" || response.JSONSchema == nil {
		return nil, fmt.Errorf("%w: response_format type %q", ErrNativeStructuredUnsupported, response.Type)
	}
	if response.JSONSchema.Strict != nil && !*response.JSONSchema.Strict {
		return nil, fmt.Errorf("%w: json_schema strict=false", ErrNativeStructuredUnsupported)
	}
	schema := bytes.TrimSpace(response.JSONSchema.Schema)
	if len(schema) == 0 || bytes.Equal(schema, []byte("null")) {
		return nil, fmt.Errorf("%w: json_schema schema is required", ErrNativeStructuredUnsupported)
	}
	if stops == nil {
		if p == nil || p.tok == nil || p.m == nil {
			return nil, fmt.Errorf("%w: tokenizer or model is unavailable", ErrNativeStructuredUnsupported)
		}
		stops = StopIDs(p.tok, p.m.Cfg)
	}
	stopID := -1
	for id, enabled := range stops {
		if enabled && id >= 0 && id < p.tok.Vocab() && (stopID < 0 || id < stopID) {
			stopID = id
		}
	}
	if stopID < 0 {
		return nil, fmt.Errorf("%w: no in-vocabulary stop token", ErrNativeStructuredUnsupported)
	}
	mask, err := grammar.CompileJSONSchema(schema, p.tok, grammar.CompileOptions{EOS: stopID})
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNativeStructuredUnsupported, err)
	}
	return &nativeStructuredConstraint{mask: mask, stopID: stopID, openID: -1, closeID: -1}, nil
}

// armReasoning lets a reasoning span precede the schema response. A seeded prompt
// is already inside the span; otherwise the first token may open one or start the
// JSON object. Both delimiters must be atomic in-vocabulary non-stop tokens so the
// trigger is a token ID the model cannot forge with ordinary text.
func (c *nativeStructuredConstraint) armReasoning(p *InKernelPlanner, seeded bool, stops map[int]bool) error {
	closeID, err := p.nativeStructuredReasoningDelimiter(thinkClose, stops)
	if err != nil {
		return err
	}
	c.closeID, c.openID, c.initial = closeID, -1, nativeStructuredReasoning
	if !seeded {
		openID, err := p.nativeStructuredReasoningDelimiter(thinkOpen, stops)
		if err != nil {
			return err
		}
		c.openID, c.initial = openID, nativeStructuredOpenOrJSON
	}
	c.reset()
	return nil
}

func (p *InKernelPlanner) nativeStructuredReasoningDelimiter(marker string, stops map[int]bool) (int, error) {
	ids, err := p.tok.Encode(marker)
	if err != nil || len(ids) != 1 {
		return -1, fmt.Errorf("%w: reasoning delimiter %q is not one token", ErrNativeStructuredUnsupported, marker)
	}
	id := ids[0]
	if id < 0 || id >= p.tok.Vocab() || id >= p.m.Cfg.VocabSize {
		return -1, fmt.Errorf("%w: reasoning delimiter %q is outside the model vocabulary", ErrNativeStructuredUnsupported, marker)
	}
	if piece, err := p.tok.Decode(ids); err != nil || piece != marker {
		return -1, fmt.Errorf("%w: reasoning delimiter %q does not decode exactly", ErrNativeStructuredUnsupported, marker)
	}
	if stops[id] {
		return -1, fmt.Errorf("%w: reasoning delimiter %q is a stop token", ErrNativeStructuredUnsupported, marker)
	}
	return id, nil
}

func (c *nativeStructuredConstraint) reset() {
	c.history = c.history[:0]
	c.phase = c.initial
	c.closed = false
}

// reasoningActive reports whether the JSON object has not started yet.
func (c *nativeStructuredConstraint) reasoningActive() bool {
	return c != nil && c.phase != nativeStructuredJSON
}

// inReasoningSpan reports whether the atomic open has been seen (or seeded) and
// the atomic close has not.
func (c *nativeStructuredConstraint) inReasoningSpan() bool {
	return c != nil && c.phase == nativeStructuredReasoning
}

func (c *nativeStructuredConstraint) reasoningArmed() bool {
	return c != nil && c.initial != nativeStructuredJSON
}

// takeReasoningClosed reports, once, that the close trigger was just accepted.
func (c *nativeStructuredConstraint) takeReasoningClosed() bool {
	if c == nil || !c.closed {
		return false
	}
	c.closed = false
	return true
}

// splitOutput separates reasoning from the schema response at contentStart, the
// byte offset just past the emitted close trigger (negative when none was
// emitted). An unclosed span is reasoning and never content.
func (c *nativeStructuredConstraint) splitOutput(raw string, contentStart int) (reasoning, content string) {
	if !c.reasoningArmed() {
		return "", raw
	}
	if contentStart < 0 {
		if c.phase == nativeStructuredJSON {
			return "", raw
		}
		return c.trimReasoning(raw), ""
	}
	reasoningEnd := min(max(contentStart-len(thinkClose), 0), len(raw))
	return c.trimReasoning(raw[:reasoningEnd]), raw[min(contentStart, len(raw)):]
}

func (c *nativeStructuredConstraint) trimReasoning(text string) string {
	if c.initial == nativeStructuredOpenOrJSON {
		text = strings.TrimPrefix(text, thinkOpen)
	}
	return strings.TrimSpace(text)
}

func (c *nativeStructuredConstraint) maskLogits(logits []float32, stops map[int]bool) ([]float32, error) {
	if c == nil || c.mask == nil || len(logits) == 0 {
		return nil, ErrNativeStructuredNoToken
	}
	if cap(c.maskBuffer) < len(logits) {
		c.maskBuffer = make([]float32, len(logits))
	} else {
		c.maskBuffer = c.maskBuffer[:len(logits)]
	}
	copy(c.maskBuffer, logits)
	switch c.phase {
	case nativeStructuredReasoning:
		// Reasoning is unconstrained except that the response cannot end before
		// its JSON object starts; the stop mask below applies.
	case nativeStructuredOpenOrJSON:
		if c.openID < 0 || c.openID >= len(c.maskBuffer) {
			return nil, ErrNativeStructuredNoToken
		}
		open := c.maskBuffer[c.openID]
		c.mask.MaskLogits(c.history, c.maskBuffer)
		c.maskBuffer[c.openID] = open
	default:
		c.mask.MaskLogits(c.history, c.maskBuffer)
	}
	for _, logit := range c.maskBuffer {
		if math.IsNaN(float64(logit)) || math.IsInf(float64(logit), 1) {
			return nil, ErrNativeStructuredNoToken
		}
	}

	complete := c.phase == nativeStructuredJSON && c.stopID >= 0 && c.stopID < len(c.maskBuffer) && nativeStructuredLogitFinite(c.maskBuffer[c.stopID])
	if !complete {
		for id, enabled := range stops {
			if enabled && id >= 0 && id < len(c.maskBuffer) {
				c.maskBuffer[id] = float32(math.Inf(-1))
			}
		}
	}
	survivor := false
	for _, logit := range c.maskBuffer {
		if nativeStructuredLogitFinite(logit) {
			survivor = true
		}
	}
	if survivor {
		return c.maskBuffer, nil
	}
	return nil, ErrNativeStructuredNoToken
}

func (c *nativeStructuredConstraint) accept(token int, effective []float32) error {
	if c == nil || token < 0 || token >= len(effective) {
		return ErrNativeStructuredNoToken
	}
	switch c.phase {
	case nativeStructuredReasoning:
		// The close trigger may be forced by the thinking budget regardless of
		// its score; any other reasoning token must have been selectable.
		if token == c.closeID {
			c.phase, c.closed = nativeStructuredJSON, true
			return nil
		}
		if !nativeStructuredLogitFinite(effective[token]) {
			return ErrNativeStructuredNoToken
		}
		return nil
	case nativeStructuredOpenOrJSON:
		if token == c.openID && nativeStructuredLogitFinite(effective[token]) {
			c.phase = nativeStructuredReasoning
			return nil
		}
		c.phase = nativeStructuredJSON
	}
	if !nativeStructuredLogitFinite(effective[token]) {
		return ErrNativeStructuredNoToken
	}
	if token != c.stopID {
		c.history = append(c.history, token)
	}
	return nil
}

func nativeStructuredLogitFinite(logit float32) bool {
	return !math.IsNaN(float64(logit)) && !math.IsInf(float64(logit), 0)
}
