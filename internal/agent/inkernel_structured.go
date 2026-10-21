package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/grammar"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var (
	ErrNativeStructuredUnsupported = errors.New("native structured output unsupported")
	ErrNativeStructuredNoToken     = errors.New("native structured output has no allowed token")
)

type nativeStructuredConstraintContextKey struct{}

type nativeStructuredConstraint struct {
	mask       model.LogitMask
	stopID     int
	history    []int
	maskBuffer []float32
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
	return &nativeStructuredConstraint{mask: mask, stopID: stopID}, nil
}

func (c *nativeStructuredConstraint) reset() {
	c.history = c.history[:0]
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
	c.mask.MaskLogits(c.history, c.maskBuffer)
	for _, logit := range c.maskBuffer {
		if math.IsNaN(float64(logit)) || math.IsInf(float64(logit), 1) {
			return nil, ErrNativeStructuredNoToken
		}
	}

	complete := c.stopID >= 0 && c.stopID < len(c.maskBuffer) && nativeStructuredLogitFinite(c.maskBuffer[c.stopID])
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
	if c == nil || token < 0 || token >= len(effective) || !nativeStructuredLogitFinite(effective[token]) {
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
