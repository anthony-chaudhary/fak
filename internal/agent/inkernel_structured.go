package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"regexp"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/grammar"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var (
	ErrNativeStructuredUnsupported = errors.New("native structured output unsupported")
	ErrNativeStructuredNoToken     = errors.New("native structured output has no allowed token")
)

type nativeGuidedCarrierError struct{ Carrier string }

func (e *nativeGuidedCarrierError) Error() string {
	return fmt.Sprintf("%v: guided carrier %q", ErrNativeStructuredUnsupported, e.Carrier)
}

func (e *nativeGuidedCarrierError) Unwrap() error { return ErrNativeStructuredUnsupported }

// nativeGuidedRegexCarriers are the guided regex fields the native route
// enforces; the first one present names the request's pattern.
var nativeGuidedRegexCarriers = [...]string{"guided_regex", "regex"}

// nativeGuidedCarriers are every guided field the native route enforces.
var nativeGuidedCarriers = [...]string{"guided_choice", "guided_json", "guided_regex", "regex"}

// guided_choice compiles to a regex alternation; these bounds keep its escaped
// pattern and automaton inside the regex mask's caps.
const (
	maxNativeGuidedChoices     = 128
	maxNativeGuidedChoiceBytes = 1024
)

// nativeGuided is the one guided constraint a request carries: a regex pattern
// (guided_regex, regex, guided_choice) or a JSON schema (guided_json).
type nativeGuided struct {
	carrier string
	pattern string
	schema  json.RawMessage
}

// refuseNativeGuidedCarriers rejects every provider-native guided-decode field
// the native route cannot enforce, so no carrier yields unconstrained text as if
// it were constrained.
func refuseNativeGuidedCarriers(guided map[string]json.RawMessage) error {
	_, err := parseNativeGuided(guided)
	return err
}

// parseNativeGuided returns the request's guided constraint, if any. Only one
// kind may be sent; both regex carriers may be sent only with the same pattern.
func parseNativeGuided(guided map[string]json.RawMessage) (nativeGuided, error) {
	refused, kinds := "", []string{}
	for key, raw := range guided {
		if nativeGuidedAbsent(raw) {
			continue
		}
		if !slices.Contains(nativeGuidedCarriers[:], key) {
			if refused == "" || key < refused {
				refused = key
			}
			continue
		}
		if key == "regex" && !nativeGuidedAbsent(guided["guided_regex"]) {
			continue
		}
		kinds = append(kinds, key)
	}
	if refused != "" {
		return nativeGuided{}, &nativeGuidedCarrierError{Carrier: refused}
	}
	if len(kinds) == 0 {
		return nativeGuided{}, nil
	}
	slices.Sort(kinds)
	if len(kinds) > 1 {
		return nativeGuided{}, &nativeGuidedCarrierError{Carrier: kinds[0]}
	}
	switch kinds[0] {
	case "guided_json":
		schema, err := nativeGuidedJSONSchema(guided["guided_json"])
		return nativeGuided{carrier: "guided_json", schema: schema}, err
	case "guided_choice":
		pattern, err := nativeGuidedChoicePattern(guided["guided_choice"])
		return nativeGuided{carrier: "guided_choice", pattern: pattern}, err
	}
	var spec nativeGuided
	for _, key := range nativeGuidedRegexCarriers {
		if nativeGuidedAbsent(guided[key]) {
			continue
		}
		var value string
		if json.Unmarshal(guided[key], &value) != nil || (spec.carrier != "" && value != spec.pattern) {
			return nativeGuided{}, &nativeGuidedCarrierError{Carrier: key}
		}
		if spec.carrier == "" {
			spec.carrier, spec.pattern = key, value
		}
	}
	return spec, nil
}

// nativeGuidedJSONSchema accepts a schema object or, as vLLM does, a JSON string
// holding one.
func nativeGuidedJSONSchema(raw json.RawMessage) (json.RawMessage, error) {
	schema := bytes.TrimSpace(raw)
	if len(schema) > 0 && schema[0] == '"' {
		var text string
		if json.Unmarshal(schema, &text) != nil {
			return nil, &nativeGuidedCarrierError{Carrier: "guided_json"}
		}
		schema = bytes.TrimSpace([]byte(text))
	}
	if len(schema) == 0 || schema[0] != '{' || !json.Valid(schema) {
		return nil, &nativeGuidedCarrierError{Carrier: "guided_json"}
	}
	return schema, nil
}

// nativeGuidedChoicePattern turns a guided_choice list into a full-match
// alternation of escaped literals, so the response is exactly one member.
func nativeGuidedChoicePattern(raw json.RawMessage) (string, error) {
	refused := &nativeGuidedCarrierError{Carrier: "guided_choice"}
	var members []any
	if json.Unmarshal(raw, &members) != nil || len(members) == 0 {
		return "", refused
	}
	seen := make(map[string]bool, len(members))
	quoted := make([]string, 0, min(len(members), maxNativeGuidedChoices))
	size := 0
	for _, member := range members {
		choice, ok := member.(string)
		if !ok || choice == "" {
			return "", refused
		}
		if seen[choice] {
			continue
		}
		seen[choice] = true
		size += len(choice)
		if len(quoted) == maxNativeGuidedChoices || size > maxNativeGuidedChoiceBytes {
			return "", refused
		}
		quoted = append(quoted, regexp.QuoteMeta(choice))
	}
	return strings.Join(quoted, "|"), nil
}

// nativeGuidedSameSchema compares schemas as parsed JSON, so key order and
// whitespace do not matter.
func nativeGuidedSameSchema(a, b json.RawMessage) bool {
	decode := func(raw json.RawMessage) (any, bool) {
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.UseNumber()
		var value any
		err := decoder.Decode(&value)
		return value, err == nil
	}
	left, okLeft := decode(a)
	right, okRight := decode(b)
	return okLeft && okRight && reflect.DeepEqual(left, right)
}

func nativeGuidedAbsent(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null"))
}

// nativeStructuredRequested reports whether the request carries a constraint
// the native route enforces.
func nativeStructuredRequested(sp SampleParams) bool {
	if nativeStructuredJSONSchemaRequested(sp.ResponseFormat) {
		return true
	}
	guided, err := parseNativeGuided(sp.GuidedDecode)
	return err == nil && guided.carrier != ""
}

// nativeStructuredRunePrefix splits off a trailing incomplete UTF-8 rune so a
// streamed piece never ends inside a multi-byte rune.
func nativeStructuredRunePrefix(text string) (complete, tail string) {
	for start := len(text) - 1; start >= 0 && start >= len(text)-utf8.UTFMax; start-- {
		if utf8.RuneStart(text[start]) {
			if !utf8.FullRuneInString(text[start:]) {
				return text[:start], text[start:]
			}
			break
		}
	}
	return text, ""
}

// nativeRegexTokenDecoder hides special tokens from a regex mask: TokenBytes
// spells them literally, so a permissive pattern would otherwise admit control
// tokens as text. The mask admits the stop token by id alone.
type nativeRegexTokenDecoder struct {
	tok     grammar.TokenDecoder
	special map[int]string
}

func (d nativeRegexTokenDecoder) TokenBytes(id int) []byte {
	if _, ok := d.special[id]; ok {
		return nil
	}
	return d.tok.TokenBytes(id)
}

// newNativeRequestConstraint builds the request's native constraint: a guided
// carrier, or otherwise its response_format. A guided carrier binds the whole
// response, so it refuses any response_format other than text, except that
// guided_json may repeat the same schema as a json_schema response_format.
func (p *InKernelPlanner) newNativeRequestConstraint(sp SampleParams, stops map[int]bool) (*nativeStructuredConstraint, error) {
	guided, err := parseNativeGuided(sp.GuidedDecode)
	if err != nil {
		return nil, err
	}
	if guided.carrier == "" {
		return p.newNativeStructuredConstraint(sp.ResponseFormat, stops)
	}
	if !nativeGuidedAbsent(sp.ResponseFormat) {
		var response struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(bytes.TrimSpace(sp.ResponseFormat), &response) != nil {
			return nil, &nativeGuidedCarrierError{Carrier: guided.carrier}
		}
		switch {
		case response.Type == "text":
		case response.Type == "json_schema" && guided.schema != nil:
			schema, err := nativeResponseFormatSchema(sp.ResponseFormat)
			if err != nil {
				return nil, err
			}
			if !nativeGuidedSameSchema(schema, guided.schema) {
				return nil, &nativeGuidedCarrierError{Carrier: guided.carrier}
			}
		default:
			return nil, &nativeGuidedCarrierError{Carrier: guided.carrier}
		}
	}
	if guided.schema != nil {
		return p.newNativeJSONSchemaConstraint(guided.schema, stops)
	}
	stopID, err := p.nativeStructuredStopID(stops)
	if err != nil {
		return nil, err
	}
	mask, err := grammar.CompileRegex(guided.pattern, nativeRegexTokenDecoder{tok: p.tok, special: p.tok.SpecialTokens()}, grammar.CompileOptions{EOS: stopID})
	if err != nil {
		return nil, fmt.Errorf("%w: %s: %w", ErrNativeStructuredUnsupported, guided.carrier, err)
	}
	return &nativeStructuredConstraint{mask: mask, stopID: stopID, openID: -1, closeID: -1}, nil
}

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
	schema, err := nativeResponseFormatSchema(raw)
	if err != nil || schema == nil {
		return nil, err
	}
	return p.newNativeJSONSchemaConstraint(schema, stops)
}

// nativeResponseFormatSchema returns the schema of a json_schema
// response_format, or nil for an absent or text format.
func nativeResponseFormatSchema(raw json.RawMessage) (json.RawMessage, error) {
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
	return schema, nil
}

// newNativeJSONSchemaConstraint compiles the native response subset shared by
// response_format json_schema and guided_json.
func (p *InKernelPlanner) newNativeJSONSchemaConstraint(schema json.RawMessage, stops map[int]bool) (*nativeStructuredConstraint, error) {
	stopID, err := p.nativeStructuredStopID(stops)
	if err != nil {
		return nil, err
	}
	mask, err := grammar.CompileJSONSchema(schema, p.tok, grammar.CompileOptions{EOS: stopID})
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrNativeStructuredUnsupported, err)
	}
	return &nativeStructuredConstraint{mask: mask, stopID: stopID, openID: -1, closeID: -1}, nil
}

// nativeStructuredStopID picks the lowest in-vocabulary stop token as the one
// EOS a structured mask admits once the response is complete.
func (p *InKernelPlanner) nativeStructuredStopID(stops map[int]bool) (int, error) {
	if p == nil || p.tok == nil || p.m == nil {
		return -1, fmt.Errorf("%w: tokenizer or model is unavailable", ErrNativeStructuredUnsupported)
	}
	if stops == nil {
		stops = StopIDs(p.tok, p.m.Cfg)
	}
	stopID := -1
	for id, enabled := range stops {
		if enabled && id >= 0 && id < p.tok.Vocab() && (stopID < 0 || id < stopID) {
			stopID = id
		}
	}
	if stopID < 0 {
		return -1, fmt.Errorf("%w: no in-vocabulary stop token", ErrNativeStructuredUnsupported)
	}
	return stopID, nil
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
