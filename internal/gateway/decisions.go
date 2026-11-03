package gateway

// decisions.go — POST /v1/decisions: the single-choice decision-probability
// surface. One validated option set in, one engine-scored softmax out.
//
// ADAPTED from TensorFold's /v1/decisions reply contract (ashhart/TensorFold,
// zig/src/server/decisions.zig @9e8c368b0b4c2c7892a0c9f0f302d7c164016209,
// Apache-2.0): prompt_format_version=1 only, temperature strictly positive,
// unique option ids, positional A-Z labels, logits supplied by an engine hook,
// stable softmax plus per-label mass on the way out, and closed refusals for a
// missing hook, duplicate ids, more than 26 options, and overflowed (non-finite)
// logits. No upstream code was copied; the idea is re-expressed against this
// gateway's host-injection seam conventions.
//
// The scoring hook is process-scoped rather than a Server field: a fak process
// serves exactly one engine, so the host CLI installs the scorer once at serve
// start (InstallDecisionsScorer) and every Server instance in the process sees
// it. A gateway with no installed scorer keeps the route inert (404), the same
// convention as the other unwired provider routes.

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"strconv"
	"sync"
)

const (
	// DecisionsPromptFormatVersion is the only prompt_format_version the wire
	// serves; any other value is refused before the engine is consulted.
	DecisionsPromptFormatVersion = 1
	// maxDecisionsOptions bounds one request to the A-Z label alphabet: the i-th
	// option (0-based) carries label 'A'+i, so the 27th option has no label.
	maxDecisionsOptions = 26
)

// Closed refusal sentinels. Callers branch on these with errors.Is, never on
// message prose; each maps to exactly one HTTP status + error code below.
var (
	// ErrDecisionsMalformedBody: unparseable JSON, an empty option id, or an
	// empty prompt — shape violations of the request contract.
	ErrDecisionsMalformedBody = errors.New("gateway: decisions malformed body")
	// ErrDecisionsPromptFormat: prompt_format_version is not 1.
	ErrDecisionsPromptFormat = errors.New("gateway: decisions unsupported prompt_format_version")
	// ErrDecisionsTemperature: temperature is NaN, infinite, or <= 0.
	ErrDecisionsTemperature = errors.New("gateway: decisions temperature must be finite and > 0")
	// ErrDecisionsNoOptions: the request carried zero options.
	ErrDecisionsNoOptions = errors.New("gateway: decisions requires at least one option")
	// ErrDecisionsDuplicateOptionID: two options share an id.
	ErrDecisionsDuplicateOptionID = errors.New("gateway: decisions duplicate option id")
	// ErrDecisionsTooManyOptions: more than 26 options (the A-Z label cap).
	ErrDecisionsTooManyOptions = errors.New("gateway: decisions more than 26 options")
	// ErrDecisionsHookUnwired: no engine scoring hook is installed (route inert).
	ErrDecisionsHookUnwired = errors.New("gateway: decisions scoring hook not installed")
	// ErrDecisionsOverflow: a non-finite logit, or a partition that cannot
	// normalize — the softmax overflow class.
	ErrDecisionsOverflow = errors.New("gateway: decisions logit overflow")
	// ErrDecisionsScoringFailed: the installed hook errored or returned a logit
	// count that does not match the option count.
	ErrDecisionsScoringFailed = errors.New("gateway: decisions engine scoring failed")
)

// DecisionsOption is one candidate choice. ID must be non-empty and unique
// across the request; Text is optional free text handed to the engine hook.
type DecisionsOption struct {
	ID   string `json:"id"`
	Text string `json:"text,omitempty"`
}

// DecisionsRequest is the POST /v1/decisions body.
type DecisionsRequest struct {
	// PromptFormatVersion must equal DecisionsPromptFormatVersion (1).
	PromptFormatVersion int `json:"prompt_format_version"`
	// Temperature divides the logits before the softmax; finite and > 0.
	Temperature float64 `json:"temperature"`
	// Prompt is the decision context the engine scores the options against.
	Prompt string `json:"prompt"`
	// Model is optional and echoed back; a gateway serves one resident model,
	// so it selects nothing.
	Model   string            `json:"model,omitempty"`
	Options []DecisionsOption `json:"options"`
}

// DecisionsResult is one option's scored outcome, in request order.
type DecisionsResult struct {
	// Label is the option's positional A-Z label ('A' for the first option).
	Label       string  `json:"label"`
	ID          string  `json:"id"`
	Logit       float64 `json:"logit"`
	Probability float64 `json:"probability"`
}

// DecisionsResponse is the 200 body. Object is always "decisions"; the result
// probabilities sum to 1 within floating-point rounding, and LabelMass carries
// the probability mass per label (each label owns exactly one option, so its
// mass equals that option's probability).
type DecisionsResponse struct {
	Object              string             `json:"object"`
	PromptFormatVersion int                `json:"prompt_format_version"`
	Temperature         float64            `json:"temperature"`
	Model               string             `json:"model,omitempty"`
	Results             []DecisionsResult  `json:"results"`
	LabelMass           map[string]float64 `json:"label_mass"`
}

// DecisionsScoreInput is everything the engine hook needs to score one
// validated request: the context, the option identities, and their optional
// texts, all in request order.
type DecisionsScoreInput struct {
	Model       string
	Prompt      string
	OptionIDs   []string
	OptionTexts []string
}

// DecisionsScorerFunc is the engine scoring hook behind POST /v1/decisions: it
// returns exactly one FINITE logit per option, in option order. It must be safe
// for concurrent use (requests are served concurrently).
type DecisionsScorerFunc func(ctx context.Context, in DecisionsScoreInput) ([]float64, error)

var (
	decisionsScorerMu sync.Mutex // guards decisionsScorerFn
	decisionsScorerFn DecisionsScorerFunc
)

// InstallDecisionsScorer installs (or, with nil, clears) the process-wide
// engine scoring hook. The host CLI wires it to the resident engine at serve
// start; tests install deterministic fakes. Passing nil detaches the hook and
// makes the route inert again.
func InstallDecisionsScorer(fn DecisionsScorerFunc) {
	decisionsScorerMu.Lock()
	decisionsScorerFn = fn
	decisionsScorerMu.Unlock()
}

// decisionsScorer returns the installed hook, or nil when none is wired.
func decisionsScorer() DecisionsScorerFunc {
	decisionsScorerMu.Lock()
	defer decisionsScorerMu.Unlock()
	return decisionsScorerFn
}

// validateDecisionsRequest checks the closed client contract and returns the
// matching sentinel. Check order is fixed (and documented in the OpenAPI spec)
// so a request violating several rules always gets the same refusal:
// prompt_format_version, temperature, option count, then option ids.
func validateDecisionsRequest(req *DecisionsRequest) error {
	if req.PromptFormatVersion != DecisionsPromptFormatVersion {
		return ErrDecisionsPromptFormat
	}
	t := req.Temperature
	if math.IsNaN(t) || math.IsInf(t, 0) || t <= 0 {
		return ErrDecisionsTemperature
	}
	if len(req.Options) == 0 {
		return ErrDecisionsNoOptions
	}
	if len(req.Options) > maxDecisionsOptions {
		return ErrDecisionsTooManyOptions
	}
	seen := make(map[string]struct{}, len(req.Options))
	for _, opt := range req.Options {
		if opt.ID == "" {
			return ErrDecisionsMalformedBody
		}
		if _, dup := seen[opt.ID]; dup {
			return ErrDecisionsDuplicateOptionID
		}
		seen[opt.ID] = struct{}{}
	}
	if req.Prompt == "" {
		return ErrDecisionsMalformedBody
	}
	return nil
}

// decisionsRefusalStatus maps a validation sentinel to its HTTP status and
// typed error code — the closed client-facing refusal table.
func decisionsRefusalStatus(err error) (status int, code string) {
	switch {
	case errors.Is(err, ErrDecisionsMalformedBody):
		return http.StatusBadRequest, "decisions_malformed_body"
	case errors.Is(err, ErrDecisionsPromptFormat):
		return http.StatusBadRequest, "decisions_bad_prompt_format"
	case errors.Is(err, ErrDecisionsTemperature):
		return http.StatusBadRequest, "decisions_bad_temperature"
	case errors.Is(err, ErrDecisionsNoOptions):
		return http.StatusBadRequest, "decisions_no_options"
	case errors.Is(err, ErrDecisionsDuplicateOptionID):
		return http.StatusBadRequest, "decisions_duplicate_option_id"
	case errors.Is(err, ErrDecisionsTooManyOptions):
		return http.StatusBadRequest, "decisions_too_many_options"
	default:
		return http.StatusBadRequest, "decisions_invalid_request"
	}
}

// softmaxDecisionProbs maps raw engine logits to option probabilities at the
// request temperature using the numerically stable form (subtract the maximum
// logit before exponentiating, so no exp argument is ever positive). It refuses
// the overflow class: a non-finite logit has no defined exp, and a partition
// sum that is not finite and positive cannot normalize. On success the
// probabilities are finite, in [0,1], and sum to 1 within rounding.
func softmaxDecisionProbs(logits []float64, temperature float64) ([]float64, error) {
	if len(logits) == 0 {
		return nil, ErrDecisionsNoOptions
	}
	maxLogit := math.Inf(-1)
	for _, l := range logits {
		if math.IsNaN(l) || math.IsInf(l, 0) {
			return nil, ErrDecisionsOverflow
		}
		if l > maxLogit {
			maxLogit = l
		}
	}
	probs := make([]float64, len(logits))
	sum := 0.0
	for i, l := range logits {
		e := math.Exp((l - maxLogit) / temperature)
		if math.IsNaN(e) || math.IsInf(e, 0) {
			return nil, ErrDecisionsOverflow
		}
		probs[i] = e
		sum += e
	}
	if math.IsNaN(sum) || math.IsInf(sum, 0) || sum <= 0 {
		return nil, ErrDecisionsOverflow
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs, nil
}

// handleDecisions serves POST /v1/decisions. Refusal precedence is fixed:
// method (405), then inert-unwired (404 decisions_hook_unwired), then the
// closed client-contract refusals (400, typed codes), then engine-side
// failures (503 decisions_scoring_failed for a hook error or a mismatched
// logit count; 400 decisions_logit_overflow for a non-finite logit).
func (s *Server) handleDecisions(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	fn := decisionsScorer()
	if fn == nil {
		writeErrCode(w, http.StatusNotFound, "decisions_hook_unwired", ErrDecisionsHookUnwired.Error())
		return
	}
	b, err := io.ReadAll(io.LimitReader(r.Body, maxBody))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "unreadable request body")
		return
	}
	var req DecisionsRequest
	if json.Unmarshal(b, &req) != nil {
		writeErrCode(w, http.StatusBadRequest, "decisions_malformed_body", "malformed request body: want a decisions JSON object")
		return
	}
	if verr := validateDecisionsRequest(&req); verr != nil {
		status, code := decisionsRefusalStatus(verr)
		writeErrCode(w, status, code, verr.Error())
		return
	}
	in := DecisionsScoreInput{
		Model:       req.Model,
		Prompt:      req.Prompt,
		OptionIDs:   make([]string, len(req.Options)),
		OptionTexts: make([]string, len(req.Options)),
	}
	for i, opt := range req.Options {
		in.OptionIDs[i] = opt.ID
		in.OptionTexts[i] = opt.Text
	}
	logits, err := fn(r.Context(), in)
	if err != nil {
		writeErrCode(w, http.StatusServiceUnavailable, "decisions_scoring_failed", err.Error())
		return
	}
	if len(logits) != len(req.Options) {
		writeErrCode(w, http.StatusServiceUnavailable, "decisions_scoring_failed",
			"engine hook returned "+strconv.Itoa(len(logits))+" logits for "+strconv.Itoa(len(req.Options))+" options")
		return
	}
	probs, err := softmaxDecisionProbs(logits, req.Temperature)
	if err != nil {
		writeErrCode(w, http.StatusBadRequest, "decisions_logit_overflow", err.Error())
		return
	}
	resp := DecisionsResponse{
		Object:              "decisions",
		PromptFormatVersion: req.PromptFormatVersion,
		Temperature:         req.Temperature,
		Model:               req.Model,
		Results:             make([]DecisionsResult, len(req.Options)),
		LabelMass:           make(map[string]float64, len(req.Options)),
	}
	for i, opt := range req.Options {
		label := string(rune('A' + i))
		resp.Results[i] = DecisionsResult{
			Label:       label,
			ID:          opt.ID,
			Logit:       logits[i],
			Probability: probs[i],
		}
		resp.LabelMass[label] = probs[i]
	}
	writeJSON(w, http.StatusOK, resp)
}
