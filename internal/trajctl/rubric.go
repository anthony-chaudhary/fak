package trajctl

// rubric.go — issue #2544, the rubric rung of the trajectory-control epic
// (#2533): an issue-specific rubric generated ONCE at objective-declare time
// (one model call) and cached on the Objective as ledger metadata, so every
// later judge call scores against concrete criteria instead of the bare
// objective statement. SWE-TRACE (arXiv:2604.14820) shows issue-specific
// rubrics — localization constraints, edit constraints, trajectory
// discipline, budget awareness — judge long-horizon work far better than a
// bare goal prompt; this file moves the W1 judge toward that without
// inventing a new witness rung: rubric-based rows are still W1.
//
// Two properties are enforced, not merely documented:
//   - GENERATE ONCE, CACHE WITH THE OBJECTIVE: the rubric is produced by
//     GenerateObjectiveRubric at declare time and travels inside the
//     objective's ledger row (Objective.Rubric). Score-time code only READS
//     the cached rubric — there is no per-score regeneration path, so the
//     one-call cost model of the issue cannot balloon.
//   - BUDGET FAIL-CLOSED, LIKE THE JUDGE: the generation call carries a
//     per-call token cap forwarded as the request ceiling AND enforced on the
//     returned usage; a nil client, non-positive cap, failed call, over-cap
//     return, or empty/invalid rubric yields an error and NO rubric — never a
//     silently degraded one.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const (
	// DefaultRubricMaxCallTokens is the per-call output-token cap used when a
	// rubric generation is requested with a non-positive cap. A rubric is a
	// handful of short criteria; 1024 is generous headroom.
	DefaultRubricMaxCallTokens = 1024

	// MaxRubricCriteria caps how many criteria a generated rubric may carry.
	// Past a handful, criteria stop being attributable and start being prose.
	MaxRubricCriteria = 8
)

// RubricCriterion is one concrete, checkable criterion of an issue-specific
// rubric. ID is the stable key per-criterion attribution cites.
type RubricCriterion struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

// Rubric is the issue-specific rubric cached on an Objective at declare time.
// Source records provenance (e.g. the generating model id) so a later audit
// can tell a generated rubric from a hand-authored one.
type Rubric struct {
	Criteria []RubricCriterion `json:"criteria"`
	Source   string            `json:"source,omitempty"`
}

// RubricFinding is one criterion's slice of a judge verdict: how far this
// criterion is satisfied in [0,1] and a short note. It is what "rows cite
// which rubric criteria moved" means — each finding becomes one evidence ref
// on the emitted score row.
type RubricFinding struct {
	ID       string  `json:"id"`
	Progress float64 `json:"progress"`
	Note     string  `json:"note,omitempty"`
}

// RubricRequest is the declare-time generation request: the objective
// statement (plus its plan-phase titles for grounding) and the per-call token
// cap the client MUST forward as the request's output ceiling.
type RubricRequest struct {
	Objective string
	Plan      []PlanPhase
	MaxTokens int
}

// RubricClient serves one rubric-generation call. Like JudgeClient it is the
// injected impurity: GenerateObjectiveRubric folds its result, the client
// owns the network and the pinned-schema request shape.
type RubricClient interface {
	GenerateRubric(req RubricRequest) (Rubric, JudgeUsage, error)
}

// GenerateObjectiveRubric runs the ONE declare-time model call and returns the
// rubric to cache on the objective. It fails closed at every boundary: a nil
// client, a failed call, a returned spend over the cap, or a rubric that
// normalizes to nothing all return an error rather than a degraded rubric.
// A non-positive maxCall gets the conservative default cap.
func GenerateObjectiveRubric(client RubricClient, obj Objective, maxCall int) (*Rubric, error) {
	if client == nil {
		return nil, fmt.Errorf("trajctl: rubric generation needs a client")
	}
	if maxCall <= 0 {
		maxCall = DefaultRubricMaxCallTokens
	}
	r, usage, err := client.GenerateRubric(RubricRequest{
		Objective: obj.Statement,
		Plan:      obj.Plan,
		MaxTokens: maxCall,
	})
	if err != nil {
		return nil, fmt.Errorf("trajctl: rubric generation: %w", err)
	}
	if usage.Tokens > maxCall {
		return nil, fmt.Errorf("trajctl: rubric call spent %d tokens, cap %d (fail-closed)", usage.Tokens, maxCall)
	}
	norm := normalizeRubric(r)
	if err := validateRubric(norm); err != nil {
		return nil, err
	}
	return &norm, nil
}

// normalizeRubric trims criteria, drops empty-text entries, synthesizes a
// stable id for any the model left blank, and caps the count, so a slightly
// sloppy generation still yields a deterministic, attributable rubric.
func normalizeRubric(r Rubric) Rubric {
	out := Rubric{Source: strings.TrimSpace(r.Source)}
	for _, c := range r.Criteria {
		text := strings.TrimSpace(c.Text)
		if text == "" {
			continue
		}
		id := strings.TrimSpace(c.ID)
		if id == "" {
			id = fmt.Sprintf("c%d", len(out.Criteria)+1)
		}
		out.Criteria = append(out.Criteria, RubricCriterion{ID: id, Text: text})
		if len(out.Criteria) == MaxRubricCriteria {
			break
		}
	}
	return out
}

// validateRubric checks a rubric is attributable: at least one criterion,
// every criterion carrying a unique id and non-empty text. It guards both the
// generation fold above and the objective ledger row (validateObjective).
func validateRubric(r Rubric) error {
	if len(r.Criteria) == 0 {
		return fmt.Errorf("trajctl: rubric has no criteria")
	}
	if len(r.Criteria) > MaxRubricCriteria {
		return fmt.Errorf("trajctl: rubric has %d criteria, max %d", len(r.Criteria), MaxRubricCriteria)
	}
	seen := map[string]bool{}
	for _, c := range r.Criteria {
		if strings.TrimSpace(c.ID) == "" {
			return fmt.Errorf("trajctl: rubric criterion id is required")
		}
		if strings.TrimSpace(c.Text) == "" {
			return fmt.Errorf("trajctl: rubric criterion %q text is required", c.ID)
		}
		if seen[c.ID] {
			return fmt.Errorf("trajctl: duplicate rubric criterion %q", c.ID)
		}
		seen[c.ID] = true
	}
	return nil
}

// FormatRubricForPrompt renders the cached rubric as the stable RUBRIC block
// the judge prompt carries. The rendering is deterministic (criteria in
// declared order, one numbered line each) so re-scoring the same objective
// reuses the provider prompt cache.
func FormatRubricForPrompt(r *Rubric) string {
	if r == nil || len(r.Criteria) == 0 {
		return ""
	}
	var b strings.Builder
	for i, c := range r.Criteria {
		fmt.Fprintf(&b, "%d. [%s] %s\n", i+1, c.ID, c.Text)
	}
	return b.String()
}

// rubricToolName is the single forced-choice tool a generation call must use;
// its parameters ARE the pinned rubric shape.
const rubricToolName = "emit_rubric"

// rubricSystemPrompt is the STABLE generation instruction — kept verbatim so
// the provider prompt cache reuses it; the objective/plan is the variable tail.
const rubricSystemPrompt = "You are a rubric author for judging long-horizon software work. " +
	"Given an OBJECTIVE (and optionally its PLAN), produce 3-8 concrete, checkable criteria " +
	"an impartial judge can score independently: localization constraints (the right files/areas), " +
	"edit constraints (the right kind and size of change), trajectory discipline (no detours or scope creep), " +
	"and budget awareness (finishing within declared turns/tokens). " +
	"Each criterion gets a short stable id (c1, c2, ...) and one testable sentence. " +
	"Respond only by calling the " + rubricToolName + " tool."

// rubricToolParameters is the PINNED JSON Schema of the generation tool.
var rubricToolParameters = json.RawMessage(`{
  "type": "object",
  "properties": {
    "criteria": {
      "type": "array",
      "minItems": 1,
      "maxItems": 8,
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string", "description": "short stable criterion id, e.g. c1"},
          "text": {"type": "string", "description": "one concrete, checkable criterion"}
        },
        "required": ["id", "text"],
        "additionalProperties": false
      }
    }
  },
  "required": ["criteria"],
  "additionalProperties": false
}`)

// GatewayRubricClient is a RubricClient backed by the same OpenAI-compatible
// chat surface GatewayJudgeClient uses: pinned schema, forced tool choice,
// max_tokens forwarded from the request cap.
type GatewayRubricClient struct {
	// BaseURL is the OpenAI-compatible API root. A trailing slash is tolerated.
	BaseURL string
	// APIKey, when set, is sent as a Bearer credential.
	APIKey string
	// Model is the model id the generation call requests; it is also recorded
	// as the returned rubric's Source.
	Model string
	// Client is the HTTP client; nil uses the judge client's 60s default.
	Client *http.Client
}

// GenerateRubric implements RubricClient against <BaseURL>/chat/completions.
// Any transport, status, or shape error is returned so the declare-time fold
// can fail closed.
func (c *GatewayRubricClient) GenerateRubric(req RubricRequest) (Rubric, JudgeUsage, error) {
	if strings.TrimSpace(c.BaseURL) == "" {
		return Rubric{}, JudgeUsage{}, fmt.Errorf("trajctl: rubric client base URL is required")
	}
	var user strings.Builder
	fmt.Fprintf(&user, "OBJECTIVE:\n%s\n", req.Objective)
	if len(req.Plan) > 0 {
		fmt.Fprintf(&user, "\nPLAN:\n")
		for i, p := range req.Plan {
			title := p.Title
			if title == "" {
				title = p.ID
			}
			fmt.Fprintf(&user, "%d. %s\n", i+1, title)
		}
	}
	body := chatVerdictRequest{
		Model: c.Model,
		Messages: []chatMessage{
			{Role: "system", Content: rubricSystemPrompt},
			{Role: "user", Content: user.String()},
		},
		Tools: []chatTool{{
			Type: "function",
			Function: chatToolFunction{
				Name:        rubricToolName,
				Description: "Emit the issue-specific judging rubric.",
				Parameters:  rubricToolParameters,
			},
		}},
		ToolChoice:  chatToolChoice{Type: "function", Function: chatToolChoiceName{Name: rubricToolName}},
		MaxTokens:   req.MaxTokens,
		Temperature: 0,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return Rubric{}, JudgeUsage{}, err
	}

	url := strings.TrimRight(c.BaseURL, "/") + "/chat/completions"
	proxy := GatewayJudgeClient{Client: c.Client}
	ctx, cancel := context.WithTimeout(context.Background(), proxy.timeout())
	defer cancel()
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return Rubric{}, JudgeUsage{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	resp, err := proxy.httpClient().Do(httpReq)
	if err != nil {
		return Rubric{}, JudgeUsage{}, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return Rubric{}, JudgeUsage{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return Rubric{}, JudgeUsage{}, fmt.Errorf("trajctl: rubric call %s: status %d: %s", url, resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	var parsed chatVerdictResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return Rubric{}, JudgeUsage{}, fmt.Errorf("trajctl: rubric response: %w", err)
	}
	if len(parsed.Choices) == 0 || len(parsed.Choices[0].Message.ToolCalls) == 0 {
		return Rubric{}, JudgeUsage{}, fmt.Errorf("trajctl: rubric response carried no tool call")
	}
	call := parsed.Choices[0].Message.ToolCalls[0]
	if call.Function.Name != rubricToolName {
		return Rubric{}, JudgeUsage{}, fmt.Errorf("trajctl: rubric call answered %q, want %q", call.Function.Name, rubricToolName)
	}
	var rubric Rubric
	if err := json.Unmarshal([]byte(call.Function.Arguments), &rubric); err != nil {
		return Rubric{}, JudgeUsage{}, fmt.Errorf("trajctl: rubric arguments: %w", err)
	}
	rubric.Source = c.Model
	return rubric, JudgeUsage{Tokens: parsed.Usage.TotalTokens}, nil
}
