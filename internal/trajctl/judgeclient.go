package trajctl

// judgeclient.go — issue #2543: the gateway-backed JudgeClient. It turns a
// JudgeRequest into a structured verdict call against an OpenAI-compatible
// endpoint (fak's own gateway, or any compatible upstream it fronts) and folds
// the reply back into a JudgeVerdict + JudgeUsage. Two request-shape properties
// make the call cheap and reliable:
//   - PINNED SCHEMA + FORCED TOOL CHOICE: the request advertises exactly one
//     tool whose parameters ARE the JudgeVerdict schema and forces the model to
//     call it, so the reply is always well-formed structured JSON, never prose
//     the scorer would have to salvage.
//   - CACHE-FRIENDLY PROMPT SHAPE: a fixed system instruction leads every call,
//     so the provider prompt cache reuses the stable prefix across objectives;
//     only the short objective/state tail varies.
//
// The per-call output ceiling (req.MaxTokens) is forwarded as max_tokens so the
// budget cap the scorer enforces is also applied at the source.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// judgeToolName is the single forced-choice tool the model must call. Its
// parameters schema is the pinned JudgeVerdict shape.
const judgeToolName = "emit_verdict"

// judgeSystemPrompt is the STABLE instruction prefix — kept verbatim across
// every call so the provider prompt cache reuses it. It never contains the
// objective or state (those are the variable tail).
const judgeSystemPrompt = "You are a strict progress judge. Given an OBJECTIVE and the CURRENT STATE, " +
	"estimate how much of the objective is complete as a number in [0,1] and whether it is fully met. " +
	"Be conservative: partial or unverified progress is not completion. " +
	"Respond only by calling the " + judgeToolName + " tool with your verdict."

// GatewayJudgeClient is a JudgeClient backed by an OpenAI-compatible chat
// endpoint. BaseURL is the API root (e.g. http://127.0.0.1:8000/v1); the client
// POSTs to <BaseURL>/chat/completions.
type GatewayJudgeClient struct {
	// BaseURL is the OpenAI-compatible API root. A trailing slash is tolerated.
	BaseURL string
	// APIKey, when set, is sent as a Bearer credential.
	APIKey string
	// Model is the model id the verdict call requests.
	Model string
	// Client is the HTTP client; nil uses a 60s-timeout default.
	Client *http.Client
}

// verdictToolParameters is the PINNED JSON Schema advertised as the verdict
// tool's parameters — the machine contract the model's structured output must
// satisfy.
var verdictToolParameters = json.RawMessage(`{
  "type": "object",
  "properties": {
    "progress": {"type": "number", "minimum": 0, "maximum": 1, "description": "fraction of the objective complete, 0..1"},
    "met": {"type": "boolean", "description": "true only if the objective is fully satisfied"},
    "rationale": {"type": "string", "description": "one or two sentences justifying the estimate"}
  },
  "required": ["progress", "rationale"],
  "additionalProperties": false
}`)

// chatVerdictRequest is the OpenAI-compatible request body the judge call sends.
type chatVerdictRequest struct {
	Model       string         `json:"model"`
	Messages    []chatMessage  `json:"messages"`
	Tools       []chatTool     `json:"tools"`
	ToolChoice  chatToolChoice `json:"tool_choice"`
	MaxTokens   int            `json:"max_tokens"`
	Temperature float64        `json:"temperature"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatTool struct {
	Type     string           `json:"type"`
	Function chatToolFunction `json:"function"`
}

type chatToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// chatToolChoice forces the model to call exactly the named function.
type chatToolChoice struct {
	Type     string             `json:"type"`
	Function chatToolChoiceName `json:"function"`
}

type chatToolChoiceName struct {
	Name string `json:"name"`
}

// chatVerdictResponse is the subset of the OpenAI-compatible reply the judge
// call reads: the first tool call's arguments (the verdict) and token usage.
type chatVerdictResponse struct {
	Choices []struct {
		Message struct {
			ToolCalls []struct {
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// Judge implements JudgeClient: it POSTs a pinned-schema, forced-tool-choice
// verdict request and folds the reply into a JudgeVerdict + JudgeUsage. The
// per-call token cap rides as max_tokens. Any transport, status, or shape error
// is returned so the scorer can fail closed.
func (c *GatewayJudgeClient) Judge(req JudgeRequest) (JudgeVerdict, JudgeUsage, error) {
	if strings.TrimSpace(c.BaseURL) == "" {
		return JudgeVerdict{}, JudgeUsage{}, fmt.Errorf("trajctl: judge client base URL is required")
	}
	body := chatVerdictRequest{
		Model: c.Model,
		Messages: []chatMessage{
			{Role: "system", Content: judgeSystemPrompt},
			{Role: "user", Content: fmt.Sprintf("OBJECTIVE:\n%s\n\nCURRENT STATE:\n%s", req.Objective, req.State)},
		},
		Tools: []chatTool{{
			Type: "function",
			Function: chatToolFunction{
				Name:        judgeToolName,
				Description: "Emit the structured progress verdict.",
				Parameters:  verdictToolParameters,
			},
		}},
		ToolChoice:  chatToolChoice{Type: "function", Function: chatToolChoiceName{Name: judgeToolName}},
		MaxTokens:   req.MaxTokens,
		Temperature: 0,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return JudgeVerdict{}, JudgeUsage{}, err
	}

	url := strings.TrimRight(c.BaseURL, "/") + "/chat/completions"
	ctx, cancel := context.WithTimeout(context.Background(), c.timeout())
	defer cancel()
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return JudgeVerdict{}, JudgeUsage{}, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	resp, err := c.httpClient().Do(httpReq)
	if err != nil {
		return JudgeVerdict{}, JudgeUsage{}, err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return JudgeVerdict{}, JudgeUsage{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return JudgeVerdict{}, JudgeUsage{}, fmt.Errorf("trajctl: judge call %s: status %d: %s", url, resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	var parsed chatVerdictResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return JudgeVerdict{}, JudgeUsage{}, fmt.Errorf("trajctl: judge response: %w", err)
	}
	if len(parsed.Choices) == 0 || len(parsed.Choices[0].Message.ToolCalls) == 0 {
		return JudgeVerdict{}, JudgeUsage{}, fmt.Errorf("trajctl: judge response carried no tool call")
	}
	call := parsed.Choices[0].Message.ToolCalls[0]
	if call.Function.Name != judgeToolName {
		return JudgeVerdict{}, JudgeUsage{}, fmt.Errorf("trajctl: judge called %q, want %q", call.Function.Name, judgeToolName)
	}
	var verdict JudgeVerdict
	if err := json.Unmarshal([]byte(call.Function.Arguments), &verdict); err != nil {
		return JudgeVerdict{}, JudgeUsage{}, fmt.Errorf("trajctl: judge verdict arguments: %w", err)
	}
	return verdict, JudgeUsage{Tokens: parsed.Usage.TotalTokens}, nil
}

func (c *GatewayJudgeClient) httpClient() *http.Client {
	if c.Client != nil {
		return c.Client
	}
	return &http.Client{Timeout: 60 * time.Second}
}

func (c *GatewayJudgeClient) timeout() time.Duration {
	if c.Client != nil && c.Client.Timeout > 0 {
		return c.Client.Timeout
	}
	return 60 * time.Second
}
