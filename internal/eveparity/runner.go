package eveparity

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// FixtureSuite is the deterministic Eve eval suite #2605 names: t.succeeded,
// t.calledTool, an exact content gate, a soft score, PLUS a deliberately-failing hard
// gate and a soft score that only fails under --strict. Every assertion is satisfiable
// by a fixed scripted response (fixtureModelReply), so no external model key is needed.
//
// Postcondition: Returns an immutable suite containing all five canonical test cases with deterministic expectations.
func FixtureSuite() Suite {
	return Suite{
		Name: "eve-fixture-parity",
		Cases: []Case{
			{
				ID:              "succeeded-and-tool",
				Prompt:          "[[case=succeeded-and-tool]] search the docs",
				Tools:           []string{"search"},
				ExpectSucceeded: true,
				ExpectToolCall:  "search",
			},
			{
				ID:              "content-exact",
				Prompt:          "[[case=content-exact]] what is six times seven",
				ExpectSucceeded: true,
				ExpectContent:   "The answer is 42.",
			},
			{
				ID:              "soft-quality",
				Prompt:          "[[case=soft-quality]] write a thorough summary",
				ExpectSucceeded: true,
				Soft:            &SoftSpec{Name: "quality", Score: 0.90, Threshold: 0.70},
			},
			{
				// The deliberately-failing gate: the fixture model never calls "write",
				// so t.calledTool("write") FAILS — on BOTH arms, with the same reason.
				ID:              "deliberate-gate-fail",
				Prompt:          "[[case=deliberate-gate-fail]] refuse to write the file",
				Tools:           []string{"write"},
				ExpectSucceeded: true,
				ExpectToolCall:  "write",
			},
			{
				// A soft score below threshold: passes without --strict, fails under it —
				// the strict soft-threshold behavior #2605 requires be preserved on both arms.
				ID:              "soft-strict-fail",
				Prompt:          "[[case=soft-strict-fail]] give a terse answer",
				ExpectSucceeded: true,
				Soft:            &SoftSpec{Name: "coverage", Score: 0.50, Threshold: 0.70},
			},
		},
	}
}

// scriptedReply is one fixture model turn: the assistant content and, optionally, a
// single tool call. Deterministic — no sampling.
type scriptedReply struct {
	content string
	tool    string // "" => no tool call
}

// fixtureReplies maps a case id to the fixture model's scripted turn. The upstream
// resolves the case from the [[case=...]] tag the runner embeds in the prompt, so the
// SAME script answers whether the request arrived raw or through fak's proxy.
var fixtureReplies = map[string]scriptedReply{
	"succeeded-and-tool":   {content: "searching the docs now", tool: "search"},
	"content-exact":        {content: "The answer is 42."},
	"soft-quality":         {content: "a thorough, well-structured summary"},
	"deliberate-gate-fail": {content: "I will not write the file", tool: ""},
	"soft-strict-fail":     {content: "terse."},
}

// FixtureUpstream is the deterministic mock model: an http.Handler that serves the
// OpenAI /v1/chat/completions wire, replying per case from fixtureReplies. It is the
// single upstream BOTH arms hit — the raw arm points at it directly, the fak arm
// reaches it through fak's gateway proxy — so any transcript difference is fak's doing.
func FixtureUpstream() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Model    string `json:"model"`
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(body, &req)
		caseID := extractCaseTag(req.Messages)
		reply := fixtureReplies[caseID]

		msg := map[string]any{"role": "assistant", "content": reply.content}
		finish := "stop"
		if reply.tool != "" {
			msg["tool_calls"] = []map[string]any{{
				"id":       "call_" + reply.tool,
				"type":     "function",
				"function": map[string]any{"name": reply.tool, "arguments": "{}"},
			}}
			finish = "tool_calls"
		}
		resp := map[string]any{
			"id":      "chatcmpl-fixture-" + caseID,
			"object":  "chat.completion",
			"created": 0,
			"model":   "fixture",
			"choices": []map[string]any{{"index": 0, "message": msg, "finish_reason": finish}},
			// Non-zero usage so token metadata is observably PRESENT on the raw arm and
			// provably preserved after fak proxies it.
			"usage": map[string]any{"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})
	return mux
}

// extractCaseTag pulls the [[case=<id>]] marker out of the request messages. The tag
// rides in the user prompt and survives fak's transparent proxy, so the upstream can
// pick the right scripted reply for either arm without any out-of-band state.
func extractCaseTag(messages []struct {
	Content string `json:"content"`
}) string {
	for _, m := range messages {
		if i := strings.Index(m.Content, "[[case="); i >= 0 {
			rest := m.Content[i+len("[[case="):]
			if j := strings.Index(rest, "]]"); j >= 0 {
				return rest[:j]
			}
		}
	}
	return ""
}

// RunArm drives every case in the suite against baseURL (a fixture upstream for the
// raw arm, or a fak gateway for the fak arm), evaluates each transcript with the shared
// scorer, and folds the run into an ArmResult. command is the human-facing command line
// recorded in the witness. Any per-case transport error surfaces as a failed t.succeeded
// gate rather than aborting the run, so a broken arm is a visible divergence, not a panic.
//
// Precondition: Target base URL must expose an active completions endpoint responding to chat completions.
// Postcondition: ArmResult captures execution outcomes along with session and token metadata preservation flags.
func RunArm(arm, command, baseURL string, suite Suite, strict bool, client *http.Client) ArmResult {
	if client == nil {
		// A bounded default so a hung fixture server (or a real gateway that stalls)
		// surfaces as a per-case transport error, never an indefinite hang.
		client = &http.Client{Timeout: 30 * time.Second}
	}
	res := ArmResult{Arm: arm, Command: command, SessionIDsPresent: true, TokenMetadataPresent: true}
	for _, c := range suite.Cases {
		tr := runCase(client, baseURL, c)
		if tr.SessionID == "" {
			res.SessionIDsPresent = false
		}
		if tr.PromptTokens == 0 && tr.CompletionTokens == 0 {
			res.TokenMetadataPresent = false
		}
		res.Cases = append(res.Cases, Evaluate(c, tr, strict))
	}
	return res
}

// runCase sends one case's turn over the OpenAI chat wire and reads the assistant
// message back into a Transcript. It is arm-agnostic: the only difference between the
// raw and fak arms is the baseURL it is handed.
func runCase(client *http.Client, baseURL string, c Case) Transcript {
	tr := Transcript{CaseID: c.ID}
	reqBody := map[string]any{
		"model":    "fixture",
		"messages": []map[string]any{{"role": "user", "content": c.Prompt}},
	}
	if len(c.Tools) > 0 {
		var tools []map[string]any
		for _, name := range c.Tools {
			tools = append(tools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        name,
					"description": "fixture tool " + name,
					"parameters":  map[string]any{"type": "object", "properties": map[string]any{}},
				},
			})
		}
		reqBody["tools"] = tools
	}
	raw, _ := json.Marshal(reqBody)
	httpResp, err := client.Post(strings.TrimRight(baseURL, "/")+"/v1/chat/completions", "application/json", bytes.NewReader(raw))
	if err != nil {
		tr.Err = "transport: " + err.Error()
		return tr
	}
	defer httpResp.Body.Close()
	respBody, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		tr.Err = fmt.Sprintf("upstream status %d: %s", httpResp.StatusCode, strings.TrimSpace(string(respBody)))
		return tr
	}
	var parsed struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct {
				Content   string `json:"content"`
				ToolCalls []struct {
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		tr.Err = "decode: " + err.Error()
		return tr
	}
	if len(parsed.Choices) == 0 {
		tr.Err = "no choices in response"
		return tr
	}
	tr.SessionID = parsed.ID
	tr.Succeeded = true
	tr.FinalText = parsed.Choices[0].Message.Content
	for _, tc := range parsed.Choices[0].Message.ToolCalls {
		if tc.Function.Name != "" {
			tr.ToolCalls = append(tr.ToolCalls, tc.Function.Name)
		}
	}
	tr.PromptTokens = parsed.Usage.PromptTokens
	tr.CompletionTokens = parsed.Usage.CompletionTokens
	return tr
}
