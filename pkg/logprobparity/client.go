package logprobparity

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/referenceoracle"
)

// Endpoint APIs a Client speaks.
const (
	// APIOpenAI is POST /v1/completions with a raw text prompt. It prefers fak's
	// native decode token ids, then legacy completion logprob rows, then text.
	APIOpenAI = "openai"
	// APILlama is llama-server's POST /completion with n_probs: token ids and
	// top-k log-probabilities.
	APILlama = "llama"
)

// ErrEndpoint wraps every transport, status and decode failure of a Client.
var ErrEndpoint = errors.New("logprobparity: endpoint")

var errBadRequest = errors.New("bad request")

// Client is one greedy-decoding endpoint.
type Client struct {
	BaseURL string
	APIKey  string
	Model   string
	API     string
	HTTP    *http.Client
}

// Generation is one greedy continuation. Positions is nil when the API reports
// neither token IDs nor usable token-piece logprob rows.
type Generation struct {
	Text         string
	Positions    []Position
	PromptTokens int
	Method       string
}

// apiProfile is one endpoint wire format: the generator that posts the raw
// prompt under the normalized base URL and decodes the response.
type apiProfile struct {
	generate func(ctx context.Context, c Client, base, prompt string, maxTokens, topK int) (Generation, error)
}

// apiProfiles maps Client.API names to their wire format; an empty API uses
// the OpenAI-compatible profile.
var apiProfiles = map[string]apiProfile{
	APIOpenAI: {generate: openAIGenerate},
	APILlama:  {generate: probsGenerate},
}

// Generate decodes maxTokens greedy tokens after the raw prompt (no chat
// template). topK is the requested top-k for an API that reports one.
func (c Client) Generate(ctx context.Context, prompt string, maxTokens, topK int) (Generation, error) {
	base := strings.TrimSuffix(strings.TrimRight(c.BaseURL, "/"), "/v1")
	p, known := apiProfiles[c.API]
	if !known {
		if c.API != "" {
			return Generation{}, fmt.Errorf("%w: unknown api %q", ErrEndpoint, c.API)
		}
		p = apiProfiles[APIOpenAI]
	}
	return p.generate(ctx, c, base, prompt, maxTokens, topK)
}

func (c Client) post(ctx context.Context, url string, body any, out any) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return fmt.Errorf("%w: %v", ErrEndpoint, err)
	}
	req.Header.Set("Content-Type", "application/json")
	if k := strings.TrimSpace(c.APIKey); k != "" {
		req.Header.Set("Authorization", "Bearer "+k)
	}
	hc := c.HTTP
	if hc == nil {
		hc = http.DefaultClient
	}
	resp, err := hc.Do(req)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrEndpoint, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return fmt.Errorf("%w: %s: %v", ErrEndpoint, url, err)
	}
	if resp.StatusCode != http.StatusOK {
		snippet := strings.TrimSpace(string(data))
		if len(snippet) > 240 {
			snippet = snippet[:240]
		}
		if resp.StatusCode == http.StatusBadRequest {
			return fmt.Errorf("%w: %w: %s: HTTP %d: %s", ErrEndpoint, errBadRequest, url, resp.StatusCode, snippet)
		}
		return fmt.Errorf("%w: %s: HTTP %d: %s", ErrEndpoint, url, resp.StatusCode, snippet)
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("%w: %s: decode: %v", ErrEndpoint, url, err)
	}
	return nil
}

// RefPrompt is one reference continuation to score a candidate against.
type RefPrompt struct {
	Text string
	// InputTokens is the reference's prompt length, 0 when unknown.
	InputTokens int
	Positions   []Position
}

// OraclePrompts adapts a recorded oracle: position i is greedy id i with the
// oracle's top-5 row (whose first entry carries the greedy piece).
func OraclePrompts(o referenceoracle.Oracle) []RefPrompt {
	var out []RefPrompt
	for _, p := range o.Prompts {
		rp := RefPrompt{Text: p.Text, InputTokens: len(p.InputTokenIDs)}
		for i, id := range p.GreedyIDs {
			pos := Position{ID: id}
			if i < len(p.Top5) {
				pos.TopK = []Candidate{}
				for _, c := range p.Top5[i] {
					pos.TopK = append(pos.TopK, Candidate{ID: c.ID, Piece: c.Piece, Logprob: c.Logprob})
				}
				if len(p.Top5[i]) > 0 {
					pos.Piece = p.Top5[i][0].Piece
				}
			}
			rp.Positions = append(rp.Positions, pos)
		}
		out = append(out, rp)
	}
	return out
}

// PromptResult is one prompt's parity plus the prompt-length cross-check: a
// candidate that tokenized the prompt to a different length (for example
// without the BOS the reference used) is not conditioned on the same input.
type PromptResult struct {
	Result
	ActualPromptTokens    int    `json:"actual_prompt_tokens,omitempty"`
	ReferencePromptTokens int    `json:"reference_prompt_tokens,omitempty"`
	PromptTokensMatch     *bool  `json:"prompt_tokens_match,omitempty"`
	Error                 string `json:"error,omitempty"`
}

// Score generates each reference prompt on actual and checks it. A failed
// generation is recorded per prompt (verdict EMPTY) rather than aborting the
// battery; the returned error is the first such failure.
func Score(ctx context.Context, actual Client, refs []RefPrompt, topK int) ([]PromptResult, error) {
	var out []PromptResult
	var first error
	for _, ref := range refs {
		pr := PromptResult{ReferencePromptTokens: ref.InputTokens}
		g, err := actual.Generate(ctx, ref.Text, len(ref.Positions), topK)
		if err != nil {
			if first == nil {
				first = err
			}
			pr.Result = Result{Prompt: ref.Text, Verdict: Empty}
			pr.Error = err.Error()
			out = append(out, pr)
			continue
		}
		method, positions := g.Method, g.Positions
		if method == "" {
			method = TokenIDs
		}
		if positions == nil {
			method, positions = TextPieces, SegmentText(g.Text, ref.Positions)
		}
		pr.Result = Check(positions, ref.Positions, topK, method)
		if method == TextPieces && pr.DivergedAt != nil {
			// Segmentation stops at the divergence; the candidate still
			// generated against every reference position.
			pr.Compared = len(ref.Positions)
		}
		pr.Prompt = ref.Text
		pr.ActualPromptTokens = g.PromptTokens
		if g.PromptTokens > 0 && ref.InputTokens > 0 {
			same := g.PromptTokens == ref.InputTokens
			pr.PromptTokensMatch = &same
		}
		out = append(out, pr)
	}
	return out, first
}

// FoldPrompts folds prompt results into a report.
func FoldPrompts(prs []PromptResult) Report {
	rs := make([]Result, len(prs))
	for i, p := range prs {
		rs[i] = p.Result
	}
	return Fold(rs)
}
