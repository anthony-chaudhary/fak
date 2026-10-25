package logprobparity

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
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

// Generate decodes maxTokens greedy tokens after the raw prompt (no chat
// template). topK is the requested top-k for an API that reports one.
func (c Client) Generate(ctx context.Context, prompt string, maxTokens, topK int) (Generation, error) {
	base := strings.TrimSuffix(strings.TrimRight(c.BaseURL, "/"), "/v1")
	switch c.API {
	case APILlama:
		body := map[string]any{"prompt": prompt, "n_predict": maxTokens, "temperature": 0, "seed": 42,
			"n_probs": topK, "post_sampling_probs": false, "cache_prompt": false, "return_tokens": true}
		var out struct {
			Content     string `json:"content"`
			Tokens      []int  `json:"tokens"`
			Evaluated   int    `json:"tokens_evaluated"`
			Probability []struct {
				ID      int     `json:"id"`
				Token   string  `json:"token"`
				Logprob float64 `json:"logprob"`
				Top     []struct {
					ID      int     `json:"id"`
					Token   string  `json:"token"`
					Logprob float64 `json:"logprob"`
				} `json:"top_logprobs"`
			} `json:"completion_probabilities"`
		}
		if err := c.post(ctx, base+"/completion", body, &out); err != nil {
			return Generation{}, err
		}
		g := Generation{Text: out.Content, PromptTokens: out.Evaluated}
		for _, p := range out.Probability {
			pos := Position{ID: p.ID, Piece: p.Token, TopK: []Candidate{}}
			for _, t := range p.Top {
				pos.TopK = append(pos.TopK, Candidate{ID: t.ID, Piece: t.Token, Logprob: t.Logprob})
			}
			g.Positions = append(g.Positions, pos)
		}
		if len(g.Positions) == 0 && len(out.Tokens) > 0 {
			return Generation{}, fmt.Errorf("%w: %s returned tokens without completion_probabilities", ErrEndpoint, base)
		}
		return g, nil
	case APIOpenAI, "":
		reportK := topK
		if reportK < 0 {
			reportK = 0
		}
		if reportK > 5 {
			reportK = 5
		}
		body := map[string]any{"prompt": prompt, "max_tokens": maxTokens, "temperature": 0, "seed": 42,
			"logprobs": reportK, "fak_decode_trace": true, "fak": map[string]any{"native_decode_token_ids": true}}
		if m := strings.TrimSpace(c.Model); m != "" {
			body["model"] = m
		}
		var out struct {
			Choices []struct {
				Text     string `json:"text"`
				Logprobs *struct {
					Tokens        []string              `json:"tokens"`
					TokenLogprobs []*float64            `json:"token_logprobs"`
					TopLogprobs   []map[string]*float64 `json:"top_logprobs"`
				} `json:"logprobs"`
			} `json:"choices"`
			Usage struct {
				PromptTokens int `json:"prompt_tokens"`
			} `json:"usage"`
			Fak *struct {
				TokenIDs *struct {
					TokenIDs []int `json:"token_ids"`
				} `json:"native_decode_token_ids"`
			} `json:"fak"`
		}
		err := c.post(ctx, base+"/v1/completions", body, &out)
		if errors.Is(err, errBadRequest) {
			// A non-native route or a strict OpenAI server refuses the extension.
			delete(body, "fak_decode_trace")
			delete(body, "fak")
			err = c.post(ctx, base+"/v1/completions", body, &out)
		}
		if err != nil {
			return Generation{}, err
		}
		if len(out.Choices) == 0 {
			return Generation{}, fmt.Errorf("%w: %s returned no choices", ErrEndpoint, base)
		}
		g := Generation{Text: out.Choices[0].Text, PromptTokens: out.Usage.PromptTokens}
		if out.Fak != nil && out.Fak.TokenIDs != nil {
			g.Positions = make([]Position, 0, len(out.Fak.TokenIDs.TokenIDs))
			for _, id := range out.Fak.TokenIDs.TokenIDs {
				g.Positions = append(g.Positions, Position{ID: id})
			}
			return g, nil
		}
		rows := out.Choices[0].Logprobs
		if rows == nil {
			return g, nil
		}
		if rows.Tokens == nil || rows.TokenLogprobs == nil || rows.TopLogprobs == nil {
			return Generation{}, fmt.Errorf("%w: %s returned a completion logprobs object without required row arrays", ErrEndpoint, base)
		}
		if len(rows.Tokens) != len(rows.TokenLogprobs) || len(rows.Tokens) != len(rows.TopLogprobs) {
			return Generation{}, fmt.Errorf("%w: %s returned misaligned completion logprob rows: tokens=%d token_logprobs=%d top_logprobs=%d", ErrEndpoint, base, len(rows.Tokens), len(rows.TokenLogprobs), len(rows.TopLogprobs))
		}
		if len(rows.Tokens) == 0 && out.Choices[0].Text != "" {
			return Generation{}, fmt.Errorf("%w: %s returned nonempty completion text without logprob rows", ErrEndpoint, base)
		}
		g.Positions = make([]Position, 0, len(rows.Tokens))
		g.Method = TextPieces
		for i, piece := range rows.Tokens {
			if rows.TokenLogprobs[i] == nil {
				return Generation{}, fmt.Errorf("%w: %s returned null token_logprobs row %d", ErrEndpoint, base, i)
			}
			if rows.TopLogprobs[i] == nil {
				return Generation{}, fmt.Errorf("%w: %s returned null top_logprobs row %d", ErrEndpoint, base, i)
			}
			type entry struct {
				piece   string
				logprob float64
			}
			entries := make([]entry, 0, len(rows.TopLogprobs[i]))
			for candidatePiece, logprob := range rows.TopLogprobs[i] {
				if logprob == nil {
					return Generation{}, fmt.Errorf("%w: %s returned null top_logprobs value at row %d", ErrEndpoint, base, i)
				}
				entries = append(entries, entry{piece: candidatePiece, logprob: *logprob})
			}
			sort.Slice(entries, func(i, j int) bool {
				if entries[i].logprob == entries[j].logprob {
					return entries[i].piece < entries[j].piece
				}
				return entries[i].logprob > entries[j].logprob
			})
			pos := Position{ID: UnknownID, Piece: piece, TopK: make([]Candidate, 0, len(entries)), ReportedTopK: reportK}
			for _, candidate := range entries {
				pos.TopK = append(pos.TopK, Candidate{ID: UnknownID, Piece: candidate.piece, Logprob: candidate.logprob})
			}
			g.Positions = append(g.Positions, pos)
		}
		return g, nil
	}
	return Generation{}, fmt.Errorf("%w: unknown api %q", ErrEndpoint, c.API)
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
