package logprobparity

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// openAIGenerate speaks the /v1/completions wire: native decode token ids
// first, then legacy completion logprob rows, then text.
func openAIGenerate(ctx context.Context, c Client, base, prompt string, maxTokens, topK int) (Generation, error) {
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
