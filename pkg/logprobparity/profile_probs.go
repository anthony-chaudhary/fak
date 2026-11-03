package logprobparity

import (
	"context"
	"fmt"
)

// probsGenerate speaks the /completion wire: token ids plus top-k
// log-probabilities per position.
func probsGenerate(ctx context.Context, c Client, base, prompt string, maxTokens, topK int) (Generation, error) {
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
}
