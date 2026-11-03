package beam

// The bounded live/completed search and generated-length score are adapted from
// vLLM 5c8ae7c74aaa99d1f7fbeea942fd7282e3413810,
// vllm/entrypoints/generate/beam_search/{offline,utils}.py (Apache-2.0).
// This portable implementation deliberately replays owned CPU sessions; it does
// not use BatchDriver's indirection table or claim shared KV state.
import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
)

const (
	MaxSearchWidth        = 8
	MaxSearchSteps        = 128
	MaxSearchContext      = 4096
	MaxSearchVocab        = 262144
	MaxSearchReplayTokens = 1 << 20
	// Covers cumulative candidate-token/header allocation at all fixed limits,
	// not model/session allocations, allocator overhead, RSS, or a global budget.
	SearchWorkspaceBudgetBytes int64 = 32 << 20
)

var ErrSearchUnsupported = errors.New("beam search: unsupported request")

type SearchConfig struct {
	Width         int
	MaxTokens     int
	VocabSize     int
	LengthPenalty float64
	EOS           []int
}

type SearchSequence struct {
	TokenIDs          []int   `json:"token_ids"`
	GeneratedTokens   int     `json:"generated_tokens"`
	CumulativeLogprob float64 `json:"cumulative_logprob"`
	Score             float64 `json:"score"`
	FinishReason      string  `json:"finish_reason"`
}

type SearchResult struct {
	Sequences        []SearchSequence
	ForwardCalls     int
	ReplayedTokens   int
	ReplayTokenBound int
}

// NextLogits must return the raw logits for the next token of the complete
// owned prefix. It is invoked serially. A production adapter must close its
// session before returning and must not retain prefix. No result is published
// if admission, cancellation, a forward, or finite-logit validation fails.
type NextLogits func(context.Context, []int) ([]float32, error)

func ValidateSearch(prompt []int, cfg SearchConfig) (int, error) {
	bad := func(why string) (int, error) { return 0, fmt.Errorf("%w: %s", ErrSearchUnsupported, why) }
	if cfg.Width < 1 || cfg.Width > MaxSearchWidth || cfg.MaxTokens < 1 || cfg.MaxTokens > MaxSearchSteps {
		return bad("width or generated-token limit")
	}
	if cfg.VocabSize < 1 || cfg.VocabSize > MaxSearchVocab || len(prompt) < 1 || len(prompt) > MaxSearchContext-cfg.MaxTokens {
		return bad("vocabulary or context limit")
	}
	if math.IsNaN(cfg.LengthPenalty) || math.IsInf(cfg.LengthPenalty, 0) || cfg.LengthPenalty < 0 || cfg.LengthPenalty > 2 {
		return bad("length penalty must be finite in [0,2]")
	}
	for _, id := range prompt {
		if id < 0 || id >= cfg.VocabSize {
			return bad("prompt token outside vocabulary")
		}
	}
	if len(cfg.EOS) > cfg.VocabSize {
		return bad("EOS metadata exceeds vocabulary")
	}
	for _, id := range cfg.EOS {
		if id < 0 || id >= cfg.VocabSize {
			return bad("EOS outside vocabulary")
		}
	}
	// One initial prefix, then at most Width surviving prefixes for every
	// remaining step. All operands were bounded above before this calculation.
	n := int64(cfg.MaxTokens - 1)
	bound := int64(len(prompt)) + int64(cfg.Width)*(n*int64(len(prompt))+n*(n+1)/2)
	if bound > MaxSearchReplayTokens {
		return bad("total replay-token limit")
	}
	return int(bound), nil
}

// SearchScore counts generated tokens INCLUDING a terminal EOS, not prompt
// tokens or the visible token slice (which omits EOS), matching pinned vLLM.
func SearchScore(generated int, cumulative, penalty float64) float64 {
	return cumulative / math.Pow(float64(max(generated, 1)), penalty)
}

type searchToken struct {
	id    int
	logit float64
}

func topSearchTokens(logits []float32, vocab, count int) ([]searchToken, float64, float64, error) {
	if len(logits) != vocab {
		return nil, 0, 0, fmt.Errorf("%w: logits shape", ErrSearchUnsupported)
	}
	top := make([]searchToken, 0, min(count, vocab))
	maximum := math.Inf(-1)
	for id, v := range logits {
		x := float64(v)
		if math.IsNaN(x) || math.IsInf(x, 0) {
			return nil, 0, 0, fmt.Errorf("%w: nonfinite logits", ErrSearchUnsupported)
		}
		if x > maximum {
			maximum = x
		}
		at := len(top)
		for at > 0 && (x > top[at-1].logit || (x == top[at-1].logit && id < top[at-1].id)) {
			at--
		}
		if at >= count {
			continue
		}
		if len(top) < count {
			top = append(top, searchToken{})
		}
		copy(top[at+1:], top[at:len(top)-1])
		top[at] = searchToken{id: id, logit: x}
	}
	sum := 0.0
	for _, v := range logits {
		sum += math.Exp(float64(v) - maximum)
	}
	if sum <= 0 || math.IsNaN(sum) || math.IsInf(sum, 0) {
		return nil, 0, 0, fmt.Errorf("%w: softmax denominator", ErrSearchUnsupported)
	}
	return top, maximum, math.Log(sum), nil
}

func rankSearchSequences(rows []SearchSequence, width int) []SearchSequence {
	// Equal scores preserve parent order and ascending next-token ID. Completed
	// rows retain arrival order; final completed/live ties favor completed rows.
	sort.SliceStable(rows, func(i, j int) bool { return rows[i].Score > rows[j].Score })
	if len(rows) > width {
		rows = rows[:width]
	}
	return rows
}

func Search(ctx context.Context, prompt []int, cfg SearchConfig, next NextLogits) (*SearchResult, error) {
	bound, err := ValidateSearch(prompt, cfg)
	if err != nil {
		return nil, err
	}
	if ctx == nil || next == nil {
		return nil, fmt.Errorf("%w: context or logits provider missing", ErrSearchUnsupported)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Capture caller-owned metadata before any forward; the provider receives its
	// own complete token prefix on every call.
	prompt = append([]int(nil), prompt...)
	eos := make(map[int]bool, len(cfg.EOS))
	for _, id := range cfg.EOS {
		eos[id] = true
	}
	live := []SearchSequence{{}}
	completed := make([]SearchSequence, 0, cfg.Width)
	out := &SearchResult{ReplayTokenBound: bound}
	for step := 0; step < cfg.MaxTokens && len(live) > 0; step++ {
		candidates := make([]SearchSequence, 0, 2*cfg.Width*cfg.Width)
		for _, parent := range live {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			prefix := make([]int, len(prompt)+len(parent.TokenIDs))
			copy(prefix, prompt)
			copy(prefix[len(prompt):], parent.TokenIDs)
			if out.ReplayedTokens+len(prefix) > bound {
				return nil, fmt.Errorf("%w: replay accounting", ErrSearchUnsupported)
			}
			logits, err := next(ctx, prefix)
			out.ForwardCalls++
			out.ReplayedTokens += len(prefix)
			if err != nil {
				return nil, err
			}
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			top, m, logsum, err := topSearchTokens(logits, cfg.VocabSize, 2*cfg.Width)
			if err != nil {
				return nil, err
			}
			for _, token := range top {
				lp := (token.logit - m) - logsum
				generated := parent.GeneratedTokens + 1
				cumulative := parent.CumulativeLogprob + lp
				child := SearchSequence{GeneratedTokens: generated, CumulativeLogprob: cumulative, Score: SearchScore(generated, cumulative, cfg.LengthPenalty)}
				if eos[token.id] {
					child.TokenIDs = append([]int(nil), parent.TokenIDs...)
					child.FinishReason = "stop"
					completed = append(completed, child)
				} else {
					child.TokenIDs = make([]int, len(parent.TokenIDs)+1)
					copy(child.TokenIDs, parent.TokenIDs)
					child.TokenIDs[len(parent.TokenIDs)] = token.id
					child.FinishReason = "length"
					candidates = append(candidates, child)
				}
			}
		}
		// A completed score never changes. Keeping only its best Width rows is
		// lossless for final top-Width selection; do not apply an early-stop heuristic.
		completed = rankSearchSequences(completed, cfg.Width)
		live = rankSearchSequences(candidates, cfg.Width)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out.Sequences = rankSearchSequences(append(completed, live...), cfg.Width)
	if len(out.Sequences) == 0 {
		return nil, fmt.Errorf("%w: no candidates", ErrSearchUnsupported)
	}
	return out, nil
}
