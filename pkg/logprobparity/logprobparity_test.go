package logprobparity_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/logprobparity"
	"github.com/anthony-chaudhary/fak/pkg/referenceoracle"
)

func pos(id int, topk ...int) logprobparity.Position {
	p := logprobparity.Position{ID: id}
	if topk != nil {
		p.TopK = []logprobparity.Candidate{}
		for _, c := range topk {
			p.TopK = append(p.TopK, logprobparity.Candidate{ID: c})
		}
	}
	return p
}

func TestCheckFollowsCheckLogprobsClose(t *testing.T) {
	cases := []struct {
		name          string
		actual, ref   []logprobparity.Position
		k             int
		match, at     int
		verdict       string
		actualInRefOK *bool
	}{
		{"identical", []logprobparity.Position{pos(1), pos(2)}, []logprobparity.Position{pos(1), pos(2)}, 5, 2, -1, logprobparity.Match, nil},
		{"close both ways", []logprobparity.Position{pos(1), pos(3, 3, 2)}, []logprobparity.Position{pos(1), pos(2, 2, 3)}, 5, 1, 1, logprobparity.Close, nil},
		{"actual outside ref top-k", []logprobparity.Position{pos(9, 9, 2)}, []logprobparity.Position{pos(2, 2, 3)}, 5, 0, 0, logprobparity.NotClose, nil},
		{"k truncates", []logprobparity.Position{pos(3, 3, 2)}, []logprobparity.Position{pos(2, 2, 4, 3)}, 2, 0, 0, logprobparity.NotClose, nil},
		{"one-sided top-k", []logprobparity.Position{pos(3)}, []logprobparity.Position{pos(2, 2, 3)}, 5, 0, 0, logprobparity.Close, nil},
		{"no top-k", []logprobparity.Position{pos(3)}, []logprobparity.Position{pos(2)}, 5, 0, 0, logprobparity.Unchecked, nil},
		{"later tokens ignored after divergence", []logprobparity.Position{pos(3, 3, 2), pos(7)}, []logprobparity.Position{pos(2, 2, 3), pos(8)}, 5, 0, 0, logprobparity.Close, nil},
		{"shorter actual", []logprobparity.Position{pos(1)}, []logprobparity.Position{pos(1), pos(2)}, 5, 1, -1, logprobparity.Match, nil},
		{"empty", nil, []logprobparity.Position{pos(1)}, 5, 0, -1, logprobparity.Empty, nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := logprobparity.Check(c.actual, c.ref, c.k, logprobparity.TokenIDs)
			if r.GreedyMatchTokens != c.match || r.Verdict != c.verdict {
				t.Fatalf("match=%d verdict=%s, want %d %s", r.GreedyMatchTokens, r.Verdict, c.match, c.verdict)
			}
			if (r.DivergedAt == nil) != (c.at < 0) || (r.DivergedAt != nil && *r.DivergedAt != c.at) {
				t.Fatalf("diverged_at=%v, want %d", r.DivergedAt, c.at)
			}
		})
	}
}

func readV41(t *testing.T) referenceoracle.Oracle {
	t.Helper()
	o, err := referenceoracle.ReadFile("../referenceoracle/testdata/v41-reference-oracle.json")
	if err != nil {
		t.Fatal(err)
	}
	return o
}

// The candidate text is what fak inkernel V4.1 on strix3 answered on
// 2026-10-16; the oracle continues " the city's population" at position 8.
func TestSegmentTextAgainstV41Oracle(t *testing.T) {
	refs := logprobparity.OraclePrompts(readV41(t))
	var france logprobparity.RefPrompt
	for _, r := range refs {
		if r.Text == "The capital of France is" {
			france = r
		}
	}
	if len(france.Positions) != 16 || france.InputTokens != 6 {
		t.Fatalf("france ref: %d positions, %d input tokens", len(france.Positions), france.InputTokens)
	}
	got := logprobparity.SegmentText(" Paris. In 1923, the International Advertising Convention", france.Positions)
	r := logprobparity.Check(got, france.Positions, 5, logprobparity.TextPieces)
	if r.GreedyMatchTokens != 8 || r.DivergedAt == nil || *r.DivergedAt != 8 {
		t.Fatalf("result %+v", r)
	}
	full := logprobparity.SegmentText(" Paris. In 1923, the city's population was around 2.", france.Positions)
	if r := logprobparity.Check(full, france.Positions, 5, logprobparity.TextPieces); r.Verdict != logprobparity.Match || r.GreedyMatchTokens != 16 {
		t.Fatalf("identical text: %+v", r)
	}
}

func TestSegmentTextPrefersLongestPiece(t *testing.T) {
	ref := []logprobparity.Position{{ID: 1, Piece: " in", TopK: []logprobparity.Candidate{{ID: 1, Piece: " in"}, {ID: 2, Piece: " international"}}}}
	got := logprobparity.SegmentText(" international", ref)
	if len(got) != 1 || got[0].ID != 2 {
		t.Fatalf("got %+v", got)
	}
	r := logprobparity.Check(got, ref, 5, logprobparity.TextPieces)
	if r.Verdict != logprobparity.Close || r.GreedyMatchTokens != 0 {
		t.Fatalf("result %+v", r)
	}
	unknown := logprobparity.SegmentText(" zebra", ref)
	if r := logprobparity.Check(unknown, ref, 5, logprobparity.TextPieces); r.Verdict != logprobparity.NotClose {
		t.Fatalf("unexplained token: %+v", r)
	}
}

func TestScoreOpenAIAndLlamaEndpoints(t *testing.T) {
	refs := logprobparity.OraclePrompts(readV41(t))[:2]
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		switch r.URL.Path {
		case "/v1/completions":
			text := " Paris."
			if body["prompt"] == "The chemical symbol for gold is" {
				text = " \"Au\" from"
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"text": text}}, "usage": map[string]any{"prompt_tokens": 6}})
		case "/completion":
			_ = json.NewEncoder(w).Encode(map[string]any{"content": " \"", "tokens_evaluated": 7,
				"completion_probabilities": []map[string]any{{"id": 582, "token": " \"", "logprob": -1.1,
					"top_logprobs": []map[string]any{{"id": 582, "token": " \"", "logprob": -1.1}}}}})
		default:
			http.Error(w, "nope", http.StatusNotFound)
		}
	}))
	defer srv.Close()

	prs, err := logprobparity.Score(context.Background(), logprobparity.Client{BaseURL: srv.URL + "/v1", API: logprobparity.APIOpenAI}, refs, 5)
	if err != nil {
		t.Fatal(err)
	}
	rep := logprobparity.FoldPrompts(prs)
	if rep.Verdict != logprobparity.Match || rep.GreedyMatchTokens != 4+2 {
		t.Fatalf("openai report %+v", rep)
	}
	if prs[0].PromptTokensMatch == nil || *prs[0].PromptTokensMatch {
		t.Fatalf("gold prompt is 7 tokens, endpoint reported 6: %+v", prs[0])
	}

	prs, err = logprobparity.Score(context.Background(), logprobparity.Client{BaseURL: srv.URL, API: logprobparity.APILlama}, refs[:1], 5)
	if err != nil || prs[0].Method != logprobparity.TokenIDs || prs[0].GreedyMatchTokens != 1 {
		t.Fatalf("llama: %+v %v", prs, err)
	}

	prs, err = logprobparity.Score(context.Background(), logprobparity.Client{BaseURL: srv.URL + "/missing", API: logprobparity.APIOpenAI}, refs[:1], 5)
	if !errors.Is(err, logprobparity.ErrEndpoint) || prs[0].Verdict != logprobparity.Empty {
		t.Fatalf("missing endpoint: %+v %v", prs, err)
	}
}
