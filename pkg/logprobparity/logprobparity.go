// Package logprobparity compares a candidate engine's greedy continuation with a
// reference engine's, token by token, in the shape of vLLM's
// check_logprobs_close (Apache-2.0, vllm-project/vllm@92dc3dd393
// tests/models/utils.py): the two sequences must agree token for token until
// the first divergence, and at that position each side's token must sit in the
// other side's top-k. After the first divergence the sequences condition on
// different prefixes, so nothing later is compared.
//
// It is a PUBLIC package so the probe tooling in any repository scores parity
// with one rule. The reference is either a recorded fak.reference_oracle.v1
// (pkg/referenceoracle) or a live llama-server; the candidate is either a
// llama-server (token ids and top-k: the exact two-sided check) or an
// OpenAI-compatible /v1/completions endpoint that returns text only, which is
// segmented against the reference's pieces (Method TextPieces).
package logprobparity

import "strings"

// Methods name how the candidate's token ids were obtained.
const (
	// TokenIDs: the candidate reported its own generated token ids.
	TokenIDs = "token_ids"
	// TextPieces: the candidate returned text only; its tokens are recovered
	// by matching the text against the reference's greedy and top-k pieces.
	TextPieces = "text_pieces"
)

// Verdicts, one per prompt and one for a whole report (the worst prompt).
const (
	// Match: every compared token agrees.
	Match = "MATCH"
	// Close: the sequences diverge, and every top-k membership check that
	// could run passed (vLLM's assertion holds).
	Close = "CLOSE"
	// NotClose: a token at the divergence is outside the other side's top-k.
	NotClose = "NOT_CLOSE"
	// Unchecked: the sequences diverge and neither side reported a top-k at
	// the divergence, so closeness is unknown.
	Unchecked = "UNCHECKED"
	// Empty: nothing was compared (an empty side).
	Empty = "EMPTY"
)

// UnknownID marks a candidate token the text segmentation could not identify.
const UnknownID = -1

// Candidate is one top-k entry.
type Candidate struct {
	ID      int     `json:"id"`
	Piece   string  `json:"piece,omitempty"`
	Logprob float64 `json:"logprob"`
}

// Position is one generated token and, when the engine reported it, the top-k
// at that step, most likely first. A nil TopK means "not reported".
type Position struct {
	ID    int         `json:"id"`
	Piece string      `json:"piece,omitempty"`
	TopK  []Candidate `json:"top_k,omitempty"`
}

// Result is the parity of one prompt.
type Result struct {
	Prompt string `json:"prompt,omitempty"`
	Method string `json:"method"`
	// Compared is how many positions both sides produced.
	Compared int `json:"compared_tokens"`
	// GreedyMatchTokens is the length of the agreeing prefix.
	GreedyMatchTokens int `json:"greedy_match_tokens"`
	// DivergedAt, ActualID and ReferenceID describe the first divergence.
	DivergedAt  *int `json:"diverged_at,omitempty"`
	ActualID    *int `json:"actual_id,omitempty"`
	ReferenceID *int `json:"reference_id,omitempty"`
	// ActualInRefTopK / RefInActualTopK are the two vLLM assertions; nil when
	// that side reported no top-k.
	ActualInRefTopK *bool  `json:"actual_in_reference_top_k,omitempty"`
	RefInActualTopK *bool  `json:"reference_in_actual_top_k,omitempty"`
	Verdict         string `json:"verdict"`
}

// Check scores actual against ref with a top-k of k (k <= 0 uses every
// reported candidate).
func Check(actual, ref []Position, k int, method string) Result {
	r := Result{Method: method}
	n := min(len(actual), len(ref))
	r.Compared = n
	if n == 0 {
		r.Verdict = Empty
		return r
	}
	for i := 0; i < n; i++ {
		a, b := actual[i], ref[i]
		if a.ID == b.ID && a.ID != UnknownID {
			r.GreedyMatchTokens++
			continue
		}
		at, aid, bid := i, a.ID, b.ID
		r.DivergedAt, r.ActualID, r.ReferenceID = &at, &aid, &bid
		r.ActualInRefTopK = inTopK(a.ID, b.TopK, k)
		r.RefInActualTopK = inTopK(b.ID, a.TopK, k)
		r.Verdict = closeness(r.ActualInRefTopK, r.RefInActualTopK)
		return r
	}
	r.Verdict = Match
	return r
}

func inTopK(id int, topk []Candidate, k int) *bool {
	if topk == nil {
		return nil
	}
	if k > 0 && len(topk) > k {
		topk = topk[:k]
	}
	found := false
	for _, c := range topk {
		if c.ID == id && id != UnknownID {
			found = true
			break
		}
	}
	return &found
}

func closeness(checks ...*bool) string {
	ran := false
	for _, c := range checks {
		if c == nil {
			continue
		}
		ran = true
		if !*c {
			return NotClose
		}
	}
	if !ran {
		return Unchecked
	}
	return Close
}

// SegmentText recovers a text-only candidate's tokens against ref: at each
// position the candidate token is the longest of the reference's greedy piece
// and top-k pieces that prefixes the remaining text. Segmentation stops at the
// first position whose token is not the reference's greedy token (the
// sequences diverge there) or when the text runs out. A divergent token no
// reference piece explains is UnknownID, which never counts as in the top-k.
//
// Without the candidate's tokenizer this is an approximation: a reference
// piece that is a strict prefix of the candidate's real token (" " before
// " Paris" when " Paris" is outside the reference top-k) counts as one more
// agreeing token. Prefer a candidate API that reports token ids.
func SegmentText(text string, ref []Position) []Position {
	var out []Position
	rest := text
	for _, p := range ref {
		if rest == "" {
			break
		}
		best, bestPiece := UnknownID, ""
		consider := func(id int, piece string) {
			if piece != "" && strings.HasPrefix(rest, piece) && len(piece) > len(bestPiece) {
				best, bestPiece = id, piece
			}
		}
		consider(p.ID, p.Piece)
		for _, c := range p.TopK {
			consider(c.ID, c.Piece)
		}
		if best == UnknownID {
			out = append(out, Position{ID: UnknownID})
			break
		}
		out = append(out, Position{ID: best, Piece: bestPiece})
		if best != p.ID {
			break
		}
		rest = rest[len(bestPiece):]
	}
	return out
}

// Report is the parity of a prompt battery.
type Report struct {
	Schema    string   `json:"schema"`
	Reference string   `json:"reference"`
	Actual    string   `json:"actual"`
	TopK      int      `json:"top_k"`
	Results   []Result `json:"results,omitempty"`
	// GreedyMatchTokens and Compared sum over prompts.
	GreedyMatchTokens int    `json:"greedy_match_tokens"`
	Compared          int    `json:"compared_tokens"`
	Verdict           string `json:"verdict"`
}

// ReportSchema is the schema id of a Report.
const ReportSchema = "fak.logprob_parity.v1"

var verdictRank = map[string]int{Match: 0, Close: 1, Unchecked: 2, Empty: 3, NotClose: 4}

// Fold sums results into a report whose verdict is the worst prompt's.
func Fold(results []Result) Report {
	rep := Report{Schema: ReportSchema, Results: results, Verdict: Empty}
	for i, r := range results {
		rep.GreedyMatchTokens += r.GreedyMatchTokens
		rep.Compared += r.Compared
		if i == 0 || verdictRank[r.Verdict] > verdictRank[rep.Verdict] {
			rep.Verdict = r.Verdict
		}
	}
	return rep
}
