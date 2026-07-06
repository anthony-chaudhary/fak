package toon

import (
	"encoding/json"
	"fmt"
	"strings"
)

// AccuracyNotYet is the honest placeholder for the WITNESSED accuracy half of issue #3068.
// The scorecard's FIRST deliverable is the OBSERVED, deterministic token-delta measurement
// (this file) — fully shippable with no model call. The SECOND deliverable, a paired
// on-JSON / on-TOON model eval graded by a held answer key (accuracy-per-1K-tokens), needs
// live model access/cost and is NOT run here. Every FamilyResult carries this string in its
// AccuracyNote so the report can never silently imply an accuracy witness that was not run —
// the repo rule: report incomplete work as `not yet` with the missing witness, never claim a
// witness you didn't produce. The follow-on methodology is documented in the scorecard README.
const AccuracyNotYet = "not yet — requires live paired model eval (follow-on)"

// Family is one named payload family the scorecard measures: a human label plus the decoded
// JSON value (encoding/json-native shapes — map[string]any, []any, string/float64/bool/nil)
// to encode both ways. The CALLER supplies real fak payloads (index rows, nightrun telemetry,
// a nested config); Scorecard fabricates no data. Keeping Scorecard a pure function of its
// input families is what lets the test feed it REAL data (rows read off a docs/nightrun/*.jsonl
// file in the tree) while the measurement itself stays deterministic.
type Family struct {
	Name    string
	Payload any
}

// FamilyResult is the OBSERVED token-delta measurement for one family plus the gate's
// fire/skip verdict. Every number is measured off the deterministic encoders (compact
// json.Marshal vs Encode) and the same bytes/4 yardstick memview uses (memview.tokenEstimate,
// and the codec's own Decide fallback) — no model call, no modeled dollars, so the numbers
// compose with memview's SweepFormats on one consistent axis. Accuracy is deliberately NOT a
// number here: AccuracyNote holds AccuracyNotYet (see that const).
type FamilyResult struct {
	Name         string     `json:"name"`
	JSONTokens   int        `json:"json_tokens"` // tokens of the compact canonical JSON encoding
	TOONTokens   int        `json:"toon_tokens"` // tokens of the TOON encoding (0 if Encode failed)
	DeltaPct     float64    `json:"delta_pct"`   // (TOON-JSON)/JSON*100; negative = TOON wins (fewer tokens)
	Eligibility  float64    `json:"eligibility"` // TabularEligibility(payload), the shape signal the gate reads
	Verdict      string     `json:"verdict"`     // "FIRE" or "SKIP(<reason>)"
	Fire         bool       `json:"fire"`        // Decide(...).Encode — the governed auto-fire decision
	SkipReason   SkipReason `json:"skip_reason,omitempty"`
	EncodeErr    string     `json:"encode_err,omitempty"` // non-empty iff Encode returned an error
	AccuracyNote string     `json:"accuracy_note"`        // always AccuracyNotYet — the accuracy half is not run here
}

// Won reports whether TOON produced STRICTLY fewer tokens than JSON for this family — the
// one true "win" definition. It is deliberately conservative: an encode failure or a tie is
// not a win, so the both-directions honesty gate cannot be fooled by a -100% delta from a
// TOONTokens==0 encode error masquerading as a saving.
func (f FamilyResult) Won() bool {
	return f.EncodeErr == "" && f.TOONTokens > 0 && f.TOONTokens < f.JSONTokens
}

// Report is the full scorecard: one FamilyResult per family plus the both-directions summary
// the honesty gate asserts. Wins counts families where TOON strictly wins on tokens; Losses
// counts every other family (tie, loss, or encode failure). A report with Losses==0 is the
// "silent truncation" anti-pattern issue #3068 rejects — a scorecard that only shows wins.
type Report struct {
	Families []FamilyResult `json:"families"`
	Wins     int            `json:"wins"`
	Losses   int            `json:"losses"`
}

// estimateTokens is the coarse bytes/4 token yardstick — byte-identical to
// memview.tokenEstimate (internal/memview/format.go) and to the codec's own Decide tokenizer
// fallback (decide.go) — so a scorecard number and a SweepFormats number are on one scale and
// can be compared directly. It is an OBSERVED estimate, not the model's true tokenizer; the
// scorecard reports it as such. A live caller can supply the model's real tokenizer to Decide.
func estimateTokens(n int) int {
	if n <= 0 {
		return 0
	}
	return (n + 3) / 4
}

// Scorecard measures the OBSERVED token delta between compact JSON and TOON for each family
// and attaches the governed fire/skip verdict from Decide. It is PURE and deterministic: no
// I/O, no model, no randomness (json.Marshal sorts map keys; Encode walks keys sorted), so a
// fixed set of families always yields the same Report. The verdict comes from the real gate
// (Decide) fed the SAME bytes/4 tokenizer the scorecard counts with, so a family's FIRE/SKIP
// line is consistent with its reported token counts.
func Scorecard(families []Family) Report {
	rep := Report{Families: make([]FamilyResult, 0, len(families))}
	tokenizer := func(b []byte) int { return estimateTokens(len(b)) }
	for _, f := range families {
		fr := FamilyResult{
			Name:         f.Name,
			Eligibility:  TabularEligibility(f.Payload),
			AccuracyNote: AccuracyNotYet,
		}
		if jsonBytes, err := json.Marshal(f.Payload); err == nil {
			fr.JSONTokens = estimateTokens(len(jsonBytes))
		}
		if enc, err := Encode(f.Payload, Options{}); err != nil {
			fr.EncodeErr = err.Error()
		} else {
			fr.TOONTokens = estimateTokens(len(enc))
		}
		if fr.JSONTokens > 0 {
			fr.DeltaPct = float64(fr.TOONTokens-fr.JSONTokens) / float64(fr.JSONTokens) * 100
		}
		dec := Decide(f.Payload, DecideInput{Tokenizer: tokenizer})
		fr.Fire = dec.Encode
		if dec.Encode {
			fr.Verdict = "FIRE"
		} else {
			fr.Verdict = "SKIP(" + string(dec.Reason) + ")"
			fr.SkipReason = dec.Reason
		}
		if fr.Won() {
			rep.Wins++
		} else {
			rep.Losses++
		}
		rep.Families = append(rep.Families, fr)
	}
	return rep
}

// String renders the report as a fixed-width table plus the both-directions summary — the
// text a run pastes into docs/toon-scorecard/README.md. It is `-v`-friendly: a test can
// t.Logf(rep.String()) and the whole scorecard shows up under `go test -run Scorecard -v`.
func (r Report) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-28s %8s %8s %9s %6s  %s\n", "family", "json_tok", "toon_tok", "delta%", "elig", "verdict")
	fmt.Fprintf(&b, "%s\n", strings.Repeat("-", 84))
	for _, f := range r.Families {
		verdict := f.Verdict
		if f.EncodeErr != "" {
			verdict = "ENCODE_ERR: " + f.EncodeErr
		}
		fmt.Fprintf(&b, "%-28s %8d %8d %+8.1f%% %6.2f  %s\n",
			f.Name, f.JSONTokens, f.TOONTokens, f.DeltaPct, f.Eligibility, verdict)
	}
	fmt.Fprintf(&b, "%s\n", strings.Repeat("-", 84))
	fmt.Fprintf(&b, "both-directions: %d win(s) (TOON fewer tokens), %d loss/tie(s)\n", r.Wins, r.Losses)
	fmt.Fprintf(&b, "accuracy delta:  %s\n", AccuracyNotYet)
	return b.String()
}
