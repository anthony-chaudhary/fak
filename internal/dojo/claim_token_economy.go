package dojo

import "fmt"

// claim_token_economy.go is the token-economy/tokens_saved_ratio KPI cell (#4487):
// the ONE overall tokens-saved ratio — the input-side tokens fak's token-economy
// levers (compaction shed, cache warmth, resume posture) save versus a NO-LEVER
// baseline, folded from a PAIRED on/off measurement. It is the single headline an
// operator cites for "what does fak save", calibrated in the gym so the number
// can't drift from billed reality or double-count per-lever savings.
//
// The claim registers through the additive RegisterClaim seam, so this cell never
// edits — and never conflicts on — the central Registry literal.
//
// Layering (why the fold lives here but the lever does not): internal/dojo is the
// gym's pure core (architest pins it tier 1 — the corpus/ledger-scanning levers
// live in the cmd/fak shell). So the SCORING is a pure, total fold over the
// paired ON/OFF token counts the shell reduces from the billed corpus; this file
// reads no corpus and does no I/O, which keeps the fold unit-testable without a
// corpus on disk and keeps the tier legal.

// The one anchored literal for this cell — the single target the RSI loop's
// RECALIBRATE arm rewrites, carried in the cell's own file rather than the shared
// map. It is an ESTIMATE (not a floor): a best-guess central tendency the loop may
// re-point at the measured ratio, never a guarantee reality must not breach.
var _ = RegisterClaim("token-economy", "tokens_saved_ratio", claim(0.30,
	"seed theory (#4487): fak's token-economy levers (compaction shed, cache warmth, resume posture) TOGETHER save about 30% of the input-side tokens a no-lever baseline would bill — the ONE overall headline saving an operator cites for 'what does fak save'. It is folded from a PAIRED on/off measurement (fak-ON billed tokens vs a no-lever OFF baseline over the same workload), NOT summed per lever: per-lever savings overlap and summing them double-counts. A genuine estimate the RSI loop recalibrates toward the measured ratio. Until a paired OFF baseline exists — the no-lever counterfactual is not billed in an ordinary session corpus, the same gap compaction has (#953) — the cell scores UNMEASURED, naming the missing paired baseline: it cannot tell 'saved nothing' from 'no baseline to measure against', so it never fabricates a ratio"))

// PairedTokenCorpus is the reduced view of a paired on/off token measurement the
// fold needs — the two token totals plus the honesty bit, nothing about the
// corpus's on-disk shape. The cmd/fak shell adapts the billed corpus into this;
// the dojo core never learns the corpus's schema. "Input-side tokens" means the
// tokens fak's levers actually move: input + cache_read + cache_creation (output
// is the model's generation, not something the token economy sheds).
type PairedTokenCorpus struct {
	// OffBaselineTokens is the input-side tokens a NO-LEVER baseline billed over the
	// workload — the counterfactual denominator. Meaningful only when BaselinePaired
	// is true.
	OffBaselineTokens uint64
	// OnTokens is the input-side tokens fak billed with its token-economy levers ON
	// over the same workload — the ON side of the saving. Carried as the sample even
	// on the UNMEASURED path so the report shows what fak CAN see (the ON side) while
	// naming what it lacks (the OFF baseline).
	OnTokens uint64
	// BaselinePaired reports whether a paired NO-LEVER OFF baseline exists at all. It
	// is the honesty bit that separates a genuine "saved nothing" from "there is no
	// baseline to measure against": false forces UNMEASURED so a corpus that carries
	// only the ON side (every ordinary session corpus — the OFF counterfactual is
	// never billed) can never be read as a fabricated saving ratio (#4487; the same
	// missing-counterfactual gap as compaction #953).
	BaselinePaired bool
}

// TokenEconomyEpisodes folds the paired token corpus into the dojo's (prediction,
// outcome) pair for token-economy/tokens_saved_ratio — ONE episode scored against
// the registered claim. It is pure and total so the fold is unit-testable without
// a corpus on disk.
//
// Honesty rules (#4487's confusion risks): the ratio is folded from the PAIRED
// measurement (off - on)/off, NEVER summed from per-lever savings (which overlap
// and double-count). A corpus with no paired OFF baseline — BaselinePaired false,
// or a zero OFF denominator — scores UNMEASURED naming the missing baseline rather
// than a fabricated number. A negative saving (levers billed MORE than the no-lever
// baseline) is NOT clamped to zero: it surfaces honestly as an over-claim so the
// gym cannot hide fak's levers costing more than they save; calibErr caps the
// downstream magnitude so one wild window cannot dominate the fold.
func TokenEconomyEpisodes(c PairedTokenCorpus) []ScoredInput {
	pred := Registry.MustPredict("token-economy", "tokens_saved_ratio", "fraction")
	if !c.BaselinePaired || c.OffBaselineTokens == 0 {
		return []ScoredInput{{
			Prediction: pred,
			Outcome: Outcome{
				Measured: false,
				Sample:   int(c.OnTokens),
				Source: fmt.Sprintf("%d input-side token(s) billed with fak's levers ON, but no paired NO-LEVER OFF baseline to fold a saving against (the no-lever counterfactual is not billed in an ordinary session corpus; the same gap as compaction #953) — tokens_saved_ratio is UNMEASURED, not a fabricated number",
					c.OnTokens),
			},
		}}
	}
	off := int64(c.OffBaselineTokens)
	on := int64(c.OnTokens)
	ratio := float64(off-on) / float64(off)
	return []ScoredInput{{
		Prediction: pred,
		Outcome: Outcome{
			Realized:   ratio,
			Provenance: Witnessed,
			Measured:   true,
			Sample:     int(c.OffBaselineTokens),
			Source: fmt.Sprintf("(off %d - on %d) / off %d input-side billed tokens over the paired no-lever/fak-on corpus — the overall tokens-saved ratio folded from a PAIRED measurement, never summed per lever (WITNESSED)",
				c.OffBaselineTokens, c.OnTokens, c.OffBaselineTokens),
		},
	}}
}
