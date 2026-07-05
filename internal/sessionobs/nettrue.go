package sessionobs

// nettrue.go -- the LINK rung's missing OUTCOME tie, made into a verdict: the
// per-session NET-TRUE ledger. The scorecard in sessionobs.go grades whether a
// session can be tied to its value/waste outcome at all (outcome_link_rate,
// value_waste_separable); this is what that link is FOR once it exists -- it lets
// the corpus answer the one question cost accounting cannot: did fak's own
// mediation HELP, WASH, or HURT this session, net of the cost the mediation added?
//
// THE QUESTION IT ANSWERS (net-true Question #2). fak inserts itself into the hot
// path of every tool call and result: adjudication, result admission, transforms,
// quarantines. Each insertion has a cost (tokens it ADDED, wall-clock ns it spent)
// and sometimes a benefit (tokens it SAVED via reuse -- vDSO / radix / compaction).
// docs/standards/net-true-value.md is the rubric: a gain is net-true only after you
// subtract the cost the change itself introduced. This ledger mechanizes that rubric
// at the session boundary -- it ties the mediation COST to the value/waste OUTCOME
// and emits one verdict:
//
//	HELPED -- mediation saved materially more than it added, on a session that did
//	          not waste: fak's overhead paid for itself.
//	HURT   -- mediation added materially more than it saved, OR the session stalled
//	          while fak spent cost on it: the mediation was a net loss for the session.
//	WASH   -- within the band: neither clearly helped nor hurt (the honest middle).
//
// PROVENANCE, NOT BLAME (net-true Question #4). Every row carries one of WITNESSED /
// OBSERVED / MODELED for its DOMINANT evidence term, so a HELPED driven by a provider
// prompt-cache hit (a value relayed from the API -> OBSERVED) is never quoted with the
// same authority as one driven by fak's own local radix reuse (a fact fak authored and
// controls -> WITNESSED), and a row with no measured mediation is MODELED, never a
// silent zero dressed as a measurement.
//
// REUSING cadencereport -- THE PATTERN, NOT THE IMPORT. internal/cadencereport already
// owns the durable-ledger discipline this rung needs: a uniform per-tick Row, a pure
// per-tick Trend vs the last row, and a JSONL history under docs/. This row mirrors that
// shape (NetTrueRow is the per-SESSION analog of cadencereport.LedgerRow) so the impure
// shell can append rows + trend them with the SAME machinery. It deliberately does NOT
// IMPORT cadencereport: internal/architest pins sessionobs at tier 1 (stdlib-only,
// imports nothing internal, off the hot path), and cadencereport is a tier-3 composer
// that shells to git/python via collect.go. Importing it would break that invariant and
// drag I/O into a pure scorer. The durable JSONL append + cross-session trend therefore
// lives in the `fak sessions` shell (where cadencereport IS reachable) -- the named
// follow-on; this file owns only the deterministic per-session verdict.
//
// It stays pure and deterministic like the rest of the package: stdlib-only, no clock,
// no RNG, same (Record, Mediation) in -> same NetTrueRow out. That determinism is what
// lets a verdict be a witness a third party can re-derive (net-true Question #5), not a
// one-run reading.

import (
	"fmt"
	"io"
)

// netTrueSchema tags the per-session ledger row so a reader can validate the line,
// the same self-describing discipline cadencereport.LedgerSchema gives its rows.
const netTrueSchema = "fak.sessionobs.nettrue.v1"

// minNetBand is the absolute floor (in tokens) on the WASH band, so a tiny session
// cannot flip HELPED/HURT on a handful of tokens. Below a real, scale-relative move
// the verdict honestly abstains to WASH -- the same empty-journal-honesty law the
// contrast and guard-verdict loops obey.
const minNetBand = 256

// netBandDivisor sets the scale-relative WASH band: a session's net token move must
// clear 1/divisor of its own token throughput (output + mediation activity) to count
// as a real HELPED/HURT rather than noise. 20 -> 5% of throughput.
const netBandDivisor = 20

// compactionCacheReadMarginal is the fraction of full input a compaction-shed token is
// worth on a WARM fire -- the B1 basis (epic #2783, #2794/#2798). On a warm fire the
// shed tokens were already provider cache-reads (billed at the read marginal), so
// dropping them saves the read marginal per turn, NOT full input; booking the shed at
// 1.0x over-values the saving ~7.7x-10x on warm traffic. This mirrors
// internal/cachevaluereport/track2.go's `providerCacheReadMultiplier = 0.1` so the
// per-session net-true ledger and the Track-2 report agree on the compaction net
// (#2804). sessionobs is tier-1 (imports nothing internal), so the constant is copied,
// not imported -- the test pins both to the same 0.1 so the mirror cannot silently drift.
const compactionCacheReadMarginal = 0.1

// Provenance labels how a reported number was obtained, the net-true vocabulary
// (docs/standards/net-true-value.md Question #4). The zero value is MODELED -- an
// unlabeled figure is a projection until a caller proves otherwise, never a silent
// WITNESSED.
type Provenance uint8

const (
	// ProvModeled is a deterministic projection or an unobserved default -- the
	// honest label for a row with no measured mediation.
	ProvModeled Provenance = iota
	// ProvWitnessed is a fact fak AUTHORED and CONTROLS: the tokens its interventions
	// added, the ns it spent mediating, the reuse its own kernel realized locally.
	ProvWitnessed
	// ProvObserved is a value RELAYED from an external party: a provider prompt-cache
	// hit reported by the API. fak did not author it, so it never carries fak's authority.
	ProvObserved
)

// String renders the provenance as its upper-case wire token; an out-of-range value
// renders "MODELED" (the safe, lowest-authority default) rather than panicking.
func (p Provenance) String() string {
	switch p {
	case ProvWitnessed:
		return "WITNESSED"
	case ProvObserved:
		return "OBSERVED"
	default:
		return "MODELED"
	}
}

// NetTrueVerdict is the per-session net-true judgment of fak's own mediation. The zero
// value is WASH -- the honest middle a session falls into until the evidence clears the
// band in one direction.
type NetTrueVerdict uint8

const (
	// NetWash: the net mediation effect is within the band -- neither clearly a help
	// nor a hurt for this session.
	NetWash NetTrueVerdict = iota
	// NetHelped: mediation saved materially more than it added, on a non-waste session.
	NetHelped
	// NetHurt: mediation added materially more than it saved, or it spent cost on a
	// session that stalled with no net saving to show for it.
	NetHurt
)

// String renders the verdict as its upper-case wire token; an out-of-range value
// renders "WASH" rather than panicking.
func (v NetTrueVerdict) String() string {
	switch v {
	case NetHelped:
		return "HELPED"
	case NetHurt:
		return "HURT"
	default:
		return "WASH"
	}
}

// Mediation is the per-session COST/BENEFIT of fak's own mediation -- the input the
// pure verdict needs that the scrubbed Record does not carry. The impure shell folds
// it from the lifecycle cost spans (epic #1147 L0/L1: EvSubmit->EvDecide adjudication
// ns; transform/quarantine token-delta added; vDSO/radix/compaction token-delta saved)
// and passes it in, keeping the verdict a pure function.
type Mediation struct {
	// TokensAdded is the tokens fak's interventions ADDED to the session (transform /
	// quarantine rewrites, injected adjudication notes). A mediation COST.
	TokensAdded int64 `json:"tokens_added"`
	// TokensSaved is the tokens fak's reuse SAVED (vDSO tool-result reuse, RadixAttention
	// prefix reuse, compaction). A mediation BENEFIT. Its provenance is SavedProvenance.
	// For compaction, this is the GROSS shed; the warm-fire discount below is applied by
	// effectiveSaved before the number reaches the net -- so callers still fold the raw
	// shed here, and the basis lives in one place.
	TokensSaved int64 `json:"tokens_saved"`
	// CompactionShedTokens is the portion of TokensSaved that came from compaction shed
	// (as opposed to vDSO/radix LOCAL reuse, which is worth its full marginal). On a warm
	// fire those shed tokens were provider cache-reads, so this portion is re-valued from
	// 1.0x down to compactionCacheReadMarginal before it counts toward the net -- the B1
	// basis (#2804). 0 (the default) means "no compaction shed in this saving", so a pure
	// local-reuse session is unchanged. Must be <= TokensSaved.
	CompactionShedTokens int64 `json:"compaction_shed_tokens,omitempty"`
	// ColdFire records that the compaction fire was OBSERVED-cold: the shed tokens were
	// NOT live provider cache-reads, so they keep the full 1.0x input basis. The default
	// (false = warm) is the honest, over-valued-if-unfixed common case -- a caller must
	// prove cold to claim full value, never the reverse.
	ColdFire bool `json:"cold_fire,omitempty"`
	// MediationNanos is the wall-clock ns fak spent mediating (adjudication + result
	// admission), the time-domain cost surfaced beside the token-domain net.
	MediationNanos int64 `json:"mediation_nanos"`
	// SavedProvenance labels TokensSaved: ProvWitnessed for reuse fak's own kernel
	// realized locally, ProvObserved for a provider prompt-cache hit relayed from the
	// API (the provider-vs-local split net-true keeps intact). Defaults to ProvModeled.
	SavedProvenance Provenance `json:"saved_provenance"`
}

// any reports whether the session carried ANY mediation cost or benefit -- a session
// with no measured mediation has nothing to judge and reads WASH/MODELED.
func (m Mediation) any() bool {
	return m.TokensAdded != 0 || m.TokensSaved != 0 || m.MediationNanos != 0
}

// effectiveSaved is TokensSaved with the warm-fire compaction-shed portion re-valued
// from full input (1.0x) down to the cache-read marginal (compactionCacheReadMarginal),
// the B1 basis (#2804). Pure local reuse (CompactionShedTokens == 0) and observed-cold
// fires (ColdFire) are returned untouched, so this only ever REMOVES phantom value --
// it can never inflate a saving. The discount is floor-rounded, so it never credits a
// fractional token the provider would not have billed.
func (m Mediation) effectiveSaved() int64 {
	shed := m.CompactionShedTokens
	if shed <= 0 || m.ColdFire {
		return m.TokensSaved // no warm compaction shed to discount
	}
	if shed > m.TokensSaved {
		shed = m.TokensSaved // clamp: the shed portion cannot exceed the saving it is part of
	}
	// The shed was booked at 1.0x; on a warm fire it is worth only the read marginal.
	// Remove the over-count = shed - floor(shed * marginal).
	marginal := int64(float64(shed) * compactionCacheReadMarginal)
	return m.TokensSaved - (shed - marginal)
}

// NetTrueRow is one session's net-true ledger row -- the per-SESSION analog of
// cadencereport.LedgerRow, a flattened, self-describing projection safe to append to a
// durable JSONL history and trend across sessions. Like a Record it carries only
// structured signal, never prose.
type NetTrueRow struct {
	Schema         string `json:"schema"`
	SessionID      string `json:"session_id"`
	Outcome        string `json:"outcome"`         // the value-vs-waste class (wire token)
	Verdict        string `json:"verdict"`         // HELPED | WASH | HURT
	Provenance     string `json:"provenance"`      // WITNESSED | OBSERVED | MODELED (dominant term)
	TokensAdded    int64  `json:"tokens_added"`    // mediation cost (tokens)
	TokensSaved    int64  `json:"tokens_saved"`    // mediation benefit (tokens)
	NetTokens      int64  `json:"net_tokens"`      // TokensSaved - TokensAdded (signed)
	MediationNanos int64  `json:"mediation_nanos"` // mediation cost (wall-clock ns)
	Band           int64  `json:"band"`            // the WASH band (tokens) the net was judged against
	Detail         string `json:"detail"`          // one-line human summary tying cost to outcome
}

// NetTrue is the whole per-session net-true engine: a pure, deterministic function from
// one scrubbed Record and its mediation cost/benefit to the verdict row. Same inputs ->
// identical row, always.
//
// THE POLICY (documented so the verdict is auditable, not a black box):
//   - band = max(minNetBand, throughput/netBandDivisor), throughput = output + added +
//     saved tokens. The net must clear the band to count as a real move.
//   - HELPED: net >= +band AND the session did not waste (you cannot claim to have
//     helped a session that stalled).
//   - HURT: net <= -band (mediation cost more than it returned, any outcome), OR the
//     session stalled (Stopped) with no net saving while fak spent token/ns cost on it.
//   - WASH: otherwise.
//
// Provenance is the DOMINANT evidence term's label, independent of the verdict: a
// saving-dominated row carries SavedProvenance, a cost-dominated row is WITNESSED (fak
// authored the tokens it added and the ns it spent), a row with no mediation is MODELED.
func NetTrue(rec Record, med Mediation) NetTrueRow {
	// saved is the net-true benefit: TokensSaved with any warm compaction shed re-valued
	// at the cache-read marginal (B1, #2804). The net and the reported saving both use it
	// so the row's NetTokens = TokensSaved - TokensAdded invariant still holds honestly.
	saved := med.effectiveSaved()
	net := saved - med.TokensAdded
	// throughput measures WORK done (the tokens that actually flowed), so it uses the
	// gross shed -- the band scales with activity, not with the discounted value.
	throughput := rec.OutputTokens + med.TokensAdded + med.TokensSaved
	band := throughput / netBandDivisor
	if band < minNetBand {
		band = minNetBand
	}

	verdict := classifyNet(rec.Outcome, net, band, med)
	row := NetTrueRow{
		Schema:         netTrueSchema,
		SessionID:      rec.SessionID,
		Outcome:        rec.Outcome.String(),
		Verdict:        verdict.String(),
		Provenance:     netProvenance(med).String(),
		TokensAdded:    med.TokensAdded,
		TokensSaved:    saved,
		NetTokens:      net,
		MediationNanos: med.MediationNanos,
		Band:           band,
	}
	row.Detail = netDetail(rec, med, verdict, net, band)
	return row
}

// classifyNet applies the documented band policy. It is the verdict's whole decision,
// kept separate so a test can pin the boundary cases without rebuilding a Row.
func classifyNet(outcome Outcome, net, band int64, med Mediation) NetTrueVerdict {
	switch {
	case net >= band && outcome != OutcomeStopped:
		return NetHelped
	case net <= -band:
		return NetHurt
	case outcome == OutcomeStopped && net <= 0 && med.any():
		// A stall whose mediation gave no net saving yet cost tokens/ns: the cost was
		// spent on a session that produced nothing -- a HURT the token band alone misses.
		return NetHurt
	default:
		return NetWash
	}
}

// netProvenance picks the label of the DOMINANT evidence term (saved vs added), so the
// verdict is never quoted with more authority than its strongest input earns.
func netProvenance(med Mediation) Provenance {
	if !med.any() {
		return ProvModeled // nothing measured -- honestly a projection, not a witness
	}
	// Dominance is judged on the EFFECTIVE (post-B1-discount) saving, so a warm compaction
	// saving that shrinks below its own cost is honestly reported cost-dominated (WITNESSED),
	// not quoted with the saved figure's authority for value it no longer has.
	if med.effectiveSaved() > med.TokensAdded {
		return med.SavedProvenance // saving-dominated: carries the saved figure's label
	}
	return ProvWitnessed // cost-dominated: fak authored the tokens it added + the ns it spent
}

// netDetail renders the one-line human summary that ties the mediation cost to the
// value/waste outcome -- the sentence an operator reads to see WHY the verdict landed.
func netDetail(rec Record, med Mediation, verdict NetTrueVerdict, net, band int64) string {
	side := "neutral"
	switch {
	case rec.Outcome.value():
		side = "value"
	case rec.Outcome == OutcomeStopped:
		side = "waste"
	}
	saved := med.effectiveSaved()
	// When the warm-fire B1 discount actually removed value, say so -- an operator reading
	// the ledger sees the gross shed was re-valued, not silently shrunk.
	basis := ""
	if saved < med.TokensSaved {
		basis = fmt.Sprintf(" [warm compaction shed %d re-valued at %.1fx: gross %d]",
			med.CompactionShedTokens, compactionCacheReadMarginal, med.TokensSaved)
	}
	return fmt.Sprintf(
		"%s: mediation saved %d, added %d tokens (net %+d vs band %d) over %dns on a %s (%s) session%s",
		verdict, saved, med.TokensAdded, net, band, med.MediationNanos, side, rec.Outcome, basis)
}

// RenderNetTrue writes the human one-liner for a row, the terminal view the shell
// prints per session (the net-true sibling of Render / RenderContrast).
func RenderNetTrue(w io.Writer, row NetTrueRow) {
	fmt.Fprintf(w, "net-true %-6s [%s]  %s  net %+d (band %d)  %dns\n",
		row.Verdict, row.Provenance, row.SessionID, row.NetTokens, row.Band, row.MediationNanos)
	fmt.Fprintf(w, "  %s\n", row.Detail)
}
