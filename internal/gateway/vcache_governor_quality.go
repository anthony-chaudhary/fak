package gateway

import "github.com/anthony-chaudhary/fak/internal/vcachegov"

// vcache_governor_quality.go -- the verdict-quality metric over the live M5 governor
// journal (#1492).
//
// The governor is already wired into a live loop: every served turn flows through
// logInferenceTurn -> observeVCacheTurn -> observeVCacheGovernorDecision, which classifies
// the turn's prefix family and appends one hash-chained row (vcache_governor_journal.go).
// What that journal lacked was a way to ASK whether the verdicts were any good, and a way
// to prove the rows had not been rewritten after the fact. This file supplies both, which
// is the issue's "deterministic verdict-quality metric ... scorable and non-forgeable" bar.
//
// Non-forgeable. Each row carries prev_hash/hash over its own decision + economics fields.
// verifyVCacheGovernorChain recomputes every link with the same hashVCacheGovernorDecision
// the writer used, so editing a decision, a token count, or a seq to flatter the score
// breaks the chain at that row. Scoring is FAIL-CLOSED: an unverified chain scores 0.0 and
// reports the first bad seq rather than grading rows nobody can trust.
//
// Deterministic + scorable (the guard-RSI keep-bit pattern). Each row earns one keep-bit:
// did the posture the verdict implies match the cache activity the SAME row witnessed?
//
//	ride_natural, heartbeat_pin   -> the prefix is meant to be warm  -> keep iff read > 0
//	lazy_rebuild, evict           -> the prefix is meant to lapse    -> keep iff read == 0
//	no_cache, explicit_cache      -> never implicitly warmed (D4)    -> keep iff read == 0
//	                                                                    and creation == 0
//
// This is the warmth-belief reconciliation the per-family view already reports as
// false_warm/false_cold, reduced to one bit per verdict. It reads only fields that are
// INSIDE the hash preimage, so a row cannot be re-graded more favourably without breaking
// its own hash — that is what makes the bit non-forgeable rather than merely deterministic.
// An unrecognized decision string scores 0 (fail-closed; a new verdict must opt in here).
//
// Law A2: this metric is observational. Nothing in the request path reads it, so a bad
// score costs money and never corrupts a result.

const vcacheGovernorQualitySchema = "fak.vcache.governor-quality.v1"

// vcacheGovernorDecisionQuality is one decision class's slice of the score.
type vcacheGovernorDecisionQuality struct {
	Records int `json:"records"`
	Kept    int `json:"kept"`
}

// vcacheGovernorQualityVars is the /debug/vars `vcache_governor_quality` block: the
// non-forgeable audit of the governor journal plus its deterministic verdict score.
type vcacheGovernorQualityVars struct {
	Schema string `json:"schema"`
	// Records is how many journal rows were audited (the retained window).
	Records int `json:"records"`
	// ChainVerified is the non-forgeability witness: every prev_hash/hash link and seq
	// step recomputed. False means the ledger was rewritten or truncated mid-window.
	ChainVerified bool `json:"chain_verified"`
	// ChainBreakSeq is the seq of the first row that failed verification (0 when clean).
	ChainBreakSeq uint64 `json:"chain_break_seq,omitempty"`
	// Kept is the number of verdicts whose implied warmth posture matched the observed
	// cache activity. Score is Kept/Records, and is 0 whenever the chain does not verify.
	Kept  int     `json:"kept"`
	Score float64 `json:"score"`
	// ByDecision breaks the keep-bits down per verdict so an operator can see WHICH class
	// of decision the governor is getting wrong, not just that it is.
	ByDecision map[string]vcacheGovernorDecisionQuality `json:"by_decision,omitempty"`
	// Provenance labels the whole block a DECISION (fak's verdict over the provider's
	// OBSERVED counters), matching the vcache_families provenance contract.
	Provenance string `json:"provenance"`
}

// verifyVCacheGovernorChain recomputes the hash chain over the retained journal window.
// It returns the seq of the first row that fails and false; (0, true) when the window is
// intact or empty.
//
// The first retained row's PrevHash is taken as the anchor rather than required to be
// empty: the journal is a bounded ring (vcacheGovernorJournalRecentCap), so after
// drop-oldest the window legitimately begins mid-chain. Every link AFTER the anchor must
// still tie back to its predecessor's Hash, and every row's own Hash must recompute from
// its recorded PrevHash and payload.
func verifyVCacheGovernorChain(records []vcacheGovernorDecisionRecord) (breakSeq uint64, ok bool) {
	for i, r := range records {
		if i > 0 {
			prev := records[i-1]
			if r.PrevHash != prev.Hash || r.Seq != prev.Seq+1 {
				return r.Seq, false
			}
		}
		if r.Hash != hashVCacheGovernorDecision(r.PrevHash, r) {
			return r.Seq, false
		}
	}
	return 0, true
}

// vcacheGovernorKeepBit is the per-verdict quality bit: true when the warmth posture the
// decision implies agrees with the cache activity the row actually witnessed. It reads
// only hashed fields, so it cannot be re-derived more favourably after the fact.
func vcacheGovernorKeepBit(r vcacheGovernorDecisionRecord) bool {
	keep, _ := vcacheGovernorKeepBitOK(r)
	return keep
}

// vcacheGovernorKeepBitOK is vcacheGovernorKeepBit plus the recognized bit, so a test can
// assert that every verdict vcachegov can emit is scored on purpose rather than silently
// swept into the fail-closed default.
//
// The switch matches vcachegov's own GovernorDecision constants, never the raw strings the
// journal happens to hold today. The journal stores the decision as a string, so a literal
// switch would keep compiling if a constant's value were ever changed or a verdict renamed —
// every row would then fall to the default and the score would collapse to 0.0 with no
// build error. Binding to the constants makes that a compile-time concern, and the
// companion coverage test turns a NEW verdict into a red test rather than a silent zero.
func vcacheGovernorKeepBitOK(r vcacheGovernorDecisionRecord) (keep, recognized bool) {
	warm := r.CacheReadTokens > 0
	switch vcachegov.GovernorDecision(r.Decision) {
	case vcachegov.DecisionRideNatural, vcachegov.DecisionHeartbeatPin:
		// Meant to be held warm — a read is the vindication.
		return warm, true
	case vcachegov.DecisionLazyRebuild, vcachegov.DecisionEvict:
		// Meant to lapse — a read means the governor gave up a prefix that was still hot.
		return !warm, true
	case vcachegov.DecisionNoCache, vcachegov.DecisionExplicitCache:
		// Law D4: never implicitly warmed. Any read OR create on this family is a breach
		// of the posture, not merely a missed saving.
		return !warm && r.CacheCreationTokens == 0, true
	default:
		// Fail-closed: an unrecognized verdict is never credited.
		return false, false
	}
}

// vcacheGovernorQuality audits and scores the retained governor journal. It returns nil
// (the /debug/vars block is omitted — the same no-phantom guard the vcache blocks keep)
// until at least one verdict has been journaled.
func vcacheGovernorQuality(records []vcacheGovernorDecisionRecord) *vcacheGovernorQualityVars {
	if len(records) == 0 {
		return nil
	}
	out := &vcacheGovernorQualityVars{
		Schema:     vcacheGovernorQualitySchema,
		Records:    len(records),
		Provenance: "DECISION",
	}
	breakSeq, ok := verifyVCacheGovernorChain(records)
	out.ChainVerified = ok
	if !ok {
		// Fail-closed: refuse to grade a ledger that does not verify. Score stays 0.0 and
		// ByDecision stays empty so a tampered window cannot present a flattering slice.
		out.ChainBreakSeq = breakSeq
		return out
	}
	byDecision := make(map[string]vcacheGovernorDecisionQuality, len(vcacheGovernorDecisionOrder))
	for _, r := range records {
		slice := byDecision[r.Decision]
		slice.Records++
		if vcacheGovernorKeepBit(r) {
			slice.Kept++
			out.Kept++
		}
		byDecision[r.Decision] = slice
	}
	out.ByDecision = byDecision
	out.Score = float64(out.Kept) / float64(out.Records)
	return out
}

// vcacheGovernorQualityVarsFor renders the block from the live journal. Nil metrics or an
// empty journal omit the block.
func (m *gatewayMetrics) vcacheGovernorQualityVars() *vcacheGovernorQualityVars {
	if m == nil {
		return nil
	}
	return vcacheGovernorQuality(m.vcacheGovernorDecisionRecords())
}
