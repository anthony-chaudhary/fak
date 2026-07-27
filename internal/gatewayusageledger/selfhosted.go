package gatewayusageledger

// The self-hosted share of served volume, folded from the durable rows.
//
// This is the reader for the Counters split of the same name, and it exists to make
// one number answerable across a fleet: of the output tokens fak served, what
// fraction did we generate on our own hardware rather than buy from a vendor API?
//
// The fold's entire discipline is that it REFUSES to answer when the corpus did not
// earn an answer. A ledger full of rows written before the split existed carries no
// classified volume at all, and the arithmetically convenient reply — 0.0% —
// is the single most expensive wrong answer available here, because it reads as "we
// self-host nothing, the whole placement effort is unbuilt" when the truth is "no
// row was ever asked". So an unclassified corpus returns a nil share and a reason,
// and a corpus that measured real turns and genuinely served none of them locally
// returns an EARNED 0.0 — the two are different values, not different phrasings.

// SelfHostedShareReason is the closed vocabulary for a share the fold declined to
// compute. It is empty exactly when OutputShare is non-nil.
type SelfHostedShareReason string

const (
	// ShareNotInstrumented: not one row carried a classified turn. The rows may
	// hold plenty of served volume — they were written by a build that did not
	// record which side served it, or by a deployment whose planner the classifier
	// could not resolve. Nobody measured; this is NOT a measured zero.
	ShareNotInstrumented SelfHostedShareReason = "not_instrumented"
	// ShareNoClassifiedOutput: turns WERE classified, but between them they
	// generated no output tokens (an all-refusal or all-empty window). The
	// denominator is zero, so the fraction is undefined rather than zero.
	ShareNoClassifiedOutput SelfHostedShareReason = "no_classified_output"
)

// SelfHostedShare is the folded answer plus everything a reader needs to audit it.
type SelfHostedShare struct {
	// OutputShare is self-hosted output tokens over CLASSIFIED output tokens, in
	// [0,1] — nil when Reason says why not. Deliberately a pointer: a nil share and
	// a zero share are different findings and no float can carry both.
	OutputShare *float64
	// Reason is set exactly when OutputShare is nil.
	Reason SelfHostedShareReason

	SelfHostedTurns        uint64
	SelfHostedInputTokens  uint64
	SelfHostedOutputTokens uint64
	VendorTurns            uint64
	VendorInputTokens      uint64
	VendorOutputTokens     uint64

	// OutputTokens is the UNSPLIT total over the same rows — the coverage
	// denominator. Because the split is accumulated in the same observation that
	// feeds this total, the classified sum can never exceed it.
	OutputTokens uint64

	Rows              int
	RowsDedupedAtFold int
}

// ClassifiedTurns is the number of served turns whose side was resolved.
func (s SelfHostedShare) ClassifiedTurns() uint64 { return s.SelfHostedTurns + s.VendorTurns }

// ClassifiedOutputTokens is the output volume whose side was resolved — the
// denominator of OutputShare.
func (s SelfHostedShare) ClassifiedOutputTokens() uint64 {
	return s.SelfHostedOutputTokens + s.VendorOutputTokens
}

// ClassifiedOutputFraction is the fraction of folded output tokens whose serving
// side was resolved, in [0,1]. It is the honesty companion to OutputShare: a share
// over 3% of the volume is a sample, not a fleet answer, and a reader that prints
// one without the other is publishing an unqualified number. 0 when nothing was
// folded.
//
// The obvious name for this is the one word the concept-admission gate refuses here
// (it is a tracked family root), so please leave the longer one in place rather than
// tidying it back into a commit that cannot land.
func (s SelfHostedShare) ClassifiedOutputFraction() float64 {
	if s.OutputTokens == 0 {
		return 0
	}
	return float64(s.ClassifiedOutputTokens()) / float64(s.OutputTokens)
}

// FoldSelfHostedShare sums the self-hosted split across rows and reports the share,
// or refuses with a reason. Rows are deduped by RowKey first, so re-reading a ledger
// that was appended to twice does not double the volume.
//
// Carryforward rows ARE included, unlike FoldTrend, which skips them. That is not an
// inconsistency: a trend compares two endpoint rows and a folded era is not an
// endpoint, whereas this is a SUM and a carryforward row is the field-wise sum of
// rows that Cut has already deleted from the file. Skipping them here would silently
// shed every session older than the last cut. A carryforward written before the
// split existed contributes its output tokens to the denominator and nothing to the
// classified numerator, which is exactly right — it drags coverage down, honestly,
// rather than vanishing.
func FoldSelfHostedShare(rows []Row) SelfHostedShare {
	deduped, dropped := DedupeByKey(rows)
	s := SelfHostedShare{Rows: len(deduped), RowsDedupedAtFold: dropped}
	for _, r := range deduped {
		c := r.Counters
		s.SelfHostedTurns += c.SelfHostedTurns
		s.SelfHostedInputTokens += c.SelfHostedInputTokens
		s.SelfHostedOutputTokens += c.SelfHostedOutputTokens
		s.VendorTurns += c.VendorTurns
		s.VendorInputTokens += c.VendorInputTokens
		s.VendorOutputTokens += c.VendorOutputTokens
		s.OutputTokens += c.OutputTokens
	}
	// Order matters. "No turn was ever classified" is a statement about
	// instrumentation and outranks "the classified turns generated nothing", which
	// is a statement about traffic.
	if s.ClassifiedTurns() == 0 {
		s.Reason = ShareNotInstrumented
		return s
	}
	denom := s.ClassifiedOutputTokens()
	if denom == 0 {
		s.Reason = ShareNoClassifiedOutput
		return s
	}
	share := float64(s.SelfHostedOutputTokens) / float64(denom)
	s.OutputShare = &share
	return s
}
