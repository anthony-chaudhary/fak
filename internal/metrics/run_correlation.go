package metrics

import "errors"

// run_correlation.go — the one canonical correlation identity every measurement
// series describing a single run must COPY (#5688, a leaf of session-analytics
// epic #2822).
//
// fak folds five measurement streams over one logical run — worker, tool,
// session, annotation, summary. Each of them can name the run. The failure this
// file exists to stop is not a parse failure and produces no error anywhere:
// when one stream copies the producer-issued identity and another RECONSTRUCTS
// one from a display label or a nested identifier, the two are well-formed,
// they parse, they render — and they silently split one run into two
// unrelated-looking series. Provenance is then lost, or worse, evidence is
// attached to the wrong execution.
//
// The distinction this contract makes explicit is COPIED vs RECONSTRUCTED:
//
//   - CorrelationSource is the closed vocabulary for HOW a series obtained the
//     identity it carries. Only CorrelationSourceProducerIssued is a copy.
//   - CorrelateRunSeries folds the presented series into RunCorrelationResult,
//     the typed machine-readable receipt.
//   - RunCorrelationResult.RefusalError reports the same verdict as an error
//     wrapping ErrRunCorrelationRefused, for a caller that branches on the
//     contract. It is named apart from the Refusal FIELD deliberately: the field
//     is the typed token the receipt serializes, the method is the error a Go
//     caller checks, and Go forbids one name for both.
//   - ValidateRunCorrelationResult re-derives a receipt from the inputs it
//     preserved, so a hand-edited or drifted receipt cannot pass as a witness.
//
// EQUAL STRINGS ARE NOT AGREEMENT. Two streams can carry byte-identical run IDs
// and still have established them independently; that they agree today is then
// a coincidence of the fixture, not a property of the pipeline. This contract
// therefore prices PROVENANCE before it prices equality: a series whose source
// is a display label is refused even when its identity matches every peer.
//
// The contract also fails closed on absence rather than passing vacuously: a
// series that records no source at all cannot be shown to have copied anything,
// and a single stream cannot witness cross-stream correlation with itself.
const RunCorrelationSchema = "fak.run_correlation.v1"

// CorrelationSource names how one measurement series obtained the correlation
// identity it carries. Only CorrelationSourceProducerIssued is a copy of the
// producer's identity; the rest are independent reconstructions that can split
// one run into unrelated-looking series.
type CorrelationSource string

const (
	// CorrelationSourceUnrecorded means the series did not record where its
	// identity came from. The copied/reconstructed distinction cannot be
	// established, so the contract fails closed rather than assuming a copy.
	CorrelationSourceUnrecorded CorrelationSource = ""
	// CorrelationSourceProducerIssued means the series copied the identity the
	// run's producer issued. This is the only accepted source.
	CorrelationSourceProducerIssued CorrelationSource = "producer_issued"
	// CorrelationSourceDisplayLabel means the series derived an identity from a
	// human-facing label, which is not stable across renames or duplicates.
	CorrelationSourceDisplayLabel CorrelationSource = "display_label"
	// CorrelationSourceNestedIdentifier means the series lifted an identifier
	// out of a nested record, which identifies that record and not the run.
	CorrelationSourceNestedIdentifier CorrelationSource = "nested_identifier"
)

// CorrelationSources returns the closed source vocabulary in declaration order.
func CorrelationSources() []CorrelationSource {
	return []CorrelationSource{
		CorrelationSourceUnrecorded,
		CorrelationSourceProducerIssued,
		CorrelationSourceDisplayLabel,
		CorrelationSourceNestedIdentifier,
	}
}

// CorrelationRefusal is the closed, machine-readable vocabulary of reasons one
// canonical correlation identity could not be established across a run's series.
type CorrelationRefusal string

const (
	// CorrelationRefusalNone is the accepted case: no refusal.
	CorrelationRefusalNone CorrelationRefusal = ""
	// CorrelationRefusalNoSeries means nothing was presented to correlate.
	CorrelationRefusalNoSeries CorrelationRefusal = "no_series"
	// CorrelationRefusalUnnamedStream means a series does not say which stream
	// it came from, so a divergence could never be attributed to a producer.
	CorrelationRefusalUnnamedStream CorrelationRefusal = "unnamed_stream"
	// CorrelationRefusalMissingIdentity means a series carries no identity at all.
	CorrelationRefusalMissingIdentity CorrelationRefusal = "missing_identity"
	// CorrelationRefusalUnrecordedSource means a series carries an identity but
	// not its provenance, so it cannot be shown to have copied anything.
	CorrelationRefusalUnrecordedSource CorrelationRefusal = "unrecorded_source"
	// CorrelationRefusalFallbackIdentity means a series reconstructed its
	// identity instead of copying the producer-issued one.
	CorrelationRefusalFallbackIdentity CorrelationRefusal = "fallback_identity"
	// CorrelationRefusalDivergentIdentity means two series copied different
	// identities, so they describe two runs and not one.
	CorrelationRefusalDivergentIdentity CorrelationRefusal = "divergent_identity"
	// CorrelationRefusalSingleStream means only one stream was presented, which
	// agrees with itself and therefore witnesses no cross-stream correlation.
	CorrelationRefusalSingleStream CorrelationRefusal = "single_stream"
)

// CorrelationRefusals returns every typed refusal the contract can emit, in
// the order CorrelateRunSeries checks them.
func CorrelationRefusals() []CorrelationRefusal {
	//enumlint:exempt CorrelationRefusalNone is the accepted outcome, not a refusal.
	return []CorrelationRefusal{
		CorrelationRefusalNoSeries,
		CorrelationRefusalUnnamedStream,
		CorrelationRefusalMissingIdentity,
		CorrelationRefusalUnrecordedSource,
		CorrelationRefusalFallbackIdentity,
		CorrelationRefusalDivergentIdentity,
		CorrelationRefusalSingleStream,
	}
}

// CorrelationSeries is one measurement stream's claim on the run: which stream
// it is, which identity it carries, and how it obtained that identity. Source is
// deliberately a separate field from RunID — the identity alone cannot say
// whether it was copied or reconstructed, which is the whole distinction.
type CorrelationSeries struct {
	Stream string            `json:"stream"`
	RunID  string            `json:"run_id"`
	Source CorrelationSource `json:"source"`
}

// RunCorrelationResult is the typed receipt for one run's series set. It
// preserves the series it judged so the verdict stays auditable after the fact,
// and names the offending series by stream AND index (streams may repeat, the
// index never does).
type RunCorrelationResult struct {
	Schema         string              `json:"schema"`
	Accepted       bool                `json:"accepted"`
	Refusal        CorrelationRefusal  `json:"refusal,omitempty"`
	Reason         string              `json:"reason"`
	CanonicalRunID string              `json:"canonical_run_id,omitempty"`
	RefusedStream  string              `json:"refused_stream,omitempty"`
	RefusedIndex   int                 `json:"refused_index"`
	Streams        int                 `json:"streams"`
	Series         []CorrelationSeries `json:"series"`
}

// ErrRunCorrelationRefused is the sentinel every correlation refusal wraps, so a
// consumer branches on the contract instead of on a message.
var ErrRunCorrelationRefused = errors.New("metrics: run correlation refused the conclusion")

// RefusalError returns nil when the series set establishes one canonical
// identity, and an error wrapping ErrRunCorrelationRefused when it does not.
func (r RunCorrelationResult) RefusalError() error {
	if r.Accepted {
		return nil
	}
	return runCorrelationError{refusal: r.Refusal, reason: r.Reason}
}

type runCorrelationError struct {
	refusal CorrelationRefusal
	reason  string
}

func (e runCorrelationError) Error() string {
	return "metrics: run correlation refused the conclusion (" + string(e.refusal) + "): " + e.reason
}
func (e runCorrelationError) Unwrap() error { return ErrRunCorrelationRefused }

// CorrelateRunSeries folds a run's measurement series into the typed receipt.
//
// It accepts only when every series copied the same producer-issued identity AND
// at least two distinct streams did so. Checks run in series order and stop at
// the first defect, so the receipt names one specific offending series rather
// than a set the caller has to re-derive.
func CorrelateRunSeries(series []CorrelationSeries) RunCorrelationResult {
	result := RunCorrelationResult{
		Schema:       RunCorrelationSchema,
		RefusedIndex: -1,
		Series:       append([]CorrelationSeries(nil), series...),
	}
	distinct := make(map[string]struct{}, len(series))
	for _, s := range series {
		if s.Stream != "" {
			distinct[s.Stream] = struct{}{}
		}
	}
	result.Streams = len(distinct)

	if len(series) == 0 {
		result.Refusal = CorrelationRefusalNoSeries
		result.Reason = "no measurement series was presented: an empty set correlates nothing and must not read as one agreed run"
		return result
	}

	canonical := ""
	for i, s := range series {
		refuse := func(refusal CorrelationRefusal, reason string) RunCorrelationResult {
			result.Refusal = refusal
			result.Reason = reason
			result.RefusedStream = s.Stream
			result.RefusedIndex = i
			return result
		}
		switch {
		case s.Stream == "":
			return refuse(CorrelationRefusalUnnamedStream,
				"series "+itoa(int64(i))+" does not name its stream: a divergence could not be attributed to a producer")
		case s.RunID == "":
			return refuse(CorrelationRefusalMissingIdentity,
				"stream "+quote(s.Stream)+" carries no correlation identity")
		case s.Source == CorrelationSourceUnrecorded:
			return refuse(CorrelationRefusalUnrecordedSource,
				"stream "+quote(s.Stream)+" does not record how it obtained "+quote(s.RunID)+
					": copied and reconstructed cannot be told apart, so the contract fails closed")
		case s.Source != CorrelationSourceProducerIssued:
			return refuse(CorrelationRefusalFallbackIdentity,
				"stream "+quote(s.Stream)+" reconstructed its identity from "+string(s.Source)+
					" instead of copying the producer-issued one (equal strings are not agreement)")
		}
		if i == 0 {
			canonical = s.RunID
			continue
		}
		if s.RunID != canonical {
			return refuse(CorrelationRefusalDivergentIdentity,
				"stream "+quote(s.Stream)+" copied "+quote(s.RunID)+" but the series set already established "+
					quote(canonical)+": these describe two runs, not one")
		}
	}

	if result.Streams < 2 {
		result.Refusal = CorrelationRefusalSingleStream
		result.Reason = "only stream " + quote(series[0].Stream) +
			" was presented: one stream agrees with itself and witnesses no cross-stream correlation"
		return result
	}

	result.Accepted = true
	result.CanonicalRunID = canonical
	result.Reason = "all " + itoa(int64(len(series))) + " series across " + itoa(int64(result.Streams)) +
		" streams copied the producer-issued identity " + quote(canonical)
	return result
}

// ValidateRunCorrelationResult checks a receipt against the series it preserved,
// so a hand-edited or drifted receipt cannot stand in as a witness.
func ValidateRunCorrelationResult(result RunCorrelationResult) error {
	if result.Schema != RunCorrelationSchema {
		return errors.New("run correlation receipt carries schema " + quote(result.Schema) +
			", want " + quote(RunCorrelationSchema))
	}
	derived := CorrelateRunSeries(result.Series)
	if derived.Accepted != result.Accepted || derived.Refusal != result.Refusal ||
		derived.Reason != result.Reason || derived.CanonicalRunID != result.CanonicalRunID ||
		derived.RefusedStream != result.RefusedStream || derived.RefusedIndex != result.RefusedIndex ||
		derived.Streams != result.Streams {
		return errors.New("run correlation receipt does not match the series it preserved")
	}
	if result.Accepted && result.CanonicalRunID == "" {
		return errors.New("accepted run correlation receipt names no canonical identity")
	}
	if !result.Accepted && result.Refusal == CorrelationRefusalNone {
		return errors.New("refused run correlation receipt carries no typed refusal")
	}
	return nil
}
