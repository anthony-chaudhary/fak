package routing

import (
	"net/http"
	"strings"
)

// ClientRefusal is the closed vocabulary of client-facing error tokens the
// fabric names when the class of the last relayed failure would mislead the
// caller. It sits beside FailureClass, never replaces it: the class still owns
// failover and credential verdicts, and the refusal only owns what the client
// is told.
type ClientRefusal string

const (
	// RefusalFallbackMaskedPrimary: every primary rung that serves the model
	// was skipped or failed, and a fallback that does not serve it answered
	// with a model wall. The fallback's class is secondary evidence; the
	// primary's cause is the real one.
	RefusalFallbackMaskedPrimary ClientRefusal = "ROUTING_FALLBACK_MASKED_PRIMARY"
	// RefusalFreeTierClientOnly: a provider serves this free-tier model only
	// from inside its own client (opencode.ai/zen 403 FreeTierError). No retry
	// through the router can cure it, so the client must fail fast.
	RefusalFreeTierClientOnly ClientRefusal = "PROVIDER_FREE_TIER_CLIENT_ONLY"
)

var clientRefusals = []ClientRefusal{
	RefusalFallbackMaskedPrimary,
	RefusalFreeTierClientOnly,
}

// ClientRefusals returns every representable client refusal token.
func ClientRefusals() []ClientRefusal {
	out := make([]ClientRefusal, len(clientRefusals))
	copy(out, clientRefusals)
	return out
}

// IsKnown reports whether r is a member of the closed refusal vocabulary.
func (r ClientRefusal) IsKnown() bool {
	for _, known := range clientRefusals {
		if r == known {
			return true
		}
	}
	return false
}

// RungOutcome is one rung's result as the gateway observed it: a pass without
// dialing (Skipped with its SkipReason, e.g. "not_ready:leased(<holder>)") or a
// dialed attempt with its classified Failure.
type RungOutcome struct {
	Backend    string
	Skipped    bool
	SkipReason string
	Failure    Failure
}

// ClientVerdict is the fabric's answer to "what does the client see" for a walk
// that ended without a served response.
type ClientVerdict struct {
	Refusal   ClientRefusal
	Status    int
	Retryable bool
	// PrimaryBackend, PrimaryCause and Holder name the masked primary: the
	// cause without its holder suffix ("not_ready:leased") and the holder when
	// the skip reason carried one.
	PrimaryBackend string
	PrimaryCause   string
	Holder         string
	// FallbackBackend and FallbackClass are the secondary evidence.
	FallbackBackend string
	FallbackClass   FailureClass
}

// FailoverToken renders the verdict for the X-Fak-Failover header and the error
// body: the refusal token first, then the primary cause, then the fallback.
func (v ClientVerdict) FailoverToken() string {
	var b strings.Builder
	b.WriteString(string(v.Refusal))
	if v.PrimaryBackend != "" {
		b.WriteString(" primary=")
		b.WriteString(v.PrimaryBackend)
	}
	if v.PrimaryCause != "" {
		b.WriteString(" cause=")
		b.WriteString(v.PrimaryCause)
	}
	if v.Holder != "" {
		b.WriteString(" holder=")
		b.WriteString(v.Holder)
	}
	if v.FallbackBackend != "" {
		b.WriteString(" fallback=")
		b.WriteString(v.FallbackBackend)
	}
	if v.FallbackClass != "" {
		b.WriteString(" fallback_class=")
		b.WriteString(string(v.FallbackClass))
	}
	return b.String()
}

// ExhaustedWalkVerdict decides the client-visible error for a walk whose
// primaries (the rungs that serve the requested model, in walk order) were all
// skipped or failed and whose last answer came from fallback. ok is false when
// the fallback's own error is the honest answer and should be relayed as is.
//
// A masked primary outranks a free-tier refusal: when a primary could have
// served, the fallback's wall is a symptom, and the primary's cause is
// retryable once its lease or outage ends (503). A free-tier refusal with no
// masked primary is terminal for this caller (403, not retryable), so a client
// that retries 5xx fails fast instead of hanging.
//
// Pure and allocation-light: it runs on the error path of the hot path and
// reads only what the walk already recorded.
func ExhaustedWalkVerdict(primaries []RungOutcome, fallback RungOutcome) (ClientVerdict, bool) {
	if fallback.Skipped || fallback.Failure.Class != FailureModelMissing {
		return ClientVerdict{}, false
	}
	if len(primaries) > 0 && allPrimariesMissed(primaries) {
		cause, holder := primaryCause(primaries[0])
		return ClientVerdict{
			Refusal:         RefusalFallbackMaskedPrimary,
			Status:          http.StatusServiceUnavailable,
			Retryable:       true,
			PrimaryBackend:  primaries[0].Backend,
			PrimaryCause:    cause,
			Holder:          holder,
			FallbackBackend: fallback.Backend,
			FallbackClass:   fallback.Failure.Class,
		}, true
	}
	if fallback.Failure.Refusal == RefusalFreeTierClientOnly {
		return ClientVerdict{
			Refusal:         RefusalFreeTierClientOnly,
			Status:          http.StatusForbidden,
			FallbackBackend: fallback.Backend,
			FallbackClass:   fallback.Failure.Class,
		}, true
	}
	return ClientVerdict{}, false
}

func allPrimariesMissed(primaries []RungOutcome) bool {
	for _, p := range primaries {
		if !p.Skipped && p.Failure.Class == "" {
			return false
		}
	}
	return true
}

// primaryCause splits a skip reason of the form "cause(holder)" into its closed
// cause and holder; a dialed primary's cause is its failure class.
func primaryCause(p RungOutcome) (cause, holder string) {
	if !p.Skipped {
		return string(p.Failure.Class), ""
	}
	reason := p.SkipReason
	open := strings.IndexByte(reason, '(')
	if open < 0 || !strings.HasSuffix(reason, ")") {
		return reason, ""
	}
	return reason[:open], reason[open+1 : len(reason)-1]
}
