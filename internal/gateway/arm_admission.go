package gateway

// arm_admission.go — the gateway's rendering of a research-arm capacity refusal, and the
// queue telemetry that makes the wait observable (internal/researcharm/saturation.go).
//
// The arm path used to answer a saturated arm with `writeErr(w, 429, err.Error())` — the
// exception sentence as the body, no machine-readable code, and a hardcoded
// `Retry-After: 1` regardless of how long a slot actually took to free. A client could not
// branch on the failure, and the hint it was told to honor was a guess: the same free-text
// drift the closed refusal vocabulary exists to kill, and why the sweep in
// docs/benchmarks/anthony-concurrency-20261026.md saw 538 identical unactionable 429s.
//
// This file renders it the way the rest of the gateway already renders a capacity refusal
// (writeSessionRefusal's SESSION_CEILING_SATURATED arm): a typed OpenAI-style error
// envelope carrying a machine-readable `code`, the closed `reason` token, and the
// structured counters a client or autoscaler needs to act — plus a `Retry-After` derived
// from MEASURED queue wait rather than a constant.

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/researcharm"
)

// armRefusalCode is the machine-readable OpenAI-style `code` an arm capacity refusal
// carries (parallel to "session_ceiling_saturated" for the session ceiling).
const armRefusalCode = "arm_concurrency_saturated"

// writeArmAdmissionRefusal renders a refused research-arm admission. It maps the
// coordinator's typed error onto a truthful HTTP response:
//
//   - a *researcharm.SaturationError (the arm is out of capacity) → 429 with the closed
//     reason ARM_CONCURRENCY_SATURATED, the arm's live counters, a wait-derived
//     Retry-After, and the machine-readable code;
//   - any other coordinator error (a held lease, a closed coordinator) is left to the
//     caller — a lease conflict is not a capacity signal and must not be dressed as one.
//
// It reports whether it wrote a response, so a caller can fall through to its own
// rendering when the error is not a capacity refusal.
func writeArmAdmissionRefusal(w http.ResponseWriter, err error) bool {
	if w == nil || err == nil {
		return false
	}
	var sat *researcharm.SaturationError
	if !errors.As(err, &sat) {
		return false
	}

	retry := sat.RetryAfter
	if retry < time.Second {
		retry = time.Second
	}
	w.Header().Set("Retry-After", fmt.Sprintf("%d", int(retry.Round(time.Second).Seconds())))

	message := fmt.Sprintf(
		"research arm %q is at its concurrency ceiling (%d/%d in flight); this request was refused for capacity after waiting %s. Retry after the Retry-After delay, or spread the work across arms / raise the arm ceiling.",
		sat.ArmID, sat.Active, sat.Limit, sat.Waited.Round(time.Millisecond))
	switch {
	case sat.Full:
		message = fmt.Sprintf(
			"research arm %q is at its concurrency ceiling (%d/%d in flight) and its admission queue is full (bound %d, %d waiting); this request was refused without waiting. Retry after the Retry-After delay, or spread the work across arms / raise the arm ceiling.",
			sat.ArmID, sat.Active, sat.Limit, sat.QueueBound, sat.Queued)
	case sat.CtxErr != nil:
		message = fmt.Sprintf(
			"research arm %q is at its concurrency ceiling (%d/%d in flight) and this request's own deadline expired after waiting %s (%v). Retry after the Retry-After delay, or spread the work across arms / raise the arm ceiling.",
			sat.ArmID, sat.Active, sat.Limit, sat.Waited.Round(time.Millisecond), sat.CtxErr)
	}

	writeJSON(w, http.StatusTooManyRequests, map[string]any{
		"error": errObject(http.StatusTooManyRequests, armRefusalCode, message, map[string]any{
			"arm_id":     sat.ArmID,
			"active":     sat.Active,
			"limit":      sat.Limit,
			"queued":     sat.Queued,
			"waited_ms":  sat.Waited.Milliseconds(),
			"queue_full": sat.Full,
		}),
		"reason": sat.Reason(),
	})
	return true
}

// refuseArmAdmission writes the response for a refused research-arm admission: the typed
// 429 for a capacity shed (saturation.go), or — for any other coordinator refusal (a held
// exclusive lease, a closed coordinator) — the historical prose 429 this path always
// emitted. A lease conflict is a different condition from a busy slot and keeps its own
// rendering rather than being dressed as a capacity signal.
func (s *Server) refuseArmAdmission(w http.ResponseWriter, err error) {
	if w == nil || err == nil {
		return
	}
	if writeArmAdmissionRefusal(w, err) {
		return
	}
	w.Header().Set("Retry-After", "1")
	writeErr(w, http.StatusTooManyRequests, err.Error())
}

// writeArmQueueMetrics renders the research-arm admission-queue family: per arm, how many
// requests are WAITING for a slot, how saturated the arm is, and the cumulative
// queue/shed totals. Absent entirely until a coordinator is wired, the same
// quiet-until-armed shape the other optional families use.
func (s *Server) writeArmQueueMetrics(b *strings.Builder) {
	c := s.ResearchArmCoordinator()
	if c == nil || b == nil {
		return
	}
	sats := c.Saturations()
	if len(sats) == 0 {
		return
	}

	writeHelpType(b, "fak_gateway_arm_queued_requests",
		"Requests currently WAITING for a concurrency slot on a research arm's bounded admission queue. Nonzero means the arm is saturated and callers are held rather than refused.",
		"gauge")
	writeHelpType(b, "fak_gateway_arm_active_requests",
		"Requests currently in flight per research arm, against that arm's effective concurrency ceiling (fak_gateway_arm_concurrency_limit).",
		"gauge")
	writeHelpType(b, "fak_gateway_arm_concurrency_limit",
		"Effective concurrency ceiling per research arm, from fak arms limit or a lease. 0 means unlimited.",
		"gauge")
	writeHelpType(b, "fak_gateway_arm_waited_total",
		"Cumulative requests that entered a research arm's admission queue (waited for a slot) rather than being admitted immediately. May exceed admitted_after_wait + sheds_total by the callers that abandoned the wait (their own context or client deadline expired), which is neither a grant nor a capacity shed.",
		"counter")
	writeHelpType(b, "fak_gateway_arm_admitted_after_wait_total",
		"Cumulative queued requests that were GRANTED a slot after waiting. A high waited:admitted ratio with a rising shed count means the arm is persistently over capacity, not bursting.",
		"counter")
	writeHelpType(b, "fak_gateway_arm_sheds_total",
		"Cumulative research-arm requests refused for CAPACITY (bounded wait exhausted, or the queue full). Each carries reason ARM_CONCURRENCY_SATURATED. A caller whose own context or deadline expired mid-wait is refused the same typed way but is NOT counted here.",
		"counter")

	for _, sat := range sats {
		lbl := fmt.Sprintf("arm=\"%s\"", promQuote(sat.ArmID))
		fmt.Fprintf(b, "fak_gateway_arm_queued_requests{%s} %d\n", lbl, sat.Queued)
		fmt.Fprintf(b, "fak_gateway_arm_active_requests{%s} %d\n", lbl, sat.Active)
		fmt.Fprintf(b, "fak_gateway_arm_concurrency_limit{%s} %d\n", lbl, sat.Limit)
		fmt.Fprintf(b, "fak_gateway_arm_waited_total{%s} %d\n", lbl, sat.WaitedTotal)
		fmt.Fprintf(b, "fak_gateway_arm_admitted_after_wait_total{%s} %d\n", lbl, sat.AdmittedAfterWait)
		fmt.Fprintf(b, "fak_gateway_arm_sheds_total{%s} %d\n", lbl, sat.ShedsTotal)
	}
}
