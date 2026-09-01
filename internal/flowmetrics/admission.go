package flowmetrics

import (
	"fmt"
	"strings"
	"time"
)

// OverloadReasonCode is stable for callers that need to route an admission refusal.
const OverloadReasonCode = "FLOW_ARRIVAL_EXCEEDS_SERVICE"

// AdmissionIntent classifies why issue or worker work is being admitted.
type AdmissionIntent string

const (
	IntentFresh        AdmissionIntent = "fresh"
	IntentRecovery     AdmissionIntent = "recovery"
	IntentLanding      AdmissionIntent = "landing"
	IntentSafety       AdmissionIntent = "safety"
	IntentContinuation AdmissionIntent = "continuation"
)

// ParseAdmissionIntent validates the closed set accepted by the native CLI.
func ParseAdmissionIntent(raw string) (AdmissionIntent, error) {
	intent := AdmissionIntent(strings.ToLower(strings.TrimSpace(raw)))
	switch intent {
	case IntentFresh, IntentRecovery, IntentLanding, IntentSafety, IntentContinuation:
		return intent, nil
	default:
		return "", fmt.Errorf("unknown admission intent %q (want fresh, recovery, landing, safety, or continuation)", raw)
	}
}

// ArrivalServiceWindow is the existing arrival-versus-service measurement in typed form.
type ArrivalServiceWindow struct {
	Opened      int       `json:"arrivals"`
	Closed      int       `json:"service"`
	ArrivalRate float64   `json:"arrival_rate_per_day"`
	ServiceRate float64   `json:"service_rate_per_day"`
	Ratio       *float64  `json:"ratio,omitempty"`
	WindowDays  float64   `json:"window_days"`
	WindowStart time.Time `json:"window_start"`
	WindowEnd   time.Time `json:"window_end"`
}

// AdmissionReceipt is the reversible overload decision returned to issue/worker admission.
type AdmissionReceipt struct {
	Schema     string               `json:"schema"`
	Verdict    string               `json:"verdict"`
	ReasonCode string               `json:"reason_code,omitempty"`
	Reason     string               `json:"reason"`
	Intent     AdmissionIntent      `json:"intent"`
	Threshold  float64              `json:"threshold"`
	Observed   ArrivalServiceWindow `json:"observed"`
}

// MeasureArrivalService folds the same issue spans used by the arrival_vs_service KPI.
func MeasureArrivalService(spans []Span, since, now time.Time) ArrivalServiceWindow {
	days := now.Sub(since).Hours() / 24
	if days <= 0 {
		days = 1
	}
	m := ArrivalServiceWindow{WindowDays: days, WindowStart: since, WindowEnd: now}
	for _, s := range spans {
		if !s.OpenedAt.Before(since) {
			m.Opened++
		}
		if s.Closed() && !s.ClosedAt.Before(since) {
			m.Closed++
		}
	}
	m.ArrivalRate = float64(m.Opened) / days
	m.ServiceRate = float64(m.Closed) / days
	if m.Closed > 0 {
		ratio := float64(m.Opened) / float64(m.Closed)
		m.Ratio = &ratio
	}
	return m
}

// AdmitWIP refuses only fresh discretionary work when measured intake is overloaded.
// Recovery, landing, safety, and already-owned continuations remain admitted so the
// queue can recover. A ratio exactly at the threshold is admitted.
func AdmitWIP(intent AdmissionIntent, observed ArrivalServiceWindow, threshold float64) AdmissionReceipt {
	out := AdmissionReceipt{
		Schema:    "fak-flow-admission/1",
		Verdict:   "ADMIT",
		Reason:    "arrival is within the service envelope",
		Intent:    intent,
		Threshold: threshold,
		Observed:  observed,
	}
	if intent != IntentFresh {
		out.Reason = "recovery, landing, safety, and already-owned continuations are exempt from fresh-WIP overload refusal"
		return out
	}
	overloaded := observed.Opened > 0 && (observed.Closed == 0 || (observed.Ratio != nil && *observed.Ratio > threshold))
	if overloaded {
		out.Verdict = "REFUSE"
		out.ReasonCode = OverloadReasonCode
		out.Reason = "fresh discretionary WIP is refused while measured arrival exceeds landed service"
	}
	return out
}
