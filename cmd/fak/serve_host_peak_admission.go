package main

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

var errServeHostLoadPeakUnavailable = errors.New("HOST_LOAD_PEAK_UNAVAILABLE")

// serveHostLoadPeakUnavailableError distinguishes an unqualified streamed load
// from a capacity refusal with a measured available-byte budget.
type serveHostLoadPeakUnavailableError struct {
	Reason string
	Cause  error
}

func (e *serveHostLoadPeakUnavailableError) Is(target error) bool {
	return target == errServeHostLoadPeakUnavailable
}

func (e *serveHostLoadPeakUnavailableError) Unwrap() error { return e.Cause }

func (e *serveHostLoadPeakUnavailableError) Error() string {
	detail := e.Reason
	if e.Cause != nil {
		detail += ": " + e.Cause.Error()
	}
	return fmt.Sprintf("%s: streamed-expert host peak is not qualified (%s); refusing before tensor payloads are read; device-ring fit does not bound host staging; use a qualified load route or set %s=off to load anyway", errServeHostLoadPeakUnavailable, detail, ggufload.HostLoadPeakOverrideEnv)
}

// The device fit checks cover device memory. This gate independently checks
// the loader's predicted anonymous host peak before tensor payloads are read.
func serveNativeHostLoadPeakAdmission(ggufPath string, mapped bool, opts []ggufload.Q4KLoadOption, getenv func(string) string) (gateway.StartupMessage, error) {
	if serveHostPeakAdmissionDisabled(getenv) {
		return serveStartupMessage("host-peak-admission", "warning", fmt.Sprintf(
			"native host-peak admission disabled by %s; the load is not checked against host RAM", ggufload.HostLoadPeakOverrideEnv)), nil
	}
	peak, estErr := estimateServeNativeHostLoadPeak(ggufPath, mapped, opts)
	total, free, known := compute.HostSystemMemoryInfo()
	return serveNativeHostLoadPeakDecision(peak, estErr, free, known && total > 0, getenv, opts...)
}

// The mapped reader can fall back to owned storage when mapping is unavailable.
// Admission therefore charges the owned-copy peak until load and estimate share
// one proven mapped source lifetime.
func estimateServeNativeHostLoadPeak(ggufPath string, _ bool, opts []ggufload.Q4KLoadOption) (ggufload.HostLoadPeak, error) {
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return ggufload.HostLoadPeak{}, err
	}
	defer ws.Close()
	effects := ggufload.ApplyQ4KLoadOptions(opts)
	if effects.StreamedExperts {
		// Estimate the actual option list, never a hypothetical bounded route.
		// A zero dense bound creates no model retention ledger and is unbounded.
		if !effects.StreamedDenseQ4K || !effects.StreamedDenseBounded || effects.StreamedDenseBytes <= 0 {
			return ggufload.HostLoadPeak{}, fmt.Errorf("%w: streamed-expert admission requires an active, positive bounded dense working set", ggufload.ErrQ4KLoadEstimateUnsupported)
		}
		return ws.EstimateStreamedExpertHostLoadPeak(opts...)
	}
	return ws.EstimateQ4KHostLoadPeak(false, opts...)
}

// Streamed-expert loads fail closed while their full host staging peak is
// unqualified. Ordinary resident routes retain their existing capacity policy.
func serveNativeHostLoadPeakDecision(peak ggufload.HostLoadPeak, estErr error, availBytes int64, availKnown bool, getenv func(string) string, opts ...ggufload.Q4KLoadOption) (gateway.StartupMessage, error) {
	if serveHostPeakAdmissionDisabled(getenv) {
		return serveStartupMessage("host-peak-admission", "warning", fmt.Sprintf(
			"native host-peak admission disabled by %s; the load is not checked against host RAM", ggufload.HostLoadPeakOverrideEnv)), nil
	}
	streamed := ggufload.ApplyQ4KLoadOptions(opts).StreamedExperts
	unavailable := func(reason string, cause error) (gateway.StartupMessage, error) {
		err := &serveHostLoadPeakUnavailableError{Reason: reason, Cause: cause}
		return serveStartupMessage("host-peak-admission", "error", err.Error()), fmt.Errorf("fak serve: %w", err)
	}
	if estErr != nil {
		if streamed {
			return unavailable("header-estimate-unavailable", estErr)
		}
		level := "warning"
		if errors.Is(estErr, ggufload.ErrQ4KLoadEstimateUnsupported) {
			level = "info"
		}
		return serveStartupMessage("host-peak-admission", level, fmt.Sprintf(
			"native host-peak admission skipped: no header estimate for this load route (%v)", estErr)), nil
	}
	margin := serveHostPeakMarginBytes(getenv)
	if !availKnown || availBytes < 0 {
		if streamed {
			return unavailable("host-available-memory-unknown", nil)
		}
		return serveStartupMessage("host-peak-admission", "info", fmt.Sprintf(
			"native host-peak admission skipped: host available memory is not probeable; predicted peak %s", serveHostPeakSummary(peak))), nil
	}
	if streamed && peak.PeakAnonBytes <= 0 {
		return unavailable("header-estimate-nonpositive", nil)
	}
	if err := ggufload.AdmitHostLoadPeak(peak, availBytes, true, margin); err != nil {
		return serveStartupMessage("host-peak-admission", "error", err.Error()), fmt.Errorf("fak serve: %w", err)
	}
	if streamed {
		// The streamed estimator is useful refusal evidence, not yet an upper
		// bound for every loader lifetime: it omits the F32 arena coalescing
		// copy, tied-head Q8 storage and some transform/retention paths. A
		// fitting estimate cannot authorize staging until those are qualified.
		return unavailable("streamed-staging-peak-unqualified", nil)
	}
	return serveStartupMessage("host-peak-admission", "info", fmt.Sprintf(
		"predicted host peak %s fits %.2f GiB available with a %.2f GiB margin",
		serveHostPeakSummary(peak), float64(availBytes)/(1<<30), float64(margin)/(1<<30))), nil
}

func serveHostPeakAdmissionDisabled(getenv func(string) string) bool {
	switch strings.ToLower(strings.TrimSpace(getenv(ggufload.HostLoadPeakOverrideEnv))) {
	case "0", "off", "false", "no":
		return true
	}
	return false
}

// Invalid margin overrides retain the default instead of weakening admission.
func serveHostPeakMarginBytes(getenv func(string) string) int64 {
	v := strings.TrimSpace(getenv(ggufload.HostLoadPeakMarginEnv))
	if v == "" {
		return ggufload.HostLoadPeakDefaultMarginBytes
	}
	gib, err := strconv.ParseFloat(v, 64)
	if err != nil || gib < 0 || math.IsNaN(gib) || gib > 1<<20 {
		return ggufload.HostLoadPeakDefaultMarginBytes
	}
	return int64(gib * (1 << 30))
}

func serveHostPeakSummary(p ggufload.HostLoadPeak) string {
	const gib = float64(1 << 30)
	s := fmt.Sprintf("%.2f GiB (steady %.2f + transient %.2f", float64(p.PeakAnonBytes)/gib, float64(p.SteadyAnonBytes)/gib, float64(p.TransientBytes)/gib)
	if p.MappedBytes > 0 {
		s += fmt.Sprintf("; %.2f file-backed", float64(p.MappedBytes)/gib)
	}
	return s + ")"
}
