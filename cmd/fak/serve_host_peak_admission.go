package main

import (
	"errors"
	"fmt"
	"math"
	"runtime/debug"
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
	msg, _, _, err := serveNativeHostLoadPeakAdmissionPeak(ggufPath, mapped, opts, getenv)
	return msg, err
}

// serveNativeHostLoadPeakAdmissionPeak also returns the admitted resident-route peak; peakOK is
// false when admission is disabled, the estimate is missing, or the route is streamed (which
// installs its own predicted-peak memory limit).
func serveNativeHostLoadPeakAdmissionPeak(ggufPath string, mapped bool, opts []ggufload.Q4KLoadOption, getenv func(string) string) (gateway.StartupMessage, ggufload.HostLoadPeak, bool, error) {
	if serveHostPeakAdmissionDisabled(getenv) {
		return serveStartupMessage("host-peak-admission", "warning", fmt.Sprintf(
			"native host-peak admission disabled by %s; the load is not checked against host RAM", ggufload.HostLoadPeakOverrideEnv)), ggufload.HostLoadPeak{}, false, nil
	}
	peak, estErr := estimateServeNativeHostLoadPeak(ggufPath, mapped, opts)
	total, free, known := compute.HostSystemMemoryInfo()
	msg, err := serveNativeHostLoadPeakDecision(peak, estErr, free, known && total > 0, getenv, opts...)
	peakOK := err == nil && estErr == nil && peak.PeakAnonBytes > 0 && !ggufload.ApplyQ4KLoadOptions(opts).StreamedExperts
	return msg, peak, peakOK, err
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

// Streamed-expert loads fail closed on any estimate, probe or memory-limit gap, and
// are admitted only under a Go memory limit equal to the predicted peak. Ordinary
// resident routes retain their existing capacity policy.
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
		// The streamed estimator charges the arena coalescing copy, tied-head Q8, KV-b buffering
		// and canonical copies (fak#13798), and a heap-sampled real-loader witness holds the load
		// under it. The garbage it does not retain (dead dequant buffers, per-upload staging,
		// per-fault expert strides) is bounded only by a Go memory limit. The limit is the
		// predicted peak itself, not peak + margin, so GC slack cannot carry RssAnon past the
		// prediction; the margin stays headroom for non-Go (driver) allocations.
		limit, ok := serveCapHostPeakMemoryLimit(peak)
		if !ok {
			return unavailable("streamed-staging-peak-unqualified", fmt.Errorf("could not install a Go memory limit of the predicted peak"))
		}
		return serveStartupMessage("host-peak-admission", "info", fmt.Sprintf(
			"predicted streamed-expert host peak %s fits %.2f GiB available with a %.2f GiB margin; Go memory limit %.2f GiB bounds unretained staging",
			serveHostPeakSummary(peak), float64(availBytes)/(1<<30), float64(margin)/(1<<30), float64(limit)/(1<<30))), nil
	}
	return serveStartupMessage("host-peak-admission", "info", fmt.Sprintf(
		"predicted host peak %s fits %.2f GiB available with a %.2f GiB margin",
		serveHostPeakSummary(peak), float64(availBytes)/(1<<30), float64(margin)/(1<<30))), nil
}

// serveSetMemoryLimit is debug.SetMemoryLimit, injectable so admission tests never lower the
// test process's own limit.
var serveSetMemoryLimit = debug.SetMemoryLimit

// serveCapHostPeakMemoryLimit lowers the Go memory limit to the predicted peak (never raising an
// operator's lower GOMEMLIMIT) and reports the limit now in force.
func serveCapHostPeakMemoryLimit(peak ggufload.HostLoadPeak) (int64, bool) {
	if peak.PeakAnonBytes <= 0 {
		return 0, false
	}
	want := peak.PeakAnonBytes
	if cur := serveSetMemoryLimit(-1); cur > want {
		serveSetMemoryLimit(want)
	}
	cur := serveSetMemoryLimit(-1)
	return cur, cur <= want
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
