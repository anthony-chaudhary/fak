package main

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func hostPeakTestEnv(values map[string]string) func(string) string {
	return func(key string) string { return values[key] }
}

func requireBoundedHostPeakDiagnostic(t *testing.T, msgLevel, msgKind, msgText string) {
	t.Helper()
	if msgKind != "host-peak-admission" {
		t.Fatalf("diagnostic kind = %q", msgKind)
	}
	if msgLevel != "info" && msgLevel != "warning" && msgLevel != "error" {
		t.Fatalf("diagnostic level = %q", msgLevel)
	}
	if len(msgText) == 0 || len(msgText) > 2048 {
		t.Fatalf("diagnostic length = %d, want 1..2048", len(msgText))
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestServeNativeHostLoadPeakDecisionClosesCapacityBoundary(t *testing.T) {
	const gib = int64(1 << 30)
	peak := ggufload.HostLoadPeak{
		SteadyAnonBytes: 18 * gib,
		TransientBytes:  6 * gib,
		PeakAnonBytes:   24 * gib,
		Rows: []ggufload.HostLoadTypeRow{
			{Type: "Q2_K", F32Bytes: 6 * gib},
			{Type: "IQ3_XXS", PackedBytes: gib, Q8Bytes: 3 * gib},
		},
	}
	empty := hostPeakTestEnv(nil)

	msg, err := serveNativeHostLoadPeakDecision(peak, nil, 26*gib, true, empty)
	if err != nil {
		t.Fatalf("exact peak+default-margin boundary refused: %v", err)
	}
	requireBoundedHostPeakDiagnostic(t, msg.Level, msg.Kind, msg.Text)

	msg, err = serveNativeHostLoadPeakDecision(peak, nil, 26*gib-1, true, empty)
	if !errors.Is(err, ggufload.ErrHostLoadPeakTooBig) {
		t.Fatalf("over-budget error = %v, want ErrHostLoadPeakTooBig", err)
	}
	requireBoundedHostPeakDiagnostic(t, msg.Level, msg.Kind, msg.Text)
	var typed *ggufload.HostLoadPeakError
	if !errors.As(err, &typed) || typed.NeedBytes != peak.PeakAnonBytes || typed.AvailBytes != 26*gib-1 || typed.MarginBytes != ggufload.HostLoadPeakDefaultMarginBytes {
		t.Fatalf("typed refusal = %+v", typed)
	}

	halfGiB := hostPeakTestEnv(map[string]string{ggufload.HostLoadPeakMarginEnv: "0.5"})
	if _, err := serveNativeHostLoadPeakDecision(peak, nil, 24*gib+gib/2, true, halfGiB); err != nil {
		t.Fatalf("exact configured-margin boundary refused: %v", err)
	}
	for _, invalid := range []string{"-3", "not-a-number", "NaN"} {
		env := hostPeakTestEnv(map[string]string{ggufload.HostLoadPeakMarginEnv: invalid})
		if _, err := serveNativeHostLoadPeakDecision(peak, nil, 25*gib, true, env); !errors.Is(err, ggufload.ErrHostLoadPeakTooBig) {
			t.Fatalf("invalid margin %q error = %v, want default-margin refusal", invalid, err)
		}
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestServeNativeHostLoadPeakDecisionObservesFailOpenCases(t *testing.T) {
	const gib = int64(1 << 30)
	peak := ggufload.HostLoadPeak{PeakAnonBytes: 24 * gib}
	empty := hostPeakTestEnv(nil)

	for name, tc := range map[string]struct {
		peak     ggufload.HostLoadPeak
		estimate error
		avail    int64
		known    bool
		level    string
	}{
		"unknown memory":       {peak: peak, avail: 0, known: false, level: "info"},
		"unsupported estimate": {estimate: ggufload.ErrQ4KLoadEstimateUnsupported, avail: gib, known: true, level: "info"},
		"failed estimate":      {estimate: errors.New("fixture estimate failure"), avail: gib, known: true, level: "warning"},
	} {
		t.Run(name, func(t *testing.T) {
			msg, err := serveNativeHostLoadPeakDecision(tc.peak, tc.estimate, tc.avail, tc.known, empty)
			if err != nil || msg.Level != tc.level {
				t.Fatalf("message=%+v error=%v, want level %s without refusal", msg, err, tc.level)
			}
			requireBoundedHostPeakDiagnostic(t, msg.Level, msg.Kind, msg.Text)
		})
	}

	for _, spelling := range []string{"off", "0", "FALSE", " no "} {
		env := hostPeakTestEnv(map[string]string{ggufload.HostLoadPeakOverrideEnv: spelling})
		msg, err := serveNativeHostLoadPeakDecision(peak, nil, gib, true, env)
		if err != nil || msg.Level != "warning" {
			t.Fatalf("override %q: message=%+v error=%v", spelling, msg, err)
		}
		requireBoundedHostPeakDiagnostic(t, msg.Level, msg.Kind, msg.Text)
	}
	if _, err := serveNativeHostLoadPeakDecision(peak, nil, gib, true, hostPeakTestEnv(map[string]string{ggufload.HostLoadPeakOverrideEnv: "on"})); !errors.Is(err, ggufload.ErrHostLoadPeakTooBig) {
		t.Fatalf("non-off override error = %v, want ErrHostLoadPeakTooBig", err)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestEstimateServeNativeHostLoadPeakChargesOwnedFallbackForMappedPlan(t *testing.T) {
	path := createMappedQ4KServeCompleteGGUF(t)
	opts := serveDenseKQuantOptions(nil)

	owned, err := estimateServeNativeHostLoadPeak(path, false, opts)
	if err != nil {
		t.Fatal(err)
	}
	plannedMapped, err := estimateServeNativeHostLoadPeak(path, true, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plannedMapped, owned) {
		t.Fatalf("mapped-plan admission peak = %+v, want conservative owned fallback %+v", plannedMapped, owned)
	}
	if plannedMapped.MappedBytes != 0 || plannedMapped.PeakAnonBytes <= 0 {
		t.Fatalf("mapped-plan admission peak = %+v, want positive anonymous owned fallback with no mapped charge", plannedMapped)
	}
}

// fak#13668: the device ring's actual options must not bypass admission merely
// because the ordinary resident estimator does not support streamed experts.
// fak-test:runtime fast est=20ms lane=default
func TestServeStreamedHostPeakAdmissionUsesActualRoute(t *testing.T) {
	path := writeServeStreamedSynthGGUF(t, "host-peak-streamed.gguf")
	ringOpts := serveActivatedExpertRingLoadOptions(serveActivatedExpertRing{RingBytes: 1 << 20})
	for name, denseOpts := range map[string][]ggufload.Q4KLoadOption{
		"ring without dense bound": nil,
		"unbounded dense":          {ggufload.WithStreamedDenseQ4K(true)},
		"zero dense bound":         {ggufload.WithStreamedDenseQ4KWorkingSet(0)},
		"negative dense bound":     {ggufload.WithStreamedDenseQ4KWorkingSet(-1)},
		"disabled dense route":     {ggufload.WithStreamedDenseQ4KWorkingSet(1024), ggufload.WithStreamedDenseQ4K(false)},
	} {
		t.Run(name, func(t *testing.T) {
			opts := append(append([]ggufload.Q4KLoadOption(nil), ringOpts...), denseOpts...)
			before := ggufload.ApplyQ4KLoadOptions(opts)
			msg, err := serveNativeHostLoadPeakAdmission(path, false, opts, hostPeakTestEnv(nil))
			var unavailable *serveHostLoadPeakUnavailableError
			if !errors.Is(err, errServeHostLoadPeakUnavailable) || !errors.As(err, &unavailable) ||
				unavailable.Reason != "header-estimate-unavailable" || !errors.Is(err, ggufload.ErrQ4KLoadEstimateUnsupported) {
				t.Fatalf("message=%+v error=%v, want typed unavailable actual route", msg, err)
			}
			if msg.Level != "error" || ggufload.ApplyQ4KLoadOptions(opts) != before {
				t.Fatalf("admission changed the route or failed open: message=%+v", msg)
			}
			requireBoundedHostPeakDiagnostic(t, msg.Level, msg.Kind, msg.Text)
		})
	}

	opts := append(ringOpts, ggufload.WithStreamedDenseQ4KWorkingSet(1024))
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	want, err := ws.EstimateStreamedExpertHostLoadPeak(opts...)
	if err != nil {
		t.Fatal(err)
	}
	got, err := estimateServeNativeHostLoadPeak(path, false, opts)
	if err != nil || got.PeakAnonBytes <= 0 || !reflect.DeepEqual(got, want) {
		t.Fatalf("streamed header estimate=%+v error=%v, want peer estimate %+v", got, err, want)
	}
	// An unreadable header must remain a typed refusal, retaining its I/O cause.
	_, err = serveNativeHostLoadPeakAdmission(filepath.Join(t.TempDir(), "missing.gguf"), false, opts, hostPeakTestEnv(nil))
	if !errors.Is(err, errServeHostLoadPeakUnavailable) || !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing header error=%v, want typed unavailable with file cause", err)
	}
}

// No fitting partial estimate is a license for an unqualified staging route.
// The existing capacity error retains exact estimated bytes, including the
// one-byte boundary, while unknown memory and unsupported routes fail closed.
// fak-test:runtime fast est=2ms lane=default
func TestServeStreamedHostPeakDecisionFailsClosed(t *testing.T) {
	const gib = int64(1 << 30)
	peak := ggufload.HostLoadPeak{SteadyAnonBytes: 18 * gib, TransientBytes: 6 * gib, PeakAnonBytes: 24 * gib}
	opts := []ggufload.Q4KLoadOption{ggufload.WithStreamedExperts(0), ggufload.WithStreamedDenseQ4KWorkingSet(gib)}
	for name, tc := range map[string]struct {
		peak     ggufload.HostLoadPeak
		estimate error
		avail    int64
		known    bool
		reason   string
	}{
		"unsupported estimate": {estimate: ggufload.ErrQ4KLoadEstimateUnsupported, avail: 26 * gib, known: true, reason: "header-estimate-unavailable"},
		"failed estimate":      {estimate: errors.New("fixture estimate failure"), avail: 26 * gib, known: true, reason: "header-estimate-unavailable"},
		"unknown memory":       {peak: peak, avail: 26 * gib, reason: "host-available-memory-unknown"},
		"negative memory":      {peak: peak, avail: -1, known: true, reason: "host-available-memory-unknown"},
		"nonpositive estimate": {avail: 26 * gib, known: true, reason: "header-estimate-nonpositive"},
		"exact partial fit":    {peak: peak, avail: 26 * gib, known: true, reason: "streamed-staging-peak-unqualified"},
		"ample memory":         {peak: peak, avail: 128 * gib, known: true, reason: "streamed-staging-peak-unqualified"},
	} {
		t.Run(name, func(t *testing.T) {
			msg, err := serveNativeHostLoadPeakDecision(tc.peak, tc.estimate, tc.avail, tc.known, hostPeakTestEnv(nil), opts...)
			var unavailable *serveHostLoadPeakUnavailableError
			if !errors.Is(err, errServeHostLoadPeakUnavailable) || !errors.As(err, &unavailable) || unavailable.Reason != tc.reason || msg.Level != "error" {
				t.Fatalf("message=%+v error=%v, want typed refusal %s", msg, err, tc.reason)
			}
			if tc.estimate != nil && !errors.Is(err, tc.estimate) {
				t.Fatalf("refusal lost estimate cause: %v", err)
			}
			requireBoundedHostPeakDiagnostic(t, msg.Level, msg.Kind, msg.Text)
		})
	}
	msg, err := serveNativeHostLoadPeakDecision(peak, nil, 26*gib-1, true, hostPeakTestEnv(nil), opts...)
	var capacity *ggufload.HostLoadPeakError
	if !errors.Is(err, ggufload.ErrHostLoadPeakTooBig) || !errors.As(err, &capacity) ||
		capacity.NeedBytes != peak.PeakAnonBytes || capacity.AvailBytes != 26*gib-1 || capacity.MarginBytes != 2*gib || msg.Level != "error" {
		t.Fatalf("message=%+v capacity=%+v error=%v, want exact estimated-byte refusal", msg, capacity, err)
	}
	for _, spelling := range []string{"off", "0", "FALSE", " no "} {
		env := hostPeakTestEnv(map[string]string{ggufload.HostLoadPeakOverrideEnv: spelling})
		msg, err := serveNativeHostLoadPeakDecision(peak, ggufload.ErrQ4KLoadEstimateUnsupported, 0, false, env, opts...)
		if err != nil || msg.Level != "warning" {
			t.Fatalf("override %q: message=%+v error=%v, want observable explicit override", spelling, msg, err)
		}
	}
}
