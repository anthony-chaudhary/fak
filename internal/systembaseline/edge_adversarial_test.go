package systembaseline

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestEdgeBuildInputClasses(t *testing.T) {
	tests := []struct {
		name        string
		baseline    []Snapshot
		command     []Snapshot
		rootPID     int
		interval    time.Duration
		wantVerdict string
		wantFinding string
		wantValid   bool
	}{
		{name: "empty command", baseline: quietFixture(100e6), rootPID: 10, interval: time.Second, wantVerdict: VerdictInvalid, wantFinding: "NO_SAMPLES", wantValid: true},
		{name: "empty baseline", command: fixture(0, 0), rootPID: 10, interval: time.Second, wantVerdict: VerdictInvalid, wantFinding: "INVALID_BASELINE_WINDOW", wantValid: true},
		{name: "zero root", baseline: quietFixture(100e6), command: fixture(0, 0), interval: time.Second, wantVerdict: VerdictInvalid, wantFinding: "INVALID_WINDOW", wantValid: true},
		{name: "negative interval", baseline: quietFixture(100e6), command: fixture(0, 0), rootPID: 10, interval: -time.Second, wantVerdict: VerdictInvalid, wantFinding: "INVALID_WINDOW", wantValid: true},
		{name: "negative sampler wall", baseline: withCensusWall(quietFixture(100e6), -1), command: fixture(0, 0), rootPID: 10, interval: time.Second, wantVerdict: VerdictInvestigate, wantFinding: "SAMPLER_DUTY_UNKNOWN", wantValid: true},
		{name: "overflowing sampler wall", baseline: overflowingSamplerFixture(), command: fixture(0, 0), rootPID: 10, interval: time.Second, wantVerdict: VerdictInvestigate, wantFinding: "SAMPLER_DUTY_UNKNOWN", wantValid: true},
		{name: "host busy exceeds total", baseline: quietFixture(100e6), command: inconsistentHostFixture(), rootPID: 10, interval: time.Second, wantVerdict: VerdictInvestigate, wantFinding: "HOST_CPU_UNKNOWN", wantValid: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := Build(tc.baseline, tc.command, tc.rootPID, tc.interval, DefaultPolicy(), 0, false)
			if r.Verdict != tc.wantVerdict || !hasFindingCode(r, tc.wantFinding) {
				t.Fatalf("verdict=%q findings=%+v; want %q with %s", r.Verdict, r.Findings, tc.wantVerdict, tc.wantFinding)
			}
			if err := r.Validate(); (err == nil) != tc.wantValid {
				t.Fatalf("Validate() error = %v; want valid=%v", err, tc.wantValid)
			}
		})
	}
}

func TestEdgeBuildPreservesPerRepetitionInputs(t *testing.T) {
	baseline := quietFixture(100e6)
	command := fixture(0, 0)
	baseline[0], baseline[1] = baseline[1], baseline[0]
	command[0], command[1] = command[1], command[0]
	wantBaseline := append([]Snapshot(nil), baseline...)
	wantCommand := append([]Snapshot(nil), command...)

	r := Build(baseline, command, 10, time.Second, DefaultPolicy(), 0, false)
	if err := r.Validate(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(baseline, wantBaseline) || !reflect.DeepEqual(command, wantCommand) {
		t.Fatal("Build reordered caller-owned repetition samples")
	}
}

func TestEdgeBuildNormalizesHostilePolicyThresholds(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Policy)
	}{
		{name: "non-SUT NaN", edit: func(p *Policy) { p.MaximumNonSUTCPUPercent = math.NaN() }},
		{name: "non-SUT infinity", edit: func(p *Policy) { p.MaximumNonSUTCPUPercent = math.Inf(1) }},
		{name: "non-SUT oversized", edit: func(p *Policy) { p.MaximumNonSUTCPUPercent = math.MaxFloat64 }},
		{name: "sampler NaN", edit: func(p *Policy) { p.MaximumSamplerDutyPercent = math.NaN() }},
		{name: "sampler infinity", edit: func(p *Policy) { p.MaximumSamplerDutyPercent = math.Inf(1) }},
		{name: "sampler oversized", edit: func(p *Policy) { p.MaximumSamplerDutyPercent = math.MaxFloat64 }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			policy := DefaultPolicy()
			tc.edit(&policy)
			report := Build(quietFixture(100e6), fixture(0, 0), 10, time.Second, policy, 0, false)
			if report.Policy.MaximumNonSUTCPUPercent != DefaultPolicy().MaximumNonSUTCPUPercent || report.Policy.MaximumSamplerDutyPercent != DefaultPolicy().MaximumSamplerDutyPercent {
				t.Fatalf("hostile policy was not normalized: %+v", report.Policy)
			}
			if err := report.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestAdversarialPerRepetitionDigestBinding(t *testing.T) {
	seen := map[string]bool{}
	for repetition := 1; repetition <= 3; repetition++ {
		baseline := shiftSnapshots(quietFixture(100e6), time.Duration(repetition)*time.Minute)
		command := shiftSnapshots(fixture(0, 0), time.Duration(repetition)*time.Minute)
		report := Build(baseline, command, 10, time.Second, DefaultPolicy(), 0, false)
		if err := report.Validate(); err != nil {
			t.Fatalf("repetition %d: %v", repetition, err)
		}
		if seen[report.Digest] {
			t.Fatalf("repetition %d reused digest %q", repetition, report.Digest)
		}
		seen[report.Digest] = true

		again := Build(shiftSnapshots(quietFixture(100e6), time.Duration(repetition)*time.Minute), shiftSnapshots(fixture(0, 0), time.Duration(repetition)*time.Minute), 10, time.Second, DefaultPolicy(), 0, false)
		if again.Digest != report.Digest {
			t.Fatalf("repetition %d digest is nondeterministic: %q != %q", repetition, again.Digest, report.Digest)
		}
		report.Window.StartedAtUTC = time.Unix(int64(repetition), 0).UTC().Format(time.RFC3339Nano)
		if err := report.Validate(); err == nil {
			t.Fatalf("repetition %d accepted an attestation rebound to another window", repetition)
		}
	}
}

func TestAdversarialDecodeRejectsMalformedAndOversizedInputs(t *testing.T) {
	tests := []struct {
		name string
		rd   io.Reader
		want string
	}{
		{name: "nil reader", want: "nil"},
		{name: "empty", rd: strings.NewReader(""), want: "EOF"},
		{name: "truncated", rd: strings.NewReader(`{"schema":`), want: "unexpected EOF"},
		{name: "unknown field", rd: strings.NewReader(`{"unknown":true}`), want: "unknown field"},
		{name: "multiple values", rd: strings.NewReader(`{} {}`), want: "multiple JSON values"},
		{name: "trailing malformed data", rd: strings.NewReader(`{} !`), want: "trailing data"},
		{name: "reader error", rd: errorReader{}, want: "hostile reader"},
		{name: "oversized", rd: strings.NewReader(strings.Repeat(" ", (1<<20)+1)), want: "exceeds"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Decode(tc.rd)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Decode() error = %v; want substring %q", err, tc.want)
			}
		})
	}

	report := Build(quietFixture(100e6), fixture(0, 0), 10, time.Second, DefaultPolicy(), 0, false)
	raw := fmt.Sprintf("\n\t%s \r\n", mustJSON(t, report))
	decoded, err := Decode(strings.NewReader(raw))
	if err != nil || decoded.Digest != report.Digest {
		t.Fatalf("valid report with whitespace: digest=%q err=%v", decoded.Digest, err)
	}
}

func TestAdversarialValidateRejectsHostileResealedReports(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Report)
	}{
		{name: "negative interval", edit: func(r *Report) { r.Window.IntervalNS = -1 }},
		{name: "negative process snapshots", edit: func(r *Report) { r.Coverage.ProcessSnapshots = -1 }},
		{name: "negative processes observed", edit: func(r *Report) { r.Coverage.ProcessesObserved = -1 }},
		{name: "negative process reads", edit: func(r *Report) { r.Coverage.ProcessReads = -1 }},
		{name: "negative process unreadable", edit: func(r *Report) { r.Coverage.ProcessUnreadable = -1 }},
		{name: "negative host CPU samples", edit: func(r *Report) { r.Coverage.HostCPUSamples = -1 }},
		{name: "negative host memory samples", edit: func(r *Report) { r.Coverage.HostMemorySamples = -1 }},
		{name: "too many top consumers", edit: func(r *Report) {
			r.Policy.IncludeTopConsumers = true
			r.TopNonSUT = make([]Consumer, 6)
			for i := range r.TopNonSUT {
				r.TopNonSUT[i] = Consumer{Image: fmt.Sprintf("p%d", i), PID: i + 1}
			}
		}},
		{name: "negative consumer CPU", edit: func(r *Report) {
			r.Policy.IncludeTopConsumers = true
			r.TopNonSUT = []Consumer{{Image: "safe", PID: 1, CPUSeconds: -1}}
		}},
		{name: "non-finite policy", edit: func(r *Report) { r.Policy.MaximumNonSUTCPUPercent = math.NaN() }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := Build(quietFixture(100e6), fixture(0, 0), 10, time.Second, DefaultPolicy(), 0, false)
			tc.edit(&r)
			r.Seal()
			if err := r.Validate(); err == nil {
				t.Fatal("Validate accepted hostile resealed report")
			}
		})
	}
}

func TestEdgePlatformReaderSeams(t *testing.T) {
	linuxCases := []struct {
		name        string
		ticks       []uint64
		total, idle uint64
		ok          bool
	}{
		{name: "empty"},
		{name: "undersized", ticks: []uint64{1, 2, 3}},
		{name: "minimum", ticks: []uint64{10, 20, 30, 40}, total: 100, idle: 40, ok: true},
		{name: "guest counters excluded", ticks: []uint64{100, 20, 30, 40, 5, 1, 2, 3, 90, 10}, total: 201, idle: 45, ok: true},
		{name: "counter sum overflow", ticks: []uint64{math.MaxUint64, 1, 0, 0}},
	}
	for _, tc := range linuxCases {
		t.Run("linux/"+tc.name, func(t *testing.T) {
			total, idle, ok := canonicalLinuxCPUTicks(tc.ticks)
			if total != tc.total || idle != tc.idle || ok != tc.ok {
				t.Fatalf("canonicalLinuxCPUTicks() = (%d, %d, %v); want (%d, %d, %v)", total, idle, ok, tc.total, tc.idle, tc.ok)
			}
		})
	}
	linuxScaleCases := []struct {
		name  string
		ticks uint64
		want  uint64
		ok    bool
	}{
		{name: "zero", ok: true},
		{name: "fractional second", ticks: 123, want: 1230e6, ok: true},
		{name: "conversion overflow", ticks: math.MaxUint64},
	}
	for _, tc := range linuxScaleCases {
		t.Run("linux-scale/"+tc.name, func(t *testing.T) {
			got, ok := linuxCPUTicksToNS(tc.ticks)
			if got != tc.want || ok != tc.ok {
				t.Fatalf("linuxCPUTicksToNS() = (%d, %v); want (%d, %v)", got, ok, tc.want, tc.ok)
			}
		})
	}

	windowsCases := []struct {
		name            string
		result, code    uintptr
		done, truncated bool
	}{
		{name: "continue despite stale error", result: 1, code: 5},
		{name: "normal completion", code: 18, done: true},
		{name: "zero error is truncation", done: true, truncated: true},
		{name: "access denied is truncation", code: 5, done: true, truncated: true},
		{name: "hostile large code is truncation", code: ^uintptr(0), done: true, truncated: true},
	}
	for _, tc := range windowsCases {
		t.Run("windows/"+tc.name, func(t *testing.T) {
			done, truncated := classifyWindowsProcessAdvance(tc.result, tc.code)
			if done != tc.done || truncated != tc.truncated {
				t.Fatalf("classifyWindowsProcessAdvance() = (%v, %v); want (%v, %v)", done, truncated, tc.done, tc.truncated)
			}
		})
	}
	windowsCPU := []struct {
		name               string
		idle, kernel, user uint64
		total, busy        uint64
		ok                 bool
	}{
		{name: "normal", idle: 4, kernel: 10, user: 2, total: 1200, busy: 800, ok: true},
		{name: "idle exceeds kernel", idle: 11, kernel: 10, user: 2},
		{name: "counter sum overflow", kernel: math.MaxUint64, user: 1},
		{name: "nanosecond conversion overflow", kernel: math.MaxUint64 / 2},
	}
	for _, tc := range windowsCPU {
		t.Run("windows-cpu/"+tc.name, func(t *testing.T) {
			total, busy, ok := canonicalWindowsHostCPUNS(tc.idle, tc.kernel, tc.user)
			if total != tc.total || busy != tc.busy || ok != tc.ok {
				t.Fatalf("canonicalWindowsHostCPUNS() = (%d, %d, %v); want (%d, %d, %v)", total, busy, ok, tc.total, tc.busy, tc.ok)
			}
		})
	}
	processCPU := []struct {
		name         string
		kernel, user uint64
		want         uint64
		ok           bool
	}{
		{name: "normal", kernel: 10, user: 2, want: 1200, ok: true},
		{name: "sum overflow", kernel: math.MaxUint64, user: 1},
		{name: "conversion overflow", kernel: math.MaxUint64 / 2},
	}
	for _, tc := range processCPU {
		t.Run("windows-process/"+tc.name, func(t *testing.T) {
			got, ok := canonicalWindowsProcessCPUNS(tc.kernel, tc.user)
			if got != tc.want || ok != tc.ok {
				t.Fatalf("canonicalWindowsProcessCPUNS() = (%d, %v); want (%d, %v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

func TestAdversarialOversizedProcessCensusStaysBounded(t *testing.T) {
	baseline := quietFixture(100e6)
	command := fixture(0, 0)
	for i := 0; i < 128; i++ {
		first := ProcessSample{PID: 1000 + i, PPID: 1, StartID: uint64(1000 + i), Image: fmt.Sprintf(`/host/%03d/%s`, i, strings.Repeat("x", 128)), CPUAvailable: true, CPUNS: uint64(i), RSSAvailable: true, RSSBytes: uint64(i + 1)}
		last := first
		last.CPUNS += uint64(i + 1)
		last.RSSBytes++
		command[0].Processes = append(command[0].Processes, first)
		command[1].Processes = append(command[1].Processes, last)
	}
	policy := DefaultPolicy()
	policy.IncludeTopConsumers = true
	report := Build(baseline, command, 10, time.Second, policy, 0, false)
	if len(report.TopNonSUT) != 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST the report contract returns the configured top-five non-SUT entries
		t.Fatalf("top consumer count=%d, want hard bound 5", len(report.TopNonSUT))
	}
	if err := report.Validate(); err != nil {
		t.Fatal(err)
	}
}

func withCensusWall(samples []Snapshot, wall int64) []Snapshot {
	out := append([]Snapshot(nil), samples...)
	out[0].CensusWallNS = wall
	return out
}

func overflowingSamplerFixture() []Snapshot {
	samples := quietFixture(100e6)
	middle := samples[0]
	middle.At = samples[0].At.Add(500 * time.Millisecond)
	samples = append(samples, Snapshot{})
	copy(samples[2:], samples[1:])
	samples[1] = middle
	samples[0].CensusWallNS = math.MaxInt64
	samples[1].CensusWallNS = math.MaxInt64
	return samples
}

func inconsistentHostFixture() []Snapshot {
	samples := fixture(0, 0)
	samples[1].Host.BusyCPUNS = samples[0].Host.BusyCPUNS + 5e9
	return samples
}

func shiftSnapshots(samples []Snapshot, delta time.Duration) []Snapshot {
	out := append([]Snapshot(nil), samples...)
	for i := range out {
		out[i].At = out[i].At.Add(delta)
	}
	return out
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

type errorReader struct{}

func (errorReader) Read([]byte) (int, error) { return 0, errors.New("hostile reader") }
