package systembaseline

import (
	"errors"
	"strings"
	"testing"
	"time"
)

func TestRefusalMessagesNameRecovery(t *testing.T) {
	valid := func() Report {
		policy := DefaultPolicy()
		policy.IncludeTopConsumers = true
		report := Build(quietFixture(100e6), fixture(100e6, 0), 10, time.Second, policy, 0, false)
		if err := report.Validate(); err != nil {
			t.Fatalf("fixture invalid: %v", err)
		}
		return report
	}
	reseal := func(edit func(*Report)) Report {
		report := valid()
		edit(&report)
		report.Seal()
		return report
	}
	cases := []struct {
		name     string
		report   Report
		recovery string
	}{
		{"schema", reseal(func(r *Report) { r.Schema = "old" }), "set schema"},
		{"verdict", reseal(func(r *Report) { r.Verdict = "maybe" }), "set verdict"},
		{"digest", func() Report { r := valid(); r.Digest = "sha256:tampered"; return r }(), "call Seal"},
		{"negative census", reseal(func(r *Report) { r.Coverage.ProcessReads = -1 }), "non-negative"},
		{"sampler coverage", reseal(func(r *Report) { r.BaselineSampler.CountedSamples++ }), "set counted samples"},
		{"sampler unavailable", reseal(func(r *Report) {
			r.BaselineSampler.DutyPercent = Metric{Reason: "missing"}
			r.BaselineSampler.WallNS = 1
		}), "set wall time"},
		{"sampler duty", reseal(func(r *Report) { r.BaselineSampler.DutyPercent.Value++ }), "recompute duty"},
		{"metric", reseal(func(r *Report) { r.Host.CPUPercent.Value = -1 }), "mark it unavailable"},
		{"contradictory verdict", reseal(func(r *Report) { r.Verdict = VerdictInvalid }), "policy-derived verdict"},
		{"consumer opt-in", reseal(func(r *Report) { r.Policy.IncludeTopConsumers = false }), "enable include_top_consumers"},
		{"consumer bound", reseal(func(r *Report) {
			for len(r.TopNonSUT) <= 5 {
				r.TopNonSUT = append(r.TopNonSUT, r.TopNonSUT[0])
			}
		}), "trim the list"},
		{"consumer identity", reseal(func(r *Report) { r.TopNonSUT[0].Image = "secret/path" }), "basename"},
		{"consumer CPU", reseal(func(r *Report) { r.TopNonSUT[0].CPUSeconds = -1 }), "finite non-negative"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.report.Validate()
			if err == nil {
				t.Fatal("Validate() accepted invalid report")
			}
			if !strings.Contains(err.Error(), tc.recovery) {
				t.Fatalf("Validate() error %q does not name recovery %q", err, tc.recovery)
			}
		})
	}
}

func TestErrorDecodeRefusalsNameRecovery(t *testing.T) {
	cases := []struct {
		name     string
		reader   *strings.Reader
		recovery string
	}{
		{"empty", strings.NewReader(""), "complete JSON report"},
		{"multiple", strings.NewReader("{} {}"), "exactly one JSON report"},
		{"trailing", strings.NewReader("{} !"), "remove bytes"},
		{"oversized", strings.NewReader(strings.Repeat(" ", maxEncodedReportBytes+1)), "reduce the report"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Decode(tc.reader)
			if err == nil || !strings.Contains(err.Error(), tc.recovery) {
				t.Fatalf("Decode() error = %v; want recovery %q", err, tc.recovery)
			}
		})
	}
	if _, err := Decode(nil); err == nil || !strings.Contains(err.Error(), "provide a report reader") {
		t.Fatalf("Decode(nil) error = %v; want reader recovery", err)
	}
	if _, err := Decode(recoveryErrorReader{}); err == nil || !errors.Is(err, errHostileReader) || !strings.Contains(err.Error(), "retry with readable") {
		t.Fatalf("Decode(errorReader) error = %v; want preserved cause and recovery", err)
	}
}

var errHostileReader = errors.New("hostile reader sentinel")

type recoveryErrorReader struct{}

func (recoveryErrorReader) Read([]byte) (int, error) { return 0, errHostileReader }
