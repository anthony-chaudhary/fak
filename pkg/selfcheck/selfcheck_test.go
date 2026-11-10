package selfcheck

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestNewReport(t *testing.T) {
	before := time.Now().UTC().Add(-time.Second)
	r := NewReport("fak doctor")
	after := time.Now().UTC().Add(time.Second)

	if r == nil {
		t.Fatal("NewReport returned nil")
	}
	if r.Command != "fak doctor" {
		t.Errorf("Command = %q, want %q", r.Command, "fak doctor")
	}
	if r.Timestamp.IsZero() {
		t.Error("Timestamp is zero, want non-zero")
	}
	if loc := r.Timestamp.Location(); loc != time.UTC {
		t.Errorf("Timestamp location = %v, want UTC", loc)
	}
	if r.Timestamp.Before(before) || r.Timestamp.After(after) {
		t.Errorf("Timestamp %v out of expected window [%v, %v]", r.Timestamp, before, after)
	}
	if r.PassCount != 0 || r.WarnCount != 0 || r.FailCount != 0 {
		t.Errorf("counts = (%d,%d,%d), want (0,0,0)", r.PassCount, r.WarnCount, r.FailCount)
	}
	// An empty report has no failures and is PASS by construction.
	if !r.OverallPass {
		t.Error("OverallPass = false for a bare NewReport, want true (no failures)")
	}
	if len(r.Items) != 0 {
		t.Errorf("len(Items) = %d, want 0", len(r.Items))
	}
}

func TestAddCounts(t *testing.T) {
	tests := []struct {
		name                         string
		statuses                     []Status
		wantPass, wantWarn, wantFail int
		wantOverall                  bool
	}{
		{
			// No Add call: an empty report has no failures, so it is PASS.
			name:        "empty",
			statuses:    nil,
			wantPass:    0,
			wantWarn:    0,
			wantFail:    0,
			wantOverall: true,
		},
		{
			name:        "all pass",
			statuses:    []Status{StatusPass, StatusPass},
			wantPass:    2,
			wantOverall: true,
		},
		{
			name:        "warn only stays passing",
			statuses:    []Status{StatusWarn, StatusWarn},
			wantWarn:    2,
			wantOverall: true,
		},
		{
			name:        "pass and warn",
			statuses:    []Status{StatusPass, StatusWarn, StatusPass},
			wantPass:    2,
			wantWarn:    1,
			wantOverall: true,
		},
		{
			name:        "single fail",
			statuses:    []Status{StatusFail},
			wantFail:    1,
			wantOverall: false,
		},
		{
			name:        "mixed with fail",
			statuses:    []Status{StatusPass, StatusWarn, StatusFail, StatusPass},
			wantPass:    2,
			wantWarn:    1,
			wantFail:    1,
			wantOverall: false,
		},
		{
			name:        "unknown status ignored",
			statuses:    []Status{"OTHER"},
			wantOverall: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewReport("cmd")
			for i, s := range tt.statuses {
				r.Add(Item{Category: "cat", Name: "item", Status: s, Message: "msg"})
				if got := len(r.Items); got != i+1 {
					t.Fatalf("after Add %d: len(Items) = %d, want %d", i, got, i+1)
				}
			}
			if r.PassCount != tt.wantPass {
				t.Errorf("PassCount = %d, want %d", r.PassCount, tt.wantPass)
			}
			if r.WarnCount != tt.wantWarn {
				t.Errorf("WarnCount = %d, want %d", r.WarnCount, tt.wantWarn)
			}
			if r.FailCount != tt.wantFail {
				t.Errorf("FailCount = %d, want %d", r.FailCount, tt.wantFail)
			}
			if r.OverallPass != tt.wantOverall {
				t.Errorf("OverallPass = %v, want %v", r.OverallPass, tt.wantOverall)
			}
		})
	}
}

func TestOverallPassTransitions(t *testing.T) {
	r := NewReport("cmd")
	r.Add(Pass("net", "reachable", "ok"))
	r.Add(Warn("disk", "space", "low", "free space"))
	if !r.OverallPass {
		t.Fatal("OverallPass = false after only pass+warn, want true")
	}
	r.Add(Fail("mem", "alloc", "oom", "add ram"))
	if r.OverallPass {
		t.Fatal("OverallPass = true after FAIL, want false")
	}
}

func TestExitCode(t *testing.T) {
	tests := []struct {
		name  string
		items []Item
		want  int
	}{
		// An empty report has no failures, so a bare report exits 0.
		{name: "bare empty report", want: 0},
		{name: "passes", items: []Item{Pass("a", "b", "c")}, want: 0},
		{name: "warns only", items: []Item{Warn("a", "b", "c", "d")}, want: 0},
		{name: "fail", items: []Item{Fail("a", "b", "c", "d")}, want: 1},
		{name: "pass and fail", items: []Item{Pass("a", "b", "c"), Fail("a", "b", "c", "d")}, want: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewReport("cmd")
			for _, it := range tt.items {
				r.Add(it)
			}
			if got := r.ExitCode(); got != tt.want {
				t.Errorf("ExitCode() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestConstructors(t *testing.T) {
	p := Pass("cat", "name", "msg")
	if p.Status != StatusPass {
		t.Errorf("Pass Status = %q, want %q", p.Status, StatusPass)
	}
	if p.Category != "cat" || p.Name != "name" || p.Message != "msg" {
		t.Errorf("Pass fields = %+v", p)
	}
	if p.Remediation != "" {
		t.Errorf("Pass Remediation = %q, want empty", p.Remediation)
	}

	w := Warn("cat", "name", "msg", "fix it")
	if w.Status != StatusWarn {
		t.Errorf("Warn Status = %q, want %q", w.Status, StatusWarn)
	}
	if w.Category != "cat" || w.Name != "name" || w.Message != "msg" || w.Remediation != "fix it" {
		t.Errorf("Warn fields = %+v", w)
	}

	f := Fail("cat", "name", "msg", "fix it")
	if f.Status != StatusFail {
		t.Errorf("Fail Status = %q, want %q", f.Status, StatusFail)
	}
	if f.Category != "cat" || f.Name != "name" || f.Message != "msg" || f.Remediation != "fix it" {
		t.Errorf("Fail fields = %+v", f)
	}
}

func TestStringTitle(t *testing.T) {
	t.Run("with command", func(t *testing.T) {
		r := NewReport("fak-serve doctor")
		r.Add(Pass("x", "y", "z"))
		if !strings.Contains(r.String(), "fak-serve doctor") {
			t.Errorf("String() missing command:\n%s", r.String())
		}
	})
	t.Run("without command", func(t *testing.T) {
		r := NewReport("")
		r.Add(Pass("x", "y", "z"))
		s := r.String()
		if !strings.Contains(s, "Self-Check Report") {
			t.Errorf("String() missing default title:\n%s", s)
		}
		if strings.Contains(s, "Self-Check Report: ") {
			t.Errorf("String() unexpected command suffix:\n%s", s)
		}
	})
}

func TestStringSummary(t *testing.T) {
	tests := []struct {
		name        string
		items       []Item
		wantOverall string
		wantCounts  string
	}{
		{
			name:        "pass",
			items:       []Item{Pass("a", "b", "c"), Pass("a", "b", "c")},
			wantOverall: "(OVERALL: PASS)",
			wantCounts:  "Summary: 2 passed, 0 warnings, 0 failures",
		},
		{
			name:        "warn still pass",
			items:       []Item{Pass("a", "b", "c"), Warn("a", "b", "c", "d")},
			wantOverall: "(OVERALL: PASS)",
			wantCounts:  "Summary: 1 passed, 1 warnings, 0 failures",
		},
		{
			name:        "fail",
			items:       []Item{Pass("a", "b", "c"), Warn("a", "b", "c", "d"), Fail("a", "b", "c", "d")},
			wantOverall: "(OVERALL: FAIL)",
			wantCounts:  "Summary: 1 passed, 1 warnings, 1 failures",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := NewReport("cmd")
			for _, it := range tt.items {
				r.Add(it)
			}
			s := r.String()
			if !strings.Contains(s, "Summary:") {
				t.Errorf("String() missing Summary:\n%s", s)
			}
			if !strings.Contains(s, tt.wantCounts) {
				t.Errorf("String() missing counts %q:\n%s", tt.wantCounts, s)
			}
			if !strings.Contains(s, tt.wantOverall) {
				t.Errorf("String() missing %q:\n%s", tt.wantOverall, s)
			}
		})
	}
}

func TestStringRemediation(t *testing.T) {
	r := NewReport("cmd")
	r.Add(Warn("disk", "space", "getting low", "run df and prune"))
	s := r.String()
	if !strings.Contains(s, "Remediation: run df and prune") {
		t.Errorf("String() missing remediation line:\n%s", s)
	}
}

func TestWriteJSONRoundTrip(t *testing.T) {
	r := NewReport("fak-doctor")
	r.Add(Pass("cpu", "avx512", "available"))
	r.Add(Warn("gpu", "rocm", "old", "upgrade"))
	r.Add(Fail("net", "dns", "broken", "fix resolv.conf"))

	var buf bytes.Buffer
	if err := r.WriteJSON(&buf); err != nil {
		t.Fatalf("WriteJSON error: %v", err)
	}

	var got Report
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal error: %v\nraw: %s", err, buf.String())
	}

	if got.Command != r.Command {
		t.Errorf("Command = %q, want %q", got.Command, r.Command)
	}
	if got.PassCount != r.PassCount {
		t.Errorf("PassCount = %d, want %d", got.PassCount, r.PassCount)
	}
	if got.WarnCount != r.WarnCount {
		t.Errorf("WarnCount = %d, want %d", got.WarnCount, r.WarnCount)
	}
	if got.FailCount != r.FailCount {
		t.Errorf("FailCount = %d, want %d", got.FailCount, r.FailCount)
	}
	if got.OverallPass != r.OverallPass {
		t.Errorf("OverallPass = %v, want %v", got.OverallPass, r.OverallPass)
	}
	if len(got.Items) != len(r.Items) {
		t.Fatalf("len(Items) = %d, want %d", len(got.Items), len(r.Items))
	}
	for i := range r.Items {
		if got.Items[i] != r.Items[i] {
			t.Errorf("Items[%d] = %+v, want %+v", i, got.Items[i], r.Items[i])
		}
	}
}

func TestRender(t *testing.T) {
	tests := []struct {
		name     string
		items    []Item
		wantExit int
	}{
		{
			name:     "passing report",
			items:    []Item{Pass("a", "b", "c"), Warn("a", "b", "c", "d")},
			wantExit: 0,
		},
		{
			name:     "failing report",
			items:    []Item{Pass("a", "b", "c"), Fail("a", "b", "c", "d")},
			wantExit: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name+" text", func(t *testing.T) {
			r := NewReport("cmd")
			for _, it := range tt.items {
				r.Add(it)
			}
			var buf bytes.Buffer
			code, err := r.Render(&buf, false)
			if err != nil {
				t.Fatalf("Render(text) error: %v", err)
			}
			if code != tt.wantExit {
				t.Errorf("Render(text) exit = %d, want %d", code, tt.wantExit)
			}
			if buf.String() != r.String() {
				t.Errorf("Render(text) != String():\n got: %q\nwant: %q", buf.String(), r.String())
			}
		})
		t.Run(tt.name+" json", func(t *testing.T) {
			r := NewReport("cmd")
			for _, it := range tt.items {
				r.Add(it)
			}
			var buf bytes.Buffer
			code, err := r.Render(&buf, true)
			if err != nil {
				t.Fatalf("Render(json) error: %v", err)
			}
			if code != tt.wantExit {
				t.Errorf("Render(json) exit = %d, want %d", code, tt.wantExit)
			}
			var got Report
			if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
				t.Fatalf("Render(json) invalid JSON: %v\nraw: %s", err, buf.String())
			}
			if got.FailCount != r.FailCount || got.OverallPass != r.OverallPass {
				t.Errorf("round-trip mismatch: got fails=%d overall=%v, want fails=%d overall=%v",
					got.FailCount, got.OverallPass, r.FailCount, r.OverallPass)
			}
		})
	}
}

// errWriter fails every write, proving Render's error paths.
type errWriter struct{ err error }

func (w errWriter) Write(p []byte) (int, error) { return 0, w.err }

func TestRenderWriteError(t *testing.T) {
	sentinel := errors.New("sink is closed")
	for _, jsonOut := range []bool{false, true} {
		name := "text"
		if jsonOut {
			name = "json"
		}
		t.Run(name, func(t *testing.T) {
			r := NewReport("cmd")
			r.Add(Pass("a", "b", "c"))
			code, err := r.Render(errWriter{err: sentinel}, jsonOut)
			if !errors.Is(err, sentinel) {
				t.Fatalf("Render(%s) error = %v, want %v", name, err, sentinel)
			}
			if code != 1 {
				t.Errorf("Render(%s) exit = %d, want 1 on write error", name, code)
			}
		})
	}
}
