package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

type detectEnvelope struct {
	Present  bool `json:"present"`
	Findings []struct {
		Line       int               `json:"line"`
		Span       string            `json:"span"`
		Category   negframe.Category `json:"category"`
		Mechanical bool              `json:"mechanical"`
	} `json:"findings"`
}

func TestNegateDetect(t *testing.T) {
	var out, errb bytes.Buffer
	// A prohibition idiom -> Classify finds a mechanical finding -> exit 1.
	code := runNegateDetect(strings.NewReader("Don't forget to stamp the commit.\n"), &out, &errb, nil)
	if code != 1 {
		t.Fatalf("detect on negative prose exit = %d, want 1 (stderr=%q)", code, errb.String())
	}
	if !strings.Contains(out.String(), "mechanical") {
		t.Errorf("detect output missing mechanical tier: %q", out.String())
	}
}

func TestNegateDetectClean(t *testing.T) {
	var out, errb bytes.Buffer
	code := runNegateDetect(strings.NewReader("Remember to stamp the commit.\n"), &out, &errb, nil)
	if code != 0 {
		t.Fatalf("detect on clean prose exit = %d, want 0", code)
	}
	if !strings.Contains(out.String(), "clean") {
		t.Errorf("detect output should report clean: %q", out.String())
	}
}

func TestNegateDetectJSON(t *testing.T) {
	var out, errb bytes.Buffer
	code := runNegateDetect(strings.NewReader("do not forget to push\nShip directly.\nNo need to wait.\n"), &out, &errb, []string{"--json"})
	if code != 1 {
		t.Fatalf("detect --json exit = %d, want 1 (stderr=%q)", code, errb.String())
	}
	var got detectEnvelope
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("detect --json is not valid JSON: %v\n%s", err, out.String())
	}
	if !got.Present {
		t.Fatal("detect --json present = false, want true")
	}
	if len(got.Findings) != 2 {
		t.Fatalf("detect --json findings = %d, want 2: %#v", len(got.Findings), got.Findings)
	}
	first := got.Findings[0]
	if first.Line != 1 || first.Span != "do not forget to push" || first.Category != negframe.Prohibition || !first.Mechanical {
		t.Errorf("first finding = %#v, want line 1 exact mechanical prohibition span", first)
	}
	second := got.Findings[1]
	if second.Line != 3 || second.Span != "No need to wait" || second.Category != negframe.Absence || !second.Mechanical {
		t.Errorf("second finding = %#v, want line 3 exact mechanical absence span", second)
	}
}

func TestNegateDetectJSONCleanEnvelope(t *testing.T) {
	var out, errb bytes.Buffer
	code := runNegateDetect(strings.NewReader("Ship directly.\n"), &out, &errb, []string{"--json"})
	if code != 0 {
		t.Fatalf("clean detect --json exit = %d, want 0 (stderr=%q)", code, errb.String())
	}
	var got detectEnvelope
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("clean detect --json is not valid JSON: %v\n%s", err, out.String())
	}
	if got.Present || len(got.Findings) != 0 {
		t.Fatalf("clean envelope = %#v, want present=false and empty findings", got)
	}
}

func TestNegateResolvePositional(t *testing.T) {
	var out, errb bytes.Buffer
	code := runNegateResolve(&out, &errb, []string{"not shared"})
	if code != 0 {
		t.Fatalf("resolve \"not shared\" exit = %d, want 0 (stderr=%q)", code, errb.String())
	}
	if !strings.Contains(out.String(), "exclusive") || !strings.Contains(out.String(), "exact") {
		t.Errorf("resolve output = %q, want exact exclusive", out.String())
	}
}

func TestNegateResolveCandidatesJSON(t *testing.T) {
	var out, errb bytes.Buffer
	code := runNegateResolve(&out, &errb, []string{"--json", "--domain", "lane-kind", "--negated", "global"})
	if code != 0 {
		t.Fatalf("resolve candidates exit = %d, want 0", code)
	}
	var res negframe.Resolution
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("resolve --json invalid: %v\n%s", err, out.String())
	}
	if res.Kind != negframe.Candidates {
		t.Errorf("kind = %q, want candidates", res.Kind)
	}
}

func TestNegateResolveUnknownExit1(t *testing.T) {
	var out, errb bytes.Buffer
	code := runNegateResolve(&out, &errb, []string{"--negated", "banana"})
	if code != 1 {
		t.Fatalf("resolve unknown exit = %d, want 1 (fail-closed)", code)
	}
	if !strings.Contains(out.String(), "UNKNOWN") {
		t.Errorf("resolve unknown output = %q, want UNKNOWN", out.String())
	}
}

func TestNegateResolveList(t *testing.T) {
	var out, errb bytes.Buffer
	code := runNegateResolve(&out, &errb, []string{"--list"})
	if code != 0 {
		t.Fatalf("resolve --list exit = %d, want 0", code)
	}
	for _, want := range []string{"lock-mode", "lane-kind", "boolean"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("--list missing domain %q: %q", want, out.String())
		}
	}
}

func TestNegateResolveNoTermUsage(t *testing.T) {
	var out, errb bytes.Buffer
	code := runNegateResolve(&out, &errb, nil)
	if code != 2 {
		t.Fatalf("resolve with no term exit = %d, want 2 (usage)", code)
	}
}

func TestNegateReframe(t *testing.T) {
	var out, errb bytes.Buffer
	code := runNegateReframe(strings.NewReader("Don't forget to stamp the commit.\n"), &out, &errb, nil)
	if code != 0 {
		t.Fatalf("reframe exit = %d, want 0", code)
	}
	if !strings.Contains(strings.ToLower(out.String()), "remember to stamp") {
		t.Errorf("reframe did not flip the idiom: %q", out.String())
	}
	if strings.Contains(strings.ToLower(out.String()), "forget") {
		t.Errorf("reframe left the negative idiom in place: %q", out.String())
	}
}

func TestNegateReframeJSON(t *testing.T) {
	var out, errb bytes.Buffer
	runNegateReframe(strings.NewReader("do not forget to push\n"), &out, &errb, []string{"--json"})
	var res negframe.ReframeResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("reframe --json invalid: %v\n%s", err, out.String())
	}
	if res.Applied == 0 {
		t.Errorf("reframe --json reported 0 applied on a mechanical idiom: %+v", res)
	}
}
