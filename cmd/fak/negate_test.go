package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

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
	runNegateDetect(strings.NewReader("do not forget to push\n"), &out, &errb, []string{"--json"})
	var findings []negframe.Finding
	if err := json.Unmarshal(out.Bytes(), &findings); err != nil {
		t.Fatalf("detect --json is not valid JSON: %v\n%s", err, out.String())
	}
	if len(findings) == 0 {
		t.Error("detect --json found no findings on negative prose")
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
	if !strings.Contains(out.String(), "Remember to stamp") {
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
