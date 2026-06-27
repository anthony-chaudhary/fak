package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// runResumeAt drives the testable core and returns stdout, stderr, and the exit code.
func runResumeAt(argv ...string) (string, string, int) {
	var out, errb bytes.Buffer
	code := runResume(&out, &errb, argv)
	return out.String(), errb.String(), code
}

// TestResumePlanColdHeadline is the CLI half of the goal: the 250k / idle-2h case prints a
// COLD posture and recommends CUT.
func TestResumePlanColdHeadline(t *testing.T) {
	out, errb, code := runResumeAt("plan", "--resident-tokens", "250000", "--idle-seconds", "7200")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	if !strings.Contains(out, "posture=COLD") {
		t.Errorf("output missing COLD posture:\n%s", out)
	}
	if !strings.Contains(out, "recommended: CUT") {
		t.Errorf("output missing CUT recommendation:\n%s", out)
	}
}

// TestResumePlanJSON: --json emits a parseable Report whose recommendation is cut on the
// cold headline case.
func TestResumePlanJSON(t *testing.T) {
	out, errb, code := runResumeAt("plan", "--resident-tokens", "250000", "--idle-seconds", "7200", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var rep struct {
		Posture     string `json:"posture"`
		Recommended string `json:"recommended"`
		Reason      string `json:"reason"`
		Strategies  []struct {
			Strategy string `json:"strategy"`
		} `json:"strategies"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	if rep.Posture != "cold" || rep.Recommended != "cut" || rep.Reason != "cold_prefill_shed" {
		t.Errorf("got posture=%q recommended=%q reason=%q, want cold/cut/cold_prefill_shed", rep.Posture, rep.Recommended, rep.Reason)
	}
	if len(rep.Strategies) != 3 {
		t.Errorf("want 3 strategies, got %d", len(rep.Strategies))
	}
}

// TestResumePlanWarmKeepsFull: idle within the TTL with a short horizon recommends keeping
// the whole transcript.
func TestResumePlanWarmKeepsFull(t *testing.T) {
	out, _, code := runResumeAt("plan", "--resident-tokens", "250000", "--idle-seconds", "60", "--horizon", "3")
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !strings.Contains(out, "posture=WARM") || !strings.Contains(out, "recommended: RESUME_FULL") {
		t.Errorf("warm short-horizon case should keep full:\n%s", out)
	}
}

// TestResumeUsageErrors covers the exit-2 paths: no subcommand, unknown subcommand, bad TTL,
// and a missing resident size.
func TestResumeUsageErrors(t *testing.T) {
	cases := [][]string{
		{},             // no subcommand
		{"frobnicate"}, // unknown subcommand
		{"plan", "--resident-tokens", "100", "--ttl", "7d"}, // bad TTL
		{"plan", "--idle-seconds", "10"},                    // no resident size
	}
	for _, argv := range cases {
		if _, _, code := runResumeAt(argv...); code != 2 {
			t.Errorf("argv %v: exit = %d, want 2", argv, code)
		}
	}
}

// TestResumeHelp: the help subcommand exits 0 and prints the example.
func TestResumeHelp(t *testing.T) {
	out, _, code := runResumeAt("help")
	if code != 0 {
		t.Fatalf("help exit = %d, want 0", code)
	}
	if !strings.Contains(out, "fak resume plan") {
		t.Errorf("help missing usage:\n%s", out)
	}
}
