package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/answershape"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/binstamp"
)

func defLimits() answershape.Limits {
	return answershape.Limits{MaxRepeat: answershape.DefaultMaxRepeat, NGram: answershape.DefaultNGram}
}

func TestDiagnoseCleanHasNoFindings(t *testing.T) {
	rep := diagnose([]byte("The kernel repairs a malformed call without a model turn, then dispatches it."), defLimits())
	if rep.Findings != 0 {
		t.Fatalf("clean text: findings=%d, want 0 (%+v)", rep.Findings, rep.Recommendations)
	}
	if rep.KernelWouldHold {
		t.Fatalf("clean text should not be kernel-held")
	}
	for _, r := range rep.Recommendations {
		if r.Severity != sevOK {
			t.Fatalf("clean text: check %q severity %q, want ok", r.Check, r.Severity)
		}
	}
}

func TestDiagnoseShapeWarnButKernelAdmitsSmallLoop(t *testing.T) {
	// A small loop (< the kernel's 512-byte / >50-rep oversize floor) is degenerate
	// by the graded witness but the conservative kernel gate still ADMITS it — the
	// whole point of the consumer dual being more sensitive than the admit rung.
	rep := diagnose([]byte(strings.Repeat("loop ", 40)), defLimits())
	if rep.Findings != 1 {
		t.Fatalf("small loop: findings=%d, want 1 (%+v)", rep.Findings, rep.Recommendations)
	}
	if !rep.Shape.Degenerate {
		t.Fatalf("small loop should be shape-degenerate")
	}
	if rep.KernelWouldHold {
		t.Fatalf("a 200-byte loop is below the kernel oversize floor; it must not be kernel-held")
	}
	if sev := severityOf(rep, "answer-shape"); sev != sevWarn {
		t.Fatalf("answer-shape severity=%q, want warn", sev)
	}
	if sev := severityOf(rep, "kernel-admit"); sev != sevOK {
		t.Fatalf("kernel-admit severity=%q, want ok", sev)
	}
}

func TestDiagnoseKernelQuarantinesBlatantRepeat(t *testing.T) {
	// 16-byte chunk repeated 60× = 960 bytes: trips the context-MMU repeat-admit
	// rung (ctxmmu.ScreenBytes -> OVERSIZE), so BOTH checks warn.
	body := []byte(strings.Repeat("0123456789abcdef", 60))
	rep := diagnose(body, defLimits())
	if !rep.KernelWouldHold {
		t.Fatalf("blatant 960-byte repeat should be kernel-held; KernelAdmit=%q", rep.KernelAdmit)
	}
	if rep.Findings != 2 {
		t.Fatalf("blatant repeat: findings=%d, want 2 (%+v)", rep.Findings, rep.Recommendations)
	}
	if rep.KernelAdmit == "NONE" || rep.KernelAdmit == "" {
		t.Fatalf("expected a non-NONE kernel admit reason, got %q", rep.KernelAdmit)
	}
}

func severityOf(rep doctorReport, check string) string {
	for _, r := range rep.Recommendations {
		if r.Check == check {
			return r.Severity
		}
	}
	return ""
}

func runDoc(t *testing.T, stdin string, args ...string) (int, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := runDoctor(strings.NewReader(stdin), &out, &errb, args)
	return code, out.String()
}

func TestRunDoctorExitCodes(t *testing.T) {
	if code, out := runDoc(t, "A clean, in-shape answer that is plenty long enough to judge."); code != 0 {
		t.Fatalf("clean: exit=%d want 0\n%s", code, out)
	}
	if code, out := runDoc(t, strings.Repeat("abc", 80)); code != 1 {
		t.Fatalf("degenerate: exit=%d want 1\n%s", code, out)
	}
	if code, _ := runDoc(t, "x", "--bogus"); code != 2 {
		t.Fatalf("bad flag: exit=%d want 2", code)
	}
}

func TestRunDoctorJSON(t *testing.T) {
	code, out := runDoc(t, strings.Repeat("loop ", 40), "--json")
	if code != 1 {
		t.Fatalf("exit=%d want 1", code)
	}
	var rep doctorReport
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("not valid JSON: %v\n%s", err, out)
	}
	if len(rep.Recommendations) != 2 {
		t.Fatalf("want 2 recommendations (answer-shape + kernel-admit), got %d", len(rep.Recommendations))
	}
}

func TestWriteBinaryDoctorHumanShowsLiveDifferentProcess(t *testing.T) {
	rep := appversion.BinaryReport{
		Executable: "C:/work/fak/fak-hygiene.exe",
		Images: []appversion.BinaryImage{
			{Path: "C:/work/fak/fak-hygiene.exe", Exists: true, Current: true, Size: 10, SHA256: "111111111111abcd"},
			{Path: "C:/work/fak/fak.exe", Exists: true, Size: 10, SHA256: "222222222222abcd"},
		},
		Processes: []appversion.BinaryProcess{
			{PID: 123, Path: "C:/work/fak/fak.exe", SHA256: "222222222222abcd"},
		},
		Recommendations: []appversion.BinaryRecommendation{
			{Check: "binary-live-process", Severity: appversion.SeverityWarn, Finding: "1 live fak process is running a different sibling image"},
		},
		Findings: 1,
	}
	var out bytes.Buffer
	writeBinaryDoctorHuman(&out, rep)
	text := out.String()
	for _, want := range []string{"processes:", "pid=123", "different-from-current", "binary-live-process"} {
		if !strings.Contains(text, want) {
			t.Fatalf("binary doctor human output missing %q:\n%s", want, text)
		}
	}
}

func TestStampFreshnessRecommendation(t *testing.T) {
	const head = "abcdef1234567890"
	cases := []struct {
		name     string
		verdict  binstamp.Freshness
		cause    binstamp.Cause
		wantSev  string
		wantWord string // a distinctive word the finding must carry
	}{
		{"unstamped is the load-bearing warn", binstamp.Unknown, binstamp.CauseUnstamped, appversion.SeverityWarn, "UNVERIFIABLE"},
		{"stale warns and points at self-update", binstamp.Stale, binstamp.CauseDiverged, appversion.SeverityWarn, "newer fak exists"},
		{"fresh is ok", binstamp.Fresh, binstamp.CauseMatched, appversion.SeverityOK, "current"},
		{"dirty dev build is informational, not a finding", binstamp.Unknown, binstamp.CauseDirty, appversion.SeverityOK, "dev build"},
		{"no head is informational, not a finding", binstamp.Unknown, binstamp.CauseNoHead, appversion.SeverityOK, "no HEAD"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rec := stampFreshnessRecommendation(c.verdict, c.cause, "deadbeefcafe1234", head, "origin/main")
			if rec.Check != "binary-vcs-stamp" {
				t.Fatalf("check = %q, want binary-vcs-stamp", rec.Check)
			}
			if rec.Severity != c.wantSev {
				t.Fatalf("severity = %q, want %q (finding: %q)", rec.Severity, c.wantSev, rec.Finding)
			}
			if !strings.Contains(rec.Finding, c.wantWord) {
				t.Fatalf("finding %q missing distinctive word %q", rec.Finding, c.wantWord)
			}
			// The two warn causes must give the operator a next action.
			if c.wantSev == appversion.SeverityWarn && strings.TrimSpace(rec.Recommend) == "" {
				t.Fatalf("a WARN recommendation must carry an actionable Recommend (cause %v)", c.cause)
			}
		})
	}
}

func TestDoctorBinaryPositionalAliasReachesBinaryDiagnosis(t *testing.T) {
	var positionalOut, flagOut bytes.Buffer
	positionalRC := runDoctor(strings.NewReader(""), &positionalOut, &bytes.Buffer{}, []string{"binary", "--json"})
	flagRC := runDoctor(strings.NewReader(""), &flagOut, &bytes.Buffer{}, []string{"--binary", "--json"})
	if positionalRC != flagRC {
		t.Fatalf("positional rc=%d, flag rc=%d", positionalRC, flagRC)
	}
	var positional, flag appversion.BinaryReport
	if err := json.Unmarshal(positionalOut.Bytes(), &positional); err != nil {
		t.Fatalf("positional output is not binary report JSON: %v\n%s", err, positionalOut.String())
	}
	if err := json.Unmarshal(flagOut.Bytes(), &flag); err != nil {
		t.Fatalf("flag output is not binary report JSON: %v\n%s", err, flagOut.String())
	}
	if positional.Executable != flag.Executable || positional.Findings != flag.Findings {
		t.Fatalf("positional alias drifted: positional=%+v flag=%+v", positional, flag)
	}
}

func TestDoctorRejectsUnknownPositionalArgument(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if rc := runDoctor(strings.NewReader(""), &stdout, &stderr, []string{"binarry"}); rc != 2 {
		t.Fatalf("rc=%d, want usage error 2; stdout=%q stderr=%q", rc, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "unexpected argument") {
		t.Fatalf("missing positional rejection: %q", stderr.String())
	}
}
