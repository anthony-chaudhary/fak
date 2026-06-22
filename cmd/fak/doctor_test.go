package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/answershape"
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
