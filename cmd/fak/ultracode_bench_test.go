package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ultracodebench"
)

func TestUltracodeBenchSelfcheckRender(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runUltracode(&out, &errOut, []string{"bench", "--selfcheck"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"ULTRACODE PAIRED BENCH: GAIN", "accepted effects: single=3 fleet=3", "billed tokens:", "selfcheck: PASS (offline fixture; not a live model-performance claim)"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
}

func TestUltracodeBenchJSON(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runUltracodeBench(&out, &errOut, []string{"--selfcheck", "--json"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var report ultracodebench.Report
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != ultracodebench.Schema || report.Verdict != "GAIN" {
		t.Fatalf("report=%+v", report)
	}
}

func TestUltracodeBenchRequiresOneInput(t *testing.T) {
	for _, args := range [][]string{nil, {"--selfcheck", "--pair", "x"}} {
		var out, errOut bytes.Buffer
		if code := runUltracodeBench(&out, &errOut, args); code != 2 {
			t.Fatalf("args=%v code=%d", args, code)
		}
		if !strings.Contains(errOut.String(), "choose exactly one") {
			t.Fatalf("stderr=%s", errOut.String())
		}
	}
}
