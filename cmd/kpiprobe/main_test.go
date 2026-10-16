// fak-test:runtime fast est=1s
package main

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

func TestRunDefaultPrintsCommittedSizeKPI(t *testing.T) {
	var out, errb bytes.Buffer
	if code := run(nil, &out, &errb); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, errb.String())
	}
	want := fmt.Sprintf("KPI=%.6f\n", rsiloop.HitRate(rsiloop.DefaultCacheSize))
	if out.String() != want {
		t.Fatalf("stdout = %q, want %q", out.String(), want)
	}
}

func TestRunSizeFlagMeasuresCandidate(t *testing.T) {
	var out bytes.Buffer
	if code := run([]string{"--size", "3"}, &out, &bytes.Buffer{}); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if want := fmt.Sprintf("KPI=%.6f\n", rsiloop.HitRate(3)); out.String() != want {
		t.Fatalf("stdout = %q, want %q", out.String(), want)
	}
}

func TestRunDumpPrintsSixteenPointCurve(t *testing.T) {
	var out bytes.Buffer
	if code := run([]string{"--dump"}, &out, &bytes.Buffer{}); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 17 || !strings.HasPrefix(lines[0], "# lru_hit_rate") {
		t.Fatalf("dump shape = %d lines, header %q", len(lines), lines[0])
	}
	if want := fmt.Sprintf("size=16 KPI=%.6f", rsiloop.HitRate(16)); lines[16] != want {
		t.Fatalf("last row = %q, want %q", lines[16], want)
	}
}

func TestRunBadFlagExitsUsage(t *testing.T) {
	var out bytes.Buffer
	if code := run([]string{"--size", "x"}, &out, &bytes.Buffer{}); code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	if out.Len() != 0 {
		t.Fatalf("stdout on usage error = %q, want empty", out.String())
	}
}
