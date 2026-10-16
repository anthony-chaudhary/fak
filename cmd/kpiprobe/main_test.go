// fak-test:runtime fast est=1s
package main

import (
	"bytes"
	"fmt"
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
	var want bytes.Buffer
	fmt.Fprintf(&want, "# lru_hit_rate over %d accesses\n", rsiloop.TraceLen())
	// The advertised dump domain is sizes 1..16. Require every measured row,
	// including its order and formatting, rather than freezing the line count.
	for size := 1; size <= 16; size++ {
		fmt.Fprintf(&want, "size=%2d KPI=%.6f\n", size, rsiloop.HitRate(size))
	}
	if out.String() != want.String() {
		t.Fatalf("dump = %q, want %q", out.String(), want.String())
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
