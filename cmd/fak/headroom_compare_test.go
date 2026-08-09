package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/headroom"
)

func TestHeadroomCompareNativeAndNoneJSON(t *testing.T) {
	var out, errb bytes.Buffer
	code := runHeadroom(&out, &errb, []string{"compare", "--via", "none,native", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	var report headroom.ComparisonReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if !report.Complete || len(report.Arms) != 2 {
		t.Fatalf("report=%+v", report)
	}
}

func TestHeadroomCompareReportsMissingLLMLingua(t *testing.T) {
	var out, errb bytes.Buffer
	code := runHeadroom(&out, &errb, []string{"compare"})
	if code != 3 {
		t.Fatalf("code=%d, want 3; out=%s err=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "MISSING lingua") {
		t.Fatalf("out=%s", out.String())
	}
}
