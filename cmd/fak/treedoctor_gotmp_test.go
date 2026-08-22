package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/treedoctor"
)

func TestRenderGoTmpJSONKeepsStableMaintenanceSchema(t *testing.T) {
	report := treedoctor.GoTmpReport{
		Schema:           treedoctor.GoTmpSchema,
		Root:             "/repo/_scratch/go-tmp",
		DryRun:           true,
		ProcessSnapshots: 1,
		TotalBytes:       41,
		ReapedBytes:      37,
		Entries: []treedoctor.GoTmpEntry{
			{Name: "go-build-active", Bytes: 4, Reason: treedoctor.GoTmpReasonReferenced, ReferencedBy: []int{42}},
			{Name: "go-build-stale", Bytes: 37, Reason: treedoctor.GoTmpReasonEligible},
		},
	}
	var out bytes.Buffer
	if err := writeGoTmpJSON(&out, report); err != nil {
		t.Fatal(err)
	}
	var decoded treedoctor.GoTmpReport
	if err := json.Unmarshal(out.Bytes(), &decoded); err != nil {
		t.Fatalf("decode JSON: %v\n%s", err, out.String())
	}
	if decoded.Schema != treedoctor.GoTmpSchema || !decoded.DryRun || decoded.ProcessSnapshots != 1 {
		t.Fatalf("stable fields = %+v", decoded)
	}
	if len(decoded.Entries) != 2 || decoded.Entries[0].Reason != treedoctor.GoTmpReasonReferenced || decoded.ReapedBytes != 37 {
		t.Fatalf("entry/reclaim fields = %+v", decoded)
	}
}

func TestRenderGoTmpTextNamesKeptReasonAndReclaimedBytes(t *testing.T) {
	report := treedoctor.GoTmpReport{
		Schema:      treedoctor.GoTmpSchema,
		Root:        "/repo/_scratch/go-tmp",
		DryRun:      true,
		TotalBytes:  41,
		ReapedBytes: 37,
		Reaped:      []string{"/repo/_scratch/go-tmp/go-build-stale"},
		Entries: []treedoctor.GoTmpEntry{
			{Name: "go-build-active", Bytes: 4, Reason: treedoctor.GoTmpReasonReferenced, ReferencedBy: []int{42}, Verdict: treedoctor.GoTmpKeepLive},
			{Name: "go-build-stale", Bytes: 37, Reason: treedoctor.GoTmpReasonEligible, Verdict: treedoctor.GoTmpReap},
		},
	}
	var out bytes.Buffer
	writeGoTmpText(&out, report)
	got := out.String()
	for _, want := range []string{"would reap 1", "go-build-active: process-referenced", "PID(s) [42]", "go-build-stale: eligible", "37 bytes"} {
		if !strings.Contains(got, want) {
			t.Fatalf("text missing %q:\n%s", want, got)
		}
	}
}
