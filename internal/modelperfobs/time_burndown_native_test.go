package modelperfobs

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

func nativeBurndownFixture(t *testing.T, id string) (perfledger.Record, RequestPhaseReceipt) {
	t.Helper()
	obs, phase := bucketFixture(t, id, 28)
	row := perfledger.NewRecord(time.Unix(1000, 0), "stop", perfledger.LocalitySelfHosted, 10, 5, 90, 110*time.Millisecond, 30*time.Millisecond)
	row.RequestID, row.Model, row.Backend = id, obs.Model, "cpu"
	phase.Backend, phase.Receipt.Backend = row.Backend, row.Backend
	return row, phase
}

func writeNativeBurndownInput(t *testing.T, path string, rows ...any) {
	t.Helper()
	var b bytes.Buffer
	for _, row := range rows {
		if err := json.NewEncoder(&b).Encode(row); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(path, b.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
}

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=20ms
func TestTimeBurndownNativeLedger(t *testing.T) {
	// These CPU fixtures exercise the existing command, parser, strict join and
	// reducer. Their phase times are synthetic, not hardware measurements.
	a, pa := nativeBurndownFixture(t, "native-a")
	b, pb := nativeBurndownFixture(t, "native-b")
	b.Status = 200 // Both current legacy-zero and explicit success are supported.
	b.CachedTokens = 0
	failure := a
	failure.RequestID, failure.Model, failure.Error = "failed", "", "client_write_timeout"
	probe := a
	probe.RequestID, probe.Synthetic = "probe", true
	dir := t.TempDir()
	input, phases := filepath.Join(dir, "native.jsonl"), filepath.Join(dir, "phases.jsonl")
	writeNativeBurndownInput(t, input, a, b, failure, probe)
	writeNativeBurndownInput(t, phases, pa, pb)
	var out bytes.Buffer
	if err := RunTimeBurndown([]string{"--input", input, "--phases", phases, "--buckets", "--format", "json"}, &out); err != nil {
		t.Fatal(err)
	}
	var report TimeBurndownV2
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Requests != 4 || report.IncludedRequests != 2 || report.NativeRequests != 2 || report.FailedRequests != 1 || report.WarmupExcluded != 1 || len(report.Groups) != 2 || len(report.BucketGroups) != 1 {
		t.Fatalf("native join/exclusion coverage: %+v", report)
	}
	var g, failed TimeBurndownGroup
	for _, group := range report.Groups {
		if group.Model == a.Model {
			g = group
		} else if group.Model == "" {
			failed = group
		}
	}
	if failed.Requests != 1 || failed.FailedRequests != 1 || failed.Samples != 0 || failed.NativeRequests != 0 || failed.APIWallMS != 0 || failed.NativeWallMS != 0 {
		t.Fatalf("early failure lost its unknown model or contributed latency: %+v", failed)
	}
	buckets := report.BucketGroups[0]
	if g.CacheState != "unknown" || g.Backend != "cpu" || g.APIWallMS != 220 || g.NativeWallMS != 200 || g.OutsideNativeMS != 20 || buckets.BucketedMS != 220 {
		t.Fatalf("identity or wall reconciliation changed: group=%+v buckets=%+v", g, buckets)
	}
	if g.TTFTSamples != 0 || len(g.APIRank) != 1 || g.APIRank[0].Phase != "unpartitioned_request" || g.APIRank[0].WallMS != 220 {
		t.Fatalf("planner TTFT was promoted to observed streaming: %+v", g)
	}
	prefill, decode := stageTotal(buckets, StagePrefill), stageTotal(buckets, StageDecode)
	if prefill.Units != 20 || decode.Units != 10 || prefill.MSPerUnit == nil || *prefill.MSPerUnit != 4 || decode.MSPerUnit == nil || *decode.MSPerUnit != 9 {
		t.Fatalf("cached tokens changed native work denominators: prefill=%+v decode=%+v", prefill, decode)
	}

	t.Run("unreported-usage", func(t *testing.T) {
		a.UsageUnreported = true
		writeNativeBurndownInput(t, input, a)
		writeNativeBurndownInput(t, phases, pa)
		out.Reset()
		if err := RunTimeBurndown([]string{"--input", input, "--phases", phases, "--buckets", "--format", "json"}, &out); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(out.Bytes(), &report); err != nil {
			t.Fatal(err)
		}
		for _, stage := range report.BucketGroups[0].Stages {
			if stage.Stage != StageUnstaged && (stage.MSPerUnit != nil || stage.UnitsUnavailable == "") {
				t.Fatalf("unreported usage became a measured denominator: %+v", stage)
			}
		}
	})

	t.Run("observation-wire-unchanged", func(t *testing.T) {
		legacy := burndownObservation("legacy", "cold", 100, 30)
		writeNativeBurndownInput(t, input, legacy)
		out.Reset()
		if err := RunTimeBurndown([]string{"--input", input, "--format", "json"}, &out); err != nil {
			t.Fatal(err)
		}
		var old TimeBurndown
		if err := json.Unmarshal(out.Bytes(), &old); err != nil {
			t.Fatal(err)
		}
		if old.IncludedRequests != 1 || old.Groups[0].CacheState != "cold" || old.Groups[0].TTFTSamples != 1 {
			t.Fatalf("existing observation contract changed: %+v", old)
		}
	})
}

// fak-test:justify why=contract when=changed:internal/modelperfobs/**
// fak-test:runtime fast est=20ms
func TestTimeBurndownNativeRefusals(t *testing.T) {
	row, phase := nativeBurndownFixture(t, "native")
	for _, tc := range []struct {
		name string
		edit func(*perfledger.Record)
	}{
		{"missing-request", func(r *perfledger.Record) { r.RequestID = "" }},
		{"missing-model", func(r *perfledger.Record) { r.Model = "" }},
		{"missing-backend", func(r *perfledger.Record) { r.Backend = "" }},
		{"negative-tokens", func(r *perfledger.Record) { r.CachedTokens = -1 }},
		{"wrong-model", func(r *perfledger.Record) { r.Model = "different" }},
		{"wrong-backend", func(r *perfledger.Record) { r.Backend = "different" }},
		{"phase-exceeds-wall", func(r *perfledger.Record) { r.E2EMS = 90 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := row
			tc.edit(&bad)
			dir := t.TempDir()
			input, phases := filepath.Join(dir, "native.jsonl"), filepath.Join(dir, "phases.jsonl")
			writeNativeBurndownInput(t, input, bad)
			writeNativeBurndownInput(t, phases, phase)
			var out bytes.Buffer
			if err := RunTimeBurndown([]string{"--input", input, "--phases", phases, "--format", "json"}, &out); err == nil || out.Len() != 0 {
				t.Fatalf("invalid native evidence emitted report: err=%v out=%s", err, &out)
			}
		})
	}
	t.Run("duplicate-request", func(t *testing.T) {
		dir := t.TempDir()
		input := filepath.Join(dir, "native.jsonl")
		writeNativeBurndownInput(t, input, row, row)
		var out bytes.Buffer
		if err := RunTimeBurndown([]string{"--input", input, "--format", "json"}, &out); err == nil || out.Len() != 0 || !strings.Contains(err.Error(), "duplicate") {
			t.Fatalf("duplicate native identity admitted: err=%v out=%s", err, &out)
		}
	})
}
