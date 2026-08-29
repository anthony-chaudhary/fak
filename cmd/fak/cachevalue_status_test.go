package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/headroom"
	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
)

func TestCachevalueStatusJSONRollsUpOwnersFidelityAndDependencies(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	usage := filepath.Join(dir, "absent-usage.jsonl")
	withCachevalueStatusHeadroom(t, headroom.HeadroomName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", usage,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status --json exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.Schema != cachevalueStatusSchema || rep.Verdict != "PARTIAL" {
		t.Fatalf("unexpected status envelope: %+v", rep)
	}
	rows := cachevalueRowsByComponent(rep.Rows)

	provider := rows["provider_prompt_cache"]
	if provider.Owner != "provider" || provider.Fidelity != "lossless" || provider.Status != "measured" {
		t.Fatalf("provider_prompt_cache row = %+v", provider)
	}
	head := rows["headroom_plugin:headroom"]
	if !head.Selected || head.Owner != "external" || head.Dependency != "external_http_sidecar" ||
		head.Fidelity != "recoverable" || head.Status != "unavailable" ||
		!strings.Contains(head.SessionImpact, "not fak core") {
		t.Fatalf("headroom external row = %+v", head)
	}
	actions := rows["provider_actions"]
	if actions.Dependency != "provider_action_transport" || actions.Fidelity != "lossless" || actions.Status != "gated" {
		t.Fatalf("provider_actions row = %+v", actions)
	}
	ablate := rows["cache_ablation_runner"]
	if ablate.Dependency != "subprocess_reexec" || ablate.Status != "available" ||
		!strings.Contains(ablate.NextAction, "fak ablate --sweep") {
		t.Fatalf("cache ablation row = %+v", ablate)
	}
	if rep.Headroom.Selected != headroom.HeadroomName || rep.Headroom.HeadroomReachable {
		t.Fatalf("headroom digest = %+v", rep.Headroom)
	}
	if rep.VCache.ProviderActionTransport != "decision_only" {
		t.Fatalf("vcache digest = %+v", rep.VCache)
	}
	if rep.Attribution.Owners["external"].Problem == 0 ||
		rep.Attribution.Owners["provider"].Total == 0 ||
		rep.Attribution.Fidelities["lossless"] == 0 ||
		rep.Attribution.Fidelities["lossy"] == 0 {
		t.Fatalf("attribution missing owner/fidelity rollup: %+v", rep.Attribution)
	}
	if findingByKey(rep.Attribution.ProblemOwners, "external") == nil ||
		findingByKey(rep.Attribution.ProblemDomains, "provider_transport") == nil {
		t.Fatalf("attribution missing problem owner/domain findings: %+v", rep.Attribution)
	}
	if len(rep.NextActions) == 0 {
		t.Fatalf("expected next actions for partial status: %+v", rep)
	}
}

func TestCachevalueStatusSurfacesRejectedTierAccesses(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "cache-value.jsonl")
	args := []string{
		"--ledger", ledger,
		"--savings-ledger", filepath.Join(dir, "absent-savings.jsonl"),
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
	}
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	rowAt := time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)
	rowWithRejected := func(rejected uint64) cachevalueledger.Row {
		return cachevalueledger.NewSessionRow("guard", "active-track-1", "issue-10151", cacheobs.Stats{
			Turns:                10,
			PromptTokens:         1000,
			ReusedTokens:         700,
			RejectedTierAccesses: rejected,
			FrozenTurns:          7,
			PartialTurns:         2,
			ColdTurns:            1,
			ReuseRatio:           0.7,
		}, rowAt)
	}
	writeRow := func(row cachevalueledger.Row) {
		t.Helper()
		line, err := cachevalueledger.AppendLedgerLine(row)
		if err != nil {
			t.Fatalf("encode Track-1 row: %v", err)
		}
		if err := os.WriteFile(ledger, []byte(line+"\n"), 0o600); err != nil {
			t.Fatalf("write Track-1 ledger: %v", err)
		}
	}
	runJSON := func(row cachevalueledger.Row) (string, cachevalueStatusReport) {
		t.Helper()
		writeRow(row)
		var out, errb bytes.Buffer
		if code := runCachevalueStatus(&out, &errb, append(append([]string{}, args...), "--json")); code != 0 {
			t.Fatalf("cachevalue status --json exit=%d stderr=%s", code, errb.String())
		}
		var rep cachevalueStatusReport
		if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
			t.Fatalf("decode cachevalue status JSON: %v\n%s", err, out.String())
		}
		return out.String(), rep
	}
	runHuman := func(row cachevalueledger.Row) string {
		t.Helper()
		writeRow(row)
		var out, errb bytes.Buffer
		if code := runCachevalueStatus(&out, &errb, args); code != 0 {
			t.Fatalf("cachevalue status exit=%d stderr=%s", code, errb.String())
		}
		return out.String()
	}

	baselineRow := rowWithRejected(0)
	rejectedRow := rowWithRejected(7)
	baselineRaw, baseline := runJSON(baselineRow)
	rejectedRaw, rejected := runJSON(rejectedRow)
	if !strings.Contains(baselineRaw, `"rejected_tier_accesses": 0`) {
		t.Fatalf("zero-rejection JSON omitted always-present field:\n%s", baselineRaw)
	}
	if rejected.Value.RejectedTierAccesses != 7 || !strings.Contains(rejectedRaw, `"rejected_tier_accesses": 7`) {
		t.Fatalf("rejected-tier JSON value=%d, want 7:\n%s", rejected.Value.RejectedTierAccesses, rejectedRaw)
	}

	baselineValue, rejectedValue := baseline.Value, rejected.Value
	baselineValue.RejectedTierAccesses = 0
	rejectedValue.RejectedTierAccesses = 0
	if !reflect.DeepEqual(rejectedValue, baselineValue) {
		t.Fatalf("rejections changed accepted value digest:\nbaseline=%+v\nrejected=%+v", baselineValue, rejectedValue)
	}
	if rejected.Verdict != baseline.Verdict || rejected.Summary != baseline.Summary ||
		!reflect.DeepEqual(rejected.Counts, baseline.Counts) || !reflect.DeepEqual(rejected.Rows, baseline.Rows) ||
		!reflect.DeepEqual(rejected.NextActions, baseline.NextActions) {
		t.Fatalf("rejections changed overall status outputs:\nbaseline verdict=%s summary=%q counts=%v rows=%v actions=%v\nrejected verdict=%s summary=%q counts=%v rows=%v actions=%v",
			baseline.Verdict, baseline.Summary, baseline.Counts, baseline.Rows, baseline.NextActions,
			rejected.Verdict, rejected.Summary, rejected.Counts, rejected.Rows, rejected.NextActions)
	}

	baselineTrack1 := cachevaluereport.Fold([]cachevalueledger.Row{baselineRow}, rowAt)
	rejectedTrack1 := cachevaluereport.Fold([]cachevalueledger.Row{rejectedRow}, rowAt)
	if baselineTrack1.TotalRows != rejectedTrack1.TotalRows || baselineTrack1.TotalSessions != rejectedTrack1.TotalSessions ||
		baselineTrack1.MultiTurnSessions != rejectedTrack1.MultiTurnSessions || baselineTrack1.LatestReuseRatio != rejectedTrack1.LatestReuseRatio ||
		baselineTrack1.Verdict != rejectedTrack1.Verdict || baselineTrack1.NextAction != rejectedTrack1.NextAction ||
		baselineTrack1.Buckets[0].Turns != rejectedTrack1.Buckets[0].Turns ||
		baselineTrack1.Buckets[0].PromptTokens != rejectedTrack1.Buckets[0].PromptTokens ||
		baselineTrack1.Buckets[0].ReusedTokens != rejectedTrack1.Buckets[0].ReusedTokens ||
		baselineTrack1.Buckets[0].RealizedReuseRatio != rejectedTrack1.Buckets[0].RealizedReuseRatio {
		t.Fatalf("rejections changed accepted Track-1 counters, ratio, verdict, or next action:\nbaseline=%+v\nrejected=%+v", baselineTrack1, rejectedTrack1)
	}

	baselineHuman := runHuman(baselineRow)
	if strings.Contains(baselineHuman, "rejected_tier_accesses") {
		t.Fatalf("zero-rejection human output named rejected tier accesses:\n%s", baselineHuman)
	}
	rejectedHuman := runHuman(rejectedRow)
	if !strings.Contains(rejectedHuman, "value: rejected_tier_accesses=7") {
		t.Fatalf("human output missing rejected tier accesses:\n%s", rejectedHuman)
	}
}

func TestCachevalueStatusRejectedTierAccessesEdgeAndAdversarial(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "cache-value.jsonl")
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	rowAt := time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)
	row := func(rejected uint64) cachevalueledger.Row {
		return cachevalueledger.NewSessionRow("guard", "issue-10153", "edge-adversarial", cacheobs.Stats{
			Turns:                10,
			PromptTokens:         1000,
			ReusedTokens:         700,
			RejectedTierAccesses: rejected,
			FrozenTurns:          7,
			PartialTurns:         2,
			ColdTurns:            1,
			ReuseRatio:           0.7,
		}, rowAt)
	}
	line := func(r cachevalueledger.Row) string {
		t.Helper()
		encoded, err := cachevalueledger.AppendLedgerLine(r)
		if err != nil {
			t.Fatalf("encode Track-1 row: %v", err)
		}
		return encoded
	}

	tests := []struct {
		name        string
		ledgerBody  func() string
		wantRejects uint64
		wantHuman   bool
	}{
		{name: "empty", ledgerBody: func() string { return "" }},
		{name: "malformed row ignored", ledgerBody: func() string { return "not-json\n" + line(row(7)) + "\n" }, wantRejects: 7, wantHuman: true},
		{name: "hostile strings stay data", ledgerBody: func() string {
			r := row(7)
			r.Context = "</json>\nrejected_tier_accesses=999999"
			r.SessionID = "../../hostile-session"
			return line(r) + "\n"
		}, wantRejects: 7, wantHuman: true},
		{name: "maximum counter remains exact", ledgerBody: func() string { return line(row(^uint64(0))) + "\n" }, wantRejects: ^uint64(0), wantHuman: true},
		{name: "aggregate overflow saturates", ledgerBody: func() string {
			return line(row(^uint64(0))) + "\n" + line(row(1)) + "\n"
		}, wantRejects: ^uint64(0), wantHuman: true},
		{name: "oversized prefix keeps tail witness", ledgerBody: func() string {
			return strings.Repeat("not-json\n", 150000) + line(row(7)) + "\n"
		}, wantRejects: 7, wantHuman: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := os.WriteFile(ledger, []byte(tt.ledgerBody()), 0o600); err != nil {
				t.Fatalf("write Track-1 ledger: %v", err)
			}
			args := []string{
				"--ledger", ledger,
				"--savings-ledger", filepath.Join(dir, "absent-savings.jsonl"),
				"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
			}

			var jsonOut, jsonErr bytes.Buffer
			if code := runCachevalueStatus(&jsonOut, &jsonErr, append(append([]string{}, args...), "--json")); code != 0 {
				t.Fatalf("cachevalue status --json exit=%d stderr=%s", code, jsonErr.String())
			}
			var rep cachevalueStatusReport
			if err := json.Unmarshal(jsonOut.Bytes(), &rep); err != nil {
				t.Fatalf("decode cachevalue status JSON: %v\n%s", err, jsonOut.String())
			}
			if rep.Value.RejectedTierAccesses != tt.wantRejects {
				t.Fatalf("rejected tier accesses=%d, want %d\n%s", rep.Value.RejectedTierAccesses, tt.wantRejects, jsonOut.String())
			}
			if !strings.Contains(jsonOut.String(), `"rejected_tier_accesses":`) {
				t.Fatalf("JSON omitted rejected_tier_accesses:\n%s", jsonOut.String())
			}

			var humanOut, humanErr bytes.Buffer
			if code := runCachevalueStatus(&humanOut, &humanErr, args); code != 0 {
				t.Fatalf("cachevalue status exit=%d stderr=%s", code, humanErr.String())
			}
			gotHuman := strings.Contains(humanOut.String(), "value: rejected_tier_accesses=")
			if gotHuman != tt.wantHuman {
				t.Fatalf("human rejected-tier visibility=%v, want %v:\n%s", gotHuman, tt.wantHuman, humanOut.String())
			}
		})
	}
}
func TestCachevalueStatusTextNamesTroubleshootingAxes(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	withCachevalueStatusHeadroom(t, headroom.HeadroomName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
	})
	if code != 0 {
		t.Fatalf("status exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"cachevalue status: PARTIAL",
		"cache-plane rollup",
		"owner",
		"fidelity",
		"external_http_sidecar",
		"lossless",
		"lossy",
		"subprocess_reexec",
		"attribution:",
		"problem owners=",
		"problem domains=",
		"problem fidelity=",
		"evidence=",
		"next actions",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("status text missing %q:\n%s", want, got)
		}
	}
}

func TestCachevalueStatusSessionDiagnosisAttributesBadSession(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	session := filepath.Join(dir, "bad-session.jsonl")
	body := strings.Join([]string{
		`{"type":"assistant","timestamp":"2026-07-04T10:00:00Z","message":{"id":"m1","model":"claude-opus-4-8-20260101","usage":{"input_tokens":120000,"cache_creation_input_tokens":120000,"output_tokens":500},"content":[]}}`,
		`{"type":"assistant","timestamp":"2026-07-04T10:05:00Z","message":{"id":"m2","model":"claude-opus-4-8-20260101","usage":{"input_tokens":160000,"cache_creation_input_tokens":160000,"output_tokens":500},"content":[]}}`,
		"",
	}, "\n")
	if err := os.WriteFile(session, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--session", session,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status --session --json exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.Session == nil || rep.Session.LikelyDomain != "provider_prompt_cache" || rep.Session.TotalContextTokens != 560000 {
		t.Fatalf("session digest = %+v", rep.Session)
	}
	if rep.VCacheObserve == nil || rep.VCacheObserve.Status != "cold_write_only" ||
		rep.VCacheObserve.Turns != 2 ||
		!strings.HasPrefix(rep.VCacheObserve.Path, "session:") {
		t.Fatalf("session-derived vcache observe digest = %+v", rep.VCacheObserve)
	}
	if findingByKey(rep.Attribution.ProblemDomains, "provider_cache_window") == nil ||
		findingByKey(rep.Attribution.ProblemDomains, "workload") == nil ||
		findingByKey(rep.Attribution.ProblemOwners, "workload") == nil {
		t.Fatalf("session attribution missing provider/workload problem domains: %+v", rep.Attribution)
	}
	rows := cachevalueRowsByComponent(rep.Rows)
	provider := rows["session_provider_cache"]
	if provider.Owner != "provider" || provider.Fidelity != "lossless" ||
		provider.Status != "cold_write_only" || provider.FailureDomain != "provider_cache_window" {
		t.Fatalf("session provider row = %+v", provider)
	}
	workload := rows["session_context_pressure"]
	if workload.Owner != "workload" || workload.Status != "high_pressure" ||
		!strings.Contains(workload.SessionImpact, "not by itself a fak cache fault") {
		t.Fatalf("session workload row = %+v", workload)
	}
	fakCtx := rows["session_fak_context_events"]
	if fakCtx.Owner != "fak" || fakCtx.Fidelity != "lossy" ||
		fakCtx.Status != "not_observed_from_transcript" ||
		!strings.Contains(fakCtx.SessionImpact, "before blaming fak context planning") {
		t.Fatalf("session fak-context row = %+v", fakCtx)
	}
	observe := rows["vcache_observe_report"]
	if observe.Owner != "provider" || observe.Status != "cold_write_only" ||
		observe.Fidelity != "lossless" ||
		!strings.Contains(observe.Reason, "source=session:") {
		t.Fatalf("session-derived observe row = %+v", observe)
	}
	observeFamily := rows["vcache_observe_family:bad-session"]
	if observeFamily.Owner != "provider" || observeFamily.Status != "cold_write_only" ||
		observeFamily.FailureDomain != "provider_cache_window" {
		t.Fatalf("session-derived observe family row = %+v", observeFamily)
	}

	out.Reset()
	errb.Reset()
	code = runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--session", session,
	})
	if code != 0 {
		t.Fatalf("status --session text exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"session:", "vcache observe: status=cold_write_only", "path=session:", "likely=provider_prompt_cache", "cold_write_only", "high_pressure", "not_observed_from_transcript"} {
		if !strings.Contains(got, want) {
			t.Fatalf("session status text missing %q:\n%s", want, got)
		}
	}
}

func TestCachevalueStatusFoldsAblationReportDropsAndCacheEffects(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	ablationPath := filepath.Join(dir, "ablation.json")
	exitCode := 17
	ablationReport := ablate.Report{
		WorkloadHash: "hash-123",
		Baseline:     "all-off",
		Runs: []ablate.AblationRun{{
			ArmID:        "vdso",
			WorkloadHash: "hash-123",
			CacheEffects: []ablate.CacheEffect{{
				Feature:    "vdso",
				Owner:      "fak",
				Plane:      "kernel_tool_cache",
				Component:  "vdso",
				Dependency: "in_process",
				Fidelity:   "lossless",
				Evidence:   "witnessed",
				Status:     "active",
				Reason:     "vDSO hits were observed",
			}},
		}},
		Dropped: []ablate.DroppedArm{{
			ArmID:           "radix",
			Reason:          "child failed: exit 17",
			Stage:           "child_exit",
			ExitCode:        &exitCode,
			DurationSeconds: 0.125,
			StderrTail:      "radix child stderr tail",
		}},
	}
	raw, err := json.Marshal(ablationReport)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ablationPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--ablation-report", ablationPath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status --ablation-report --json exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.Ablation == nil || rep.Ablation.Status != "partial" || rep.Ablation.DroppedArms != 1 ||
		rep.Ablation.DroppedWithDiagnostics != 1 || rep.Ablation.DroppedChildExits != 1 ||
		rep.Ablation.CacheEffects != 1 || rep.Ablation.ActiveEffects != 1 {
		t.Fatalf("ablation digest = %+v", rep.Ablation)
	}
	rows := cachevalueRowsByComponent(rep.Rows)
	effect := rows["ablation_effect:vdso:vdso"]
	if effect.Owner != "fak" || effect.Dependency != "in_process" || effect.Fidelity != "lossless" ||
		effect.Status != "active" || !strings.Contains(effect.SessionImpact, "fak-owned") {
		t.Fatalf("ablation effect row = %+v", effect)
	}
	dropped := rows["ablation_dropped_arm:radix"]
	if dropped.Status != "dropped" || dropped.Dependency != "subprocess_reexec" ||
		dropped.FailureDomain != "fak_diagnostics_subprocess_exit" ||
		!strings.Contains(dropped.Reason, "stage=child_exit") ||
		!strings.Contains(dropped.Reason, "exit=17") ||
		!strings.Contains(dropped.Reason, "stderr=radix child stderr tail") {
		t.Fatalf("dropped arm row = %+v", dropped)
	}

	out.Reset()
	errb.Reset()
	code = runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--ablation-report", ablationPath,
	})
	if code != 0 {
		t.Fatalf("status --ablation-report text exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"ablation: status=partial", "child_exit=1", "ablation_effect:vdso:vdso", "ablation_dropped_arm:radix", "stage=child_exit", "subprocess_reexec"} {
		if !strings.Contains(got, want) {
			t.Fatalf("ablation status text missing %q:\n%s", want, got)
		}
	}
}

func TestCachevalueStatusFoldsHeadroomBenchReport(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	benchPath := filepath.Join(dir, "headroom-bench.json")
	benchReport := headroom.BenchReport{
		Compressor: headroom.NativeName,
		Samples: []headroom.BenchSample{
			{Name: "pretty-json", Kind: "json", Codec: "json-min", OrigLen: 1000, NewLen: 400, Saved: 0.6},
			{Name: "plain-prose", Kind: "text", Codec: "(none)", OrigLen: 500, NewLen: 500, Saved: 0},
		},
		OrigTotal: 1500,
		NewTotal:  900,
		Saved:     0.4,
	}
	raw, err := json.Marshal(benchReport)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(benchPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--headroom-bench-report", benchPath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status --headroom-bench-report --json exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.HeadroomBench == nil || rep.HeadroomBench.Status != "measured" ||
		rep.HeadroomBench.Compressor != headroom.NativeName || rep.HeadroomBench.SavedRatio != 0.4 {
		t.Fatalf("headroom bench digest = %+v", rep.HeadroomBench)
	}
	rows := cachevalueRowsByComponent(rep.Rows)
	aggregate := rows["headroom_bench_report"]
	if aggregate.Owner != "fak" || aggregate.Dependency != "in_process" ||
		aggregate.Fidelity != "recoverable" || aggregate.Status != "measured" {
		t.Fatalf("headroom bench aggregate row = %+v", aggregate)
	}
	saved := rows["headroom_bench_sample:pretty-json"]
	if saved.Status != "saved" || saved.Owner != "fak" {
		t.Fatalf("saved sample row = %+v", saved)
	}
	noEffect := rows["headroom_bench_sample:plain-prose"]
	if noEffect.Status != "no_effect" || !strings.Contains(noEffect.SessionImpact, "aggregate status decides") {
		t.Fatalf("no-effect sample row = %+v", noEffect)
	}

	out.Reset()
	errb.Reset()
	code = runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--headroom-bench-report", benchPath,
	})
	if code != 0 {
		t.Fatalf("status --headroom-bench-report text exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"headroom bench: status=measured", "headroom_bench_report", "headroom_bench_sample:pretty-json", "no_effect"} {
		if !strings.Contains(got, want) {
			t.Fatalf("headroom bench status text missing %q:\n%s", want, got)
		}
	}
}

func TestCachevalueStatusHeadroomBenchNoSavingIsActionable(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	benchPath := filepath.Join(dir, "noop-bench.json")
	benchReport := headroom.BenchReport{
		Compressor: headroom.NativeName,
		Samples:    []headroom.BenchSample{{Name: "plain", Kind: "text", Codec: "(none)", OrigLen: 500, NewLen: 500, Saved: 0}},
		OrigTotal:  500,
		NewTotal:   500,
		Saved:      0,
	}
	raw, err := json.Marshal(benchReport)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(benchPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--headroom-bench-report", benchPath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status no-saving headroom bench exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	aggregate := cachevalueRowsByComponent(rep.Rows)["headroom_bench_report"]
	if aggregate.Status != "no_saving" || !strings.Contains(aggregate.NextAction, "representative") {
		t.Fatalf("no-saving aggregate row = %+v", aggregate)
	}
}

func TestCachevalueStatusHeadroomBenchUnavailableBlamesExternalDependency(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	benchPath := filepath.Join(dir, "headroom-unavailable.json")
	benchReport := headroom.BenchReport{
		Compressor: headroom.HeadroomName,
		Owner:      "external",
		Dependency: "external_http_sidecar",
		Fidelity:   "recoverable",
		Evidence:   "observed",
		Status:     "unavailable",
		Reason:     "all samples passed through because the selected compressor dependency was unavailable",
		Samples: []headroom.BenchSample{{
			Name:    "sample",
			Kind:    "text",
			Status:  "unavailable",
			Codec:   "(none)",
			OrigLen: 100,
			NewLen:  100,
			Saved:   0,
			Reason:  "headroom sidecar returned HTTP 503",
		}},
		OrigTotal: 100,
		NewTotal:  100,
		Saved:     0,
	}
	raw, err := json.Marshal(benchReport)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(benchPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--headroom-bench-report", benchPath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status unavailable headroom bench exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.HeadroomBench == nil || rep.HeadroomBench.Status != "unavailable" ||
		rep.HeadroomBench.Owner != "external" || rep.HeadroomBench.Dependency != "external_http_sidecar" {
		t.Fatalf("headroom unavailable digest = %+v", rep.HeadroomBench)
	}
	rows := cachevalueRowsByComponent(rep.Rows)
	aggregate := rows["headroom_bench_report"]
	if aggregate.Status != "unavailable" || aggregate.FailureDomain != "external:headroom_unavailable" ||
		!strings.Contains(aggregate.NextAction, "start headroom proxy") {
		t.Fatalf("unavailable aggregate row = %+v", aggregate)
	}
	sample := rows["headroom_bench_sample:sample"]
	if sample.Status != "unavailable" || sample.Owner != "external" ||
		!strings.Contains(sample.Reason, "HTTP 503") {
		t.Fatalf("unavailable sample row = %+v", sample)
	}
}

func TestCachevalueStatusFoldsVCacheScoreReportPlanes(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	scorePath := filepath.Join(dir, "vcache-score.json")
	scoreReport := vcachescore.Report{
		Schema:           "fak.vcache.score.v1",
		Status:           "2x_ready",
		Grade:            "A",
		Score:            91,
		ActiveSource:     "telemetry",
		ActiveMultiplier: 2.4,
		TwoXBetter:       true,
		Planes: vcachescore.PlaneReport{
			ProviderObserved: vcachescore.PlaneValueReport{
				Available:          true,
				Provenance:         "OBSERVED",
				Multiplier:         2.4,
				SavedTokenEquiv:    1400,
				BaselineTokenEquiv: 2400,
				CostTokenEquiv:     1000,
				Reason:             "provider telemetry supplied",
			},
			KernelWitnessed: vcachescore.PlaneValueReport{
				Available:          true,
				Provenance:         "WITNESSED",
				Multiplier:         1.5,
				SavedTokenEquiv:    500,
				BaselineTokenEquiv: 1500,
				CostTokenEquiv:     1000,
				Reason:             "cachevalue ledger witness supplied",
			},
			ContextWitnessed: vcachescore.PlaneValueReport{
				Available:       true,
				Provenance:      "WITNESSED",
				SavedTokenEquiv: 600,
				Reason:          "context shed-token witness supplied",
			},
			ExternalEngineObserved: vcachescore.PlaneValueReport{
				Available:  true,
				Provenance: "OBSERVED",
				HitRate:    0.8,
				Reason:     "external prefix cache supplied",
			},
			Forecast: vcachescore.PlaneValueReport{
				Available:          true,
				Provenance:         "FORECAST",
				Multiplier:         4,
				SavedTokenEquiv:    6000,
				BaselineTokenEquiv: 8000,
				CostTokenEquiv:     2000,
				Reason:             "planned star workload",
			},
		},
		AgenticActivation: vcachescore.AgenticActivationReport{Total: 2, Active: true, KernelKVEvents: 1, ContextEvents: 1},
		DefaultUsefulness: vcachescore.DefaultUsefulnessReport{Verdict: "default_ready"},
	}
	raw, err := json.Marshal(scoreReport)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(scorePath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--vcache-score-report", scorePath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status --vcache-score-report --json exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.VCacheScore == nil || rep.VCacheScore.Status != "2x_ready" ||
		rep.VCacheScore.ProviderObserved != "OBSERVED" || !rep.VCacheScore.AgenticActivation {
		t.Fatalf("vcache score digest = %+v", rep.VCacheScore)
	}
	rows := cachevalueRowsByComponent(rep.Rows)
	aggregate := rows["vcache_score_report"]
	if aggregate.Status != "measured" || aggregate.Evidence != "WITNESSED" ||
		!strings.Contains(aggregate.SessionImpact, "provider rebates") {
		t.Fatalf("vcache score aggregate row = %+v", aggregate)
	}
	provider := rows["vcache_score_provider_observed"]
	if provider.Owner != "provider" || provider.Dependency != "provider_usage_snapshot" ||
		provider.Fidelity != "lossless" || provider.Status != "observed" {
		t.Fatalf("provider plane row = %+v", provider)
	}
	kernel := rows["vcache_score_kernel_witnessed"]
	if kernel.Owner != "fak" || kernel.Dependency != "cachevalue_ledger" ||
		kernel.Fidelity != "lossless" || kernel.Status != "measured" {
		t.Fatalf("kernel plane row = %+v", kernel)
	}
	context := rows["vcache_score_context_witnessed"]
	if context.Owner != "fak" || context.Fidelity != "lossy" ||
		context.Evidence != "WITNESSED" || context.Status != "measured" {
		t.Fatalf("context plane row = %+v", context)
	}
	external := rows["vcache_score_external_engine_observed"]
	if external.Owner != "external" || external.Dependency != "external_engine_snapshot" ||
		external.Status != "observed" {
		t.Fatalf("external plane row = %+v", external)
	}
	forecast := rows["vcache_score_forecast"]
	if forecast.Owner != "fak" || forecast.Fidelity != "forecast" ||
		forecast.Evidence != "FORECAST" || forecast.Status != "forecast" {
		t.Fatalf("forecast plane row = %+v", forecast)
	}

	out.Reset()
	errb.Reset()
	code = runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--vcache-score-report", scorePath,
	})
	if code != 0 {
		t.Fatalf("status --vcache-score-report text exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"vcache score:", "vcache_score_provider_observed", "vcache_score_context_witnessed", "vcache_score_external_engine_observed", "forecast"} {
		if !strings.Contains(got, want) {
			t.Fatalf("vcache score status text missing %q:\n%s", want, got)
		}
	}
}

func TestCachevalueStatusFoldsVCacheObserveReportTelemetry(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	observePath := filepath.Join(dir, "vcache-observe.json")
	report := vcacheobserve.Report{
		Schema:      vcacheobserve.Schema,
		Turns:       5,
		FamilyCount: 2,
		Aggregate: vcachegov.TelemetrySavingsProof{
			CacheReadTokens:     40_000,
			CacheCreationTokens: 84_000,
			SavedTokenEquiv:     12_000,
			SavedPct:            35,
		},
		HitRate:    0.42,
		Multiplier: 1.8,
		Prediction: vcachecal.PredictionError{Total: 5, TrueWarm: 2, FalseWarm: 1, TrueCold: 2},
		OwnerSlices: []vcacheobserve.OwnerSlice{
			{
				Owner:               "provider",
				Mechanism:           "prompt_cache",
				SavedTokenEquiv:     12_000,
				CacheReadTokens:     40_000,
				CacheCreationTokens: 84_000,
				Provenance:          vcacheobserve.Observed,
				Evidence:            "provider usage counters",
			},
			{
				Owner:           "fak",
				Mechanism:       "authored_cache",
				SavedTokenEquiv: 0,
				Provenance:      vcacheobserve.NotObserved,
				Evidence:        "provider telemetry has no fak KV/context counters",
			},
		},
		Families: []vcacheobserve.Family{
			{
				Key:                 "alpha",
				Turns:               3,
				CacheReadTokens:     40_000,
				CacheCreationTokens: 42_000,
				HitRate:             0.5,
				GovernorDecision:    vcachegov.DecisionRideNatural,
				Economics:           vcachegov.TelemetrySavingsProof{SavedTokenEquiv: 10_000},
				Prediction:          vcachecal.PredictionError{Total: 3, TrueWarm: 2, TrueCold: 1},
			},
			{
				Key:                 "beta",
				Turns:               2,
				CacheCreationTokens: 42_000,
				HitRate:             0,
				GovernorDecision:    vcachegov.DecisionHeartbeatPin,
				Economics:           vcachegov.TelemetrySavingsProof{SavedTokenEquiv: -500},
				Prediction:          vcachecal.PredictionError{Total: 2, FalseWarm: 1, TrueCold: 1},
			},
		},
	}
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(observePath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--vcache-observe-report", observePath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status --vcache-observe-report --json exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.VCacheObserve == nil || rep.VCacheObserve.Status != "false_warm" ||
		rep.VCacheObserve.FailureDomain != "provider_cache_prediction" ||
		rep.VCacheObserve.FalseWarm != 1 ||
		rep.VCacheObserve.CacheReadTokens != 40_000 {
		t.Fatalf("vcache observe digest = %+v", rep.VCacheObserve)
	}
	rows := cachevalueRowsByComponent(rep.Rows)
	aggregate := rows["vcache_observe_report"]
	if aggregate.Status != "false_warm" || aggregate.Owner != "provider" ||
		aggregate.Fidelity != "lossless" || !strings.Contains(aggregate.NextAction, "context-join") {
		t.Fatalf("observe aggregate row = %+v", aggregate)
	}
	provider := rows["vcache_observe_owner:provider:prompt_cache"]
	if provider.Status != "observed" || provider.Owner != "provider" ||
		provider.Dependency != "provider_usage_fields" {
		t.Fatalf("observe provider owner row = %+v", provider)
	}
	fak := rows["vcache_observe_owner:fak:authored_cache"]
	if fak.Status != "not_observed" || fak.Owner != "fak" ||
		fak.FailureDomain != "evidence_gap" ||
		!strings.Contains(fak.NextAction, "fak-owned witnesses") {
		t.Fatalf("observe fak owner row = %+v", fak)
	}
	family := rows["vcache_observe_family:beta"]
	if family.Status != "false_warm" || family.FailureDomain != "provider_cache_prediction" ||
		!strings.Contains(family.Reason, "false_warm=1") {
		t.Fatalf("observe family row = %+v", family)
	}
	if findingByKey(rep.Attribution.ProblemDomains, "provider_cache_prediction") == nil ||
		findingByKey(rep.Attribution.ProblemDomains, "evidence_gap") == nil {
		t.Fatalf("observe attribution missing prediction/evidence findings: %+v", rep.Attribution)
	}

	out.Reset()
	errb.Reset()
	code = runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--vcache-observe-report", observePath,
	})
	if code != 0 {
		t.Fatalf("status --vcache-observe-report text exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"vcache observe: status=false_warm", "provider_cache_prediction", "vcache_observe_family:beta", "not_observed", "false_warm=1"} {
		if !strings.Contains(got, want) {
			t.Fatalf("vcache observe status text missing %q:\n%s", want, got)
		}
	}
}

func TestCachevalueStatusFoldsVCacheActionsReportTransportGates(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	actionsPath := filepath.Join(dir, "vcache-actions.json")
	plan := vcacheobserve.ProviderActionPlan{
		Schema:         vcacheobserve.ProviderActionSchema,
		Turns:          4,
		FamilyCount:    2,
		Counts:         vcacheobserve.ProviderActionCounts{Ready: 1, Gated: 1},
		Transport:      vcacheobserve.ProviderActionTransport{Mode: "decision_only", Ready: false, Reason: "provider transport witness missing"},
		CorrectnessLaw: "full uncached prompt remains the correctness path",
		Actions: []vcacheobserve.ProviderAction{
			{
				Family:          "bursty",
				Decision:        vcachegov.DecisionHeartbeatPin,
				Action:          "heartbeat_pin",
				State:           vcacheobserve.ActionGated,
				Requires:        []string{"heartbeat_transport", "byte_identical_prefix"},
				Witnessed:       []string{"heartbeat_transport"},
				Reason:          "pin candidate needs active provider warm transport plus byte-identical prefix fingerprint before spending",
				Turns:           2,
				SavedTokenEquiv: 1200,
			},
			{
				Family:          "secret",
				Decision:        vcachegov.DecisionNoCache,
				Action:          "no_cache",
				State:           vcacheobserve.ActionReady,
				Reason:          "route the prefix uncached because the content is not warmable",
				Turns:           2,
				SavedTokenEquiv: 0,
			},
		},
	}
	raw, err := json.Marshal(plan)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(actionsPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--vcache-actions-report", actionsPath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status --vcache-actions-report --json exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.VCacheActions == nil || rep.VCacheActions.Status != "gated" ||
		rep.VCacheActions.Ready != 1 || rep.VCacheActions.Gated != 1 ||
		rep.VCacheActions.TransportReady {
		t.Fatalf("vcache actions digest = %+v", rep.VCacheActions)
	}
	rows := cachevalueRowsByComponent(rep.Rows)
	aggregate := rows["vcache_actions_report"]
	if aggregate.Status != "gated" || aggregate.Owner != "fak" ||
		aggregate.FailureDomain != "provider_transport" {
		t.Fatalf("actions aggregate row = %+v", aggregate)
	}
	transport := rows["vcache_actions_transport"]
	if transport.Owner != "provider" || transport.Status != "gated" ||
		transport.Evidence != "MISSING" || transport.FailureDomain != "provider_transport" {
		t.Fatalf("transport row = %+v", transport)
	}
	heartbeat := rows["vcache_action:bursty:heartbeat_pin"]
	if heartbeat.Owner != "provider" || heartbeat.Status != "gated" ||
		!strings.Contains(heartbeat.Reason, "missing=byte_identical_prefix") ||
		!strings.Contains(heartbeat.NextAction, "byte_identical_prefix") {
		t.Fatalf("heartbeat action row = %+v", heartbeat)
	}
	noCache := rows["vcache_action:secret:no_cache"]
	if noCache.Owner != "fak" || noCache.Dependency != "local_provider_manifest" ||
		noCache.Status != "ready" {
		t.Fatalf("no_cache action row = %+v", noCache)
	}
	if findingByKey(rep.Attribution.ProblemDomains, "provider_transport") == nil ||
		findingByKey(rep.Attribution.ProblemOwners, "provider") == nil {
		t.Fatalf("actions attribution missing provider transport findings: %+v", rep.Attribution)
	}

	out.Reset()
	errb.Reset()
	code = runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--vcache-actions-report", actionsPath,
	})
	if code != 0 {
		t.Fatalf("status --vcache-actions-report text exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"vcache actions: status=gated", "vcache_actions_transport", "vcache_action:bursty:heartbeat_pin", "missing=byte_identical_prefix"} {
		if !strings.Contains(got, want) {
			t.Fatalf("vcache actions status text missing %q:\n%s", want, got)
		}
	}
}

func TestCachevalueStatusFoldsVCacheContextJoinReportAttribution(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	contextJoinPath := filepath.Join(dir, "context-join.json")
	report := vcacheobserve.JoinReport{
		Schema: vcacheobserve.JoinSchema,
		Turns:  8,
		Events: 1,
		Changes: []vcacheobserve.AttributedChange{
			{
				Family:     "alpha",
				UnixMillis: 30_000,
				Change:     vcacheobserve.ChangeCacheCreateSpike,
				Cause:      vcacheobserve.CausePlanning,
				Detail:     "cache creation rose from warm baseline to 42000 tokens",
				MatchedEvent: &vcacheobserve.LifecycleEvent{
					Kind:       vcacheobserve.EventCompaction,
					Family:     "alpha",
					UnixMillis: 25_000,
					Outcome:    "compaction",
					Detail:     "shed old tool output",
				},
			},
			{
				Family:     "beta",
				UnixMillis: 1_000_000,
				Change:     vcacheobserve.ChangeHitRateDrop,
				Cause:      vcacheobserve.CauseProviderBehavior,
				Detail:     "hit rate fell from 90% to 20% with no matched lifecycle event",
			},
		},
		Summary: vcacheobserve.JoinSummary{TotalChanges: 2, PlanningAttributed: 1, ProviderAttributed: 1},
	}
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(contextJoinPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--vcache-context-join-report", contextJoinPath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status --vcache-context-join-report --json exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.VCacheContextJoin == nil || rep.VCacheContextJoin.Status != "measured" ||
		rep.VCacheContextJoin.TotalChanges != 2 ||
		rep.VCacheContextJoin.PlanningAttributed != 1 ||
		rep.VCacheContextJoin.ProviderAttributed != 1 {
		t.Fatalf("vcache context-join digest = %+v", rep.VCacheContextJoin)
	}
	rows := cachevalueRowsByComponent(rep.Rows)
	aggregate := rows["vcache_context_join_report"]
	if aggregate.Status != "measured" || aggregate.FailureDomain != "mixed_context_provider" ||
		!strings.Contains(aggregate.SessionImpact, "separates fak managed-context") {
		t.Fatalf("context-join aggregate row = %+v", aggregate)
	}
	planning := rows["context_join:alpha:cache_create_spike:1"]
	if planning.Owner != "fak" || planning.Status != "context_planning" ||
		planning.Fidelity != "lossy" || planning.FailureDomain != "fak_context_planner" ||
		!strings.Contains(planning.Reason, "matched=compaction") {
		t.Fatalf("context planning row = %+v", planning)
	}
	provider := rows["context_join:beta:hit_rate_drop:2"]
	if provider.Owner != "provider" || provider.Status != "provider_cache_behavior" ||
		provider.Fidelity != "passive" || provider.FailureDomain != "provider_cache_behavior" {
		t.Fatalf("provider behavior row = %+v", provider)
	}
	if findingByKey(rep.Attribution.ProblemDomains, "fak_context_planner") == nil ||
		findingByKey(rep.Attribution.ProblemDomains, "provider_cache_behavior") == nil ||
		findingByKey(rep.Attribution.ProblemOwners, "provider") == nil {
		t.Fatalf("context-join attribution missing fak/provider findings: %+v", rep.Attribution)
	}

	out.Reset()
	errb.Reset()
	code = runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--vcache-context-join-report", contextJoinPath,
	})
	if code != 0 {
		t.Fatalf("status --vcache-context-join-report text exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"vcache context-join: status=measured", "context_planning", "provider_cache_behavior", "matched=compaction", "mixed_context_provider"} {
		if !strings.Contains(got, want) {
			t.Fatalf("vcache context-join status text missing %q:\n%s", want, got)
		}
	}
}

func TestCachevalueStatusFoldsVCacheContextWitnessReport(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	witnessPath := filepath.Join(dir, "context-witness.json")
	report := vcacheContextWitnessReport{
		Schema:      "fak.vcache.context-witness.v1",
		Fixture:     "testdata/context-replay.jsonl",
		Wire:        "openai",
		Snapshot:    filepath.Join(dir, "context-snapshot.jsonl"),
		ReplayExit:  0,
		ScoreExit:   0,
		ScoreStatus: "2x_ready",
		ContextWitnessed: vcachescore.PlaneValueReport{
			Available:          true,
			Provenance:         "WITNESSED",
			SavedTokenEquiv:    800,
			BaselineTokenEquiv: 2000,
			CostTokenEquiv:     1200,
			Reason:             "context shed-token witness supplied",
		},
		ContextEvents:     2,
		ContextShedTokens: 800,
	}
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(witnessPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--vcache-context-witness-report", witnessPath,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status --vcache-context-witness-report --json exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.VCacheContextWitness == nil || rep.VCacheContextWitness.Status != "measured" ||
		rep.VCacheContextWitness.ContextWitnessed != "WITNESSED" ||
		rep.VCacheContextWitness.ContextEvents != 2 ||
		rep.VCacheContextWitness.ContextShedTokens != 800 {
		t.Fatalf("vcache context-witness digest = %+v", rep.VCacheContextWitness)
	}
	rows := cachevalueRowsByComponent(rep.Rows)
	aggregate := rows["vcache_context_witness_report"]
	if aggregate.Status != "measured" || aggregate.Owner != "fak" ||
		aggregate.Fidelity != "lossy" || aggregate.Evidence != "WITNESSED" ||
		!strings.Contains(aggregate.SessionImpact, "fak-owned context replay") {
		t.Fatalf("context-witness aggregate row = %+v", aggregate)
	}
	plane := rows["vcache_context_witness_plane"]
	if plane.Status != "measured" || plane.FailureDomain != "fak_context_planner" ||
		!strings.Contains(plane.Reason, "context shed-token witness") {
		t.Fatalf("context-witness plane row = %+v", plane)
	}

	out.Reset()
	errb.Reset()
	code = runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--vcache-context-witness-report", witnessPath,
	})
	if code != 0 {
		t.Fatalf("status --vcache-context-witness-report text exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"vcache context-witness: status=measured", "context=WITNESSED", "vcache_context_witness_report", "vcache_context_witness_plane", "context shed-token witness"} {
		if !strings.Contains(got, want) {
			t.Fatalf("vcache context-witness status text missing %q:\n%s", want, got)
		}
	}
}

func TestCachevalueStatusArtifactDirDiscoversReports(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	artifactDir := filepath.Join(dir, "artifacts")
	if err := os.Mkdir(artifactDir, 0o700); err != nil {
		t.Fatal(err)
	}
	observeReport := vcacheobserve.Report{
		Schema:      vcacheobserve.Schema,
		Turns:       2,
		FamilyCount: 1,
		Aggregate:   vcachegov.TelemetrySavingsProof{CacheReadTokens: 20_000, CacheCreationTokens: 21_000, SavedTokenEquiv: 5_000},
		HitRate:     0.49,
		Multiplier:  1.4,
		OwnerSlices: []vcacheobserve.OwnerSlice{{
			Owner:           "provider",
			Mechanism:       "prompt_cache",
			SavedTokenEquiv: 5_000,
			CacheReadTokens: 20_000,
			Provenance:      vcacheobserve.Observed,
			Evidence:        "provider usage counters",
		}},
		Families: []vcacheobserve.Family{{
			Key:             "bundle",
			Turns:           2,
			CacheReadTokens: 20_000,
			HitRate:         0.49,
			Economics:       vcachegov.TelemetrySavingsProof{SavedTokenEquiv: 5_000},
		}},
	}
	if err := writeJSONFile(filepath.Join(artifactDir, "vcache-observe.json"), observeReport); err != nil {
		t.Fatal(err)
	}
	witnessReport := vcacheContextWitnessReport{
		Schema:      "fak.vcache.context-witness.v1",
		Fixture:     "fixture.jsonl",
		Wire:        "openai",
		Snapshot:    filepath.Join(artifactDir, "context-snapshot.jsonl"),
		ReplayExit:  0,
		ScoreExit:   0,
		ScoreStatus: "partial",
		ContextWitnessed: vcachescore.PlaneValueReport{
			Available:       true,
			Provenance:      "WITNESSED",
			SavedTokenEquiv: 300,
			Reason:          "bundle context witness",
		},
		ContextEvents:     1,
		ContextShedTokens: 300,
	}
	if err := writeJSONFile(filepath.Join(artifactDir, "context-witness.json"), witnessReport); err != nil {
		t.Fatal(err)
	}

	withCachevalueStatusHeadroom(t, headroom.NoopName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")

	var out, errb bytes.Buffer
	code := runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--artifact-dir", artifactDir,
		"--json",
	})
	if code != 0 {
		t.Fatalf("status --artifact-dir --json exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.Sources.ArtifactDir != artifactDir {
		t.Fatalf("artifact dir source = %q want %q", rep.Sources.ArtifactDir, artifactDir)
	}
	if rep.VCacheObserve == nil || rep.VCacheObserve.Path != filepath.Join(artifactDir, "vcache-observe.json") ||
		rep.VCacheObserve.Status != "observed" {
		t.Fatalf("discovered observe digest = %+v", rep.VCacheObserve)
	}
	if rep.VCacheContextWitness == nil || rep.VCacheContextWitness.Path != filepath.Join(artifactDir, "context-witness.json") ||
		rep.VCacheContextWitness.Status != "measured" {
		t.Fatalf("discovered context witness digest = %+v", rep.VCacheContextWitness)
	}
	rows := cachevalueRowsByComponent(rep.Rows)
	if rows["vcache_observe_report"].Status != "observed" ||
		rows["vcache_context_witness_report"].Status != "measured" {
		t.Fatalf("artifact-dir rows missing discovered reports: %+v", rows)
	}

	out.Reset()
	errb.Reset()
	code = runCachevalueStatus(&out, &errb, []string{
		"--ledger", track1,
		"--savings-ledger", track2,
		"--usage-ledger", filepath.Join(dir, "absent-usage.jsonl"),
		"--artifact-dir", artifactDir,
	})
	if code != 0 {
		t.Fatalf("status --artifact-dir text exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"artifacts: dir=", "vcache observe: status=observed", "vcache context-witness: status=measured", "bundle context witness"} {
		if !strings.Contains(got, want) {
			t.Fatalf("artifact-dir status text missing %q:\n%s", want, got)
		}
	}
}

func TestCachevalueStatusGateFailOnFlipsExitCode(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)
	usage := filepath.Join(dir, "absent-usage.jsonl")
	withCachevalueStatusHeadroom(t, headroom.HeadroomName)
	t.Setenv("FAK_HEADROOM_URL", "http://127.0.0.1:9")
	t.Setenv("FAK_HEADROOM_TIMEOUT_MS", "50")
	t.Setenv("FAK_VCACHE_SNAPSHOT", filepath.Join(dir, "absent-vcache.json"))
	t.Setenv("FAK_VCACHE_CONTEXT_SNAPSHOT", "off")
	base := []string{"--ledger", track1, "--savings-ledger", track2, "--usage-ledger", usage, "--json"}

	// This corpus folds to PARTIAL (pinned by the rollup test above); the
	// default (ungated) invocation must keep exiting 0.
	var out, errb bytes.Buffer
	if code := runCachevalueStatus(&out, &errb, base); code != 0 {
		t.Fatalf("ungated status exit=%d stderr=%s", code, errb.String())
	}
	var rep cachevalueStatusReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("status JSON did not decode: %v\n%s", err, out.String())
	}
	if rep.Verdict != "PARTIAL" {
		t.Fatalf("fixture verdict = %q, want PARTIAL", rep.Verdict)
	}

	// --gate --fail-on PARTIAL on a PARTIAL corpus flips to exit 1, still
	// emitting the JSON report so a CI capture keeps the evidence.
	out.Reset()
	errb.Reset()
	code := runCachevalueStatus(&out, &errb, append(append([]string{}, base...), "--gate", "--fail-on", "PARTIAL"))
	if code != 1 {
		t.Fatalf("--gate --fail-on PARTIAL exit=%d, want 1; stderr=%s", code, errb.String())
	}
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("gated status JSON did not decode: %v\n%s", err, out.String())
	}
	if !strings.Contains(errb.String(), "gate: verdict PARTIAL is at or worse than --fail-on floor PARTIAL") {
		t.Fatalf("gate refusal not named on stderr: %s", errb.String())
	}

	// --fail-on alone implies the gate.
	out.Reset()
	errb.Reset()
	if code := runCachevalueStatus(&out, &errb, append(append([]string{}, base...), "--fail-on", "partial")); code != 1 {
		t.Fatalf("--fail-on partial exit=%d, want 1; stderr=%s", code, errb.String())
	}

	// A PARTIAL corpus passes a floor of INSUFFICIENT.
	out.Reset()
	errb.Reset()
	if code := runCachevalueStatus(&out, &errb, append(append([]string{}, base...), "--gate", "--fail-on", "INSUFFICIENT")); code != 0 {
		t.Fatalf("--gate --fail-on INSUFFICIENT exit=%d, want 0; stderr=%s", code, errb.String())
	}

	// --gate alone defaults the floor to PARTIAL.
	out.Reset()
	errb.Reset()
	if code := runCachevalueStatus(&out, &errb, append(append([]string{}, base...), "--gate")); code != 1 {
		t.Fatalf("--gate (default floor) exit=%d, want 1; stderr=%s", code, errb.String())
	}

	// An unknown floor is a usage error, not a silent pass.
	out.Reset()
	errb.Reset()
	if code := runCachevalueStatus(&out, &errb, append(append([]string{}, base...), "--gate", "--fail-on", "BOGUS")); code != 2 {
		t.Fatalf("--fail-on BOGUS exit=%d, want 2; stderr=%s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "--fail-on must be OK, PARTIAL, or INSUFFICIENT") {
		t.Fatalf("bad floor error not named: %s", errb.String())
	}

	// A blank --fail-on (e.g. a CI quoting mistake) is a loud usage error,
	// never a silently disabled gate.
	out.Reset()
	errb.Reset()
	if code := runCachevalueStatus(&out, &errb, append(append([]string{}, base...), "--fail-on", " ")); code != 2 {
		t.Fatalf("--fail-on ' ' exit=%d, want 2; stderr=%s", code, errb.String())
	}
}

func TestCachevalueStatusGateExitOrdersVerdicts(t *testing.T) {
	cases := []struct {
		verdict, floor string
		want           int
	}{
		{"OK", "PARTIAL", 0},
		{"OK", "OK", 1},
		{"PARTIAL", "PARTIAL", 1},
		{"PARTIAL", "INSUFFICIENT", 0},
		{"INSUFFICIENT", "PARTIAL", 1},
		{"INSUFFICIENT", "INSUFFICIENT", 1},
		{"ok", "partial", 0},
		{"UNKNOWN", "PARTIAL", 0},
	}
	for _, tc := range cases {
		if got := cachevalueStatusGateExit(tc.verdict, tc.floor); got != tc.want {
			t.Fatalf("cachevalueStatusGateExit(%q, %q) = %d, want %d", tc.verdict, tc.floor, got, tc.want)
		}
	}
}

func cachevalueRowsByComponent(rows []cachevalueStatusRow) map[string]cachevalueStatusRow {
	out := map[string]cachevalueStatusRow{}
	for _, row := range rows {
		out[row.Component] = row
	}
	return out
}

func findingByKey(findings []cachevalueStatusFinding, key string) *cachevalueStatusFinding {
	for i := range findings {
		if findings[i].Key == key {
			return &findings[i]
		}
	}
	return nil
}

func withCachevalueStatusHeadroom(t *testing.T, name string) {
	t.Helper()
	prev := headroom.Selected().Name()
	if !headroom.Select(name) {
		t.Fatalf("unknown headroom plugin %q", name)
	}
	t.Cleanup(func() {
		headroom.Select(prev)
	})
}
